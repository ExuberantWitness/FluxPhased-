# 激光无人机训练零击杀 — 详细根因分析报告

**日期**: 2026-06-15
**分析基线**: origin/main @ 41105f0(逐行只读分析 + 雷达 pipeline 审计)
**分析对象**: training/train_laser.py、training/radar_policy.py、radar_sim/gpu/{vec_drone,vec_battlefield,vec_weapon,vec_mfar_env}.py、configs/laser_25x25_train*.yaml
**关联文档**: LASER_CRITIC_POSTMORTEM.md(被复核对象 — 诊断方向对但机制错)

---

## 0. 执行摘要

激光无人机训练在四种 critic 架构(IPPO+fix / IPPO+no-decay / CTDE / MAPPO)下全部零击杀,best eval 瞄准误差卡在 151–329m,且都在 iter 20 退化到 >1000m。

LASER_CRITIC_POSTMORTEM.md 把根因归为"精度天花板:策略 tanh-Gaussian 噪声 ~25m ≫ 0.2m 击杀半径,差 125×"。该诊断**方向对**(确实存在精度鸿沟)但**机制错**。

逐行读代码后的真实结论:

> 这本质上是一个**"commander 拿着敌人真值坐标、被 BC 直接监督去抄它"的监督指向任务,本该 trivial**。失败的原因不是探索噪声,而是**奖励曲面在 95m 以内被截断为平台(零梯度)** + **奖励与"开火/击杀"脱钩**,使得 PPO 不仅帮不上 BC,还用带噪优势把已经学到的精确瞄准推离目标。换 critic 改变不了这条曲面,所以四个变体注定全败。

进一步审计发现第五个问题(§5):

> **radar_latents 在训练循环里是全零张量**([train_laser.py:262](training/train_laser.py#L262))。commander obs[4:68] 是空的,**敌方信息的唯一来源是 obs[68:72] 真值**。直接删除真值会让 commander 完全失明。这阻断了"完全移除真值 → 真 MARL benchmark"的路径,需要先建雷达检测 pipeline。

五类问题(详见 §4):
- **A 死区**:spatial 被 beam_spatial_cap 截断 → r<95m 零梯度。头号绑定约束。
- **B 半径错配**:dense 奖励在 500m 内累积时间 t,击杀要求 0.2m,差 2500×。
- **C 开火脱钩**:dense 奖励不看 fire_on,击杀必须 fire_on,BC 不监督开火位 → 策略从未被教会开火。
- **D 监督指向被破坏**:BC 能解但 PPO 在平台区拖后腿。
- **E 配置/代码不一致**(含 radar_latents=0):默认值分叉、死键、错注释、pipeline 断线。

---

## 1. 任务与配置(代码确认)

**武器/击杀模型** ([vec_weapon.py](radar_sim/gpu/vec_weapon.py)、[vec_battlefield.py:149-216](radar_sim/gpu/vec_battlefield.py#L149)):
- 无人机定点在 3km 高空,激光瞄准地面敌方雷达
- 击杀条件:瞄点到任一存活敌方雷达 < `kill_radius_m`,且连续保持 `illumination_time_s`,任何一拍脱靶则计时清零([vec_weapon.py:72-78](radar_sim/gpu/vec_weapon.py#L72));且必须 `fire_on=True`([vec_battlefield.py:195](radar_sim/gpu/vec_battlefield.py#L195))
- 配置:`kill_radius_m=0.2m`,`illumination_time_s=0.002s`,`dt=pri=100μs` → 0.2m 内连续 20 拍

**动作空间** ([vec_drone.py:160-183](radar_sim/gpu/vec_drone.py#L160)):commander 输出 5 维 tanh 动作:
- `action[0]`:开火位(>0.5 开火)
- `action[1] × half_x`(10000m) = 瞄准 x
- `action[2] × half_y`(10000m) = 瞄准 y
- `action[3] × 1000m` = 瞄准 z;`action[4]` 保留

**观测** ([vec_drone.py:207-278](radar_sim/gpu/vec_drone.py#L207),76 维):

| 区段 | 内容 | 来源 |
|---|---|---|
| `[0:4]` | own radar 0/1 位置 (x,y) / half_map | 真值 |
| `[4:36]` | radar 0 latent (32-dim) | **应为 radar_ac 输出,实际全 0** |
| `[36:68]` | radar 1 latent (32-dim) | 同上 |
| `[68:72]` | enemy radar 0/1 位置 (x,y) / half_map | **真值** |
| `[72:74]` | drone laser aim (x,y) / half_map | 状态 |
| `[74:76]` | [fire_on, illumination_progress] | 状态 |

**关键推论**: 若 `action[1:3] == obs[68:70]`,则瞄点 = `obs[68:70] × half_map` = 敌方雷达 0 真值位置,瞄准误差 ≈ 0、必杀。**这是个监督指向问题,本应该 trivial**。

**关键超参** ([configs/laser_25x25_train.yaml](configs/laser_25x25_train.yaml)):

| 项 | 值 | 项 | 值 |
|---|---|---|---|
| max_steps_per_episode | 500 拍 (50ms) | pulses_per_control | 5 |
| beam_reward_weight | 5.0 | beam_hit_radius_m | 500.0 |
| beam_r_ref_m | 3000.0 | beam_r_floor_m | 2.0 |
| beam_spatial_cap | **1000.0** ← 头号嫌疑 | beam_temporal_epsilon | 0.05 |
| kill_bonus | 100.0 | death_penalty | -10.0 |
| log_std_init / floor / decay | -1.0 / -6.0 / 0.20 | cmd_bc_weight init/final/iters | 1.0 / 0.1 / 10 |

---

## 2. 奖励曲面详解:全场有奖励,但 95m 处梯度断裂

dense 奖励在 [train_laser._get_rewards:324-365](training/train_laser.py#L324) 逐 team 计算:

```python
dist_all = (aim - enemy_pos).norm(dim=-1)            # 到每个敌方雷达的距离 [E, R/2]
dist_all = dist_all + (~enemy_alive) * 1e6           # 屏蔽已死敌人
min_dist = dist_all.min()                            # 取最近存活敌人 → 不绑定特定目标
hitting  = min_dist < beam_hit_radius_m              # 500m 内才累积时间
t        = where(hitting, t + dt, 0)                 # 连续 hit 时间
spatial  = clamp((r_ref / max(min_dist, r_floor))², max=spatial_cap)   # = clamp((3000/r)², ≤1000)
temporal = beam_temporal_epsilon + (t / t_max)⁴      # = 0.05 + (t/0.002)⁴
beam_reward = spatial × temporal × 5.0
cmd_rewards[:, t] += beam_reward
radar_rewards[own] += beam_reward × 0.1
```

**spatial cap 触顶距离**: `(3000/r)² = 1000` → `r = 3000/√1000 ≈ 94.87m`

代入数值,展示这条"全场奖励"的形状(temporal 仅在 500m 内累积,故外侧恒为 ε=0.05):

| 到最近敌人 r | spatial | temporal(满驻留) | beam_reward/步 | ∂reward/∂r |
|---|---|---|---|---|
| 10000m(角落) | 0.09 | 0.05 (t=0) | ~0.02 | 有(弱) |
| 3000m | 1.0 | 0.05 | ~0.25 | 有 |
| 500m(累积起点) | 36 | 0.05→1.05 | 9 → 189 | 有(强) |
| **94.9m** | **1000(触顶)** | 1.05 | **5250** | **拐点** |
| 50m | 1000(仍触顶) | 1.05 | 5250 | ≈0 |
| 10m | 1000(仍触顶) | 1.05 | 5250 | 0 |
| 0.2m(击杀) | 1000(仍触顶) | 1.05 | 5250 | 0 |

两个结论:
1. **空间上是全场的**:从 10km 到 95m,处处非零、单调拉向最近敌人——这正是策略能从 BC 的 ~1700m 压到 ~95–151m 的原因
2. **95m 以内是死区**:cap 截断后,95m 和 0.2m 拿到的奖励完全相同(5250)。最大奖励在 95m 内任意位置驻留即可拿满 → 最优确定性策略停在 ~95m,**与实测 best eval=151m 精确吻合**。奖励无法区分"差一点"和"击杀"

---

## 3. 评估是确定性的:151m 是均值误差,不是噪声

`eval_episode` / `_eval_nn_step` ([train_laser.py:631-723](training/train_laser.py#L631)):

```python
r_action, *_ = self.radar_ac.get_action(radar_flat, deterministic=True)
c_action, *_ = self.commander_ac.get_action(cmd_flat, deterministic=True)  # 用 mean,绕过 log_std
c_action[:, 0] = 1.0   # 强制开火,使 eval 测的是真实瞄准
```

`get_action(deterministic=True)`([actor_critic.py:170-171](training/ppo/actor_critic.py#L170))直接取 `mean`,完全不采样、不用 `log_std`。

→ postmortem 的"探索噪声 25m 是天花板"**在 eval 阶段根本不生效**。best eval=151m 反映的是**策略均值的瞄准误差**(击杀半径的 755 倍),不是抖动。postmortem 把"训练期探索噪声"与"评估期均值精度"混为一谈。

(补:训练期 log_std 在 iter 20 实际是 `max(-6, -1-20×0.2)=-5` → std=exp(-5)≈6.7e-3 in [-1,1] → ~67m 物理噪声。比 postmortem 说的 25m 还大,且 25 轮才到 -6;但这只影响训练采样,不影响 eval。)

---

## 4. 四类问题汇总(均经代码确认)

### A. 死区:spatial cap 造成 95m 内零梯度【头号绑定约束】

见 §2。[train_laser.py:352](training/train_laser.py#L352) `spatial.clamp(max=self.beam_spatial_cap)`。r<95m 奖励恒为平台,PPO 在击杀最需要 refine 的区间得不到任何梯度。

### B. 半径错配:奖励累积半径(500m)≫ 击杀半径(0.2m)

[train_laser.py:342](training/train_laser.py#L342) 温度 `t` 在 `beam_hit_radius_m=500m` 内就累积、20 拍后 `(t/t_max)⁴` 饱和;而击杀要 0.2m 连续 20 拍([vec_battlefield.py:194-201](radar_sim/gpu/vec_battlefield.py#L194))。奖励教的是"飞到 500m 内赖着",与击杀需要的"精确锁定 0.2m"方向相反。

### C. 开火脱钩:策略从未被教会开火

- dense 奖励([train_laser.py:341-360](training/train_laser.py#L341))只看瞄准距离 `min_dist`,**不看 fire_on** → 不开火也能拿满 dense 奖励
- 但击杀([vec_battlefield.py:195](radar_sim/gpu/vec_battlefield.py#L195) `on_target & fire_on`)和 env 侧 illumination 奖励([vec_battlefield.py:299-300](radar_sim/gpu/vec_battlefield.py#L299))都要求开火
- BC([train_laser.py:452-459](training/train_laser.py#L452))只监督 `action[1:3]`(瞄准),**完全不碰 `action[0]`(开火)**
- `kill_bonus`(+100,[vec_battlefield.py:288](radar_sim/gpu/vec_battlefield.py#L288))因精度永远收不到 → 开火位无任何训练信号
- eval 用 `c_action[:,0]=1.0` 强制开火,把这个洞掩盖了。即便修好精度,训练期若不开火仍是 0 击杀

### D. 监督指向问题被奖励设计破坏

- commander 观测里有敌人真值坐标(`obs[68:70]`),BC 直接监督抄它 → 完美 BC 瞄准误差 ≈ 0
- 但 `bc_loss` 只到 0.03(每轴 `√0.03×10000=1732m` 潜在误差),best eval 卡在 151m,说明 BC 没被压到 0
- 原因:dense 奖励在 95m 内是平的,PPO 的带噪优势把 `action_head` 推离 BC 目标,而平台区没有反向梯度把它拉回。postmortem 自述"不要更快衰减 BC,BC 是唯一把 eval 压到 1000m 以下的东西"——**正印证 BC 在干活、PPO 在帮倒忙**
- "先升后崩"(151m@iter4 → 6396m@iter20):早期 BC 权重高把均值拽到死区边缘,随 BC 衰减(→0.1)+ PPO 噪声累积,均值被推离 → 回归 >1000m

### E. 配置键与代码不一致(换配置即触发的雷)

- [train_laser.py:216-221](training/train_laser.py#L216) 的 reward 默认值与 `laser_25x25_train.yaml` 不同:代码默认 `r_ref=100/cap=100/hit=200`(死区在 10m),配置 `=3000/1000/500`(死区在 95m)。当前用 train 配置以配置为准没问题,但任何缺这些键的配置会落到完全不同的死区,诊断极易看错
- `illumination_progress_weight`(配置有,train_laser.py:207 读入 `self.illumination_weight`)全程未被使用;env 实际读 `illumination_reward`([vec_battlefield.py:300](radar_sim/gpu/vec_battlefield.py#L300),默认 1.0)→ 一个死键 + 一个走默认值
- log_std "~7.5m noise" 错误注释出现两处(配置 line 43、train_laser.py:1001),实际 25m(乘子是 `half_map=10000m`,非高度 3000m)

**时序自检**(无问题):500 拍/集 = 100 个 NN 控制步;击杀需 20 连续拍 = 4 次 aim-hold;敌人 20 m/s × 2ms = 4cm 位移 < 0.2m。→ episode 长度与控制频率足够,不是瓶颈。

---

## 5. 第五类问题:radar_latents=0 阻塞"obs 真值移除"

逐行审计中发现。这是 §4 之外的、之前所有诊断都漏掉的**架构性阻塞**。

### 5.1 现象

[training/train_laser.py:262](training/train_laser.py#L262):

```python
radar_latents = torch.zeros(self.E, self.R, 32, device=dev)
```

**commander obs[4:68] 的 64 维雷达 latent 全部填零**。commander 关于敌方的**唯一信息源**就是 `obs[68:72]` 真值。

### 5.2 代码库中存在但未接线的基础设施

| 组件 | 位置 | 状态 |
|---|---|---|
| CPI accumulator + FFT/matched filter/beamforming | [radar_policy.py:26-56, 192-200](training/radar_policy.py#L26) | ✅ 实现完整,产出 `self._spectrum` |
| Radar NN 输出含位置估计 dims [20:22] | [radar_policy.py:274-275](training/radar_policy.py#L274) | ⚠️ 注释说"是位置估计",但 NN **未被监督**产出有用估计,且该维度作为 BPSK comm payload 调制到空中 |
| BPSK encode/decode comm 链 | radar_sim 中存在 | ⚠️ 物理链路存在,但**没有"radar 检测 → BPSK 编码 → commander 解码"的闭环** |
| `radar_latents` 注入 commander obs | [vec_drone.py:257-259](radar_sim/gpu/vec_drone.py#L257) | ✅ 接口存在,接收 `[E, R, 32]` |
| 训练循环里调用注入 | [train_laser.py:262](training/train_laser.py#L262) | ❌ **传全零张量**,等价于没接 |

### 5.3 推论:obs 真值移除的依赖图

要实现"完全移除 obs[68:72] 真值 → 真 MARL",需要先建好以下 pipeline:

```
Raw IQ → CPI → FFT/spectrum → [radar NN: position regression head] → position estimate
                                                                       │
                                                                       ▼
                                                              BPSK encode → air
                                                                       │
                                                                       ▼
                                                              commander decode
                                                                       │
                                                                       ▼
                                                       commander obs[4:68] (latent)
```

每一步的现状:

1. **Raw IQ → CPI → FFT/spectrum**:✅ 已实现
2. **Radar NN 位置回归头**:❌ 未实现。当前 radar NN 是 PPO actor-critic,没有监督信号教它输出敌方位置。需要新增回归 head + 监督 loss(从 spectrum 估敌方 DOA + 距离)
3. **BPSK encode + 空口传输 + decode**:⚠️ 物理实现存在,但 commander 端没有 BPSK 解调 → 位置提取的逻辑
4. **commander 用 latent 替代真值**:✅ 接口存在,只需在训练循环里替换 `radar_latents = ...`

### 5.4 这意味着什么

"完全移除敌方真值"不是改几行能解决的。它隐含要求:

| 工作项 | 估算工作量 | 阻塞性 |
|---|---|---|
| 直接删 `obs[68:72]` | 几行 | ❌ commander 完全瞎,训不出来 |
| 接 `radar_latents = radar_ac` 中间层输出 | 1 天 | 闭环但 radar NN 未被监督,latent 是垃圾 |
| 给 radar NN 加位置回归头 + BC 监督 | 2-3 天 | 真雷达检测 pipeline |
| 完整 BPSK comm 链 | 4-7 天 | 真正多智能体(探测 → 通信 → 决策) |

---

## 6. 为什么四个 critic 变体全败(对 postmortem 的回应)

| 变体 | postmortem 说法 | 实际 |
|---|---|---|
| IPPO+fix | BC 衰减后 PPO 漂移 | 对,但根因是平台区无梯度,PPO 只能注入噪声 |
| IPPO+no-decay | BC 锚定仍被 PPO 污染 | 对,印证 §4-D |
| CTDE | 信息不对称非瓶颈 | 对,critic 改变不了奖励曲面 |
| MAPPO | 团队 critic 不解决连续控制精度 | 对 |

**共同点**:四者都只改 advantage 的估计方式,而绑定约束是奖励曲面本身(A/B)+ 开火脱钩(C)。advantage 再准,也无法在零梯度平台上产生 refine 信号。所以"再试 QMIX/MADDPG"是错的方向(这点 postmortem 说对了)。

---

## 7. 修复方案(按性价比 + 优雅度排序)

### P0-1 修奖励曲面 — 让梯度一路延伸到 0.2m(几行)

[train_laser.py:349-352](training/train_laser.py#L349),推荐方案 b(对数标度):

```python
# 方案 b(推荐,数值稳定):对数标度,全程单调有梯度
spatial = torch.log(self.beam_r_ref_m / r_eff).clamp(min=0.0)
# r: 3000→0.2 持续增长,梯度始终非零
```

同时把温度累积半径收紧,逼"精确锁定"而非"近处驻留":

```yaml
beam_hit_radius_m: 10.0   # 配置 500 → 10m
```

### P0-2 修开火脱钩 — 三种哲学,benchmark 优雅度递增

#### 方案 a: dense 奖励 × fire_on 门控(一行)

```python
# train_laser.py:342 附近
fire_on = self.env.battlefield.drone.fire_on[:, t]   # [E]
hitting = (min_dist < self.beam_hit_radius_m) & fire_on
# 让 spatial/beam_reward 也 × fire_on.float()
```
**问题**:鸡生蛋。policy 不开火 → 没信号 → 学不会。需要探索机制保证初期开火率。

#### 方案 b: 训练期也强制开火(简单)

```python
c_action[:, 0] = 1.0   # 训练 + eval 都强制
```
**问题**:跳过开火决策维度。benchmark 不优雅。

#### 方案 c (推荐): hybrid action — Bernoulli fire + Gaussian aim

承认 `action[0]` 是离散的,改用 Bernoulli 分布。同时保留 Gaussian 用于连续 aim。这是 benchmark 级做法。

```python
# actor_critic.py — CommanderActorCritic 重构
self.fire_logit_head = nn.Linear(hidden_dim, 1)   # Bernoulli
self.aim_mean_head = nn.Linear(hidden_dim, 2)     # Gaussian (x, y)
self.aim_z_head = nn.Linear(hidden_dim, 1)        # Gaussian (z)

def get_action(...):
    fire_logit = self.fire_logit_head(features)
    fire_dist = torch.distributions.Bernoulli(logits=fire_logit)
    fire = fire_dist.sample()  # 0 or 1
    # ... aim 同前
```
PPO 标准 entropy 项就驱动 fire 的探索,**不需要任何 ad-hoc bonus**。

### P0-3 kill_radius 课程(注意对象)

击杀半径在 `battlefield.laser.kill_radius_m`(VecLaser,**不是** postmortem 写的 `drone.kill_radius_m` — 那是空操作)。当前无 mid-training 改它的代码([train_laser.py:50](training/train_laser.py#L50) 构造时一次性读)。

**两个半径同步退火**(beam_hit_radius 与 kill_radius):

```python
# 每个 PSRO iter 开始时
level = max(0.0, 1.0 - psro_iter / psro_iters)   # 1.0 → 0.0
kr = 0.2 + (50.0 - 0.2) * level                   # kill_radius: 50m → 0.2m
br = kr * 5.0                                     # beam_hit_radius 同步: 250m → 1m
env.battlefield.laser.kill_radius_m = kr
self.beam_hit_radius_m = br
```

**退火判据**(自适应,推荐):
- 维持 iter-based 退火作为基线
- 叠加 eval-triggered 加速:`eval_min_aim_dist < kr * 0.5` 连续 2 iter → 跳一级

### P1 移除敌方真值(分阶段,依赖 §5)

- **第一阶段**:加 Gaussian 噪声到 `obs[68:72]`(σ 可配,默认 200m)。BC 误差下限提高,PPO 有真实工作。**今天可做**
- **第二阶段**:接 radar spectrum → 位置回归头 → commander latent(2-3 天)
- **第三阶段**:完整 BPSK comm 链(4-7 天)

### P2 卫生项

- 统一 [train_laser.py:216-221](training/train_laser.py#L216) 默认值与配置
- 删除死键 `illumination_progress_weight` 或接通它
- 改正两处 "7.5m" 注释为 25m
- `.git/config` 内嵌明文 PAT,**强烈建议轮换**并改用 credential helper

---

## 8. 验证计划

**本机**(24G,无需大显存):配置解析 + import 冒烟;对修改后的 `_get_rewards` 用小张量手验奖励在 95m→0.2m 单调递增

**训练机**(98G):按梯度判据逐级确认
1. 单集确定性 eval:`min_aim_dist` 先破 10m(证明梯度延伸进死区)
2. 再破 0.2m 并出现 kill_bonus 样本(证明端到端击杀闭环)
3. 训练期 `kills > 0`(证明开火门控/课程生效,非仅靠 eval 强制开火)
4. Phase 级 `win_rate > 0`

**成功判据**:eval `min_aim_dist < 0.2m` 且训练 `kills > 0`

---

## 9. 复发的元模式(本项目第三次"方向对、机制错")

| 诊断 | 方向 | 真实机制 | 漏掉的关键 |
|---|---|---|---|
| DIAG_WIN_RATE_ZERO(导弹) | 对(episode 太短) | 244.4 m/s 而非 62500、需 49 万步而非 600 | 跑在没接 speed_ms 的旧单体上,YAML 是死配置 |
| LASER_CRITIC_POSTMORTEM | 对(有精度鸿沟) | 平台区零梯度 + 半径错配 + 开火脱钩,非探索噪声 | 混淆 train/eval 噪声;kill_radius 打错对象(drone vs laser);漏掉 radar_latents=0 |
| 本报告 | (待验证) | — | — |

**共性**:诊断停在"看得见的症状层",没有 ① 把奖励/物理公式代入数值算梯度曲面,② 核对"哪段代码真正生效"(单体 vs 模块、drone vs laser、train vs eval、配置键 vs 代码默认值、obs 通道是否实际有数据)。

**建议固化为每次诊断的强制步骤**:
1. 把奖励/物理公式代入典型数值,画出"距离/时间 → 奖励/梯度"曲线,确认目标区间有梯度
2. 区分 train 与 eval 路径,确认指标受哪些项支配(确定性 eval 与 log_std 无关)
3. 对每个建议修复,grep 确认改的对象/键在生效代码路径上
4. 对每个观测通道,打印其在训练循环里的实际数值(确认非全零、非 placeholder)

---

## 附录:关键代码位置索引

| 主题 | 文件:行 |
|---|---|
| dense 奖励公式 + cap | [training/train_laser.py:324-365](training/train_laser.py#L324)(cap 在 :352) |
| 奖励默认值(与配置不一致) | [training/train_laser.py:216-221](training/train_laser.py#L216) |
| **radar_latents = 全零** | [training/train_laser.py:262](training/train_laser.py#L262) |
| BC 辅助损失(只监督瞄准) | [training/train_laser.py:452-459](training/train_laser.py#L452) |
| log_std 退火 + "7.5m" 错注释 | [training/train_laser.py:1004-1009](training/train_laser.py#L1004)(注释 :1001) |
| 确定性 eval + 强制开火 | [training/train_laser.py:696-723](training/train_laser.py#L696)(强制开火 :719) |
| 动作→物理映射 | [radar_sim/gpu/vec_drone.py:180-183](radar_sim/gpu/vec_drone.py#L180) |
| 观测布局(敌人真值 obs[68:72]) | [radar_sim/gpu/vec_drone.py:207-278](radar_sim/gpu/vec_drone.py#L207) |
| 激光击杀判定(连续照射) | [radar_sim/gpu/vec_battlefield.py:149-216](radar_sim/gpu/vec_battlefield.py#L149)、[vec_weapon.py:46-94](radar_sim/gpu/vec_weapon.py#L46) |
| kill_radius 真实对象 | `battlefield.laser.kill_radius_m`([vec_battlefield.py:82](radar_sim/gpu/vec_battlefield.py#L82)) |
| kill_bonus → commander_rewards | [radar_sim/gpu/vec_battlefield.py:288](radar_sim/gpu/vec_battlefield.py#L288) |
| illumination 奖励(要求开火) | [radar_sim/gpu/vec_battlefield.py:299-300](radar_sim/gpu/vec_battlefield.py#L299) |
| CPI accumulator + spectrum | [training/radar_policy.py:26-56, 192-200](training/radar_policy.py#L26) |
| Radar NN 位置估计输出(注释,未监督) | [training/radar_policy.py:274-275](training/radar_policy.py#L274) |
| 环境 dt = pri | radar_sim/gpu/vec_mfar_env.py(pri=1/prf=100μs) |
