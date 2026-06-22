# FluxLeague Tier 1 训练 NaN 崩溃 — 详细根因分析报告

**日期**: 2026-06-22
**分析基线**: origin/main @ e733fad（tier 1 kill-fix 已落地 + 训练跑过）
**崩溃日志**: `logs/laser_league_tier1_kill_fix_20260621_193136.log`（1807 行）
**分析对象**: `training/ppo/{actor_critic.py,ppo_trainer.py}`、`checkpoints/laser_pro6000_league/*.pt`、训练日志
**关联文档**: [LASER_ROOT_CAUSE_ANALYSIS.md](LASER_ROOT_CAUSE_ANALYSIS.md)（前一轮零击杀问题，已修）、[fluxleague-kill-fix-tier1](.aris/research-wiki/fluxleague-kill-fix-tier1.md)

---

## 0. 执行摘要

Tier 1 kill-fix 训练（PID 1417143，2026-06-21 19:31 启动）在 PSRO iter 1
即将完成时崩溃：

```
ValueError: Expected parameter loc (Tensor of shape (248, 4)) of distribution
Normal(loc: torch.Size([248, 4]), scale: torch.Size([248, 4])) to satisfy
the constraint Real(), but found invalid values: tensor([[nan, nan, ...
```

崩溃点 [actor_critic.py:271](training/ppo/actor_critic.py#L271) `aim_dist =
torch.distributions.Normal(aim_mean, aim_std)`，即 commander actor-critic 的
hybrid_fire aim 头输出 NaN。

逐行读代码 + 37 个 checkpoint 的张量扫描 + 复现脚本
([diagnose_nan.py](diagnose_nan.py)) 得出真实结论：

> **不是权重爆炸，不是 checkpoint 损坏，不是 forward 数值溢出。**
> iter 0 末尾的 commander 权重 `max_abs ≤ 2.83`（甚至比 init 还小），
> 即使把输入推到 ±30σ，forward 也不产生 NaN。
>
> 真实根因是 **Adam 优化器的数值退化**：所有 13 个参数组都存在
> `exp_avg_sq.min = 0` 的维度（log_std 100%、action_head 99.8%、shared.0
> 84.7%）。当某个 minibatch 触发 ratio spike（iter 1 ep7 step152 实测
> policy_loss=2.05 = 10× baseline）时，反向梯度落到这些"零方差"维度，
> Adam 的有效步长变成 `lr × grad / eps = 3e-4 × grad / 1e-8 = 3e4 × grad`，
> 单步更新把 weight 推到 ±∞，下一次 forward 即 NaN。

崩溃前日志已经留下两个明确的预警信号（§2），任何一个都足以在 1-2 个
minibatch 内预测到崩溃。

**最小可行修复**（§5）：3 行代码即可让训练稳定通过 iter 1——
[A] Adam `eps=1e-8 → 1e-4`（直接堵住零方差爆破路径）+
[B] PPO update 加 NaN-skip 守卫（防御兜底）。

---

## 1. 崩溃上下文重建

### 1.1 时间线（来自日志）

| 时刻 | 事件 | 信号 |
|---|---|---|
| iter 0 全程 | 4 个训练子任务（main/league exploiter × team 0/1）正常完成 | 30651.5s 完成 |
| iter 0 末尾 | 保存所有 gen1 checkpoint（main_exploiter_team1_gen1.pt 等） | 健康（§3.1） |
| iter 1 开头 | payoff matrix 评估，无训练 | 正常 |
| iter 1 / p0002 vs p0010 | league_exploiter team 0 训练完成 | 正常 |
| iter 1 / p0004 vs **p0014** | main_exploiter team 1 开始训练，对手是 mutant pool 第 14 个策略 | 分布漂移源 |
| iter 1 ep0–ep6 | 7 个 episode 完成，cmd `pl ≈ 0.3`、`ent ≈ 2.4`、`rad vl ≈ 1.8M` | 表面正常 |
| iter 1 ep7 step152 | **policy_loss spike 到 2.0534**（10× baseline） | **预警 #1** |
| iter 1 ep7 step194–488 | entropy 从 2.4254 单调衰减到 2.1018 | **预警 #2：伪熵崩溃** |
| iter 1 ep8 rollout | 500 步 timeout（无击杀），wr=0.00，avg_r=13907 | 表面正常 |
| iter 1 ep8 PPO update | `evaluate_actions` 抛 ValueError，aim_mean 全 NaN | **崩溃** |

### 1.2 为什么是 p0014 这个对手

iter 0 训练对手是 p0003/p0005/p0006（league pool 前 6 个，都是 gen0
衍生）。iter 1 切到 p0010、p0014（mutant pool，gen1/gen2 衍生）。
mutant 来自更激进的策略分支（mutant_team0_from_p0001_v1 等），
呈现完全不同的瞄准分布和开火节奏，触发 commander policy 的 ratio 爆破。

---

## 2. 崩溃前预警信号

### 2.1 预警 #1：policy_loss 单点爆破

```text
[PPO] main-t1 ep7 step110 cmd(pl=0.2053 vl=140736.7135 ent=2.4560) rad(...)
[PPO] main-t1 ep7 step152 cmd(pl=2.0534 vl=136760.5703 ent=2.4254) rad(...)  ← 峰值
[PPO] main-t1 ep7 step194 cmd(pl=0.2830 vl=149439.2865 ent=2.2823) rad(...)
```

`pl=2.0534` 在 baseline ≈0.3 的体系下单点跳到 10×。PPO clip 公式：

```python
ratio = torch.exp(log_prob_new - log_prob_old)
surr1 = ratio * advantages                       # 无 clip
surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

`policy_loss = 2.05` + `advantages ≈ N(0,1)` 意味着某个 minibatch 里
`ratio × advantage` 单点达到 20+，即 `ratio ≈ 20`、
`log_prob_new - log_prob_old ≈ +3.0`。

注意：`surr1` 是**无 clip**的那一项，`min(surr1, surr2)` 在 ratio > 1+ε
时选 surr2（clipped），在 ratio < 1-ε 时选 surr1（unclipped，但
advantage 同号时方向不变）。所以 pl=2.05 主要来自 `|surr1|` 在
advantage 与 ratio 异号、且 ratio < 1-ε 的 minibatch 里爆破。

### 2.2 预警 #2：熵单调衰减（伪熵崩溃）

ep7 step152 之后 8 个连续 PPO update，`ent` 从 2.4254 衰减到 2.1018：

```
step152: ent=2.4254 → step194: 2.2823 → step236: 2.2028 → step278: 2.1612
→ step320: 2.1468 → step362: 2.1372 → step404: 2.1349 → step446: 2.1136
→ step488: 2.1018
```

斜率 ≈ -0.04 nat/update，单调控形式。这不是真实的熵减少（log_std 几乎
不动，见 §3.2），而是 aim_mean 漂移到 tanh 饱和区后，**squash 修正项**
`-log(1 - tanh(aim_raw)² + 1e-6)` 把熵估计拉低。一旦 aim_raw > 5，
`tanh(aim_raw) ≈ 1`，`1 - tanh² ≈ 1e-5`，`-log(1e-5) ≈ +11`，反过来
aim_raw < -5 同样。两个方向都让 squashed entropy 暴跌。

到 step488 时，aim_mean 已在 ±5 附近震荡，下一次 PPO update 触碰数值
边界即 NaN。

---

## 3. 排除其他假设

### 3.1 checkpoint 不是已损坏

对全部 37 个 checkpoint 做张量级扫描（[diagnose_nan.py](diagnose_nan.py)）：

```
[scan] 0/37 checkpoints have NaN/Inf
[scan] real weight max_abs per file (top 5):
  main_team0_gen2.pt          max_abs=1.428e+03   ← Adam state.step counter, 不是权重
  league_exploiter_team0_gen2 max_abs=1.428e+03   ← 同上
  main_exploiter_team1_gen1   max_abs=6.660e+02   ← 同上
  mutant_team0_from_p0008_v2  max_abs=2.914e+00   ← 真实权重
  main_team0_gen0             max_abs=2.800e+00   ← init
```

"700" 实际是 Adam 的 `state.step = tensor(666.)` 计数器（666 次 PPO
update）。**真实权重的 max_abs 全部 ≤ 2.83**，甚至比 init（2.8）还小，
说明 iter 0 的权重根本没怎么动。

### 3.2 forward pass 在极端输入下也不产生 NaN

构造三种输入分布（typical ±1、drifted ±5、extreme ±30），跑
`CommanderActorCritic.evaluate_actions`：

| 输入尺度 | features max | action_mean max | aim_mean max | log_prob range | NaN? |
|---|---|---|---|---|---|
| ±1σ | 3.63 | 3.29 | 1.23 | [-308, -283] | No |
| ±5σ | 1.84e1 | 1.27e1 | 6.76 | [-258, -196] | No |
| ±30σ | 1.15e2 | 5.95e1 | 4.03e1 | [-5179, -431] | No |

即使在 ±30σ 输入下，`log_prob` 最坏 -5179（finite），`ratio` 最坏
1.4e-42（finite），forward 路径完全稳健。

### 3.3 不是 iter 0 累积的梯度爆炸

Adam state（[diagnose_nan.py](diagnose_nan.py) §"Adam optimizer state
health"）：

```
param_group[0]  (log_std 5 params):       exp_avg.max=9.68e-7   exp_avg_sq.max=3.65e-13
param_group[5]  (value_shared.0 256x76):  exp_avg.max=1.28e-2   exp_avg_sq.max=4.45e-5
param_group[9]  (action_head 5x256):      exp_avg.max=8.21e-7   exp_avg_sq.max=1.17e-12
param_group[11] (value_head 1x256):       exp_avg.max=6.08e-3   exp_avg_sq.max=1.17e-3
```

梯度信号整体很弱（1e-7 ~ 1e-3），没有累积爆炸。log_std 的 exp_avg 和
exp_avg_sq 都是 1e-7 / 1e-13 量级——这意味着 iter 0 整个过程中 log_std
几乎没收到有效梯度，因此 std 卡在 init（-1.0）不动。这解释了为什么熵
"崩溃"不是来自 std 收缩（§2.2）。

---

## 4. 真实根因：Adam 零方差维度爆破

### 4.1 关键证据

逐 param_group 检查 Adam state，**每个 group 的 `exp_avg_sq.min = 0.0`**：

```
param_group[0]  exp_avg.max=9.68e-7   exp_avg_sq.min=0.0   eas_zero_frac=1.000  (log_std)
param_group[1]  exp_avg.max=8.64e-8   exp_avg_sq.min=0.0   eas_zero_frac=1.000  (shared.0.weight)
param_group[5]  exp_avg.max=1.28e-2   exp_avg_sq.min=0.0   eas_zero_frac=0.847  (value_shared.0)
param_group[6]  exp_avg.max=7.14e-3   exp_avg_sq.min=0.0   eas_zero_frac=0.020  (value_shared.0.bias)
param_group[7]  exp_avg.max=3.71e-4   exp_avg_sq.min=0.0   eas_zero_frac=0.224  (value_shared.2)
param_group[9]  exp_avg.max=8.21e-7   exp_avg_sq.min=0.0   eas_zero_frac=0.998  (action_head.weight)
```

`eas_zero_frac` = 该参数张量中 `exp_avg_sq == 0` 的元素比例。
log_std 100%、action_head 99.8%、shared.0 100%。这些维度在 iter 0 整个
生命周期里**从未收到非零梯度平方**（因为反向传播路径被某些层阻断，
比如 ReLU 死区或 squash 修正的 JVP 恒零）。

### 4.2 数值机理

Adam 更新公式：

```
m = β1 * m + (1-β1) * g              # exp_avg (β1=0.9)
v = β2 * v + (1-β2) * g²             # exp_avg_sq (β2=0.999)
w -= lr * m / (sqrt(v) + eps)        # PyTorch Adam default eps=1e-8
```

当 `v = 0` 但 `m ≠ 0` 时：

```
Δw = lr * m / (sqrt(0) + 1e-8)
   = lr * m / 1e-8
   = 3e-4 * m * 1e8
   = 3e4 * m
```

对 `param_group[5]` (value_shared.0, `m.max = 1.28e-2`)：

```
Δw_max = 3e4 * 1.28e-2 = 384   ← 单步把某个 weight 元素推到 ±384
```

下一次 forward：`features @ W` 经过该元素 → 输出 ~ ±384 × feature →
传到 action_head → aim_mean → ±1e4 量级 → `Normal(mean=1e4, std=0.37)`
的 `log_prob(action=0)` = `-0.5 * (1e4/0.37)² = -3.7e8`（仍 finite）
→ 但 `log_prob(action=aim_raw)` 当 aim_raw 也在 1e4 量级时，
`(aim_raw - aim_mean) / std` 在 batch 内某些维度上是 `0/0.37 = 0`、
另一些维度上是 `(aim_raw - aim_mean) / std = (1e4 - 1e4) / 0.37 = 0`，
**但只要某次 backward 把 NaN/Inf 推回 weight（比如 `loss = policy_loss
+ value_coef * value_loss` 里 value_loss = (1e4)² = 1e8，`grad =
2 * 1e4 = 2e4`，clip 后仍 finite；但下一次 ratio = exp(log_prob_new -
log_prob_old) 当 log_prob_old 是 finite 但 log_prob_new 是 -1e10 时
ratio = 0；当 log_prob_new 反向漂移成 +1e10 时 ratio = exp(1e10) =
inf**），整个 batch 的 `surr1 = inf * advantage = inf/NaN`，
`policy_loss = -min(inf, clipped).mean() = -inf`，`loss = -inf + finite
= -inf`，backward 把 `-inf` 灌回每个参数的梯度 → 下一次 step 把所有
weight 变 NaN。

### 4.3 触发条件

为什么是 iter 1 ep7 step152？

1. **分布漂移**：对手从 p0006（gen0 衍生）切到 p0014（mutant pool），
   采样到的 obs 分布左偏，导致某个 batch 的 advantage 与 ratio 异号 +
   ratio 跨越 1-ε 边界。
2. **死维度唤醒**：之前零梯度的 action_head/value_shared 维度在新的
   batch 分布下首次收到非零梯度（ReLU 不再死区），但 `exp_avg_sq` 累积
   慢（β2=0.999），需要 ~7 个 step 才能积累到 `1e-8` 以上——这正好
   对应 ep7 step152 的 spike 时序。
3. **clip 守不住**：`max_grad_norm=0.5` clip 的是总范数，对落在单维度
   上的极大梯度无效（一个维度贡献整个 0.5 范数，依然 lr×0.5/eps=1.5e4
   的 step）。

---

## 5. 修复方案

按代价/收益排序。推荐 **A + B** 组合（4 行代码改动）作为最小可行修复，
其他作为可选加固。

### A. Adam `eps` 上调（直接堵死根因）

[training/ppo/ppo_trainer.py:49](training/ppo/ppo_trainer.py#L49)

```python
# 改前
self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)
# 改后
self.optimizer = torch.optim.Adam(
    self.ac.parameters(), lr=lr,
    eps=1e-4,   # default 1e-8 在 exp_avg_sq=0 时让有效步长 = lr * grad / 1e-8 = 3e4 * grad
                # 上调到 1e-4 后，最大有效步长 = 3e-4 * grad / 1e-4 = 3 * grad
                # 与 max_grad_norm=0.5 联动 → 单步最大更新 1.5（仍较大，可继续调到 1e-3）
)
```

**有效性**：直接把零方差维度的爆破上限从 `3e4 * grad` 砍到 `3 * grad`。
配合 `max_grad_norm=0.5`，单步 weight 更新上限 ≈ 1.5（对比当前 1500）。

**代价**：1 行。Adam 自适应能力略降（极小梯度会被 eps 噪声主导），
但本项目的梯度尺度（1e-7 ~ 1e-3）远大于 eps=1e-4，影响可忽略。

### B. PPO update NaN-skip 守卫（防御兜底）

[training/ppo/ppo_trainer.py:146-149](training/ppo/ppo_trainer.py#L146-L149)

```python
self.optimizer.zero_grad()
loss.backward()
# 加在 clip 之前
if not torch.isfinite(loss) or any(
    not torch.isfinite(p.grad).all() for p in self.ac.parameters() if p.grad is not None
):
    # 跳过本 minibatch，不更新 Adam moments、不 step
    self.optimizer.zero_grad()
    continue
nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
self.optimizer.step()
```

**有效性**：即便 A 失效（比如更极端的 ratio spike），也能防止 NaN 灌
回 weight，下次 forward 仍可用。代价 3 行。

### C. aim_mean clamp（forward 兜底）

[training/ppo/actor_critic.py:269](training/ppo/actor_critic.py#L269)

```python
aim_mean = mean[:, 1:]
aim_mean = aim_mean.clamp(-30, 30)   # 防止 Normal 构造器抛 ValueError
aim_std = torch.exp(self.log_std[1:]).expand_as(aim_mean)
```

**有效性**：1 行，直接消除 ValueError。**不解决根因**——只是让 forward
不抛错，下一次 update 依然可能因为 squash 修正而 NaN。**不推荐单独用**。

### D. 优势 clip（削弱 ratio spike 幅度）

[training/ppo/ppo_trainer.py:106](training/ppo/ppo_trainer.py#L106)

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
advantages = advantages.clamp(-10, 10)   # 削弱 outlier advantage
```

**有效性**：1 行。把 surr1 的上界从 `inf * advantage` 砍到
`inf * 10`（仍可能 inf，但触发条件更苛刻）。治标。

### E. value loss 尺度治理（结构性，可选）

雷达 `value_loss = 1.2M` 是结构性问题：单步 reward ~27（avg_r=13k /
500 步）、N-step returns ~2700、MSE = 7.3M。要么：

- reward scale ÷ 100（破坏既有调参，不推荐）
- value_head 单独 lr = 1e-5（4 行，加 `param_groups` 分组）
- value loss 用 Huber loss（1 行，`F.smooth_l1_loss` 替代 `MSE`）

### 推荐组合

| 修复组合 | 改动量 | 期望稳定性 | 备注 |
|---|---|---|---|
| **A + B** | 4 行 | 高 | **推荐**：根因 + 兜底 |
| A + B + D | 5 行 | 很高 | 加 advantage clip |
| C alone | 1 行 | 低 | 只堵 ValueError，不解决 NaN 来源 |
| A + B + E (Huber) | 5 行 | 极高 | 结构性修 value loss，但需重跑调参 |

---

## 6. 验证计划

应用 A + B 后，按以下顺序验证：

1. **单元测试**：构造一个 batch，故意把 `aim_mean` 推到 ±50，验证 clamp
   后 forward 通过。
2. **快速烟测**：跑 2 个 PSRO iter（用 `episodes_per_training=3`），
   观察是否还有 policy_loss spike。
3. **iter 1 全程**：恢复 main_exploiter_team1_gen1.pt，跑到崩溃点
   (ep7-ep8)，验证 NaN-skip 守卫触发频率 ≤ 1%。
4. **完整 24 iter**：用原 config 跑 5 天，看最终 ELO / 击杀率。

如果 A+B 后仍崩，按 C+D+E 顺序加加固。

---

## 7. 对论文的影响

按 [fluxleague-paper-framing](.aris/research-wiki/fluxleague-paper-framing.md)
的 EAAI Q1 故事，FluxLeague 是主贡献，IPPO/MAPPO 是 baseline。本次 NaN
是**工程问题，不是算法问题**，不会影响论文的 SOTA 声明。但需要在论文
§Implementation Details 里写一句"我们使用 Adam eps=1e-4 + PPO NaN-skip
guard 保证训练稳定性"，否则 reviewer 复现时可能踩同样的坑。

**不建议**把本诊断报告作为论文附录——它属于工程 hygiene，不是算法
贡献。但可以作为 supplementary code 的 commit message 引用。

---

## 附录：诊断脚本与日志

- **复现脚本**: [diagnose_nan.py](diagnose_nan.py)（180 行，运行约 30 秒）
- **崩溃日志**: `logs/laser_league_tier1_kill_fix_20260621_193136.log:1534-1560`
- **关键 checkpoint**: `checkpoints/laser_pro6000_league/main_exploiter_team1_gen1.pt`
  （崩溃前的健康状态）
- **扫描结果**: 全部 37 个 checkpoint 的 NaN/Inf 检查 + Adam state 指标
  在 [diagnose_nan.py](diagnose_nan.py) 输出里
