# WP-3.1 kill=0 修复 handoff spec

**日期**:2026-07-17 · **基线**:`twoteam/bc-ppo @ c515b3f`(逐文件核实)
**分工**:本 spec 由上游给出;**PRO6000 全程改代码 + 跑训练**。
**关系**:取代 `WP3_FINAL_SUMMARY.md` 的"终止到 IET floor"结论——依据见 §0(reward 最优被核实为"不开火",IET 前提不成立)。

---

## 0. 诊断结论(逐文件核实,否定 IET floor)

WP-3 4 次训练 kill=0 的**真因 = reward 错配:当前 reward 的最优策略就是"永不开火"**。

- 自博弈占 league 主体,env reward 零和 `zero_sum_reward = reward - reward.flip(dims=[1])`(`env/gpu/twoteam/twoteam_env.py:784`)→ 自博弈双方都 kill=0 时净信号 ≈ 0。
- RL 唯一可靠 dense 信号 = 非对称 shaping(仅 learning_team,`algo/_shared/pilot/twoteam/br_trainer.py:187-197`)= `+track_bonus·n_tracked − exposure_penalty·exposure`——**奖励跟踪、惩罚暴露(= 惩罚 laser 需要的 emission)、零 kill/dwell 项**。
- laser dwell-kill 链(`twoteam_env.py:689-731`)**必须持续 emission** 才能让 `radar_E` 累积到 `e_kill=2.0`;exposure 又被**双重惩罚**(env `:783` + trainer `:196`)。
- ⟹ RL 理性最优 = "跟得好 + 永不辐射 + 藏起来存活"。4 次数据吻合:survival↑、trace_P 学到、**kill 恒 0**、trace_P 低干扰反而更差(策略性被动)。**compute 修不了一个最优就是不开火的 reward(5× 无效正是此证据)。**

次要根因(核实):**PFSP 只有 f_hard 无 f_var** → 塌到 BC 单一对手(`opponent_pool.py:sample_pfsp`);**entropy 退火到 0.001** 在 kill 未现前 = 过早收敛;**BC 纯 NLL 模仿强 teacher 无 KL 锚**。

---

## 1. Fix A(THE fix)—— shaping 装正向 kill-链 + 拆 anti-kill 的 exposure 惩罚

**文件**:`br_trainer.py`(`collect_rollout`,`__init__`)+ `experiments/twoteam/wp3_train.py`(CLI)+ `run_wp2_league.py`(透传)。

### 1.1 现状代码(`br_trainer.py:185-197`)
```python
rew_lt = reward[:, learning_team] * self.reward_scale   # [E]
# WP-3 dense reward shaping (non-zero-sum, learning team only).
if self.shape_track_bonus > 0.0:
    trace_P_t = env.tracker_P[:, learning_team, :, 0, 0] + env.tracker_P[:, learning_team, :, 2, 2]
    n_tracked = ((trace_P_t < env.tau_track) & env.tracker_initialized[:, learning_team]).float().sum(dim=-1)
    rew_lt = rew_lt + self.shape_track_bonus * n_tracked
if self.shape_exposure_penalty > 0.0:
    exp_lt = info["exposure"][:, learning_team]
    rew_lt = rew_lt - self.shape_exposure_penalty * exp_lt
```

### 1.2 改法(demo)
`__init__` 新增两参(默认 0):`shape_kill_bonus: float = 0.0`、`shape_dwell_bonus: float = 0.0`;`collect_rollout` 循环**前**初始化 `prev_kills = env.team_kills[:, learning_team].clone()`。循环内改为:
```python
rew_lt = reward[:, learning_team] * self.reward_scale        # [E]
et = 1 - learning_team                                        # enemy team; radar_E[:,et]=learning_team 的 dwell 进度

# (A) dense dwell 进度(核心:让 ~20 步 dwell 可学)。绝对形式,robust;radar_E 越界会自动 kill,farm 不住。
if self.shape_dwell_bonus > 0.0:
    dwell_frac = (env.radar_E[:, et].sum(dim=-1) / env.e_kill)          # [E] ∈[0,~R]
    rew_lt = rew_lt + self.shape_dwell_bonus * dwell_frac
# (B) 显式 kill bonus(每个新 kill 一次性大奖)
if self.shape_kill_bonus > 0.0:
    now_kills = info["team_kills"][:, learning_team]
    new_kills = (now_kills - prev_kills).clamp(min=0).float()
    rew_lt = rew_lt + self.shape_kill_bonus * new_kills
    prev_kills = now_kills.clone()
# (C) track bonus 保留(小);exposure 惩罚学 kill 阶段设 0(它压制 emission=反 kill)
if self.shape_track_bonus > 0.0:
    trace_P_t = env.tracker_P[:, learning_team, :, 0, 0] + env.tracker_P[:, learning_team, :, 2, 2]
    n_tracked = ((trace_P_t < env.tau_track) & env.tracker_initialized[:, learning_team]).float().sum(dim=-1)
    rew_lt = rew_lt + self.shape_track_bonus * n_tracked
# shape_exposure_penalty 默认 0(kill 学会后再上,或只在 committed dwell 之外惩罚)
```
> **可选(理论更干净)**:dwell 用 potential-based `F=γΦ−Φ_prev`(Ng/Harada/Russell 1999,policy-invariant),`Φ=dwell_frac`,存 `self._prev_phi` 并在 `done` 归零。先用绝对形式验证 kill>0,再决定是否换 PBRS。
> **索引 assert**:实现时 assert `env.radar_E[:, et]` 在 learning_team 成功 lase 后上升(shooter=t、victim=1−t,见 `twoteam_env.py:722/728-731`)。

### 1.3 CLI 透传
`wp3_train.py:56-59` 加 `--shape-kill-bonus`(默认 50.0)`--shape-dwell-bonus`(默认 1.0);`run_wp2_league` 对应 argparse + 传给 `TwoTeamBRTrainer(...)`;学 kill 阶段 `--shape-exposure-penalty 0`。

**硬判据**:100-iter shaped run → `wp3_smoke_crossplay.py` vs BlindClassical,**RL 低干扰 kill 必须 > 0(离 0)**;对比前 4 次(全 ~0)。

---

## 2. Fix B —— PFSP 加 f_var,防塌到 BC

**文件**:`algo/_shared/pilot/twoteam/opponent_pool.py:sample_pfsp`。

### 现状 → 改法(demo)
```python
# 现状:weights = (1.0 - known_wr) ** self.pfsp_hardness_p
f_hard  = (1.0 - known_wr) ** self.pfsp_hardness_p
f_var   = known_wr * (1.0 - known_wr)                                   # AlphaStar:质量压在 ~50% 胜率对手
weights = (1.0 - self.pfsp_var_mix) * f_hard + self.pfsp_var_mix * f_var  # pfsp_var_mix∈[0,1],默认 0.5
if unknown.any():
    weights[unknown] = weights.max() if weights.max() > 0 else 1.0
```
- `__init__`/`initialize_pool`/CLI 加 `pfsp_var_mix`(默认 0.5)。
- 把已实现但没人用的 `ema_variance()`(`:158-164`)接成健康触发:ema_var<0.05 时**强制均匀采样一轮**或**注入 exploiter**。

**判据**:100-iter,pool ema_var 不破 0.05;opponent-sampling entropy(见 §4)不塌到 0。

---

## 3. Fix C —— 破自博弈弱梯度 / BC 锁 / 过早收敛

**文件**:`br_trainer.py`(entropy)、`run_wp2_league.py`(self-play)、`bc_pretrain.py`(teacher)。

- **entropy 别过早退火**:`br_trainer._entropy_coef:145-150` 现在 cosine 退到 0.001。改:**kill>0 出现前不退火**(kill-appeared flag 门控),或 `entropy_coef_min` 抬到 0.005、`entropy_decay_iters` 推后。
- **self-play 80/20**(OpenAI Five):`run_wp2_league` 对手选择处,~20% 概率用 `copy.deepcopy(br_ac)` 包 `ACCommander` 当对手(而非纯 PFSP);或加 **main-exploiter**(第二 `TwoTeamBRTrainer`,frozen_opponent 恒为 current main,每 K iter 从 main reset,checkpoint 注册进 pool)。
- **BC 锁**:`bc_pretrain.py` teacher 换 **50% BlindClassical + 50% ExtremeCommander**(避免克隆单一强 teacher);或直接**去 BC warmup** 靠 Fix A dense reward bootstrap(AlphaZero/OpenAI Five 路线);可选 soft 衰减 KL-to-BC。

**判据**:decisive-game fraction 升;对老师 exploit_gap>0;entropy 在 kill 未现前 > 0.3。

---

## 4. 检查方案(每步必做)

**A. 单元/微验证(改完立刻,不烧 GPU)**
- Fix A:8-env×50-step 微 rollout,assert(a)learning_team lase 后 `radar_E[:,et]` 上升;(b)`dwell_frac` 上升时 `rew_lt` 增量>0;(c)`team_kills` 增 1 时 kill bonus 触发一次。
- Fix B:构造 wr=[0.1(BC),0.5,0.9] 的假 pool,assert var_mix=0.5 时 0.5-胜率对手采样概率 **不为 0**(现状 f_hard-only 下它≈0)。
- 回归:`pytest tests/twoteam/` 109/109 仍 PASS;`assert_no_godview` 仍 PASS;`priv[:,4]` 归一化 assert(`br_trainer.py:251`)。

**B. Fix A 命门(100-iter,~0.4h)**
- 跑 `wp3_train.py --iters 100 --shape-kill-bonus 50 --shape-dwell-bonus 1 --shape-exposure-penalty 0`;
- `wp3_smoke_crossplay.py` → **RL 低干扰 kill > 0**(离 0);逐轮打印 train kill-rate 曲线**离 0**。
- ✅ kill>0 → 进 C;❌ 仍 0 → 先 assert 索引/信号是否真进 reward(`dwell_frac`、`new_kills` 打印),排除接线错,再谈更深问题。

**C. 全修集成(500-iter,~6h)**
- 全 Fix(A+B+C)500-iter;三方 smoke:RL vs BlindClassical vs StrongRule;
- 逐轮健康:kill-rate、decisive-fraction、opponent-sampling entropy(`-(p·log p).sum()`,从 `sample_pfsp` 的 probs)、pool_ema_var、adv_std、policy_loss≠0、无 NaN;
- **判决**:RL kill 逼近/超过 BC(低干扰 BC 0.83-0.90)→ crown 有戏 → 上 production(~5e7 steps);装了正确 kill reward 仍 kill=0 → **那才是真 IET floor(带证据)**。

**D. 运维铁律**:checkpoint `checkpoints/blind/` **严禁 /tmp**;每 100 iter `priv[:,4]` assert + 每 500 iter `assert_no_godview`。

---

## 5. 顺序(硬性)
1. **Fix A 单独先跑 100-iter 判 kill>0**(最便宜最高杠杆,几乎独立决定成败);
2. kill>0 → **Fix B + Fix C** → 500-iter 集成;
3. 逼近/超 BC → production;仍 0(正确 reward 下)→ 诚实 IET floor。

## 6. 诚实缺口 / 待定(PRO6000 扫定,不臆断)
- `shape_kill_bonus`(50?)、`shape_dwell_bonus`(1?)、是否保留极小 `shape_track_bonus`、`reward_scale` 交互——量级扫;
- 绝对 dwell vs PBRS delta——先绝对,不稳再 PBRS;
- `pfsp_var_mix`(0.5?)、80/20 比例、exploiter reset K、混合 teacher 比例——扫;
- 本方案是**重启 WP-3**,与 `WP3_FINAL_SUMMARY.md` "终止到 IET floor" 相反,依据 = reward 最优被核实为"不开火"。

---

## 附:文献锚点
- **PBRS**:Ng, Harada, Russell 1999 — potential-based shaping policy-invariant。
- **reverse curriculum**:Florensa et al. CoRL 2017。
- **demonstration reset**:Salimans & Chen 2018;Hester et al. (DQfD) 2018。
- **in-domain**:arXiv:2502.13584 — 雷达搜跟 DRL 同样 kill=0,靠 BC/shaping 修好。
- **PFSP f_hard⊕f_var + main-exploiter + league**:Vinyals et al. (AlphaStar) Nature 2019。
- **80/20 self-play + dense hand-shaped reward + zero-sum baseline**:OpenAI Five 2019。
- **对称两队 population + PBT**:Jaderberg et al. (CTF) Science 2019。
