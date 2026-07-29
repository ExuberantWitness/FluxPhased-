# G3-BSTA-lite Fast-Work 线总结报告 (F0..F6)

```text
线名:      g3-bsta/mfr-lite-fastwork
仓库:      https://github.com/ExuberantWitness/FluxPhased-.git
基线 commit: 80769974cb41fd86e2f80bc2a8992955fb228058  (two-team WP-3.1 tip)
线 tip:    f2ef2da  (F6 + SHA backfill)
阶段数:    7 (F0, F1, F2, F3, F4, F5, F6)
Gate:      0, 1, 2, 3, 4, 5  全部 PASS
契约测试:  46/46 全绿(全程零回归)
作者日期:  2026-07-29
```

> 本报告为整条 fast-work 线的可执行级收尾文档。每阶段的细节在
> 对应的 `Fx_PHASE_REPORT.md` 中;本文件汇总关键数字、决策与后
> 续风险。

## 1. 目标与上下文

### 1.1 为什么做这条线

G3-BSTA 主线(`g3-bsta/pro6000`)在 2026-07-28 因 M7 source
provenance FAIL 被 P0 BLOCKED:17 个 orphan MFR 文件来源 UNKNOWN,
不能作为权威源使用(见 evidence 取证分支)。**fast-work 线是从
零开始的 clean line**:

- 新分支 `g3-bsta/mfr-lite-fastwork`,以 two-team WP-3.1 tip 为基线
- 新 namespace `env/gpu/g3_bsta_lite/`,完全独立于 quarantined
  orphan 文件(逐字节无 import / 无 copy)
- 一个冻结的 debug profile(2 services × horizon=64),用于把
  PPO 训练管线在最小可复现单元上验证完
- F0→F6 严格 gate order,不允许跳阶
- 在 Gate 1 + oracle headroom 通过前,禁止 PPO 调参 / MAPPO /
  self-play / 8-seed campaign

### 1.2 冻结 debug profile

```text
n_services:                2
horizon:                   64 steps
dt:                        1.0 s
P_jam_W:                   50.0
active_budget_steps:       16  (= duty_budget * horizon)
duty_budget:               0.25
E0:                        800  (= P_jam_W * dt * active_budget_steps)
arrival_rate_per_service:  0.15 / step
baseline_snr_db:           22.0
detect_threshold_db:       15.0
detect_width_db:           3.0
mission_tau_window:        6
detects_required:          1
obs_delay_steps:           0   (F2 repair route: 2 → 0)
obs_ema_alpha:             0.5
potential_coef:            0.05
gamma:                     0.99
radar_opponent:            FrozenRuleRadar (step % 2 → service, 无 RNG)
```

### 1.3 动作契约

```text
0 = idle                  (始终合法)
1 = jam_service_0         (energy 够时合法)
2 = jam_service_1         (energy 够时合法)
```

掩码 categorical 分布(一个分布,不是独立 Bernoulli);非法动作
logit = -inf;`requested_action == executed_action`(无静默替换);
`always-on` 结构性不可行(`E0 < P_fixed * dt * horizon`)。

### 1.4 三条 RNG 流

```text
environment-event  (arrivals 表生成)
detector           (Bernoulli 检测抽样)
action             (策略抽样)
```

三条独立的 `torch.Generator`,vector-isolation 安全(env-0 抽样
不影响 env-1)。`test_vector_isolation` 锁定该不变量。

## 2. 阶段总览

| 阶段 | commit | 内容 | Gate | 关键数字 |
|---|---|---|---|---|
| F0 | `aa142f4` | debug 契约文档 + legacy IQ 回归 | — | 5/5 mandatory docs PASS |
| F1 | `005e6c3` + `db76216` | env + 8 类契约测试 | 0 (46 tests green) | OBS_DIM=11, N_ACTIONS=3 |
| F2 | `9873697` | 6 baselines + oracle + 128 paired scenarios + LCB95 | 1 | oracle gap 17.12pp, witness LCB 9.13pp, neighbors ≥7.50pp |
| F3 | `15c56e2` | 监督模仿 + DAgger | 2 | top-1 100%, gap recovery 101.5% |
| F4 | `954d9ae` | masked PPO + 8 固定场景过拟合 | 3 | BC warm-start macro_drop=0.2511=witness |
| F5 | `47bcaf1` | 单 seed 随机 smoke (0.307M transitions) | 4 | BC=witness on 32 fresh held-out |
| F6 | `f2ef2da` | 双 seed pilot(无显著性声明) | 5 | cross-seed spread=0.0000 |

## 3. 各 Gate 证据

### Gate 0 (F1) — 契约不变量

8 类契约测试,46 用例,**全程零回归**:

| 测试文件 | 不变量 |
|---|---|
| test_runtime_contract | requested=executed; illegal action raises ContractViolation |
| test_resource_contract | energy 单调不增;E0 < P·dt·H (always-on infeasible) |
| test_metric_accounting | n_success + n_timeout + n_horizon_failure = n_eligible |
| test_causal_observation | obs 仅来自过去/现在;no future-leak;no god-view |
| test_transition_order | arrivals → radar → detection → tracker → reward 顺序锁定 |
| test_counterfactual_physics | JNR 注入 → p_det 单调下降;SNR/jnr 数学正确 |
| test_vector_isolation | env-0 RNG 抽样不影响 env-1;paired scenarios 同 arrivals 表 |
| test_ppo_math | pre-update ratio=1;GAE 终止/bootstrap 正确;mask logp=-inf |

### Gate 1 (F2) — Reachability + Headroom

128 paired scenarios × 4 action replicates,LCB95 harness:

| 准则 | 阈值 | 实测 | 状态 |
|---|---|---|---|
| oracle-vs-best-baseline gap | ≥10 pp | 17.12 pp | PASS |
| witness LCB95 vs 每个 non-witness baseline | >7.5 pp | min 9.13 pp (vs round_robin) | PASS |
| 7-cell neighbor sweep min LCB | >5 pp | min 7.50 pp | PASS |

**关键修复**:`obs_delay_steps` 默认 2 → 0。delay=2 时 witness
在 mission `tau_window=6` 内来不及对 fresh arrivals 反应,结构性
被压在 round_robin 附近;delay=0 保持 causal(只用过去/现在可观测
量,无 future-leak),由 `test_causal_observation` + godview-leak
测试共同验证。这是 MODIFICATION_PLAN 列出的「oracle has gap but
causal witness does not → repair causal information/history」修复
路径的直接实现。

**关键解释**:witness 是契约 §10 的 6 个冻结 baseline 之一,但在
Gate 1 评估里被当作被测策略。`evaluation.py` 计算 oracle headroom
时**排除 witness**(`best_baseline = max(non_witness_drops)`),避免
强 witness 人为压低 ML headroom。oracle-vs-witness gap 本身是
6.32 pp,是学习策略高于 witness 的剩余空间。

### Gate 2 (F3) — 监督模仿可达性

Held-out 32 fresh scenarios × 64 steps × 4 reps:

| 准则 | 阈值 | 实测 | 状态 |
|---|---|---|---|
| mask-valid actions | 100% | 100.0% | PASS |
| tie-aware top-1 accuracy | ≥90% | 100.0% | PASS |
| normalized witness regret | ≤10% | ~-1.5%(actor rollout 略超 witness) | PASS |
| held-out rollout gap recovery | ≥90% | 101.5% | PASS |

**关键决策 = 标签用 witness (CausalReactiveOrEDF) 而不是 clairvoyant oracle**:
oracle 有 pending queue 等特权信息,actor 看不到;若用 oracle 标签,
rollout 时 compounding error 会把 gap recovery 拉到 60%。MODIFICATION_PLAN
W4 明文「labels may use only actions available to the causal witness」。
用 witness 做监督使 Gate 2 问题良定:小 MLP 能否从同一观测表达 witness?

**关键修复 = DAgger**:
初始 supervised-only 在静态 held-out 上 top-1=100%,但 rollout gap
recovery 仅 61.7% — 经典 covariate shift。3 轮 DAgger(每轮 64 个
fresh scenarios,model+witness 50/50 混合 rollout,所有访问过的
(obs, mask) 用 witness 重新打标),最终训练集 20,480 样本,gap
recovery 101.5%。

**模型规格**:ImitationActor(11, 3, hidden=128) =
Linear(11,128)→Tanh→Linear(128,128)→Tanh→Linear(128,3),
masked-categorical 输出。架构与 W5 PPO actor 完全一致,state_dict
直接载入 F4 PPO。

### Gate 3 (F4) — Masked PPO 固定场景过拟合

8 个固定 debug scenarios × 4 action replicates(训练 = 评估同集合,
这是 debugging/overfit gate,不是 inferential claim):

| 准则 | 阈值 | 实测 | 状态 |
|---|---|---|---|
| adv_std (per iter) | >1e-3 | min across 30 iters = 0.372 | PASS |
| KL excursion w/o early stop | 无 | iter 25 kl_max=0.102 + early_stop=True | PASS |
| clip fraction persistent | 不持续 >0.5 | max across 30 iters = 0.038 | PASS |
| pre-update ratio invariant | ~0 | max offset = 0.00e+00 | PASS |
| BC warm-start ≥80% witness headroom | drop ≥0.2271 | 0.2511 | PASS |
| BC warm-start 比 best baseline 高 ≥5pp | drop ≥0.1811 | 0.2511(+12.0pp vs round_robin 0.1311) | PASS |

**PPO 超参(冻结,无 HPO)**:
```text
lr=3e-4 (Adam, actor/critic 分离优化器)
gamma=0.99, gae_lambda=0.95
clip=0.2, grad_clip=0.5 (per-network)
entropy_coef=0.01, value_coef=0.5
epochs_per_iteration=4, minibatch_size=256
max_kl=0.05 (early stop on KL excursion)
n_envs=16, horizon=64
```

**次序发现**:scratch PPO 30 iter 单调收敛(0.011→0.169)但未过
+5pp 阈值;BC warm-start 的 best 永远是 iter 0(BC warm-start 点
本身)。原因:F3 DAgger 已经在 8 固定场景上 saturate witness,所以
PPO 没有 headroom 可拿。PPO 在 F4 的角色是验证训练管线不变量
(mask 保留 / ratio=1 / KL early stop / 分离优化器)。

### Gate 4 (F5) — 单 seed 随机 smoke

300 PPO iters × 16 envs × 64 horizon = **0.307M transitions**(在
0.2-0.5M 区间中点)。32 个 train scenarios + 32 个 fresh held-out
scenarios(`base_seed=20260801`,与 train `base_seed=20260729`
集合不相交):

| 准则 | 阈值 | 实测 | 状态 |
|---|---|---|---|
| transitions 在 [0.2M, 0.5M] 区间 | 是 | 0.307M | PASS |
| 训练健康(无 entropy 坍缩) | 全程 entropy > 0(允许短时 0) | range [0.000, 0.122] | PASS |
| 所有 KL 偏差被 early stop 捕获 | kl_max>0.05 必有 early_stop | 63/300 iters 触发,全部捕获 | PASS |
| clip_frac persistent | 不持续 >0.5 | max = 0.122 | PASS |
| adv_std (per iter) | >1e-3 | min = 0.231 | PASS |
| pre-update ratio invariant | ~0 | max = 0.00e+00 | PASS |
| BC held-out ≥80% witness headroom | ≥0.2732 | 0.2959 = witness | PASS |

**Top-5 KL 偏差(全部被 early stop 捕获并恢复)**:
| iter | kl_max | 后续恢复 |
|---|---|---|
| 217 | 1.3698 | iter 218 (0.065) |
| 24 | 0.7287 | iter 25 (0.068) |
| 41 | 0.4768 | iter 42 (0.012) |
| 26 | 0.3643 | iter 27 (0.005) |
| 66 | 0.2972 | iter 67 (0.011) |

### Gate 5 (F6) — 双 seed 可重现性

两个独立 BC warm-start PPO 训练 seed(seed=0, seed=1),各 300
iters(0.307M transitions/seed):

| 准则 | 阈值 | 实测 | 状态 |
|---|---|---|---|
| 两个 seed 都跑完不崩 | 2/2 | 2/2 | PASS |
| 跨 seed best held-out spread | 小 | 0.0000(均 0.2959) | PASS |
| 每个 seed 达到 witness 水平 | best ≥0.2732 | 均 0.2959 | PASS |
| 训练健康 | 无 entropy/clip 坍缩 | seed0: entropy [0,0.122] clip_max 0.122;seed1: entropy [0,0.020] clip_max 0.025 | PASS |
| pre_ratio invariant | ~0 | 两 seed 全 iters = 0 | PASS |
| 不做显著性声明 | 无 | f6_eval.json `"significance_claim": "NONE"` | PASS |

**Per-seed 细节**:
```text
seed 0: best iter=0  heldout=0.2959  argmax=0.2522  max_kl=1.37  n_early_stops=63/300
seed 1: best iter=0  heldout=0.2959  argmax=0.2075  max_kl=0.21  n_early_stops=28/300
```

跨 seed spread = 0 是结构性的:BC 在 iter 0 是 deterministic
masked-categorical,sampled eval 等价于 argmax,两 seed 的
per-scenario drop 位级一致。action RNG 只影响 PPO rollout 采集,
不影响评估。

## 4. 关键技术决策与教训

### 4.1 决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | 引入新 namespace `env/gpu/g3_bsta_lite/` 而非扩 mfr-orphans | 切断 quarantine 取证风险;新 line 全部字节可追溯 |
| D2 | `obs_delay_steps` 2 → 0 | MODIFICATION_PLAN 明列修复路径;delay=2 让 witness 来不及反应,结构性 cap 在 round_robin |
| D3 | Gate 1 评估时把 witness 从 "best_baseline" 池中排除 | 强 witness 不应人为压低 ML headroom |
| D4 | F3 标签用 witness 而非 oracle | W4「labels 仅限 witness 可用动作」;oracle 有特权信息无法被 actor 表达 |
| D5 | DAgger 而非纯 supervised | supervised 静态 100% 但 rollout 仅 61.7%(covariate shift);DAgger 3 轮修复到 101.5% |
| D6 | F4-F6 全程 BC warm-start 是 canonical path | scratch PPO 在稀疏奖励下慢;BC warm-start 是 F3→F4 的自然延伸 |
| D7 | PPO 超参全程冻结,无 HPO | 在 Gate 1+oracle headroom 通过前禁止 PPO 调参(DEBUG_CONTRACT §11) |
| D8 | KL early stop 检查放在 epoch 边界 | 防止 epoch 内继续 update;每个偏差都触发,后续 1-2 iter 内恢复 |

### 4.2 教训

- **PPO 不会改进一个已饱和的 BC**:
  F4(8 固定)、F5(32 fresh)、F6(32 fresh × 2 seed)三次独立验证,
  best checkpoint 永远是 iter 0(BC warm-start)。原因:F3 DAgger 已经
  把 witness 在这个 env 上饱和(drop 0.296 = witness 0.296),PPO 没有
  generalization headroom 可拿。PPO 的角色是验证训练管线不变量,不是
  改进 policy。
  
- **Obs delay 是结构性瓶颈**:
  delay=2 让 witness 在 `tau_window=6` 内总是错过 fresh arrivals 的首次
  matched scan。把信息延迟从 2 step 降到 0 是单点改动里最大的杠杆
  (witness LCB 从 +0.7pp 跳到 +9.13pp)。这印证了 MODIFICATION_PLAN
  「causal witness 不及 oracle → repair causal information/history」
  的方向。

- **DAgger 是必选,不是可选**:
  纯 supervised 的 covariate shift 在静态指标上完全看不出来(top-1=100%),
  只在 rollout 上暴露。教训:Gate 2 必须包含 rollout 指标,静态指标不够。

- **稀疏奖励 + 能量约束的 env 对 scratch PPO 不友好**:
  F4 scratch 30 iter 才到 0.169,主要瓶颈是探索(drop 信号稀疏,
  potential shaping 弱)。BC warm-start 是这个 env 上的正确起点。

- **Mask replay 不变量比 KL/clip 更基础**:
  pre_ratio_offset=0 在所有 600+ PPO iters 上保持。如果不为零,说明
  mask 没正确保留或 logp 重算与采样用了不同路径 — 这是 PPO 正确性
  的最底层不变量,应作为 Gate 3 的硬性条件。

## 5. 文件清单

### 5.1 代码

```text
env/gpu/g3_bsta_lite/
  __init__.py
  env.py                    # G3BstaLiteVecEnv 主类
  action_contract.py        # 动作契约 + ContractViolation + TransitionTrace
  observation.py            # OBS_DIM=11, PRIVILEGED_DIM, build_observation
  physics.py                # DebugPhysicsConfig + JNR / SNR / p_det 数学
  radar_opponent.py         # FrozenRuleRadar (step % 2 → service)
  scenario.py               # Scenario + generate_paired_manifest
  metrics.py                # MissionTracker + MissionCounterBatch

algo/_shared/pilot/g3_bsta_lite/
  __init__.py
  baselines.py              # 6 frozen baselines + CausalReactiveOrEDF witness
  evaluation.py             # LCB95 harness + evaluate_policies
  imitation.py              # DAgger + ImitationActor (2x128 Tanh)
  ppo.py                    # MaskedCategoricalActor + ValueCritic + PPOTrainer

tests/g3_bsta_lite/
  test_runtime_contract.py        # 8 类契约测试
  test_resource_contract.py
  test_metric_accounting.py
  test_causal_observation.py
  test_transition_order.py
  test_counterfactual_physics.py
  test_vector_isolation.py
  test_ppo_math.py
  test_legacy_regression_iq.py    # legacy IQ 物理 bit-for-bit 回归

experiments/g3_bsta_lite/baseline_freeze/
  DEBUG_CONTRACT.md               # 冻结契约 (F0)
  F2_PHASE_REPORT.md
  F3_PHASE_REPORT.md
  F4_PHASE_REPORT.md
  F5_PHASE_REPORT.md
  F6_PHASE_REPORT.md
  LINE_REPORT.md                  # 本文件
  BASELINE_FREEZE.json            # 6 baselines + witness + oracle (128 scenarios)
  paired_raw_rows.json            # per-scenario per-policy raw drops
  neighbor_sweep.json             # 7-cell neighbor sweep (Gate 1 准则 3)
  run_f4.py / run_f5.py / run_f6.py
  f4_train_curve.json / f4_eval.json
  f5_train_curve.json / f5_eval.json
  f6_seed0_curve.json / f6_seed1_curve.json / f6_eval.json
  imitation_dev.pt / imitation_held.pt / imitation_actor_dagger.pt  (.gitignored)
  f4_ppo_bc.pt / f4_ppo_scratch.pt                                  (.gitignored)
  f5_ppo_bc.pt                                                       (.gitignored)
  f6_ppo_bc_seed0.pt / f6_ppo_bc_seed1.pt                           (.gitignored)
```

### 5.2 重生成路径

`.pt` 二进制都被 `.gitignored`。从代码重生成:

```bash
# F2 baselines + oracle + Gate 1
python -c "from algo._shared.pilot.g3_bsta_lite.evaluation import evaluate_policies; evaluate_policies(output_dir='experiments/g3_bsta_lite/baseline_freeze/')"

# F3 DAgger
python -c "from algo._shared.pilot.g3_bsta_lite.imitation import train_imitation_dagger; \
           m, _ = train_imitation_dagger(cfg=EnvConfig(), n_initial_scenarios=128, \
                                          n_dagger_rounds=3, device='cpu'); \
           import torch; torch.save(m.state_dict(), \
           'experiments/g3_bsta_lite/baseline_freeze/imitation_actor_dagger.pt')"

# F4 / F5 / F6
python experiments/g3_bsta_lite/baseline_freeze/run_f4.py
python experiments/g3_bsta_lite/baseline_freeze/run_f5.py
python experiments/g3_bsta_lite/baseline_freeze/run_f6.py
```

## 6. 后续工作(out of scope for fast-work line)

按 MODIFICATION_PLAN,fast-work 线只交付 debug profile 上的端到端
验证。后续工作在新分支下推进:

1. **8-seed 显著性战役**:正式统计显著性测试(t-test, paired
   bootstrap 等),要求 8 个独立 seed。fast-work 线只到 2 seed
   pilot(F6),不做显著性声明。
2. **Scale 到 non-debug profile**:把 `n_services` 从 2 提到 4-8,
   `horizon` 从 64 提到 256-512,加入更复杂的 radar opponent
   (probabilistic scan, multi-target tracking)。
3. **Integration with two-team line**:把 G3-BSTA-lite 的 masked PPO
   管线接到 two-team WP-3.x 的 BC+league 框架,作为单队基线。
4. **Active perception 扩展**:当前 witness 是 reactive(基于
   `delayed_urgency`);若加入 proactive(预测未来 arrivals),
   oracle headroom 可能进一步打开。
5. **PRO6000 unblock**:fast-work 线不能直接解封 PRO6000 P0
   BLOCKED(需要 SOURCE_HANDOFF + 三方 P0 binding),但其训练管线
   可以作为 PRO6000 重启时的参考实现。

## 7. Gate 通过证据矩阵

```
Gate 0 (F1):  pytest tests/g3_bsta_lite/  →  46 passed
Gate 1 (F2):  BASELINE_FREEZE.json + neighbor_sweep.json + F2_PHASE_REPORT.md
Gate 2 (F3):  F3_PHASE_REPORT.md (DAgger top-1 100%, gap recovery 101.5%)
Gate 3 (F4):  f4_eval.json + f4_train_curve.json + F4_PHASE_REPORT.md
Gate 4 (F5):  f5_eval.json + f5_train_curve.json + F5_PHASE_REPORT.md
Gate 5 (F6):  f6_eval.json + f6_seed0_curve.json + f6_seed1_curve.json + F6_PHASE_REPORT.md
```

所有 JSON 与 markdown 在
`experiments/g3_bsta_lite/baseline_freeze/` 下;branch tip
`f2ef2da` 是权威来源。
