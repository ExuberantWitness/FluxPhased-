# 计划: Bet B — Domain-Randomization RL → Off-Nominal Robustness(AppInt 主注)

**日期**: 2026-07-14
**前序**: WP2_STEP0_5_V1_REPORT.md(V1 3/3 exploit FAIL)、G0_BC_PPO_REPORT.md(G0 #3 BC+PPO 打平)
**目标会议**: AppInt(主)/ IET(失败地板)
**状态**: spec,等用户批准开 Step 0

---

## 1. 定位 + EV

### 1.1 为什么转 Bet B

V1(3/3 exploit FAIL)+ G0 #3(BC+PPO tie,CI 跨 0)+ rule 设计分析(71% track + 三机制覆盖) = **三线独立证据,rule 在 nominal design point 近 Nash**。

这关上了 "RL 在 nominal 打赢 rule" 这条路(WP2 league 干净-TAES 概率从 20-25% 压到 ~10%),但打开了一条更值钱的路:

**核心杠杆 — 近 Nash at nominal ≠ 近 Nash globally**:
- Rule 是为 nominal 手调(阈值 duck=60 / jam_detect=0.30 / hop=6 / 71% track / jam_gain=6.0 / exposure_gain=200 等),最优但**脆**
- Domain-randomized RL 学的是 *覆盖分布* 而非点最优 → off-nominal 优雅退化
- AppInt 标准 applied-ML 故事:"学习式免手调 + 泛化超专家设计点"

### 1.2 EV 表(V1 后更新)

| Bet | 价值 | P(成功) | EV | 状态 |
|---|---|---|---|---|
| **B.** 域随机化 RL → off-nominal 鲁棒 → AppInt | 中 | ~55% | **最高** | 主注 |
| A. league → 干净 TAES | 最高 | ~10%(V1 后压) | 低 | 放弃(V1 + G0 #3 已关) |
| C. IET 地板(3 线近 Nash 证据) | 低-中 | ~88% 在手 | 已保 | 后备 |

### 1.3 关键判据

- **G1-robust**: DR-RL 在 off-nominal grid 的 **median WR > Rule 的 median WR**,bootstrap CI 排除 0
- **G1-clean**: DR-RL **@ nominal 不显著输 Rule**(WR ≥ -0.05,允许 ≈ 平,不允许 cliff)
- **G2**: DR-RL 的 **5%-tail WR > Nom-RL 的 5%-tail WR**(证 DR 加值,不是 RL 本身就行)

### 1.4 硬停

PASS → 写 AppInt;FAIL → 退 IET;**不开第 4 轮 DR 调参**。

---

## 2. 关键复用(已建,不重写)

| 资源 | 路径 | 用途 |
|---|---|---|
| BC 预训练 | `algo/_shared/pilot/twoteam/bc_pretrain.py` | 起点 policy(rule-equivalent,避免从零学) |
| PPO+GAE+CTDE | `algo/_shared/pilot/twoteam/br_trainer.py` | DR 训练内循环 |
| Commander AC | `algo/_shared/pilot/twoteam/commander_actor_critic.py` | 同网络 |
| 两队 env | `env/gpu/twoteam/twoteam_env.py` | 构造参数支持 DR(rich physics knobs) |
| 强规则 | `algo/_shared/baselines/twoteam_strong_rule_commander.py` | nominal baseline,off-nominal 也跑 |
| Cross-play harness | `algo/_shared/pilot/twoteam/run_g0_gate.py` | bootstrap CI |
| Extreme + exploit strategies | `algo/_shared/pilot/twoteam/extreme_commanders.py` | OOD 对手 |

**关键事实**: env 构造参数支持丰富 DR 轴(jam_gain / range_sigma_m / sigma_q / exposure_gain / radar_separation_m / 等),但 **set at construction,无 runtime reconfig**。DR 在 iter 边界重建 env。

---

## 3. DR 轴选择(基于 env API 真实可配参数)

### 3.1 DR 轴 + 分布

| 轴 | env 参数 | nominal | DR 范围 | 含义 |
|---|---|---|---|---|
| EW 强度 | `jam_gain` | 6.0 | Uniform[3.0, 9.0] | 0.5× ~ 1.5× nominal |
| 传感器精度 | `range_sigma_m` | 0.05 | Uniform[0.02, 0.10] | 更好 ~ 更坏 sensor |
| 目标动力学 | `sigma_q` | 2.0 | Uniform[1.0, 4.0] | 慢 ~ 快 target |
| Exposure 灵敏度 | `exposure_gain` | 200.0 | Uniform[100, 400] | 不敏感 ~ 敏感 |
| 编队间距 | `radar_separation_m` | 1500 | Uniform[1000, 2000] | 紧 ~ 宽 |
| 交战距离 | `map_size_m` | 8000 | Uniform[6000, 10000] | 近 ~ 远 |
| 几何模式 | `geometry` | MIRROR | {MIRROR, RANDOM} | 镜像 vs 随机 |
| 对手策略 | — | StrongRule | {rule, pure_track, pure_jam, balanced} | 4 种 |

**每 episode 开头**: 从分布中采样 → 配置 env + 选对手。

### 3.2 边界依据

- `jam_gain` [3, 9]:rule 的 hop 反应在 jam=6 时最优,3 和 9 都偏离但合理
- `range_sigma_m` [0.02, 0.10]:传感器物理范围,4× 跨度
- `sigma_q` [1, 4]:target 机动性,CV 模型标准范围
- 其他类似 — 都是 env 已支持的合理物理范围

### 3.3 不随机化的项

- `n_teams=2, n_radars_per_team=2, n_fn=4`:env 结构,固定
- `dt=0.1, episode_steps=600`:时间结构,固定
- `freq_hop_max=8.0`:rule 也吃这个上限,固定保证公平

---

## 4. Step 0 — Rule 敏感度扫描(cheap gate,~1 day)

**Why**: Bet B 的前提是 rule 在 off-nominal 掉崖。先量,再决定 DR 训练值不值得。**这是 Bet B 的 anti-strawman gate**。

### 4.1 路径

**New** `scripts/rule_sensitivity_sweep.py`(~250 LOC)

### 4.2 Sweep grid

6 轴 × 3 levels(中心 = nominal):

| 轴 | low | nominal | high |
|---|---|---|---|
| `jam_gain` | 3.0 | 6.0 | 9.0 |
| `range_sigma_m` | 0.02 | 0.05 | 0.10 |
| `sigma_q` | 1.0 | 2.0 | 4.0 |
| `exposure_gain` | 100 | 200 | 400 |
| `radar_separation_m` | 1000 | 1500 | 2000 |
| `geometry` | MIRROR | MIRROR | RANDOM |

→ **17 grid points**(6 个 1-axis 单变量 + 6 个 pair + 4 个 triple + nominal,正交子集而非全 3^6=729)。

**或者更简单**: 6 个 1-axis sweeps(每轴 3 levels)= 6×2+1 = 13 grid points(nominal 复用)。先用这个,廉价且能识别主效应。

### 4.3 每 grid 点

- Rule(StrongRuleCommander)vs `pure_track` 固定 baseline
- 双向 100 ep × horizon=200 × n_envs=8
- bootstrap 1e4 CI on WR
- 记录:rule_WR, kill_delta, draw_rate, mean tracker_P

### 4.4 Cliff 判据

- **Cliff**: rule_WR 比 nominal 下降 > 0.15
- **Brittleness score**: cliff 数 / 总 grid 数

### 4.5 决策

| Cliff 数 | 判定 | 动作 |
|---|---|---|
| 0-1 | rule 全程稳健 | **硬停,转 IET 地板**(Bet B 前提死) |
| 2-4 | 中度脆 | DR 训练,**专攻 cliff 轴** |
| ≥5 | 高度脆 | DR 训练,全轴随机化 |

### 4.6 输出

`experiments/twoteam/rule_sensitivity_sweep.md`:
- per-grid WR 表(轴 × level)
- cliff list(哪些 grid 触发)
- brittleness score + 主效应分解
- verdict(Bet B alive / dead)

### 4.7 预算

~1 day(13 grid × 200 ep × 2 dir ≈ 5200 ep,~2-3 小时跑 + 报告)

---

## 5. Step 1 — DR 训练 spec

### 5.1 路径

**New** `algo/_shared/pilot/twoteam/run_bet_b_dr.py`(~400 LOC)

### 5.2 env 重建策略

无 runtime reconfig → **每 K iter 重建一次 env**(K=10):
- 采样新 DR config
- `env = TwoTeamVecEnv(**new_config)`
- `trainer.attach_env(env)`
- 跑 K iter PPO
- 重复

**为什么 K=10**: 每 iter 重建太慢(env init ~1s),K=10 让重建开销 < 1% 训练时间。每 K iter 内部仍是同 env,K iter 间 DR config 不同 → 期望覆盖分布。

### 5.3 训练 loop

```
[Step A] BC 预训练(复用 bc_pretrain.py)
    50K samples from StrongRule @ nominal
    → BC 15 epoch → rule-equivalent 起点策略
    save checkpoints/twoteam/bet_b_dr/iter000_bc.pt

[Step B] DR-PPO 主循环
    trainer = BRTrainer(ac, env=nominal_env, ...)
    for iter in range(N_iters):           # N ~ 800
        if iter % K == 0:                 # K=10
            cfg = sample_dr_config(rng)
            env = TwoTeamVecEnv(**cfg.to_env_kwargs())
            trainer.attach_env(env)
            trainer.frozen_opponent = make_opponent(cfg.opponent_kind)
        
        # PPO iter(复用 br_trainer 完全不变)
        buf = trainer.collect_rollout(env, horizon, learning_team=0)
        trainer._compute_gae(buf)
        trainer.update(buf)
        
        # Health monitor(同 WP2:NaN, adv_std ∈ [0.1, 100], entropy 稳定)
        # Quick eval @ nominal vs rule every 50 iter(防 nominal 退化)
        if iter % 50 == 0:
            wr_nominal = quick_eval_at_nominal(commander, n_ep=20, horizon=200)
            log wr_nominal,assert ≥ -0.05 vs rule-equivalent 起点
        
        if iter % 100 == 0:
            assert_priv_normalized(env, tag=f"iter-{it}")
        
        if iter % snapshot_every == 0:    # 50
            save iter{N:03d}.pt
```

### 5.4 CLI flags

```bash
python -u algo/_shared/pilot/twoteam/run_bet_b_dr.py \
    --n-iters 800 \
    --snapshot-every 50 \
    --dr-reconfig-every 10 \
    --bc-samples 50000 --bc-epochs 15 \
    --ppo-lr-actor 1e-4 --ppo-entropy-coef 0.01 --log-std-floor -6.0 \
    --dr-jam-gain-min 3.0 --dr-jam-gain-max 9.0 \
    --dr-range-sigma-min 0.02 --dr-range-sigma-max 0.10 \
    --dr-sigma-q-min 1.0 --dr-sigma-q-max 4.0 \
    --dr-exposure-gain-min 100 --dr-exposure-gain-max 400 \
    --dr-radar-sep-min 1000 --dr-radar-sep-max 2000 \
    --dr-map-size-min 6000 --dr-map-size-max 10000 \
    --dr-geom "mirror,random" \
    --dr-opponents "rule,pure_track,pure_jam,balanced" \
    --ckpt-dir checkpoints/twoteam/bet_b_dr/ \
    --out experiments/twoteam/bet_b_dr_report.md
```

### 5.5 Health guards(继承 WP2)

- α_eff bug: priv[:,4] 归一化 assert 每 100 iter
- NaN guard: per-iter
- adv_std ∈ [0.1, 100]
- entropy 不崩(稳定在 -2.x)
- ckpt_dir 严禁 /tmp(`raise ValueError`)
- **nominal WR 不退化监控**: 每 50 iter vs rule @ nominal,WR ≥ -0.05 起点值

### 5.6 预算

~1 周(800 iter × horizon=200 × 8 envs,RTX PRO 6000)

---

## 6. Step 2 — Nominal-RL baseline(G2)

**Path**: 复用 `run_bet_b_dr.py`,DR 分布塌缩到 nominal:

```bash
python -u algo/_shared/pilot/twoteam/run_bet_b_dr.py \
    --n-iters 800 \
    --dr-jam-gain-min 6.0 --dr-jam-gain-max 6.0 \
    --dr-range-sigma-min 0.05 --dr-range-sigma-max 0.05 \
    --dr-sigma-q-min 2.0 --dr-sigma-q-max 2.0 \
    --dr-exposure-gain-min 200 --dr-exposure-gain-max 200 \
    --dr-radar-sep-min 1500 --dr-radar-sep-max 1500 \
    --dr-map-size-min 8000 --dr-map-size-max 8000 \
    --dr-geom mirror \
    --dr-opponents rule \
    --ckpt-dir checkpoints/twoteam/bet_b_nom_rl/ \
    --out experiments/twoteam/bet_b_nom_rl_report.md
```

**Why**: G2 判据需要。证明 DR 加值不是 RL 本身就行。同 budget(800 iter),只缺随机化。

预算: ~1 周(可与 Step 1 并行跑,双 GPU 卡 or 串行 ~2 周)

---

## 7. Step 3 — Off-nominal eval grid

### 7.1 路径

**New** `scripts/eval_bet_b_robust.py`(~300 LOC)

### 7.2 Eval grid(比 sweep 更密)

6 轴 × 4 levels(更密):

| 轴 | levels |
|---|---|
| `jam_gain` | 3.0, 4.5, 6.0, 9.0 |
| `range_sigma_m` | 0.02, 0.05, 0.075, 0.10 |
| `sigma_q` | 1.0, 2.0, 3.0, 4.0 |
| `exposure_gain` | 100, 200, 300, 400 |
| `radar_separation_m` | 1000, 1500, 1750, 2000 |
| `geometry` | MIRROR, RANDOM |

→ 单轴 sweep: 6 轴 × ~4 levels = ~24 grid points(不含 nominal 复用)

**对手固定为 StrongRule**(保 eval 公平,只测我方策略的 robustness)。

### 7.3 三方对比

| Player | 描述 |
|---|---|
| **DR-RL** | Bet B Step 1 训出的 commander |
| **Nom-RL** | Bet B Step 2 训出的 commander |
| **Rule** | StrongRuleCommander |

### 7.4 每 grid 每 player

- 双向 30 ep × horizon=200 × n_envs=8 = 60 ep/dir,120 ep total per grid per player
- bootstrap 1e4 CI

### 7.5 总量

24 grid × 120 ep × 3 players = 8640 ep ≈ 2-3 天

### 7.6 输出

`experiments/twoteam/bet_b_robust_eval.md`:
- 3 个 (player × grid) WR matrix
- **median WR across grid**(DR-RL vs Rule vs Nom-RL)
- **5%-tail WR**(最差 5% grid 的均值 — off-nominal 鲁棒核心指标)
- **cliff count**(WR < nominal - 0.20 的 grid 数)
- per-axis marginal WR 切片(jam_gain=3/4.5/6/9 时三方 WR)

---

## 8. Step 4 — Smoke test

**New** `tests/twoteam/test_bet_b_smoke.py`(~120 LOC)

```python
def test_dr_config_sampler():
    """DR config 采样覆盖所有 axis 边界 + nominal,边界正确."""

def test_dr_env_reconfig():
    """每 K iter 重建 env,新 config 生效(jam_gain 等)不 crash."""

def test_dr_loop_minimal():
    """DR 训练 loop 跑 3 iter(K=1),DR config 实际生效(不卡 nominal)."""
```

预算: <2 min

---

## 9. Verification(端到端)

### V0: Smoke
```bash
conda activate fluxphased
python -u tests/twoteam/test_bet_b_smoke.py
```
预算: <2 min

### V1: Rule 敏感度扫描(cheap gate)
```bash
python -u scripts/rule_sensitivity_sweep.py \
    --episodes 100 --n-envs 8 --horizon 200 \
    --out experiments/twoteam/rule_sensitivity_sweep.md
```
预算: ~2-3 小时
**硬停**: 0-1 cliff → 转 IET

### V2: DR 训练(主成本)
```bash
python -u algo/_shared/pilot/twoteam/run_bet_b_dr.py \
    --n-iters 800 --snapshot-every 50 --dr-reconfig-every 10 \
    --ckpt-dir checkpoints/twoteam/bet_b_dr/ \
    --out experiments/twoteam/bet_b_dr_report.md \
    > experiments/twoteam/bet_b_dr.log 2>&1 &
```
预算: ~1 周
**健康指标**:
- adv_std ∈ [0.1, 100] 全程,无 NaN
- entropy 稳定(-2.x)
- nominal WR 监控 ≥ 起点值 - 0.05
- training reward 从 BC 起点 -0.68 在 DR 分布上稳定 / 略降(因 off-nominal 更难)

### V3: Nom-RL baseline
```bash
# 同 V2 但 DR 塌缩到 nominal,见 Step 2
```
预算: ~1 周(可与 V2 并行)

### V4: Off-nominal eval
```bash
python -u scripts/eval_bet_b_robust.py \
    --dr-ckpt checkpoints/twoteam/bet_b_dr/iter_final.pt \
    --nom-ckpt checkpoints/twoteam/bet_b_nom_rl/iter_final.pt \
    --episodes 30 --n-envs 8 --horizon 200 \
    --out experiments/twoteam/bet_b_robust_eval.md
```
预算: ~2-3 天

### V5: 判门贴回用户

| 判据 | 阈值 | 决定 |
|---|---|---|
| **G1-robust** DR-RL median WR > Rule median WR | CI 排除 0 | 主 PASS |
| **G1-clean** DR-RL @ nominal vs Rule @ nominal | WR ≥ -0.05 | 干净 |
| **G2** DR-RL 5%-tail > Nom-RL 5%-tail | CI 排除 0 | DR 加值确认 |

---

## 10. 决策树

```
Step 0 PASS(rule 敏感度扫描 cliff ≥ 2)?
  NO  → 硬停,转 IET 地板(rule 不脆,Bet B 前提死)
  YES → Step 1-2 训练

Step 1-2 训完:
  Step 3 eval grid

G1-robust PASS + G1-clean PASS + G2 PASS
  → Bet B CLEAN PASS → 写 AppInt 主注(off-nominal 鲁棒超专家)

G1-robust PASS + G1-clean FAIL
  → 部分(DR-RL off-nominal 赢但 nominal 输 rule)
  → 写 AppInt 但弱故事("trade-off:泛化换 nominal")

G1-robust FAIL
  → Bet B FAIL → IET 地板(3 线近 Nash + 第 4 线 Bet B 也匹配)

G2 FAIL
  → DR 没加值(RL 本身就够)
  → 写 AppInt 但弱("RL 行,DR 加值不明显")
```

### 硬停条款
- **不开第 4 轮 DR 调参**: 若 800 iter 训完 G1-robust 仍 FAIL,转 IET
- **若 nominal WR 退化了**: 加 nominal-rehearsal(每 batch 混 20% nominal 样本),**最多调一次**
- **若 DR-RL 训不稳**(entropy 崩 / adv_std 爆): diagnose 不盲训
- **若 env reconfig bug**: smoke test 应捕获,fix 后续跑

---

## 11. 风险登记

| # | 风险 | P | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | rule 全程稳健(0-1 cliff) | M | H(Bet B 前提死) | Step 0 先 gate,~3 小时 |
| R2 | env reconfig 在 iter 边界有副作用 | M | M | smoke test + 每 K=10 iter 重建,日志监控 |
| R3 | DR-RL 在 nominal 退化 | M | M | nominal-rehearsal + 每 50 iter nominal WR 监控 |
| R4 | DR-RL 训不稳(entropy 崩) | M | M | log_std_floor=-6 + entropy_coef=0.01 + DR 边界收窄 |
| R5 | DR 分布太广,策略平庸 | M | M | 边界收窄,Step 0 cliff 轴专攻 |
| R6 | G2 FAIL(DR 没加值) | L | M | Nom-RL 同 budget,直接对比 |
| R7 | α_eff bug 复发 | L | H | priv[:,4] assert 每 100 iter |
| R8 | /tmp 清掉 ckpt | L | H | ckpt_dir 硬检,严禁 /tmp |
| R9 | nominal vs off-nominal WR 分离不够(eval 区分不出) | M | M | Step 3 grid 加密,加 5%-tail 指标 |

---

## 12. retreat 友好

```bash
# Step 0 gate FAIL:只需保留 sweep 结果(强化 IET 近 Nash 故事)
# Step 1-4 全部可删
rm -f algo/_shared/pilot/twoteam/run_bet_b_dr.py
rm -f scripts/eval_bet_b_robust.py scripts/rule_sensitivity_sweep.py
rm -f tests/twoteam/test_bet_b_smoke.py
rm -rf checkpoints/twoteam/bet_b_*/
rm -f experiments/twoteam/bet_b_*.md experiments/twoteam/bet_b_*.log
# rule_sensitivity_sweep.md 保留(IET 证据)
```

---

## 13. 后续(若 Bet B PASS → AppInt)

### 故事框架

"域随机化 RL 在两队 IQ 多功能对抗博弈中,off-nominal 鲁棒性超专家设计点"

### 主图

per-axis WR 切片(6 轴 × 4 levels × 3 player):
- x 轴:axis level
- y 轴:WR vs Rule
- 三条线:DR-RL / Nom-RL / Rule(自对)

### 关键数字

- median WR 差(DR-RL − Rule)
- 5%-tail WR 差(DR-RL − Rule)
- cliff count 差(DR-RL − Rule)
- DR 加值: 5%-tail WR 差(DR-RL − Nom-RL)

### 章节结构(AppInt 8-12 页)

1. Intro: 两队多功能对抗 + 近 Nash rule + off-nominal 杠杆
2. Related: domain randomization, multi-function phased array, RL vs classical
3. Env: WP0 testbed + 6 轴 DR 表
4. Method: BC 起点 + DR-PPO + CTDE
5. Results: G1-robust + G1-clean + G2
6. Ablations: 去掉每个 DR 轴的 marginal 贡献
7. Discussion: 近 Nash at nominal vs robust-global
8. Conclusion

---

## 14. 后续(若 Bet B FAIL → IET)

地板已是硬证据(3 线近 Nash),Bet B FAIL 加第 4 线:
"RL 在 nominal + off-nominal 都匹配 rule"

IET 故事:
- testbed(WP0)+ BC→PPO pipeline(G0)+ 三线近 Nash 证据 + Bet B 第 4 线
- 论文标题候选:"Two-Team Symmetric Multifunction Phased-Array Adversarial Game: Testbed + Near-Nash Verification of Hand-Tuned Rule"

---

## 15. 关键设计决策(总结)

1. **DR 在 iter 边界重建 env(K=10),不动 env API**: 30 行 wrapper,env 代码零修改
2. **Step 0 先 gate**: 廉价(~3 小时)筛掉 "rule 全程稳健" 这种 Bet B-死情况,不浪费 1-2 周 DR 训练
3. **6 轴 DR 基于 env 真实可配参数**: 不靠虚构轴,所有轴已在 env constructor 验证
4. **nominal-rehearsal 防退化**: DR 训练的通病是 nominal 退化,rehearsal + 监控
5. **5%-tail WR 是主指标**: median WR 易被大部分 grid 主导,5%-tail 才看 off-nominal 鲁棒真本事
6. **G2(Nom-RL baseline)必跑**: 不证 DR 加值,reviewer 会问 "RL 本身就够?"
7. **硬停不开第 4 轮**: 跟 WP2 相同纪律。若 800 iter 训完仍 FAIL,Bet B 也死,转 IET。

---

## 状态

- [ ] 等用户批准 spec
- [ ] Step 0: rule 敏感度扫描(~3 小时,cheap gate)
- [ ] Step 0 PASS → Step 1-2 训练(1-2 周)
- [ ] Step 3 eval(2-3 天)
- [ ] V5 判门 + 贴回用户
