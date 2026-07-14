# WP1 G0 Exploitability Gate — 完整实验报告

**日期**: 2026-07-14
**项目**: 两队对称多功能相控阵对抗(TWOTEAM_MULTIFUNCTION_PLAN.md)
**目标会议**: TAES(主)/ IET(备选)
**spec**: TWOTEAM_ENV_FIX_SPEC.md + TWOTEAM_MULTIFUNCTION_PLAN.md §WP1

---

## 1. 研究背景

**G0 命门**(用户 verbatim): "G0 是最先的命门:π_rule 可不可被 exploit?可 → 我放行 WP2;不可 → 别烧自博弈,一起退 IET"

公式:
```
exploitability(π_rule) = U(π_rule vs 镜像 π_rule) − U(π_rule vs BR(π_rule))
                      ≈ 0 − (negative) = 正值 = rule 可被 exploit 的程度
```

**判据**:
- gap ≥ 0.5 kills/episode **且** bootstrap 95% CI 排除 0 → **PASS,放行 WP2 self-play**
- gap ≈ 0 或 CI 含 0 → **FAIL,退 IET(IQ/CRLB 基线论文)**

**用户重点提醒**(2026-07-13): G0 FAIL 的原因是 "两队对称 + 多功能博弈,但 RL 榨不出" → "根因 A 平静海面" → 两次框架级失败的早期警告。

---

## 2. 上一次 G0 FAIL 的诊断与误判

**时序**:
1. 2026-07-13 WP0 PASS(mirror 对称 + 无主导策略 + CRLB ratio 0.28)
2. 2026-07-13 第一次 G0 FAIL:exploit_gap ≈ 0,BR 训不出 → 当时诊断为 "root-A 平静海面"
3. 2026-07-13 用户重诊断:**这不是 root-A**

**用户重诊断核心**(verbatim):
> "G0 FAIL 是假 FAIL:不是『多功能博弈平凡』(root-A),是『干扰成了对称的全灭按钮 → 0-0 僵局 → exploitability 无从定义』。缺的是抗干扰技能维度。"

**Cell-by-cell 证据**:
- Cell 1(mirror):严格 0-0(结构性的,不是噪声)
- Anti-strawman:rule 被 pure_jam 拖入僵局 → "TOO_WEAK" 但不是因为 rule 弱,是因为 env 缺抗干扰技能
- Cell 2:BR 没有 "干扰你同时保住自己 track" 的技能维度

**根因**:env 没有 frequency agility / burn-through skill → 干扰 = 全灭按钮 → 任何 matchup 都倾向 0-0。

---

## 3. TWOTEAM_ENV_FIX_SPEC.md(5 个 Fix)

### Fix 1 — 抗干扰技能维度(核心)**

**物理基础**(Richards ch.5, Skolnik ch.9):
- 频率捷变迫使干扰机把功率铺到 N 个频率 → 每频率 JNR 下降约 N 倍
- 但处理开销:per-frequency 相干积分脉冲数减少 → SNR 惩罚

**实现**:
```python
# 每孔径一个 freq_hop_rate ∈ [1, freq_hop_max=8]
# 每 tracker 聚合:max(贡献孔径的 hop)
freq_hop_per_tracker[:, t, r] = max over apertures k beaming at r of freq_hop_rate[:, t, k]

# 有效干扰 = jam / hop(干扰被 hop 稀释)
effective_jam = jam_level / freq_hop_per_tracker
jam_mul = 1.0 + jam_gain * effective_jam

# 处理开销 = 1/hop^0.25(亚线性,匹配滤波器组在小 hop 下恢复大部分 SNR)
processing_overhead = 1.0 / freq_hop_per_tracker.pow(0.25)
f_track_eff = f_track * processing_overhead
track_sigma = base_sigma / sqrt(f_track_eff) * jam_mul
```

**关键调参**:
- `jam_gain`: 8.0 → **6.0**(让 hop=8 时抗干扰能赢;手算:hop=8 + pure_track vs pure_jam → track_sigma=0.13 → trace_P=0.034 < tau_track=0.04 ✓)
- `processing_overhead`: 1/sqrt(hop) → **1/hop^0.25**(亚线性;sqrt 太苛刻,小 hop 数下不应该这么惩罚)

**修改文件**:
- `env/gpu/twoteam/twoteam_env.py`:加 `freq_hop_max` 参数,`_last_freq_hop` tensor,改 step() 加 jam 聚合 + track 公式
- `algo/_shared/pilot/twoteam/extreme_commanders.py`:ExtremeCommander 加 `freq_hop` 参数,新策略 `track_agile`(pure_track + hop=8)
- `algo/_shared/pilot/twoteam/commander_actor_critic.py`:加 `freq_hop_head = Linear(hidden, n_aperture * 2)`,Beta(α,β) 采样,rescale 到 [1, freq_hop_max]
- `algo/_shared/pilot/twoteam/br_trainer.py`:`_RolloutBuffer` 加 `freq_hop_rate` 字段,collect + update 处理

### Fix 2 — Exposure 生效(打破 mirror duck 锁)

**问题**:原 `exposure_gain=50`,两队同时 duck → 0-0 永远僵局
**修复**:
- `exposure_gain`: 50 → **200**(4× 更敏感)
- 新增 `exposure_overload_threshold=50`,`exposure_decay_rate=0.5`
- 当 exposure > 50:**直接膨胀 own tracker_P diag**(非对称,打破 mirror 锁不依赖 RNG)
- 物理意义:辐射过曝 → 敌方反向定位 → 自我追踪受 EM 干扰

### Fix 3 — StrongRule 升级(anti-strawman + 抗干扰反应)

`algo/_shared/baselines/twoteam_strong_rule_commander.py`:
- `exposure_duck_threshold`: 30 → **60**(减少误 duck)
- 加 `jam_detect_threshold=0.30`(敌方干扰 > 0.30 → 触发抗干扰反应)
- 加 `freq_hop_low=1.0`,`freq_hop_high=6.0`(默认低 hop,被强干扰时高 hop)
- 反应逻辑:`high_jam → task_alloc[track] += 0.15, task_alloc[jam] += 0.10, freq_hop = 6.0`

### Fix 4 — WP0-decisive 判据升级

`algo/_shared/pilot/twoteam/run_wp0_check.py`:
- **2a**(原):无主导策略(任一策略 vs 所有其他 > 0.90 胜率)
- **2b**(新):decisive_rate(出杀局比例)≥ 0.50 — 抓 "全部 0-0 僵局" 的伪 PASS
- **2c**(新):kill_density ≥ 0.5 kills/ep — 抓 "杀很少但 decisive" 退化
- **2d**(新):每个策略 best_opponent_decisive ≥ 0.30 — 抓 "任一策略没人能破" 的 unilateral lock

### Fix 5 — G0 重测(BR 500 iters + decisive gate)

`algo/_shared/pilot/twoteam/run_g0_gate.py`:
- BR 500 iters(原 200)
- 加 `lr_decay_iters` cosine LR 衰减(BR trainer 加相应逻辑)
- 报告加 anti-strawman + BR training curve + bootstrap CI

---

## 4. 实验结果

### 4.1 WP0-decisive(PASS)

```
=== Check 1: Mirror self-play symmetry ===
  max |kills_t0 - kills_t1|  = 0.0000 (target: 0)
  max |trace_P_t0 - t1|      = 0.0000 (target: 0)
  mean |reward_t0|           = 0.0000 (target: ~0)
  ✅ PASS

=== Check 2: Four-function tradeoff matrix (7 strategies) ===
  track_agile vs pure_jam: WR_t0=1.00  kills 1.00 vs 0.00  decisive=1.00  ← 抗干扰生效!
  pure_jam vs track_agile: WR_t0=0.00  kills 0.00 vs 1.00  decisive=1.00

  Dominant strategy: NONE
  Decisive rate (off-diag mean): 0.660  (target ≥ 0.50)  ✅
  Kill density (off-diag mean):  0.801  (target ≥ 0.5)   ✅
  ✅ 2a/2b/2c/2d all PASS

=== Check 3: CRLB anchor ===
  Achieved trace_P / CRLB = 0.21  ✅
```

**关键证据**:track_agile vs pure_jam = 1.00 vs 0.00 → **抗干扰技能维度真的有效**。

### 4.2 WP1 smoke tests(PASS)

```
✅ StrongRule 30 steps NaN-free; mean tracked targets per team: 1.00
✅ AC action shapes OK; task_alloc sums to 1; freq_hop ∈ [3.44, 7.17]; env step NaN-free
✅ evaluate_actions log_prob matches forward (diff 1.19e-07)
✅ BR trainer 5 iters OK; final adv_std=1.000, entropy=-0.036, kl=0.0107
```

### 4.3 G0 re-test #1(默认超参,FAIL)

**配置**: BR 500 iters, lr_actor=3e-4, lr_critic=1e-3, entropy_coef=0.01, horizon=200, n_envs=8, n_episodes=30

**Anti-strawman**: TOO_WEAK(rule vs pure_jam 33% 胜、68% 平)

**Cell 1 mirror**: margin = 0.000 ✓

**BR 训练曲线**(reward_mean):
| iter | reward | 备注 |
|---|---|---|
| 0-275 | -1.82(平台) | 无学习信号 |
| 300 | -1.55 | 学习开始 |
| 325 | -1.19 | 大跳 |
| 450-499 | -1.15 → -1.21 | 缓慢改进 |

**Cell 2**:
- rule kills: 1.96
- BR kills: 1.00
- BR 胜率: 0%
- exploit_gap: **-0.963**(CI [-0.98, -0.94])

**BR 策略检查**:
- task_alloc 均匀 [0.25, 0.25, 0.25, 0.25](Dirichlet α 没学到浓度)
- 只有 freq_hop 学到了(4-7)

**判据**:
| 检查 | 阈值 | 实际 | 通过 |
|---|---|---|---|
| exploit_gap ≥ 0.5 | 0.5 | -0.963 | ❌ |
| CI 排除 0 | > 0 | -0.98 | ❌ |
| BR 胜率 ≥ 0.55 | 0.55 | 0.00 | ❌ |
| BR 健康 | adv_std∈[0.1,100], no NaN | 1.0 | ✅ |

### 4.4 G0 re-test #2(调超参,FAIL)

**配置**: BR 500 iters, **lr_actor=1e-4**, lr_critic=1e-3, **entropy_coef=0.03**, **lr_decay_iters=500**(cosine)

**Anti-strawman**: TOO_WEAK(同前)

**Cell 1 mirror**: margin = 0.000 ✓

**BR 训练曲线**(reward_mean):
| iter | reward | 对比 re-test #1 |
|---|---|---|
| 0-100 | -1.82(平台) | 同 |
| 110 | -1.69 | 早 200 iters 出现学习信号 |
| 130 | -1.25 | 更快收敛 |
| 260 | -0.84 | 已达 re-test #1 最终水平 |
| 310 | **-0.57** | 最佳点 |
| 499 | -0.88 | 是 re-test #1 最终(-1.21)的 73% |

**Cell 2**:
- rule kills: 1.96(同前)
- BR kills: 0.98(同前)
- BR 胜率: 0%
- exploit_gap: **-0.979**(CI [-1.00, -0.96])

**BR 策略检查**(显著改进):
- task_alloc = [0.20, **0.36**, 0.30, 0.14](BR 把 36% 给 track,而非均匀)
- Dirichlet α: aperture 0 = [1.24, **2.27**, 1.88, 0.84](track 维 α 最大)
- beam + laser 都对准 enemy 1(聚焦射杀)
- freq_hop = 4.7-5.2(主动用抗干扰)
- emit aperture 0 偶尔关(step 1)

**StrongRule 对比策略**:
- task_alloc = [0.08, **0.71**, 0.12, 0.10](track 71%,高度集中)
- beam + laser = enemy 0(聚焦)
- hop = 1.0(对手是随机,无需 hop)

---

## 5. 关键分析

### 5.1 与上一次 FAIL 的对比

| 维度 | 上次 FAIL(pre-fix) | 本次 FAIL(post-fix) |
|---|---|---|
| WP0-decisive | pure_jam 不可破(2d FAIL) | 4 项全 PASS |
| 抗干扰技能 | 不存在 | track_agile 打 pure_jam 1.00 vs 0.00 |
| 训练 reward 趋势 | 平坦 -1.8 | 学习曲线 -1.82 → -0.88 |
| BR deterministic policy | near-prior [0.25,...] | 学到 track-heavy [0.20, 0.36, 0.30, 0.14] |
| Cell 2 BR kills | 0(僵局) | 1.00(能玩,但输) |
| Fail 性质 | env 退化(root-A mimic) | BR 没找到 exploit |

### 5.2 为什么 BR 没找到 exploit

**直接原因**:BR 学到的策略 "track-heavy balanced" 跟 StrongRule 同类型,但浓度低(rule 71% vs BR 36%)→ 追踪竞赛中 rule 先杀。

**根本原因**:BR 的探索没找到 "rule 不防御的弱点"。可能的 exploit:
1. StrongRule duck threshold=60,可通过 pump exposure 强制 duck → 杀
2. StrongRule anti-jam 反应阈值=0.30,可分散 jam 避免触发
3. StrongRule 聚焦射杀,可通过假目标切换欺骗

但 PPO 在 13 维连续 + 5 维离散动作空间中,500 iters(800K samples)不足以发现这些非显然 exploit。

### 5.3 训练 reward vs eval reward 的差距

- 训练(stochastic)reward: -0.88
- eval(deterministic)reward: ≈ -1.96 + 0.98 = -0.98 margin ≈ -9.8(unscaled)

 stochastic 训练中,BR 偶尔通过随机采到 "高 track + 高 hop" 的好运 sample 拿到奖励。但 Dirichlet 均值没足够集中 → deterministic 策略还是中等水平。

---

## 6. 决策点

### 6.1 严格的 spec 解读

用户 verbatim: "G0 FAIL → 别烧自博弈,一起退 IET"

按这条:G0 二次 FAIL → 退 IET。

### 6.2 但环境不再退化

WP0-decisive 4 项 PASS + track_agile 打 pure_jam 1.00 vs 0.00 → **环境是非平凡的**。BR 学到了连贯策略。这次 FAIL 不是 root-A,是 BR 训练 / 探索不够。

### 6.3 备选路径

**A. 退 IET**(strict spec)
- 论文方向:IQ/CRLB 两队对称多功能基线
- 把 env 设计 + WP0-decisive + 抗干扰技能维度作为 "framework design" 章节
- 放弃 WP2 self-play 主线

**B. 再调 BR 超参**(继续尝试)
- 去 LR decay(LR 持续高让策略继续移动)
- entropy_coef 0.03 → 0.01 后期(允许 Dirichlet 集中)
- 初始化 task_alloc_head bias 让 track 起点高
- 风险:可能还是不行,但更便宜(~30 min)

**C. 软化 StrongRule**(让 G0 能 PASS)
- track 浓度 cap 0.45(原 0.71)
- duck threshold 60 → 30(更易触发)
- anti-jam 阈值 0.30 → 0.50(反应更慢)
- 风险:rule 太弱 → anti-strawman 现在就 TOO_WEAK,会更弱 → G0 PASS 但 WP2 thesis 弱

**D. 课程学习**(更结构化)
- 先 vs ExtremeCommanders(track_agile 等)300 iters → 学到抗干扰技能
- 再 vs StrongRule fine-tune 300 iters → 把技能迁移到打 rule
- 风险:需要写 curriculum 代码,~2-3h

---

## 7. 已花费成本

| 项 | 时间 |
|---|---|
| Fix 1-3 实现 + WP0-decisive 实现 | ~3h |
| WP0-decisive 调试(processing_overhead, jam_gain 调参) | ~1h |
| WP1 smoke tests | 5 min |
| G0 re-test #1 | ~22 min |
| G0 re-test #2 | ~22 min |
| **总** | **~4.5h** |

---

## 8. 推荐与待决问题

**我的判断**:这次 FAIL 跟上次性质完全不同。环境**确实**修好了(抗干扰技能有效,WP0 PASS,BR 学到连贯策略)。问题是 BR 500 iters 不够 PPO 在 18 维动作空间找到 exploit。

**对用户的最关键的待决问题**:

> "退 IET 接受这次 FAIL" vs "再投 1-2 轮 BR 训练调参看能不能 PASS"?

如果用户更看重 **研究纪律**(spec 怎么写就怎么做),退 IET。
如果用户更看重 **不让修复工作白费**,再投一两轮。

我作为 AI 没有偏好。数据在这里,决策权在用户。

---

## 附录:文件清单

**修改**:
- `env/gpu/twoteam/twoteam_env.py` — Fix 1 + Fix 2
- `algo/_shared/pilot/twoteam/extreme_commanders.py` — freq_hop 参数 + track_agile 策略
- `algo/_shared/pilot/twoteam/commander_actor_critic.py` — freq_hop_head + Beta 分布
- `algo/_shared/pilot/twoteam/br_trainer.py` — freq_hop_rate buffer + LR decay
- `algo/_shared/pilot/twoteam/run_wp0_check.py` — WP0-decisive 4 子检查
- `algo/_shared/pilot/twoteam/run_g0_gate.py` — lr_decay_iters CLI + 报告改进
- `algo/_shared/baselines/twoteam_strong_rule_commander.py` — Fix 3
- `tests/twoteam/test_wp1_smoke.py` — freq_hop shape + range 检查

**新增**:
- `TWOTEAM_ENV_FIX_SPEC.md` — 修复 spec
- `experiments/twoteam/G0_RETEST_STATUS.md` — 中间状态
- `experiments/twoteam/G0_FULL_EXPERIMENT_REPORT.md` — 本报告
- `experiments/twoteam/wp0_check_report.md` — WP0 报告
- `experiments/twoteam/g0_gate_report.md` — G0 最终报告
- `experiments/twoteam/g0_br_training_log.csv` — BR 训练曲线
- `checkpoints/twoteam/br_vs_strong_rule_final.pt` — BR v2 checkpoint
