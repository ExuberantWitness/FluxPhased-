# TWOTEAM ENV FIX SPEC — 抗干扰技能维度 + Exposure 必须生效

**Date**: 2026-07-14
**Target**: PRO6000 (RTX PRO 6000 Blackwell, 101.9 GB)
**Status**: 修复 WP1 G0 假 FAIL 的精确 spec。不是 root-A,是 env 缺一个技能维度。

---

## 0. TL;DR(执行摘要)

**误诊纠正**: G0 FAIL ≠ root-A(传感太容易 → classical 近最优)。G0 FAIL = env 缺抗干扰技能维度 → 干扰是对称的"全灭按钮" → 镜像 0-0 僵局 → exploit_gap 在 0-0 baseline 上退化无意义。

**5 项修复**(有界,~6h 实施 + 1h 重测):
1. **【命门】加抗干扰技能维度**: 频率捷变 `freq_hop_rate` 作为新动作维度,real-radar 物理依据
2. **让 exposure 真咬**: 提高 `exposure_gain` + 加 exposure 直接损伤项,打破镜像 duck 死锁
3. **强化 StrongRule**: 给它 `freq_hop` 决策 + 抗干扰反应,从 TOO_WEAK 升到合格 baseline
4. **WP0-decisive 判据升级**: 在"无恒赢策略"基础上加"游戏分胜负"硬性检查
5. **G0 重测**: 同 exploit_gap 公式但 baseline 不再退化

**硬退场**: 修完跑 WP0-decisive + G0,若仍 draw 满天 / BR 仍打不过真强规则 → 才诚实地退 IET。

---

## 1. 数据驱动的再诊断(全部已验证)

### 1.1 WP0 Check 2 矩阵是 false PASS

原矩阵(`experiments/twoteam/wp0_check_report.md`):

| | pure_track | pure_jam | pure_comm | pure_detect | balanced | balanced_jam_heavy |
|---|---|---|---|---|---|---|
| pure_track | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| pure_jam | 0.00 | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| pure_comm | 0.00 | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| pure_detect | 0.00 | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| balanced | 0.97 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| balanced_jam_heavy | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |

- **22/36 = 0.00 (61% draws)**
- `pure_jam / pure_comm / pure_detect` 三行**整行 0.00**(碰到任何对手都 0 杀)
- 原 WP0 判据: "无单一策略 >0.90 vs ALL" → PASS(确实没恒赢)
- **缺失判据**: 游戏是否 *decisive*(能杀人、能分胜负)

### 1.2 env 零抗干扰机制

当前 track 公式 ([env/gpu/twoteam/twoteam_env.py:296](env/gpu/twoteam/twoteam_env.py#L296)):
```python
track_sigma = base_sigma / sqrt(f_track) * jam_mul
jam_mul = 1.0 + self.jam_gain * jam_level   # jam_gain=8.0, 单调乘子
```
- jam 是一个**纯放大因子**,没有任何反向项(f_track 提高、功率提高、频率捷变都**压不回去**)
- 技高分配 vs 技低分配在 jam 下**同样瞎**
- grep `burn.through|freq.agility|agile|frequency` 在整个 env = 0 命中

### 1.3 home-on-jam 镜像下退化

`p_homejam = 1 - exp(-50 * (exposure/100) * 0.001)` ([env/gpu/twoteam/twoteam_env.py:343](env/gpu/twoteam/twoteam_env.py#L343)):
- exposure=4.49(typical)→ p_homejam ≈ 0.0022/step
- `homejam_roll = torch.rand(E,1,R).expand(E,T,R)` ([L348](env/gpu/twoteam/twoteam_env.py#L348)) — 镜像广播
- 对称动作 → 对称 exposure → 同一个 roll → **双方同生共死** → 永远 0 margin

### 1.4 duck 在镜像下死锁

`duck_mask = (exposure>30) & (E_max<0.5)` 在镜像下双方同步触发 → 同步关辐射 → exposure 同步降 → 同步解除 → 循环。这是 rule-vs-rule = 0 kills / 240 的直接元凶。

### 1.5 exploit_gap 在 0-0 baseline 上无意义

mirror_margin = 0(强制);br_margin = ±ε(偶发噪声);gap = ±ε。CI [-0.133, -0.058] 是 240 episodes 的偶然涨落,不是 exploitability 信号。

---

## 2. FIX 1: 抗干扰技能维度(命门,real-radar 依据)

### 2.1 物理依据

Real radar 抗干扰三大手段(参见 Richards *Fundamentals of Radar Signal Processing* ch.5, Skolnik *Radar Handbook* ch.9):

1. **Frequency agility** (本 spec 采用): 跳频跨 N 个频点 → jammer 必须把功率铺到整个带宽 → 每频点 JNR 降 ~N 倍 → 跟踪保住。代价:信号处理开销 ↑(同步 / 多脉冲相干积累)。
2. Coherent dwell stretching: 长相干积累 → SNR ∝ √N_pulses 提升。代价:dwell 时间 ↑ → 杀伤链节奏 ↓。
3. Multistatic geometry: 收发分置 → jammer 盲不掉旁瓣。代价:几何约束 + 部署复杂。

选 (1) 频率捷变:实现简单、物理清晰、和现有 `task_alloc` 软分配天然耦合(分配更多 f_track → 处理开销 → 支撑更多跳频)。

### 2.2 动作空间扩展

每个 aperture 多一个连续动作:

```python
# algo/_shared/pilot/twoteam/commander_actor_critic.py
# 现有: task_alloc[E,R,4], beam_target[E,R], laser_target[E], emission_on[E,R]
# 新增: freq_hop_rate[E,R] ∈ [1, N_FREQ_MAX]   (连续,正实数)
```

实现选择: **Beta(α,β) * (N_FREQ_MAX-1) + 1** per aperture。N_FREQ_MAX=8(2-3 bit 等效)。
- 探索期: α=β=1 → uniform [1, 8]
- 收敛期: α/β 偏向最优值
- 替代方案: Categorical(8) — 简单但失去 spec D2=A "连续分数" 风格;不推荐

新增 head:
```python
self.freq_hop_head = nn.Linear(hidden, n_aperture)   # [B, R],每 aperture 一对 (α,β)
# reshape → [B, R, 2] → softplus + dirichlet_min_concentration → Beta(α,β)
# sample = Beta(α,β).sample() * (N_FREQ_MAX - 1) + 1   ∈ [1, N_FREQ_MAX]
```

log_prob += Beta.log_prob(每 aperture);entropy += Beta.entropy(每 aperture)。

### 2.3 物理公式升级

[env/gpu/twoteam/twoteam_env.py:268-301](env/gpu/twoteam/twoteam_env.py#L268-L301) 改造:

```python
# OLD:
jam_mul = 1.0 + self.jam_gain * jam_level   # 单调放大

# NEW:
# freq_hop 是 RECEIVING side 的属性(track 自己的 tracker 时用)
freq_hop_self = action["freq_hop_rate"]   # [E, T, R],每 aperture
# 按 aperture 聚合到 tracker r 上的有效跳频: 同一 tracker 上各 aperture 的 max
# (一个 tracker 可被多 aperture 喂测量,有效跳频 = 最敏捷的那个 aperture)
freq_hop_per_tracker = ...   # [E, T, R],向量最大值

# 频率捷变削减 jam: jammer 功率铺到 hop_rate 个频点 → 每频点 JNR 降 hop_rate 倍
effective_jam = jam_level / freq_hop_per_tracker.clamp(min=1.0)
jam_mul = 1.0 + self.jam_gain * effective_jam

# 代价:高跳频耗 signal processing overhead → f_track 有效值打折
# (用 f_track 的 sqrt 表示 SNR ∝ √N_pulses,跳频提高后 N_pulses-per-freq 降)
processing_overhead = 1.0 / freq_hop_per_tracker.sqrt().clamp(min=1.0)
f_track_effective = f_track * processing_overhead
track_sigma = base_sigma / (f_track_effective + 1e-3).sqrt() * jam_mul
```

**关键不变量**:
- 无 jam (`jam_level=0`): jam_mul=1,processing_overhead 主导 → 高跳频反而**降低** track(因为 overhead)。技高玩家只在被 jam 时才提高跳频。
- 强 jam + 高跳频: effective_jam ↓,track_sigma ↓,可恢复 track。
- 强 jam + 不跳频: track_sigma ↑↑,完全瞎。

**这就创造了技能差**: 知道何时跳频 vs 何时不跳频是 RL 能学到、classical rule 难穷举的非平凡决策。

### 2.4 obs 扩展

obs_dim 36 → 40(给 freq_hop_self 加 R=2 维,给 enemy_freq_hop 估计加 R=2 维)。详见 §2.6。

### 2.5 验证公式(写完后跑一次 sanity)

| 场景 | 预期 track_sigma |
|---|---|
| f_track=0.5, jam=0, hop=1 | 0.05/√0.5 = 0.071 |
| f_track=0.5, jam=0, hop=4 | 0.05/(√0.5·√0.25) · 1 = 0.142(overhead 主导) |
| f_track=0.5, jam=1, hop=1 | 0.05/√0.5 · 9 = 0.636(被 jam 致盲) |
| f_track=0.5, jam=1, hop=4 | 0.05/(√0.5·√0.25) · (1+8·0.25) = 0.142 · 3 = 0.426(部分恢复) |
| f_track=0.5, jam=1, hop=8 | 0.05/(√0.5·√0.125) · (1+8·0.125) = 0.2 · 2 = 0.4(更好) |

→ 跳频非单调最优,依赖 jam 强度。**这是技能差**。

### 2.6 实施清单

| 文件 | 改动 | LOC |
|---|---|---|
| `env/gpu/twoteam/twoteam_env.py` | action 多 `freq_hop_rate` key;jam 公式改;obs 加 4 维 | ~50 |
| `algo/_shared/pilot/twoteam/commander_actor_critic.py` | 新 `freq_hop_head` (Beta);forward/evaluate_actions/get_action_for_env 加这个 head | ~30 |
| `algo/_shared/pilot/twoteam/extreme_commanders.py` | 6 个 ExtremeCommander 各加 freq_hop_rate 决策 | ~20 |
| `algo/_shared/baselines/twoteam_strong_rule_commander.py` | 见 §4 | ~20 |
| `algo/_shared/pilot/twoteam/br_trainer.py` | _RolloutBuffer 加 freq_hop_rate 字段 | ~10 |
| `tests/twoteam/test_wp1_smoke.py` | 新动作 shape 测试 | ~5 |

---

## 3. FIX 2: Exposure 必须生效(打破镜像 duck 死锁)

### 3.1 双管齐下

**A. exposure_gain 50 → 200**(4× 更敏感)
```python
# env __init__
exposure_gain: float = 200.0,   # was 50.0
```
- exposure=4.49 → p_homejam ≈ 0.0090/step(从 0.0022 提升 4×)
- 200 step episode 中至少一次 home-on-jam 概率: 1 - (1-0.009)^200 ≈ 83%

**B. 加 exposure 直接损伤项**(不依赖 home-on-jam RNG,打破对称)

```python
# env step() 第 5 段,新增:
# 高 exposure 直接降激光防御(E_progress 累积率),逼双方都不能光躲
# 物理依据:暴露 = 敌方能反向 geolocate 你 → 你的反制窗口缩
exposure_overload = (self.exposure > self.exposure_overload_threshold).float()   # [E, T]
# 用在对己不利的方向: 被暴露方 track_quality 衰减
self.tracker_P = self.tracker_P + exposure_overload.unsqueeze(-1) * self.exposure_decay_rate * dt
```

参数: `exposure_overload_threshold=50.0`, `exposure_decay_rate=0.5`。

**C. duck 行为有代价**(规则层面,见 §4)

### 3.2 验证

修完后跑 `pure_jam vs pure_jam`(镜像对称)→ 期望:
- 不再 0-0 双方
- 双方高 exposure → 双方 tracker_P 衰减 → 都不能 track → 但 home-on-jam 非对称(roll 不同)→ 一方先死 → 非 0 margin
- decisive_rate ≥ 0.5(50% episodes 有 ≥1 kill)

---

## 4. FIX 3: 强化 StrongRule(从 TOO_WEAK 升级)

### 4.1 当前问题(anti-strawman 3/4 太弱)

- 打不过 pure_jam(WR=0.15,85% draw)— 被 pure_jam 拖进互干扰僵局
- vs balanced/balanced_jam_heavy 大量 draw — duck 同步死锁

### 4.2 改动

`algo/_shared/baselines/twoteam_strong_rule_commander.py`:

```python
# 1. 加 freq_hop 决策(抗干扰反应)
def get_action(self, env, team):
    ...
    # 检测敌方 jam 强度(env._last_jam_matrix 现已暴露)
    enemy_jam_on_me = env._last_jam_matrix[:, team, :].max(dim=-1).values   # [E]
    high_jam = enemy_jam_on_me > 0.3
    # 高 jam → 高跳频(4-8);无 jam → 低跳频(1)
    freq_hop = torch.where(high_jam.unsqueeze(-1),
                            torch.full((E, R), 6.0, device=dev),
                            torch.full((E, R), 1.0, device=dev))

    # 2. 放松 duck(exposure 阈值 30 → 60),让规则更敢打
    duck_mask = (env.exposure[:, team] > 60.0) & (E_max < 0.5)   # was 30.0

    # 3. 抗干扰自适应 alloc: 被强 jam 时 task_alloc track↑ + jam↑(同时反制)
    # (现已是 base 0.45 track + 0.30 jam,需在强 jam 下额外加)
    high_jam_b = high_jam.unsqueeze(-1).expand(E, R)
    base[..., 1] = base[..., 1] + 0.15 * high_jam_b   # track +0.15
    base[..., 2] = base[..., 2] + 0.10 * high_jam_b   # jam +0.10

    return {
        ..., "freq_hop_rate": freq_hop,
    }
```

### 4.3 验证(新 anti-strawman)

修完后跑 anti-strawman 检查,期望:
- vs pure_track/comm/detect: WR ≥ 0.80(保持)
- **vs pure_jam: WR ≥ 0.60**(从 0.15 升;证明 freq_hop 反制有效)
- vs balanced: WR ∈ [0.40, 0.75](不太弱也不太强)
- vs balanced_jam_heavy: WR ∈ [0.40, 0.75]

---

## 5. FIX 4: WP0-decisive 判据升级

### 5.1 现状

`algo/_shared/pilot/twoteam/run_wp0_check.py` 的 Check 2 只判 "无单一策略碾压" → 61% draws 也 PASS。

### 5.2 新增 Check 2.5 (decisive_rate) + Check 2.6 (kill_density)

```python
# 在 run_wp0_check.py::check_four_function_tradeoff 内,跑完 6×6 矩阵后:

# Check 2.5: 游戏分胜负率
decisive_rate = (matrix_kills_total > 0).mean()   # 36 格中多少格有 ≥1 kill
decisive_pass = decisive_rate >= 0.50   # 至少一半对局有杀

# Check 2.6: 平均杀伤密度
mean_kills_per_ep = matrix_kills_total.mean()   # 每局平均杀几个
density_pass = mean_kills_per_ep >= 0.5

# Check 2(原): 无单一策略碾压
dominant_pass = (matrix_winrate.max(dim=1).values < 0.90).all()   # 旧逻辑保留

check2_pass = dominant_pass and decisive_pass and density_pass
```

### 5.3 报告升级

`experiments/twoteam/wp0_check_report.md` Check 2 段加:
```
- Dominant strategy: NONE / FOUND
- Decisive rate: 0.XX (target ≥ 0.50)
- Mean kills/episode: 0.XX (target ≥ 0.5)
- ✅/❌ PASS/FAIL
```

---

## 6. FIX 5: G0 重测(metric 不变,baseline 不再退化)

### 6.1 G0 流程不变

`algo/_shared/pilot/twoteam/run_g0_gate.py` 主体不动。只是 baseline 因为 env 修复后不再退化,exploit_gap 信号有意义。

### 6.2 新增 decisive gate(防 baseline 退化)

在 G0 报告里加一个 hard check:
```python
# Cell 1 mirror 必须有 ≥50% episodes 出现 ≥1 kill
mirror_decisive_rate = (mirror_metrics["kills_t0"] + mirror_metrics["kills_t1"] > 0).mean()
mirror_decisive_pass = mirror_decisive_rate >= 0.50

# 否则报错: "Mirror baseline degenerate (0-0 stalemate). Exploit_gap meaningless."
```

加到 G0 verdict 表:
| check | threshold |
|---|---|
| mirror decisive rate ≥ 0.50 | (新,防退化) |
| exploit_gap ≥ 0.5 | (原) |
| CI excludes 0 | (原) |
| BR win rate ≥ 0.55 | (原) |
| BR healthy | (原) |

### 6.3 BR 训练加长

`br_iters` default 200 → 500(~15-20 min)。给 PPO 时间学 freq_hop × jam 的非平凡策略。

---

## 7. 实施顺序(避免 race condition)

1. **Fix 1 env+AC+extremes** (~3h): 改 env 物理 + AC head + ExtremeCommander。每改完跑 smoke (`pytest tests/twoteam/test_wp1_smoke.py -v`)。
2. **Fix 2 exposure** (~30min): env 参数改 + 直接损伤项。跑 `pure_jam vs pure_jam` sanity 确认不再 0-0。
3. **Fix 3 StrongRule** (~30min): 加 freq_hop 决策 + 放松 duck。
4. **Fix 4 WP0-decisive** (~30min): 改 run_wp0_check.py。**先跑这个**(便宜的早报警)。
5. **WP0 全跑** (~5min): 若 WP0-decisive FAIL → 回 1-3 调参;若 PASS → 进 6。
6. **Fix 5 G0** (~25min): BR 500 iters + 30 eps eval。出 G0 verdict。

总预算: ~6-7h。

---

## 8. 硬退场判据(bounded,防兔子洞)

修完跑完 WP0-decisive + G0 后:

| 场景 | 判 |
|---|---|
| WP0-decisive PASS + G0 PASS | ✅ 进 WP2 self-play |
| WP0-decisive PASS + G0 FAIL(BR 真打不过强化规则) | 诚实退 IET(root A 确认) |
| WP0-decisive FAIL(decisive_rate < 0.5) | env 仍退化,回 §2-3 再调;不允许进 G0 |
| 任何 NaN / 数值不稳 | 立即停,debug |

**只允许 1 轮再调**: 若 WP0-decisive FAIL,允许回到 §2 调一次参数(`exposure_gain`、`exposure_overload_threshold`、`processing_overhead` 公式)。第二轮仍 FAIL → 退 IET。

---

## 9. 退路(retreat friendly)

若决定退 IET:
```bash
# 保留新 env(作为 IET 的 baseline 复用)
git checkout -b twoteam-env-fix-retreat
git commit -am "retreat: twoteam env fix spec + partial impl (IET fallback baseline)"

# 回到 WP1 FAIL 状态(若用户想保留 G0 FAIL 数据)
git checkout appint/data-preflight   # 已 push 的状态
```

**复用机会**: 修好的 env(带 freq_hop)可作为 IET paper 的 baseline testbed — "我们提出一个抗干扰多功能博弈 testbed" 是 IET 的合理 contribution。

---

## 10. 风险登记

| # | 风险 | P | 缓解 |
|---|---|---|---|
| F1 | freq_hop 公式把 track 全压垮(overhead 太重) | M | §2.5 sanity 表先验证;overhead 用 √hop 而非 hop |
| F2 | exposure_gain=200 引发数值不稳 | L | 加 exposure clamp at 100;monitor NaN |
| F3 | StrongRule 修后变得太强(anti-strawman TOO_STRONG) | L | 阈值 ≤0.80 vs balanced;若超 → 降 sharpness |
| F4 | BR 训 500 iters 仍 local opt | M | 加 entropy_coef 0.02→0.03;若仍失败 → 是 root A 信号 |
| F5 | freq_hop 维度让 AC 训练不稳(Beta 分布数值问题) | M | dirichlet_min_concentration=0.5 守护;log_prob clip |

---

## 11. 立即开始?

**推荐执行顺序**:
1. 先确认 spec(本文件)无异议
2. Fix 4 (WP0-decisive 判据) — 30 min,**先做**(最便宜的早报警)
3. 跑现有 env(未改)的 WP0-decisive 看是否 FAIL(预期 FAIL,印证误诊)
4. 进 Fix 1-3 实施
5. WP0 重跑(应 PASS)→ G0 重跑

**等用户确认 spec → 开始 Fix 4**。
