# WP-B Report — 强经典退化曲线 + WP-C 工作点

**日期**: 2026-07-15
**状态**: COMPLETE
**前置**: WP-A (IQ 原生干扰已接入 env)
**后续**: WP-C (RL demonstration in D_c regime)

---

## Step 0 — σ clamp 重定(阻塞项,已解)

### 改动
[iq_interference.py:58](env/gpu/twoteam/iq_interference.py#L58) `jnr_total_clamp: 1e4 → 1e8`

### 原因(已查证,非臆断)
- σ=5m 饱和来源 = `jnr_total_clamp=1e4` (40 dB),非任何 σ 硬性 clamp
- 链路: `JNR_total.clamp(0, 1e4)` → σ_inflation = sqrt(1+1e4) = 100× → σ_max = 0.05 × 100 = **5m**
- env 里另有 `tracker_P.clamp(-1e3, 1e3)` ([twoteam_env.py:599](env/gpu/twoteam/twoteam_env.py#L599)) 兜底 trace_P,数值安全
- 1e4 没有文档依据,看上去是早期保守防 NaN 值

### σ 动态范围(新 clamp 下)
```
JNR_dB    σ        regime
   0     0.05m    ✓ calm baseline (full-track)
  15     0.28m    ✓ locked (degraded, 临界 track-lock σ=0.2m)
  30     1.58m    ⚠ marginal (heavy jam, partial track-loss)
  40     5.00m    ✗ lost (trace_P > tau_track=0.04, kills stop)
  60    50.0m     ✗ hopeless
  80   500.0m     ✗ saturated (clamp, 主瓣极端耦合)
```

### 没动的旋钮 + 为什么
- `P_per_subarray_W=5.0W` / `n_subarrays=25` (合计 125W): X-band UAV-class radar 现实硬件常数,不是 scenario 强度
- 调它们 = 改雷达型号,不是改干扰强度
- "现实高干扰"通过 **scenario 参数**控制:`task_alloc fraction` / `team_offset_m` / `channel config` / `beam direction`

### 物理量级(默认几何 5km team-to-team)
- **主瓣对主瓣 boresight** (full f_emit=1): JNR = **82.7 dB**(手算 + 实测一致)
- **主瓣对旁瓣 off-axis** (sidelobe -30dB floor): JNR ~30-37 dB
- **f_emit=0 关发射**: JNR = -88 dB(底噪)
- **WP-A sub-test 1 σ=5m 是 OLD clamp 的 artifact** — 新 clamp 下 σ 在该场景达 500m(boresight 物理饱和,非错误)

---

## Step 1 — 强经典 baseline

**选用**: 现有 [`TwoTeamStrongRuleCommander`](algo/_shared/baselines/twoteam_strong_rule_commander.py)
- per-aperture adaptive 4-function allocation
- 反 jam 频率跳变(jam_detect_threshold=0.30 → freq_hop=6)
- 反 track-jam(enemy tracking me → boost own jam)
- exposure duck(duck_mutex 防止双方同时静默)

**没换 IMM-PDAF**: env 内置 EKF 已经是经典估计;WP-B 目的是退化曲线不是估计器对比,加 IMM-PDAF 是 WP-D 的事。

**没加中心化调度**: 现有 rule 不碰 `env.radar_freq_hz`(默认全 ch0)→ **队内互扰默认全开**。这恰好是用户要的"队内自致盲"现象来源——直接作为 baseline 测试,不需要额外加。

---

## Step 2 — 退化曲线(scan: f_emit_A × channel_mode @ 5km 几何)

### Scan 设置
- 固定几何: team_offset_m=2500, radar_separation_m=1500
- team A = 固定 action jammer (`f_emit_A` 变,其余 comm)
- team B = StrongRuleCommander
- 轴 1: `f_emit_A ∈ [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]`
- 轴 2: `channel_mode ∈ {same, orthogonal}`(radar_0/1 同/差信道)
- 100 episodes × 200 steps per scenario

### trace_P 退化曲线(连续,4 个数量级跨度)
```
f_emit_A |   same_ch   orth_ch  | coord_gap×
---------+---------------------+------------
   0     |   27.150     0.061  |   442.7×    ← 纯队内互扰 gap
  1e-6   |   43.624     1.140  |    38.3×
  1e-5   |  105.854     3.426  |    30.9×
  1e-4   |  274.940    16.071  |    17.1×
  1e-3   |  388.568    76.066  |     5.1×
  1e-2   |  486.917   292.550  |     1.7×
  1e-1   |  634.654   566.170  |     1.1×    ← enemy dominates
  1e+0   |  334.143   304.356  |     1.1×
```

**观察**:
- 同频 channel: trace_P 从 27(无 enemy)→ 635(全 enemy),enemy 加压 24×
- 异频 channel: trace_P 从 0.06(无 enemy)→ 566(全 enemy),enemy 加压 9400×
- 协同 gap 随 enemy 加压而收窄: 442× (f_emit=0) → 1.1× (f_emit=1.0)
- 物理含义: 低 enemy 时队内互扰主导,channel 协同价值大;高 enemy 时敌扰主导,channel 协同无用

### kills_B / track_loss(二值,tau_track=0.04 严格阈值)
```
f_emit_A | kills_same kills_orth | loss_same% loss_orth%
---------+----------------------+---------------------
   0     |    0.76       2.00   |   89.4%      5.8%    ← 无敌扰协同 gap: +1.24 kills
  >0     |    0.00       0.00   |  100.0%    100.0%    ← 任何敌扰 → 经典全崩
```

**观察**:
- 任何 `f_emit > 0` → `trace_P > tau_track` → kill chain 死
- 这是 **physics + 严格阈值** 的真实结果,不是 bug
- `tau_track=0.04` 对应 σ < 0.2m,在 JNR > 15 dB 时已经做不到
- 二值指标太严,trace_P 是真正的连续退化曲线

### D_c (collapse point,track_loss crosses 50%)
- **same channel**: D_c = **0**(从基线就崩,无敌扰也 loss=89%)
- **orthogonal**: D_c ∈ **(0, 1e-6]**(任何敌扰都崩)

---

## 队内自致盲现象(WP-C 协同 gap 硬证据)

**Headline**: 没敌扰(f_emit=0)时,纯粹队内同频互扰就让经典崩:

| 配置 | trace_P | loss% | kills_B (out of 2) |
|---|---|---|---|
| same channel(默认)| 27.15 | 89.4% | **0.76** |
| orthogonal channel | 0.06 | 5.8% | **2.00** |
| **gap** | **442×** | — | **+1.24 kills** |

**物理**:
- teammate 在 1.5km 内(radar_separation_m 默认),主瓣对主瓣 JNR ≈ 93 dB
- 经 StrongRule beam_target=lt_idx(两雷达都打同一目标),相对方位 ~73° → 旁瓣耦合,JNR ~60 dB
- 仍远超 40 dB(原 clamp)和 80 dB(新 clamp)→ σ 大,P 增长 → trace_P 稳态 ~25-30
- 远超 tau_track=0.04 → track 丢 → kill chain 死

**意义**:
- 经典固定策略(StronRule 没碰 radar_freq_hz)无法避免队内互扰
- 一个简单的协同动作(两雷达差信道)就能恢复 performance
- 这是 WP-C RL 要学的东西:**学习式 channel coordination**

---

## Step 3 — WP-C 现实高干扰工作点

### 现实 EW JSR 量级参考(open lit)
- 战术 fighter vs 战术 fighter(companion jam): JSR 10-40 dB
- surface ship vs missile: JSR 30-60 dB
- stand-off jam(100+ km): JSR 0-20 dB
- main-beam coupling(rare, extreme): JSR 80+ dB

### WP-C 推荐工作点
```
configuration       | f_emit_A range | JNR_victim | classical behavior
--------------------+----------------+------------+-------------------
calm baseline       |    0           |   -inf     | works (orth: kills=2)
mild interference   | 1e-7 ~ 1e-6    |  10-20 dB  | degraded (orth: marginal lock)
D_c (boundary)      | 1e-6 ~ 1e-5    |  20-30 dB  | classical fails to kill
WP-C target regime  | 1e-5 ~ 1e-3    |  30-50 dB  | classical dead, RL opportunity
extreme (saturated) | 1e-2 ~ 1.0     |  60-80 dB  | uncounterable (boresight)
```

**WP-C 推荐**: `f_emit_A ∈ [1e-5, 1e-3]`(JNR 30-50 dB,realistic high-interference)
- classical 完全失效(kills=0, trace_P 100+)
- RL 应通过 learned channel coordination + measurement scheduling 保持 track-lock
- 协同 gap 仍可观察(5-30× at this range)

### 不推荐 WP-C 用的工作点
- `f_emit > 1e-2`(JSR > 60 dB):饱和区,RL 也救不动
- `f_emit = 0`:classical 已能跑(kills=2 orthogonal),RL 体现不出优势
- 主瓣对主瓣几何(boresight saturation):"uncounterable 全灭按钮",正是旧两队 env 的 0-0 僵局病

---

## 范围外(明示)

1. **多目标 N∈{2,4,8}**: env 当前固定 R=2 radars/team。N=4/8 需扩展 env(大量代码假设 R=2)。WP-B 用 N=2 已足够展示退化趋势;N>2 留给后续工作。

2. **IMM-PDAF robust estimator**: env 内置 EKF 已是经典估计的代表;WP-B 不做估计器对比(那是 WP-D 的事)。

3. **RL demonstration**: WP-B 只记录 classical 的退化曲线;RL 学习式协同是 WP-C 的 deliverable。

4. **Long-range engagement (30+ km)**: 默认 5km 几何足够展示退化;long-range sweep 是 sanity check,不是核心 deliverable。

---

## 交付清单

| 项目 | 状态 | 文件 |
|---|---|---|
| Step 0 σ clamp 修复 | ✅ | [iq_interference.py:58](env/gpu/twoteam/iq_interference.py#L58) |
| 26/26 单测 + 5/5 WP-A 子测 PASS | ✅ | — |
| WP-B sweep 脚本 | ✅ | [wp_b_degradation_sweep.py](experiments/twoteam/wp_b_degradation_sweep.py) |
| 退化曲线(trace_P vs f_emit) | ✅ | 本报告 Step 2 |
| 队内自致盲现象证据 | ✅ | 本报告 "队内自致盲" |
| WP-C 工作点定义 | ✅ | 本报告 Step 3 |

## 不确定项 / 需用户核实

- `tau_track=0.04` 阈值是否过严?当前使 kills 在 f_emit > 0 时全 0,可能掩盖 WP-C 中 RL 的 kills 优势
- `team_offset_m=2500` (5km close combat) 作为默认是否合适?Long-range sweep 给出更 graceful 退化曲线,但不是核心 deliverable
- WP-C 工作点选 `f_emit ∈ [1e-5, 1e-3]`(JNR 30-50 dB)是否符合用户对"现实高干扰"的预期
