---
name: twoteam-wp-b-classical-collapse
description: "WP-B 2026-07-15 — 强经典退化曲线已扫;f_emit=0 时同频队内互扰就让经典掉 kills 1.24/2(协同 gap 硬证据);WP-C 工作点定 f_emit∈[1e-5,1e-3] (JNR 30-50dB)。"
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

2026-07-15 WP-B 完工:扫强经典(StrongRuleCommander)在 IQ 原生干扰下的退化曲线,定 WP-C 工作点。

**关键发现(HEADLINE — 协同 gap 硬证据)**:
没敌扰(f_emit=0)时,纯粹队内同频互扰就让经典崩:
- same channel(默认)vs orthogonal channel
- trace_P gap = **442×** (27 vs 0.06)
- kills_B gap = **+1.24** (0.76 vs 2.00 out of 2)
- 物理来源:teammate 在 1.5km,主瓣对旁瓣耦合 JNR≈60dB,σ_inflation≈1000×,trace_P 稳态远超 tau_track=0.04
- 这是 WP-C RL 要学的东西:**learned channel coordination**(env.radar_freq_hz 当前不是 action,StrongRule 也不碰它)

**Why**:用户 WP-B 交接的核心命题——"协同 gap 是 RL 立命点的物理来源"。WP-B 用 NO-ENEMY baseline 直接证明:不用任何敌扰,仅靠队内同频,经典固定策略就崩。这比"经典在狠处崩"更具体、更可证明。

**退化曲线(trace_P vs f_emit,4 个数量级跨度)**:
```
f_emit | trace_P_same | trace_P_orth | coord_gap×
0      |    27.15     |     0.06     |   442×   ← 纯队内互扰
1e-6   |    43.6      |     1.14     |    38×
1e-5   |   105.9      |     3.43     |    31×
1e-4   |   275        |    16.1      |    17×
1e-3   |   389        |    76.1      |     5×
1e-2   |   487        |   293        |     2×   ← enemy 主导
1e-1   |   635        |   566        |     1×
1e+0   |   334        |   304        |     1×   ← 饱和
```

**WP-C 工作点**: `f_emit_A ∈ [1e-5, 1e-3]`(JNR 30-50 dB,realistic high-interference per open EW lit)。classical 完全失效(kills=0, trace_P 100+),RL 应通过 learned channel coordination 保持 track-lock。`f_emit > 1e-2` 是饱和区(RL 也救不动);`f_emit = 0` 太轻(classical 已能跑)。

**Step 0 σ clamp 修复**:`jnr_total_clamp 1e4 → 1e8` (40→80 dB)。原因:5km 几何 + 125W 满功率主瓣耦合 JNR=82.7dB,在 1e4 clamp 下瞬间饱和 σ=5m,把 [10,50]dB useful 区间压成单点("saturated calm sea")。1e8 让饱和只发生在 boresight 极端场景;trace_P clamp(-1e3,1e3)仍兜底数值。

**How to apply**:进 WP-C 时把 env 默认 `task_alloc[:, enemy_team, :, 2] = uniform(1e-5, 1e-3)` 作为 DR 区间;`channel_mode` 默认 randomize 让 RL 自己学协调(不要 default orthogonal——那就把 gap 白送);**别扫 P_per_subarray/n_subarrays**(硬件常数,不是 scenario 强度)。

**潜在争议项(留给用户判断)**:
- `tau_track=0.04` 阈值可能过严:f_emit>0 时 kills 全 0,可能掩盖 RL 在 WP-C 的 kills 优势。trace_P 是真正的连续指标。
- 默认几何 5km 是 "close combat" 极端;long-range (30-100km) 给更 graceful 退化曲线但不是 WP-B 核心。
- 现有 StrongRule 的 jam_detect 反应在新 IQ JNR scale 下可能误触发(_last_jam_matrix clamp 到 [0,1],>0dB 就触发)。

详细报告 [[twoteam-wp-a-iq-native-pass]] 后续: [experiments/twoteam/WP_B_REPORT.md](/home/ubuntu/CODE/FluxPhased-/experiments/twoteam/WP_B_REPORT.md)。
