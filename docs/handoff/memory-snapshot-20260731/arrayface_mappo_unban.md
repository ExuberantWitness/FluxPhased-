---
name: arrayface-mappo-unban
description: 2026-07-31 用户解禁 array_face 路线的 MAPPO forbidden;路线从 4 phase 扩展到 7 phase(S1-S4 单 agent + S5-S7 multi-agent)
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c76cde4-1c23-4df6-94f5-ee29ee81afbc
---

2026-07-31 用户裁定前 plan §11 的 "MAPPO forbidden" 是脑残设计,删除。Array-face 路线从 4 phase 扩展到 7 phase,**终点 = 完整 two-team MAPPO**(2 jammer vs 2 radar)。

**Why**: 用户终局目标是 MAPPO 级多 agent 对抗,原 plan §11 因 MFR-IQ + two-team WP-3 历史失败而过度保守。S1 multi-seed 结果(3/5 broke,探索策略不能救 stuck seed)证明单 agent PPO 也有局部最优问题,继续把 multi-agent 当禁忌没意义。

**How to apply**:
- 路线变为 S1-S7。S2-S4 单 agent PPO(jammer ULA → cell binding → 2D UPA),S5-S7 引入 multi-agent(2 jammer 协同 → radar 升级 → 完整 two-team MAPPO)
- 每个新 phase 必跑 multi-seed(≥3 seed),不再单 seed
- S5 启动前要复现 S4 饱和点作为对照(不能直接跳 multi-agent)
- 历史 MAPPO 失败教训(MFR-IQ Phase B / two-team WP-3 / WP-3.2 Phase A2 / Idea E)仍作为 §12 风险备忘,不作为禁忌
- 详细规格见 [/home/ubuntu/.claude/plans/memory-mossy-cupcake.md](/home/ubuntu/.claude/plans/memory-mossy-cupcake.md) §1 phase 表 + §6.5/6.6/6.7 S5-S7 设计 + §11.2 历史教训

相关:[g3-bsta-lite F5+F6 line done](g3_bsta_lite_f56_line_done.md) 是 lite 单 agent PPO 的天花板(0.2628),S1 已达 0.2372(gap 2.6pp),S2-S4 单 agent 路径在物理复杂度上扩展,期待饱和点 ~0.25-0.30,S5-S7 multi-agent 期待显著超此。
