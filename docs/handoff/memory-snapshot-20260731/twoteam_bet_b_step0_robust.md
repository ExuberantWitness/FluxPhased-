---
name: twoteam-bet-b-step0-robust
description: Bet B Step 0 — Rule kill capacity invariant to 6-axis physics perturbation; Bet B premise dead; 4th near-Nash evidence line
metadata: 
  node_type: memory
  type: project
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

Bet B Step 0 sweep 完成 (2026-07-14): Rule (StrongRuleCommander) vs pure_track baseline 跨 6 物理轴 + geometry 共 14 grid points,**0/13 cliff 触发**(threshold = WR drop > 0.15 from nominal)。

**结果**(bidirectional 100 ep × horizon=200 × n_envs=8 per grid):
- Nominal WR (rule vs pure_track): 0.983
- 所有 14 grids WR range: 0.964 - 1.000
- 最大下降: `exposure=400` Δ=-0.019(远低于 cliff 阈值)
- **Rule kill 容量几乎不变**: 1.90 - 1.97 / 2 radars across all grids(CV ~1.3%)

**关键诚实评估**:
- WR vs pure_track 是**弱证据**(pure_track trivial,margin 本来就 0.95+)
- **强证据是 kill 容量**: rule 的机制级 kill 能力对 6 轴物理扰动(jam_gain / range_sigma / sigma_q / exposure_gain / radar_separation / map_size + geometry)几乎完全不变
- 若 rule 在某轴脆,kill 会显著下降(jam_gain=9 应让 rule track 被 jam 严重影响)→ 但 kill 不变,证明 anti-jam hop reaction 在所有 jam 强度下都正常工作

**Why**: Bet B 前提("rule off-nominal 脆 → DR-RL 在 off-nominal 击败 rule")被证伪。Rule 不脆。

**How to apply**:
- 4 线近 Nash 证据已成(G0 #3 + V1 exploits + 设计分析 + Step 0 kill 容量不变)
- IET 地板现在是硬证据,不是 fallback
- 转 IET:testbed + BC pipeline + 4 线近 Nash 验证
- 不要再开 Bet B Step 1(DR 训练)— Step 0 hard stop 触发,违反 spec 纪律
- 相关:[[twoteam-g0-discipline]] [[twoteam-multifunction-pivot]] [[twoteam-wp2-exploits-fail]] [[taes-wp12-g1-partial]]
