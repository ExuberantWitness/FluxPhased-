---
name: fluxleague-anchor-root-cause
description: FluxLeague 真根因是 anchor 坏（min_dist_init=139m），不是 alpha-collapse；所有 alpha/F1+F2 实验在 anchor 修好前都 inconclusive
metadata: 
  node_type: memory
  type: project
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

# FluxLeague 真根因 = anchor 坏，不是 alpha

P2 A baseline（F1+F2 OFF）3 iter 结果证伪了 "F1+F2 是 kill_rate=1.00 唯一原因" 的假设：A 与 B 在 team_value_loss (~1e5)、team_adv_std (~1e2)、kill_rate (1.00) 上**无差异**。

dart 诊断暴露真问题：
- `min_dist_init` 平均 **139m**（应 ≤1m）—— Kalman-fused anchor 偏离敌人 139m
- `fire_rate` = 0.524 —— policy 没学 fire commitment，仍 Bernoulli(0.5) 初始
- `min_dist_min` = 0.23m —— episode 内偶发，非策略学到
- kill_rate=1.00 是 kr=24.5m 宽松阈值白送，**不是学习信号**

**Why**: 用户（2026-06-28）明确指出战略顺序错误——先修 anchor 再跑 tight-kr。若 anchor 偏 139m 时收紧到 kr=5m，A 和 B 都会 kill_rate→0（瞄偏 139m 不可能击中 5m），又一次 inconclusive。alpha-collapse 可能只是下游现象：策略从没学会瞄准，所谓 "α>0.5 崩" 也许只是脆弱策略被 team 噪声扰动。

**How to apply**:
1. **不要**在 anchor 没修好前跑 §5 tight-kr 或 HEADLINE-LINEAR —— 都会 inconclusive
2. 必须先按 [[ANCHOR_DIAGNOSIS.md]] 的 5 候选 print 定位根因（最可能是 Kalman warm-start 没运行或 obs[68:70] 传递路径被覆盖）
3. 修好后回归验证：`[dart] min_dist_init` ≤1m 才能继续 alpha 实验
4. 论文叙事影响：当前**不能**写 "F1+F2 是 kill_rate 恢复的原因"；可以写 "kr=24.5m 下 kill_rate=1.00 是宽松阈值产物，不反映学习"

承认错误：之前宣称的 "F1 让 team_returns 改善 1917×" 是合成测试夸大；真实苹果对苹果只 ~15×，A/B 在 team_value_loss 上几乎无差。

相关：[[fluxleague-r01234-training]]（R0-R5 league 修复历史）；[[fluxleague-paper-framing]]（论文叙事）
