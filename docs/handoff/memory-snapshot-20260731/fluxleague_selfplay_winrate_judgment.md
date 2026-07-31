---
name: fluxleague-selfplay-winrate-judgment
description: FluxLeague self-play league 中 cum red 自然回落到 0.88-0.90 是健康均衡（≈ 4090 验证值），不是退化；判定要点
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

在 FluxLeague self-play league 训练中判定"是否退化"时，**不要把 cum red 从 1.0 自然回落到 0.88-0.90 当作失败信号**。

**Why**：早先在 4090 上验证过的、稳定成功的联赛，终值就是 cum red=0.88、blue=0.00。对手池里装的是 RED 自己近期的强快照——RED 打不过（也不该打过）自己的强副本，所以平均胜率从 1.0 降到 ~0.9 是"池子变强了"的标志，不是退化。0.90 ≈ 4090 成功值 0.88，本身就是好结果。

**How to apply**：
- 真正的退化信号是 **cum red 继续往下掉到 0.6-0.7**（v2 实测 0.58），不是从 1.0 回落到 0.9。
- v2 vs stable 的真正区别不在 iter-16 的 0.90，而在之后：
  - v2 继续崩到 0.58（因为 adv_std 爆到 235 把训练带垮）
  - stable 应该在 ~0.85-0.90 处平台化（adv_std 受控，峰值 40 vs v2 的 235）
- 判定抗退化是否成立，看一件事：iter 18-20，cum red 是稳在 0.85-0.90（成功，复现 4090），还是继续往下掉。
- kr 卡 0.34m（不追 0.24m）也是 OK 的——课程是 eval_kill_rate≥0.5 才收紧，0.34m 对多样对手池还能 cum red 0.90 说明这是稳健点，不是失败。
- 万一 cum red 真的继续往下掉（不平台化），那是单一策略打不过多样种群的非传递性极限——这是 Phase 2.3（3-role AlphaStar + Nash 混合）要解决的，**不该在 Phase 1 单策略 league 里硬怼**。一个 "main" 策略本来就不该独自压制整个池子。

**核心成果的定义**（不要被 cum red 转移注意力）：
- adv_std 全程稳定不爆（v2 235 → stable ≤40）= dartboard=0 起作用
- cmd_pl 不再"冲到 +0.2 再坍缩"（v2 模式）= PPO 不坍缩
- aim_res 2× 高（log_std_floor=-4 探索熵恢复）
- 训练 2.5× 快（dartboard=0 减负）

这四项才是 Phase 1 stable 配置的成功指标。cum red 是否平台化是辅助判定，不是主判定。

关联：[[fluxleague-paper-framing]]、[[fluxleague-r01234-training]]、[[fluxleague-anchor-root-cause]]
