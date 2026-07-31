---
name: twoteam-tier223-pfspfix-outcome
description: "三方案终审(2026-07-24) — PFSP 修复/asymmetric fine-tune/cv 机动全负;规则教师鲁棒性是算法性的,ResiP 结构继承不了也训不出;差距结构性非预算性"
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

# 三方案(PFSP 修复 + Tier 2.3 非对称 + Tier 2.2 机动)终审(2026-07-24)

**Why**: Tier 2.1 G3 gate PASS 后用户要求三方案全测。核心问题:能否找到 RL 反超规则教师的场景/训练配置。

**How to apply**: 后续任何"换场景/加多样性就能反超教师"的想法必须先过这个负结果先验;论文写法应转向"in-distribution 追平 + OOD 脆性"的诚实分析。相关 [[twoteam-tier21-adaptive-outcome]]。

## ① PFSP 修复(commits f7794ee 前启动,单变量消融)
- 机制:--pfsp-var-mix 0.5 --ema-var-uniform-floor 0.05(早已内置,G2 误用默认 0)
- 多样性修复确认:ema_var 0.023→0.062,对手轮换 vs 原版 90% 锁 hard_jam_focus
- **但性能显著变差**:vs AJR 0.445/0.445(原版 0.570/0.547,p=0.03/0.006);vs BC 0.766/0.688 打平
- 教训:多样性稀释针对性;"坍缩到最难对手"在单场景下反而接近最优采样

## ② Tier 2.3 asymmetric fine-tune(决定性负)
- env: ASYMMETRIC_GEOMETRY(两队独立采样,默认不变);league flags f7794ee
- 规则层:BC vs BC 镜像地板 0.000→0.05;教师 vs AJR 0.375/0.266→0.484/0.438(非对称削弱 follow-jam)
- RL 零训练迁移崩:0.156/0.188(教师 0.922)
- **200-iter 场景内 fine-tune 仅 +0.07/+0.09 → 0.227/0.281,离教师 -0.71/-0.66 (p<0.001)**
- @random 保持 0.766/0.750 → 场景间无干扰,差距纯结构性

## ③ Tier 2.2 cv 机动(诊断级)
- env: target_motion="cv"(heading 随机游走 15°/s,死人冻结,种子可复现),microverify 4/4
- 速度悬崖(BC kills):0→10 m/s 无损;20 m/s 塌 80%;50 m/s 全灭(激光 50m 半径+跟踪滞后)
- RL 迁移:10 m/s 0.27-0.30,20 m/s 归零;教师 20 m/s 仍 0.4

## 总结论
规则教师鲁棒性 = 算法性(IMM 跟踪器 + 几何不变波束数学),ResiP(冻结 BC+±Δ+KL 锚)
继承不了、200-iter 也训不出几何/运动泛化。**差距结构性,非训练预算问题。**
论文方向:in-distribution 追平教师 + OOD 脆性分析(三场景定量)是诚实且可发表的贡献。
