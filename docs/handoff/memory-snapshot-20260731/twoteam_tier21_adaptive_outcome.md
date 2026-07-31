---
name: twoteam-tier21-adaptive-outcome
description: "Tier 2.1 自适应对手场景全程结果 — G3 曾判 PASS,但 2026-07-24 P0 收益矩阵证伪:5σ 是跨协议伪影,同协议对照下 RL≈BC 统计不可分"
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

# Tier 2.1 自适应对手场景(AdaptiveJamRule)全程结果(2026-07-23~24)

**Why**: 对称镜像"师徒同源"死局后,用户选 Tier 2 非对称提前,首个场景=自适应对手+随机性。核心假设:教师确定性分集跳频对 follow-jam 可预测,RL 的 Beta 采样是反预测资产——首个 gate 可设"RL > 教师"的场景。

**How to apply**: 后续 Tier 2 场景设计、评价 RL vs 规则基线时,必须同时报绝对 kill 和 kill 差分两个视角;PFSP 坍缩时用 crossplay 实测而非训练内 wr。

## 产物
- AdaptiveJamRuleCommander(继承 BlindClassical 全 kill 链 + follow-jam + 3 路随机化:p_follow=0.6 episode 级风格切换、probe_eps=0.15、jam_boost∈[0.20,0.45],全部种子派生自 env._reset_count)
- commit 4be261f(commander+注册+G0 4/4+pool 13→14)

## G1 基线(n=64/条件)
- 教师 vs AJR: **0.344/0.344**(从对 BC 的 0.87 崩塌,假设证实);AJR 反杀教师 0.33-0.39 → 差分≈-0.02(教师其实只略劣)
- V1-long iter500 零训练 vs AJR: 0.438/0.562

## G2 训练(V1-long init,200-iter ResiP,池含 AJR,252.6min)
- kl 0.015-0.024、ent -2.24~-2.37、adv_std=1.00,健康
- **PFSP 坍缩**:后半程 ~90% 打 hard_jam_focus,ema_var 0.023

## G3 终态(final n=128,snapshots n=64)
- vs AJR 曲线(RL kills 低/高): it0 0.438/0.562 → it50 0.422/0.438 → it100 0.500/0.578 → it150 0.531/0.500 → **it200 0.570/0.547**
- AJR 反杀 RL(final): 0.633/0.664 → 差分 -0.062(p=0.31)/-0.117(p=0.055)
- 泛化 vs BC: RL 0.797/0.797 vs BC 0.875/0.906(p=0.24/0.08)——与 V1-long 终态 0.734/0.812 持平,无顾此失彼(risk#3 排除)

## 判决
- **预注册 gate PASS**: RL 0.570/0.547 > 教师 0.344 + 2SEM(0.088)=0.432,≈5σ —— 首次"RL 超越师傅"成立
- **⚠️ 2026-07-24 被 P0 收益矩阵证伪(Idea A pilot)**: 上述 5σ 是**跨协议伪影**——G3 用 seed42/64eps×2batch,G1 用另一 harness;同协议 8 节点收益矩阵对照下 rl worst-case 0.391-0.484 ≈ bc 0.406-0.422,统计不可分。**方法论教训:跨 harness/跨协议的 kill 数字绝不可直接比较,任何"超越师傅"结论必须同电池复测**
- **诚实阴影①**: RL 未赢 AJR 本人(低平/高边缘负)
- **诚实阴影②**: 差分视角教师≈-0.02 优于 RL≈-0.09 —— RL↔AJR 对抗更"血腥"(双方互杀都升),绝对 kill 高不等于相对优势
- 训练增量有限:200-iter 仅低干扰 +0.13、高干扰持平(vs 零训练基线)

## 下一步候选
- Tier 2.2 机动目标真值(env step 改动 ~40 LOC,IMM 已 CV+CT 就绪)
- Tier 2.3 非对称初始条件(reset L283-298 镜像硬编码需改)
- PFSP 坍缩修复(防单一对手过拟合)后再跑长训
