---
name: twoteam-d2-multiteacher-pilot
description: D2 多教师路线 pilot PASS — BC 风格变体真实多样;V1 track-heavy 纯克隆(零 RL) kill 0.703/0.719 超全部 10 次 RL 跑;教师 task_alloc 30% jam 容量浪费是首个定量确认的教师弱点
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

# D2 多教师 BC 路线 pilot(2026-07-23)

## 背景

WP-3.2 Phase B.1 后诊断:RL 无法超越 BC 的真约束是"师徒同源"——ResiP 把 RL 锚定在
克隆附近(冻结 base + ±Δ 残差 + KL 锚),结构上禁止学新招。用户提出两条破路线:
D1 BC+联赛(漂移多样性)vs D2 多教师 BC 混合(来源多样性)。分析结论:D1 纯形式被
G2/G3 数据证伪(kl≈0.005/iter 漂移≈0,pool ema_var 0.07 同质化);D2 是 AlphaStar
初始种群的实际做法(多设置监督蒸馏种子)。

## Pilot 结果(决定性)

**Extreme 教师不能用作 BC 教师**: 全部走 legacy beam_target 路径,env 内部经 true_pos
换算方位 = godview(env/gpu/twoteam/twoteam_env.py:655)。变体方案:BlindClassical 加
`alloc_base` 参数移植风格(~6 LOC),保住盲态 + beam_direction 原生。

**神经-vs-神经 cross-play 矩阵是错误工具**: 4 个变体互打全 0-0——同架构同初始化的
确定性神经策略信道锁步更严重,矩阵测的是锁步程度不是风格强度。正确工具:各变体
vs 真实规则教师(G1 式 crossplay)。

**正确工具的判决(n=64/条件,SEM≈0.06)**:

| 变体 | 低干扰 | 高干扰 | vs V0(bc5: 0.484/0.609) |
|------|--------|--------|--------------------------|
| V1 track-heavy (alloc [0.05,0.70,0.15,0.10]) | **0.703** | **0.719** | **+0.22/+0.11** |
| V2 jam-heavy (alloc [0.10,0.15,0.60,0.15]) | 0.484 | 0.469 | -0.00/-0.14 |
| V3 agile-ECCM (阈值0.15+谨慎发射) | 0.422 | 0.438 | -0.06/-0.17 |

## 关键发现

1. **教师 task_alloc 是首个定量确认的弱点**: 默认 alloc 30% 容量给 jam,但 vs 该教师
   jam 无收益(杀伤全走 track→laser 链)。V1 压 jam→15% 加 track→70%,零 RL 超过
   全部 10 次 RL 跑的最好成绩(0.609)。V2 反向变差佐证同一机制。
2. **RL 够不到这个 +0.2**: beam_only 冻结 alloc 头;all 模式 logit 残差在
   kl≈0.005/iter 下 100 iter 挪不动 0.45→0.70 量级。"徒弟被绑在师傅身上"的定量证据。
3. **D2 定位**: 不是替代 RL,而是给 RL 更强更多样的起点。V1 base (0.70 起点)
   vs 教师差距 -0.20 才是 RL 真正攻关区间。

## 资产

- 变体 ckpt: checkpoints/blind/wp3_phaseC1_bcV{1_trackheavy,2_jamheavy,3_agile_eccm}/iter000_bc.pt
  (各 30k 样本 × 12 epochs,val_beam≤0.002,~1.2min/个)
- 矩阵脚本: /tmp/wp3_phaseC1_crossmatrix.py;变体训练: /tmp/wp3_phaseC1_bc_variants.py
- 报告: experiments/twoteam/wp3_smoke_phaseC1_V*_report.md, wp3_phaseC1_crossmatrix.md

## V1-long 500-iter 判决(2026-07-23 完成)

11 快照 crossplay(32 eps×2 dir,SEM≈0.057): 0.703/0.719 → it100 谷底 0.656/0.656
→ it450 0.797/0.797 → it500 0.734/0.812。后半段均值 vs 前半段 +0.047/+0.031(≈2σ/
1.3σ),斜率 +0.02/100it;it450/500 与教师差距收窄到 -0.05~-0.09(p=0.15-0.49,
统计打平)。教师自身 kill 随 RL 变强从 0.91 降到 0.85。**判决: "强起点+长训"部分
成立**——首次出现可辨上升趋势(bc5 G3 100-iter 完全平),但 500 iter 只爬到与教师
打平,未超越。图: experiments/twoteam/wp3_phaseC1_V1long_curve.png。
Ckpt: checkpoints/blind/wp3_phaseC1_V1_long500/iter{050..500,final}.pt。
结论强化 D2 定位: 来源多样性(起点)是当前最大杠杆(+0.2),长训只提供 +0.05 量级
慢爬坡;要"超越师傅"仍需结构松绑(解冻 alloc 头/对手建模/Tier 2 非对称场景)。

## 混合比例扫描(2026-07-23,决定性负结果)

V0:V1 = 100:0 → 70:30 → 50:50 → 30:70 → 0:100 五点(纯 BC 克隆,32 eps×2 dir):
低干扰 0.484 / 0.641 / 0.547 / 0.516 / 0.703;高干扰 0.609 / 0.562 / 0.547 / 0.484 / 0.719。
**所有混合都比两端点差**——BC 多模态平均:两教师差异主要在 task_alloc(track 0.45
vs 0.70),相似 obs 上冲突标签使网络学到风格平均,beam/ECCM 与任一风格不自洽。
**"不同 BC 比例混合产生种群"轴被证伪;D2 多样性必须来自多个独立蒸馏的纯种子**
(AlphaStar 实际做法: per-teacher 分别蒸馏,不混数据)。

## 下一步
相关: [[twoteam-wp3-2-phaseB-redesign]] [[twoteam-wp2-exploits-fail]]
