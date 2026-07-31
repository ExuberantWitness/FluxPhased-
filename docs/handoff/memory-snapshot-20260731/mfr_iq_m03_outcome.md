---
name: mfr-iq-m03-outcome
description: "MFR-IQ env M0-M3 结局 — G1 PASS(规则 0.09 vs 随机 0.251),G2 负结果(MAPPO 两轮平台 0.35-0.40 差于随机,用户裁定采纳);规则 near-Nash 第三条独立证据线;下一步=复杂化环境(对抗/联赛)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c76cde4-1c23-4df6-94f5-ee29ee81afbc
---

# MFR-IQ(env/gpu/mfr)M0-M3 结局(2026-07-27)

**G1 PASS;G2 FAIL,用户裁定作为负结果采纳(2026-07-27,"进入2"),不再追加训练预算。**

- G1:负载 1.5 下 rule drop 0.090-0.092(3 seeds)vs random 0.251,差 15.9pp;负载扫描 1.0/1.5/2.0 = rule 0.021/0.091/0.097。G1 带从规格 [0.25,0.45] 修订为 [0.05,0.20](env 比 Wang 抽象模型严格更宽容:无 alloc 冲突、coalition 线性跨 step 保留、自动伺服;≥10pp margin 未动)。
- G2:MAPPO 两轮(run1 基线 8 seeds→iter74;run2 drop shaping ×7 + entropy 地板 0.005→iter78-79),8/8 seeds 平台在 drop 0.353-0.397,iter 40→78 斜率≈0,**差于均匀随机 0.251,远差于规则 0.090**;无训练崩溃(entropy/kl/pl 健康)。shaping 只压住 run1 的回升,不改变平台高度。
- 结论:**规则 near-Nash 第三条独立证据线**(前两条:two-team WP-2 手工 exploit 3/3 全负、WP-3 production smoke 4/4 全负)。机制:纯调度+全可观队列是规则擅长的结构化问题;team-reward 信度分配摊薄丢弃因果;判据瓶颈类任务(通信每step≥2链/精确跟踪tracker预热)是协议式行为,试错学习极不友好。

**Why:** Wang 2025 的 RL 数字(drop 33.67% @151% 负载)与我们 MAPPO(35-40%)量级一致——RL 侧不差,是规则侧在规格语义下上限太高。在静态纯调度 MFR 里继续做新算法预期收益低。

**How to apply:**
- 下一步路线(用户已定):复杂化环境打破规则 near-Nash = scripted jammer → learning jammer → 联赛;之后新算法 vs MAPPO。
- 对照基线/floor 已固定:rule 0.090、random 0.251、MAPPO 0.35-0.40(负载 1.5)。
- 报告:`experiments/mfr/REPORT.md`;回归 136 passed(2026-07-27)。
- 坑:`experiments/mfr/mappo_l1.5_s*/train_curve.csv` 被最后一次 nohup 重启覆盖成 iter 0,完整曲线只在会话记录/报告里;复现用 `launch_8seeds.sh`。
- 负载口径:_LAM_SCALE 按 Σλτdη/(K·0.25)=1.5 标定(规格原文 /K 是 600% 过载,已注明偏差)。
- 最难任务类:通信(type5,判据强制每 step ≥2 子阵驻留)与精确跟踪(type0,需 trace_P<4 tracker 收敛)——IQ grounding 新增的"判据瓶颈"维度。

相关:[[twoteam_wp2_exploits_fail]]、[[twoteam_wp3_production_smoke_fail]]、[[twoteam_mfr_pivot_20260725]]、[[wang2025_params]]
