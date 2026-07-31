---
name: twoteam-mfr-pivot-20260725
description: "2026-07-25 方向终裁:放弃负结果论文路线,改走 MFR 场景复现路线:调研 MFR 论文场景(优先多智能体)→复现场景→跑 MAPPO 基线→掌握 MFR 任务特点→复杂化环境(联赛等)→新算法 vs MAPPO。用户明确否决了'写负结果论文'的建议"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c76cde4-1c23-4df6-94f5-ee29ee81afbc
---

# MFR 场景复现路线(2026-07-25 用户拍板)

**Why**: 文献综述发现我们的 two-team env 抽象掉了 MFR 经典结构(无任务调度/dwell/更新率/Q-RAM),而"MFR × 对抗"是经典算法真空地带。用户判断:不要写负结果论文,应该回到 MFR 文献的场景设定,先复现、跑通 MAPPO、掌握 MFR 任务特点,再从联赛等角度复杂化,做出新算法相对 MAPPO 的优势。

**How to apply**:
- 路线六步: ①调研 MFR 论文场景(优先多智能体+MFR)→ ②设置/复现场景 → ③跑 MAPPO → ④掌握任务特点 → ⑤复杂化(联赛/对抗)→ ⑥新算法 vs MAPPO
- **用户明确否决"诚实负结果论文"路线,不要再提**
- 关键文献锚点: Witherell 2024 RadarConf MAPPO 子阵分配(多智能体 MFR 最接近者)、George 2022 DQN、Kosuru 2022 元控制器、Xu&Zhang 2020 雷达组网 Q-learning、Hashmi 2022 IET RSN 综述、Shaghaghi 2018 MCTS+蒸馏(性能上限基准)
- MFR 任务特点(待验证): 任务丢失率/时间占用率为主指标;高负载(过载)是区分度来源;更新率约束/dwell 时间是核心决策变量
- two-team 资产可复用: IQ 物理、IMM-PDAF、联赛 harness、PPO/MAPPO 训练栈(见 [[twoteam-idea-e-em-contest]] 深度归因: PPO 微调破坏 BC 链 = offline-to-online 退化, 文献对应 Cal-QL/PEx, 解法首选 loss 侧 BC/KL 锚定)
