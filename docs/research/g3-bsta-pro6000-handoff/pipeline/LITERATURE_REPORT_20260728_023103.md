# 文献调研报告：FluxPhased MFR-IQ G2'a 可达性重构

**调研日期**：2026-07-28  
**范围**：雷达检测/跟踪物理、认知雷达与干扰资源分配、受约束决策、RL 评估方法  
**问题锚点**：不是寻找“更强 PPO 超参”，而是判断怎样重构环境，才能形成物理合理、公平且在训练前可证伪的 learning-jammer 决策问题。

## 1. 结论摘要

1. 当前 `progress *= max(0.1, 1/sqrt(1+JNR))` 不是可直接由通用雷达检测理论推出的模型。若 `progress` 表示累计 SINR、非中心参数或 Fisher 信息，在固定信号与噪声条件下，每步信息量通常随
   \[
   \mathrm{SINR}=\frac{S}{N+J}
   =\frac{\mathrm{SNR}_0}{1+\mathrm{JNR}}
   \]
   缩放，而不是随其平方根缩放；若 `progress` 表示标准差或其他量，平方根关系可能出现，但必须由具体估计器定义和 IQ Monte Carlo 校准，而不能再叠加任意 0.1 地板。
2. 现有公式在 `JNR >= 99`（约 20 dB）时达到 0.1 地板，因此该 progress 通道对更强干扰不再敏感。这**可能**让 50% duty 随机动作与 100% always-on 在最终 `drop_ratio` 上难以区分，但是否相等还取决于 jammer 聚合、cue、queue 和任务到达，必须通过 duty sweep 实测，不能从 floor 单独推出。
3. 认知雷达和协同干扰文献普遍把问题写成有限功率/能量、有限波束、目标选择、频谱匹配和时间分配，而不是给 learner 单独加 soft active-cost。最小合理重构应使用对 learned、scripted 和 oracle 完全相同的硬资源约束。
4. 小规模精确搜索或 branch-and-bound、较大规模 MCTS/beam search 可以在训练前充当“可行策略见证”。只有先找到相对冻结脚本基线留有 7.5pp 左右保守余量的策略，才值得投入 PPO。
5. “波束/目标/功率联合分配”本身已有大量先例；本项目可辩护的新贡献更可能是：IQ 校准的 MFR 任务进度、共享硬预算的 jammer POMDP、oracle-first 可达性 gate，以及区分“学到策略”和“随机策略碰巧进入饱和区”的控制实验。不要把通用 PPO 或资源分配本身包装成算法新颖性。

## 2. 检索方法与证据边界

- 并行检索两条独立主题线：
  - 检测/跟踪物理：SINR、`P_d/P_fa`、Swerling 起伏、Fisher 信息、互信息、IQ 校准；
  - 认知雷达/干扰决策：目标、波束、频率、功率、能量、POMDP、oracle 和强启发式。
- 另外检查 RL 实验设计的一手论文，重点是独立 training seeds、置信区间和小样本不确定性。
- 优先使用论文原文、IEEE/出版社页面、arXiv 作者版本和官方技术资料。
- 本轮没有目标仓库中 M7 的可执行代码和 raw artifacts，因此所有物理结论都是“应实现并校准的模型要求”，不是对未见源码的运行结果声明。

## 3. 检测与任务进度：应从什么量累积

### 3.1 检测任务：累计非中心参数/SINR，再映射到固定 `P_fa` 下的 `P_d`

Marcum 的脉冲雷达检测理论给出高斯噪声中阈值检测的基准：在固定 `P_fa` 下，检测概率由累计 SNR/非中心参数和积分脉冲数决定。Swerling 模型进一步说明，起伏目标必须对 RCS 分布积分，不能把平均 RCS 当作恒定回波。Shnidman 给出了若干标准目标模型下、给定 `P_d/P_fa/N` 所需 SNR 的近似反解。

对 FluxPhased 的直接含义：

- 先从 IQ 链或经过校准的链路预算得到每个 task、每个 dwell 的后处理 `SINR`；
- 检测任务累计与 detector 一致的非中心参数或充分统计量；
- 任务完成应由 `P_d >= P_d,target`、累计统计量过阈值，或一次显式检测事件触发；
- `P_fa` 必须固定并记录，不能只提高 `P_d` 而忽略 false alarm；
- Swerling case、相干/非相干积分和 looks 相关性必须进入配置及测试。

核心来源：

- Marcum, “A Statistical Theory of Target Detection by Pulsed Radar,” IEEE TIT 1960, [DOI](https://doi.org/10.1109/TIT.1960.1057560)。
- Swerling, “Probability of Detection for Fluctuating Targets,” IEEE TIT 1960, [DOI](https://doi.org/10.1109/TIT.1960.1057561)。
- Shnidman, “Determination of Required SNR Values,” IEEE TAES 2002, [DOI](https://doi.org/10.1109/TAES.2002.1039422)。
- Shnidman et al., “Radar detection probabilities and their calculation,” IEEE TAES 1995, [IEEE](https://ieeexplore.ieee.org/document/395246)。

### 3.2 跟踪/估计任务：累计 Fisher 信息，而不是直接缩放“进度”

主动阵列参数估计的 Fisher 信息矩阵在特定 likelihood、模型正确且条件独立的情形下可随 SNR 增长，并按正确的状态坐标变换组合；协方差下界约为其逆。该关系只能作为量测信息模块，不能替代包含状态转移、过程噪声、missed detection、data association 和 update 的完整动态跟踪递推。CRB 在低 SINR threshold 区还可能过度乐观。

对 FluxPhased 的直接含义：

- 若 task 是 track update，不应只让 `progress` 变慢；应让 `R_meas` 或 information increment 随 post-processing SINR 改变；
- 同时检查 missed detection、track continuity 和 covariance，避免“进度慢了但量测仍完美”的物理矛盾；
- 通过 raw-IQ Monte Carlo 比较经验量测误差与模型 covariance/CRB。

核心来源：

- Dogandžić & Nehorai, “Cramér–Rao Bounds for Estimating Range, Velocity, and Direction with an Active Array,” IEEE TSP 2001, [DOI](https://doi.org/10.1109/78.923295)。
- Zhang et al., “Power Allocation Scheme for Multi-Static Radar to Stably Track Self-Defense Jammers,” Remote Sensing 2024, [DOI](https://doi.org/10.3390/rs16152699)。

### 3.3 识别/表征任务：互信息可作 progress，但必须对齐最终任务

Bell 将能量受限波形设计写成最大化目标响应与观测之间的互信息；在高斯模型中，独立条件观测的信息可累计。互信息适合表征/识别类 task，但不自动等价于检测概率或 `drop_ratio`。

对 FluxPhased 的直接含义：

- 只有当 task 的语义是“减少目标不确定性”时才使用 MI progress；
- 必须验证 MI 与 `P_d`、估计误差或任务完成率的单调关系；
- 不要把一个通用 `progress_factor` 同时用于 detect、track、classify 等不同任务。

核心来源：

- Bell, “Information Theory and Radar Waveform Design,” IEEE TIT 1993, [DOI](https://doi.org/10.1109/18.259642)。
- Zhang et al., “Anti-Jamming Power Allocation … Based on Mutual Information,” Digital Signal Processing 2024, [DOI](https://doi.org/10.1016/j.dsp.2023.104335)。

### 3.4 频谱重叠和 jammer power 必须先进入 in-band interference

固定总功率下，spot 与 barrage jammer 的带内功率不同；天线增益、距离、频谱重叠、带宽和接收滤波决定进入 detector 的 `J`。因此推荐统一计算：

\[
J_{r,k,t}=\sum_b z_{b,t}P_{b,t}
G_{b\rightarrow r,t}\,\rho_{b,k,t}\,L^{-1}_{b,r,t},
\qquad
\mathrm{SINR}_{r,k,t}=\frac{S_{r,k,t}}
{N_{r,k,t}+J_{r,k,t}},
\]

其中 `z` 是波束/目标选择，`rho` 是 `[0,1]` 的频谱重叠，`L` 是传播损耗。所有策略必须调用同一个函数。

来源：

- NAWCWD, *Electronic Warfare and Radar Systems Engineering Handbook*, TP 8347, [PDF](https://ed-thelen.org/pics/Radar-NAWCWD-TP-8347.pdf)。
- Geng et al., “Radar and Jammer Intelligent Game under Jamming Power Dynamic Allocation,” Remote Sensing 2023, [全文](https://www.mdpi.com/2072-4292/15/3/581)。
- MIT Lincoln Laboratory, “Detection of Signals in Noise and Pulse Compression,” [课程入口](https://www.ll.mit.edu/outreach/web-based-course-radar-introduction-radar-systems)。

### 3.5 对当前 `sigma-progress` 的判定

当前模型：

```text
sigma_scale = sqrt(1 + JNR)
progress_factor = clamp(1 / sigma_scale, min=0.1)
```

可作如下严格判断：

- `min=0.1` 在 `JNR=99` 起生效，制造确定的硬饱和；
- 没有文献证据支持在所有 MFR task 上统一使用这条曲线；
- 若 progress 是累计 SINR/信息，候选倍率是 `1/(1+JNR)`，但必须通过 IQ/detector calibration；
- 若 progress 是标准差的倒数，则平方根可能合理，但应从 covariance update 推出，不能与 detector progress 混用；
- 新实现应保留 `legacy_sqrt` feature flag 只用于回归/消融，主模式应按 task type 选择 `pd_accumulation`、`fisher_information` 或 `mutual_information`。

## 4. 为什么硬资源分配比 active-cost 更合理

### 4.1 多波束、多目标、频谱与功率约束已有直接先例

协同干扰工作将动作显式写成 jammer-to-radar 波束矩阵和功率矩阵，并约束每台 jammer 的波束数、指向可达性、频谱匹配与总功率。另有工作使用分层策略：上层选 target/task，下层分配连续功率。

对 FluxPhased 的直接含义：

- `always-on` 不应通过 learner-only reward 变贵，而应在可行域内因总功率、episode 能量或 active-beam cap 受限；
- scripted、random、learned 和 planner 都必须经过同一 allocator/projector；
- 让动作表达“把有限资源给哪个 target/channel、给多少、何时给”，才有状态依赖策略空间。

核心来源：

- Xin et al., “Cooperative Jamming Resource Allocation with Joint Multi-Domain Information Using Evolutionary Reinforcement Learning,” Remote Sensing 2024, [全文](https://www.mdpi.com/2072-4292/16/11/1955)。
- Wang et al., “Hierarchical Reinforcement-Learning-Based Joint Allocation of Jamming Task and Power for Countering Networked Radar,” IEEE TAES 2025, [IEEE](https://ieeexplore.ieee.org/document/10693358)。
- Shang et al., “Research on Resource Allocation Strategy of One-to-Many Radar Jamming Based on Reinforcement Learning,” 2022, [出版社页面](https://xuebao.sjtu.edu.cn/sjtu_en/EN/abstract/abstract44207.shtml)。

### 4.2 2025–2026 closest work / novelty boundary

截至 2026-07-28，以下工作比一般的“认知干扰”文献更接近本项目拟议的 target/beam/power/time allocation。它们共同表明：**受限资源下用 RL/PPO 选择干扰目标、波束、类型或功率已经是拥挤方向，候选 A 的算法新颖性低。**

| Closest work | 与本项目的直接重叠 | 尚未被该文覆盖、但仍需本项目实验证明的差异 |
|---|---|---|
| Yang et al., “Optimizing Jamming Type Selection and Power Allocation for Countering Multifunctional Radar Network Based on IMAHPPO Algorithm,” IEEE TAES 2025, [DOI](https://doi.org/10.1109/TAES.2025.3564286) | 直接面向 MFR network；多 beam；功率约束；混合 PPO/MARL；长期干扰效能 | 未见 raw-IQ detector calibration、共享 episode-energy 或训练前 headroom stop gate |
| Wang et al., “Joint Optimal Allocation of Resources for Multiple Jammer Based on Multi-Agent Deep Reinforcement Learning,” IET RSN 2025, [DOI](https://doi.org/10.1049/rsn2.70031) | 多 jammer 动态选择 beam/target 并分配功率，以降低组网雷达融合检测概率；与 PSO、SAO、MADDPG、MAPPO 比较 | 未见因果 observation 审计、跨 episode 硬能量或独立 training-seed margin gate |
| Hao et al., “Dynamic Jamming Policy Generation for Netted Radars Using Hybrid Policy Network,” Applied Sciences 2025, [DOI](https://doi.org/10.3390/app15168898) | PPO2 hybrid policy 联合 beam selection/power allocation，并随 radar state switching 自适应 | 功率主要进入 reward/逐步资源指标，不等同于共享硬 episode-energy feasible set；仍是功能级仿真 |
| Cai et al., “A Cooperative Jamming Decision-Making Method Based on Multi-Agent Reinforcement Learning,” Autonomous Intelligent Systems 2025, [DOI](https://doi.org/10.1007/s43684-025-00090-4) | 每步选择 idle/target/jamming type/power；用干扰后的 SNR 计算 `P_d`，并驱动 search–confirmation–tracking 状态转换 | 无 MFR-IQ task/drop calibration；没有 oracle-first reachability protocol |
| Song et al., “VAM-Enhanced Deep Reinforcement Learning for Cooperative Jamming Task Allocation,” Symmetry 2026, [DOI](https://doi.org/10.3390/sym18020295) | PPO、多 jammer–多 target、平台约束、动态 target assignment 和强启发式预训练 | 对象是通信目标，未建立 radar detector/Fisher task semantics |
| Yang et al., “Joint Beam and Power Allocation for Cooperative Jamming in Terrain-Constrained Environments,” IET RSN 2026, [DOI](https://doi.org/10.1049/rsn2.70174) | 动态可见 radar 集、beam-count cap、每平台总功率和联合 beam/power allocation；以检测/定位概率为目标 | 非 RL，且未使用 episode-energy；但已直接覆盖 constrained beam/target/power formulation |
| Zhang et al., “An Optimized Soft Actor-Critic-Based Quantized Decision Method for Jamming Power Allocation Against Networked Radars,” Signal Processing 2026, [DOI](https://doi.org/10.1016/j.sigpro.2026.110765) | 有限干扰资源、动态量化功率决策、SAC/PPO/TD3 比较和累计能耗 | 不覆盖本项目拟议的 IQ-calibrated MFR task progress 与 causal headroom gate |

据此，新颖性边界必须写成：

- **Novelty: LOW**，适用于 target/beam/power allocation、masked action、PPO/MARL 和一般 radar-state adaptation；
- 可能可发表的差异只在“可复现的 IQ-calibrated MFR benchmark 修复、严格因果 observation contract、same-feasible-set comparator、oracle-first admissibility protocol，以及由实验确认的 benchmark pathology/negative finding”；
- `5pp` superiority margin、`7.5pp` pre-PPO headroom、相邻 budget sensitivity、最低 training-seed 数和具体置信区间均是**项目预注册选择**，不是上述论文或通用 RL 方法学推出的自然常数；
- 不得声称 first RL/PPO jammer allocator、first target/beam/power allocation、first SINR-to-`P_d`/Fisher progress、first oracle comparison，或任何算法优先权；
- 即便组合此前未以完全相同的工程形式出现，“把已有组件接入 FluxPhased”也不能单独构成算法新颖性；必须由公开 artifact、物理校准和非平凡经验发现支撑。

### 4.3 频率敏捷适合作为第二阶段，而不是第一修复

频率捷变雷达与 jammer 可写成 POMDP/不完美信息博弈；历史观测或 recurrent policy 对延迟频谱感知有价值。但若第一阶段同时引入 radar hopping、jammer sensing delay 和 co-learning radar，会混入非平稳性并扩大 M7 范围。

因此：

- 若现有两 carrier 已有物理语义，可在核心动作中加入 `channel_id` 和 spectral overlap；
- 不要在 v3 第一版同时训练 radar hopping policy；
- 只有当预算化目标/功率分配的 planner headroom 仍不足时，才升级为动态频率 POMDP。

来源：

- Ak & Brüggenwirth, “Avoiding Jammers: A Reinforcement Learning Approach,” [arXiv](https://arxiv.org/abs/1911.08874)。
- Li et al., “Counterfactual Regret Minimization for Anti-Jamming Game of Frequency Agile Radar,” [arXiv](https://arxiv.org/abs/2202.10049)。
- Xia et al., “GA-Dueling DQN Jamming Decision-Making Method for Intra-Pulse Frequency Agile Radar,” Sensors 2024, [全文](https://www.mdpi.com/1424-8220/24/4/1325)。

## 5. Baseline 与 oracle 的文献启示

MFR 调度文献常见三级比较：

1. 可解释的 threshold、priority、equal-share、greedy/Q-RAM 启发式；
2. 小实例 brute force 或 branch-and-bound 精确解；
3. 大实例 MCTS、policy-guided search、进化优化等近似 planner。

这比“noise/reactive/blink 三个简单脚本”更适合作为 G2'a 评估结构。对本项目：

- reduced environment 用 exhaustive DP/B&B，给出真正 exact optimum；
- full environment 用 MCTS/beam search/CEM，称为 `planner witness`，除非有证明，不得称 upper bound；
- baseline 至少加入 budgeted barrage、budgeted blink、reactive channel/beam follower 和 myopic marginal-drop-per-joule；
- 使用 calibration seeds 冻结脚本 family 和参数，再锁 test seeds。

来源：

- Shaghaghi et al., “Multifunction Cognitive Radar Task Scheduling Using Monte Carlo Tree Search and Policy Networks,” [arXiv](https://arxiv.org/abs/1805.07069)。
- Krishnamurthy & Djonin, “Optimal Threshold Policies for Multivariate POMDPs in Radar Resource Management,” [作者 PDF](https://people.ece.ubc.ca/vikramk/DK09.pdf)。
- Durst & Brüggenwirth, “Quality of Service Based Radar Resource Management Using Deep Reinforcement Learning,” [arXiv](https://arxiv.org/abs/2010.10210)。
- Lu et al., “Multi-Objective Reinforcement Learning for Cognitive Radar Resource Management,” [arXiv](https://arxiv.org/abs/2506.20853)。

## 6. RL 评估：G2'a 统计设计必须重写

原报告的 8 个 evaluation RNG seeds 不能替代 8 个独立 training seeds。单 checkpoint 的多次 rollout 只能降低该 checkpoint 的评估噪声，不能估计训练算法方差。

建议：

- 8 个独立 training seeds 只是项目的最低工程下限，不是文献保证的充分样本量；正式 N 必须基于预注册备择效应、full-budget pilot/外部方差及其保守上界做 power analysis；
- 每个 checkpoint 在共同的 locked environment seeds 上评估；
- stochastic action RNG 与 environment RNG 分离并记录；
- 比较单位是 training seed 聚合后的差值；
- 检验 `H0: E[learning - best_script] <= 0.05`，不是 `H0: delta=0`；
- PASS 等价于一侧 95% lower confidence bound 严格大于 0.05；
- 报告点估计、区间和每 seed 原始数据，不只给 p 值；
- parity 必须使用预注册 non-inferiority/equivalence margin，不能用 `p>0.05` 推断“相同”。

来源：

- Henderson et al., “Deep Reinforcement Learning that Matters,” AAAI 2018, [论文页](https://ojs.aaai.org/index.php/AAAI/article/view/11694)。
- Colas et al., “How Many Random Seeds?,” [arXiv](https://arxiv.org/abs/1806.08295)。
- Agarwal et al., “Deep Reinforcement Learning at the Edge of the Statistical Precipice,” NeurIPS 2021, [论文页](https://papers.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)。
- Jordan et al., “Evaluating the Performance of Reinforcement Learning Algorithms,” ICML 2020, [PMLR](https://proceedings.mlr.press/v119/jordan20a.html)。
- Patterson et al., “Empirical Design in Reinforcement Learning,” JMLR 2024, [JMLR](https://jmlr.org/papers/v25/23-0183.html)。

## 7. 研究空白与对本项目的诚实定位

### 已被文献充分覆盖

- jammer 的目标/任务选择与功率联合分配；
- 多 jammer 的波束与功率约束；
- frequency-agile radar/jammer 的 POMDP 或不完美信息博弈；
- MFR 资源调度的 exact/heuristic/learning 比较；
- SINR 到 `P_d`、Fisher 信息或 MI 的物理桥。

### 本项目仍有价值、但需实验证明的组合

- 在同一 MFR-IQ 环境中，用经过 detector calibration 的 task-specific progress；
- learned/scripted/oracle 共用硬 team-power、episode-energy、beam/channel feasibility；
- 在训练前做 headroom gate，避免为不可达的 5pp 目标盲目训练；
- 用 untrained、random、observation-shuffled controls 证明 PPO 确实利用状态；
- 使用 raw `drop_ratio` superiority 与资源效率次指标并列、但不互相替代。

### 不能从文献推出的结论

- 不能保证推荐环境中 PPO 一定超过脚本 5pp；
- 不能保证 `1/(1+JNR)` 对所有 task 精确；
- 不能把近似 planner 称为全局 upper bound；
- 不能仅凭 sampled PPO 与 noise 四舍五入相等，就说 PPO 学到了最优策略；
- 不能通过后验选择预算或 baseline 参数来“制造”可学习余量。

## 8. 对后续方案的约束

后续候选与最终实施规格必须满足：

1. 首先恢复 M7 源码、脚本和原始结果，并建立 symbol/shape/unit map；
2. 物理模型先以 feature flag 落地，并用 IQ/detector grid 校准；
3. hard resource projector 是唯一动作入口；
4. 在 PPO 前先跑 fixed-duty sweep、competent baselines 和 oracle/headroom；
5. headroom 不足就停止训练并修改问题，而不是继续调超参；
6. 最终 gate 以独立 training seeds 和正确的 5pp margin null 为准。
