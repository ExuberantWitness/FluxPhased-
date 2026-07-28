# Research Brief：FluxPhased MFR-IQ Learning Jammer 可达性重构

## Problem Statement

FluxPhased MFR-IQ Phase B 的 G2'a gate 要求 learning jammer 在 rule radar 对手下的 `drop_ratio` 比最强 scripted jammer 高至少 5 percentage points。当前报告称：

- `noise` scripted jammer 是 always-on，`drop_ratio≈0.520`；
- PPO 的 sampled Bernoulli 策略约为 0.519–0.520，greedy 约为 0.471–0.491；
- 环境通过 `progress *= max(0.1, 1/sqrt(1+JNR))` 将干扰耦合到任务进度；
- 现有动作只有每个 off-board jammer 的 on/off。

该设计可能同时存在 dominant-action、硬饱和、reward/metric 不一致、随机策略不可辨识和统计 gate 错配。目标不是继续调 PPO，而是重构一个物理合理、对所有策略公平、在训练前可证明存在 ≥5pp headroom 的 jammer decision problem。

## Bottom-Line Problem Anchor

在保持 MFR-IQ 仿真主线、rule radar 对手和 raw `drop_ratio` 主要评价指标的前提下，构造一个具有真实受限资源分配决策的 learning-jammer 环境，使：

1. always-on 不再是无代价的支配动作；
2. 学习策略的优势来自因果 observation 下的 target/channel/time allocation，而非后验调参或 baseline 故意弱化；
3. 在启动 PPO 前，通过 oracle/headroom gate 证明“超过最强 competent scripted 5pp”至少在动作空间内可达；
4. 结果可由独立 training seeds、版本化脚本和原始数据复现。

## Research Questions

1. JNR 应如何通过检测概率、信息积累或 measurement covariance 影响 MFR progress，避免人为硬饱和？
2. 哪种最小资源约束能让 jammer 策略产生真正的时空/频谱分配问题？
3. hard energy budget、simultaneous-beam cap、frequency allocation、target selection 中，哪种组合最小且足够？
4. 如何定义 oracle upper bound、competent scripted family 和统计 superiority gate？
5. 如何区分 PPO 学习到状态依赖策略与初始化 Bernoulli 随机策略碰巧命中饱和指标？

## Constraints

- Target executor：PRO6000 agent。它是否具备未同步的 M7 源码尚未验证；实施前必须提供 `SOURCE_HANDOFF.json`（repo、commit、URI/path、hash、owner），否则只能执行 P0 盘点并返回 provenance blocker。
- 当前审计机找不到 `env/gpu/mfr/*`、`algo/_shared/pilot/mfr/*`、M7 checkpoint 或 `/tmp` gate scripts，因此本轮输出必须是符号解析优先、文件级可执行的实施规格，而不是假装已应用的 patch。
- 保持 rule radar 主对手和 raw `drop_ratio`，除非明确建立独立的资源效率次指标。
- 所有 learned/scripted/oracle 策略必须遵守相同 observation、power、energy、bandwidth、beam 和 timing 约束。
- 不允许根据 test seeds 调物理参数以“制造 RL 胜利”。
- 先运行低成本诊断与 reduced oracle；headroom 不足则停止训练。

## What We Already Tried

- 无 σ-coupling：scripts 与 PPO 都约 0.203–0.205，缺乏干扰物理信号。
- 加 `1/sqrt(1+JNR)` 和 0.1 floor：scripted 分离，但 always-on noise 达到 0.518–0.520。
- PPO entropy annealing：sampled 指标不变，greedy 更差。
- 原 gate 报告错误地对 blink 而非 best scripted noise 做 t-test，且只有 training seed 0。

## Non-Goals

- 不通过降低 scripted baseline 能力来让 RL 过 gate。
- 不用 learner-only active-cost 修复 raw-drop superiority。
- 不把 “sampled mean 与 noise 四舍五入相同”解释为已学习。
- 不在本阶段执行大规模 GPU 训练或声称实验结果。
- 不后验把原 superiority gate 改写为 parity 并宣称 PASS。

## Success Criteria

- PRO6000 agent 能按照文件级任务清单完成代码、单元测试、诊断脚本和实验 tracker。
- 物理更新有明确公式、单位、边界和 IQ/detector calibration test。
- `always-on` 在受限动作空间中不再可行或不再支配，且约束对所有策略一致。
- reduced/full oracle 相对冻结的 best scripted 具有保守 7–10pp headroom。
- PPO 显著超过 untrained、iid Bernoulli 和 shuffled-observation controls。
- 最终使用至少 8 个独立 training seeds；正式样本量还须由预注册备择效应和保守方差决定，并正确检验 `H0: Δ≤0.05`。
- 所有脚本、config、raw episode records、seed IDs、checkpoint hashes 和 reproduction commands 均版本化。
