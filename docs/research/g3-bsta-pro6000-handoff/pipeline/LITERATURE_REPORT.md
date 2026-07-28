# 文献调研报告：FluxPhased MFR-IQ G2'a 可达性重构

当前固定版见 [LITERATURE_REPORT_20260728_023103.md](LITERATURE_REPORT_20260728_023103.md)。该版本综合雷达检测/跟踪物理、认知干扰资源分配、MFR 调度 oracle 与 RL 评估方法，形成后续实施规格的证据基础。

## 固定结论

- 当前 `max(0.1, 1/sqrt(1+JNR))` 在约 20 dB 起产生硬饱和，可能造成策略不可辨，但最终 duty parity 必须实测；它不能作为跨任务的通用物理模型。
- 新进度应按 task type 从 post-processing SINR 映射到 `P_d`、Fisher 信息或互信息，并与 IQ Monte Carlo 校准。
- jammer 必须在 learned/scripted/oracle 共享、且有平台依据的 per-emitter power、energy、beam、service 和 channel 可行域中行动。
- 训练前必须有 reduced exact oracle 或 full planner witness 证明保守 headroom；文献本身不保证 5pp。
- target/beam/power allocation + PPO 已有直接近邻，算法新颖性低；可辩护方向是 benchmark pathology、IQ 校准修复与 admissibility protocol。
- 独立 training seeds 的工程下限为 8，但正式 N 必须由备择效应、full-budget 方差和 power analysis 决定；正确原假设是 `H0: delta <= 0.05`。
