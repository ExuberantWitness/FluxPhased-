# 相控阵雷达 Rule-Based 资源管理策略 —— 文献调研报告

**日期**: 2026-05-30
**目的**: 为 FluxPhased 项目的 critic 预训练数据生成，调研现有相控阵雷达 rule-based 启发式策略

---

## Executive Summary

相控阵雷达资源管理有 **30 年以上的研究积累**，形成了从简单启发式（EDF）到复杂混合算法（HPEDF + PSO + 脉冲交错）的完整方法体系。

核心结论：
1. **HPEDF**（Highest Priority Earliest Deadline First）是简单启发式中性能最优的
2. **综合优先级函数**是关键设计要素——融合任务类型、威胁度、截止期、时间偏移率
3. **脉冲交错**可提升时间利用率 30-50%
4. **两阶段方法**（启发式预选 + 优化精调）平衡实时性与最优性

---

## 一、雷达任务模型

标准相控阵雷达任务表示为 dwell tuple：

```
Task = {priority, arrival_time, deadline, dwell_time, PRI, n_pulses, power, beam_dir}
```

| 任务类型 | 典型优先级 | dwell (ms) | 截止期特征 |
|---------|-----------|-----------|-----------|
| 验证/确认 | **5 (最高)** | ~6.8 | 紧，一次性 |
| 精密跟踪 | 4 | ~3.6 | 中等，周期性 |
| 普通跟踪 | 3 | ~6.8 | 宽松，周期性 |
| 区域搜索 | 2 | ~8.4 | 无硬截止期 |
| 空域搜索 | **1 (最低)** | ~7.2 | 批量调度 |

---

## 二、主要算法分类

### 2.1 基准算法（模板法）

| 算法 | 原理 | 性能 |
|------|------|------|
| EST | 按起始时间 head-to-tail 调度 | 低负载可用，高负载差 |
| ED (Earliest Deadline) | 按截止期排序 | 基线对比用 |
| 固定模板 | 预分配时间槽 | 资源利用率 <50% |

### 2.2 改进启发式（推荐应用于 FluxPhased）

| 算法 | 排名规则 | 特点 |
|------|---------|------|
| **HPEDF** | Np + Nd，优先 Np | **命中价值率最高** |
| MHPF | Np 优先，Nd 次要 | 高优先级保障 |
| MEDF | Nd 优先，Np 次要 | 低截止期遗漏率 |
| MCF/EMCF | 最小代价优先 | 时间利用率 ~6× EST |

**HPEDF 伪代码**:
```
1. 对所有待调度任务，计算综合优先级 S_i = Np_i + Nd_i
2. 按 S_i 降序排序
3. 取队首任务，检查剩余时间窗是否容纳 dwell_time
4. 若可容纳 → 调度到最早可行时刻
5. 若不可容纳 → 丢弃或降级到下个调度间隔
6. 重复 3-5 直至窗口满或任务空
```

### 2.3 脉冲交错技术

利用发射脉冲与接收脉冲之间的**等待期**插入其他任务，是提升资源利用率的关键：

```
TX ──[wait]── RX ──[wait]── TX ──[wait]── RX
        ↑ 插入其他任务 TX/RX
```

- **单波束模式**：发射期不允许其他任务发射
- **多波束模式（DBF）**：接收期支持同时多波束

### 2.4 两阶段混合方法（启发式 + 优化）

程婷团队（UESTC）的代表性工作：
- **阶段 1**：启发式调度（HPEDF）快速生成可行解
- **阶段 2**：粒子群优化（PSO）或遗传算法（GA）迭代改进

### 2.5 AI/ML 方法（近期趋势）

| 方法 | 代表工作 |
|------|---------|
| Q-Learning | 自适应驻留调度，MDP 建模（JSEE 2025） |
| MAPPO | 全数字相控阵 RRM（IEEE RadarConf 2024） |
| DRL + Transfer | 动态环境调度（IEEE TAES 2024） |
| MCTS + Policy Net | 认知雷达调度（2018） |

---

## 三、Benchmark 评估体系

| 指标 | 公式 | 含义 |
|------|------|------|
| **调度成功率 (SSR)** | N_scheduled / N_total | 成功调度任务占比 |
| **时间利用率 (TUR)** | Σ dwell_time / T_interval | 间隔利用率 |
| **时间偏移率 (TSR)** | Σ|t_actual - t_desired| / N | 偏差度量 |
| **命中价值率 (HVR)** | Σ w_i · I(scheduled) / Σ w_i | 加权成功率 |
| **任务丢失率** | N_dropped / N_total | 资源不足被删率 |

标准仿真工具：**Adapt_MFR**（DRDC Ottawa），因果式波束级仿真，IMM 跟踪器。

---

## 四、应用于 FluxPhased 的策略

### 4.1 任务映射

| FluxPhased 任务 | 雷达任务类比 | 优先级设计 |
|----------------|-------------|-----------|
| Recon (0) | 区域搜索 | 低优先级，填充剩余资源 |
| Detect (1) | 验证/跟踪 | **最高优先级**，紧截止期 |
| Jam (2) | 电子对抗 | 中等，按威胁度加权 |
| Comm (3) | 通信链路 | 中等，周期性 |

### 4.2 推荐 Rule-Based 调度策略

```
Algorithm: AdaptiveRadarScheduler

Input:  array [625 elems], tactical_state
Output: task_assignment [625], beam_dir [625], params [625, ...]

1. 威胁评估
   - 检测到目标的方向 → 高优先级 Detect
   - 未检测方向 → 低优先级 Recon

2. HPEDF 优先级排序
   S_i = w_task(task_type) * 2.5 + w_deadline * 1.5
   按 S_i 降序分配阵元

3. 阵元分配
   Detect: ~200 elems → 目标方向，LFM 波形
   Jam:    ~150 elems → 敌方方向，噪声波形
   Comm:   ~100 elems → Missile 方向，BPSK
   Recon:  剩余 elems → 扫描未覆盖方向

4. 能量约束
   瞬时功率 ≤ P_max
```

### 4.3 优先级权重

```
w_task = {
    "detect":  0.4,   # 最高
    "jam":     0.3,
    "comm":    0.2,
    "recon":   0.1,   # 最低
}
```

---

## 五、下一步行动

1. **实现 HPEDF 调度器** → `training/scripted_policy.py`
2. **设计综合优先级函数**：融入 FluxPhased 战术状态
3. **生成高质量 demo 数据**：rule-based 策略 50-100 episodes
4. **训练 critic**：MC returns 回归，不做 GAE 折扣
5. **联赛训练**：critic 初始化后纯 RL，不 force-launch

---

## 参考来源

- 程婷 et al., "基于启发式回溯的实时相控阵雷达波束驻留调度方法," CN114609589B
- H. Zhang et al., "Dynamic Adaptive Resource Scheduling for Phased Array Radar," arXiv:2409.19201, 2024
- 程婷 et al., "Real-Time Dwell Scheduling Based on a Unified Pulse Interleaving Framework," Tsinghua Science and Technology, 2024
- 程婷 et al., "Adaptive Dwell Scheduling for Digital Array Radar Based on Online Pulse Interleaving," Chinese Journal of Electronics, 2009
- A. Charlish et al., *Adaptive Radar Resource Management*, Elsevier, 2015
- J. Akbar et al., "Transfer-based DRL for Radar Task Scheduling," IEEE TAES, 2024
- M. Shaghaghi et al., "MAPPO for All-Digital MFPAR Resource Management," IEEE RadarConf, 2024
- "Adaptive Dwell Scheduling Based on Q-Learning for MFR," JSEE, 2025
