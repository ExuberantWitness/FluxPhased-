# Concerto-RRM 详细方案（给 PRO6000 agent）— 时间轴交错的经典+RL 认知雷达资源管理

**动机**：纯 RL 反复输给 ClassicalMPC 且退化成瞄准射击(见 EAAI_C2_GATE0_REPORT)。改走 **CRL2RT(arXiv 2502.10429)的"时间轴交错经典+RL+规则编排器"** 思想——RL 不替代经典,而是**沿调度时间轴与经典调度器分时掌控**。这解决三痛点:退化(经典掌控主节律保 QoS 平衡)、计算慢(RL 只训自己的时隙)、打不过 classical(交错的 RL 时隙补自适应 EW)。

## 0. CRL2RT 参考机制(从 `240611_探索创新点.py` 读出的实际写法)
- **逐步交替**:`i_step%2!=0` → 自适应 PID 掌控(L705-752);`i_step%2==0 且 IS_RL` → RL 掌控(`agent.select_action`,L678-691);
- **RL 只在自己的步训练**(L678 注释"只有执行了 RL 后再更新",L975 `update_parameters_policy`);
- **CPG 基础节律**(`HORIZON_CONTROL_ALLOCATION`,L653)给经典控制器跟踪的参考轨迹;
- 结果:快而稳的经典控制 + 学习的自适应控制在时间轴交错共存,提升 18–60% vs 纯 PID/MRAC。

## 1. 映射到认知雷达 RM(核心设计)
**调度时间轴** = 控制步(每个 `pulses_per_control` 块 = 一次孔径分配决策)。三角色沿轴交错:

| CRL2RT 元素 | Concerto-RRM 对应 |
|---|---|
| CPG 基础节律 | **经典 QoS-RRM 调度器的标称循环调度**(侦/干/通/探标准驻留模式) |
| PID 掌控奇数步 | **经典调度器掌控"标称时隙"**(稳定、按构造平衡四功能、不退化) |
| RL 掌控偶数步 | **RL 掌控"争用时隙"**(重调度孔径:反干扰/动态重分配/欺骗) |
| `i_step%2` 固定交替 | **编排器**:v1 固定比例 N:1;**v2 事件触发(创新点,见 §3)** |
| RL 只训自己步 | **RL buffer 只收 RL 掌控的时隙**(非 RL 步 mask 掉) |

## 2. 实现计划(FluxPhased 代码触点)
### 2.1 经典 QoS-RRM 调度器(扩 `algo/_shared/baselines/classical_mpc.py`)
现 `ClassicalMPC` 只会"瞄 anchor+开火"。扩成**四功能 QoS 调度器**:每步按各功能 QoS 余量(detect SNR、track 协方差、comm CRC、jam 需求)+ 优先级规则,**分配 25 个子阵(5×5)给侦/干/通/探/track**。这是经典底座(规则/贪心/或简单 MPC),**无学习**。产出:每步的子阵→功能分配 + 波束指向。

### 2.2 时间交错主循环(改 rollout / `train_laser.py` 的 step 循环)
```
每个 control step t:
  owner = composer(t, ew_indicators)          # 见 §3
  if owner == CLASSICAL:
      action = classical_rrm.schedule(state)   # 经典分配,不进 RL buffer
  else:  # RL 掌控
      action = rl_policy.act(state)            # 现 MAPPO/CTDE(噪声鲁棒)
      rl_buffer.add(state, action, ...)        # 只有 RL 步进 buffer(照 CRL2RT L678)
  env.step(action)
# 更新:只用 rl_buffer(RL 只学争用时隙的窄自适应策略 → 样本少、收敛快)
```
- 交接连续性:切换时经典/RL 共享同一 env 状态 + 上一步分配(避免抖动)。

### 2.3 四功能 QoS 奖励/指标(重激活被搁置的多功能奖励)
现 `algo/_shared/laser/reward.py` 已窄化成击杀。**从 FluxLeague/env 路径重激活四功能 QoS 项**(detect_snr/detect_coverage/jam_effectiveness/comm_reliability/recon_intel,原 `DenseRewardShaper`/`league_25x25.yaml` 里有)。**目标函数 = 四功能 QoS 能力满足度 under JSR**(不是击杀率)——RL 掌控的时隙用这个 reward,**避免退化的关键:经典掌控主时隙已按构造平衡四功能**。

### 2.4 自适应 EW(现成)
`jam_gain:8`、`exposure_gain:50`、`jam_cost`(ew_exposure 里有)。敌方自适应干扰某功能 → 经典固定调度无法重分配 → 失效 → 给 RL 争用时隙留出价值空间。

## 3. 编排器(composer)—— 这里是超越 CRL2RT 的创新点
- **v1 固定交替(baseline,复现 CRL2RT)**:经典:RL = N:1 时隙(如 3:1),`owner = RL if t%(N+1)==N else CLASSICAL`;
- **v2 事件触发(本方法创新)**:经典掌控标称;**RL 被触发进入接下来 K 个时隙当且仅当检测到自适应 EW**:
  ```
  if JSR > θ1  OR  trace(Kalman协方差 _trk_P) 膨胀 > θ2  OR  某功能 QoS 余量 < ε:
      接下来 K 步 owner=RL
  else: owner=CLASSICAL
  ```
  指标全可观测(JSR、`_trk_P`、四功能 QoS)。**v2 的贡献 = 自适应编排(vs CRL2RT 的固定交替)——干扰活跃时多给 RL 时隙,标称时经典主导。**

## 4. 实验(EAAI headline)
四功能 QoS + 自适应 EW,四方对比(同 env/预算/seed,QoS-under-JSR 指标):
| 方法 | 说明 | 预期 |
|---|---|---|
| **纯经典 QoS-RRM** | 固定调度 | 自适应 EW 下**失效**(不能重分配) |
| **纯 MAPPO** | 从零学 | 退化/不稳/慢 |
| **固定交错 Concerto(v1)** | 复现 CRL2RT | 赢纯经典 |
| **事件触发 Concerto(v2,本方法)** | 自适应编排 + 噪声鲁棒 critic | **赢纯经典 + 赢纯 MAPPO + 赢 v1** |
**消融**:v2 −事件触发(退回 v1)、−噪声鲁棒 critic、composer 阈值扫描(θ1/θ2/ε/K/N)。

## 5. Pilot 判据(先证伪,~1-2 周)
1. **证经典失效**:纯经典 QoS-RRM 在自适应 EW 下某功能 QoS 明显违约;
2. **证交错赢**:Concerto(v1/v2)QoS 满足度 **显著高于**纯经典 且 高于纯 MAPPO;
3. **证不退化**:四功能驻留分配全程非零(经典主时隙保平衡),不塌成单一击杀;
4. **证快**:RL 只训争用时隙 → 单 run 明显快于之前从零训整策略。
- 全过 → EAAI 故事成立(时间交错融合 + 四功能 QoS + 自适应编排创新),铺全套 + CRL2RT/CIRL 引用;
- 交错 ≈ 纯经典 → 调 EW 强度/功能冲突/composer;
- 都不行 → 退 Path A(传感 TAES)。

## 6. 贡献框架
- **C2(headline)**:**事件触发时间交错的经典+RL 认知多功能雷达资源管理**——规则编排器按 EW 指标沿时间轴自适应交替经典 QoS 调度器与噪声鲁棒 RL,QoS-under-JSR 上赢纯经典/纯 MARL;迁移并改进 CRL2RT(2502.10429)的时间交错(固定→事件触发),CIRL 作方法论锚;
- **C0**:四功能 MFAR EW 基准;**C1**:多基地融合 + CRLB。

## 7. 诚实风险
- **composer 阈值/交接**是关键超参:RL 时隙太多→退化,太少→无贡献,θ/K/N 要扫;切换要状态连续防抖动;
- 经典 QoS-RRM 调度器要写对(是本方案的底座,决定不退化);
- 仍需 pilot 证"经典在自适应 EW 下真败";
- 引用锚:CRL2RT(2502.10429,时间交错)、CIRL(2024,混合>纯RL/纯经典)、Constrained DRL for Cognitive Radar RM(2606.05526,QoS-RRM)。
