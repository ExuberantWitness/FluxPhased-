# FluxPhased M7 / G2'a 实验审计报告

**日期**：2026-07-28  
**项目**：FluxPhased- / MFR-IQ Phase B / M7  
**审计对象**：`G2'a Gate 结构性负结果诊断报告`  
**审计员**：GPT-5.6-Sol ultra，新鲜只读审计代理  
**独立性**：same-family；结论为 provisional  

## 总体结论：FAIL / BLOCK CLAIMS

原因码：

- `ARTIFACTS_ABSENT`
- `BASELINE_SUBSTITUTION`
- `WRONG_STATISTICAL_NULL`
- `SINGLE_TRAIN_SEED`
- `SIMULATION_ONLY`
- `STRUCTURAL_OVERCLAIM`

当前最窄且可辩护的表述是：

> 文字报告称两个 training-seed-0 PPO checkpoint 未通过原 G2'a gate；由于实现、评估脚本、原始结果和 checkpoint 均不可访问，该结果目前不能独立复现。现有摘要显示严重的指标饱和与策略不可辨识，但尚未证明“任何学习算法都不可能超过 noise 5pp”，也未证明 PPO 学会了 noise-equivalent 策略。

原 G2'a 不应在看过结果后改写成 parity gate。应保留为 **FAIL（headroom/protocol 未证实）**，另行预注册 reachability、learnability 和 parity/superiority gate。

## 报告中的原始结果

| 干扰策略 | drop_ratio（mean ± std） |
|---|---:|
| noise | 0.520 ± 0.017 |
| reactive | 0.489 ± 0.018 |
| blink | 0.450 ± 0.022 |
| learning v1 | 0.491 ± 0.018 |
| learning v2 | 0.475 ± 0.018 |

| 报告检验 | Δdrop | t | 报告 p |
|---|---:|---:|---:|
| v1 − blink | +0.041 | 16.4 | <0.0001 |
| v2 − blink | +0.026 | 13.3 | <0.0001 |

附加评估摘要：

| 策略 | greedy | sampled |
|---|---:|---:|
| v1 | 0.486 | 0.520 |
| v2 | 0.471 | 0.519 |
| noise | — | 0.520 |

以上数字仅来自文字报告，未能追溯到原始 JSON/CSV。

## 关键发现

### 1. Gate 比较对象错误

Gate 要求 learning 超过 **best scripted** 5pp；报告自己认定 best scripted 是 `noise=0.520`，但两次 t 检验都换成了最弱的 `blink=0.450`。

| 模型 | learning − noise | 达到 +5pp 所需分数 | 距 gate |
|---|---:|---:|---:|
| v1 greedy/gate 表值 | −2.9pp | 0.570 | 7.9pp |
| v2 greedy/gate 表值 | −4.5pp | 0.570 | 9.5pp |
| v1 sampled | 0.0pp | 0.570 | 5.0pp |
| v2 sampled | −0.1pp | 0.570 | 5.1pp |

因此“只差 0.9pp/2.4pp”不是原 gate 的结论。

### 2. 统计原假设错误

报告的 p 值检验近似是 \(H_0:\Delta=0\)，但 +5pp superiority gate 应检验：

\[
H_0:\Delta\le 0.05,\qquad H_1:\Delta>0.05.
\]

根据报告四舍五入后的 blink 比较可反推：

| 比较 | 均差 | 配对 SE | 配对 SD | 95% 双侧 CI | \(p_{\Delta=0}\) | \(p_{\Delta>0.05}\) |
|---|---:|---:|---:|---:|---:|---:|
| v1 − blink | 0.041 | 0.002500 | 0.007071 | [0.03509, 0.04691] | 7.64e-7 | 0.995630 |
| v2 − blink | 0.026 | 0.001955 | 0.005529 | [0.02138, 0.03062] | 3.18e-6 | 0.999997 |

即使对 blink，两组结果也在统计上排除了 +5pp，而不是接近通过。由于缺少逐 seed 配对值，learning 相对 noise 的精确配对 CI 无法恢复。

### 3. “8 seeds”不是 8 个独立训练 seed

两个 checkpoint 路径都带 `s0`；v1/v2 是不同配置的单一 training seed，不能视为训练重复。报告中的 8 seeds 看起来是固定 checkpoint 的 evaluation RNG seeds，只能描述该 checkpoint 的 rollout 变化，不能估计 PPO 的训练稳定性或算法泛化。

`32 envs × 4 episodes` 嵌套于 seed 内，也不能直接当成 1024 个独立样本。

### 4. sampled parity 不证明 PPO 学习成功

零附近 logits 的 Bernoulli 策略很可能从初始化开始就约为 \(p=0.5\)。在饱和环境中，未训练策略、固定 iid Bernoulli(0.5) 和训练后策略可能得到同一 drop_ratio。

缺少以下必要对照：

- untrained/frozen-init actor；
- 固定 Bernoulli 占空比 sweep；
- 打乱或固定 observation 的 trained actor；
- checkpoint 随训练进度的 held-out 曲线；
- action-state mutual information 或条件动作统计。

因此目前只能称为“sampled endpoint metric 与 noise 四舍五入后接近”，不能称为“PPO 学到 noise-equivalent 行为”。

### 5. 结构性不可达是合理假设，但尚未证明

若完整 MDP 同时满足：

1. jammer 动作只会非负增加 JNR；
2. 没有能量、热、暴露、反辐射或反制代价；
3. cue、admission、调度、队列和终止 drop 对 jammer 活跃度保持同一偏序；
4. all-on 脚本在每一步取该偏序最大动作；

则可用路径耦合与归纳证明 all-on 弱支配所有历史依赖策略，原 +5pp gate 数学上不可达。

报告中的 `prog_factor` 片段只证明局部 JNR→progress 单调性，未证明完整转移的偏序保持。雷达 cue/反干扰/队列反应仍可能使选择性晚时干扰优于 always-on。

另外：

\[
\max\left(0.1,\frac{1}{\sqrt{1+\mathrm{JNR}}}\right)
\]

在 JNR ≥ 99（约 20dB）时固定为 0.1，之后所有更强动作完全没有边际作用。必须先报告 JNR 分布与 `prog_factor==0.1` 的时间比例；否则该 clamp 很可能直接制造了平台区。

### 6. learner-only active cost 不能修复 raw-drop gate

仅加入：

```text
jam_rew -= alpha * active_fraction
```

不会改变 always-on noise 的物理轨迹或 raw `drop_ratio=0.520`；它只会让 learner 在奖励中牺牲 raw drop 换能耗。若 gate 仍以 raw drop 比较，headroom 不会因此出现。

有效改造必须二选一：

- 对所有 learned/scripted 策略施加相同的硬能量、热或同时波束约束；
- 将 gate 改为所有策略一致计算的成本调整 utility / Pareto 指标。

## 完整性检查

### A. 指标/ground-truth provenance：FAIL

找不到 `drop_ratio` 的分子、分母、reset、聚合和 seed 数组。评估类型只能归类为 `simulation_only`。

### B. score normalization：FAIL（不可审计）

文字中未发现显式自归一化，但实现缺失，无法排除。后验加入的 σ-progress coupling 也没有提供物理校准或 held-out calibration。

### C. 结果文件存在性：FAIL

下列 13 个关键目标均不存在于给定路径、当前本地 refs、不可达 Git 对象或远端最新公开分支：

- 四个 `/tmp` 评估脚本；
- 四个 MFR 实现/训练文件；
- `g2a_summary.json`；
- 两个 checkpoint；
- 两个 `train_curve.csv`。

本地 checkout 为 `fa485ad4`；公开最新 `twoteam/bc-ppo` 快照为 `80769974`，二者均无 MFR/G2a 路径。

### D. dead code / call path：FAIL

没有可达实现，无法确认报告中的函数是否执行、metric 是否写入结果文件。

### E. scope：FAIL

一个 training seed、两个后验配置、8 个评估 RNG seed，不支持算法稳定性或 universal structural claim。

### F. evaluation type：PASS（分类）

`simulation_only`；所有性能主张都必须带 “in this simulator/configuration” 限定。

## 内部矛盾

- 报告写 `0.491 < 0.489`，算术错误；v1 点估计比 reactive 高 0.2pp，显著性未知。
- 前文称两个 greedy 都为 0.484，后文又给出 0.486/0.471，未解释差异。
- v2 sampled 为 0.519，不是与 0.520 “精确相等”。
- “near-optimal”不足以推出“所有策略都不能多 5pp”；需要可验证上界。
- sampled 与 greedy 是两个不同部署策略，不应在看过结果后选择更有利者。

## 建议的确认性实验

### D0：先恢复 provenance

将所有脚本从 `/tmp` 移入版本控制；保存：

- commit SHA、完整 config 和 reproduction command；
- raw per-episode/per-environment/per-seed counts；
- training seed、environment seed、action-sampling seed；
- checkpoint SHA-256；
- metric 定义、调用路径和 aggregation test。

### D1：零训练成本的饱和/学习诊断

| 维度 | 条件 |
|---|---|
| jammer power | 30W、100W |
| policy | always-off、iid Bernoulli p=0:0.1:1、always-on、untrained actor、trained sampled、trained shuffled-observation |
| replication | 16 个 held-out environment seeds；随机策略每个环境 seed 4 个 action seeds |
| 指标 | drop_ratio、active fraction、any-on fraction、JNR quantiles、`prog_factor==0.1` 比例、cue-admission rate、queue occupancy |

若 untrained≈trained≈Bernoulli(0.5)≈always-on，且打乱 observation 不降低性能，才支持“无状态学习 + 指标饱和”。

### D2：reachability 上界

- 在简化四动作模型中用 exact DP；
- 在完整环境中用 MCTS/CEM/有限状态控制器搜索；
- 用 common exogenous randomness 检查 all-on 对可达状态/轨迹的偏序支配；
- 正式不可达条件应为：

\[
\mathrm{UCB}_{95\%}(U_{\mathrm{oracle}}-B_{\mathrm{best\ script}})<0.05.
\]

### D3：最小物理重构

优先级建议：

1. **硬共享能量/热预算 + 同时活跃上限**，对所有策略一致；
2. **目标/波束选择 + 异质任务 deadline/value**，形成真实资源分配；
3. 平滑且经 IQ detector/tracker 校准的 information/Pd/covariance 更新，移除无依据的统一硬地板；
4. 频率敏捷只在动作包含 channel/bandwidth/power allocation、固定总 EIRP 且 observation 因果可获得时加入。

单纯 reward active-cost 不推荐作为 raw-drop gate 修复。

## 建议的新 gate

### G2'a-0：headroom feasibility（训练前）

先冻结 scripted family，在独立 calibration seeds 上选定 best script。要求 oracle 相对 best script 的保守 headroom 至少 7–10pp；否则 +5pp learner gate 标为 `INAPPLICABLE_NO_HEADROOM`。

### G2'a-1：learnability

trained policy 必须显著超过 frozen-init、iid Bernoulli 和 shuffled-observation control；否则标为 `NO_LEARNING_SIGNAL`。

### G2'a-2：确认性 superiority

使用至少 8 个独立 training seeds；每个 checkpoint 在相同 held-out environment seeds 上评估，sampled policy 另设独立 action seeds。预先固定 sampled 或 deterministic 语义。

\[
\mathrm{PASS}\iff
\min_j\left[
\bar d_j-t_{0.95,K-1}\frac{s_{d,j}}{\sqrt K}
\right] > 0.05,
\]

其中 \(d_j=D_{\mathrm{learner}}-D_{\mathrm{script}\ j}\)，且所有预注册脚本均须通过。

若工程目标只是 parity，应另建 non-inferiority/TOST gate，并预先规定容忍度 \(\epsilon\)；不能以“不显著不同”宣称等价。

## 四个决策点

1. **Gate**：不要后验把原 gate 改为 parity。保留 FAIL，并新增 headroom gate。
2. **物理改造**：选择“硬共享资源约束 + 稀疏目标分配”；active-cost-only 无效。
3. **M8**：可作为独立 radar robustness 轨继续，但当前 sampled PPO 在证明超过 untrained iid 前只能标为 randomized/noise-equivalent-at-metric baseline，不应宣传为 adaptive learned jammer。
4. **发表价值**：目前适合作为 benchmark-design diagnostic/ablation，不足以成为 Phase B 核心 finding。若补齐形式化支配证明、饱和相图、多算法/多 seed 控制和完整 artifact，可上升为“cost-free monotone jammer 的 dominant-action trap”方法学负结果。

## Claim impact

| Claim | 审计影响 |
|---|---|
| 两个 checkpoint 未通过 G2'a | 与摘要方向一致，但不可复现 |
| PPO 学会 noise-equivalent 策略 | 不支持 |
| 任意 learning jammer 均不可能超过 noise 5pp | 不支持，需上界/支配证明 |
| p 值验证了 gate | 错误；比较对象和原假设均不对 |
| negative result 已可发表 | 否 |

## 审计追踪

- 输入报告 SHA-256：`0c381b0ca6c9d29dda1e258b11b6eadd80a5139a17a8a62a1a7559b89c7fdad4`
- 本地目标仓库 HEAD：`fa485ad4cf6314df8a747498f4179d702a7c4923`
- 公开最新目标分支快照：`80769974cb41fd86e2f80bc2a8992955fb228058`
- Trace：`.aris/traces/experiment-audit/2026-07-28_run01/`

