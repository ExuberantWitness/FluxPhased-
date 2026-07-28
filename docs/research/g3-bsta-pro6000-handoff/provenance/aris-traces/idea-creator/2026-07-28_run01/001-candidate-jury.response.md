# Independent Idea Jury Decision

## Verdict

**唯一推荐核心候选：A，Budgeted Sparse Beam–Target Allocation。**

但当前行动不是直接训练 PPO：

> **候选 E 必须作为强制 Phase 0。只有恢复 provenance、修正 gate，并证明 A 的 same-observation causal planner 存在保守 headroom 后，才允许进入 PPO。**

原 G2'a 继续记为 `FAIL / unverified reachability`。A 必须作为新环境版本、新 gate，不能回写成原 gate PASS。

## Score Table

所有分数均为 1–5，越高越好；最后一列是抗制造 headroom 能力，5 代表风险最低。

| 候选 | 物理可辩护 | 非平凡因果决策 | 脚本公平性 | 训练前可证伪 | 实施范围可控 | 文献相对新颖性 | 抗制造 headroom |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A** | **5** | **5** | **5** | **5** | 3 | 2 | **4** |
| B | 4 | 2 | 5 | 5 | **5** | 1 | 2 |
| C | 3 | 5 | 5 | 4 | 2 | 2 | 2 |
| D | 2 | 4 | 4 | 4 | 1 | 3 | 1 |
| E | 3 | 1 | 5 | 5 | 5 | 3 | 5 |

E 不引入物理机制，其物理分是中性，而不是“当前物理已验证”。

## Strongest Argument For/Against Each Candidate

- **A**
  - 支持：硬峰值功率、能量和单波束约束直接消除 all-target/all-time 支配动作，并产生目标、时间和资源之间的真实机会成本。
  - 反对：competent marginal-information-per-joule 或 knapsack 脚本可能接近 oracle；预算和目标异质性若无平台依据，也可能成为人为制造余量的旋钮。
- **B**
  - 支持：最小、最快、可用短时域枚举直接证伪。
  - 反对：若任务同质且信息增量近似可加，发射时机不重要，问题仍会退化成固定 duty cycle；能量桶容易被调到“刚好有 headroom”。
- **C**
  - 支持：延迟截获和有限频谱覆盖可形成真正的 belief-state 决策。
  - 反对：当前系统是否有可信频率状态尚未确认；hop 可预测性、感知延迟和泄漏程度都能轻易制造优势，且该方向文献拥挤。
- **D**
  - 支持：即使单目标，也能形成 burst/cooldown/track-loss 的长时程决策。
  - 反对：off-board jammer 的 exposure、反制和热参数目前缺少校准依据，且 `tau`、阈值和 countermeasure latency 等自由参数极易制造结果。
- **E**
  - 支持：直接修复 artifact 缺失、错误 comparator、错误统计原假设和单 training seed，并阻止对不可达 gate 继续耗费训练。
  - 反对：E 本身不创造 jammer 决策空间，也不能构成 learned-jammer 方法，只能是 Phase 0。

## Recommended Minimal Mechanism

A 的第一版进一步收缩为：

- 团队每步只允许 `{idle}` 或 `{选择一个可观测 target，以固定校准功率干扰}`；
- `K=1`，共享硬峰值功率和 episode energy；
- 单一 masked categorical allocation template；
- learner、script 和 planner 共用同一 observation、mask、validator、JNR 与能量扣减路径；
- observation 仅含 remaining energy、time、延迟/带噪 emission、bearing/target proxy 和任务紧迫度 proxy，不含真实 queue、progress 或未来到达；
- JNR 只影响所选目标，并通过 IQ 校准的 detection/Fisher-information update 进入任务完成逻辑；
- 主模型中删除任意 0.1 progress floor。

第一版不加入连续功率。固定功率下的 target–time–energy 选择已经足以检验 A 的核心假设。

## Excluded Add-ons

- learner-only active-cost；
- frequency hopping、频谱预测和 retune POMDP；
- exposure、home-on-jam、nulling、thermal/cooldown；
- 连续功率控制和多波束；
- co-learning radar；
- clairvoyant oracle 作为可部署 headroom 证据；
- 后验 parity gate；
- legacy sqrt 作为主结果；仅允许回归/消融。

## Mandatory Pre-PPO Gates and Exact Falsifiers

### E0：provenance

必须恢复并版本化 metric 定义、源代码、gate scripts、raw episode records、所有 seed 类型、config 和 checkpoint hash。任一关键项缺失：`BLOCK_PPO`。

### E1：原环境可达性

使用冻结 best scripted 和 admissible current-action bound：

\[
\operatorname{UCB}_{95}
(D_{\text{upper bound}}-D_{\text{best script}})<0.05
\]

时，结论为 `STOP_CURRENT_G2A_INFEASIBLE`。若 bound 不紧，只能记 `INCONCLUSIVE`，不能声称结构性不可达。

### A0：物理与守恒

- `Pmax/E/K` 来自平台或预注册 scenario，而非 test seeds；
- trajectory 能量守恒归一化误差不超过 `1e-6`；
- 预注册 IQ grid 上模型 `P_d` 与 Monte Carlo 最大绝对误差不超过 `0.03`；
- 任一 J 增加导致信息增量反向改善，或出现人为硬 floor：拒绝 A。

### A1：公平动作路径

相同 state/action replay 下，learner、script、planner 的 executed action、能耗和 JNR 必须逐项一致。任一 policy-type 特判、约束绕过或 privileged observation：实验无效。

### A2：因果决策非平凡

- reduced exact DP 中必须存在至少一对相同 `(time, energy)`、但因果 observation history 不同的可达状态，其唯一最优 target action 不同；
- causal oracle 相对最佳 open-loop schedule 的 raw-drop paired one-sided 95% LCB 大于 `0.02`；
- 任一条件不满足：缺乏足够状态依赖，不启动 PPO。

### A3：competent scripts 冻结

至少包括 budgeted round-robin、periodic blink、EDF/threat-first、reactive follower、marginal-information-per-joule/assignment。只允许 calibration split 调参，随后冻结。使用 test seeds 选择脚本或预算：整项 headroom 证据作废。

### A4：headroom witness

只能使用 same-observation causal policy witness；clairvoyant bound 不计：

\[
\operatorname{LCB}_{95}\left(
D_{\text{causal witness}}-\max_jD_{\text{script},j}
\right)>0.075.
\]

未达到 7.5pp：不启动 PPO。

### A5：抗调参脆弱性

在预注册相邻 energy settings 和 detector calibration 的两端模型上，headroom LCB 均仍大于 `0.05`。只有单一“甜点”配置通过：判为 manufactured headroom，拒绝该 operating point。

训练后的独立 falsifier：

- trained policy 未显著超过 frozen-init、iid random 和 shuffled-observation control；或
- 至少 8 个独立 training seeds 下，对 best script 的一侧 95% LCB 不大于 `0.05`。

任一发生都禁止 superiority claim。

## Engineering vs Publication Novelty

- **工程上**：A 是最合理选择，因为它直接修复 dominant action 和缺少机会成本；E 是不可省略的入口门。
- **发表上**：target/beam/power resource allocation 与 PPO 本身不新颖。潜在贡献只能来自 IQ 校准的 task-specific progress、统一硬可行域、oracle-first reachability gate，以及证明策略真实使用状态的控制实验。

## Confidence and Missing Evidence

- 工程选择 A、E 必须 Phase 0：`0.86`；
- A 实际产生至少 5pp headroom：`0.35`；
- 独立算法/方法新颖性：`0.25`。

未决证据：M7 源码和 raw records、真实平台功率/能量/波束限制、target urgency 是否因果可观测、IQ detector calibration，以及 competent index baseline 是否吃掉全部余量。当前没有任何实验成功主张。