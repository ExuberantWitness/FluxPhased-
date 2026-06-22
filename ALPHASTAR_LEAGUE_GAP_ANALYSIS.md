# FluxLeague vs AlphaStar 联赛机制 — 差距分析与修改建议

**目的**: 你想实现 AlphaStar 式联赛但"效果非常差"。本文对照 AlphaStar 的**实测实现模式**逐条找出
FluxLeague 的偏离,并给出修改建议。结论:**结构对了,但 6 个实现细节偏离了 AlphaStar 的关键设计,
而且全部建立在一个被 NaN/0.50 bug 污染的胜率信号之上——必须先修信号,联赛机制才可能 work。**

---

## 1. AlphaStar 联赛的标准实现(联网核实)

**三类 agent(每个种族/队各一套)**:
| 角色 | 训练对手 | 入池条件 | 重置 |
|---|---|---|---|
| **Main Agent (MA)** | **0.5 自博弈 + 0.35 PFSP(f_hard)对全联赛 + 0.15 对"遗忘的"对手** | 定期(~2e9 步)快照入池 | **从不重置**(连续学习的主干) |
| **Main Exploiter (ME)** | **只打当前 MA** | 胜 MA >70% 或步数上限 | 入池后**重置到 supervised 模型** |
| **League Exploiter (LE)** | **PFSP(f_hard)对整个联赛历史** | 胜全联赛 >70% 或步数上限 | 入池后**以 0.25 概率重置到 supervised** |

**PFSP 优先采样**(核心):对手按权重 ∝ 优先函数采样,
- `f_hard(x) = (1 − x)^p`(x = 自己对该对手的胜率;**胜率越低=越难=权重越高**,p≈1~2);
- `f_var(x) = x(1 − x)`(偏好**均势**对手,用于"太难打"时退而求其次)。
**直接按 f 归一化采样,不是 softmax。**

**两点最关键的设计哲学**:
1. **PFSP(胜率加权)是训练 matchmaking 的核心,Nash 只用于最终评估** —— PFSP 在任何规模都鲁棒;
   Nash 需要良态、稠密的 payoff 矩阵,小规模/退化时塌缩。
2. **Main Agent 永不重置 + 50% 自博弈** —— 自博弈给"始终在自己水平上的稳定学习信号",是防崩主干。

---

## 2. FluxLeague 现状对照(逐条)

| AlphaStar 设计 | FluxLeague 现状 | 差距 |
|---|---|---|
| MA = 0.5 SP + 0.5 PFSP | `ROLE_MAIN` 用 `_sample_opponents`(纯 PFSP/Elo,**无自博弈**) | **缺 0.5 自博弈**(防崩主干没了) |
| 每周期对**一群**对手训练 | `_train_against(…, opp_id, …)` + `n_samples=1`(**每轮只打 1 个对手**) | **单对手训练** → 高方差、过拟合、循环遗忘 |
| 重置到 **supervised(强)模型** | `_maybe_reset` 载入 **parent checkpoint(弱)** | **重置锚点弱**(无人类数据,parent 也菜)→ exploiter 起不来 |
| PFSP ∝ (1−x)^p 直接归一 | `sample_pfsp`: `softmax(loss_rate / temperature)` | softmax 抹平差异;且 **win_rate 默认 0.5 → loss_rate 全 0.5 → 退化成 uniform** |
| Nash 仅用于最终评估 | Nash/TC-DAMS 驱动 `meta_strategies` + 诊断 | 小规模 Nash 塌缩(sigma=[1,0,0])加噪 |
| 12 agents × 数周 × TPU | 极小规模(few iters/games,小池) | 规模差几个数量级(机制可学,期望要现实) |

---

## 3. 为什么"效果非常差" —— 根因排序

1. **【信号被污染,最致命】** PFSP / exploiter 优先级 / Nash **全部依赖胜率/payoff 矩阵**。但该矩阵正被
   两个 bug 压成退化:**win_rate=0.50 bug**(`LASER_LEAGUE_NAN_FULL_ANALYSIS.md §4`)+ **NaN 崩溃**。
   矩阵全 0.5 → loss_rate 全 0.5 → **PFSP 退化成 uniform → 联赛等于随机对打,毫无 AlphaStar 效果**。
   **先修这个,其它才有意义。**
2. **缺 0.5 自博弈** → MA 没有稳定的同水平学习信号,纯 PFSP 打退化池 = 弱信号 → 学不动。
3. **单对手训练(n_samples=1)** → 每轮只对一个对手优化 → 过拟合它、下轮换对手就遗忘 → 策略在池里循环、不进步。
4. **重置锚点弱** → exploiter 重置到弱 parent,找不到强 exploit,无法逼 MA 补漏。
5. **softmax PFSP + 默认 0.5** → 即便有信号也被抹平。
6. **小规模 + Nash 加噪** → meta-strategy 不稳。

---

## 4. 修改建议(按优先级,对照 AlphaStar)

### R0(前置,必须先做)— 修复信号源
先落地 NaN 的 **F1**(env/buffer 动作解耦)+ sensing 的 **clamp-to-±1 → 软 clamp/±(1−ε)**,让对局
不再全 0.5、不再 NaN。**胜率矩阵有方差后,下面的 PFSP/exploiter 才有意义。** 这是 hard gate。

### R1 — 给 Main Agent 加 0.5 自博弈(单点最高回报)
`ROLE_MAIN` 的对手采样改为:**以 0.5 概率 = 自己(当前 MA);否则 PFSP(f_hard)对全池**。
镜像 AlphaStar 的 MA 分布(可进一步拆成 0.5 SP / 0.35 PFSP / 0.15 遗忘对手)。位置:`flux_league.py` 第 ~377 行
`if record.role == ROLE_MAIN:` 分支。

### R2 — 每周期对"一群"对手训练,而非 1 个
把"每轮训练"从"对 1 个采样对手跑一次 `_train_against`"改为:**采样 K(如 4~8)个 PFSP 对手,在一个训练
周期内对这批对手轮流/混合采集 rollout 再更新**。消除单对手过拟合与循环遗忘。位置:`_train_against` 调用处 +
`n_samples`。

### R3 — 修 PFSP 为真正的 AlphaStar 形式
`sample_pfsp`(opponent_pool.py:94)改为 **概率 ∝ f_hard(x)=(1−x)^p**(直接归一,**去掉 softmax**),
p 取 1~2;"太难打"(对手胜率>某阈值)时混入 `f_var(x)=x(1−x)`。并**保证 win_rate 充分估计**(加 eval 局数,
未知时别默认 0.5 而是标记为"未评估、优先评估")。

### R4 — 修 exploiter 重置锚点
无监督数据时,把 `_maybe_reset` 的目标从 **parent** 改为 **当前最强 MA 的快照**(或维护一个"历史最强"锚点)。
让 exploiter 每次从强点重启去找新 exploit —— 这是 AlphaStar exploiter 有效的关键。

### R5 — Nash/TC-DAMS 降级为"仅评估"
训练 matchmaking **只用 PFSP**(鲁棒);Nash/NashConv/effK **只在最终报告/诊断**算,不驱动训练对手选择。
这样小规模下 Nash 的塌缩不再污染训练。(论文里 TC-DAMS 可作为"评估期多样性度量"的贡献,而非训练驱动。)

### R6 — 现实化规模与期望
AlphaStar 是 12 agents×数周×TPU。本项目做不到。务实配置:每队 1 MA(永不重置)+ 1 ME + 1~2 LE;
eval 局数拉到能稳定估胜率(≥20~50);周期内多对手多局。**期望**:得到"对多样对手鲁棒"的策略,
**不是**超人棋力 —— 论文也应据实陈述,别过度宣称(见 EAAI 计划的 claim 纪律)。

---

## 5. 推荐落地顺序
**R0(修信号)→ R1(0.5 自博弈)→ R2(多对手训练)→ R3(真 PFSP)→ R4(强重置锚点)→ R5(Nash 降级)。**
R0 不过,R1–R5 全是在随机信号上做文章,白费。每步后看 win_rate 矩阵是否出现方差、effK 是否 >1、
策略池是否单调变强(用 held-out exploitability 衡量,而非自评胜率)。

---

## Sources
- [Grandmaster level in StarCraft II (AlphaStar, Nature/DeepMind)](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
- [TStarBot-X: Efficient League Training in StarCraft II](https://arxiv.org/pdf/2011.13729)
- [SCC: Efficient DRL Agent Mastering StarCraft II](https://arxiv.org/pdf/2012.13169)
- [A Survey on Self-play Methods in RL](https://arxiv.org/pdf/2408.01072)
- [A Robust and Opponent-Aware League Training Method for StarCraft II (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf)
- [mini-AlphaStar](https://arxiv.org/pdf/2104.06890)
