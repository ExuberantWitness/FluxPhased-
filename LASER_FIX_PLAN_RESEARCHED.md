# 激光训练零击杀 — 文献支撑的修复方案

**日期:** 2026-06-15
**方法:** research-pipeline Stage 1(文献调研)→ 在 Gate 1 产出方案,**未进入 GPU 实验阶段**(本机 24G 不可跑,训练需 98G)
**代码基线:** origin/main @ 41105f0
**前置分析:** `LASER_ROOT_CAUSE_ANALYSIS.md`

---

## 0. 文献交叉确认的诊断

四个独立方向的文献都指向同一套机制,且与代码逐行分析吻合:

1. **把 capped (1/r²)·t⁴ 直接当奖励,是 Ng et al. 1999 明确警告的反模式。** 状态量直接作奖励既会饱和(95m 内零梯度),又不保证 policy-invariant。[Ng 1999]
2. **平台区 = PPO 梯度信噪比崩溃。** 策略梯度的"信号"项随奖励方差消失而消失,但"噪声/熵"项不消失 → 在 BC 最优点附近 PPO 净贡献是漂移而非细化。这正是 best eval 从 ~0 漂到 151m 的命名机制。[Razin et al. 2023, ICLR'24 "Vanishing Gradients in RL Finetuning";Bolland et al. 2024]
3. **开火位拿不到梯度的根因是"阈值化在分布之外"。** `tanh(action[0])>0.5` 是分布外的不可导阈值,二元开火事件没有自己的 log-prob,PPO 无法对它分配 advantage。[MP-DQN 诊断;PPO score-function 机制]
4. **稠密塑形奖励会让策略"过拟合塑形项"而非真目标。** 极简 min-time 稀疏奖励反而在精度上胜过稠密塑形。[Vasan et al. 2024 "Revisiting Sparse Rewards", arXiv:2407.00324]

---

## 1. 收敛的修复方案(五个组件,纯 RL、无钉死动作、最小改动)

五个组件互相正交、可叠加,共同构成一条"真目标(+100)+ 良态稠密引导 + 可学开火 + 保护 BC 精度 + 课程让目标可达"的闭环。

### 组件 A — 奖励曲面:换成势能塑形(去死区,且不改最优策略)

**机制(文献):** 用势能塑形 `F = γ·Φ(s') − Φ(s)`,取 **Φ = −log r**(尺度不变,梯度 ∝ 1/r 在近目标处变强,跨 3000m→0.2m 四个数量级都有梯度,不饱和)。Ng et al. 1999 证明该形式**不改变最优策略**,只改善梯度条件;动态势能(Devlin & Kudenko 2012)允许训练中安全地锐化 Φ。[Ng 1999;Devlin&Kudenko 2012, AAMAS;Harutyunyan 2015, AAAI]

**对应代码改动**(`train_laser.py:349-360`,替换 spatial 段):
```python
# 旧:spatial = clamp((r_ref/max(r,r_floor))², ≤cap)  → 95m 内零梯度
# 新:对数距离势能塑形(尺度不变,全程有梯度)
phi      = -torch.log(min_dist.clamp(min=self.beam_r_floor_m))          # Φ(s')
phi_prev = -torch.log(self._prev_min_dist[:, t].clamp(min=self.beam_r_floor_m))
shaping  = self.gamma * phi - phi_prev                                  # F = γΦ(s')−Φ(s)
self._prev_min_dist[:, t] = min_dist
beam_reward = shaping * self.beam_reward_weight
```
保留真目标 `+100 kill_bonus`(`vec_battlefield.py:288`)作为唯一未塑形项;PBRS 只调节路径。
**替代(更省事)**:若不想引入 Φ(s')−Φ(s) 的跨步状态,直接把 spatial 换成 `1 − tanh(r/σ)` 或 `−exp(−r/σ)`——但单个 σ 跨不了四个数量级,需分段或用 log-distance,故**首选 PBRS log-distance**。[reaching 经典奖励 `1−tanh(10·d)`, Neurocomputing 2025]

### 组件 B — 开火位:独立 Bernoulli 头 + 奖励对开火门控(可学,不钉死)

**机制(文献):** 把开火从 Gaussian 里拆出,做成独立的 **Bernoulli logit 头**;联合 log-prob = `log Bernoulli(fire) + log Gaussian(aim)`,共用同一 advantage 与 PPO clip。这样开火是一个**真正的采样事件**,有自己的 log-prob,PPO 能对它分配 advantage。再让**奖励只在开火且瞄准好时兑现**(开火-未命中给小惩罚),则"瞄准好就开火"有正 advantage、"乱开火"有负 advantage——开火位纯靠奖励梯度学出来,无任何强制。[H-PPO, Fan et al. IJCAI 2019, arXiv:1903.01344;TAAC, Yu et al. NeurIPS 2021;OpenDILab PPOxFamily hybrid recipe]

**为什么不用 Gumbel-softmax:** PPO 是 score-function 方法,不需要对离散动作做可导松弛;纯 Bernoulli 头更简单且更正确。Gumbel-ST 留给 SAC/DDPG 类重参数化骨干。[Jang et al. 2017]

**对应代码改动:**
- `actor_critic.py` `CommanderActorCritic`:新增 `self.fire_head = nn.Linear(hidden, 1)`;`get_action`/`evaluate_actions` 里 aim 用现有 4 维 tanh-Gaussian、fire 用 `Bernoulli(logits=fire_head(features))`;log-prob 相加。
- `_get_rewards`:奖励乘开火门控(用 commander 自己的决策,不掺 radar_valid):
```python
fire = self.env.battlefield.drone._commander_fire[:, t].float()
beam_reward = shaping * self.beam_reward_weight + fire * illum_term      # 兑现需开火
beam_reward += fire * (min_dist > self.beam_r_floor_m).float() * fire_miss_penalty  # 开火-未命中小惩罚
```
- fire Bernoulli 头加一个小熵奖励,防过早塌缩到"永不开火"。
- **撤掉 eval 的强制开火拐杖** `_eval_nn_step:719 c_action[:,0]=1.0`(开火能学出来后,eval 应反映真实策略)。

### 组件 C — kill_radius 课程:按成功率门控退火(让真目标可达)

**机制(文献):** 把击杀半径从大退到小,但**不按固定时钟,按成功率门控**:维持在成功率约 50%(梯度信号最大)的半径,滚动成功率越过 ~0.7 才收紧,跌向 0 则回放大(防退火过快塌缩)。标量单参数用手写的成功率门控调度即可;要自动化用 ALP-GMM。[GOID 中等难度区间, Goal-GAN, Florensa et al. ICML 2018;PCCL 连续退火, Luo et al. IJCNN 2020, arXiv:2002.02697;ALP-GMM, Portelas et al. CoRL 2019;退火速度的 KL 上界, Self-Paced RL, Klink et al. NeurIPS 2020]

**对应代码改动**(`train_laser.py` PSRO 主循环,注意对象是 **`battlefield.laser.kill_radius_m`** 不是 drone):
```python
# 每个 PSRO 迭代结束:按上轮 eval 成功率门控收紧
if eval_success_rate > 0.7:
    new_kr = max(0.2, env.battlefield.laser.kill_radius_m * 0.7)   # 收紧
elif eval_success_rate < 0.2:
    new_kr = env.battlefield.laser.kill_radius_m * 1.3             # 放大(安全阀)
else:
    new_kr = env.battlefield.laser.kill_radius_m                  # 保持(~50%区间)
env.battlefield.laser.kill_radius_m = new_kr
```
课程同时收紧"组件 B 的 illum_term 兑现半径",使稠密照射奖励逐步要求更高精度。当前无此基础设施(`:50` 构造时一次性读),需新增上述十几行。

### 组件 D — 保护 BC 精度:残差 RL 或 KL 锚定(防 PPO 拖垮 BC)

**机制(文献):** 本任务 obs 里有敌人真值坐标、BC 直接监督抄它,几乎是监督指向;问题是平台区的 PPO 噪声把 BC 均值推离。两条文献支撑的修法:
- **残差 RL(最强结构性修复):** 冻结 BC 瞄准头,PPO 只训练一个**零初始化、带界**的小残差 `aim = aim_BC + aim_res`,漂移在架构上不可能超过残差界。ResiP 在精密装配上 50 演示→98%、且**胜过直接 RL 微调**。[ResiP, Ankile et al. 2024, arXiv:2407.16677;Residual Policy Learning, Silver et al. 2018;Johannink et al. ICRA 2019]
- **DAPG/KL 锚定(改动更小):** 把现有固定权重 BC-MSE 换成**随 advantage 尺度缩放、并按迭代衰减(λ₁^k)**的 demo 梯度,或加 `β·KL(π‖π_BC)` 信任域并随 aim 误差下降而收紧 β——使早期 BC 不被噪声 advantage 盖过,精度达标后再松。[DAPG, Rajeswaran et al. RSS 2018, arXiv:1709.10087;ABM, Siegel et al. ICLR 2020;AWAC, Nair et al. 2020]
- **廉价缓解(与上面叠加):** 收敛附近退火/关闭熵奖励;按 reward std 缩放 PPO 学习率(平台区贡献≈0 更新)。[Razin et al. 2023]

**建议:** 先上"DAPG 式 advantage 缩放 + 衰减 + 收敛附近退熵"(改动小);若仍漂移,再上残差头(改动较大但根治)。

### 组件 E — 对照基线(验证塑形确实是元凶)

跑一个**极简 min-time 稀疏奖励**(每步 −1,0.2m 内终止)做对照。若它追平或胜过塑形版,即直接证明 (1/r²) 塑形在伤害精度。[Vasan et al. 2024, arXiv:2407.00324]。HER 不适用于纯 on-policy PPO(需 replay buffer);其 on-policy 类比是 Hindsight Policy Gradients(需 goal-conditioned 策略),本任务单目标不必引入。[HER, Andrychowicz et al. NeurIPS 2017;HPG, Rauber et al. ICLR 2019]

---

## 2. 落地顺序(按"改动最小 × 杠杆最大")

| 层级 | 组件 | 改动量 | 说明 |
|---|---|---|---|
| **T0 必做** | A 奖励曲面 PBRS log-distance | ~8 行 | 去死区,根治。单独做即可让 eval min_aim_dist 破 10m |
| **T0 必做** | B Bernoulli 开火头 + 奖励门控 | ~25 行 | 开火可学;撤 eval 强制开火 |
| **T1 强烈建议** | C kill_radius 成功率门控课程 | ~15 行 | 让 +100 训练期可达 |
| **T1 强烈建议** | D BC 保护(先 DAPG 缩放+衰减+退熵) | ~10 行 | 防 PPO 拖垮 BC 精度 |
| **T2 验证** | E min-time 稀疏对照基线 | 1 个配置 | 证伪/证实塑形假设 |

A+B 是最小可行集(去死区 + 可学开火);C+D 提升收敛与稳定;E 是对照。**A 必须先于其它**——曲面不修,开火学对也白搭(瞄准仍卡 95m)。

---

## 3. 验证计划(本机做代码,训练机做训练)

**本机(24G,无 GPU 训练):**
- 代码静态验证:配置解析、import 冒烟;
- 小张量手验:新 PBRS 奖励在 r=3000→0.2m 单调、梯度处处非零;Bernoulli 头 log-prob/advantage 维度对齐。

**训练机(98G):** 按梯度判据逐级确认
1. eval `min_aim_dist` 先破 **10m**(证明梯度延伸进原死区);
2. 再破 **0.2m** 且出现 `kill_bonus` 样本(端到端闭环);
3. 训练期 `kills>0`(证明 Bernoulli 开火头 + 课程生效,非靠 eval 强制);
4. 对照:min-time 稀疏基线 vs 塑形版(组件 E)。

**成功判据:** eval `min_aim_dist < 0.2m` 且训练 `kills > 0`。

---

## 4. 引用清单(已核验;不确定项已标注)

**势能塑形 / 奖励形式**
- Ng, Harada, Russell. Policy Invariance Under Reward Transformations. ICML 1999.
- Devlin & Kudenko. Dynamic Potential-Based Reward Shaping. AAMAS 2012.
- Harutyunyan et al. Expressing Arbitrary Reward Functions as Potential-Based Advice. AAAI 2015.
- Wiewiora et al. Principled Methods for Advising RL Agents. ICML 2003.
- Berducci et al. HPRS: Hierarchical Potential-Based Reward Shaping. Front. Robot. AI 2024.
- Bolland et al. Behind the Myth of Exploration in Policy Gradients. 2024. arXiv:2402.00162.
- Vasan et al. Revisiting Sparse Rewards for Goal-Reaching RL. 2024. arXiv:2407.00324.

**课程 / 目标容差**
- Luo et al. PCCL: Continuous Curriculum Learning for Reaching. IJCNN 2020. arXiv:2002.02697.
- Florensa et al. Reverse Curriculum Generation. CoRL 2017. arXiv:1707.05300.
- Florensa et al. Automatic Goal Generation (Goal-GAN). ICML 2018. arXiv:1705.06366.
- Portelas et al. ALP-GMM. CoRL 2019. arXiv:1910.07224.
- Jiang et al. Prioritized Level Replay. ICML 2021. arXiv:2010.03934.
- Andrychowicz et al. Hindsight Experience Replay. NeurIPS 2017. arXiv:1707.01495.
- Rauber et al. Hindsight Policy Gradients. ICLR 2019. arXiv:1711.06006.
- Klink et al. Self-Paced Deep RL. NeurIPS 2020.〔arXiv id 待核〕

**混合动作 / 可学触发**
- Fan et al. H-PPO: Hybrid Actor-Critic in Parameterized Action Space. IJCAI 2019. arXiv:1903.01344.
- Xiong et al. P-DQN. 2018. arXiv:1810.06394.
- Bester et al. MP-DQN. 2019. arXiv:1905.04388.
- Hausknecht & Stone. Deep RL in Parameterized Action Space. ICLR 2016. arXiv:1511.04143.
- Li et al. HyAR. ICLR 2022. arXiv:2109.05490.
- Yu et al. TAAC. NeurIPS 2021. arXiv:2104.06521.
- Jang et al. Categorical Reparameterization with Gumbel-Softmax. ICLR 2017. arXiv:1611.01144.

**BC+RL / 残差 / 防漂移**
- Rajeswaran et al. DAPG (Learning Complex Dexterous Manipulation). RSS 2018. arXiv:1709.10087.
- Ankile et al. ResiP (Imitation to Refinement). 2024. arXiv:2407.16677.
- Silver et al. Residual Policy Learning. 2018. arXiv:1812.06298.
- Johannink et al. Residual RL for Robot Control. ICRA 2019. arXiv:1812.03201.
- Siegel et al. ABM (Keep Doing What Worked). ICLR 2020. arXiv:2002.08396.
- Nair et al. AWAC. 2020. arXiv:2006.09359.
- Galashov et al. Information Asymmetry in KL-Regularized RL. ICLR 2019. arXiv:1905.01240.
- Razin et al. Vanishing Gradients in RL Finetuning. ICLR 2024. arXiv:2310.20703.

**待核(引用前需复核 id/作者):** TRPO arXiv:1502.05477、PPO arXiv:1707.06347(凭通识,本次未重新抓取);Cliff Diving arXiv:2205.07015;On Pathologies in KL-Regularized RL from Expert Demos arXiv:2212.13936;Goal-GAN 的 GOID 区间具体数值;Self-Paced RL 的 arXiv id;HTRPO arXiv:1907.12439 作者/venue。

---

## 5. Gate 1 决策

研究管线在此暂停:**已产出文献支撑、机制清晰、可直接实现的方案,不进入 Stage 3-4 的 GPU 实验**(本机不可跑,且符合用户"先不要运行"的约束)。

下一步可选:① 我在本机一个分支上落地 T0(组件 A+B)+ T1(C+D),做代码级静态验证,训练验证留给 98G 机器;② 或先只落地 T0 最小集快速验证去死区 + 可学开火。
