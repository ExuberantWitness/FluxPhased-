# TAES 主线执行计划 — 自适应 EW 下的认知雷达-激光联合交战:学习式指挥官 vs 强经典/博弈论基线(给 PRO6000 agent)

**目标期刊**:IEEE TAES(甜点,IF~5,Q1,radar 圈硬通货);兜底 IET Radar Sonar & Navigation(高概率)。
**系统范畴(不得脱离)**:双相控阵雷达 + 指挥官 + 激光 DEW。多基地 Kalman 融合传感 → 指挥官分配 25 子阵(侦/跟/干/通)+ 指挥激光 → 激光对目标积能出 kill;敌方自适应干扰 + home-on-jam 暴露。
**已发表坐标(证明这是高概率、非梭哈)**:Dolinger 2025 IET(多智能体 RL vs 启发式+3 博弈论策略,高保真 GLRT sim)· Li et al. 2022 TAES(NFSP 雷达抗干扰博弈,exploitability/Nash)· Qin 2023 IET(LSTM-TD3 认知雷达 RRM POMDP)· Xiong 2023 TAES(coalition-game MARL 雷达网 MTT + 收敛证明)。**我们的 infra 比这四篇都强(IQ 级 + MATLAB 校验 + 多基地 CRLB + 激光 DEW),是超配不是够不着。**

---

## 三层贡献(对齐 TAES 口味)
| | 贡献 | TAES 认的点 |
|---|---|---|
| **C1** | 高保真 IQ 级 双雷达+指挥官+激光 DEW + 自适应 EW **闭环交战 testbed**(MATLAB 83/83 校验,多基地 CRLB) | 高保真仿真 + benchmarking |
| **C2(headline)** | **自适应 EW 下雷达-DEW 联合交战的学习式指挥官**——**激光驻留-kill-链耦合**下打赢强模块化经典 + 博弈论基线 | 新问题 + 强 baseline + 实证 |
| **C3(严谨)** | 博弈论刻画(**exploitability/best-response**,对标 Li'22)+ 操作包线相变 + **CRLB/PCRLB** 锚 | 博弈论 + CRLB(TAES 母语)+ 统计 |

**差异化钩子(把 IET 抬到 TAES)**:**激光 DEW + 认知雷达 + 自适应 EW 的联合交战 + 驻留-kill-链耦合**——文献几乎没有;RRM 论文只优化 track/detection,没人做感知-资源-DEW火控在自适应干扰下的联合调度。

---

## 博弈论形式化(C3 的骨架,TAES 差异化关键)
指挥官(雷达+激光)与干扰机是**部分可观测、近零和**博弈(零和量 = kill-under-jamming 或 −survival)。
- **exploitability(π) = U(π vs 静态干扰) − U(π vs best-response 干扰机 BR(π))** = 策略被最优对手拉下的幅度;
- 固定/规则策略 = 纯策略 → 高 exploitability;均衡(自适应)策略 → 低 exploitability;
- **对每个指挥官(经典模块化 / 博弈论经典 / 学习式)都算 exploitability + 头对头 kill/survival**——这是超越"赢一次固定经典"、达到 Li'22 严谨度的核心。

---

## 已核实代码触点(别改错)
- 传感 `algo/_shared/laser/sensing.py`:`KalmanTracker`(`_trk_P`,`trace_P`),`fused_sensing(track=True)`,`jam_mul`;
- 指挥官 `CommanderActorCritic`(obs 76,`commander_privileged_dim:10` CTDE,`task_head` 四功能);
- 激光/奖励 `algo/_shared/laser/reward.py`:kill 半径 `kr`、`jam_level`/`jam_cost`、`exposure_gain`、`race_death_penalty`;
- 对手 `env/gpu/qos_rrm/adversary.py`:`StaticJammer(L0)/ReactiveJammer(L1,τ)/LearnedJammer(L3,PPO)`;
- 经典基座 `algo/_shared/baselines/{classical_mpc,classical_qos_rrm}.py`;训练 `training.train_laser` + `SimpleMAPPOTrainer` + league;
- 配置 `configs/laser_25x25_p14.yaml`(基线)、`configs/laser_25x25_ew_exposure.yaml`(EW:jam_gain=8,exposure_gain=50)。
- **⚠️ checkpoint_dir → 持久盘(`checkpoints/taes_mainline/`),严禁 /tmp;log_std_floor=-6(勿 -4)。**

---

## WP0 — 高保真闭环 testbed(C1,~1 周)
**0.1 多目标**:`N_targets ∈ {1,2,4,8}`,CV+转弯机动,多基地融合每目标维护 track。
**0.2 激光驻留-kill-链(核心耦合机制)**:激光对目标 i **只在** track 达标(`kr_i<kr_thresh` 或 `trace_P_i<τ_track`)时**累积** kill 能量 `E_i += dwell_rate·dt`;**track 丢(σ 被干扰抬过阈)→ E_i 衰减/清零**;`E_i≥E_kill` 出 kill。→ "必须持续保 track 才出 kill"的时序耦合。
**0.3 暴露(现成)**:照射/干扰辐射 → exposure 累积 → home-on-jam(`exposure_gain`)→ 被反杀(`race_death_penalty`)。→ track 质量 vs 暴露权衡。
**0.4 自适应 EW**:干扰机 L0→L3,`jam_mul` 抬被照射目标 σ;L3 观测指挥官 task 直方图 → 干扰**当前正在 kill 的目标**(制造预判价值)。
**0.5 CRLB/PCRLB(C3 锚)**:实现多基地定位 CRLB 与跟踪 PCRLB;报告各方法 track 质量对下界的差距(经典近 CRLB 是"传感已解"的诚实证据;干扰下偏离 CRLB 是难点所在)。
**0.6 验证(必过)**:逐目标打印 track_loss/kill 链累积-清零/exposure/survival;单目标无 EW 时经典近满 kill(sanity),多目标+L3 经典明显掉(硬 regime 成立)。

## WP1 — 强基线套件(anti-strawman 命门,~1.5 周)
**决定成败。经典必须强,否则 RL 赢了不可信。三类基线:**
1. **强模块化经典指挥官**:IMM-PDAF 多目标跟踪(**Stone Soup** `dstl/Stone-Soup`)+ Q-RAM/短视界调度 + shoot-look-shoot 最优火控 + 反应式 ECCM(频率捷变+重分配)。**模块各自强、接口固定(有损)。**
2. **博弈论经典指挥官(Li'22 严谨要求)**:对模块化经典做 **fictitious-play / best-response 迭代**,逼近对当前干扰机分布的均衡纯/混策略。**这是"若固定 Nash 就赢你 RL 则故事塌"的防线——必须打赢它,不只打赢固定经典。**
3. (对照)纯 IPPO / 纯 MAPPO。
- **验证(必过)**:强模块化 + 博弈论经典在**低难度**下 kill 近满、survival 高、exploitability 低(证非稻草人);硬 regime 掉。

## WP2 — 学习式联合指挥官 + 判别 pilot + Gate(~1.5 周)
**2.1 架构(联合 CTDE 指挥官)**:扩 `CommanderActorCritic`:
- **obs**:每目标 (track 均值 + `trace_P_i` + kill 进度 `E_i`) + JSR/jam_level(分带)+ exposure 状态 + 资源占用;
- **(方法加分,可选)belief-conditioned**:把 `trace_P_i`(定位不确定度)显式喂 actor → 学"低置信收火/换被动传感/预置";CTDE critic 用 `trace_P` 反向加权 team-advantage(干扰活跃时更信 per-agent)。这从 POMDP+辐射对偶机理推出,给 TAES 一个机制新意(非 vanilla PPO);
- **action**:子阵→功能分配 + 波束指向 + 激光目标选择 + 辐射控制(降暴露);
- **reward**:kill-rate + survival(−exposure)+ −time-to-kill(`DenseRewardShaper`);
- **训练**:PPO/MAPPO(CTDE),num_envs 拉满显存,**league/PSRO 与 L3 自适应干扰机共训**(生成 best-response 对手池),log_std_floor=-6,kr 课程沿用 p14。
**2.2 难度扫描**:`N_targets × 干扰 L0→L3(+τ 连续) × exposure 强度`,每点 ≥5 seed;**报完整曲线含低难度经典够用区(诚实防 p-hack)。**
**2.3 指标**:kill-rate、time-to-kill、**survival-rate**、per-target track-loss、**exploitability/Nash-gap**、track 质量 vs CRLB;cross-play 双向平均 + 共同 held-out 干扰机。

### Gate(逐 cell + bootstrap CI)
| # | 判据 | 阈值 | 决定 |
|---|---|---|---|
| **G1(命门)** | 学习式 **> 博弈论经典**(不只固定经典)@ 硬 regime,kill-rate gap>0.10 **或** exploitability 显著更低,95%CI 不含 0 | 打赢**最强**经典 | TAES 命题成立 |
| G2(诚实) | 低难度 学习式 ≈ 经典 | 照实报 | 相变在 |
| G3(机制) | 消融 −belief/−耦合/−预判 → 丢 G1 gap | gap 消失 | 证赢来自耦合+belief,非碰巧 |
| G4(强baseline) | 模块化+博弈论经典 @ 低难度 kill 近满/exploitability 低 | 证非稻草人 | WP1 验证 |
| G5(工程使能兜底) | 联合最优的 joint-MPC/POMDP 经典在此规模**在线不可解** | amortization=实时使能 | 即便只赢速度,TAES/IET 认此工程价值 |

## WP3 — 全套(仅 G1 PASS,~3-4 周)
全难度扫描 × {学习式, 强模块化经典, 博弈论经典, IPPO/MAPPO} × ≥5 seed → **操作包线图**(kill/survival vs 难度,经典 D_c 断崖、学习式扩 ΔD)+ **exploitability 曲线** + **track-vs-CRLB** + 消融(G3)+ 统计(bootstrap 1e4、Welch-t/Mann-Whitney、Cohen's d、Holm-Bonferroni、D_c bootstrap CI)。

## 决策树
- **G1 PASS + G3 证机制** → **TAES 命题成立** → WP3 铺全套 + 写作,主投 TAES,兜底 IET;
- **G1 FAIL(博弈论经典没被打破)但 G5 成立(joint-classical 在线不可解)** → 走**工程使能**叙事(实时逼近不可解联合最优)→ **IET/TAES 仍可**;
- **G1 FAIL 且 G5 不成立(经典在线也能追平)** → 耦合不够 → **先升难度**(更多目标/更狠 L3/更紧 kill-链-暴露耦合)重测;仍不破 → **诚实退**:收 C1(多基地 CRLB 传感)+ C0(基准)→ IET 传感论文。

## GPU 预算 + 时间线
| WP | 内容 | GPU-周 |
|---|---|---|
| WP0 | testbed + CRLB + 验证 | 1 |
| WP1 | 强模块化 + 博弈论经典 baseline | 1.5 |
| WP2 | 学习式指挥官 + pilot + Gate | 1.5 |
| WP3 | 全扫描 + exploitability + 消融 + 统计 | 3-4 |
| 合计 | | **~2-2.5 月** |

## 回报格式
每 WP 贴回:逐 cell(N_targets × 干扰档 × exposure)的 **{学习式, 强模块化经典, 博弈论经典}** 的 kill-rate/time-to-kill/survival/**exploitability**(per-seed+mean+95%CI)+ 低难度 sanity + track-vs-CRLB + 消融 −belief/−耦合。据此判 G1 → WP3 / 工程使能叙事 / 升难度 / 退。

## 诚实缺口 / 风险
- **R1(最大)**:博弈论经典(fictitious-play)若追平学习式 → G1 FAIL;缓解:WP0 把 kill-链+暴露耦合做足 + WP2 belief/预判架构 + G5 兜底工程使能;
- **R2**:L3 干扰机没训到位 → 低估硬度;缓解:多 seed + 收敛监控 + fallback ReactiveJammer(τ=1);
- **R3**:多目标+激光是新代码 → WP0 验证必扎实(kill 链/清零/暴露逐一核对);
- **R4**:纯仿真+军事色彩 → CRLB/损伤真实性做满;框架用"效应器有驻留约束的联合调度",剥 kill-chain 措辞;venue TAES/IET 对纯仿真+EW 友好(Li'22/Dolinger'25 均纯仿真)。

## 论文结构(TAES)
1. Intro:认知雷达-DEW 交战 under 自适应 EW + 经典模块化的失效边界 + 学习式联合调度扩包线;
2. Related:认知雷达 RRM(Qin'23)、雷达抗干扰博弈(Li'22 NFSP)、多智能体雷达(Xiong'23、Dolinger'25)、CTDE;
3. 系统 C1(IQ 级 testbed,MATLAB 校验)+ 博弈形式化 + CRLB;
4. 方法 C2(联合 belief-conditioned CTDE 指挥官 + league 自博弈);
5. 实验 C3(操作包线 + exploitability + 消融 + 统计 + track-vs-CRLB);
6. 讨论:工程使能(实时逼近不可解联合最优)、局限(仿真)、双用途。
