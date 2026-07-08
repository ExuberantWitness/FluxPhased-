# EAAI 主线执行计划 — 联合 RL 指挥官 vs 强模块化经典指挥官(给 PRO6000 agent)

**系统范畴(不得脱离)**:双相控阵雷达 + 指挥官 + 激光。多基地 Kalman 融合传感 → 指挥官分配 25 子阵(侦/跟/干/通)+ 指挥激光 → 激光对目标积能出 kill;敌方自适应干扰 + home-on-jam 暴露。
**核心命题(EAAI headline)**:在**多目标 + 自适应 EW + 激光驻留-kill-链 + 暴露权衡**的硬 regime 下,**联合 RL 指挥官在 kill 目标上显著打赢强模块化经典指挥官**——因为经典把 跟踪/调度/火控 分解成局部最优的模块,**紧耦合下这个分解有损**,而联合策略优化了模块间耦合 + 预判自适应干扰机。
**为什么是 EAAI 不是 TAES**:单侧 AI 决策方法解一个工程感知-行动联合调度系统,打赢强经典,benchmark 可复现;联合最优是在线不可解的高维 POMDP → RL 的 amortization = 工程实时可部署(EAAI 认此价值)。
**纪律(这一路 5 次失败换来的)**:① 强 baseline 必须是**强模块化经典**(IMM-PDAF + Q-RAM + 最优火控 + 反应式 ECCM),不是"瞄 anchor + 开火"稻草人;② 在**硬 regime**测,不在单目标无 EW 的平静海面;③ 早停:强模块化经典若在**任何** regime 都打不破,就是分解不够有损 → 加耦合难度或诚实退。

---

## 已核实代码触点(别改错)
- 传感:`algo/_shared/laser/sensing.py` — `KalmanTracker`(`_trk_x/_trk_P`,`trace_P`=P00+P11),`fused_sensing(track=True)` in-place,`jam_mul` 放大 range+crossrange σ;
- 指挥官:`CommanderActorCritic`(obs_dim=76,`commander_privileged_dim:10` CTDE critic,`task_head` 四功能);
- 激光/奖励:`algo/_shared/laser/reward.py` — kill 半径 `kr`、`jam_level`/`jam_cost`、`exposure_gain`(home-on-jam)、`race_death_penalty`;
- 对手:`env/gpu/qos_rrm/adversary.py` — `StaticJammer(L0)/ReactiveJammer(L1,τ)/LearnedJammer(L3,PPO)`;
- 经典基座:`algo/_shared/baselines/classical_mpc.py`(现只会瞄+开火)、`classical_qos_rrm.py`(water-fill);
- 训练:`training.train_laser`(入口),`SimpleMAPPOTrainer`,league/PSRO 开关(`league`/`use_mappo`);
- 配置:`configs/laser_25x25_p14.yaml`(基线,勿用 local)、`configs/laser_25x25_ew_exposure.yaml`(EW:jam_gain=8,exposure_gain=50)。
- **⚠️ checkpoint_dir 必须指持久盘(如 `checkpoints/eai_joint/`),严禁 /tmp(会写满崩)。log_std_floor 用 -6(p14 proven-stable),勿用 -4。**

---

## WP0 — 硬 regime 环境:多目标 + 激光驻留-kill-链 + 暴露(~3-5 天)
把现单目标环境扩成能让"模块化分解真掉链子"的最小硬 regime。

**0.1 多目标**:`N_targets ∈ {1,2,4,8}`,各自机动(CV + 转弯),多基地融合对每目标维护 IMM/Kalman track。
**0.2 激光驻留-kill-链(核心新机制,决定耦合)**:激光对目标 i **只在** track 质量达标(`kr_i < kr_thresh` 或 `trace_P_i < τ_track`)时**累积** kill 能量 `E_i += dwell_rate·dt`;**一旦 track 丢(σ 被干扰抬过阈)→ E_i 清零或衰减**;`E_i ≥ E_kill` 出 kill。→ 制造"必须持续保住 track 才出 kill"的时序耦合。
**0.3 暴露(现成,接上)**:雷达照射/干扰辐射 → `exposure` 累积 → home-on-jam 概率(`exposure_gain`)→ 被反杀(`race_death_penalty`)。→ 制造"track 质量 vs 暴露"的权衡。
**0.4 自适应 EW**:干扰机 L0→L3,`jam_mul` 抬被照射目标的 σ;L3 `LearnedJammer` 观测指挥官 task 直方图 → 干扰**当前正在 kill 的目标**(制造"预判/预置"价值)。
**0.5 验证(必过)**:打印每目标 track_loss、kill 链累积/清零、exposure、survival;确认单目标无 EW 时经典近满 kill(平静海面 sanity),多目标+L3 时经典明显掉(硬 regime 成立)。

**耦合点清单(RL 要利用、模块化经典抓不到的)**:
- 跨目标资源权衡(为目标 B 将成之 kill 牺牲目标 A 的 track);
- 暴露-track 权衡(为降暴露主动收火/换被动传感);
- 预判 L3 干扰机(预置资源/波形捷变,先于它抢占)。

## WP1 — 强模块化经典指挥官(anti-strawman 命门,~5-7 天)
**这是全计划成败前提。经典必须强,否则 RL 赢了也不可信。** 四个模块各自强、但接口固定(有损):
| 模块 | 强实现 | 库/触点 |
|---|---|---|
| **多目标跟踪** | **IMM-PDAF**(非单 KF;CV+CT 模型库 + 概率数据关联)| **Stone Soup** `dstl/Stone-Soup`(IMM/PDAF/PMBM 现成) |
| **资源调度** | **Q-RAM / 短视界 rollout**:按各目标 kill 优先级 × track 余量分配 25 子阵(非 equal/round-robin)| 强化 `classical_qos_rrm.py` |
| **火控排序** | **shoot-look-shoot / 最优目标选择**:选 kill-progress-per-exposure 最高目标驻留 | 扩 `classical_mpc.py` |
| **ECCM** | **反应式**:检测被干扰功能 → 频率捷变 + 子阵重分配 | 新增反应逻辑 |
- **模块化 = 每模块优化自己的目标 + 固定接口**(调度器不知火控 kill 进度;跟踪器不知该为暴露收火)——**这个"有损接口"正是 RL 要打的点**;
- **验证(必过)**:此强模块化经典在**单目标无 EW / 低难度**下 kill-rate 近满、survival 高(证它是强 baseline 非稻草人);在硬 regime 掉(证 regime 真难)。

## WP2 — 判别 pilot:联合 RL 指挥官 vs 强模块化经典(决定成败,~1 周)
**2.1 联合 RL 指挥官**:扩 `CommanderActorCritic`:
- **obs**:每目标 (track 均值 + `trace_P_i` + kill 进度 `E_i`) + JSR/jam_level(分带)+ exposure 状态 + 资源占用;（CTDE:privileged critic 加干扰机状态/暴露对偶）;
- **action**:子阵→功能分配 + 波束指向 + **激光目标选择** + 辐射控制(emission on/off 降暴露);
- **reward**:kill-rate + survival(−exposure 惩罚)+ −time-to-kill,shaped(`DenseRewardShaper`);
- **训练**:PPO/MAPPO(CTDE,`use_mappo:true`),num_envs 按 PRO6000 显存拉满,psro/league 用于 L3 自适应干扰机共训;log_std_floor=-6;kr 课程沿用 p14。

**2.2 难度扫描(操作包线,headline 方法学)**:沿 `N_targets × 干扰机 L0→L3(+ ReactiveJammer τ 连续)× exposure 强度` 扫全程,每点 ≥5 seed。**报告完整曲线含低难度经典够用区(诚实,防 p-hacking)。**

**2.3 指标**:kill-rate、time-to-kill、**survival-rate**(未被反杀)、per-target track-loss、暴露累积;cross-play(若 L3 自适应)双向平均 + 共同 held-out 干扰机。

## Gate(逐 cell + bootstrap CI,决定进 WP3 还是退)
| # | 判据 | 阈值 | 决定 |
|---|---|---|---|
| **G1(命门)** | 联合 RL **> 强模块化经典** @ 硬 regime(多目标+L3)| kill-rate gap > 0.10,95%CI 不含 0 | EAAI 命题成立 |
| G2(诚实) | 低难度 RL ≈ 经典(单目标无 EW)| 照实报,不 p-hack | 相变在 + 诚实 |
| G3(机制) | 消融 −耦合(RL 只给模块化分解的 obs/reward)→ **丢掉 G1 的 gap** | gap 消失 | 证赢来自耦合优化,非碰巧 |
| G4(强baseline) | 强模块化经典 @ 低难度 kill 近满/survival 高 | 证非稻草人 | WP1 验证 |
| G5(EAAI框架) | 联合最优的经典版(joint-MPC/POMDP)在此规模**在线不可解**(或解得慢到不可部署)| 证 amortization = 工程使能 | EAAI 卖点 |

## WP3 — 全套(仅 G1 PASS,~3-4 周)
- 全难度扫描 × {联合RL, 强模块化经典, (可选)纯 IPPO/MAPPO} × ≥5 seed → **操作包线图**(kill-rate vs 难度;经典在 D_c 断崖,联合 RL 扩 ΔD);
- 消融:−耦合(G3)、−预判(去 L3 共训)、−暴露项、模块逐个换强/弱;
- 物理锚:多基地 **CRLB/PCRLB**(track 质量下界)+ 激光能量-驻留模型真实性;
- 统计:mean±95%CI(bootstrap 1e4)、Welch-t/Mann-Whitney、Cohen's d、Holm-Bonferroni、D_c bootstrap CI。

## 决策树
- **G1 PASS + G3 证耦合** → **EAAI 命题成立** → WP3 铺全套;论文框架"受扰条件下雷达-激光联合调度的智能决策(工程使能:实时逼近在线不可解的联合最优)";
- **G1 FAIL(强模块化经典没被打破)** → 分解不够有损 → **先升耦合难度**(更多目标/更狠 L3/更紧驻留-暴露耦合)重测;仍不破 → **诚实退**:收 C0(IQ MFAR EW 基准)+ C1(多基地 CRLB)→ IET/TAES,别硬凑 EAAI。

## GPU 预算 + 时间线
| WP | 内容 | GPU-周 |
|---|---|---|
| WP0 | 硬 regime env + 验证 | 0.5-1 |
| WP1 | 强模块化经典 baseline + 验证 | 1-1.5 |
| WP2 | 判别 pilot + Gate | 1-1.5 |
| WP3 | 全扫描 + 消融 + 统计(仅过 gate) | 3-4 |
| 合计 | | **~2-2.5 月** |

## 回报格式
每 WP 贴回:**逐 cell(N_targets × 干扰档 × exposure)的 {联合RL, 强模块化经典} kill-rate/time-to-kill/survival(per-seed + mean + 95%CI)+ 低难度 sanity + 消融 −耦合结果 + 强经典非稻草人验证**。据此判 G1 → 进 WP3 还是升难度/退。

## 诚实缺口 / 风险
- **R1(最大)**:强模块化经典若把接口做通(joint-MPC)就追平 RL → 说明耦合不够紧;缓解:WP0 把驻留-kill-链 + 暴露耦合做足,G5 验证 joint-classical 在线不可解;
- **R2**:L3 自适应干扰机没训到位 → 低估硬度;缓解:多 seed + 收敛监控,fallback ReactiveJammer(τ=1);
- **R3**:多目标 + 激光扩展是新代码 → WP0 验证必须扎实(kill 链/清零/暴露逐一打印核对);
- **R4**:全仿真 + 军事色彩对 EAAI 偏险 → 框架用"感知-行动联合调度/效应器有驻留约束",剥 kill-chain 措辞;C1 CRLB 真实性做满;venue 兜底 TAES/IET。
