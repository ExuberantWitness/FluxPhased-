# TAES 实现级详细 spec(给 PRO6000 agent)— `TAES_MAINLINE_PLAN.md` 的执行伴档

> 策略/贡献/venue 见 `TAES_MAINLINE_PLAN.md`。本文件给**可直接落地的实现细节**:环境参数、kill-链数学、强基线算法、网络架构、超参数、训练配置、评估协议、成功指标、要建的配置文件。标 `⟨TUNE-WP0⟩` 的是需在 WP0 标定的量。
> **地雷**:checkpoint_dir→`checkpoints/taes_mainline/`(严禁 /tmp);`log_std_floor=-6`(勿 -4);基线用 p14 值系,勿用 local。

---

## 0. 文件映射(每块插在哪)
| 组件 | 新建/改 | 位置 |
|---|---|---|
| 多目标+kill链+暴露 env | 改 | `algo/_shared/laser/` env + `reward.py` |
| CRLB/PCRLB | 新建 | `algo/_shared/laser/crlb.py` |
| IMM-PDAF 基线 | 新建(Stone Soup) | `algo/_shared/baselines/imm_pdaf_commander.py` |
| Q-RAM 调度 | 强化 | `algo/_shared/baselines/classical_qos_rrm.py` |
| 火控/ECCM | 扩 | `algo/_shared/baselines/classical_mpc.py` |
| 博弈论经典(fictitious-play) | 新建 | `algo/_shared/baselines/fictitious_play.py` |
| exploitability harness | 新建 | `algo/_shared/eval/exploitability.py` |
| 联合 RL 指挥官 | 扩 | `CommanderActorCritic`(`actor_critic.py`) |
| 训练入口 | 改 | `training.train_laser` + `SimpleMAPPOTrainer` |
| 配置 | 新建 | `configs/taes_*.yaml`(见 §6) |

---

## 1. WP0 — 环境实现细节

### 1.1 目标与几何
- `N_targets ∈ {1,2,4,8}`;初始位置均匀撒在 [2.5, 3.5] km 环带、方位 [−60°,60°];
- 运动:近常速 CV,速度 `v ∈ [150,300] m/s`;机动 = 泊松触发转弯,`p_turn=0.05/step`,`turn_rate ∈ [5,15]°/s`;⟨TUNE-WP0:机动强度使强经典在多目标掉但单目标不掉⟩;
- 雷达:双阵,`min_radar_baseline_m=5000`(p14 值),各 25 子阵(5×5);
- 量测噪声(取自 `ew_exposure`):`range_sigma_m`(cm 级)、`crossrange_factor=7.4e-5`(σ_cross=R×factor);`residual_scale_m=6.0`。

### 1.2 激光驻留-kill-链(核心耦合,formula)
每目标 i 维护 kill 能量 `E_i`,每 control step dt:
```
track_ok_i = (kr_i < kr_thresh) AND (trace_P_i < τ_track)      # track 达标才积能
if laser_assigned_to == i AND track_ok_i:
    E_i += dwell_rate * dt
elif not track_ok_i:                                            # track 丢 → 惩罚性衰减
    E_i *= decay_factor          # decay_factor < 1(硬耦合可设 0=清零)
kill_i = (E_i >= E_kill)
```
默认(⟨TUNE-WP0⟩ 起点):`kr_thresh=0.5m`、`τ_track` = 单目标无干扰稳态 trace_P 的 3×、`dwell_rate=1.0/s`、`E_kill=2.0`(=需 2s 连续达标 track)、`decay_factor=0.5`。**标定目标:单目标无 EW 时经典 ~2s 稳出 kill;多目标+L3 时经典因 track 抖频繁触发衰减、time-to-kill 爆或 kill 失败。**

### 1.3 暴露 / home-on-jam(formula)
```
exposure += emit_power * dt            # 照射/干扰辐射累积;emit_power ∝ 分配给该目标的子阵数
p_homejam = 1 - exp(-exposure_gain * exposure_norm)     # exposure_gain=50(ew_exposure)
if bernoulli(p_homejam): own_death → reward -= race_death_penalty(=30)
```
指挥官可 `emission_off`(被动传感)降 exposure,代价是 track 质量降 → **暴露-track 权衡**。

### 1.4 自适应干扰机(`adversary.py`,3 档)
- **L0** `StaticJammer(jam_level=0.3)`;**L1** `ReactiveJammer(τ∈{16,8,4,2,1})` EMA on red task hist;
- **L3** `LearnedJammer`:MLP,输入 = 指挥官 task 直方图 + 自身 jam 历史,输出 = jam_level + 目标 onehot(**干扰当前正被 kill 的目标**);PPO 训,与指挥官 league 共训;
- `jam_mul = 1 + jam_gain(=8) * jam_level` 抬被照射目标的 range+crossrange σ(`fused_sensing` L112-114)。

### 1.5 CRLB/PCRLB(`crlb.py`)
- 多基地定位 **CRLB**:FIM `J = Σ_r H_r^T R_r^{-1} H_r`(r=两雷达),`H_r`=量测雅可比(range+crossrange),`R_r`=含 jam_mul 的量测协方差;`CRLB = trace(J^{-1})`;
- 跟踪 **PCRLB** 递归:`J_{k+1} = (Q + F J_k^{-1} F^T)^{-1} + Σ_r H^T R^{-1} H`;
- 报告:各方法 `trace_P` vs `PCRLB`(经典近 PCRLB=传感已解证据;干扰下偏离=难点)。

### 1.6 WP0 验证清单(必过)
逐目标打印:track_loss 率、`E_i` 累积-衰减轨迹、kill/time-to-kill、exposure、survival;**单目标无 EW:经典 kill≈1.0、time-to-kill≈E_kill/dwell_rate、trace_P≈PCRLB(sanity);N=4+L3:经典 kill 明显掉、time-to-kill 爆(硬 regime 成立)。**

---

## 2. WP1 — 强基线实现细节(anti-strawman 命门)

### 2.1 IMM-PDAF 多目标跟踪(`imm_pdaf_commander.py`,Stone Soup)
- 模型库:CV + 协调转弯 CT(2 模型 IMM);转移矩阵 `[[0.95,0.05],[0.05,0.95]]`;
- PDAF:门 `gate_prob=0.997`(3σ),杂波密度按 env false-alarm 设;
- 每目标一个 IMM-PDAF track;量测 = env 的 fused 量测(含 jam_mul);
- 输出每目标 (状态估计, 协方差) 喂下游调度/火控。

### 2.2 Q-RAM 调度(强化 `classical_qos_rrm.py`)
- 各功能 QoS 效用:detect(SNR)、track(−trace_P/PCRLB)、jam(有效性)、comm(CRC);
- 分配:Lagrangian/water-fill 把 25 子阵按 **kill 优先级 × track 余量**分给各目标各功能;`qos_floor_per_fn` 保底;
- **短视界 rollout 版**(强化):1-2 步前瞻(非纯 myopic),作更强对照。

### 2.3 火控 shoot-look-shoot(扩 `classical_mpc.py`)
- 目标优先级 = `kill_progress_i / exposure_cost_i`(kill 进度高、暴露低者优先);
- 激光指派给最高优先级且 track_ok 的目标;shoot→look(核 kill)→ 切换。

### 2.4 反应式 ECCM
- 检测某功能被干扰(JSR>θ 或 trace_P 膨胀)→ 频率捷变(换 band)+ 子阵重分配到受扰目标。

### 2.5 博弈论经典(`fictitious_play.py`,Li'22 严谨要求)
```
初始化经典策略 π_c^0(=强模块化经典)
for iter in range(K):
    BR_jammer = best_response(π_c^iter)          # 训/搜对当前经典的最优干扰
    π_c^{iter+1} = best_response(BR_jammer 分布)  # 经典对干扰机分布重优化(调 ECCM/调度参数)
    平均策略 π̄_c = 时间平均
返回 π̄_c(逼近对干扰机分布的均衡)
```
- **G1 必须打赢 π̄_c,不只打赢 π_c^0**——防"固定 Nash 就赢 RL 则故事塌"。

### 2.6 exploitability harness(`exploitability.py`)
```
exploitability(π) = U(π vs 静态干扰L0) − U(π vs BR(π))
U = kill_rate  或  survival_rate  或  −(1−kill)*death 的近零和标量
BR(π): 冻结 π,训一个 PPO 干扰机 ~N steps 打它,取收敛值
```
对 {π_c^0, π̄_c, π_RL} 各算 exploitability + 头对头。

### 2.7 WP1 验证(必过)
强模块化 + π̄_c 在**低难度**:kill≈1.0、survival 高、exploitability 低(证非稻草人);硬 regime 掉。

---

## 3. WP2 — 联合 RL 指挥官实现细节

### 3.1 观测向量(逐维)
per-target block(× N_targets,pad 到 max=8):
`[x̂_i(2), v̂_i(2), trace_P_i(1), E_i kill进度(1), JSR_i 分带(4), track_ok_i(1)]` = 11 维/目标;
global:`[exposure(1), 各功能子阵占用(4), 自身存活(1), step_norm(1)]` = 7 维。
**总 obs ≈ 11×8 + 7 = 95 维**(pad-mask 无效目标)。CTDE privileged critic 额外加:干扰机 jam_level 真值 + 我方 emission/exposure 对偶(`commander_privileged_dim` 扩到 ~20)。

### 3.2 网络架构
- 共享 trunk:per-target 11维 → shared MLP(64) → **attention/mean-pool over targets**(处理变长目标数)→ concat global 7维 → MLP(256,256);
- **belief-conditioning(方法加分)**:trace_P_i 既进 obs 也作 gating(low-confidence 门控 actor 的开火/激光头);
- actor 头:子阵→功能分配(25×4 logits 或 Dirichlet)+ 波束指向(连续)+ 激光目标选择(N_targets softmax)+ emission on/off(Bernoulli);
- critic:CTDE 中心化,吃 privileged obs → V(s);noise-robust:`α_eff = α_max·exp(−β·trace_P_norm)`,`adv = (1−α_eff)A_agent + α_eff A_team`。

### 3.3 超参数(PPO/MAPPO,起点=p14 proven-stable)
| 超参 | 值 |
|---|---|
| optimizer | Adam, lr=3e-4(actor)/1e-3(critic), anneal |
| PPO clip | 0.2;epochs=4;minibatch=按显存 |
| GAE | λ=0.95;γ=0.99 |
| entropy coef | 0.01→anneal;value coef=0.5 |
| **log_std_floor** | **−6(勿 −4)** |
| grad clip | 0.5 |
| num_envs | 拉满 PRO6000 显存(参考本机 2→ PRO6000 大幅上调) |
| rollout | 每 control step 块;total ~2e7-5e7 steps/run |
| kr 课程 | 50→0.5m(p14) |

### 3.4 League/PSRO(与 L3 干扰机共训)
- `league:true`;对手池初始 {L0,L1(各τ),L3};
- PFSP 采样:`wr[opp]` EMA 更新(`update_pool_winrate`,已修),优先采难打对手;
- 定期把当前指挥官/干扰机快照入池;生成 best-response 对手多样性(→ 低 exploitability 的关键)。

### 3.5 奖励 shaping(`DenseRewardShaper`)
`R = w_kill·kill + w_surv·survival − w_exp·exposure − w_ttk·time_to_kill_penalty − w_jamcost·jam_cost`
起点权重:`w_kill=10, w_surv=5, w_exp=1(exposure_gain 内), w_ttk=0.1, race_death_penalty=30`(ew_exposure 值)。

### 3.6 难度扫描网格
`N_targets{1,2,4,8} × 干扰{L0,L1-τ16,τ8,τ4,τ2,τ1,L3} × exposure{low,high}` = 4×7×2 cells,每 cell ≥5 seed。**报完整曲线含低难度经典够用区(诚实)。**

---

## 4. 评估协议 + 成功指标(formula)

### 4.1 指标
- **kill_rate** = 击杀目标数 / 总目标数(per episode 平均);
- **time_to_kill** = 首 kill 的 step 数(未 kill = 上限);
- **survival_rate** = 未被 home-on-jam 反杀的 episode 比;
- **track_loss_rate** = trace_P 超发散阈的目标-步比;
- **exploitability**(§2.6);
- **track-vs-CRLB** = mean(trace_P / PCRLB)(越近 1 越优);
- **cross-play**:π_RL vs {经典} 在共同 held-out 干扰机集上双向平均。

### 4.2 统计
≥5 seed;mean±95%CI(bootstrap 1e4);头对头 Welch-t / Mann-Whitney;效应量 Cohen's d;多重比较 Holm-Bonferroni;操作包线 D_c(kill 跌破阈的难度)用 bootstrap CI;曲线交叉点 CI。

### 4.3 Gate(逐 cell + CI)
| # | 判据 | 阈值 |
|---|---|---|
| **G1** | π_RL > **π̄_c(博弈论经典)** @ 硬 regime:kill gap>0.10 **或** exploitability 显著更低,95%CI 不含 0 |
| G2 | 低难度 π_RL ≈ 经典(诚实报) |
| G3 | 消融 −belief / −耦合 / −预判 → 丢 G1 gap(证机制) |
| G4 | 强模块化+π̄_c @ 低难度 kill≈1.0、exploitability 低(非稻草人) |
| G5 | joint-MPC/POMDP 经典在此规模在线不可解(工程使能兜底) |

---

## 5. 决策树(同 mainline)
G1 PASS+G3 → WP3 全套→TAES;G1 FAIL 但 G5 → 工程使能叙事→IET/TAES;G1 FAIL 且 G5 不成立 → 升耦合难度重测;仍不破 → 退 C1(CRLB 传感)+C0→IET。

## 6. 要建的配置文件
- `configs/taes_env.yaml`:§1 全部 env 参数(N_targets、kill链、exposure、jammer、CRLB 开关);
- `configs/taes_rl_commander.yaml`:§3 架构+超参+league;
- `configs/taes_baselines.yaml`:§2 IMM-PDAF/Q-RAM/火控/ECCM/fictitious-play 参数;
- 各 `checkpoint_dir: checkpoints/taes_mainline/<name>`(**非 /tmp**)。

## 7. 时间线 + 回报
WP0(1w)→WP1(1.5w)→WP2(1.5w,判 G1)→WP3(3-4w)。每 WP 贴回:逐 cell {π_RL, 强模块化经典, π̄_c} 的 kill/ttk/survival/exploitability(per-seed+mean+95%CI)+ 低难度 sanity + track-vs-CRLB + 消融。**先跑 WP0+WP1 验证(sanity + 强经典非稻草人)再动 WP2——别在没验证 baseline 前判 G1。**

## 8. 风险(实现层)
- kill-链/暴露参数标定是 WP0 成败(耦合太松→经典追平,太紧→RL 也学不动)⟨TUNE-WP0 反复调⟩;
- IMM-PDAF(Stone Soup)与我们 IQ 量测接口对齐要仔细(坐标系/协方差单位);
- L3 干扰机 + league 收敛监控(NaN/adv_std 爆→先诊断,fallback ReactiveJammer τ=1);
- exploitability 的 BR 干扰机必须训到位(否则低估 exploitability=假优)。
