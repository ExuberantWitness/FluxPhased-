# 传感层 pilot — 精确执行清单(给 PRO6000 agent)

**关联**:`EXPLORATION_SENSING_TRACKER.md` §2(判别 pilot)。
**目的**:用最便宜的一步(~1-2 天)裁决"传感层算法能否赢经典"——**学习式跟踪器 @ L3 是否显著优于经典 Kalman**。这决定 EAAI 传感层路线成不成立;不过则退 Path A。
**命门 G**:学习 KalmanNet 式跟踪器 @ L3 的 **track RMSE 显著 < best-tuned 经典 KF**,95%CI 不含 0(且经典 KF 的 `trace_P_norm` 复现 ≈0.07 塌陷)。

---

## 前置事实(已核实,别改错)
- **经典 KF baseline** = `KalmanTracker`(`algo/_shared/laser/sensing.py:182`):
  - state `_trk_x` [E,T,2敌,2坐标] = **纯位置随机游走 + 固定 `track_q_m=0.05`**(⚠️ 无速度项、无机动模型 → 对机动目标本就失配);
  - `_trk_P` [E,T,2,2,2];`trace_P` 属性 = P00+P11 → [E,T,2];
  - 量测经 `fused_sensing(track=True)` **in-place** 更新 tracker。
- **量测噪声** `add_sensing_noise`(`sensing.py:45`):`nr = randn × range_sigma_m`、`nc = randn × (R × crossrange_factor)`;`jam_mul` **同时**放大 sr、cf(`fused_sensing` L112-114)。
- **干扰机**(`env/gpu/qos_rrm/adversary.py`):接口 `jammer.step(red_task_hist, jam_history) → jam_level [E, n_teams]`,`reset(num_envs, n_teams, device)`;
  - L0 `StaticJammer(jam_level=0.3)` / L1 `ReactiveJammer(τ 延迟 EMA)` / L3 `LearnedJammer(小 MLP,PPO 训)`。
- **track QoS** = `trace_P_norm`(`env/gpu/qos_rrm/spectrum_metrics.py:73`,归一化协方差迹 [0,1];L3 实测塌到 0.07)。

> **机会洞察(立论核心)**:经典 KF 是**纯位置随机游走 + 固定 Q** → 对"机动目标 + 自适应抬噪"是**模型失配**,**不一定是信息极限**。学习跟踪器(建模速度/机动 + 自适应有效噪声 + 用 JSR 时序)有**结构性赢面**。pilot 就是区分"模型失配(可学)vs 信息极限(不可学)"。

---

## Step 1 — 数据生成(`algo/_shared/sensing_pilot/gen_tracks.py`,新建)
1. 生成 **N ≥ 5000** 条 2D 目标真轨迹 `x_t`:**CV(匀速)+ 转弯机动**混合,尺度对齐 env(~3km 距离、`half_map`);
2. 轨迹 **disjoint** 切 train/val/test ≈ 3500/750/750;
3. **固定雷达 task 分配**(用现 classical scheduler 或代表性固定 pattern)→ 喂 jammer 得 `jam_level[t]`;**关键:经典 KF 与学习跟踪器面对同一 jam 流 = 公平对比**;
4. 每步用 `add_sensing_noise` + `jam_mul = 1 + jam_gain·jam_level` 产量测 `z_t`;
5. 每难度导出 `(x_t, z_t, jam_level_t)` → 存 `experiments/sensing_pilot/data/{L0, L1_tau16, L1_tau8, L1_tau4, L1_tau2, L1_tau1, L3}.pt`。

## Step 2 — 经典 KF baseline(强对照,决定 G 的可信度)
1. 跑 `KalmanTracker` 处理 `z_t` → `x̂_t`、`trace_P`;算 **RMSE = ‖x̂ − x‖**;
2. **强经典对照(必做)**:对每难度**网格搜** `track_q_m ∈ {0.01, 0.05, 0.2, 1.0}` 取最优 → **best-tuned KF**;(可选)加 CV-model KF 或 IMM;
3. **学习跟踪器必须赢 best-tuned KF 才算数**(防"只赢一个乱调的 KF"这一审稿死穴);
4. 记录 `trace_P_norm` 曲线(把结果连回已知的 0.07 塌陷)。

## Step 3 — 学习跟踪器(`algo/_shared/sensing_pilot/kalmannet.py`,新建)
- **主方法 · KalmanNet 式**:保留 KF 的 predict/update 结构,GRU 吃 innovation `δ_t = z_t − H·x̂⁻_t` + 特征 + **JSR 指标(jam_level)** → 输出 **Kalman 增益 K_t**(替代解析增益/固定 Q);
- **最小可行(先跑 go/no-go)**:简版 GRU `(z_t, jam_level, x̂_{t-1}) → x̂_t`,监督 MSE(`x̂_t, x_t`);
- 训练:Adam,train 集 = **L1 + L3 混合**,early-stop on val,小网(GRU hidden 64-128),分钟-小时级(便宜)。

## Step 4 — 指标 + 门 + 机制核查
- **主指标 track RMSE**(vs 真值,test 集)vs **对手自适应性**(L0 → L1 各 τ → L3);bootstrap 1e4 per-trajectory 95%CI;
- **G(命门)**:learned RMSE @ L3 **显著 < best-tuned KF**,95%CI 不含 0(且 KF `trace_P_norm` ≈ 0.07 塌陷复现);
- **机制核查(防碰巧)**:learned 的增益/修正幅度**随 jam_level 上升而变**(证它真在用干扰信息);
- **消融**:−JSR 输入(证 JSR 有因果贡献)、−recurrence(退前馈,证时序建模的价值);
- **诚实**:L0 learned ≈ KF 是正常(经典够用),照实报,不 p-hack。

## Step 5 — 回报格式
逐难度贴回 **{经典 KF, best-tuned KF, learned} 的 RMSE(per-traj + mean + 95%CI)** + 相变曲线(RMSE vs 自适应性)+ `trace_P_norm` + 机制核查图(gain vs JSR)+ 消融结果。
- **G PASS**(learned @ L3 赢 best-tuned KF)→ **传感层路成立** → 铺 `EXPLORATION_SENSING_TRACKER.md §3` 全套(CRLB/PCRB 物理锚 + 全难度扫描 + 泛化到 held-out 干扰机);
- **G FAIL**(learned 赢不了 KF)→ **track 是真信息极限**,任何估计器都救不了 → **退 Path A**(传感 C1 + 基准 C0 → IEEE TAES/TGRS,或 Applied Intelligence 2 区)。

---

## 风险 / fallback
- **R4(L3 干扰机不收敛)**:`LearnedJammer` 若在 pilot 预算内 PPO 不收敛 → 用 `ReactiveJammer(τ=1)` 当高难度代理,**别让没训好的干扰机污染判据**。
- **R-scale(量测尺度)**:`add_sensing_noise` 的 `range_sigma_m`/`crossrange_factor`/`jam_gain` 取现 env 配置真实值(`configs/laser_25x25_ew_exposure.yaml` 有全套),别自造尺度导致难度失真。

## 纪律(这一路踩坑换来的)
1. **逐 cell + CI**,不看聚合单数;头对头比 learned vs **best-tuned** KF @ L3;
2. **强经典是前提**(best-tuned KF / IMM,不是 fixed-Q 稻草人),否则赢了也不可信;
3. **机制核查**(gain vs JSR)查 WHY,不只看 RMSE 数字;
4. **诚实**:低难度 learned≈KF 照实报;G FAIL 就是 G FAIL,退 Path A 不硬撑。
