# 探索方案 — 传感层:学习式自适应跟踪打赢自适应 EW 下塌陷的经典 Kalman(冲 EAAI)

> **方向**: 把算法从"控制/调度层"(反复输经典)挪到"传感/估计层"——**经典 Kalman 在 L3 自适应干扰下已证塌陷(track QoS→0.07);做一个学习式自适应跟踪器,在经典失效处维持 track,即"AI 在经典可证失效的层赢经典" = EAAI 正中靶心。**
> **纪律**: 探索 = pilot 先证伪(~1-2 天)。赢经典 → 铺全套;赢不了(track 是真信息极限)→ 退 Path A。

---

## 1. 假设与机会(为什么可能赢)
- **经典 baseline**: `KalmanTracker`(`algo/_shared/laser/sensing.py:182`),**固定** `track_q_m=0.05`、线性高斯、固定量测模型;
- **为什么在 L3 塌**: `LearnedJammer`(`env/gpu/qos_rrm/adversary.py`,PPO 训的 MLP)**自适应、非平稳、定向**地抬高量测噪声(`jam_mul` 放大 range+crossrange σ)→ **固定 Q 的 KF 模型失配 → trace(P) 爆 → track QoS 0.07**;
- **机会**: **这是模型失配失效,不一定是信息极限**。学习式跟踪器能 ① 自适应有效过程/量测模型、② 利用干扰的时序结构、③ 预判自适应干扰机 → **在 KF 失效处维持 track**。文献背书:**KalmanNet(Revach 2022,学习式 KF 在模型失配下打赢经典 KF)**、认知雷达跟踪 LSTM-TD3。
- **诚实风险**: 若干扰真把量测信息毁掉(JSR 高到任何估计器都恢复不了状态)→ 信息极限 → 学习也没用。**pilot 就是区分"模型失配(可学)vs 信息极限(不可学)"。**

## 2. 判别 pilot(决定这条路成不成立,~1-2 天)
**问题**: 学习式跟踪器在 L3 能否把 track 维持到明显优于经典 KF(0.07)?

### 2.1 数据(复用现有量测模型 + 干扰机)
- 用 env 的量测生成(`add_sensing_noise`/`fused_sensing` + `jam_mul`)+ `{StaticJammer(L0), ReactiveJammer(L1), LearnedJammer(L3)}` 生成**量测流 z_t + 真状态 x_t + 可观测干扰指标(JSR/jam_level)**;
- 每难度档 ≥ 数千条轨迹(便宜,纯仿真无需大训练)。

### 2.2 两个估计器对比(同一量测流,公平)
| | 经典 KF(baseline) | 学习式跟踪器(本方法) |
|---|---|---|
| 实现 | 现 `KalmanTracker`(固定 track_q) | **KalmanNet 式**:GRU 吃 (innovation, 先验估计, JSR 指标) → 输出 Kalman 增益/修正(替代固定 Q/解析增益);或简版 GRU:(z_t, jam 指标)→ 状态估计 |
| 训练 | 无(解析) | 监督:预测真状态 MSE(便宜,~分钟-小时级,非 RL) |
| 输入的干扰信息 | 无 | **JSR/jam_level 指标 + 时序**(这是它超越 KF 的来源) |

### 2.3 指标(相变,直接复用方法学)
- **track RMSE** 与 **trace(P)-等价 QoS** vs **对手自适应性(L0→L1→L3,+ ReactiveJammer τ 连续扫)**;
- 画曲线:**经典 KF 在 D_c 塌陷,学习跟踪器把包线往后推**(Y=track 质量,X=自适应性)——headline 图照用。

### 2.4 判据(门)
| 判据 | 阈值 | 决定 |
|---|---|---|
| **G(命门)** | 学习跟踪器 track RMSE @ L3 **显著 < 经典 KF**(track QoS 从 0.07 明显抬升,CI 不含 0)| **PASS → 传感层这条路成立,铺 EAAI 全套** |
| 低难度诚实 | L0 学习≈KF(经典够用) | 诚实汇报,不 p-hack |
| 机制核查 | 学习跟踪器的增益随 JSR 自适应变化(证"它真在用干扰信息")| 防碰巧 |
- **G FAIL** → track 是真信息极限,任何估计器都救不了 → **退 Path A**(传感 C1+基准 C0)。

## 3. 若 pilot PASS → EAAI 全套(~2-3 月)
- **主实验**: 全难度扫描 × {经典 KF, 学习跟踪器, (可选)EKF/UKF/IMM 更强经典} × ≥5 seed → **操作包线图(track 质量 vs 自适应性)+ ΔD 包线扩展**;
- **物理锚**: **CRLB/PCRB** —— 学习跟踪器逼近后验 CRLB、经典 KF 在 L3 远离 → 证"学习恢复了 KF 丢失的可达精度"(C1 变核心);
- **消融**: 去 JSR 指标输入、去时序(退成前馈)、KF process-noise 自适应版对照(证增益不只是"调大 Q");
- **泛化**: 对**未见干扰策略**(held-out LearnedJammer,联赛/自博弈生成)鲁棒;
- **升级(可选)**: 从"被动学习跟踪"到"**主动认知感知**"——RL 控波形/驻留主动对抗干扰维持 track(动作空间进一步进传感层)。

## 4. EAAI 贡献框架(全保留现有资产)
- **C0**: IQ 级四功能 MFAR EW 对抗基准(MATLAB 校验);
- **C1**: 多基地融合 + **CRLB/PCRB**(现在是学习跟踪器逼近的理论下界,变核心);
- **C2(headline)**: **自适应 EW 下的学习式雷达跟踪** —— 在经典 KF 可证塌陷处维持 track,**AI 在经典失效的传感层赢经典**;
- **C3**: 操作包线相变刻画(track 质量 vs 自适应性,ΔD)。
- **引用锚**: KalmanNet(Revach'22)、认知雷达跟踪 LSTM-TD3、多基地 CRLB(Godrich/Haimovich)、CRL2RT/CIRL(混合)。

## 5. 为什么这条比"控制层"强(一句话论证给审稿人)
经典 KF 塌陷是**模型失配**(固定 Q 对自适应干扰),**不是信息极限**——学习恢复了 KF 丢失的可达精度(逼近 PCRB)。这是"AI 在经典可证失效、且难点真正所在的传感层赢经典",而非在控制层硬凑(控制层物理主导、经典近最优、赢不了)。

## 6. 诚实的门与备选
- **先做 §2 pilot**(~1-2 天):学习跟踪器 @ L3 赢 KF 吗?
  - **赢** → 传感层 EAAI 路成立,铺 §3 全套;
  - **不赢** → track 是真信息极限 → **退 Path A**(传感+基准 → TGRS 一区 / TAES),或 **Applied Intelligence(2区)** 收现有框架+基准+表征;
- **纪律**: 逐 cell+CI、低难度诚实汇报、机制核查(增益随 JSR 自适应)、强经典对照(EKF/UKF/IMM 不只是 fixed-Q KF)。

---

## 立刻做的一步
**§2 判别 pilot**:用现有量测模型+三档干扰机生成 (z_t, x_t, JSR),训一个 KalmanNet 式 GRU 跟踪器,对比经典 `KalmanTracker` 的 track RMSE vs 自适应性曲线。**G:L3 学习 track 显著 > KF(0.07)→ 这条路成立。** 这是判断"传感层算法能否赢经典"的最便宜一步。
