# 风暴确认 Gate — S1 前的廉价决定性裁决(给 PRO6000 agent)

**关联**:`RESEARCH_STORM_STRATEGY.md`(四轴文献综合) · `EXPLORATION_SENSING_TRACKER.md`(已证伪的传感 pivot) · `env/gpu/qos_rrm/adversary.py`(PPO 干扰机)
**背景**:~15 agent 文献舰队确证——**检测/跟踪/调度层调好的强经典 = oracle,学习无诚实 gap**(Greenberg NeurIPS'23 / Mehrfard'24 / CFARnet / Brekke'11 / Davey'08 全指此)。**唯一 oracle 不存在、AI 可证赢经典处 = S1:闭环认知雷达 vs 自适应对手**(固定策略必被 best-responder 反演,只有均衡达 minimax)。
**本 gate 的唯一问题**:我们那个**完整、调好的强经典雷达栈**,是否真被一个 best-responding 自适应对手**显著 exploit**?exploit 得动 → storm 真,建 S1 冲 TAES;exploit 不动(经典已近均衡)→ 连这 regime 都良态 → 退 IET / 收 C0 基准。
**决定性指标(文献锚定,非拍脑袋)**:**exploitability / Nash-gap**(NFSP,Li 2022 TAES)+ **熵率**(Thornton 2023:熵率预测在线学习价值)。

> **为什么这是"cheap gate 用对地方"**:前 5 次在良态区造 toy 必平;这次在**文献证明唯一有 gap 的 regime**里,用**博弈论可测的 exploitability**裁决,而非我脑补的难度。

---

## 前置:非稻草人的强经典雷达栈(命门,先建对)
⚠️ **exploitability 只有对"调好的强经典"才有意义**。必须先把经典雷达栈配强(Stone Soup 提供 IMM/PDAF/PMBM,别自己造弱版):
1. **跟踪**:IMM-PDAF(非单模型 KF)——Stone Soup `dstl/Stone-Soup` 现成;
2. **ECCM/抗干扰**:规则 ECCM = 频率捷变 + 反应式子阵/驻留重分配 + 门控/振幅辅助关联(Brekke PDAFAI);
3. **调度**:Q-RAM 或非近视 rollout(非 equal-allocation),复用/强化 `algo/_shared/baselines/classical_qos_rrm.py`;
4. **验证**:此栈在**静态/非自适应干扰**下应近最优(QoS 高、track-loss 低)——若这里就弱,先修,别拿弱经典测 exploitability。

## 对手:best-response 自适应干扰机(现成 PPO)
- 用 `env/gpu/qos_rrm/adversary.py` 的 `LearnedJammer`(PPO)+ 联赛/自博弈 infra;给它**足够容量 + 观测雷达行为**(task 直方图/频率/驻留),训到对固定雷达策略的 best-response。

---

## Step G1 — 经典栈的 exploitability(必要条件,storm 存不存在)
1. **固定**强经典雷达栈 π_c(上面配好的,确定或固定随机策略);
2. 训 best-response PPO 干扰机 BR(π_c) 打它;
3. 算 **exploitability(π_c) = U(π_c vs 静态干扰) − U(π_c vs BR(π_c))**,U = QoS-under-jamming 或 −track-loss/OSPA;
4. 意义:best-responder 能把固定经典拉下多少 = 经典的可利用度。

## Step G2 — 自博弈雷达的 exploitability(充分条件,学习能否吃下 storm)
1. 用联赛/PSRO 自博弈训一个 RL 雷达 π_r*(vs 共同进化的干扰机池)——**S1 的最小版**;
2. 同法算 **exploitability(π_r*)**(用 BR(π_r*) 打它);
3. 对照:**exploitability(π_c) vs exploitability(π_r*)**。

## Step G3 — 熵率 storm 判据(证在风暴里,非嘴说)
- 测 best-response 干扰机诱导的信道/行为**熵率**(Thornton'23):高熵率/非平稳 → 学习有价值(storm);对角/状态持久 → 固定规则够(calm)。
- 附:头对头 **π_r* vs π_c(cross-play,双向)** 在**共同 held-out 自适应干扰机**上比 QoS/track-loss。

---

## Gate(逐 cell + bootstrap CI,不看单数)
| 判据 | 阈值 | 决定 |
|---|---|---|
| **G-storm(命门)** | **exploitability(π_c) ≫ exploitability(π_r*)**,gap 显著、95%CI 不含 0;且熵率落在"学习有价值"区 | ✅ **storm 真 + 学习吃得下** → 建 S1 冲 TAES |
| G-headtohead | π_r* 在共同 held-out 自适应干扰机上显著赢 π_c(QoS/track-loss) | 佐证学习价值 |
| G-strong | π_c 在静态干扰下近最优(QoS 高/track-loss 低) | 证经典非稻草人(前提) |
| G-null(退场) | exploitability(π_c) ≈ exploitability(π_r*)(经典已近均衡) | ❌ 无 storm → 退 IET/收 C0 基准 |

## 决策树
- **G-storm PASS** → **命题成立** → 建 **S1 完整闭环**(自博弈雷达 vs 自适应对手),守 **delta over Li 2022 TAES NFSP**:IQ 级 + **四功能 MFAR** + **多基地融合/CRLB** + **操作包线/熵率相变刻画** + **co-adaptive 欺骗 vs 完整 ECCM 栈**;主投 **TAES 一区**;并行切 **C0 基准发布 + 一个 AI slice** 投 EAAI/ESWA。
- **G-null FAIL**(经典近均衡、exploit 不动)→ 该 regime 也良态 → **退 Path A**:C0(IQ MFAR EW 基准)+ C1(多基地 CRLB)→ IET RSN(安全)/ 诚实相变表征。

## 指标 + 统计 + 诚实守则
- exploitability(Nash-gap)、熵率、QoS-under-jamming、track-loss/OSPA、cross-play 双向平均 + 共同 held-out;
- ≥5 seed,mean±95%CI(bootstrap 1e4),Welch-t/Mann-Whitney;
- **强经典是前提**(IMM-PDAF/Q-RAM/规则 ECCM,用 Stone Soup,不是 CV-KF/equal-alloc 稻草人);
- **诚实**:若经典已近均衡就是 G-null,老实退 IET,不硬凑;best-response 干扰机要训到位(否则低估 exploitability = 假 null)。

## 范围 / 时间 / 风险
- **~1-2 周**(比整套 S1 的 ~2-3 月便宜),纯裁决用;
- **R1**:best-response 干扰机没训到位 → 低估 exploitability → 假 G-null;缓解:多 seed + 收敛监控 + 熵率交叉验证;
- **R2**:强经典栈没配好(弱)→ 假 exploitable;缓解:G-strong 前置验证;
- **R3**:zero-sum 近似——QoS/track-loss 要定成雷达-干扰机近零和的清晰标量,否则 exploitability 无定义。

跑完贴回:**exploitability(π_c) vs exploitability(π_r*)(逐 seed + CI)+ 熵率 + cross-play 头对头 + π_c 静态干扰下的 QoS/track-loss**,据此判 G-storm → 建 S1 还是退 Path A。
