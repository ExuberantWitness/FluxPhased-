# 文献锚定的选靶战略 — 从"自造 toy"到"文献确立的 storm"(四轴文献综述综合)

**动机**:此前 5 次自造 toy 快速测试全部"AI 打平/输 classical",根因不是 AI 无用,而是**自造问题必落在 classical 假设成立的良态区**(平静海面筛不出船长)。本文档是四轴雷达工程文献扫描(2022-2026,TAES/TSP/TGRS/IET RSN/EAAI/ESWA/DSP)的综合,把难度和 SOTA 从文献锚定,而非脑补。

---

## 0. 压倒性收敛结论(四轴独立指向同一处)
**唯一可辩护的"AI 赢经典雷达"thesis = 自适应对手 / 非平稳 regime。** 凡环境平稳、模型已知处,经典**可证近最优**——这正是我们 5 次全输的原因,不是 bug:
- **检测**:Kelly-GLRT / AMF / **ANMF/ACE 是 UMPI(一致最优不变)+ texture-CFAR**(Kraut-Scharf 2001/2005;Conte-Lops-Ricci 1995/96);非高斯海杂波检测经典深厚,学习撞"oracle-model 经典也赢"死结 → **别碰**。
- **调度**:Q-RAM 在最优 **0.1%** 内(Ghosh/Rajkumar RTSS'04);"RL 解 NP-hard 更快"= 可反驳的 speed win,死于 Q-RAM/B&B-MCTS(Shaghaghi IET'18)→ **别碰**。
- **跟踪**:线性高斯下 KF 可证最优(我们 sensing-pilot 亲测 CV-KF 到噪声底 0.58m);只打赢单模型 KF = 稻草人。
- **机理**:经典 ECCM/滤波/调度**可证建立在平稳干扰机/固定模板/均匀杂波/已知模型假设上**;一个在 OODA 环内 best-responding 的认知对手**恰好打破这个前提**——博弈论无歧义:**对自适应对手,任何固定策略可被反演,只有均衡(自适应/混合)策略达 minimax 值。**
- **领域自己的判断**:Calatrava et al. 综述(arXiv:2503.00285, 2025)把"**自适应对抗动态 / 闭环交战**""**无标准 benchmark**"列为头号 open problem——正是我们 infra 卡的缺口。

---

## 1. 选靶表(候选 storm,文献锚定)

| # | storm | 经典为何**可证**失效 | 必打赢的强 baseline(非稻草人) | 我们 infra 契合 | storm 判据(证在风暴里) | venue |
|---|---|---|---|---|---|---|
| **S1(主)** | **闭环认知 MFAR vs co-adaptive 学习型 EW 对手**(自博弈→均衡) | 固定/规则策略被 best-responder 反演;只有均衡策略达 minimax | **Nash/fictitious-play(Li 2022 NFSP)** + 规则 ECCM + Q-RAM | **满配**:PPO 干扰机 + 联赛/自博弈 + IQ + 多基地 = 全套零件 | **exploitability / Nash gap** + **熵率(Thornton 2023)** + SINR/track-loss | **TAES 主** / IET |
| **S2** | 多基地认知资源管理保跟踪 under 协同欺骗(RGPO/VGPO/假目标) | 协同欺骗同时满足 IMM 运动模型 + PDAF 似然,滤波无统计依据拒绝(Calatrava TAES'24:PDAF 均匀杂波模型是失效点) | **IMM-PDAF + PCRLB**(=Blair-Watson benchmark 的标准解,reviewer 说不出"没比对口 ECCM")+ 前沿跟踪 + MHT | 强:多基地 Kalman + CRLB + 对手 | track-loss / OSPA vs 对手自适应性(相变包线 ΔD) | TAES / IET |
| **S3(EAAI 钥匙)** | **发布 C0 基准 + 一个 AI slice 打赢强经典** | (承 S1/S2 的一个组件,如学习干扰机策略推断 / belief-CRLB 决策) | CRLB/myopic-最优 + 强经典 optimum | C0 已 MATLAB 83/83 校验 | 公开可复现 + 显著性/消融 | **EAAI/ESWA** |

---

## 2. 把"反 toy 纪律"操作化(这次不重蹈覆辙)
1. **用数字证明在 storm 里,不靠嘴**:算 **熵率**(Thornton-Buehrer, IEEE T-RadarSys 2023:熵率预测在线学习价值,对角性预测固定规则够用)/ **exploitability / Nash gap**——把"平静海面 vs 风暴"变成可测量。
2. **打赢对的 baseline(购物清单,白纸黑字)**:
   - 抗干扰:规则 ECCM **+ 博弈论 Nash/fictitious-play**(Li 2022),**不是 Q-learning-vs-DQN**;
   - 调度/波形:myopic/greedy + **PCRLB/CRLB 分配** + B&B/Q-RAM;
   - 跟踪:**IMM/IMM-EKF + PCRLB**(不是单模型 KF);
   - 检测:ANMF/ACE(若涉及)。
   - **"如果一个固定 Nash 策略就能赢你的 RL,整个自适应故事就塌了"** → Nash baseline 必做。
3. **别碰经典可证最优区**:非高斯杂波检测、良态高斯跟踪、调好的静态调度——文献确认的死路。

---

## 3. 双交付物路线图(同一套 infra,两篇)
- **TAES 主论文(最强科学)**:S1 完整闭环自博弈 vs 自适应对手系统 + 强 baseline(Nash/IMM-PDAF/PCRLB)+ exploitability/熵率 storm 刻画 + 收敛/消融/统计。
- **EAAI/ESWA slice(Q1-IF 奖杯)**:S3 = 发布 C0 IQ-MFAR EW 基准(公开可复现,命中 EAAI scope 白字"using public data sets")+ 一个 AI 组件打赢强经典最优。**发布基准本身 = 最大 EAAI de-risker**,同时干掉稻草人质疑。

---

## 4. novelty 约束(必须守的 delta)
**Li, Jiu, Pu, Liu, Peng (2022) IEEE TAES "NFSP for Radar Antijamming Dynamic Game"(47 引)几乎就是 S1 的基座——已发表。** 我们的 delta 必须清晰:**IQ 级(Li 是抽象博弈)+ 四功能 MFAR(非仅频率捷变)+ 多基地融合 + 操作包线/熵率相变刻画 + co-adaptive 欺骗 vs 完整 ECCM 栈**。守不住 delta = 重复劳动。

---

## 5. 诚实的 venue 与概率
- **TAES 纯仿真 + EW + RL 完全 OK**(Technical Area 明列;Li'22/Xiong'23/Zhou'23 全纯仿真且高引),但**要 rigor**:强 baseline/收敛证明/硬数字;**弱 baseline 是唯一真风险**。→ S1 现实录取**高**。
- **IET RSN**:最宽松安全港(Wang'23 无强 baseline 都进),IF~1.9 非 Q1-IF。
- **ESWA**:重构成"intelligent decision/inference"slice 后最现实的 Q1-IF(DMIIRL, Feng ESWA'25 纯仿真对抗雷达认知有先例);中。
- **EAAI**:梦想 IF~7.8 但最险(整套军事仿真有 scope desk-reject 风险);**只有 S3=组件 slice + 公开 benchmark + 强经典 baseline 才行**;低-中。
- **Neurocomputing / TGRS 出局**(前者要 SAR-ATR 真实数据 DL-vs-DL;后者要真实测量数据)。

**关键澄清**:纯仿真 + 军事色彩**不是** desk-reject 主因(RDJCNN@EAAI'23、DMIIRL@ESWA'25 都是);**弱 baseline + 不公开数据**才是。我们的 CRLB/多基地 infra 恰好用来**造强经典对手**——把劣势变资产。

---

## 6. 之前的"失败"全是零件(没白费)
PPO 干扰机、联赛/自博弈、多基地融合、IQ sim、甚至 RGPO 发现——**正是 S1/S2 storm 需要的全部原料**。控制层"失败"不是终点,是在错误的良态 regime 里测了对的工具;换到文献确立的自适应对手 regime,同一套工具就是 SOTA。

---

## 7. 推荐的第一步(廉价决定性 gate,这次锚定在真 storm)
**别直接建整套。先确认我们真在风暴里**:让 co-adaptive 对手(现 PPO 干扰机)对**完整强经典栈**(Nash/fictitious-play + IMM-PDAF + 规则 ECCM),**测 exploitability / Nash gap / 熵率**:
- 强经典栈**可被 best-responder 显著 exploit**(exploitability ≫ 0,熵率落在"学习有价值"区)→ **storm 确认** → 建学习型雷达闭环(S1);
- 强经典栈就是均衡、exploit 不动 → 连这个 regime 都良态 → 老实退 IET/收 S3 基准。
**这是"cheap decisive gate first"这次用对地方**:文献锚定的 storm regime + 对的强 baseline + 可测 storm 指标。

---

## 附录 A. 完整文献舰队(~15 agent)对"经典是估计/检测层的 oracle"的确证
四轴 + 各自子调研独立、反复得出**同一结论**:**凡模型已知、环境平稳处,调好的强经典 = oracle,学习只能逼近或靠 amortized 速度赢;发表的"AI 赢经典"绝大多数是(a)打未调稻草人、(b)同生成器 in-distribution 仿真、或(c)推理提速——不是根本最优性 gap。** 这解释了我们 5 次全输:全在良态区。冒烟证据:
- **Greenberg, Yannay, Mannor, "Optimization or Architecture: How to Hack Kalman Filtering," NeurIPS 2023**:神经 KF 的优势来自**优化(用训练 loss 拟合 Q,R)而非架构**;一个 Optimized-KF 就追平神经滤波。
- **Mehrfard et al.(Mercedes-Benz)2024**:真实汽车雷达上**未调 IMM 打赢 KalmanNet**(位置 RMSE 1.08 vs 1.23m),判 KalmanNet"不适合安全攸关系统"。
- **CFARnet(Diskin/Wiesel, Signal Processing 2024)**:证 DL 检测器**渐近等价 GLRT**(只赢 ~30× 速度);朴素 DL 检测器**丢掉 CFAR 保证**本身。
- **海杂波**:唯一正面 benchmark 强 ANMF 的工作(Ovarlez 组 CVAE)最终是**与 ANMF 融合**,不是取代。
- **TBD(Davey/Rutten/Cheung 2008)**:DP-TBD/grid-Bayes/PF-TBD 检测性能**统计不可分**,6dB 都 Pd≈1、3dB 全 <0.5——经典就是强 baseline,学习买 dB 不买物理(SNR 墙)。
- **MTT(Chalmers Svensson 组 MT3/Can-DL)**:学习只在"模型未知/复杂"时赢近似的经典;模型已知则**打平或输**;DeepDA vs JPDA = 0.394 vs 0.399 是**平局**。
→ **我们 sensing-pilot 的 CV-KF 到噪声底不是运气,是定理。** 唯一 oracle 不存在的地方 = **反应式、策略性、非平稳的对手策略**(S1)。

**这次 novelty 的 delta 再确认**:Li 2022 TAES NFSP 已占"自博弈雷达 vs 干扰机"基座 → 我们必须 IQ 级 + 四功能 MFAR + 多基地 + 操作包线/熵率相变 + co-adaptive 欺骗 vs 完整 ECCM 栈。

**必打赢的强 baseline(舰队白纸黑字汇总)**:抗干扰→**Nash/fictitious-play(Li'22)**+ 规则 ECCM;跟踪→**IMM-PDAF / PMBM / PCRLB**(非单 KF);检测→**ANMF/GLRT**(非 CA-CFAR);调度→**Q-RAM / 非近视 POMDP-rollout**(非 equal-allocation);TBD→PF-TBD/DP-TBD。复现基座:**Stone Soup**(DSTL 开源,含这些经典 + OSPA/GOSPA),用它当经典 baseline 谁都说不出稻草人。
