# 文献支撑:一体化协同雷达 + 多智能体RL + 精确激光击杀

**日期:** 2026-06-16
**用途:** §5(去真值 → 真 MARL)的方法学引用骨架。所有条目经 IEEE/arXiv/CrossRef 核验;不确定项已标注。

---

## 1. 多基地/分布式雷达定位(S2 空间融合的理论基础)

- **Godrich, Haimovich, Blum, "Target Localization Accuracy Gain in MIMO Radar-Based Systems," IEEE Trans. IT 56(10), 2010**(arXiv:0809.4058). **核心定理**:分布式 MIMO 雷达定位 CRLB ∝ 1/(有效带宽),优化传感器位置使 CRLB 下降 ≈ #Tx×#Rx 倍。→ **我们"6cm 融合 vs 10m 单雷达"的硬理论锚点**。
- **Haimovich, Blum, Cimini, "MIMO Radar with Widely Separated Antennas," IEEE SP Mag 25(1), 2008**(DOI 10.1109/MSP.2008.4408448)。广分布天线的空间分集 → 分辨率"远超波形支持"。架构层锚点。
- **Mutambara, "Decentralized Estimation and Control for Multisensor Systems," CRC 1998** + **Maybeck Vol.1 (1979) §5.7 逆协方差形式**。→ **我们用的信息滤波融合 Σ_fused⁻¹=ΣΣ_k⁻¹ 的精确出处**。
- **Durrant-Whyte & Henderson, "Multisensor Data Fusion," Springer Handbook of Robotics ch.25, 2008**。多传感器融合=信息矩阵相加。

## 2. GDOP / 几何(解释 6cm↔10m 的几何依赖)

- **Bishop, Fidan, Anderson, Doğançay, Pathirana, "Optimality Analysis of Sensor-Target Localization Geometries," Automatica 46(3), 2010**。FIM 证明:正交几何最小化 CRLB,共线退化最大化。→ **我们 90°→6cm、近共线→10m 的直接解释**。
- **Nguyen & Doğançay, "Optimal Geometry Analysis for Multistatic TOA Localization," IEEE Trans. SP 64(16), 2016**(正确 DOI 10.1109/TSP.2016.2566611,网上有错 DOI)。**最贴合我们的单篇**:多基地 TOA 最优几何在 ±60°,共线最差。
- **Doğançay & Hmam, "Optimal Angular Sensor Separation for AOA Localization," Signal Processing 88(5), 2008**。

## 3. 多帧 Kalman 跟踪(S2-track 的理论基础)

- **Bar-Shalom, Li, Kirubarajan, "Estimation with Applications to Tracking and Navigation," Wiley 2001**。稳态 Riccati / α-β 滤波;**位置误差floor 由 Q/R 比决定**。
- **Kalata, "The Tracking Index," IEEE Trans. AES 20(2), 1984**。跟踪指数 Λ=σ_w·T²/σ_v 给出"过程噪声 vs 测量噪声"的闭式权衡;**慢目标(小 Λ)→ 稳态误差远低于单帧测量噪声**。⚠️ 注意:**严格 1/√N 只在静止目标成立;运动目标 floor 在 Q 决定的稳态值**——文中要如实陈述。
- **★ Nardone & Aidala, "Observability Criteria for Bearings-Only Target Motion Analysis," IEEE Trans. AES 17(2), 1981**(DOI 10.1109/TAES.1981.309141)。**关键观测性定理:观测器必须机动才能观测目标**。→ **这是"雷达移动(20m/s)打破共线退化"的严格依据——正是 S2-track 能突破融合 floor 的根本原因**。
- **Mallick & Bar-Shalom, GMTI 跟踪(Proc. SPIE 4728, 2002)**。两近正交 GMTI 传感器的椭圆误差协方差相交 → 大幅提精度;**移动单传感器在时间上扫过旋转的误差椭圆,KF 逐帧"求交" = 我们的时间积分机制**。
- **Sherman & Barton, "Monopulse Principles and Techniques"**;**Track-Before-Detect 文献**:时间积分实现亚分辨单元/亚波束定位。
- **Blom & Bar-Shalom, "The IMM Algorithm," IEEE Trans. AC 33(8), 1988**;**VS-IMM 地面目标(Kirubarajan et al. 2000)**——目标机动时的保险。

## 4. ISAC / 一体化网络化感知("侦查通信探测干扰一体化"叙事)

- **Liu, Cui, Masouros, Xu, Han, Eldar, Buzzi, "Integrated Sensing and Communications...," IEEE JSAC 40(6), 2022**(DOI 10.1109/JSAC.2022.3156632)。ISAC 领域权威综述;"integration gain + coordination gain"。⚠️ 作者非"Zhang/Heath"。
- **★ Chernyak, "Fundamentals of Multisite Radar Systems," Gordon & Breach 1998**。**最强"一体化即优势"陈述**:MSRS 优于单雷达**也优于未整合的雷达集合**。
- **Zhang et al., "Perceptive Mobile Networks," IEEE VT Mag 16(2), 2021**。网络化 JCAS。
- **Griffiths & Farina, "Multistatic and Networked Radar," IEEE RadarConf 2021**。覆盖/抗干扰/精度优势。
- 协同 ISAC 标度律 ln²N(Meng & Masouros, arXiv:2403.20228, 2024,**预印本**);通信速率→融合精度耦合(arXiv:2408.03174 等,**2024-25 预印本**)——证明"BPSK 链路是感知链的一部分"。
- 雷达+干扰一体化波形(多篇 2023-25,**新兴、有性能折衷**)——四合一(侦查+通信+探测+干扰)是研究愿景,非已证成熟能力。

## 5. RL 控制认知雷达 + 定向能(系统层)

- **Haykin, "Cognitive Radar: A Way of the Future," IEEE SP Mag 23(1), 2006**。感知-动作闭环。
- **★ Xiong, Zhang, Cui, Wang, Kong, "Coalition Game of Radar Network for Multitarget Tracking via Model-Based MARL," IEEE Trans. AES 59(3), 2023**(DOI 10.1109/TAES.2022.3208865)。**最接近我们"协同相控阵=MARL agent"架构的已发表工作**。
- **Thornton et al., "Deep RL Control for Radar Detection and Tracking...," IEEE TCCN 6(4), 2020**(arXiv:2006.13173)。RL 选波形/带宽提跟踪质量。
- **Jiang, Ren, Wang, MADRL 抗干扰认知雷达, Digital Signal Processing 135, 2023**。MARL 抗干扰保跟踪。
- 定向能 ATP:**NPS/Agrawal HEL 光束指向测试床(SPIE 7587, 2010)**;DTIC ADA601166/ADA554679(光束抖动控制,微弧度级);CRS R46925/R44175 + HELIOS。雷达提供粗跟踪+捕获,FSM 闭最后微弧度。⚠️ 无同行评审"RL 激光火控"文献。

## 核心:本工作的新颖性(文献空白)

各支柱**单独**都有扎实文献,但**没有一篇**展示完整端到端链:
> **网络化协同雷达(MARL 控制)→ BPSK 融合链路 → 亚米定位 → 定向能(激光)火控/指挥决策**,在一个学习闭环里。

- 雷达侧 RL([1]-[12])与激光侧控制([13]-[15])是**两条不相交的文献**;
- **MARL 协同雷达 + 定向能火控耦合 = 本工作的新颖点**,应作为"基于已证组件的系统级提议"呈现,而非复现已有结果。

## 论文优先引用(全核验,可直接进参考文献)

Godrich/Haimovich/Blum 2010 · Bishop et al. 2010 · Nguyen & Doğançay 2016 · Bar-Shalom/Li/Kirubarajan 2001 · Kalata 1984 · **Nardone & Aidala 1981** · Blom & Bar-Shalom 1988 · Liu et al. JSAC 2022 · Chernyak 1998 · Haykin 2006 · Xiong et al. 2023 · Thornton et al. 2020。

(标 ⚠️/预印本者引用前复核 vol/pp/作者。)
