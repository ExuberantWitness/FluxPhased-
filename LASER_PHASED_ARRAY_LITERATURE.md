# 文献支撑:相控阵雷达 — 物理建模 / 控制 / 阵列处理

**日期:** 2026-06-16
**用途:** 为 25×25 X 波段相控阵的物理模型、RL 控制、以及"侦查通信探测干扰一体化"各支柱提供核验引用。
所有条目经 IEEE/Wiley/Artech/Springer 元数据核验;页码等不确定项已标注。

---

## 1. 相控阵物理(直接支撑我们的 radar_sim 模型)

我们的参数:25×25 阵元、X 波段 10GHz(λ=30mm)、0.5λ 间距(孔径 D≈0.375m)、带宽 200MHz。文献给出的公式与我们实测/推导**逐项吻合**:

| 量 | 公式 | 我们的值 | 出处 |
|---|---|---|---|
| 波束宽度 | θ_B ≈ 0.886λ/(N·d·cosθ₀) | 0.886/12.5 ≈ **4.06°** | Balanis 2016;Mailloux 2018 |
| 无栅瓣条件 | d ≤ λ/(1+|sinθ|) → 全扫 d≤λ/2 | 0.5λ ✅ | Mailloux 2018 |
| 横向(交叉距离)分辨 | Δx ≈ R·θ_B | 1km 处 ≈ **71m** | Balanis;Skolnik Radar Hbk 2008 |
| **单脉冲测角精度** | σ_θ ≈ θ_B/(k_m·√(2·SNR)),k_m≈1.6 | 20dB → ~0.18°→3km 处 ~9m | **Sherman & Barton 2011** |
| 距离分辨 | ΔR = c/2B | **0.75m** | Richards 2010 |
| 距离精度 | σ_R ≈ ΔR/√(2·SNR) | 20dB → **~5cm** | Richards 2010;Skolnik Radar Hbk |

**核心引用**(全核验):
- **Balanis, C.A. _Antenna Theory: Analysis and Design_, 4th ed., Wiley, 2016**(ISBN 978-1-118-64206-1)— 阵因子、0.886 波束常数。
- **Mailloux, R.J. _Phased Array Antenna Handbook_, 3rd ed., Artech, 2018**(978-1-63081-029-0)— 栅瓣、扫描损失、相移量化。
- **Sherman, S.M. & Barton, D.K. _Monopulse Principles and Techniques_, 2nd ed., Artech, 2011** — 我们用的 σ_θ≈θ_B/(k_m√(2SNR)) 出处。
- **Richards, M.A. et al. _Principles of Modern Radar: Basic Principles_, SciTech, 2010** — ΔR=c/2B、CRLB 测距精度。
- **Skolnik, M.I. _Introduction to Radar Systems_, 3rd ed., 2001 / _Radar Handbook_, 3rd ed., 2008**。

> **这直接论证了我们一路的核心物理**:距离向 cm 级(带宽决定,与频率无关)、横向几十米(衍射限 λ/D)、单脉冲亚波束。横向墙是衍射物理,不是 bug。

---

## 2. 相控阵控制 / 多功能资源管理(支撑 RL 控制 radar)

我们用 RL 调度波束/驻留、设自适应重访——这正是经典 RRM 问题的学习化:

- **★ van Keuk, G. & Blackman, S.S. "On Phased-Array Radar Tracking and Parameter Control," IEEE Trans. AES 29(1):186-194, 1993**(DOI 10.1109/7.249124)— **最经典**:联合优化波束调度 + 雷达参数 vs 跟踪负载,track-sharpness σ_θ 控制律 → 自适应重访。**这就是我们 RL 智能体在学的 reward/cost 结构,也是"机动捕获自适应跟踪"的经典出处。**
- **Charlish, A. et al. "The Development From Adaptive to Cognitive Radar Resource Management," IEEE AES Mag 35(6):8-19, 2020** — adaptive→Q-RAM(QoS)→cognitive 的演进,**把 RL 定位为认知 RRM 的终点**。论证"为什么用 RL"。
- **Hero, Castañón, Cochran, Kastella (eds). _Foundations and Applications of Sensor Management_, Springer, 2008** — 把传感器管理形式化为 POMDP/随机控制,**论证 RL formulation 的合法性**。
- **Moo, P.W. & Ding, Z. _Adaptive Radar Resource Management_, Academic Press, 2015**(978-0-12-802902-2)— 问题形式化(任务集/时间-能量预算/调度指标)= 我们的 MDP 状态/动作/约束。
- **Moo & Ding. "Coordinated radar resource management for networked phased array radars," IET RSN 9(8):1009-1020, 2015**(DOI 10.1049/iet-rsn.2013.0368)— **网络化多雷达协同 RRM**,我们多雷达协同的经典对照。
- **Daeipour, Bar-Shalom, Li. "Adaptive beam pointing control... using an IMM estimator," ACC 1994**;**Narykov, Krasnov, Yarovoy. "...multiple phased array radars for target tracking," FUSION 2013** — IMM 自适应重访 + 多雷达选择/交接(我们 Kalman 跟踪 + 多基地的经典 baseline)。

---

## 3. 数字波束形成 / 自适应抗干扰 / STAP / MIMO(支撑"探测+干扰"一体化)

抗干扰(干扰支柱)与精细测角的阵列处理基础:

**自适应置零(抗干扰)— R⁻¹ 加权:**
- **Capon, J. Proc. IEEE 57(8), 1969**(DOI 10.1109/PROC.1969.7278)— MVDR/Capon 波束形成器,w=R⁻¹a/(aᴴR⁻¹a)。
- **Applebaum, S.P. IEEE T-AP 24(5):585-598, 1976** — Howells-Applebaum 最大 SINR 自适应阵 + 旁瓣对消器,**抗干扰阵的经典**。
- **Frost, O.L. Proc. IEEE 60(8), 1972** — LCMV/Frost(多约束,可预置已知干扰方向零点)。
- **Reed, Mallett, Brennan. IEEE T-AES 10(6):853-863, 1974** — **SMI / RMB 规则**:K≈2N 训练样本 → 3dB SINR 损失。决定抗干扰所需样本预算。
- **Van Trees, H.L. _Optimum Array Processing_, Wiley, 2002** — 所有自适应置零波束形成器的权威推导 + 对角加载鲁棒化。

**STAP(联合抑制干扰+杂波):**
- **Ward, J. "Space-Time Adaptive Processing for Airborne Radar," MIT LL TR-1015, 1994** — STAP 权威教程,w=R⁻¹v。
- **Guerci, J.R. _STAP for Radar_, 2nd ed., Artech, 2014**;**Klemm, R. _Principles of STAP_, 3rd ed., IET, 2006**。

**数字阵列 + MIMO(精细测角的路径):**
- **Herd & Conway. "The Evolution to Modern Phased Array Architectures," Proc. IEEE 104(3):519-529, 2016**;**Talisa et al. "Benefits of Digital Phased Array Radars," Proc. IEEE 104(3):530-543, 2016** — 元级数字化 → 自适应干扰对消 + 测角精度优势。
- **★ Bliss & Forsythe. "MIMO Radar and Imaging: Degrees of Freedom and Resolution," Asilomar 2003**(DOI 10.1109/ACSSC.2003.1291865)— 正交波形 → N_tx·N_rx **虚拟孔径**,角分辨超物理接收孔径。**这是另一条把横向墙打下来的路(虚拟孔径 vs 我们用的多基地融合)。**
- **Li, J. & Stoica, P. (eds). _MIMO Radar Signal Processing_, Wiley-IEEE, 2009**。

---

## 4. 连到我们的工作:横向墙的三条破解路(都有文献)

我们撞到的横向几十米墙是衍射物理(§1)。把它打到亚米/0.2m 有三条独立、互补的路,**我们已实现/验证了两条**:

| 路径 | 机制 | 文献 | 我们的状态 |
|---|---|---|---|
| **多基地距离三角融合** | 多雷达 cm 级距离测量从不同角度求交 | Godrich/Haimovich(MIMO 定位 CRLB)、信息滤波 | ✅ 已实现(单测 6cm) |
| **多帧 Kalman 跟踪 + 机动捕获** | 时间积分 + 移动雷达扫掠几何 | van Keuk & Blackman、Nardone-Aidala、Bar-Shalom | ✅ 已实现(单测 8cm) |
| **MIMO 虚拟孔径** | 正交波形合成 N_tx·N_rx 虚拟阵 | Bliss-Forsythe、Li-Stoica | ⬜ 未做(第三条备选) |

并行的抗干扰能力(§3 自适应置零 + STAP)是"干扰"支柱,可作为对抗环境鲁棒性的下一步。

---

## 论文优先引用(全核验,可直接进参考)

**物理:** Balanis 2016 · Mailloux 2018 · Sherman & Barton 2011 · Richards 2010 · Skolnik 2001/2008。
**控制/RRM:** van Keuk & Blackman 1993 · Charlish et al. 2020 · Hero et al. 2008 · Moo & Ding 2015(书+IET)。
**阵列处理/抗干扰:** Capon 1969 · Applebaum 1976 · Reed-Mallett-Brennan 1974 · Van Trees 2002 · Ward 1994 · Bliss-Forsythe 2003 · Li-Stoica 2009 · Herd/Talisa 2016。

(页码标注"待核"者引用前复核。)
