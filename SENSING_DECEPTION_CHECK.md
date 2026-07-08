# 欺骗干扰廉价决定性检验(给 PRO6000 agent)— Stage-A:欺骗是否真打破强经典?

**关联**:`EXPLORATION_SENSING_TRACKER.md` · `SENSING_TRACKER_PILOT_CHECKLIST.md` · `experiments/sensing_pilot/STEP2_FINDING.md`
**背景**:Step1/2 已证伪原前提——"L3 track QoS→0.07"是零初值 burn-in 假象;proper CV-KF @ L3 = **0.58m(已到量测噪声底)**。根因:`jam_mul` **只放大高斯 σ = KF 可证最优的正是这种情形** → 学习无 gap。**结论:高斯抬噪这条路死了,调旋钮(单雷达/加机动/加 jam_gain)都是造假胜,不做。**
**本检验的唯一问题**:换成**非高斯欺骗干扰**(现 sim 缺此物理),经典 KF **不再可证最优**——那么**强经典(robust-KF / PDAF / 粒子滤波)是否被真打破?** 破 → 有诚实 EAAI gap(进 Stage-B 建学习跟踪器);连 PF 都吃得下 → 该域全域良态 → **退 Path A,不再赌。**

> **为什么这是诚实的、不是 p-hack**:RGPO(距离波门拖引)/ 假目标 / glint 是**真实 EW 手段**,现高斯-only sim **恰好把这个物理略掉了** → 加它是**补全缺失物理**,不是调难度旋钮。且它**破坏 KF 的最优性假设本身**(高斯→多模态/非高斯),不是调 KF 的旋钮 → 与 A1/A2/A3 性质完全不同。

---

## 两阶段门(本次只做 Stage-A,~半天)
- **Stage-A(本检验)**:强经典在欺骗下**是否被打破**?——纯经典滤波器,**不训任何 NN**,CPU 可跑。**先答"有没有 room"再决定是否投入建学习跟踪器。**
- **Stage-B(仅 Gate-A PASS 后,+1-2 天)**:学习跟踪器是否赢强经典?——这才是真 EAAI 门,Stage-A 不过就永远不做。

---

## Step D1 — 欺骗量测模型(扩 `gen_tracks.py`,加开关不动原高斯管线)
在原 `add_sensing_noise`(高斯热噪声保留)之上叠加**逐 scan 欺骗**,新增难度档 `{clean, RGPO, FT, RGPO+FT}`:

**RGPO(距离波门拖引)** — 单条**有偏走离**的诱饵量测:
```
per episode: 以 p_rgpo 触发,随机 onset t0,walk_rate(m/step),max_offset(m)
for t < t0:            z = 真值 + 高斯热噪声                      # 未捕获
for t0 ≤ t < t0+T_walk: z_range = 真range + walk_rate·(t−t0)      # 波门被拖走(真回波被压制)
for t ≥ t0+T_walk:     诱饵 drop,z 回到真值 + 热噪声              # 目标重现(制造跳变)
```
**假目标(FT)** — 关联多模态:
```
per scan: 以 p_ft 触发,注入 k 个假量测(在 gate 区内 plausible 偏移)
          与真量测并存(或按 P_detect 替换)→ 滤波器须做数据关联
```
- 参数取**物理合理**范围(walk_rate、max_offset ~ gate 尺度;k=1-3;p_rgpo/p_ft ∈ [0.3,0.7]),记录在 config;
- 导出 `experiments/sensing_pilot/data/{clean,rgpo,ft,rgpo_ft}.pt`(x_t, z_t 含欺骗, deception_meta)。

## Step D2 — 强经典 suite(公平配强,**防稻草人的命门**)
⚠️ **绝不能只跑 CV-KF**(它被欺骗打破是废话、无意义)。**必须**跑这套,且各给**合理固定假设、不喂 episode 级 oracle 欺骗模型**:
| 经典 | 应对欺骗的机制 | 公平假设(非 oracle) |
|---|---|---|
| CV-KF | 参照(会被打破,证欺骗有效) | 固定 Q(Step2 best-tuned) |
| **robust-KF** | Huber loss + 3σ 门控**剔除离群量测** | 已知热噪声 σ、门控阈 |
| **PDAF** | 概率数据关联(软加权多量测) | 已知虚警密度、gate 概率 |
| **bootstrap PF** | 多模态后验(粒子群跟真+诱饵两支) | 500-1000 粒子、已知热噪声 likelihood、**不知**诱饵轨迹 |
| (有余力)IMM-PDAF | 机动+关联 | — |
- **公平性核心**:经典拿"合理但非 oracle"的假设(知道有欺骗这回事 + 热噪声统计,但**不知道**这一集诱饵何时起、往哪走);**若给 oracle 欺骗模型 PF 就最优、学习也赢不了**——那不是现实、也不是我们的 claim。

## Step D3 — 指标 + Gate-A
- **track RMSE**(vs 真值)per 滤波器 per 欺骗档;
- **track-loss 率** = RMSE 超发散阈的 episode 占比(= RGPO 捕获成功率);
- **room headroom** = (最强经典 欺骗下 RMSE) − (clean 噪声底 0.58m)。
- **sanity**:clean 档所有滤波器 RMSE 应 ≈ Step2(0.58m 附近),证管线没改坏。

**Gate-A(决策,逐档+bootstrap CI)**:
| 结果 | 结论 |
|---|---|
| **最强经典(robust-KF/PDAF/PF 里最好的)在欺骗下 RMSE ≫ 噪声底 或 track-loss 率显著>0** | ✅ **有诚实 room** → 进 Stage-B(建 KalmanNet/GRU,vs 这套强经典,真 EAAI 门) |
| **最强经典在欺骗下仍 ≈ 噪声底、track-loss≈0** | ❌ **无 room**(PF 吃得下欺骗)→ **退 Path A,不再赌** |
| **只有 CV-KF 破、robust-KF/PF 都稳** | ❌ 同上(CV-KF 破是废话)→ 退 Path A |

## 决策树
- **Gate-A PASS** → Stage-B:学习跟踪器(KalmanNet 式,训在**自适应欺骗干扰机的策略分布**上)vs **合理假设的强经典**;赢(95%CI)= "自适应/未知欺骗下学习跟踪器打破经典的模型失配" = **诚实 EAAI 故事**(anchor:KalmanNet Revach'22 + 认知抗干扰跟踪 + RGPO/deception ECM 文献);
- **Gate-A FAIL** → **退 Path A**:该域即便加欺骗物理、强经典(PF)仍鲁棒 → 传感本就良态;把 C0(IQ 级 MFAR EW 基准)+ C1(多基地 CRLB 亚米定位)+ 诚实相变表征收成 **IEEE TAES / TGRS 一区传感论文**。EAAI 放弃,不硬凑。

## 纪律 / 诚实守则(这一路 4 次踩坑换来的)
1. **强经典 = PF/PDAF/robust-KF,不是 CV-KF 靶**;经典给合理假设、不喂 oracle 欺骗模型;
2. **逐档 + bootstrap CI**;clean 档 sanity 必对上 0.58m;
3. **只有 naive CV-KF 破 = FAIL**(废话胜不算);
4. **Stage-A 就能廉价枪毙整个想法**(PF 吃得下 → 退,别再投 Stage-B);
5. **诚实**:欺骗是补全真实 EW 物理(现 sim 缺),不是调难度让经典输——若审稿觉得像 p-hack,就是没通过 Gate-A 该退的信号。

跑完贴回:**逐档 {CV-KF, robust-KF, PDAF, PF} 的 RMSE + track-loss 率(mean+CI)+ clean sanity + room headroom**,据此判 Gate-A → 进 Stage-B 还是退 Path A。
