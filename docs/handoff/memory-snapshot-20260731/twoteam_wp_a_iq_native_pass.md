---
name: twoteam-wp-a-iq-native-pass
description: "WP-A PASS 2026-07-15 — scalar jam_mul 已删,IQ 原生干扰(含队内互扰)接入 env;5/5 物理验证全 PASS,26/26 单测全 PASS。"
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

2026-07-15 FLUXPH_MARL WP-A 完工:IQ 原生 co-channel 干扰物理替换 scalar `jam_mul`,作为 env 的物理事实而非调参旋钮。

**Why**:Bet B 7 次「经典赢/近 Nash」全在错的「平静海面」regime 测得(`jam_mul` 把 IQ 级密集多辐射源干扰抽象掉了,只聚合敌方 jam,**完全不含队内互扰**)。WP-A 让「两雷达同频/同向会互相致盲」成为 env 的物理事实,这是协同问题的物理来源。

**5 条物理验证**(全 PASS):
- ① 敌扰生效(JNR ≥10 dB,σ_inflation ≥3×)
- ② 队内他扰生效(同信道 JNR >0 dB,差信道/hop=8 衰减 >4×)
- ③ 4 源共道叠加线性(rel err <1%,NOT dB-sum)
- ④ 镜像自博弈无偏(reward asymmetry mean≈0)
- ⑤ 训练健康(NaN-free + priv[:,4] assert 不触发)

**3 项已确认的架构决策**(详见 [[twoteam-multifunction-pivot]] + plan `tidy-wiggling-codd.md`):
- Subarray→孔径耦合:`D_eff = D·sqrt(f_emit)`
- freq_hop_rate 语义:扩到自己 channel 内 hop 个子信道,共道污染 1/hop 衰减
- Legacy scalar jam_mul:**WP-A 末尾已删**(`interference_mode` flag + `jam_gain` ctor 参数全去掉)

**How to apply**:WP-A 已完工 → 进 WP-B(在高干扰 regime 下记录经典估计崩)。WP-B 标定常数时调 `P_per_subarray_W` / `aperture_D_m` / `n_subarrays` 这些物理常数,**不要**再加 scalar `jam_gain`/`jam_mul` 这种历史抽象。下游戏/学习基线时记得 obs_dim 已从 36 → 40(4 个 freq-channel slot),AC 网络 default 已同步。

镜像无偏的关键:`reset()` 中 team_B 的 freq/beam_az 必须严格 = team_A 镜像(`f_B=f_A`, `beam_az_B = π + beam_az_A`);单测断言 jnr 矩阵镜像对称在 1% relative 内(sinc² -30dB floor 引入的数值 floor artifact,非物理对称性违反)。
