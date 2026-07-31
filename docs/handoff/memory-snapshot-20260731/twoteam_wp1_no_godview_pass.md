---
name: twoteam-wp1-no-godview-pass
description: WP-1 (去 god-view 反 toy) 4 milestones 全 PASS 2026-07-16; §2.6 7/7 items green; 69/69 tests; ready for WP-2 盲态经典栈
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

WP-1 (反 toy / no-godview / partial observability) 完成于 2026-07-16。分支 `twoteam/bc-ppo`,基于 spec `FLUXPH_BLIND_ADVERSARIAL_SPEC.md` (commit fa485ad)。

**4 milestones all committed (513472e ← 73fa874 ← bcec72d ← 408a7c9 等):**
- M1: detection chain 替换 god-view `z=true_pos+noise`; enemy shutdown (enemy_emitting flag)
- M2: assert_no_godview (permutation test, 44/44 dims invariant); obs_dim 40→44 (fsld/search_cov/n_det); proactive-detect exposure bonus; enemy_freq masked when shutdown
- M3: beam_direction Beta(α,β) → azimuth [-π,π] 替代 beam_target∈{0,1} god-view leak; soft transition (env accepts both)
- M4: σ_meas-gated NN association (5σ_meas + 500m floor); mirror unbiased verified (mean=0, std=0 perfect symmetry)

**§2.6 全 7 项 PASS:**
- ① no-godview assert — test_no_godview.py
- ② 关机/隐藏 — test_shutdown_disappears.py (fsld grows, trace_P inflates)
- ③ 主动探→exposure — test_proactive_detect_raises_exposure.py (bonus = emit_inc + 2·events)
- ④ 虚警+关联 — test_association_under_clutter.py (P_fa=1e-3 stress, ratio 0.03)
- ⑤ IQ 干扰 — test_iq_interference.py (13 tests)
- ⑥ mirror unbiased — test_mirror_unbiased.py (50 episodes, mean/std=0)
- ⑦ NaN-free — implicit in 69/69 tests

**Why:** spec 诊断 prior env 是 god-view-lite toy (`z=true_pos+noise`, beam_target 按敌索引, 敌方永不隐藏); WP-C D1-D5 IET FLOOR 已证假胜根因。WP-1 是反 toy 第一步,no-godview + 部分可观测。承 [[twoteam-wp-c-d1235-outcome]] 之后。

**How to apply:** WP-1 done → 进入 WP-2 (盲态经典栈:CFAR/IMM-PDAF/RRM/ECCM production-quality)。WP-2 须 demonstrate "经典低干扰 kill + 高干扰失效"。WP-3 (production RL ~5e7 steps) 跟上。WP-4 cross-play 比较。

**Deferred (out of M4 scope per plan):**
- IMM-PDAF (CV+CT 2-model + Mahalanobis gate) — 当前 σ-gate 已通过 §2.6④
- hard-cut legacy beam_target env alias — assert_no_godview 已 PASS(leak 在 step() 不在 get_obs());迁移 ~10 baselines + bc_pretrain + br_trainer 大爆炸,延到 WP-3 前期

**关键技术细节(供 WP-2/3 参考):**
- coherent_processing_gain_db=20 必需 (LFM/Barker+CPI),否则单脉冲 SNR 0dB → P_detect≈0
- 队内互扰:同 ch0 → JNR 81dB,所有检测死;必须 distinct channels per radar
- 多目标 NN init bug:slot 1 会锁在 slot 0 第一次测量上(B1 未检测时);PDAF/JPDA 是真修复
- mirror RNG pattern `rand(E,1,R).expand(E,T,R)` 必须贯穿所有 stochastic step(detection Bernoulli + FA + homejam)
- FA 位置不是镜像对称的(own_centroid+极坐标偏移),但 p_fa=1e-6 默认下 FAs 罕见到无影响
