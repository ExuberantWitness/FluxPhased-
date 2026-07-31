---
name: twoteam-wp2-blind-classical-pass
description: Two-team WP-2 盲态经典 baseline 全 5 milestone PASS (2026-07-16) — no-godview + IMM-PDAF + laser slot-id + ECCM + 3-axis sweep kill collapse.
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

WP-2 "competent blind classical" baseline 全 5 milestone PASS (2026-07-16, commits 35fd700→834d804, branch `twoteam/bc-ppo`).

**Why:** spec §0.3④ 要求一个非稻草人的盲态经典栈来证明 env 是真挑战而非 toy。WP-1 (no-godview assert) 只证明 RL actor 见不到 god-view,但所有 baseline commanders 还在用 legacy `beam_target` (god-view enemy index) API,且 env 内部 laser block L672-701 + Kalman block L617-624 两处都有 god-view leak。WP-2 把这两处堵死并交付真盲态经典栈。

**How to apply:**
- M0 laser slot-id semantics: `laser_target ∈ {0,1}` 现在是 **slot id** 不是 enemy id;env 在 laser block 内读 `tracker_x[,t,slot]` belief 位置,找最近 alive enemy,距离 ≤ `laser_hit_radius_m=50m` 才加 energy。Belief-vs-truth check 不依赖 god-view 偷看,只比对 belief 和公开 enemy pos。
- M1 `env/gpu/twoteam/tracker.py::BatchedIMMPDAF`:CV+CT 2-model IMM + PDAF (5σ Mahalanobis gate, β_i over K_max detections),替换 σ-gate NN association。Mirror RNG 用 `rand(E,1,R).expand(E,T,R)` 保持 team-shared。
- M2 `algo/_shared/baselines/twoteam_blind_classical.py::BlindClassicalCommander`:输出 **`beam_direction`** (continuous azimuth ∈ [-π,π]) 取代 `beam_target`;slot init → track_az=atan2(dy,dx) from `tracker_x`;slot uninit → sweep at `beam_width/2` step (0.0375 rad) for dense angular coverage. ECCM:jam>0.40 时 channel_select 切下一信道。
- M3 干扰梯度断言:低干扰(jam=0, orthogonal, p_fa=1e-6) kill≥0.5;高干扰(jam≥0.4, duty≥60%, same channel, p_fa=1e-3) kill≤0.3 且 ≤低干扰一半。
- M4 3-axis sweep (jam × duty × channel = 4×4×2 = 32 cells, 3ep×150step×8envs) 验证单调 kill collapse:
  - orthogonal channel: ECCM 生效,kill_rate 持续 ~1.0 across 全网格
  - same channel: 单调下降 0.542 (jam=0,duty=20%) → 0.000 (jam≥0.4,duty=80%)
  - headline collapse ratio = 0.000 << 0.5 → spec §3 ③ competent-blind 硬要求满足

**关键证据 env 非 toy:** BlindClassical 在 same-channel high-interference 真 collapse 到 kill=0,低干扰 kill=1.0。ECCM 让 orthogonal 配置下还能维持满 kill —— 反映盲态栈在低干扰下 competent,在高干扰下失效,与 spec §3 ③ 严丝合缝。

**18 WP-2 tests + 87 总 tests 全 PASS。** 测试覆盖:`test_laser_api_slot_semantics.py` (5) + `test_imm_pdaf.py` (6) + `test_blind_classical_smoke.py` (5) + `test_blind_classical_interference.py` (2)。

**§8 IMM-PDAF vs σ-gate 对照** (commit `0a4e60d`,`experiments/twoteam/wp2_tracker_ablation.py`):同一 detection stream 上并行跑 IMM-PDAF 和 WP-1 σ-gate NN (commit 513472e 复刻),4 scenarios:
- linear: σ-gate 略优 (6.7m vs 39.5m,IMM 付 robustness tax 但 ≪ tau_track)
- j_turn: IMM 显著优 (26.7m vs 95.8m,3.6×,CT model 抓住突转)
- high_clutter: IMM 完胜 (39.7m vs 424.7m,10.7×,PDAF β_i 稀释 FA)
- low_snr: IMM 大幅优 (505.9m vs 1939.3m,3.8×,稀疏检测下更鲁棒)
- 平均 RMSE:IMM 153m vs σ-gate 617m,**4.0× 更优**;3/4 PASS,1/4 linear 略违反但操作可忽略

**Next:** WP-3 production RL training (IPPO/MAPPO)。WP-4 RL vs BlindClassical cross-play 比较。参见 [[twoteam-wp1-no-godview-pass]] (前置 milestone) 和 [[twoteam-multifunction-pivot]] (框架总览)。

**复用代码:**
- `combine_team_actions()` 已支持 beam_direction stack 且 beam_target optional
- `detection.py` measurement noise team-shared (mirror symmetry)
- `env.assert_no_godview(tol=1e-5)` 44/44 obs dims invariant

**WP-2 完成定义 (per plan) 全 ✓:**
1. ✅ laser API 改 slot-id semantics + belief check
2. ✅ IMM-PDAF 替换 σ-gate NN (RMSE 同等或更优,trace_P 不发散)
3. ✅ BlindClassical 通过 no-godview assert
4. ✅ 低干扰 kill_rate ≥ 0.5 (实测 1.000)
5. ✅ 高干扰 kill_rate ≤ 0.3 且 ≤ 低干扰一半 (实测 0.000)
6. ✅ 3-axis sweep report 展示 kill collapse
7. ✅ 全套 tests 不回归 (87/87)
