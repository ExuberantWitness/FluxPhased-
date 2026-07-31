---
name: twoteam-wp3-m0123-pass
description: Two-team WP-3 production RL pipeline 4/4 milestone PASS (2026-07-17) — 代码完成 + 109/109 tests。但 production RL 训练 4 次 smoke 全 FAIL (Δ_kills 集中在 -0.85/-0.15)。详见 [[twoteam-wp3-production-smoke-fail]]。
metadata: 
  node_type: memory
  type: project
  originSessionId: bff8f7dd-12bf-41ff-9620-849bc96406c2
---

WP-3 production RL pipeline 4/4 milestone PASS (2026-07-17). 全套代码 + 22 verification tests + 87 existing tests = 109/109 全 PASS。

**Why:** spec §4 要求 production 规模 RL MARL，与 BlindClassical 完全同样的盲态(obs/action)，CTDE PPO/MAPPO，PFSP 自博弈 league，~5e7 steps / 1000+ iter。WP-2 闭环后 RL actor 还在用 legacy `beam_target` (god-view)，buffer 不存 `beam_direction` (PPO 梯度静默失效)，BC teacher 是 StrongRule (god-view)，league pool 不含 BlindClassical。

**How to apply:**
- M0 `algo/_shared/pilot/twoteam/commander_actor_critic.py`: 删 `beam_target_head`，加 `detect_mlp` (DeepSets: Linear(5,32)→Tanh→Linear(32,32)→Tanh)，加 `_detect_embedding`/`_trunk_input` (mean-pool over K)。AC forward/evaluate_actions/get_action_for_env 全部新签名 `(obs, detect_list, privileged)`。trunk 输入 44→76 (44 obs + 32 detect_emb)。env 加 `get_detect_list()` 返回 `[E,T,K_max,5]` (z_x, z_y, snr_db, is_fa, mask)。
- M1 `algo/_shared/pilot/twoteam/br_trainer.py::_RolloutBuffer`: 加 `beam_direction[H,E,R]` + `detect_list[H,E,K_max,5]`，删 `beam_target`。`collect_rollout` 调 `env.get_detect_list()[:,learning_team]` 喂 AC。`update()` 用 cosine anneal `entropy_coef_max→entropy_coef_min` (默认 0.01→0.001)。actor param group 加 `beam_direction_head` + `detect_mlp`，去 `beam_target_head`。
- M2 `algo/_shared/pilot/twoteam/bc_pretrain.py`: teacher `StrongRuleCommander` → `BlindClassicalCommander`，buffer `beam_target` → `beam_direction`，加 `detect_list`。`_bc_loss` 新签名 `_bc_loss(obs_b, detect_b, priv_b, action_b)`。
- M3 `algo/_shared/pilot/twoteam/run_wp2_league.py`: `initialize_pool` 加 `BlindClassicalCommander` 作为 pool 成员 (与 StrongRule 并列，共 13 records = rule×2 + 7 extreme + 3 exploit + 1 BC)。`ACCommander.get_action` 喂 detect_list。每 100 iter 健康监控: entropy < 0.3 / |policy_loss| < 1e-4 (PID 1296303 signature) / pool ema_var < 0.05 → warning。CLI 加 `--blind-teacher/--strong-rule-teacher` toggle + `--ppo-entropy-coef-min`。
- M4 `experiments/twoteam/wp3_train.py`: 薄 wrapper around `run_wp2_league.py`，默认 production 参数 (1000 iter × 300 horizon × 256 envs = 7.65e7 steps ≥ 5e7 spec)。Hard guard: `--ckpt-dir` 和 `--report` 拒 /tmp (spec §4.3)。默认 ckpt 路径 `checkpoints/blind/wp3_<ts>/`。

**关键测试 (22 新 + 87 旧 = 109/109 全 PASS):**
- `test_actor_critic_blind.py` (6): no_beam_target_head / has_beam_direction_head / has_detect_mlp / detection_encoder_permutation_invariant (DeepSets 不变性) / no_godview (44/44 dims invariant) / runs_full_episode (200 step NaN-free)
- `test_br_trainer_beam_direction.py` (7): buffer_has_beam_direction / buffer_has_detect_list / buffer_no_beam_target / ppo_update_changes_beam_direction_head_weights (梯度真回流) / ppo_update_changes_detect_mlp_weights / entropy_coef_anneal / no_nan_after_5_updates
- `test_bc_pretrain_blind.py` (5): bc_uses_blind_classical_teacher / buffer_has_beam_direction_no_beam_target / buffer_has_detect_list / bc_loss_decreases
- `test_league_pool_blind.py` (4): pool_includes_blind_classical / pool_size_is_13 / pool_blind_classical_produces_beam_direction / make_factory_commander_blind

**Spec §6 反 toy checklist 全 PASS (RL AC):**
1. no-godview assert: 44/44 dims invariant
2. enemy shutdown support: env.enemy_emitting 属性存在
3. IMM-PDAF tracker: BatchedIMMPDAF
4. IQ interference physics: IqInterference
5. classical is blind competent: BlindClassicalCommander exists
6. priv[:,4] normalized: max=0.250 << 100
7. AC no beam_target_head: True
8. AC has detect_mlp (DeepSets): True
9. AC has beam_direction_head (blind azimuth): True

**API cascade fixed (8 sites):** AC 签名变化级联到 `run_wp2_league.py::ACCommander` + `opponent_pool.py::action_fn` + `run_wp2_crossplay.py::fn` + `run_g0_gate.py::make_ac_action_fn` + `wp_c_d3_killer_contrast.py` + `wp_c_r3_rl_demo.py` 全部加 detect_list 参数。

**复用代码:**
- `combine_team_actions` 已 stack beam_direction (M2 早已)
- `env.assert_no_godview(tol=1e-5)` 直接对 RL AC 工作
- `BatchedIMMPDAF` env 自带，RL 不重写
- `TwoTeamOpponentPool` PFSP + EMA win-rate 已实现
- `α_eff blend` priv[:,4] guard 在 br_trainer.py:207 已咬过

**Production launch (user runs, 6-12h RTX PRO 6000):**
```bash
cd /home/ubuntu/CODE/FluxPhased-
python experiments/twoteam/wp3_train.py --iters 1000 --n-envs 256 --horizon 300 2>&1 | tee checkpoints/blind/wp3_train_log.txt
```

**Next:** WP-4 RL vs BlindClassical cross-play 报告 (kill/survival/track vs 干扰强度 + Welch-t 统计)。参见 [[twoteam-wp2-blind-classical-pass]] (前置 baseline) 和 [[twoteam-wp1-no-godview-pass]] (no-godview 前置) 和 [[twoteam-multifunction-pivot]] (框架总览)。

**WP-3 完成定义 (per plan) 状态更新 (2026-07-17 实测后):**
1. ✅ actor-critic 无 `beam_target_head` + detection encoder permutation-invariant (6 tests)
2. ✅ trainer `_RolloutBuffer` 含 `beam_direction` + `detect_list`，PPO update 真回流梯度 (7 tests)
3. ✅ BC teacher = `BlindClassicalCommander`，buffer `beam_direction` (5 tests)
4. ✅ league pool 含 BlindClassical + BC step 用 blind teacher (4 tests)
5. ⚠️ RL 训练稳定不发散 (entropy/KL/NaN 都 OK)，但 **4 次 smoke cross-play 全 FAIL**：100-iter ×3 + 500-iter ×1，Δ_kills ∈ [-0.925, -0.825] 低干扰 / [-0.150, -0.125] 高干扰。详见 [[twoteam-wp3-production-smoke-fail]]。
6. ✅ spec §6 反 toy checklist 全 PASS (9/9 items)
7. ❌ smoke cross-play: **RL 低干扰没打平 BlindClassical** (RL kill 0.05 vs BC kill 0.90)。用户决策接受 IET floor，进入 WP-4。

**500-iter 实测结论**: 5 倍 compute (100→500 iter) 没让 kill 从 0 出现，根因是结构性问题 (BC teacher lock + PFSP collapse + zero-sum mirror symmetry + 缺 kill shaping)，**不是 compute 不足**。诚实判断 1000-iter (5e7 steps) 不会改变结论。用户选 "写总结，终止训练" 2026-07-17。最终总结：`experiments/twoteam/WP3_FINAL_SUMMARY.md`。
