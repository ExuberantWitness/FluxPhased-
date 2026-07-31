---
name: fluxleague-r01234-training
description: R0-R5 AlphaStar league fixes applied 2026-06-23. Re-runs formal 24-iter training (PID 1296303) after NaN root-cause was corrected from Adam-eps misdiagnosis to _apply_residual_aim buffer mutation.
metadata: 
  node_type: memory
  type: project
  originSessionId: 902a7f7f-2d60-4a53-b927-2a75af5c8fc4
---

R0–R5 AlphaStar league fixes applied 2026-06-23, superseding the earlier Adam-eps NaN diagnosis.

**Why:** Previous training (PID 1417143, launched 2026-06-21 19:31) crashed at iter 1 ep7 step152 with ValueError at actor_critic.py:271. Initial LASER_LEAGUE_NAN_ROOT_CAUSE.md blamed Adam eps=1e-8 + zero exp_avg_sq dimensions — this was **wrong** (the fix was never even applied to code). Real root cause per LASER_LEAGUE_NAN_FULL_ANALYSIS.md (committed by user via force-push 81f1abb): `_apply_residual_aim` in ppo_trainer.py mutated the buffer action in-place from residual (in (-1,1)) to absolute aim clamped to ±1, which downstream atanh turned into log(0)=-∞ → NaN gradients → NaN weights → NaN aim_mean crash.

**How to apply (the 7 concrete fixes that landed):**
1. **F1** (root cause, ppo_trainer.py:351): `_apply_residual_aim` returns separate env-action tensor; `get_own_actions` stores raw residual in transition, returns absolute aim only for env.step.
2. **F2** (defense, actor_critic.py:264,275): clamp actions to (-1+1e-6, 1-1e-6) before atanh.
3. **F3** (desensitize): `log_std_floor=-3` config + forward-time clamp on all commander log_std sites.
4. **F4** (guardrail, ppo_trainer.py:112): ratio log-ratio clamp [-20,20] + NaN-skip guard around loss.backward().
5. **win_rate=0.50 bug** (opponent_pool.py:115,157): unknown opponents → NaN priority (evaluate first, don't default to 0.5); `update_win_rate` first-obs-replaces instead of EMA-from-0.5.
6. **R1+R2** (flux_league.py:387): `_sample_ma_opponents` — K=4 slots per cycle, 0.5 self-play prob (AlphaStar MA distribution). `_train_against` takes `n_episodes` for K-way budget split.
7. **R3** (opponent_pool.py:94): PFSP `f_hard(x)=(1-x)^p` direct normalization (softmax removed). `pfsp_hardness_p` config.
8. **R4** (flux_league.py:712): `_maybe_reset` resets exploiters to strongest MA snapshot via `_strongest_main_snapshot(team)`, parent as fallback.
9. **R5**: Nash already eval-only (verified); added clarifying comment.

**Training relaunched:** 2026-06-23 07:06, **PID 1296303**. Config `configs/laser_25x25_pro6000_league.yaml` (24 iter × 10 ep × 500 steps × 12 envs). Log `logs/laser_league_R01234_20260623_070612.log`.

**Status as of 2026-06-25 21:39:** PID 1296303 was killed by user. After ~62.5h it had only reached kill_radius=12.5m (4th anneal tier), with PPO policy_loss → 0 (stable ignorance at local optimum). Diagnosis: reward scale imbalance — kill_bonus=100 vs cumulative shaping ~20000 (200:1 ratio) drowns the kill signal so policy_loss gradient is dominated by value-regression noise.

**Follow-up fixes (F2-F8) implemented 2026-06-25:**
- **F2** (flux_league.py ~L688): PPO update trigger `if episodes % 10 == 0` → `if episodes >= 1` — was never firing for K=4 main agent (episodes_per_opp = max(1, 10//4) = 2 < 10).
- **F3+F7** (payoff_matrix.py): sticky `last_kill_rate` (always updated even on cached eval) + `re_eval_interval=3` periodic re-evaluation.
- **F4** (flux_league.py ~L468): `total_iters = max(30, 1)` → `max(1, self.psro_iterations)` + `psro_iterations` param added to __init__ — linear alpha schedule was saturating at iter 15 instead of iter 12.
- **F5** (flux_league.py): `_sample_ma_opponents` uses AlphaStar 35/50/15 mass split (35% self-play / 50% PFSP-hard / 15% forgotten) + new `_sample_forgotten_opponents` method.
- **F6** (opponent_pool.py): `_main_pids_to_evict` evicts intermediate main generations per team (keeps latest + strongest), fixing population_cap=12 violation.
- **F8** (buffer.py + ppo_trainer.py + flux_league.py + train.py): return-based scaling via RunningMeanStd — `reward_normalize` flag. Normalizes rewards by running std before GAE so value targets stay O(1).
- **Elo-band disabled** in ablation configs (`use_elo_band: false`) — bypass was re-introducing the 0.5 win-rate bug.

**Ablation launched 2026-06-25 23:29:** 4 yaml variants × 2h sequential, master PID 617929 (runner) → v4_control PID 617934. Variants in `configs/ablation_f1f8/`:
- `v4_control.yaml` — F2-F8 code fixes only, no reward changes (baseline)
- `v1_conservative.yaml` — kill_bonus 100→20000
- `v2_aggressive.yaml` — v1 + illum 50→5, commit 8→1, guidance 5→1
- `v3_scaling.yaml` — reward_normalize=true (F8 active), no manual reward tuning

Launch: `bash run_ablation.sh` (auto-kills each at 2h). Analyze: `python scripts/analyze_ablation.py`. Expected completion ~2026-06-26 07:29.

Sanity check confirmed bug reproduction: v4 control shows value_loss=1.65M on first PPO update — exactly the scale collapse F8 was designed to fix.

**Old PID 1417143 is dead. Old PID 1296303 is dead.** Do not confuse prior training runs.

Related: [[fluxleague-kill-fix-tier1]] (Tier 1 fire head bias + beam_az threading — still applies), [[fluxleague-paper-framing]] (EAAI Q1 story).
