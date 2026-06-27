# FluxLeague v5 Dart-Board Report — aim_z Drift Root Cause

**Date**: 2026-06-26 (initial), 2026-06-27 (corrected after deeper audit)
**Status**: True root cause identified and fixed; verification training pending

---

## TL;DR

All 4 v5 ablation variants (v5a/v5b/v5c/v5d) failed to produce kills. **The initial Kalman-stale-state hypothesis was WRONG** — `LaserEpisodeRunner.reset()` already resets the Kalman tracker every episode (episode.py:174-176). The actual root cause is **`aim_z` drift**: `vec_drone.py:195` scales `commander_action[3]` by 1000 m, but enemies are ground radars at z=0. The untrained policy samples `action[3] ~ N(0, 0.37²)` (init log_std=-1), producing `|aim_z| ≈ 300 m` on average — which dominates `min_dist` and prevents any kill.

**Confirmed via direct probe during training** (`[aim-dbg]` output):
```
[aim-dbg t=0] aim_z_avg=355.8m  |res_x|=0.339  |res_y|=0.218
[aim-dbg t=0] aim_z_avg=258.0m  |res_x|=0.264  |res_y|=0.274
```

`min_dist = √(res_x² + res_y² + aim_z²) ≈ √(1.8² + 1.5² + 270²) ≈ 270 m` — exactly matches the dart-board diagnostic.

The Kalman filter is verified working: warm-start converges to ~2-3 cm (10 separate warm-starts observed during eval, all <5 cm).

---

## Correction after deeper audit

### What I got wrong

The initial investigation (Phase 4-5 below) concluded the Kalman tracker was never reset. That was based on a code search that missed `LaserEpisodeRunner.reset()` (episode.py:168-176), which already does:

```python
for trainer in (red_trainer, blue_trainer):
    ...
    shaper = getattr(trainer, "reward_shaper", None)
    if shaper is not None and hasattr(shaper, "reset_episode"):
        shaper.reset_episode(E, n_teams)
    kalman = getattr(trainer, "kalman_tracker", None)
    if kalman is not None and hasattr(kalman, "reset"):
        kalman.reset()
```

So the Kalman tracker IS reset every episode (via `runner.reset()` in both `_train_against:609` and `evaluate_pair:117`). The `test_kalman_bias.py` "no-reset 2838 m" scenario was an artificial test that didn't match the actual training path.

### What's actually broken

`vec_drone.py:195`:
```python
self._commander_aim[..., 2] = commander_actions[..., 3] * 1000.0  # z scale
```

`commander_action[3]` is interpreted as aim_z, scaled by 1000 m. So:
- `action[3] = 0` → aim_z = 0 (correct for ground target)
- `action[3] = ±1` → aim_z = ±1000 m
- Random init with `log_std = -1` → `std ≈ 0.37` → `E[|action[3]|] ≈ 0.30` → `E[|aim_z|] ≈ 300 m`

In `reward.py`, `min_dist` is a 3D Euclidean norm including z. Ground enemies sit at z=0. So any nonzero `aim_z` directly inflates `min_dist`:

```
min_dist = sqrt(Δx² + Δy² + aim_z²)
         ≈ sqrt(2² + 2² + 300²)
         ≈ 300 m
```

This explains the 270 m average perfectly. And since `kill_radius_init = 50 m`, no kill is possible until the policy learns to drive `action[3] → 0` — which PPO can't easily learn because:

1. `beam_guidance` uses `log(r_ref / r_eff)` which saturates (log(3000/300)=2.3 vs log(3000/2)=7.4 — weak gradient)
2. `kill_bonus = 100000` is sparse and never triggered
3. Entropy bonus actively pushes `action[3]` to explore

The `min=0.75 m` in the dart metric comes from rare moments when `action[3]` happens to sample near 0 by chance.

### The fix

In `ppo_trainer._apply_residual_aim`, force `env_action[..., 3] = 0`:

```python
env_action[..., 3] = 0.0  # aim_z=0 (ground targets only)
```

The PPO buffer still stores the raw sampled `action[3]` (so log_prob is preserved), but the env never sees it. This is equivalent to removing the z DoF entirely for the laser-kill task.

---

## Investigation Path (original, with corrections)

### Phase 1 — Ground-truth oracle test (env mechanics) ✅

`test_oracle_kill.py` verified env kill mechanics work. Scenario A (perfect aim, always fire) → 40 kills/5 ep. Scenario C (270 m offset) → 0 kills. Conclusion stands: env is correct, problem is in the policy's aim.

### Phase 2 — Where does the policy's aim come from? ✅

`ppo_trainer._apply_residual_aim` builds `env_action[1:3] = anchor + residual`. At init (residual ≈ 0), XY aim is anchored at Kalman enemy position. Correct.

### Phase 3 — Dart-board diagnostic (v5c) ⚠️ partial

The 270 m `min_dist_init` was real, but the interpretation was wrong. It's not KF bias — it's `aim_z` drift entering the 3D norm. The `min=0.75 m` clue (KF is actually fine) was overlooked.

### Phase 4 — Direct Kalman probe ⚠️ misleading

`test_kalman_bias.py` measured the tracker in isolation, where `reset()` truly was missing. But in the actual training path, `LaserEpisodeRunner.reset()` already handles it. The 2.5 km error was a test artifact.

### Phase 5 — Code search ❌ incomplete

`grep` for `kalman_tracker.reset` missed the indirection through `LaserEpisodeRunner.reset()` which calls `getattr(trainer, "kalman_tracker", None).reset()` dynamically.

### Phase 6 — Re-audit with debug prints ✅

Added per-call debug prints:
- `[fused_sensing] warm-start done: err_x=0.028m err_y=0.014m` — confirmed Kalman converges to 2-3 cm every episode
- `[aim-dbg t=0] aim_z_avg=355.8m` — confirmed aim_z drift is the actual cause

These two prints together are conclusive: anchor is correct, aim_xy residual is small (~1.8 m), but aim_z is ~300 m — the only thing that can produce a 270 m `min_dist`.

---

## Fix Applied

### 1. `training/ppo/ppo_trainer.py` — `_apply_residual_aim`

```python
env_action[..., 3] = 0.0  # aim_z=0 (ground targets only)
```

### 2. `training/ppo/ppo_trainer.py` — `reset_episode` is now a no-op

`LaserEpisodeRunner.reset()` already handles the reset. Kept the method as a no-op hook to avoid breaking any caller that invokes it.

### 3. `training/flux_league.py` — kept the `reset_episode()` calls

Harmless (they call a no-op). Removed in a follow-up if desired.

---

## Why the v5 Ablation Was Doomed

All 4 variants were testing fixes for a non-existent Kalman bug:

| Variant | What it tested | Why it failed |
|---|---|---|
| v5a_big_residual | residual_scale=500 | XY residual can't compensate for Z drift |
| v5b_no_residual | residual_aim=false | Same aim_z issue, just via different code path |
| v5c_anchor_noise | +50 m noise on anchor | Adding noise to XY anchor doesn't help Z |
| v5d_combo | v5a+v5c combined | Stack of two ineffective fixes |

The anchor_noise feature is now known to be unnecessary. Recommend setting `anchor_noise_std_m = 0` in all configs.

---

## Recommendation

### Immediate

1. **Run v3_scaling for 30 min with the aim_z fix applied** — dart-board metric should show:
   - `min_dist_init` drops from ~270 m to **< 10 m** (dominated by first-step zero-aim artifact)
   - `min_dist_final` drops to **< 5 m** (true aim accuracy)
   - Real kills within first 2-3 episodes (kill_radius=50 m easily achievable)
2. **Re-run oracle test C** — sanity check no env regression

### Short-term

3. **Drop the v5 ablation entirely.** All 4 configs were testing fixes for the wrong bug.
4. **Set `dartboard_weight = 0` and `anchor_noise_std_m = 0` in all configs.** Both are unnecessary now; they pollute the reward / observation.
5. **Keep the F1/F4/F8 stability fixes** (env/buffer action separation, NaN skip-guard, reward normalization) — these are real and unrelated to aim_z.
6. **Keep the C1 commander buffer bootstrap fix** (`compute_returns(last_value=...)`) — also unrelated and real.

### Medium-term (after verification passes)

7. Re-run the original 4-variant v2 ablation (v4_control / v1_conservative / v2_aggressive / v3_scaling) — previous results were invalidated by the aim_z bug.
8. Consider removing `aim_z` DoF entirely from the commander action space (drop action dim from 5 to 4). The env supports air targets for future work, but for the current ground-target task, the policy shouldn't even see this DoF.

---

## Lessons Learned

1. **Read the entire call chain before concluding.** I grep'd for `kalman_tracker.reset` and missed `LaserEpisodeRunner.reset()` which calls it via `getattr`.
2. **Test artifacts ≠ production bugs.** `test_kalman_bias.py` showed a real 2.5 km error in a synthetic scenario, but the production path didn't have that scenario.
3. **The min= clue was ignored.** `min_dist_min ≈ 1 m` was a strong signal that KF was working — I should have weighted it more heavily.
4. **Run debug prints BEFORE forming hypotheses.** Adding `[aim-dbg]` took 5 minutes and immediately pointed to the real bug. I should have done this first instead of writing a synthetic test.

---

## Artifacts

- `test_oracle_kill.py` — Phase 1 oracle test (still valid)
- `test_kalman_bias.py` — Phase 4 probe (valid test, but its "no-reset" scenario doesn't reproduce in production)
- `logs/diag/v3_aimz_20260626_234854.log` — Phase 6 smoking-gun output
- `logs/diag/v5c_anchor_noise_20260626_201124.log` — original dart-board diagnostic
- `configs/ablation_f1f8/v5{a,b,c,d}_*.yaml` — deprecated; do not use
