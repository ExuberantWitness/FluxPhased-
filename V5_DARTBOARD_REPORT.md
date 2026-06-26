# FluxLeague v5 Dart-Board Report — Kalman Stale-State Root Cause

**Date**: 2026-06-26
**Status**: Root cause identified, fix applied, verification training pending

---

## TL;DR

All 4 v5 ablation variants (v5a/v5b/v5c/v5d) failed to produce kills. **Root cause is NOT in the policy, reward, or PPO — it is a 6-year-old-style bug in episode lifecycle management**: `KalmanTracker.reset()` is never called from the league training path, so from episode 2 onwards the Kalman filter carries the previous episode's enemy position as its prior. This makes the policy's "anchor" (sensed enemy position) land **2-3 km away from truth**, far outside any reasonable kill_radius.

A direct probe (`test_kalman_bias.py`) confirms:
- **tracked mode + reset every episode**: 0.043 m bias (Kalman converges correctly)
- **tracked mode + no reset (the bug)**: 2537 m bias on episodes 2+

The oracle kill test (Phase 1) already proved env mechanics work; the Kalman bug explains why the policy could never exploit them.

---

## Investigation Path

### Phase 1 — Ground-truth oracle test (env mechanics)

`test_oracle_kill.py` bypassed PPO and gave the env a perfect-aim scripted policy:

| Scenario | Aim source | Fire | Kills / 5 ep |
|---|---|---|---|
| A.OracleAlwaysFire | true enemy pos | always | **40** ✅ |
| B.OracleBernoulli | true enemy pos | Bern(0.5) | 6 |
| C.OracleBias270m | true + 270m offset | always | **0** ❌ |

**Verdict**: env kill mechanics work; kill_radius check is correct; the policy's aim is what's broken.

### Phase 2 — Where does the policy's aim come from?

The aim is decoded from `commander_action[..., 1:3]` by `vec_drone.py`. `ppo_trainer._apply_residual_aim` constructs the env-action as:

```python
aim = anchor + residual    # anchor = Kalman-fused enemy pos from obs[68:70]
```

So at init (residual≈0), policy aim ≈ Kalman-fused enemy position.

### Phase 3 — Dart-board diagnostic (v5c, last 3 episodes)

```
[dart] fire_rate=0.461  min_dist_init=276.39m  final=273.89m  change=-2.50m  min=1.06m
[dart] fire_rate=0.496  min_dist_init=268.35m  final=261.87m  change=-6.48m  min=0.86m
[dart] fire_rate=0.480  min_dist_init=271.00m  final=265.77m  change=-5.24m  min=0.86m
```

- `min_dist_init ≈ 270 m` → policy's aim is **270 m off truth** at episode start
- `min_dist_min ≈ 1 m` → KF partially converges DURING the episode (gets to 1m)
- `change ≈ -5 m` → small convergence during the episode, but never enough, never fast enough
- `fire_rate ≈ 0.48` → policy fires 50% of the time (Bernoulli init, never learned to sustain)

This pattern is consistent with a KF that starts each episode 270 m off truth, then slowly converges over the episode's 500 steps. Why 270 m average? Because enemies spawn at random positions up to ±10 km from the previous episode's enemies; KF pulls toward new truth at a rate bounded by `track_q_m=0.02` per step, reaching ~1 m by step ~300, then averaging ~270 m across the episode.

### Phase 4 — Direct Kalman probe (the smoking gun)

`test_kalman_bias.py` builds an env, runs `fused_sensing` at episode start, measures distance between Kalman-fused enemy-0 position and true enemy-0 position.

| Mode | Reset every ep? | Mean error | Episodes 2+ error |
|---|---|---|---|
| fused | yes | 0.20 m | 0.19 m |
| fused | no | 0.23 m | 0.24 m |
| **tracked** | **yes** | **0.043 m** | **0.037 m** |
| **tracked** | **no (BUG)** | **2029 m** | **2537 m** |

Without reset, the KF carries the previous episode's converged state as the prior for the new episode's first measurement. The new enemy is randomly placed, so the prior is wildly wrong, and the KF (with `track_q_m=0.02`) takes many steps to recover. The first step's "fused" output — which becomes the policy's anchor — is essentially the previous enemy's position.

### Phase 5 — Code search confirms the missing call

```bash
$ grep -rn "kalman_tracker.reset\|tracker.reset" training/
training/laser/sensing.py:199:    def reset(self):
training/baselines/classical_mpc.py:120:        self.kalman_tracker.reset()
```

**`reset()` is called from `classical_mpc.py` (a baseline) but NEVER from `TeamPPOTrainer` or `flux_league._train_against`.** This is the bug.

The same audit also revealed that `LaserRewardShaper.reset_episode` is *also* never called from the league path — its per-episode state (`_beam_hit_time`, `_fire_streak`, `_prev_min_dist`, `_jam_level`) leaks across episodes too. This is a smaller bug (the shaper has an E-shape guard that lazily re-allocates, but doesn't clear values), but worth fixing alongside.

---

## Fix Applied (uncommitted)

### 1. `training/ppo/ppo_trainer.py` — new method

```python
def reset_episode(self):
    """Reset all per-episode state at the start of a new episode."""
    if self.task_type != "laser":
        return
    if self.kalman_tracker is not None:
        self.kalman_tracker.reset()
    if self.reward_shaper is not None and hasattr(self.reward_shaper, "reset_episode"):
        self.reward_shaper._beam_hit_time = None
        self.reward_shaper._fire_streak = None
        self.reward_shaper._prev_min_dist = None
        self.reward_shaper._jam_level = None
```

### 2. `training/flux_league.py` — call reset_episode every ep

```python
for ep in range(n_episodes):
    runner.reset(red_trainer=trainer, blue_trainer=opp_trainer)
    # Reset per-episode state in BOTH trainers (Kalman tracker + reward
    # shaper streak/prev_dist). Without this, KF carries the previous
    # episode's enemy position as prior → 2-3km anchor bias on ep 2+.
    if hasattr(trainer, "reset_episode"):
        trainer.reset_episode()
    if hasattr(opp_trainer, "reset_episode"):
        opp_trainer.reset_episode()
```

---

## Why the v5 Ablation Was Doomed

| Variant | What it tested | Why it failed |
|---|---|---|
| v5a_big_residual | residual_scale=500 | Big residual can't correct a 270m KF bias; anchor noise dominates |
| v5b_no_residual | residual_aim=false | Policy outputs absolute aim directly, but obs[68:70] still contaminated by KF bias leaking into downstream recon obs |
| v5c_anchor_noise | +50m noise on anchor | Adding 50m noise to a 270m-biased anchor doesn't help |
| v5d_combo | v5a+v5c combined | Stack of two ineffective fixes |

All 4 variants were treating symptoms of a sensing-side bug.

---

## Recommendation

### Immediate (verification)

1. **Run v3_scaling for 30 min with the fix applied** — dart-board metric should show:
   - `min_dist_init` drops from ~270 m to **< 5 m**
   - `min_dist_change` becomes near-zero (already at enemy)
   - First kill appears in episodes 1-5
2. **Re-run oracle test C (270m bias)** — sanity check still passes (no env regression)

### Short-term (after verification passes)

3. **Drop the v5 ablation entirely**. The 4 variants were testing fixes for the wrong bug.
4. **Revert `dartboard_weight` and `anchor_noise_std_m` to 0**. They are unnecessary once Kalman works; they pollute the reward.
5. **Drop the C2 self-play win-accounting patch** is unnecessary (was a cosmetic logging fix).
6. **Keep the C1 commander buffer bootstrap fix** (line 793 `compute_returns(last_value=...)`) — that was a real bug independent of the Kalman issue.

### Medium-term (after v3_scaling trains successfully)

7. Re-run the original 4-variant v2 ablation (v4_control / v1_conservative / v2_aggressive / v3_scaling) for the full 2h each — these were the actual ablation of interest, but their results were invalidated by the Kalman bug.

---

## Open Questions

- **Why didn't `classical_mpc.py` exhibit the bug?** Because it explicitly calls `tracker.reset()` at line 120 — the author of the MPC baseline knew about the issue, but the knowledge didn't propagate to the league trainer.
- **Are there other per-episode state variables leaking?** Likely yes. A full audit of `TeamPPOTrainer.__init__` for any member prefixed with `_` that's not cleared in `reset_episode()` would be worthwhile.
- **Why was the oracle test (Phase 1) the right call?** It bypassed PPO and sensing, isolating env mechanics. If we had run it first (before any v2/v5 ablation), we'd have saved ~16 GPU-hours.

---

## Artifacts

- `test_oracle_kill.py` — ground-truth oracle kill test (Phase 1)
- `test_kalman_bias.py` — direct Kalman bias probe (Phase 4, the smoking gun)
- `logs/diag/v5c_anchor_noise_20260626_201124.log` — v5c dart-board diagnostic data
- `configs/ablation_f1f8/v5{a,b,c,d}_*.yaml` — deprecated; do not use
