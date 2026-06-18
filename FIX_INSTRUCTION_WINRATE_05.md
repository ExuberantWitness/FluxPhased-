# 🔧 FIX INSTRUCTION — FluxLeague win_rate ≡ 0.50

**For the agent on the PRO 6000.** This is a *precise* fix + verification spec.
Do exactly this. Do NOT re-extract sensing code or touch `training/laser/sensing.py`
— that module is correct. The bug is a **call-ordering / timing** bug in the episode
runner, not a sensing bug.

---

## 1. Root cause (read this first — the previous diagnosis was wrong)

`win_rate ≡ 0.50` because, in tracked-sensing mode, the **Kalman warm-start runs on
the un-spread, near-collinear random radar geometry**, locking the tracker onto a
degenerate fused anchor for the whole episode. Both teams therefore aim at the map
centre, never illuminate the enemy → `illumination_progress = 0` for both → the payoff
tiebreaker degenerates to 0.5.

**Why:** `enforce_radar_baseline` (which spreads each team's two radars to ≥5 km so the
triangulation geometry is non-degenerate) is **NOT called at episode reset** in the
FluxLeague path. It is only called *per-step inside `get_own_actions`, AFTER the
observation is already built* — too late for the warm-start.

**Proof (the working path does it right):** `train_laser.py` applies
`_enforce_radar_baseline` **at reset** (train_laser.py:317 "at reset"), *before* the
first `commander_obs` is built (train_laser.py:367) and before the warm-start — so its
fusion always sees good geometry and it reaches kr=0.20m. FluxLeague's
`LaserEpisodeRunner.reset()` (training/laser/episode.py:90-111) resets env + cpi +
reward shaper + Kalman tracker but **never calls `enforce_radar_baseline`**.

Earlier misreads to discard: the anchor is not "saturated to ±1 by a sensing.py bug",
and `laser_aim=(10000,10000)` is the **map centre** of the [0,20000]² map (the
residual≈0 aim with a degenerate/centre anchor), not a "corner". The sensing module and
`enforce_radar_baseline` themselves are correct.

---

## 2. THE FIX (primary — ~6 lines, `training/laser/episode.py`)

In `LaserEpisodeRunner.reset()`, immediately **after** `self.env.reset(env_ids)` and
**before** anything else, enforce the baseline using the trainers' configured value:

```python
    def reset(self, env_ids=None, red_trainer=None, blue_trainer=None):
        """Reset env + CPI buffer + clear cached actions + per-episode trainer state."""
        self.env.reset(env_ids)
        # FIX(win_rate=0.5): enforce the radar baseline AT RESET — before the first
        # commander_obs is built and before the Kalman warm-start. Mirrors
        # train_laser._enforce_radar_baseline (applied at reset). Without this the
        # warm-start locks the tracker onto the un-spread near-collinear geometry →
        # degenerate fused anchor → aim at map centre → progress=0 → win_rate=0.5.
        from training.laser.sensing import enforce_radar_baseline
        for _trainer in (red_trainer, blue_trainer):
            _base = float(getattr(_trainer, "min_radar_baseline_m", 0.0)) if _trainer else 0.0
            if _base > 0.0:
                enforce_radar_baseline(self.env, _base)  # spreads ALL teams in one call
                break
        self.cpi_buffer.reset()
        # ... (rest of reset() unchanged)
```

Notes:
- `enforce_radar_baseline` already loops over all teams, so **one call covers both
  teams**; the `break` is correct.
- It is idempotent (already-≥baseline radars are left unchanged), so the existing
  per-step call in `get_own_actions` stays harmless.
- If you ever call `reset()` with a **partial** `env_ids`, scope the enforce to those
  ids (the full-batch reset used by PSRO eval/training needs no change).

## 3. Optional secondary cleanup (`training/ppo/ppo_trainer.py`, get_own_actions)

The per-step `enforce_radar_baseline(env, self.min_radar_baseline_m)` currently runs
*after* `_get_observations(...)` builds `commander_obs`, so `obs[0:4]` is stale by one
step. With the reset fix this is cosmetic, but for exactness move that `enforce_...`
call to **before** the `state, commander_obs = self._get_observations(...)` line.
Do NOT remove it — keep the radars spread as they move (20 m/s) through the episode.

---

## 4. Verification (do ALL three)

### 4.1 Static
```bash
grep -n "enforce_radar_baseline" training/laser/episode.py    # must now appear inside reset()
```

### 4.2 Diagnostic — anchor/progress no longer degenerate
Re-run your previous diagnostic (the one that printed `illumination_progress` and
`laser_aim`). Expected AFTER fix, within the first eval episode:
- `illumination_progress` **> 0** for at least the stronger team (was 0.0000 / 0.0000);
- `laser_aim` near the **enemy** position, **not** `(10000, 10000)` (map centre).

### 4.3 Functional — payoff no longer uniform 0.5
```bash
python -m training.train \
  --config configs/laser_25x25_pro6000_league.yaml \
  --override training.psro_iterations=3 \
  --override env.num_envs=8 \
  --override league.episodes_per_training=3 \
  --override league.n_eval_games=4
```

**PASS criteria (ALL must hold):**
| Check | PASS condition |
|---|---|
| Payoff entries | cross-team win-rates are **NOT all 0.50** (some >0.5, some <0.5) |
| Meta-Nash | `NashConv > 0`; `sigma` is **not** `[1, 0, 0, ...]`; effK > 1 |
| Kills happen | eval logs show `kills > 0` / `illumination_progress > 0` for a winner |
| Curriculum | `kill_radius` starts tightening (kr drops below 50m across iters) |

If 4.3 still shows uniform 0.5 → the warm-start is still seeing bad geometry; double
check the fix is in the code path actually used (print `env.radar_pos` pairwise
distance right after reset — each team's two radars must be ≥ `min_radar_baseline_m`).

---

## 5. Rollback / safety
- Single-method change, lazy import, guarded by `min_radar_baseline_m > 0` → zero impact
  on non-laser tasks and on configs without the baseline set.
- If anything regresses, revert the `reset()` hunk; behaviour returns to the (broken) 0.5.

## 6. After it passes
Report back: the 3×3 (or NxN) payoff matrix, `NashConv`, and the eval `[Eval]` lines
(kr + cum red/blue/draw). Then we can scale `env.num_envs` up and run the full
`laser_25x25_pro6000_league.yaml` (30 iters) for real.

> Reminder: the simpler `train_laser.py` built-in PSRO-lite league
> (`python -m training.train_laser --config configs/laser_25x25_pro6000.yaml`) already
> works (red≈0.88 vs frozen pool, kr→0.2m) and does NOT have this bug — use it if you
> don't need FluxLeague's full Nash/exploiter/TC-DAMS meta-game.
