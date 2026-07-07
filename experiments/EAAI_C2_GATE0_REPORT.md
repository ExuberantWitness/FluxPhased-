# EAAI C2 — Gate 0 Report: EW Injection + Cross-Play vs ClassicalMPC

**Branch**: `phase1.5/three-way-baselines`
**Plan ref**: `EAAI_C2_EW_BELIEF_PLAN.md` (commit `a244f4c`)
**Date**: 2026-07-07
**Verdict**: **Gate 0 FAIL** — and the failure is not EW-parameter-shaped.

---

## 1. Goal

Gate 0 of the two-gate EAAI-C2 strategy: inject electronic-warfare (EW) params
into existing MAPPO/IPPO baselines, train two new arms (`ew_mappo`, `ew_ippo`),
cross-play each vs `ClassicalMPC` under the same EW environment, and verify the
"AI beats classical" requirement: NN win-rate > 0.5 vs the rule-based
beam-follow controller. Pass → proceed to Gate 1 (belief-conditioned CTDE).
Fail → tune EW params or pivot.

## 2. What was built

### 2.1 Two new APP arms (clone + EW delta)

- `algo/ew_mappo/{AGENTS.md, README.md, APP_PUBLICATION.json, environment/, code/config.yaml, data/.gitkeep}`
- `algo/ew_ippo/{AGENTS.md, README.md, APP_PUBLICATION.json, environment/, code/config.yaml, data/.gitkeep}`

Both inherit the MAPPO/IPPO scaffold byte-identically and inject the same 8 EW
fields:

| Field (reward_shaping) | Value | Effect |
|---|---|---|
| `jam_gain` | 8.0 | Enemy jam multiplies my range+cross-range σ by (1 + 8·jam) — SNR collapses |
| `jam_cost` | 0.01 | Per-step emission cost — jamming is not free |
| `exposure_gain` | 50.0 | Home-on-jam beacon: my emission is an isotropic info source the enemy can localize me from |
| `race_time_cost` | 0.01 | Time pressure: survive-then-kill race |
| `race_death_penalty` | 30.0 | Death penalty for the race framing |
| `jam_log_std` (training) | -1.0 | Initial jam action std |
| `jam_kr_threshold_m` (training) | 0.5 | Curriculum gates EW on when `kill_radius ≤ 0.5m` |
| `commander_privileged_dim` (training) | 10 | Centralized critic privileged-info dim (for jam level + ally state) |

`env.kill_radius_m` lowered from 0.5 → 0.2 so the curriculum anneals past the
`jam_kr_threshold` and actually triggers EW in Phase B.

### 2.2 EW-aware cross-play harness

The previous `crossplay_mpc.py` and `ClassicalMPC` both called `fused_sensing()`
without EW params — meaning EW effects were silently dropped during eval,
invalidating any "AI beats classical under EW" verdict.

Fix (this commit):

- `ClassicalMPC.__init__`: new optional kwargs `jam_gain=0.0`, `exposure_gain=0.0`; `get_own_actions` passes them + `self.jam_level` to `fused_sensing`
- `NNPolicyAdapter.__init__`: same EW kwargs; `get_own_actions` updates `self.jam_level[:, team]` from `c_action[:, 4]` (matches `train_laser.py:1276`)
- `directional_match`: creates a shared `jam_level[E, n_teams]` tensor, hands both adapters a reference, zeros `jam_level[:, mpc_team]` each step (classical never jams)
- 1-step lag semantics preserved (matches training): fused_sensing at step t reads jam_level set at step t-1

Back-compat: with no-EW config, `jam_gain=0` short-circuits the EW branch in
`fused_sensing`; the adapter behaves identically to the pre-patch version.
Verified by re-running original `mappo` baseline.

## 3. Training results (20-iter, seed=42, RTX PRO 6000)

| Arm | Wall-clock | iter 20 kr | iter 20 jam | cum_red | eval_kr | adv_std | EW onset |
|---|---|---|---|---|---|---|---|
| mappo (baseline) | — | 0.20 | — | 0.96 | 0.50 | 8.5 | n/a |
| ippo (baseline) | — | 0.20 | — | ~0.70 | 0.45 | 18.0 | n/a |
| **ew_mappo** | 4h27m | 0.24 | 0.47 | **0.62** | 0.333 | 14.68 | iter 17 |
| **ew_ippo** | ~5h | 0.20 | 0.53 | **0.87** | 0.958 | 34.95 | iter 17 |

**Headline**: under EW, IPPO beats MAPPO (0.87 vs 0.62 cum_red). The reverse of
the clean-env ordering (MAPPO 0.96 vs IPPO ~0.70). Jam settles at intermediate
levels (0.47, 0.53) — not max — confirming the home-on-jam exposure term makes
the policy learn a timed-jam strategy rather than brute-force denial.

## 4. Cross-play results vs ClassicalMPC (72 games each, both directions)

| Arm | NN WR | W-L-D | Notes |
|---|---|---|---|
| mappo (no EW, historical) | 0.472 | 34-38-0 | n=72, prior commit |
| ippo (no EW, historical) | 0.167 | 12-60-0 | n=72, prior commit |
| pspfix (no EW, historical) | 0.431 | 31-41-0 | n=72, prior commit |
| mappo baseline (re-run, no EW) | 0.417 | 10-14-0 | n=24, validates adapter |
| **ew_mappo** | **0.069** | 5-67-0 | n=72 |
| **ew_ippo** | **0.264** | 19-53-0 | n=72 |

## 5. Gate 0 verdict: FAIL

The pass criterion was "all arms beat MPC under EW, NN WR > 0.5." None of the
four arms (with or without EW) exceeds 0.5. Best is `mappo` no-EW at 0.472,
statistically indistinguishable from a coin flip (SE ≈ 0.06 at n=72).

**Crucially**: this is not an EW-parameter problem. The classical baseline was
**already** competitive-or-better in the no-EW baselines. Tuning
`jam_gain` 8 → 16 or `exposure_gain` 50 → 100 will not gate-pass, because the
NN arms also degrade under stronger EW (cross-play WRs move with training
cum_red, which falls as EW intensifies).

## 6. Diagnostic: why doesn't RL beat the classical controller?

### 6.1 ClassicalMPC shares the Kalman-fused frontend

`ClassicalMPC.get_own_actions` calls the same `fused_sensing()` (multi-static
range triangulation + 2×2 Kalman tracker) as the NN policies. The SOTA sensing
frontend — the actual contribution of the project — is identical on both sides.
The NN's only marginal value over MPC is the learned actor replacing a fixed
rule (beam-steer to anchor + always-fire). The data says that marginal value
is ≈ zero (or negative) in this obs/action space.

### 6.2 CTDE reversal — confirmed in cross-play

The training-side reversal (IPPO > MAPPO under EW) reproduces in cross-play:

- Without EW: mappo (0.472) > ippo (0.167) — centralized critic wins in clean env
- With EW: ew_ippo (0.264) > ew_mappo (0.069) — centralized critic loses under anchor noise

This is consistent with the hypothesis that MAPPO's team V-critic learns
wrong credit assignment when the anchor is noisy (the noisy range/cross-range
injects variance into team rewards that the centralized critic can't factor
out). IPPO's per-agent critic sees a cleaner per-agent signal.

This is a real, publishable ablation result — but it does not salvage "AI beats
classical."

### 6.3 Anchors track fused enemy position; both teams drift

`ANCHOR-DD` debug trace (env_id 0, both teams, EW on):

```
n=5  team=0 aim=(526.1, 1560.9)m
n=5  team=1 aim=(10.0,  -34.5)m
n=10 team=0 aim=(1123.9, 3327.9)m
n=10 team=1 aim=(29.4, -100.0)m
```

Team 0's aim drifts by ~600m over 5 control steps (≈2.5 ms real time at
PRF=10kHz, pulses_per_control=5). At `vehicle_speed_ms=20`, physical motion
accounts for only 0.05m — so the drift is Kalman convergence, not physics.
Both teams converge toward different enemy track estimates. The relative
geometry of (own_radars, enemy_radars) determines whose Kalman estimate is
tighter; the rule-based beam-follow controller exploits whichever side it's
on. RL adds a small residual correction on top of the same anchor — not
enough to flip the winner.

## 7. Implications for the EAAI paper framing

The original EAAI-C2 framing ("belief-CTDE beats classical + standard MARL")
requires two independent wins:

1. RL beats classical — **fails** at baseline, before EW
2. Belief-CTDE beats standard MARL under EW — undemonstrated, since (1) fails

Without (1), there is no stage on which belief-CTDE can perform.

## 8. Options for next step

| Option | Effort | Risk | Expected outcome |
|---|---|---|---|
| **A. Pivot to C0+C1** | Low (writing only) | Low | Frame: "multi-base Kalman fusion + IQ-level MARL benchmark." Classical becomes a strong baseline, not a competitor to beat. CTDE-reversal becomes an ablation. **Recommended if paper deadline is tight.** |
| **B. PFSP vs MPC** | ~6h retrain | Medium | Add ClassicalMPC to the league as a frozen opponent; PFSP mixes it in during training. Forces NN to learn anti-beam-follow. **Highest information value if it works.** |
| **C. Weaken MPC** | 1h code + 1h eval | High (reviewer pushback) | Drop Kalman tracker from MPC → raw fused anchor only. Gives NN room. But this isengineering the result; EAAI reviewers will notice. |
| **D. Skip to Gate 1 (belief-CTDE)** | 1-2 days | High | Bet that belief info (covariance trace in obs) lets NN finally beat classical under EW. Doesn't address the "NN doesn't beat classical in clean env" root cause. |

Default recommendation: **B**, with **A** as fallback. **B** has the highest
upside (might fix Gate 0 cleanly); **A** is the safe publication path either
way.

## 9. Reproduce

```bash
cd /home/ubuntu/CODE/FluxPhased-
conda activate fluxphased

# Train (each arm, ~4-5h on RTX PRO 6000)
python main.py --config algo/ew_mappo/code/config.yaml
python main.py --config algo/ew_ippo/code/config.yaml

# Cross-play vs ClassicalMPC (under EW, ~10min each)
python scripts/crossplay_mpc.py \
    --config algo/ew_mappo/code/config.yaml \
    --arms ew_mappo \
    --n-games-per-direction 36 \
    --num-envs 12 \
    --out experiments/crossplay_mpc_ew_mappo.md

python scripts/crossplay_mpc.py \
    --config algo/ew_ippo/code/config.yaml \
    --arms ew_ippo \
    --n-games-per-direction 36 \
    --num-envs 12 \
    --out experiments/crossplay_mpc_ew_ippo.md
```

## 10. Artifacts (this commit)

**New**:
- `algo/ew_mappo/{AGENTS.md, README.md, APP_PUBLICATION.json, environment/README.md, code/config.yaml, sanity_config.yaml, code/.git_commit}`
- `algo/ew_ippo/{AGENTS.md, README.md, APP_PUBLICATION.json, environment/README.md, code/.git_commit}`
- `experiments/crossplay_mpc_ew_mappo.md` (WR=0.069)
- `experiments/crossplay_mpc_ew_ippo.md` (WR=0.264)
- `experiments/EAAI_C2_GATE0_REPORT.md` (this file)

**Modified**:
- `algo/_shared/baselines/classical_mpc.py` (+ jam_gain/exposure_gain kwargs)
- `scripts/crossplay_mpc.py` (+ EW param threading + shared jam_level tensor)

**Sanity check artifacts (not committed)**:
- Checkpoints `algo/ew_mappo/data/checkpoints/iter_{000..019}.pt`
- Checkpoints `algo/ew_ippo/data/checkpoints/iter_{000..019}.pt`
- Training logs `algo/ew_{mappo,ippo}/data/logs/full_run.log`
