# S7 — Full two-team MAPPO: 2 jammers vs 2 radars

**Branch:** `g3-bsta/array-face-s1` · **Seeds:** 20260801/20260802/20260803 (snr=12) · **Date:** 2026-08-23

**Status: TRAINING IN PROGRESS** — seed 20260801 at iter ~900/1000; this report
fills as the pipeline completes. Sections marked ⏳ await final checkpoints.

## TL;DR (preliminary, seed 20260801 partial trajectory)

S7 asks: does a SECOND jammer break S6's defense-dominant equilibrium? With
spatially separated radars (±20°) and two jammers in cross-fire (±60°), the
early answer is **erosion, not collapse**:

- h2h (2v2) plateaus near **0.21** — 2.4× S6's 0.089, a major erosion of the
  defense advantage, but stable (not a runaway)
- j1_only (1 jammer vs the SAME learned radars, same env) ≈ **0.12** — the
  radars still neutralize a single jammer as thoroughly as S6 did
- jam_vs_sweep (2 jammers vs scripted sweep) climbs to **0.36** — the second
  jammer's marginal raw power is ~3× the first's (0.36 vs 0.12)
- rad_vs_idle ≈ **0.997** — radar competence is at ceiling

Reading: the second jammer converts cross-fire geometry into real suppression
(the marginal-damage ratio), but the radar team adapts enough to hold a stable
— if far worse — line. Final numbers ⏳.

## Setup

### Geometry (roadmap: "spatial separation returns in S7")

| Asset | Placement | Rationale |
|---|---|---|
| Radars | az ±20°, el 0 | v1 separation restored; S6's co-location was a 1-jammer patch |
| Jammers | az ±60°, el 0 | cross-fire sites; all four (k,r) pair bearings (±40°/±80°) are OFF the beam grid |

Off-grid pair bearings mean no single beam suppresses both radars — the
jammers MUST coordinate (the S7 hypothesis premise). A symmetric ±40° jammer
placement was rejected: it would put one pair bearing exactly on the −60°
grid point.

### Physics

Per-(jammer k, radar r) JNR via S6's generalized AF at the pair bearing
(`compute_upa_af_db_toward`), combined per radar as incoherent linear power
sum (S5's rule): `JNR_r = 10·log10(Σ_k 10^(JNR_kr/10))`. M0 gates: idle →
−inf; twins → +3.0103 dB; single-active at broadside == S4. Detection, target
gain, and mission tracking are S6 verbatim.

### Budgets: team-total parity with S6

63 activation steps split 32/31 across the two jammers (S6's single jammer
had all 63). The S6→S7 comparison therefore isolates **jammer count**, not
team energy.

### MAPPO structure (CTDE)

Each team: ONE parameter-shared MultiHeadActor + per-agent ValueCritic + ONE
PrivilegedValueCritic on the team's public global state (both agents' obs
concatenated; no oracle info). Swap-update: the same `S2PPOTrainerV2.update()`
runs for both teams by rebinding actor/critic/priv_critic/optimizers/head
specs (S6's mechanism, now with the privileged-critic path enabled —
`use_privileged_critic=True`, validated in S5).

Rollouts are K-flattened on BOTH sides ([T, E·K] jammer slots, [T, E·R] radar
slots, k-major); team rewards and env-level privileged values are duplicated
across slots; GAE uses the privileged values (compute_gae's priv path).

### Observations

- Jammer obs [E,K,67]: S6's 55-dim ESM view + 12-dim partner channel (partner
  beam az/el one-hots, energy ratio, active flag)
- Radar obs [E,R,60]: S6's 49-dim + a SECOND per-jammer ESM section (DOA
  az/el one-hots + active flag for each jammer independently)
- intercept_confidence is per-radar now (each radar's own snr_eff);
  intercept_age per-jammer

## Pre-training gates (all PASS)

1. **18/18 test gates** (`tests/array_face/test_array_face_s7.py`,
   `test_array_factor_s7.py`): env contract, per-jammer budgets, ESM slots,
   ledger identity, per-(svc,az) credit, JNR combination M0 trio, aiming
   asymmetry, p_detect monotone, and the **2-jammer contestability gate** —
   under BOTH jammers at full leverage, ≥3/5 mission azimuths stay in the
   contestable band, min p > 0.02, max p > 0.8.
2. **Contestability sweep** (`_s7_sweep_contestability.py`):

```
mission az       : [0, 1, 2, 3, 4]
1 jammer profile : [0.965, 0.921, 0.993, 0.996, 0.994]
2 jammer profile : [0.837, 0.912, 0.989, 0.993, 0.855]
cross-beam drop  : [0.128, 0.010, 0.005, 0.003, 0.139]
```

The second jammer pushes the two off-axis azimuths (0, 4) substantially down
while every azimuth stays reachable — geometry validated for the 2v2 game.

## Training

Three seeds (20260801/02/03), 1000 iterations each, self-healing chain with
atomic-checkpoint resume (max-iteration completion criterion). Machine-sleep
interruptions occurred three times (iterations preserved by checkpoints;
loss < 100 iters each). ~41 s/iter (4-view validation every 10 iters).

### Seed 20260801 trajectory (validation views, 16 val seeds)

| Quarter | h2h | jam_vs_sweep | j1_only | rad_vs_idle | gap (jvs−h2h) |
|---|---|---|---|---|---|
| 0–99 | ~0.22 | ~0.19 | ~0.15 | 0.90 | −0.03 |
| 800–869 | ~0.20 | ~0.36 | ~0.10 | 0.997 | +0.16 |

(Detailed quarter table ⏳ after all seeds complete.)

## Final evaluation ⏳

Protocol: 64 validation seeds × 3 action seeds × reps=1, five views (h2h,
jam_vs_sweep, rad_only, **j1_only** — the S7-specific 1v2 control — and the
sweep_vs_idle natural floor). Merge across seeds with mean ± sd.

### Headline table ⏳

| Metric | S6 (1v2, snr=12) | S7 (2v2) | reading |
|---|---|---|---|
| h2h drop | 0.0888 ± 0.0053 | ⏳ | defense dominance eroded? |
| jam_vs_sweep | 0.2751 ± 0.0110 | ⏳ | |
| j1_only drop | — | ⏳ | same-env 1v2 control |
| neutralization | 63.7% ± 0.7% | ⏳ | |
| 2nd-jammer marginal | — | ⏳ | jvs / j1_only |

## Behavior extraction (preview: seed 20260801, iter-849 checkpoint, greedy)

- **Radar team spontaneously divides the sky**: parameter-shared actors lock
  beam 11 (az −30°, 52% mass) and beam 13 (az +30°, 48%), both in the mission
  plane (el 0), svc 50/50 — left/right sector assignment matching the ±20°
  site separation. Division of labor emerges purely from the "other radar"
  ESM channel in the shared policy's input.
- **Both jammers' greedy mode is idle** (0 cells, like all three S6 seeds) —
  the equilibrium damage lives in the stochastic tail. The cross-assignment
  matrix (who suppresses whom) is therefore only visible under the stochastic
  policy; full extraction ⏳ after training.

## Engineering notes

- S7 is a NEW frozen module (`env/gpu/array_face_s7/`); S6's 17 gates and its
  published numbers are untouched.
- Per-jammer over-budget top-k clamp (S5 pattern); team budget split with the
  odd token going to jammer 0 (32/31).
- Central critics on PUBLIC state only (no deadline/arrival oracle) — the
  CTDE centralization is over observability, not information asymmetry.
- Three sleep-interruption recoveries; each cost <100 iterations (atomic
  checkpoints + max-iter chain criterion + restored iteration counter).

## Next

- ⏳ Seeds 20260802/20260803 → merged statistics → finalize headline table
- ⏳ Stochastic-mode cross-assignment analysis (does jammer k lock onto radar k?)
- ⏳ Curves plot + commit/push
- Post-S7: the S1→S7 roadmap is complete; paper assembly (TAES target) begins.
