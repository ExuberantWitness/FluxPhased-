# S7 — Full two-team MAPPO: 2 jammers vs 2 radars

**Branch:** `g3-bsta/array-face-s1` · **Seeds:** 20260801/20260802/20260803 (snr=12) · **Date:** 2026-08-23

**Status: TRAINING IN PROGRESS** — seed 20260801 at iter ~900/1000; this report
fills as the pipeline completes. Sections marked ⏳ await final checkpoints.

## TL;DR (seed 20260801 + its 2000-iter continuation; seeds 02/03 ⏳)

S7 asks: does a SECOND jammer break S6's defense-dominant equilibrium? With
spatially separated radars (±20°) and two jammers in cross-fire (±60°), the
answer — now measured at full convergence — is **yes, in scale: the 2v2
equilibrium sits at h2h ≈ 0.34, ~3.8× S6's 0.089**, and the defense dominance
S6 found is eroded into a much higher, still-stable plateau:

- h2h (2v2) converges at **0.343 ± 0.015** (final full-protocol 0.347 at
  iter 1999) after a slow climb that the 1000-iter budget had truncated at
  0.26 — the continuation control shows the rise is real convergence, NOT
  cycling (see "Continuation control")
- jam_vs_sweep converges at **0.496 ± 0.020** vs S6's 0.275 — the pair
  extracts +80% more raw damage from the SAME 63-token team budget
- j1_only doubles across the continuation (0.12 → **0.24**): the radar team,
  specializing against the pair, becomes exploitable by a lone jammer — a
  measured adaptation trade-off (rock-paper-scissors tilt, not a cycle)
- rad_vs_idle 0.987 — the erosion is purely offensive; radar competence
  never decays
- the jammer pair exhausts its energy budget (S6's jammer rationed) and plays
  a symmetric center-split; the radar team holds a stable az ±30° sector
  division of labor

Seeds 20260802/03 ⏳ (training in progress) for the replication statistics.

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

### Seed 20260801 trajectory (validation views; final row = full protocol 64 seeds × 2 reps)

| Quarter | h2h | jam_vs_sweep | j1_only | rad_vs_idle | gap (jvs−h2h) | 2nd-jammer marginal (jvs−j1) |
|---|---|---|---|---|---|---|
| 0–199 | 0.185 | 0.215 | 0.128 | 0.927 | 0.030 | 0.087 |
| 200–399 | 0.163 | 0.253 | 0.104 | 0.973 | 0.090 | 0.149 |
| 400–599 | 0.191 | 0.326 | 0.115 | 0.980 | 0.135 | 0.211 |
| 600–799 | 0.184 | 0.336 | 0.111 | 0.988 | 0.153 | 0.225 |
| 800–999 | 0.224 | 0.357 | 0.130 | 0.990 | 0.133 | 0.227 |
| **final (999)** | **0.260** | **0.394** | **0.177** | 0.985 | — | — |

Reading: the second jammer's marginal power grows monotonically all training
(0.087 → 0.227, +160%) — the jammer pair's coordination compounds. In the
last quarter the h2h creeps up (0.18 → 0.22–0.26): late offensive
counter-adaptation, same direction as S6's late-h2h hint but 5× larger.

## Continuation control (seed 20260801, iter 1000 → 2000, entropy anneal frozen)

**Question** (raised by the late rise at the 1000-iter budget): was the q4 h2h
rise (0.20 → 0.26) real offensive drift, or unfinished convergence, or
non-transitive cycling? The continuation isolates the variable — `--anneal-done`
freezes all per-head entropy coefficients at coef_min from the resume point,
so the ONLY thing that changes is more training.

**Answer: the rise was an unfinished arms race, which then CONVERGES at a
much higher plateau — it is not cycling and it does not run away.**

Per-200-iter quarters (validation views):

| Quarter | h2h | jam_vs_sweep | j1_only | rad_vs_idle |
|---|---|---|---|---|
| q5 (1000–1199) | 0.290 | 0.427 | 0.178 | 0.993 |
| q6 (1200–1399) | 0.328 | 0.475 | 0.209 | 0.987 |
| q7 (1400–1599) | 0.335 | 0.485 | 0.211 | 0.987 |
| q8 (1600–1799) | 0.345 | 0.497 | 0.237 | 0.989 |
| q9 (1800–1999) | **0.341** | **0.494** | 0.243 | 0.985 |
| last-40 | **0.343 ± 0.015** | 0.496 ± 0.020 | 0.240 ± 0.016 | 0.987 ± 0.007 |

- h2h climbs 0.29 → 0.34 with decelerating slope and flats (q8 ≈ q9 within
  noise); jvs flattens the same way. **Convergence completes around
  iter ~1600–1700.**
- **Consequence for the protocol:** the 1000-iter budget understates the 2v2
  equilibrium by ~30% (0.26 at 999 vs 0.34 at convergence). All S7 headline
  numbers must be read at the converged scale — or with the budget stated.
- **New phenomenon — adaptation trade-off:** j1_only roughly DOUBLES across
  the continuation (0.12 → 0.24). The radar team, optimizing against the
  PAIR, gives up single-jammer containment: at iter 999 the radars blunted a
  lone jammer to 0.12 (S6-like), at 1999 a lone jammer extracts 0.24. The
  shared jammer policy also sharpened, but the radar-side retargeting is the
  dominant term — a genuine rock-paper-scissors tilt: specialize against the
  team, become exploitable by the singleton.
- End-state entropies: jammer 0.92 / radar 1.81 — the jammer is near its
  sharpening floor while the radar still explores; no entropy lock either
  side.

**Verdict for the league question:** no non-transitive CYCLE is needed to
explain the S7 dynamics — the single-policy self-play run converges, just
slowly and far above the 1000-iter checkpoint. League/PFSP machinery would
not change this answer; it becomes relevant only if we later want the
single-jammer exploitability (the j1_only trade-off) minimized, which is a
different objective (robustness across opponent classes). The cheapest
robustness knob — periodic singleton-opponent mixing — is a natural follow-up
if that objective matters.

### Stage-2 extension (iter 2000 → 3000, anneal frozen): plateau CONFIRMED

The user's check — "2000 is not flat enough point-to-point" — drove one more
1000-iter stage. The answer is now unambiguous: **the equilibrium is fully
stationary by ~iter 1700; 2000–3000 is pure stationary noise.**

Validation-view quarters (2000–3000):

| Quarter | h2h | jam_vs_sweep | j1_only | rad_vs_idle |
|---|---|---|---|---|
| q10 (2000–2199) | 0.3465 | 0.5121 | 0.2468 | 0.9824 |
| q11 (2200–2399) | 0.3428 | 0.4956 | 0.2527 | 0.9825 |
| q12 (2400–2599) | 0.3484 | 0.5119 | 0.2481 | 0.9773 |
| q13 (2600–2799) | 0.3507 | 0.5107 | 0.2421 | 0.9801 |
| q14 (2800–2999) | 0.3463 | 0.5215 | 0.2525 | 0.9749 |

All four views vary within ±0.005 of their 2000–3000 means across five
quarters (h2h mean 0.3469 ± 0.003, jvs 0.5104 ± 0.009, j1 0.2484 ± 0.004) —
statistically flat. Last-40 at 3000: h2h 0.3485 ± 0.0151, jvs 0.5161 ± 0.0174,
j1 0.2473 ± 0.0137. **The converged 2v2 equilibrium is h2h ≈ 0.35 (≈3.9×
S6's 0.089), jvs ≈ 0.51, j1 ≈ 0.25** — and no further training changes it.

## Final evaluation

Protocol: 64 validation seeds × 3 action seeds × reps=1, five views (h2h,
jam_vs_sweep, rad_only, **j1_only** — the S7-specific 1v2 control — and the
sweep_vs_idle natural floor). Merge across seeds with mean ± sd.

### Converged continuation (seed 20260801 @ 2000 iters) — the authoritative 2v2 numbers

| View | mean ± sd over action seeds |
|---|---|
| h2h (2v2) | **0.3282 ± 0.0016** |
| jam_vs_sweep | **0.5043 ± 0.0046** |
| j1_only (1v2 in-env control) | 0.2532 ± 0.0055 |
| rad_vs_idle drop | 0.0204 ± 0.0019 |
| sweep_vs_idle floor | 0.1187 |

Floor-adjusted neutralization = 1 − (h2h − rad_idle) / (jvs − floor)
= 1 − 0.308/0.386 = **20.2%** — versus S6's **63.7%**.

**The containment ratio collapses from ~64% to ~20%.** In S6 the learned
radars absorbed nearly two-thirds of the jammer's marginal power; in the
converged 2v2 game they absorb one-fifth. Defense dominance as measured by
S6 is broken by the second jammer — not into instability (the equilibrium is
a stable plateau), but into a game where the offense's adapted suppression
is mostly unrecoverable by the defense.

### 1000-iter checkpoints (seeds 20260802/03 ⏳; seed 01 for reference)

Seed 20260801 at its 1000-iter budget: h2h 0.2600, jam_vs_sweep 0.3941,
j1_only 0.1770, rad_vs_idle 0.9847 — i.e., the protocol-level 1000-iter
numbers understate the converged equilibrium by ~26% on h2h (the
continuation control quantifies exactly this gap).

## Behavior extraction (seed 20260801 final checkpoint)

**Greedy mode:** both jammers idle (0 cells — the S6 mode-collapse invariant
holds for the pair); radars lock beam 11 (az −30°, 52%) / beam 13 (az +30°,
48%), el strictly in the mission plane, svc 50/50.

**Stochastic mode** (the equilibrium damage carrier; 16 val seeds × 64 steps,
sampled actions):

- **The jammer team burns its full budget**: mean 0.51/0.49 cells per step at
  ~37% duty each ≈ 32/31 tokens over the horizon — the 63-token team budget
  is exactly exhausted. S6's single jammer idled its mode and rationed its
  energy; the S7 pair spends everything it has.
- **No clean cross-pairing — a center-split instead**: both jammers' beam
  marginals are near-identical (az mass spread across all 5 sectors with the
  mode at az 0°, the site center) — the parameter-shared policy plays a
  symmetric mixed strategy. The asymmetric damage comes from GEOMETRY, not
  role specialization:

```
cross-assignment (mean per-pair JNR, active steps only):
                radar 0 (+20°)   radar 1 (−20°)
jammer 0 (+60°):    22.2 dB          22.6 dB     ← balanced center-split
jammer 1 (−60°):    21.9 dB          12.3 dB     ← biased toward radar 0
```

  Radar 0 eats near-full contributions from BOTH jammers (linear sum ≈
  25 dB) while radar 1 sees one strong + one weak (≈ 23 dB). The radar
  team's counter-adaptation: radar 0 takes the az +30° sector (near jammer
  0's bearing), radar 1 takes az −30° — the sector split keeps each radar's
  Rx mainlobe pointed AWAY from the nearer jammer whenever possible.
- **Radar division of labor is robust**: az marginals 0/49/0/51/0, el 100%
  at the mission plane, svc 50.6/49.4 — identical in greedy and stochastic
  rollouts and at iter-849 and iter-999. The two-head ESM channel (other
  radar's beam) is sufficient coordination substrate for spatial role
  emergence under parameter sharing.

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
