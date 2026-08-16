# S4 Report — 2D UPA Beam Steering + 25-Cell Binding (IQ-Level EW Benchmark)

**Stage**: S4 of the S1→S7 roadmap (HANDOFF §11.2)
**Environment**: 5×5 UPA both sides, Categorical(25) 2D beam + Bernoulli(25) cell binding,
no base head (idle = all-zero cells), no service head (always matched), 63-token budget.
**Config (final, "expD")**: PPO, no reward shaping beyond pending-potential, shared trunk,
beam entropy anneal_frac = 0.9 (extended exploration window), cell sparse-init bias −3.0.

## 1. Headline Results (3 seeds × 1000 iters)

| seed | val_drop@999 | idx12 (broadside) share | beam entropy | cell entropy | mean cells |
|------|-------------|------------------------|--------------|--------------|------------|
| 20260729 | **0.1080** | 0.44 | 1.81 | 2.90 | 1.0 |
| 20260730 | 0.0980 | 0.85 | 0.89 | 2.82 | 1.0 |
| 20260801 | 0.0770 | 0.95 | 0.41 | 3.45 | 0.9 |
| **mean** | **0.0943 ± 0.0158** | **3/3 → broadside** | | | |

Cross-stage comparison (same 3-seed protocol, 64 val scenarios):

| Stage | Action space | val_drop (PPO) | Oracle ceiling | Relative efficiency |
|-------|-------------|----------------|----------------|---------------------|
| S2 | 3×5 MultiDiscrete | 0.2114 ± 0.0006 | — | — |
| S3 | + Bernoulli(5) cells | 0.4229 ± 0.0095 | 0.538 | 79% |
| S4 | Bernoulli(25) + Categorical(25) | **0.0943 ± 0.0158** | **0.132** | **71%** |

## 2. The Physics Ceiling: Why S4's Absolute Number Is 4× Smaller

Oracle experiments (fixed 1-cell continuous jamming, 64 val seeds) measure the
environment's achievable maximum:

| Oracle policy | S4 drop | note |
|---------------|---------|------|
| idle | 0.0119 | floor |
| beam idx12 (broadside), 1 cell | **0.1322** | **ceiling** |
| beam idx12, 2 cells | 0.1039 | budget burn halves coverage |
| beam idx12, 3 cells | 0.1242 | |
| beam idx17 (el=+15° ridge), 1 cell | 0.0547 | symmetric local optimum |
| beam idx2 (el=−30°), 1 cell | 0.0341 | |

The 25-direction 2D radar Rx sweep is the cause. The jammer's signal enters the
radar's receive pattern at full gain only 1/25 of steps (vs 1/5 in S3); the mean
Rx loss over the sweep is ≈ −22 dB (vs ≈ −13.5 dB in S3). During off-broadside
sweep steps JNR falls 10–30 dB, p_detect rises sharply, and most missions are
still detected → the timeout (drop) ceiling collapses from 0.538 to 0.132.

**Interpretation**: adding the 2D dimension made the *defender* 4× stronger, not
the learning problem proportionally harder. PPO retains ~71–82% relative
optimality (vs 79% in S3) and *spontaneously* learns (a) energy minimization —
1 cell/step so the 63-token budget covers ~63 of 64 steps, matching the oracle —
and (b) broadside aiming (all 3 seeds, no exceptions).

## 3. The Diagnostic Journey (negative results that matter)

### 3.1 Initial misdiagnosis: "beam head lacks gradient signal"

The first S4 run (v1: shared trunk, beam anneal 0.5) reached val ≈ 0.1035 with
the beam head only 47% concentrated. We hypothesized the beam head was
starved of reward signal and added (1) potential-based beam-alignment shaping
and (2) an independent beam trunk (v2).

**v2 result**: beam locked 97.8% onto idx17 (az=0°, el=+15°) — a *symmetric
local ridge* at −6.9 dB TxAF — and val dropped to 0.0830.

### 3.2 Ablation (400-iter runs, all falsified the shaping hypothesis)

| Variant | val@399 | beam locked onto |
|---------|---------|------------------|
| v1 (no shaping) | **0.0985** | **idx12 ✓** |
| v2 (avg shaping 0.01 + trunk) | 0.0464 | idx17 ✗ |
| expA (tx-only shaping 0.003 + trunk) | 0.0494 | idx17 ✗ |
| expB (avg shaping 0.001 + trunk) | 0.0403 | idx17 ✗ (not locked, still 2.4× behind) |
| expD (no shaping, anneal 0.9) | 0.0994 | idx12 ✓ |

p_detect measurements show the true reward *already* distinguishes idx12 from
idx17 by 100× (1.5e-4 vs 1.4e-2): there was no tie to break. The shaping's
coarse proxy (coef × TxAF differences ≈ 0.02/step) could not resolve ridge
members, pushed early commitment before entropy annealing froze exploration,
and (expB evidence) polluted the critic's regression targets with
radar-sweep-correlated noise of the same magnitude as the true reward.

**Lesson**: potential-based shaping guarantees policy invariance, not
acceleration. A weak proxy of an already-strong signal only adds noise.

### 3.3 Correct diagnosis: exploration window too short

Beam entropy annealed to 0 at iter 200 (frac 0.5), freezing the beam head
before the true 100× signal could resolve broadside. expD (anneal 0.9 →
exploration to iter 360/900 at 400/1000-iter scales) confirmed: mid-training
speed +66% over v1, firm broadside convergence, final 0.1080 vs 0.1035.

## 4. Notable Secondary Findings

1. **Beam-lock ↔ performance inversion across seeds**: the seed with the
   softest beam lock (44%) achieved the best val (0.1080); the hardest lock
   (95%) the worst (0.0770). Residual headroom lies in the *cell/timing*
   policy, not beam precision — over-committing the beam head appears to
   crowd out cell-policy exploration.
2. **Energy minimization is optimal under this budget regime**: all oracle
   multi-cell variants underperform 1-cell continuous jamming (2 cells: 0.104,
   3 cells: 0.124 vs 0.132), explaining why greedy all-cells baselines fail.
3. **Sampler bug (fixed)**: `torch.rand` in float32 returns exactly 0.0 with
   p≈6e-8; combined with `clamp(min=1e-12)` on masked Bernoulli cells this
   sampled illegal ON cells at exhausted energy (~1.5e-3 crash risk/iter).
   Fixed by forcing masked cells' sampled actions to zero
   (`actor_heads.py::sample_multihead`); regression-tested.

## 5. Artifacts

- `run_s4_ppo.py` — parameterized driver (shaping mode/coef, trunk, anneal, iters)
- `trainer_s4.py`, `env/gpu/array_face_s4/*` — trainer + env (beam shaping code
  retained but **disabled** in the final config: `beam_shaping_coef=0`)
- `s4_ppo_output_seed{20260729,20260730,20260801}_expD_1k/` — final 3-seed runs
- `s4_ppo_output_seed20260729_{v1_baseline,v2_shaping_trunk,expA_txonly,expB_lowcoef,expD_anneal09}/` — ablation history
- `s4_expD_vs_v1_v2_curves.png`, `s4_v1_vs_v2_training_curves.png` — curves

## 6. Implications for S5 (two-jammer cooperation)

- S4's single-jammer saturation point (0.094 ± 0.016, 71% of a 0.132 ceiling)
  is the control condition for measuring cooperation gains.
- The ceiling analysis generalizes: a second jammer's value will be gated by
  Rx-sweep geometry (temporal coverage of broadside steps), suggesting
  time-division coordination (staggered energy budgets) as the natural
  cooperative structure for IPPO + central critic to discover.
- The anneal-frac finding (0.9) should carry over to per-jammer beam heads.
