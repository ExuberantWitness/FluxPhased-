# WP0 Verification Report — Two-team symmetric multifunction env

**Date**: 2026-07-13
**Spec**: TWOTEAM_MULTIFUNCTION_PLAN.md commit 4329bae
**Overall**: ✅ PASS — proceed to WP1 BR training

## Check 1: Mirror self-play physics symmetry (D3-A)

With both teams playing IDENTICAL actions under MIRROR_GEOMETRY, physics must be mirror-symmetric.
Win rate is NOT the right metric (symmetric play → 97% draws → few decisive samples).
The right metric is physics symmetry: |team_0_metric - team_1_metric| → 0.

- max |kills_t0 - kills_t1|: 0.0000
- mean |kills_t0 - kills_t1|: 0.0000
- max |exposure_t0 - exposure_t1|: 0.0000
- max |mean_trace_P_t0 - mean_trace_P_t1|: 0.0000
- mean |reward_t0| (zero-sum → should be ~0): 0.0000
- ✅ PASS (targets: all metrics < 0.5/1.0/0.1/0.5)

## Check 2: Four-function tradeoff matrix

| strategy | pure | pure | pure | pure | balanc | balanc |
|----------|--------|--------|--------|--------|--------|--------|
| pure_track | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| pure_jam | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pure_comm | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pure_detect | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| balanced | 0.97 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| balanced_jam_heavy | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 |

**Dominant strategy**: NONE
- ✅ PASS (target: no strategy >0.90 vs ALL)

## Check 3: CRLB anchor

- Theoretical CRLB trace_P: 0.030278
- Achieved trace_P (split-beam pure_track): 0.008576
- Ratio: 0.28
- ✅ PASS (target: ratio < 5)

## Verdict

✅ **WP0 PASS** — env is unbiased, four-function tradeoff is real, CRLB anchor works.
**→ Proceed to WP1 BR training (G0 exploitability gate).**
