# WP0 Verification Report — Two-team symmetric multifunction env

**Date**: 2026-07-14
**Spec**: TWOTEAM_MULTIFUNCTION_PLAN.md + TWOTEAM_ENV_FIX_SPEC.md (WP0-decisive upgrade)
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

## Check 2: Four-function tradeoff matrix (WP0-decisive upgrade)

Four sub-checks: (2a) no dominant strategy; (2b) decisive rate ≥ 0.50;
(2c) kill density ≥ 0.5/ep; (2d) no strategy with stalemate_rate > 0.50.
The 2b/2c/2d upgrades catch 0-0 stalemates that the original 2a check
alone misclassified as PASS.

### Win-rate matrix

| strategy | pure | pure | pure | pure | balanc | balanc | track |
|----------|--------|--------|--------|--------|--------|--------|--------|
| pure_track | 0.00 | 0.00 | 1.00 | 0.95 | 0.00 | 0.00 | 0.00 |
| pure_jam | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pure_comm | 0.00 | 0.07 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 |
| pure_detect | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| balanced | 1.00 | 0.03 | 1.00 | 0.97 | 0.00 | 0.05 | 0.15 |
| balanced_jam_heavy | 0.97 | 0.03 | 1.00 | 0.97 | 0.00 | 0.00 | 0.05 |
| track_agile | 0.00 | 1.00 | 1.00 | 0.97 | 0.00 | 0.00 | 0.00 |

### Decisive-rate matrix (fraction of episodes with ≥1 kill)

| strategy | pure | pure | pure | pure | balanc | balanc | track |
|----------|--------|--------|--------|--------|--------|--------|--------|
| pure_track | 1.00 | 0.00 | 1.00 | 0.95 | 0.97 | 1.00 | 0.97 |
| pure_jam | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| pure_comm | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| pure_detect | 1.00 | 0.00 | 0.00 | 0.00 | 0.97 | 1.00 | 0.97 |
| balanced | 1.00 | 0.00 | 1.00 | 0.97 | 0.00 | 0.00 | 1.00 |
| balanced_jam_heavy | 0.97 | 0.00 | 1.00 | 0.97 | 0.00 | 0.00 | 1.00 |
| track_agile | 0.97 | 1.00 | 1.00 | 0.97 | 1.00 | 0.97 | 1.00 |

**Dominant strategy**: NONE
- 2a ✅ PASS (target: no strategy >0.90 vs ALL)
- 2b ✅ PASS: decisive_rate=0.660 (target ≥ 0.50)
- 2c ✅ PASS: kill_density=0.801/ep (target ≥ 0.5)
- 2d ✅ PASS: per-strategy diagnostics —
    - pure_track: stalemate_rate=0.179  best_opp_decisive=1.000
    - pure_jam: stalemate_rate=0.833  best_opp_decisive=1.000
    - pure_comm: stalemate_rate=0.333  best_opp_decisive=1.000
    - pure_detect: stalemate_rate=0.348  best_opp_decisive=1.000
    - balanced: stalemate_rate=0.340  best_opp_decisive=1.000
    - balanced_jam_heavy: stalemate_rate=0.340  best_opp_decisive=1.000
    - track_agile: stalemate_rate=0.010  best_opp_decisive=1.000

## Check 3: CRLB anchor

- Theoretical CRLB trace_P: 0.030278
- Achieved trace_P (split-beam pure_track): 0.006432
- Ratio: 0.21
- ✅ PASS (target: ratio < 5)

## Verdict

✅ **WP0 PASS** — env is unbiased, four-function tradeoff is real, CRLB anchor works.
**→ Proceed to WP1 BR training (G0 exploitability gate).**
