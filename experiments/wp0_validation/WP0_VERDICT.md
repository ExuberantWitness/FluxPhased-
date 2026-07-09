# WP0 Validation Verdict
**Run**: dt=0.1, e_kill=2.0, dwell_rate=1.0, jam_gain=8.0, exposure_gain=50.0
**Cells**: 6 | **Seeds**: 5 | **Envs/seed**: 4 | **Steps**: 600
**Elapsed**: 92.9s

## Summary
| Cell | N | Jammer | kill_rate | ttk_first | survived | track_loss | P/PCRLB | exposure_end |
|---|---|---|---|---|---|---|---|---|
| n1_l0 | 1 | L0 | 1.00±0.00 | 20±0 | 1.00 | 0.000 | 1.77 | 0.3 |
| n4_l0 | 4 | L0 | 1.00±0.00 | 20±0 | 1.00 | 0.000 | 3.53 | 1.0 |
| n4_l1 | 4 | L1 | 0.89±0.08 | 27±2 | 0.85 | 0.593 | 48.42 | 2.3 |
| n4_l3 | 4 | L3 | 0.39±0.40 | 315±237 | 0.65 | 0.815 | 55.93 | 2.6 |
| n8_l0 | 8 | L0 | 0.97±0.05 | 20±0 | 0.90 | 0.000 | 4.01 | 1.9 |
| n8_l3 | 8 | L3 | 0.23±0.18 | 267±188 | 0.55 | 0.928 | 56.35 | 2.8 |

## Gate Checks (spec §1.6)
### Sanity (n1_l0): kill=1.00 (target >0.8: **PASS**), ttk=20 (target ≈20: **PASS**), P/PCRLB=1.77 (target [0.5, 3.0]: **PASS**)

### Hard regime (n4_l0 vs n4_l3): kill drop = 0.61 (target >0.20: **PASS**), ttk blow-up 20→315 (target 2×: **PASS**)

### Extreme regime (n8_l3): kill=0.23, ttk=267, survived=0.55

## Verdict
✅ **WP0 PASS** — env mechanics correct, hard regime established.
Proceed to WP1: build IMM-PDAF (Stone Soup) + fictitious-play baseline, verify strong classical is non-strawman at low difficulty.
