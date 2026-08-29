# Paper Claim Audit Report

**Paper:** *Scaling the Attack Breaks Defense Containment in Task-Level Radar--Jammer Self-Play*
**Audit scope:** current `paper/main.tex` and `paper/sections/*.tex` against raw S6/S7/R5 `final_eval.json` files.
**Status:** corrected after independent zero-context review.

## Verdict

**PASS for reported numeric values and stated scope.** The manuscript now uses the valid two-seed 12-dB S6 baseline, three converged S7 seeds at 3000 iterations, the corrected co-located-jammer evaluation, and the R5 ratio metrics with their gradient-budget limitation.

## Verified claims

| Claim family | Evidence | Status |
|---|---|---|
| S6 valid 12-dB h2h/jvs | `experiments/array_face_s6/learning_repair/s6_selfplay_output_seed20260730/final_eval.json`, seed 20260731 | Match: h2h 0.0888 +/- 0.0053, jvs 0.2751 +/- 0.0110 |
| S6 neutralization | same two valid files, floor-adjusted formula | Match: 63.7% with across-seed SD rounded to 1.0 percentage point |
| S7 converged h2h/jvs | `s7_continue2_output_seed20260801`, `s7_seed02_cont_output_seed20260802`, `s7_seed03_cont_output_seed20260803` | Match: 0.3366 +/- 0.0143 and 0.5294 +/- 0.0215 across three training seeds |
| S7 idle-radar floor | same three S7 files | Match: 0.0194 +/- 0.0008 |
| S7 neutralization | same three S7 files, explicit eta formula | Match: 23.0% +/- 1.1% |
| Mechanism control | corrected `s7_ablation_output_seed20260811/final_eval.json` | Match: h2h 0.2927 +/- 0.0119, jvs 0.4947 +/- 0.0026, eta 28.4%; scope limited to separated S7 radar geometry |
| R5 dose response | `s7_continue_output_seed20260801` and three `s7_r5_mix*` files | Match: j1/jvs 0.502 -> 0.440 -> 0.277 -> 0.262; eta 20.2 -> 23.5 -> 26.1 -> 24.6 |

## Corrections applied

- Changed S6 sample description from three seeds to two valid 12-dB seeds.
- Replaced stale S6 rad-idle value 0.0110 with valid aggregate 0.0275 +/- 0.0110.
- Replaced S6 eta uncertainty 0.7 percentage points with the valid two-seed estimate rounded to 1.0 percentage point.
- Changed Figure 3 uncertainty caption from action-seed SD to training-seed SD.
- Removed unsupported ``pre-registered'' wording; the manuscript now says ``pre-specified'' or ``matched.''
- Limited count attribution: S6-to-S7 changes attacker count and radar-site geometry; the co-located control decomposes the two effects only within the separated S7 geometry.
- Specified that S6 does not contain the S7-specific j1-only view.
- Corrected R5 h2h/jvs wording: it reaches its minimum at 50% exposure and is slightly higher at 75%.
- Replaced ``remains unchanged'' with ``remains low'' where the valid S6/S7 floors are not identical.

## Residual submission notes

- The mechanism control has one training seed.
- R5-lite has one training seed per mixture condition and skips jammer updates on singleton iterations; absolute drops are not causal comparisons across mix fractions.
- The manuscript's claims are benchmark-level and simulation-based, not universal deployed-radar claims.
