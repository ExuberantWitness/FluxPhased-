# Figure & Table Plan — TAES Submission

Every figure is a belief-change step; data sources are all in-repo.

| # | Figure (belief change) | Reader question | Claim | Data source | Status |
|---|---|---|---|---|---|
| 1 | Benchmark anatomy + contestability oracle spectra (1-jammer vs 2-jammer best-response profiles; S6 snr-12/22 + S7 cross/co-located) | Why trust the game? | The oracle predicts balance before training | `_s7_sweep_contestability.py` outputs; S6 sweep in REPORT.md | **final: fig1_benchmark_anatomy.pdf** |
| 2 | 3000-iter four-view trajectory, 3-segment (1000/2000/3000), plateau band + S6 reference lines | Is this converged or drifting? | Equilibrium, not cycling/undertraining | `arms_race_curves_s7_full.png` (regenerate as PDF, single-column) | **final: fig2_training_curves.pdf** |
| 3 | **Money figure**: neutralization 63.7%±1.0 → 23.0%±1.1 (two bars + seed dots) | What did scaling the attack do? | Containment collapse, replicated | 3-seed merged stats (this session) | **final: fig3_neutralization.pdf** |
| 4 | Mechanism decomposition: S6 / co-located / cross-fire neutralization bars + pre-specified spectra below | Count or geometry? | Count primary (→28.4%), geometry secondary (→24.2%) | ablation `final_eval.json` + sweep profiles | **final: fig4_mechanism_ablation.pdf** |
| 5 | Behavior panel: (a) radar az-sector division (beam 11/13 shares), (b) cross-assignment JNR matrix, (c) budget-exhaustion time series vs S6 rationing | What does the equilibrium look like? | Interpretable structure at equilibrium | `_s7_policy_extract.py` outputs | **final: fig5_behavior.pdf** |
| 6 | [R5] Dose-response: j1/jvs (singleton leverage) and eta vs mix in {0,.25,.5,.75}; saturation between 50-75% | Can mixing fix the trade-off? | Mixing halves singleton leverage AND raises eta (ratio-level free lunch) | R5-lite final evals (4 conditions, complete 2026-08-29) | **final: fig6_r5_dose_response.pdf** |

| T# | Table | Content | Status |
|---|---|---|---|
| T1 | Env/physics parameters | grids, budgets, deadlines, link-budget constants | from code |
| T2 | Three-seed converged statistics | h2h/jvs/j1/rad_idle/floor/η per seed + mean±sd | final |
| T3 | S6 vs S6-regime (snr 12/22) invariance | η ≈ const across regimes | from S6 REPORT |
| T4 | Mechanism ablation | cross vs co-located, all views + η | final |
| T5 | [R5] mixing dose-response | mix → h2h, j1_only, η (ratio-metric claims) | **final** (2026-08-29) |

## Key numbers quick sheet (final)

- S6 (1v2, 2 valid 12-dB training seeds, snr=12): h2h 0.0888±0.0053, jvs 0.2751±0.0110, η 63.7%±1.0%
- S6 snr=22 (1 seed): h2h 0.0267, η 60.5% (regime invariance)
- S7 (2v2, 3 seeds @3000): h2h 0.3366±0.0143, jvs 0.5294±0.0215, j1 0.2086±0.0745, rad_idle 0.0194±0.0008, floor 0.1174±0.0011, **η 23.0%±1.1%**
- Ablation co-located (1 seed @2000): h2h 0.2927±0.0119, jvs 0.4947±0.0026, η 28.4%
- Reference 0% mix @2000 (seed01): h2h 0.3282, j1 0.2532, η 20.2%
- Convergence: completes ~iter 1700; 2000–3000 flat within ±0.005
- Training: 1000 normal + 2000 frozen anneal; team budget 63 tokens (32/31)
