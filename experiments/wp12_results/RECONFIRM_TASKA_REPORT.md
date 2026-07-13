# RECONFIRM Task A — learned vs classical kill+survival (α_eff FIXED)

**Methods**: ['strong_classical', 'mappo', 'ippo']
**Cells**: ['n4_L0', 'n4_L1-tau1', 'n4_L3-trained', 'n8_L0']
**Seeds**: [42, 43, 44, 45, 46]
**L3 jammer**: `/home/ubuntu/CODE/FluxPhased-/checkpoints/appint/jammer_L3_final.pt`

## Per-cell × method (mean over 5 seeds ± 95% CI)

| Cell | Method | kill (mean ± CI) | survival (mean ± CI) |
|------|--------|------------------|----------------------|
| n4_L0 | ippo | 4.00 ± 0.00 | 0.90 ± 0.20 |
| n4_L0 | mappo | 4.00 ± 0.00 | 0.95 ± 0.14 |
| n4_L0 | strong_classical | 3.98 ± 0.07 | 0.85 ± 0.13 |
| n4_L1-tau1 | ippo | 3.85 ± 0.28 | 0.78 ± 0.34 |
| n4_L1-tau1 | mappo | 3.67 ± 0.28 | 0.62 ± 0.25 |
| n4_L1-tau1 | strong_classical | 3.62 ± 0.11 | 0.40 ± 0.20 |
| n4_L3-trained | ippo | 3.90 ± 0.13 | 0.78 ± 0.20 |
| n4_L3-trained | mappo | 3.58 ± 0.14 | 0.65 ± 0.17 |
| n4_L3-trained | strong_classical | 3.58 ± 0.08 | 0.40 ± 0.20 |
| n8_L0 | ippo | 4.65 ± 0.30 | 0.62 ± 0.16 |
| n8_L0 | mappo | 5.45 ± 0.53 | 0.53 ± 0.07 |
| n8_L0 | strong_classical | 8.00 ± 0.00 | 0.80 ± 0.14 |

## n4_L1-τ1 headline cell (gate decision)

| Method | kill mean | survival mean | survival CI |
|--------|-----------|---------------|-------------|
| mappo | 3.67 | 0.62 | ±0.25 |
| ippo | 3.85 | 0.78 | ±0.34 |
| strong_classical | 3.62 | 0.40 | ±0.20 |

## Gate verdict (per learned method)

### mappo
- kill_wins (learned kill > classical + 0.05): ❌ (3.67 vs 3.62)
- survival-Pareto (tie kill ±0.20 AND surv > classical+CI): ❌ (surv 0.62 vs 0.40)
### ippo
- kill_wins (learned kill > classical + 0.05): ✅ (3.85 vs 3.62)
- survival-Pareto (tie kill ±0.20 AND surv > classical+CI): ❌ (surv 0.78 vs 0.40)

## Overall

**Task A gate: PASS**

At least one learned method wins n4_L1-τ1 on kill OR survival-Pareto.
**→ APPROVE full R2 grid (1120 eps).**
