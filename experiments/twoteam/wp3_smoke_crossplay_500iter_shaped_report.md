# WP-3 Smoke Cross-Play Report

**RL ckpt**: `checkpoints/blind/wp3_500iter_shaped/iter_final.pt` (iter 500)
**Episodes per direction**: 20
**Horizon**: 200
**Elapsed**: 0.7 min

## Results

| Condition | RL kills | BC kills | Δ | Welch t | p-value | RL survival | RL trace_P |
|---|---|---|---|---|---|---|---|
| low_interference | 0.050±0.218 | 0.900±0.300 | -0.850 | -14.32 | 0.000 | 0.832 | 316.696 |
| high_interference | 0.075±0.263 | 0.225±0.418 | -0.150 | -1.90 | 0.062 | 0.951 | 200.413 |

## Verdict

- **low_interference**: Δ_kills=-0.850 → FAIL
- **high_interference**: Δ_kills=-0.150 → MARGINAL/FAIL
