# WP-3 Smoke Cross-Play Report

**RL ckpt**: `checkpoints/blind/wp3_100iter_shaped2/iter_final.pt` (iter 100)
**Episodes per direction**: 20
**Horizon**: 200
**Elapsed**: 0.7 min

## Results

| Condition | RL kills | BC kills | Δ | Welch t | p-value | RL survival | RL trace_P |
|---|---|---|---|---|---|---|---|
| low_interference | 0.025±0.156 | 0.900±0.300 | -0.875 | -16.16 | 0.000 | 0.750 | 218.089 |
| high_interference | 0.050±0.218 | 0.200±0.400 | -0.150 | -2.06 | 0.044 | 0.885 | 87.470 |

## Verdict

- **low_interference**: Δ_kills=-0.875 → FAIL
- **high_interference**: Δ_kills=-0.150 → MARGINAL/FAIL
