# WP-3 Smoke Cross-Play Report

**RL ckpt**: `checkpoints/blind/wp3_100iter_dynamics/iter_final.pt` (iter 100)
**Episodes per direction**: 20
**Horizon**: 200
**Elapsed**: 0.6 min

## Results

| Condition | RL kills | BC kills | Δ | Welch t | p-value | RL survival | RL trace_P |
|---|---|---|---|---|---|---|---|
| low_interference | 0.075±0.263 | 1.000±0.000 | -0.925 | -21.93 | 0.000 | 0.765 | 292.091 |
| high_interference | 0.075±0.263 | 0.225±0.418 | -0.150 | -1.90 | 0.062 | 0.811 | 309.593 |

## Verdict

- **low_interference**: Δ_kills=-0.925 → FAIL
- **high_interference**: Δ_kills=-0.150 → MARGINAL/FAIL
