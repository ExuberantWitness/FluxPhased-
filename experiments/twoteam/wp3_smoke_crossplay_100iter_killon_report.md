# WP-3 Smoke Cross-Play Report

**RL ckpt**: `checkpoints/blind/wp3_100iter_killon_20260718_002800/iter_final.pt` (iter 100)
**Episodes per direction**: 20
**Horizon**: 200
**Elapsed**: 0.7 min

## Results

| Condition | RL kills | BC kills | Δ | Welch t | p-value | RL survival | RL trace_P |
|---|---|---|---|---|---|---|---|
| low_interference | 0.025±0.156 | 1.000±0.000 | -0.975 | -39.00 | 0.000 | 0.850 | 357.852 |
| high_interference | 0.025±0.156 | 0.200±0.400 | -0.175 | -2.55 | 0.014 | 0.956 | 264.466 |

## Verdict

- **low_interference**: Δ_kills=-0.975 → FAIL
- **high_interference**: Δ_kills=-0.175 → MARGINAL/FAIL
