# WP-3 Smoke Cross-Play Report

**RL ckpt**: `checkpoints/blind/wp3_20260718_090802/iter_final.pt` (iter 100)
**Episodes per direction**: 50
**Horizon**: 200
**Elapsed**: 1.8 min

## Results

| Condition | RL kills | BC kills | Δ | Welch t | p-value | RL survival | RL trace_P |
|---|---|---|---|---|---|---|---|
| low_interference | 0.031±0.174 | 0.938±0.242 | -0.906 | -34.26 | 0.000 | 0.815 | 284.952 |
| high_interference | 0.039±0.194 | 0.211±0.408 | -0.172 | -4.29 | 0.000 | 0.939 | 326.627 |

## Verdict

- **low_interference**: Δ_kills=-0.906 → FAIL
- **high_interference**: Δ_kills=-0.172 → MARGINAL/FAIL
