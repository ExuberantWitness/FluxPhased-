# WP-3 Smoke Cross-Play Report

**RL ckpt**: `checkpoints/blind/wp3_100iter_shaped/iter_final.pt` (iter 100)
**Episodes per direction**: 20
**Horizon**: 200
**Elapsed**: 0.6 min

## Results

| Condition | RL kills | BC kills | Δ | Welch t | p-value | RL survival | RL trace_P |
|---|---|---|---|---|---|---|---|
| low_interference | 0.000±0.000 | 0.825±0.380 | -0.825 | -13.56 | 0.000 | 0.771 | 269.149 |
| high_interference | 0.000±0.000 | 0.125±0.331 | -0.125 | -2.36 | 0.023 | 0.793 | 87.120 |

## Verdict

- **low_interference**: Δ_kills=-0.825 → FAIL
- **high_interference**: Δ_kills=-0.125 → MARGINAL/FAIL
