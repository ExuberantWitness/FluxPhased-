# PfspFix — Phase 1.5 reference baseline

> Verified recipe (commit `911c5ef`), seed=42 bit-exact.
> Reference every Phase 1.5+ run compares against.

**Algorithm**: AlphaStar PFSP f_hard priority sampling + EMA win-rate + per-agent critic.

## Headline results
| Metric | Value |
|---|---|
| kr (final train) | 0.5 m |
| eval_kill_rate | 0.667 |
| cum_red | 0.810 |
| aim_residual | 0.034 m |

## Quick start
```bash
cd /home/ubuntu/CODE/FluxPhased-
bash algorithms/pspfix/code/run.sh
```

Outputs land in `data/{checkpoints,logs}/`. Wall-clock: ~3.4 h on RTX 4090.

## See also
- [`AGENTS.md`](AGENTS.md) — full agent-facing entry (6 APP sections)
- [`../../experiments/phase1_pfsp_seed42/`](../../experiments/phase1_pfsp_seed42/) — archived metrics + figures + comparison
- [`environment/README.md`](environment/README.md) — shared conda env spec
