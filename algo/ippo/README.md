# IPPO — Phase 1.5 candidate (vanilla per-agent critic)

> Frozen at commit `af0d4c2`, seed=42 bit-exact.
> Vanilla IPPO baseline for three-way comparison.

**Algorithm**: Independent PPO (per-agent critic, no CTDE) + uniform opponent
sampling. Both PFSP and team critic disabled.

**Three-way comparison roles**:
- MAPPO vs IPPO → effect of team critic (CTDE)
- PfspFix vs IPPO → effect of PFSP f_hard priority sampling

## Headline results (TBD — not yet run)
After running, results archive to `experiments/phase1.5_ippo_seed42/`.

| Metric | Expected |
|---|---|
| kr (final train) | 0.5 m |
| eval_kill_rate | 0.5 – 0.8 |
| cum_red | 0.6 – 0.9 |

## Quick start
```bash
cd /home/ubuntu/CODE/FluxPhased-
bash algorithms/ippo/code/run.sh
```

Outputs land in `data/{checkpoints,logs}/`. Wall-clock: ~3.4 h on RTX 4090.

## See also
- [`AGENTS.md`](AGENTS.md) — full agent-facing entry (6 APP sections)
- [`environment/README.md`](environment/README.md) — shared conda env spec
