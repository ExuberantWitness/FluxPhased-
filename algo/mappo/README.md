# MAPPO — Phase 1.5 candidate (CTDE team critic)

> Frozen at commit `5c24f0d`, seed=42 bit-exact.
> Answers EAAI reviewer: "why AlphaStar league instead of MAPPO?"

**Algorithm**: Multi-Agent PPO with team critic (CTDE) + uniform opponent sampling.
Same env / reward / curriculum as PfspFix, only `use_mappo=true` and `pfsp_p=0`.

## Headline results (PASS vs PfspFix on all gates)
| Metric | Value | vs PfspFix |
|---|---|---|
| kr (final train) | 0.5 m | same |
| eval_kill_rate | 0.875 | > 0.667 |
| cum_red | 0.970 | > 0.810 |
| aim_residual | 0.032 m | slightly tighter |

## Quick start
```bash
cd /home/ubuntu/CODE/FluxPhased-
bash algorithms/mappo/code/run.sh
```

Outputs land in `data/{checkpoints,logs}/`. Wall-clock: ~3.8 h on RTX 4090.

## See also
- [`AGENTS.md`](AGENTS.md) — full agent-facing entry (6 APP sections)
- [`../../experiments/phase1.5_mappo_seed42/`](../../experiments/phase1.5_mappo_seed42/) — archived metrics + figures + comparison
- [`environment/README.md`](environment/README.md) — shared conda env spec
