# COMA — Phase 1.5 candidate (counterfactual credit assignment)

> Frozen at commit TBD (filled post-sanity), seed=42 bit-exact.
> Answers EAAI reviewer: "why MAPPO/V-critic instead of COMA/Q-critic?"

**Algorithm**: COMA-S (sample-based Counterfactual Multi-Agent Policy Gradients,
Foerster et al. 2018) + uniform opponent sampling. Same env / reward / curriculum
as MAPPO/IPPO/PfspFix, only `use_coma=true` and `pfsp_p=0`.

Replaces team V critic with centralized Q critic conditioning on the 1222-dim
joint action vector. Per-agent advantage = Q(s,a) − E[Q(s,a_{−i},a_i^')] where
the expectation is approximated by sampling K=8 counterfactual actions per agent.

## Headline results (TBD pending full run)
| Metric | Target | vs MAPPO |
|---|---|---|
| kr (final train) | 0.5 m | same |
| eval_kill_rate | ≥ 0.5 | compare to 0.875 |
| cum_red | ≥ 0.6 | compare to 0.970 |
| aim_residual | < 0.1 m | compare to 0.032 m |

## Quick start
```bash
cd /home/ubuntu/CODE/FluxPhased-
python main.py --config algo/coma/code/config.yaml
```

Outputs land in `algo/coma/data/{checkpoints,logs}/`. Wall-clock: ~3.5–4 h on RTX 4090.

## See also
- [`AGENTS.md`](AGENTS.md) — full agent-facing entry (7 APP sections)
- [`environment/README.md`](environment/README.md) — shared conda env spec
- [`../../algo/_shared/ppo/coma_critic.py`](../../algo/_shared/ppo/coma_critic.py) — Q critic
- [`../../algo/_shared/ppo/coma_advantage.py`](../../algo/_shared/ppo/coma_advantage.py) — counterfactual advantage
