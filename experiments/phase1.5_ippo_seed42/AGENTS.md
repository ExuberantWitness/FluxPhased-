# phase1.5_ippo_seed42

> **IPPO baseline** — same env/reward/curriculum as PfspFix and MAPPO, but with **per-agent critic (no CTDE)** AND **PFSP off (uniform opponent sampling)**. This is the third arm of the three-way comparison and isolates the joint effect of dropping both AlphaStar league mechanisms.

## TL;DR

- **kr (final train)**: `0.5m`  (curriculum floor: 0.5m)
- **eval kill_rate @ iter 20**: `0.792`
- **cum red / blue / draw**: `0.810 / 0.180 / 0.010`
- **aim residual**: `0.035m`
- **adv_std (last)**: `14.911`  (health: 1e-3 < x < 50)
- **cmd policy_loss (last)**: `-0.00966`  (collapse watch: |x| > 1e-4)

## Gate vs PfspFix baseline

- **Gate status**: PASS ✅ (cum_red ≥ 0.75, eval_kr ≥ 0.5, aim_res ≤ 0.1)
- PfspFix reference: cum_red=0.810, aim_res=0.034m, eval_kr=0.667
- This run:        cum_red=0.810, aim_res=0.035m, eval_kr=0.792
- See `comparison.md` for full PASS/FAIL table and per-iter deltas.

## Three-way comparison (iter 20)

| Metric | PfspFix (PFSP only) | MAPPO (CTDE only) | IPPO (neither) | CTDE Δ (M−I) | PFSP Δ (P−I) |
|---|---|---|---|---|---|
| cum red win share       | 0.810 | 0.970 | 0.810 | +0.160 | +0.000 |
| eval kill_rate @ iter 20 | 0.667 | 0.875 | 0.792 | +0.083 | −0.125 |
| aim residual (m)         | 0.034 | 0.032 | 0.035 | −0.003 | −0.001 |
| adv_std (last)           | 9.489 | 18.71 | 14.91 | +3.80  | −5.42  |
| \|cmd policy_loss\| (last) | 0.00436 | 0.01029 | 0.00966 | +0.00067 | −0.00530 |

**Reading**:
- **CTDE effect (MAPPO vs IPPO)**: large cum_red gain (+0.16), moderate eval_kr gain (+0.08), small aim_res gain. CTDE is the dominant lever for self-play red win share.
- **PFSP effect (PfspFix vs IPPO)**: nearly identical cum_red (0.81 vs 0.81) — PFSP alone does not move self-play red share. PfspFix actually shows *lower* deterministic eval_kr (0.667 vs 0.792), consistent with PFSP's harder-priority opponents creating a stronger training signal without inflating red/red win share (since PFSP balances opponent strength rather than just red-favored).

## Files in this directory

| File | What |
|---|---|
| `config.yaml` | Frozen config snapshot (survives edits to live yaml) |
| `metadata.json` | seed, git commit, host, reproduce_cmd, wall-clock |
| `metrics.json` | Per-iter structured metrics (machine-readable) |
| `train.log` | Raw stdout from training |
| `figures/*.png` | Curve plots (kr, cum_red, aim_res, adv_std, cmd_pl, eval_kr) |
| `reproduce.sh` | One-command re-run |
| `AGENTS.md` | This file |
| `comparison.md` | Diff vs PfspFix baseline |

## Reproduce

```bash
python main.py --config algo/ippo/code/config.yaml
```

(Older path: `bash scripts/run_train.sh configs/laser_25x25_ippo.yaml logs/phase1.5_ippo.log` — still works as legacy.)

_Commit: `af0d4c20fd2a`  branch: `phase1.5/three-way-baselines`  dirty: `True`  seed: `42`  wall: `3.938h`_

## Parse / re-plot

```bash
# regenerate metrics.json from train.log
python scripts/experiments/parse_log_to_metrics.py \
    --log train.log \
    --out metrics.json --run-id phase1.5_ippo_seed42

# regenerate figures
python scripts/experiments/make_figures.py \
    --run phase1.5_ippo_seed42:metrics.json --out-dir figures
```
