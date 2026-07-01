# phase1_pfsp_seed42

> **PfspFix reference baseline** — verified recipe (commit `911c5ef`), seed=42 bit-exact. This is the reference every Phase 1.5+ run compares against.

## TL;DR

- **kr (final train)**: `0.5m`  (curriculum floor: 0.5m)
- **eval kill_rate @ iter 20**: `0.667`
- **cum red / blue / draw**: `0.810 / 0.190 / 0.000`
- **aim residual**: `0.034m`
- **adv_std (last)**: `9.489`  (health: 1e-3 < x < 50)
- **cmd policy_loss (last)**: `-0.00436`  (collapse watch: |x| > 1e-4)

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
| `comparison.md` | (Candidate runs only) Diff vs PfspFix baseline |

## Reproduce

```bash
bash scripts/run_train.sh configs/laser_25x25_pro6000_stable.yaml logs/phase1_seed42_run1.log
```

_Commit: `911c5ef236fe`  branch: `main`  dirty: `True`  seed: `42`  wall: `3.367h`_

## Parse / re-plot

```bash
# regenerate metrics.json from train.log
python scripts/experiments/parse_log_to_metrics.py \
    --log train.log \
    --out metrics.json --run-id phase1_pfsp_seed42

# regenerate figures
python scripts/experiments/make_figures.py \
    --run phase1_pfsp_seed42:metrics.json --out-dir figures
```
