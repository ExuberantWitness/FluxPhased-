# phase1.5_mappo_seed42

> **MAPPO baseline** — same env/reward/curriculum as PfspFix, but switches the critic to a **team critic** (CTDE) and **disables PFSP** (uniform opponent sampling). Answers the EAAI reviewer question: *why AlphaStar league instead of MAPPO?*

## TL;DR

- **kr (final train)**: `0.5m`  (curriculum floor: 0.5m)
- **eval kill_rate @ iter 20**: `0.875`
- **cum red / blue / draw**: `0.970 / 0.000 / 0.030`
- **aim residual**: `0.032m`
- **adv_std (last)**: `18.711`  (health: 1e-3 < x < 50)
- **cmd policy_loss (last)**: `-0.01029`  (collapse watch: |x| > 1e-4)

## Gate vs PfspFix baseline

- **Gate status**: PASS ✅ (cum_red ≥ 0.75)
- PfspFix reference: cum_red=0.810, aim_res=0.034m, eval_kr=0.667
- This run:        cum_red=0.970, aim_res=0.032m, eval_kr=0.875
- See `comparison.md` for full PASS/FAIL table and per-iter deltas.

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
bash scripts/run_train.sh configs/laser_25x25_mappo.yaml logs/phase1.5_mappo.log
```

_Commit: `911c5ef236fe`  branch: `main`  dirty: `True`  seed: `42`  wall: `3.817h`_

## Parse / re-plot

```bash
# regenerate metrics.json from train.log
python scripts/experiments/parse_log_to_metrics.py \
    --log train.log \
    --out metrics.json --run-id phase1.5_mappo_seed42

# regenerate figures
python scripts/experiments/make_figures.py \
    --run phase1.5_mappo_seed42:metrics.json --out-dir figures
```
