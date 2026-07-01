# Experiments index

> Headline numbers across all runs in this directory.

> 📄 **Full Phase 1.5 experiment report**: [`../PHASE1_5_MAPPO_REPORT.md`](../PHASE1_5_MAPPO_REPORT.md) — setup, results, reproduction, discussion.

> Add a new run by copying `scripts/experiments/_template/` (TODO) or by
> following the recipe in `scripts/experiments/README.md`.

## Run comparison

| Run | seed | iters | kr (m) | eval_kr | cum_red | aim_res (m) | verdict |
|---|---|---|---|---|---|---|---|
| [phase1.5_mappo_seed42](phase1.5_mappo_seed42/AGENTS.md) | 42 | 20 | 0.5 | 0.875 | 0.970 | 0.032 | PASS ✅ |
| [phase1_pfsp_seed42](phase1_pfsp_seed42/AGENTS.md) | 42 | 20 | 0.5 | 0.667 | 0.810 | 0.034 | reference |

## Tooling

- `scripts/experiments/parse_log_to_metrics.py` — log → metrics.json
- `scripts/experiments/make_figures.py` — metrics.json → PNG curves
- `scripts/experiments/compare_runs.py` — two metrics.json → comparison.md
- `scripts/experiments/write_metadata.py` — seed/commit/reproduce → metadata.json
- `scripts/experiments/write_agents_md.py` — this file (regenerates AGENTS.md)

## Reproducing the whole comparison

```bash
# 1. run PfspFix (already done, ~3h)
bash scripts/run_train.sh configs/laser_25x25_pro6000_stable.yaml logs/phase1_seed42_run1.log

# 2. run MAPPO (~3h)
bash scripts/run_train.sh configs/laser_25x25_mappo.yaml logs/phase1.5_mappo.log

# 3. regenerate every artifact
for d in experiments/phase1_*/ experiments/phase1.5_*/ ; do
  python scripts/experiments/parse_log_to_metrics.py \
      --log $d/train.log --out $d/metrics.json --run-id $(basename $d)
  python scripts/experiments/make_figures.py \
      --run $(basename $d):$d/metrics.json --out-dir $d/figures
  python scripts/experiments/write_agents_md.py --run-dir $d
done
```
