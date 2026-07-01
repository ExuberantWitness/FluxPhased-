---
protocol: APP
protocol_version: "1.0"
paper_title: "FluxLeague: AlphaStar-style Self-Play for Phased Array Laser Defense"
authors: ["anonymous"]
domain: rl
algorithm: pspfix
commit_pinned: 911c5ef236fed7db894a79041e3e6609ae74172a
seed: 42
license: TBD
---

# PfspFix Baseline (Phase 1.5 reference)

## 1. Identity
- **Algorithm**: PfspFix (AlphaStar PFSP f_hard priority sampling + EMA win-rate + per-agent critic)
- **Role**: Phase 1.5 reference baseline — every PfspFix/MAPPO/IPPO run compares against this
- **Commit pinned**: `911c5ef` (bit-exact reproduction)
- **Key novelty**: Prioritized Fake Self-Play (PFSP) — agents sample opponents by `w_i = (pool_winrate[i] + ε)^p`, concentrating training on the most informative matchups

## 2. Summary
Verified recipe locked at commit `911c5ef`, seed=42 bit-exact. Activates PFSP f_hard
opponent sampling (default `pfsp_p = 0` here, but the league framework supports it)
and EMA win-rate estimation, with per-agent (non-CTDE) critic.

## 3. Key results (from `experiments/phase1_pfsp_seed42/`)
| Metric | Value | Health band |
|---|---|---|
| kr (final train) | 0.5 m | curriculum floor (started 50 m) |
| eval_kill_rate | 0.667 | — |
| cum_red / blue / draw | 0.810 / 0.062 / 0.128 | — |
| aim_residual | 0.034 m | < 0.1 m target |
| adv_std (last) | (see metrics.json) | 1e-3 < x < 50 (healthy GAE) |
| cmd_policy_loss (last) | (see metrics.json) | |x| > 1e-4 (non-collapse) |

## 4. Repo structure (APP layout)
- `code/config.yaml` — frozen config (absolute `checkpoint_dir`)
- `code/.git_commit` — pinned SHA (WARN-only tripwire)
- `data/checkpoints/` — algorithm outputs (iter_*.pt, populated at runtime)
- `data/logs/` — training log (populated at runtime)
- `environment/README.md` — shared conda env spec
- `LICENSE` — symlink to repo root LICENSE
- `APP_PUBLICATION.json` — manifest

Entry point: repo-root `main.py` (pure-Python launcher; activates `fluxphased`
conda env, enables faulthandler, writes PID, then forwards to
`algo._shared.train_laser:main`).

## 5. Computational requirements
- GPU: 1x RTX 4090 (24 GB) — verified; also runs on 98 GB
- Wall-clock: ~3.4 h for 20 PSRO iterations
- Conda env: `fluxphased` (see `environment/README.md`)
- Disk: ~290 MB checkpoints + ~13 MB log per run

## 6. Reproduce
```bash
cd /home/ubuntu/CODE/FluxPhased-
python main.py --config algo/pspfix/code/config.yaml
# Outputs land in algo/pspfix/data/{checkpoints,logs}/
```

Legacy entry (still works):
```bash
bash scripts/run_train.sh algo/pspfix/code/config.yaml algo/pspfix/data/logs/run.log
```

For the archived result (frozen commit + experiment artifacts):
see `experiments/phase1_pfsp_seed42/`.

## 7. Citation
(To be filled when paper is public.)
