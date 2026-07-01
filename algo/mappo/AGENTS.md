---
protocol: APP
protocol_version: "1.0"
paper_title: "FluxLeague: AlphaStar-style Self-Play for Phased Array Laser Defense"
authors: ["anonymous"]
domain: rl
algorithm: mappo
commit_pinned: 5c24f0d497563e0914edf337238ddb09c8a7e566
seed: 42
license: TBD
---

# MAPPO Baseline (Phase 1.5 — CTDE team critic)

## 1. Identity
- **Algorithm**: MAPPO (Multi-Agent PPO with team critic, CTDE)
- **Role**: Phase 1.5 candidate — answers EAAI reviewer question "why AlphaStar league instead of MAPPO?"
- **Commit pinned**: `5c24f0d` (bit-exact reproduction)
- **Key difference vs PfspFix**: `use_mappo=true` → 104-dim team state critic (CTDE); `pfsp_p=0` → uniform opponent sampling

## 2. Summary
Same env / reward / curriculum as PfspFix stable, but switches the critic to a
**team critic** (centralized training, decentralized execution) and **disables
PFSP** (uniform opponent sampling). Isolates the effect of CTDE team critic.

## 3. Key results (from `experiments/phase1.5_mappo_seed42/`)
| Metric | Value | Health band |
|---|---|---|
| kr (final train) | 0.5 m | curriculum floor |
| eval_kill_rate | 0.875 | > PfspFix 0.667 |
| cum_red / blue / draw | 0.970 / 0.005 / 0.025 | — |
| aim_residual | 0.032 m | < 0.1 m target |
| adv_std (last) | (see metrics.json) | 1e-3 < x < 50 |
| cmd_policy_loss (last) | (see metrics.json) | |x| > 1e-4 |

**Gate vs PfspFix**: PASS (cum_red=0.97 ≥ 0.75, eval_kill_rate improved).

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
- GPU: 1x RTX 4090 (24 GB) — verified
- Wall-clock: ~3.8 h for 20 PSRO iterations
- Conda env: `fluxphased` (see `environment/README.md`)
- Disk: ~290 MB checkpoints + ~13 MB log per run

## 6. Reproduce
```bash
cd /home/ubuntu/CODE/FluxPhased-
python main.py --config algo/mappo/code/config.yaml
# Outputs land in algo/mappo/data/{checkpoints,logs}/
```

Legacy entry (still works):
```bash
bash scripts/run_train.sh algo/mappo/code/config.yaml algo/mappo/data/logs/run.log
```

For the archived result (frozen commit + experiment artifacts):
see `experiments/phase1.5_mappo_seed42/`.

## 7. Citation
(To be filled when paper is public.)
