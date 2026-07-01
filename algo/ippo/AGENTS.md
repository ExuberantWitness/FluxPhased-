---
protocol: APP
protocol_version: "1.0"
paper_title: "FluxLeague: AlphaStar-style Self-Play for Phased Array Laser Defense"
authors: ["anonymous"]
domain: rl
algorithm: ippo
commit_pinned: af0d4c20fd2a14eed05f3e8d39f28ad6f43dd1d6
seed: 42
license: TBD
---

# IPPO Baseline (Phase 1.5 — vanilla per-agent critic)

## 1. Identity
- **Algorithm**: IPPO (Independent PPO — per-agent critic, no CTDE, no PFSP)
- **Role**: Phase 1.5 candidate — isolates PFSP effect by holding the critic fixed (per-agent like PfspFix) while disabling opponent prioritization
- **Commit pinned**: `af0d4c2` (bit-exact reproduction)
- **Key difference vs PfspFix**: only `pfsp_p=0` (PFSP off) — same per-agent critic
- **Key difference vs MAPPO**: `use_mappo=false` (per-agent critic, no CTDE)

## 2. Summary
Same env / reward / curriculum as PfspFix stable, with **both** PFSP and CTDE
disabled. Provides the vanilla IPPO baseline for the three-way comparison:
  - MAPPO  vs IPPO   → effect of team critic (CTDE)
  - PfspFix vs IPPO  → effect of PFSP f_hard priority sampling

## 3. Key results (TBD — not yet run)
This baseline has not yet been executed. After running, results will be archived
to `experiments/phase1.5_ippo_seed42/` and surfaced here.

Expected range (from Phase 1.5 design):
| Metric | Expected | Note |
|---|---|---|
| kr (final train) | 0.5 m | same curriculum floor |
| eval_kill_rate | 0.5 – 0.8 | between PfspFix (0.667) and MAPPO (0.875) |
| cum_red | 0.6 – 0.9 | lower than MAPPO if PFSP matters |
| aim_residual | < 0.1 m | same target |

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
- Wall-clock: ~3.4 h for 20 PSRO iterations (estimated, same as PfspFix)
- Conda env: `fluxphased` (see `environment/README.md`)
- Disk: ~290 MB checkpoints + ~13 MB log per run

## 6. Reproduce
```bash
cd /home/ubuntu/CODE/FluxPhased-
python main.py --config algo/ippo/code/config.yaml
# Outputs land in algo/ippo/data/{checkpoints,logs}/
```

Legacy entry (still works):
```bash
bash scripts/run_train.sh algo/ippo/code/config.yaml algo/ippo/data/logs/run.log
```

## 7. Citation
(To be filled when paper is public.)
