---
protocol: APP
protocol_version: "1.0"
paper_title: "FluxLeague: AlphaStar-style Self-Play for Phased Array Laser Defense"
authors: ["anonymous"]
domain: rl
algorithm: full_league
commit_pinned: "6375030"
seed: 42
license: TBD
---

# FullLeague (PFSP + CTDE on) — Phase 1.5 2×2 ablation 4th cell

## 1. Identity
- **Algorithm**: PPO with team critic (CTDE) + AlphaStar-style PFSP opponent sampling
- **Role**: 4th cell of Phase 1.5 2×2 ablation (CTDE × PFSP)
- **Commit pinned**: `6375030` (bit-exact reproduction)
- **Key difference vs MAPPO**: `pfsp_p=2` (was 0) — turns PFSP f_hard opponent sampling ON
- **Key difference vs PfspFix**: `use_mappo=true` (was false) — turns CTDE team critic ON

## 2. Summary
Same env / reward / curriculum as MAPMO/IPPO/PfspFix, but **both** CTDE and PFSP on.
Closes the 2×2 ablation matrix:
- IPPO     = CTDE off + PFSP off
- MAPPO    = CTDE on  + PFSP off
- PfspFix  = CTDE off + PFSP on
- **FullLeague = CTDE on + PFSP on** (this run)

Cross-play (Exp B) showed MAPPO wins head-to-head vs PfspFix (0.764). This run
tests whether combining both gives a strict improvement over MAPPO alone.

## 3. Key results (TBD — populated after iter 20)
| Metric | Value | Health band |
|---|---|---|
| kr (final train) | TBD | curriculum floor |
| eval_kill_rate | TBD | > MAPPO 0.875 to PASS |
| cum_red | TBD | > MAPPO 0.97 to PASS |
| aim_residual | TBD | < 0.1 m target |
| adv_std (last) | TBD | 1e-3 < x < 50 |
| cmd_policy_loss (last) | TBD | |x| > 1e-4 |

**Gate vs MAPPO**: PASS = cum_red ≥ 0.97 AND eval_kill_rate ≥ 0.875.

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
- GPU: 1x RTX PRO 6000 (101.9 GB) — verified
- Wall-clock: ~4 h for 20 PSRO iterations
- Conda env: `fluxphased` (see `environment/README.md`)
- Disk: ~290 MB checkpoints + ~13 MB log per run

## 6. Reproduce
```bash
cd /home/ubuntu/CODE/FluxPhased-
python main.py --config algo/full_league/code/config.yaml
# Outputs land in algorithms/full_league/data/{checkpoints,logs}/
# (config.checkpoint_dir is absolute path to algorithms/, not algo/)
```

## 7. Citation
(To be filled when paper is public.)
