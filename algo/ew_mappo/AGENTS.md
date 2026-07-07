# EW-MAPPO — Phase 1.5 Gate 0 (CTDE under EW)

> 目标:验证 EW 任务下 classical MPC 是否 < 0.5(EW 让 classical 失效),
> EW-MAPPO 是否反超 classical。决定 EAAI C2 (belief-CTDE) 路线是否成立。
> Frozen at commit `a244f4c`, seed=42 bit-exact.

**算法**: MAPPO (team critic, CTDE) + **EW 任务环境** (jam_gain=8 / exposure_gain=50)
+ uniform opponent sampling。
**相对 mappo 的差异**: 注入 EW 字段(jam_gain/jam_cost/exposure_gain/race_*) +
`commander_privileged_dim=10` 让 CTDE critic 看到 EW 状态 +
`env.kill_radius_m: 0.5→0.2` 让 curriculum 经过 0.5 触发 jam_on。

## 1. Identity
- **Algorithm**: EW-MAPPO (CTDE under EW)
- **Role**: Phase 1.5 Gate 0 — falsification test for "EW makes classical fail, RL wins"
- **Commit pinned**: `a244f4c` (bit-exact reproduction)
- **Key difference vs mappo**: +8 EW config fields, kr floor 0.5→0.2

## 2. Summary
Same Phase 1.5 mappo stable recipe, with the full EW segment from
`configs/laser_25x25_ew_exposure.yaml` injected. Purpose: create the EW
environment where the Kalman anchor is no longer trustworthy (jamming adds
5× noise to range/crossrange), so classical "point at anchor + always fire"
fails. Then test if MAPPO with EW-aware critic can learn the strategic
counter (timed intermediate jamming, suppress-fire under uncertainty).

## 3. Key results (Gate 0 metrics, populates after run)
| Metric | Value | Health band / Gate |
|---|---|---|
| kr (final train) | TBD | curriculum floor 0.2 |
| eval_kill_rate | TBD | > classical-mpc WR |
| cum_red | TBD | > 0.5 (vs classical) |
| aim_residual | TBD | < 0.5m EW-degraded |
| adv_std (last) | TBD | 1e-3 < x < 50 |
| cmd_policy_loss (last) | TBD | \|x\| > 1e-4 |

**Gate 0 (vs ClassicalMPC under same EW env)**: PASS = ew_mappo WR > 0.5.

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
- Wall-clock: ~4 h for 20 PSRO iterations (ew_mappo + kr to 0.2 takes slightly longer)
- Conda env: `fluxphased` (see `environment/README.md`)
- Disk: ~290 MB checkpoints + ~13 MB log per run

## 6. Reproduce
```bash
cd /home/ubuntu/CODE/FluxPhased-
python main.py --config algo/ew_mappo/code/config.yaml
# Outputs land in algo/ew_mappo/data/{checkpoints,logs}/
# Cross-play vs ClassicalMPC: scripts/crossplay_mpc.py --arms ew_mappo
```

## 7. Citation
(To be filled when paper is public.)
