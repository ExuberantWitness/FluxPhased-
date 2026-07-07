# EW-IPPO — Phase 1.5 Gate 0 (无 CTDE 对照)

> 目标:Gate 0 的无-CTDE 对照臂 — 验证 EW 任务下 CTDE 是否带来额外增益。
> Frozen at commit `a244f4c`, seed=42 bit-exact.

**算法**: IPPO (per-agent critic, 无 CTDE) + **EW 任务环境**
(jam_gain=8 / exposure_gain=50) + uniform opponent sampling。
**相对 ippo 的差异**: 注入 EW 字段(jam_gain/jam_cost/exposure_gain/race_*) +
`commander_privileged_dim=10` (在此臂中 critic 不会用上,保留同 config 格式) +
`env.kill_radius_m: 0.5→0.2` 让 curriculum 经过 0.5 触发 jam_on。

## 1. Identity
- **Algorithm**: EW-IPPO (no-CTDE under EW)
- **Role**: Phase 1.5 Gate 0 — control arm for "CTDE effect under EW"
- **Commit pinned**: `a244f4c` (bit-exact reproduction)
- **Key difference vs ippo**: +8 EW config fields, kr floor 0.5→0.2

## 2. Summary
Same Phase 1.5 ippo stable recipe, with the full EW segment from
`configs/laser_25x25_ew_exposure.yaml` injected. Counterpart to `ew_mappo`:
isolates whether CTDE team critic helps under EW (vs the no-CTDE IPPO).

## 3. Key results (Gate 0 metrics, populates after run)
| Metric | Value | Health band / Gate |
|---|---|---|
| kr (final train) | TBD | curriculum floor 0.2 |
| eval_kill_rate | TBD | — |
| cum_red | TBD | — |
| aim_residual | TBD | < 0.5m EW-degraded |
| adv_std (last) | TBD | 1e-3 < x < 50 |
| cmd_policy_loss (last) | TBD | \|x\| > 1e-4 |

**Gate 0 (vs ClassicalMPC under same EW env)**: PASS = ew_ippo WR > 0.5.
**CTDE contribution**: Δ(ew_mappo WR − ew_ippo WR) shows the value of CTDE under EW.

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
python main.py --config algo/ew_ippo/code/config.yaml
# Outputs land in algo/ew_ippo/data/{checkpoints,logs}/
# Cross-play vs ClassicalMPC: scripts/crossplay_mpc.py --arms ew_ippo
```

## 7. Citation
(To be filled when paper is public.)
