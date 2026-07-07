# EW-MAPPO — Phase 1.5 Gate 0

> Frozen at commit `a244f4c`, seed=42 bit-exact.
> Phase 1.5 Gate 0: **EW 任务下 CTDE team critic 能否反超 classical MPC**。

**算法**: MAPPO + EW 任务 + uniform opponent sampling。
**vs mappo**: 注入 8 个 EW 字段(jam_gain=8, exposure_gain=50, race_* 等) +
`commander_privileged_dim=10` + `env.kill_radius_m: 0.5→0.2`。

## 快速复现
```bash
cd /home/ubuntu/CODE/FluxPhased-
python main.py --config algo/ew_mappo/code/config.yaml
```
输出: `algo/ew_mappo/data/{checkpoints,logs}/`。Wall-clock: ~4 h on RTX 4090。

## Cross-play vs ClassicalMPC
```bash
python scripts/crossplay_mpc.py --config algo/ew_mappo/code/config.yaml \
    --arms ew_mappo --out experiments/crossplay_mpc_ew_mappo.md
```
**Gate 0 判据**: ew_mappo NN WR > 0.5 vs ClassicalMPC(在 EW 任务下)。

## 关联
- [`AGENTS.md`](AGENTS.md) — 完整 7-段 APP entry
- [`../../EAAI_C2_EW_BELIEF_PLAN.md`](../../EAAI_C2_EW_BELIEF_PLAN.md) — Gate 0/1 方案
- [`environment/README.md`](environment/README.md) — 共享 conda env
