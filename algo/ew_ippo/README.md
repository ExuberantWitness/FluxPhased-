# EW-IPPO — Phase 1.5 Gate 0 (无 CTDE 对照)

> Frozen at commit `a244f4c`, seed=42 bit-exact.
> EW 任务下 IPPO(per-agent critic)对照 — 分离 CTDE 的贡献。

**算法**: IPPO + EW 任务 + uniform opponent sampling。
**vs ippo**: 注入 8 个 EW 字段 + `commander_privileged_dim=10` +
`env.kill_radius_m: 0.5→0.2`。

## 快速复现
```bash
cd /home/ubuntu/CODE/FluxPhased-
python main.py --config algo/ew_ippo/code/config.yaml
```
输出: `algo/ew_ippo/data/{checkpoints,logs}/`。Wall-clock: ~4 h on RTX 4090。

## Cross-play vs ClassicalMPC
```bash
python scripts/crossplay_mpc.py --config algo/ew_ippo/code/config.yaml \
    --arms ew_ippo --out experiments/crossplay_mpc_ew_ippo.md
```

## 关联
- [`AGENTS.md`](AGENTS.md) — 完整 7-段 APP entry
- [`../../EAAI_C2_EW_BELIEF_PLAN.md`](../../EAAI_C2_EW_BELIEF_PLAN.md) — Gate 0/1 方案
- [`./ew_mappo/`](../ew_mappo/) — CTDE 配对臂
- [`environment/README.md`](environment/README.md) — 共享 conda env
