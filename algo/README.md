# Algorithms — Phase 1.5 three-way comparison

Three APP-compliant algorithm folders for the Phase 1.5 baseline comparison.
Each folder is an independent, reproducible research artifact following the
[Agentic Publication Protocol (APP)](https://arxiv.org/abs/2606.27386).

## Comparison

| Algorithm | Critic | Opponent sampling | Commit | Wall-clock | eval_kill_rate | cum_red | aim_res |
|---|---|---|---|---|---|---|---|
| [pspfix](pspfix/AGENTS.md) | per-agent | PFSP f_hard (default p=0) | `911c5ef` | ~3.4h | 0.667 | 0.810 | 0.034 m |
| [mappo](mappo/AGENTS.md) | team (CTDE) | uniform (pfsp_p=0) | `5c24f0d` | ~3.8h | 0.875 | 0.970 | 0.032 m |
| [ippo](ippo/AGENTS.md) | per-agent | uniform (pfsp_p=0) | `af0d4c2` | ~3.4h | (TBD) | (TBD) | (TBD) |

## What each comparison isolates

- **MAPPO vs IPPO** → effect of team critic (CTDE)
- **PfspFix vs IPPO** → effect of PFSP f_hard priority sampling
- **PfspFix vs MAPPO** → combined effect of CTDE + uniform sampling

## Shared environment

All three algorithms share the same conda env (`fluxphased`) and Warp kernel
cache (`$HOME/.cache/warp/`). See each algorithm's `environment/README.md` for
details. Only config / seeds / pinned commits differ.

## Repository layout (paper-ready)

```
FluxPhased-/
├── main.py                # Pure-Python entry (conda subprocess + faulthandler + PID)
├── env/                   # Phased-array EW simulation (renamed from radar_sim/)
├── algo/
│   ├── _shared/           # Shared training code (moved from training/)
│   │   ├── train_laser.py # ← main.py forwards here
│   │   ├── ppo/
│   │   ├── self_play/
│   │   ├── laser/
│   │   ├── flux_league.py
│   │   └── ...
│   ├── pspfix/            # APP-compliant (config + .git_commit, no run.sh)
│   ├── mappo/
│   └── ippo/
├── scripts/run_train.sh   # Legacy launcher (still works; uses algo._shared.train_laser)
└── configs/               # Legacy configs (frozen)
```

## Run in any order (fully isolated)

```bash
cd /home/ubuntu/CODE/FluxPhased-

# Each writes to its own algo/<algo>/data/checkpoints/ (absolute path)
python main.py --config algo/pspfix/code/config.yaml
python main.py --config algo/mappo/code/config.yaml
python main.py --config algo/ippo/code/config.yaml
```

`main.py` auto-activates `fluxphased` (via `conda activate` subprocess) if not
already in it, enables `faulthandler`, writes `/tmp/train_laser.pid`, then
forwards to `algo._shared.train_laser:main`.

Outputs land in each algorithm's `data/{checkpoints,logs}/`. They do not
overwrite each other.

## APP compliance (6 components per folder)

| Component | pspfix | mappo | ippo |
|---|---|---|---|
| `AGENTS.md` (frontmatter) | ✅ | ✅ | ✅ |
| `README.md` (human) | ✅ | ✅ | ✅ |
| `LICENSE` (symlink) | ✅ | ✅ | ✅ |
| `code/` (config + commit pin) | ✅ | ✅ | ✅ |
| `data/` (checkpoints + logs) | ✅ | ✅ | ✅ |
| `environment/` (env spec) | ✅ | ✅ | ✅ |

## See also

- [`../AGENTS.md`](../AGENTS.md) — repo-level APP entry
- [`../APP_PUBLICATION.json`](../APP_PUBLICATION.json) — repo-level manifest
- [`../experiments/`](../experiments/) — frozen-result archives per algorithm
- [`../PHASE1_5_MAPPO_REPORT.md`](../PHASE1_5_MAPPO_REPORT.md) — full Phase 1.5 experiment report
