---
protocol: APP
protocol_version: "1.0"
paper_title: "FluxLeague: AlphaStar-style Self-Play for Phased Array Laser Defense"
authors: ["anonymous"]
domain: rl
repo: FluxPhased-
license: TBD
---

# FluxPhased- — Phase 1.5 Three-Way Baseline Comparison

This repository hosts three APP-compliant algorithm folders for the Phase 1.5
three-way comparison (PfspFix vs MAPPO vs IPPO). Each folder is an independent,
reproducible research artifact following the
[Agentic Publication Protocol (APP)](https://arxiv.org/abs/2606.27386).

## Algorithm folders (each is APP-compliant)

| Algorithm | Role | Commit pinned | Status |
|---|---|---|---|
| [`algorithms/pspfix/`](algorithms/pspfix/AGENTS.md) | Phase 1.5 reference | `911c5ef` | ✅ reference |
| [`algorithms/mappo/`](algorithms/mappo/AGENTS.md) | Phase 1.5 candidate (CTDE) | `5c24f0d` | ✅ PASS |
| [`algorithms/ippo/`](algorithms/ippo/AGENTS.md) | Phase 1.5 candidate (vanilla) | `af0d4c2` | ⏳ pending |

## Shared infrastructure
- **Training code**: [`training/train_laser.py`](training/train_laser.py) — single codebase, config-flag switching
- **Launcher**: [`scripts/run_train.sh`](scripts/run_train.sh) — handles conda activate + faulthandler + PID persistence
- **Conda env**: `fluxphased` (Python 3.11 + torch + warp-lang) — see each algorithm's `environment/README.md`
- **Warp kernel cache**: `$HOME/.cache/warp/` (cwd-independent)

## Quick start
```bash
# Run any algorithm in any order — they are fully isolated by absolute checkpoint_dir
bash algorithms/pspfix/code/run.sh   # ~3.4h
bash algorithms/mappo/code/run.sh    # ~3.8h
bash algorithms/ippo/code/run.sh     # ~3.4h
```

## Experiment archive
Each algorithm has a frozen-results archive under `experiments/`:
- [`experiments/phase1_pfsp_seed42/`](experiments/phase1_pfsp_seed42/) — PfspFix
- [`experiments/phase1.5_mappo_seed42/`](experiments/phase1.5_mappo_seed42/) — MAPPO
- (IPPO archive pending run)

## Reproducibility
- All three algorithms use seed=42 with `set_global_seed()` (random/np/torch/cudnn.deterministic)
- Bit-exact equivalence between `algorithms/<algo>/code/run.sh` and legacy `scripts/run_train.sh configs/laser_25x25_*.yaml` is enforced by sharing the same `train_laser.py` and identical config values (only `league.checkpoint_dir` differs — absolute vs relative)

## Repo structure (top-level APP entry)
- [`LICENSE`](LICENSE) — placeholder (TBD)
- [`AGENTS.md`](AGENTS.md) — this file (repo-level APP entry)
- [`APP_PUBLICATION.json`](APP_PUBLICATION.json) — repo-level manifest
- [`algorithms/`](algorithms/) — three APP-compliant algorithm folders
- [`experiments/`](experiments/) — frozen-result archives
- [`training/`](training/) — shared training code
- [`scripts/`](scripts/) — shared launchers + experiment tooling
- [`configs/`](configs/) — legacy config paths (frozen, do not modify)

## Citation
(To be filled when paper is public.)
