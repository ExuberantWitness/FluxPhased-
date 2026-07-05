# Environment (shared across PfspFix / MAPPO / IPPO / COMA)

All four algorithms use the **same** conda env and Warp kernel cache — only
config / seeds / pinned commits differ.

## Conda env
- Name: `fluxphased`
- Python: 3.11
- Activated automatically by `scripts/run_train.sh` (no manual `conda activate` needed)
- Key deps: `torch>=2.1`, `numpy`, `warp-lang`, `pyyaml`, `matplotlib`

## Warp kernel cache
- Location: `$HOME/.cache/warp/`
- Cwd-independent — switching between `algorithms/<algo>/` folders does **not**
  invalidate or recompile Warp kernels.

## GPU
- Verified: 1x RTX 4090 (24 GB) — also runs comfortably on 98 GB
- Memory: ~3 GB fixed overhead + ~6 GB per env (num_envs=12 → ~75 GB peak on 98 GB,
  or set `num_envs=4` for 4090)

## Reproduce
```bash
conda activate fluxphased
cd /home/ubuntu/CODE/FluxPhased-
bash algorithms/<algo>/code/run.sh
```
