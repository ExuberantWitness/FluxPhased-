# Environment (shared across all phase1.5 algorithms)

All phase1.5 algorithms use the **same** conda env and Warp kernel cache — only
config / seeds / pinned commits / EW params differ.

## Conda env
- Name: `fluxphased`
- Python: 3.11
- Activated automatically by `main.py` (no manual `conda activate` needed)
- Key deps: `torch>=2.1`, `numpy`, `warp-lang`, `pyyaml`, `matplotlib`

## Warp kernel cache
- Location: `$HOME/.cache/warp/`
- Cwd-independent — switching between `algo/<algo>/` folders does **not**
  invalidate or recompile Warp kernels.

## GPU
- Verified: 1x RTX PRO 6000 (101.9 GB) — also runs on 4090 (24 GB)
- Memory: ~3 GB fixed overhead + ~6 GB per env (num_envs=12 → ~75 GB peak on
  PRO 6000, or set `num_envs=4` for 4090)

## Reproduce
```bash
conda activate fluxphased
cd /home/ubuntu/CODE/FluxPhased-
python main.py --config algo/ew_mappo/code/config.yaml
```
