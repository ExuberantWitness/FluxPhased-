"""WP3.2 helper — split the multi-cell robustness sweep into 5 standalone configs.

The catalog configs/wp3_robustness_sweep.yaml documents all 5 damage cells in
one file for reviewability, but training.train expects a flat config dict (one
top-level cell per file). This script extracts each cell into its own runnable
config file.

Usage:
    python scripts/split_wp3_configs.py
    # → writes configs/wp3_<cell>.yaml for each of the 5 cells

After splitting, each cell is runnable directly:
    python -m training.train --config configs/wp3_clutter_weibull_neg10.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sweep_path = repo_root / "configs" / "wp3_robustness_sweep.yaml"
    out_dir = repo_root / "configs"

    with open(sweep_path) as f:
        sweep = yaml.safe_load(f)

    cells = [k for k in sweep.keys()
             if isinstance(sweep[k], dict) and "task_type" in sweep[k]]

    if not cells:
        print(f"ERROR: no cells with task_type found in {sweep_path}",
              file=sys.stderr)
        sys.exit(2)

    for cell in cells:
        cfg = sweep[cell]
        out_path = out_dir / f"{cell}.yaml"
        with open(out_path, "w") as f:
            f.write(f"# Auto-generated from configs/wp3_robustness_sweep.yaml\n")
            f.write(f"# Cell: {cell}\n")
            f.write(f"# Run: python -m training.train --config configs/{cell}.yaml\n\n")
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        print(f"wrote {out_path.relative_to(repo_root)}")

    print(f"\nSplit {len(cells)} cells. Each runnable via:")
    print(f"  python -m training.train --config configs/<cell>.yaml")


if __name__ == "__main__":
    main()
