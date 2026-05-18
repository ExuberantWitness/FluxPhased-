"""Run the TC-DAMS ablation matrix at 25x25 / P=4 / bins=64 / num_envs=1.

Cells (selected from the IDEA_REPORT v2 matrix):
  R0  Nash baseline + standard PFSP
  R1  TC-DAMS lambda=0.3 + standard PFSP
  R3  TC-DAMS lambda=0.3 + Elo-band PFSP

Each cell runs a Phase A -> B -> C -> D curriculum at the (intentionally
modest) budget set in configs/league_tcdams.yaml. Per-iteration diagnostics
and per-cell summaries are written under artifacts/tcdams_ablation/<cell>/.

Usage:
    python run_tcdams_ablation.py --cells R0 R1 R3 --seed 42
    python run_tcdams_ablation.py --cells R0 --phase a   # phase-restricted

Cells are independent. We launch them sequentially with a single GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


CELLS = {
    "R0": {
        "meta_solver": "nash",
        "tcdams_lambda": 0.0,
        "use_elo_band": False,
    },
    "R1": {
        "meta_solver": "tc_dams",
        "tcdams_lambda": 0.3,
        "use_elo_band": False,
    },
    "R2": {
        "meta_solver": "nash",
        "tcdams_lambda": 0.0,
        "use_elo_band": True,
    },
    "R3": {
        "meta_solver": "tc_dams",
        "tcdams_lambda": 0.3,
        "use_elo_band": True,
    },
    "R1lo": {
        "meta_solver": "tc_dams",
        "tcdams_lambda": 0.1,
        "use_elo_band": False,
    },
    "R1hi": {
        "meta_solver": "tc_dams",
        "tcdams_lambda": 1.0,
        "use_elo_band": False,
    },
}


def run_cell(
    cell: str,
    spec: dict,
    config: str,
    seed: int,
    phase: str | None,
    py: str,
    log_dir: str,
) -> dict:
    ckpt_dir = f"checkpoints/league_tcdams/{cell}_seed{seed}"
    log_file = os.path.join(log_dir, f"{cell}_seed{seed}.log")
    os.makedirs(log_dir, exist_ok=True)

    overrides = [
        f"league.meta_solver={spec['meta_solver']}",
        f"league.tcdams_lambda={spec['tcdams_lambda']}",
        f"league.use_elo_band={'true' if spec['use_elo_band'] else 'false'}",
        f"league.checkpoint_dir={ckpt_dir}",
    ]

    cmd = [py, "-m", "training.train", "--config", config, "--seed", str(seed)]
    for ov in overrides:
        cmd += ["--override", ov]
    if phase:
        cmd += ["--phase", phase]

    print(f"[matrix] === Cell {cell} seed={seed} ===")
    print(f"[matrix] cmd: {' '.join(cmd)}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"# Cell {cell} seed={seed} start {datetime.now().isoformat()}\n")
        f.write(f"# Cmd: {' '.join(cmd)}\n\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    summary = {
        "cell": cell,
        "seed": seed,
        "spec": spec,
        "ckpt_dir": ckpt_dir,
        "log_file": log_file,
        "returncode": result.returncode,
        "elapsed_s": elapsed,
    }
    diag_path = os.path.join(ckpt_dir, "diag_history.json")
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            summary["diag_history"] = json.load(f)
    print(f"[matrix] Cell {cell} done in {elapsed/60:.1f}min, rc={result.returncode}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/league_tcdams.yaml")
    parser.add_argument("--cells", nargs="+", default=["R0", "R1", "R3"],
                        choices=list(CELLS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=["a", "b", "c", "d"], default=None,
                        help="Restrict to one curriculum phase (default: all)")
    parser.add_argument("--py", default=r"C:\Users\zhang\.conda\envs\env_isaacsim\python.exe")
    args = parser.parse_args()

    out_dir = "artifacts/tcdams_ablation"
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "logs")

    matrix_summary = []
    for cell in args.cells:
        summary = run_cell(
            cell, CELLS[cell],
            config=args.config, seed=args.seed, phase=args.phase,
            py=args.py, log_dir=log_dir,
        )
        matrix_summary.append(summary)
        # Persist after each cell so a crash doesn't lose prior cells.
        with open(os.path.join(out_dir, "matrix_summary.json"), "w") as f:
            json.dump(matrix_summary, f, indent=2)

    print(f"\n[matrix] All cells done. Summary at {out_dir}/matrix_summary.json")


if __name__ == "__main__":
    main()
