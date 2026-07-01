#!/usr/bin/env python3
"""
write_metadata.py — emit metadata.json for a run directory.

Captures the fields a downstream agent needs to reproduce / understand a run:
  - config snapshot path
  - git commit SHA (and dirty flag)
  - seed
  - start/end timestamps, wall-clock hours
  - GPU
  - command to reproduce
  - host / pid

USAGE:
  python scripts/experiments/write_metadata.py \
      --run-dir experiments/phase1.5_mappo_seed42 \
      --config configs/laser_25x25_mappo.yaml \
      --seed 42 \
      --start "2026-07-01T11:30:00+08:00" \
      --end   "2026-07-01T14:35:00+08:00" \
      --reproduce "bash scripts/run_train.sh configs/laser_25x25_mappo.yaml logs/phase1.5_mappo.log"
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git(args: list[str], default: str = "") -> str:
    try:
        out = subprocess.run(
            ["git"] + args,
            check=False, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or default
    except Exception:
        return default


def _gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default=None, help="ISO-8601 start timestamp")
    ap.add_argument("--end", default=None, help="ISO-8601 end timestamp")
    ap.add_argument("--reproduce", required=True,
                    help="exact shell command that re-runs this experiment")
    ap.add_argument("--notes", default="", help="free-form notes field")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    commit_sha = _git(["rev-parse", "HEAD"])
    commit_dirty = bool(_git(["status", "--porcelain"]))
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])

    start_iso = args.start or datetime.now(timezone.utc).isoformat()
    end_iso = args.end or datetime.now(timezone.utc).isoformat()
    wall_h: float | None = None
    try:
        s = datetime.fromisoformat(start_iso)
        e = datetime.fromisoformat(end_iso)
        wall_h = round((e - s).total_seconds() / 3600.0, 3)
    except Exception:
        pass

    meta = {
        "run_id": args.run_dir.name,
        "purpose": (
            "Phase 1.5 MAPPO baseline — team critic (CTDE) + uniform opponent "
            "sampling. Tests whether AlphaStar PFSP league is necessary vs "
            "vanilla MAPPO under identical recipe."
        ) if "mappo" in args.run_dir.name.lower() else (
            "Phase 1 PfspFix baseline — verified recipe, seed=42 bit-exact. "
            "Reference run for all Phase 1.5+ comparisons."
        ),
        "seed": args.seed,
        "config_path": str(args.config),
        "config_snapshot": str(args.run_dir / "config.yaml"),
        "git": {
            "commit": commit_sha,
            "branch": branch,
            "dirty": commit_dirty,
            "repo_root": str(repo_root),
        },
        "host": {
            "hostname": os.uname().nodename,
            "gpu": _gpu_name(),
        },
        "timing": {
            "start_iso": start_iso,
            "end_iso": end_iso,
            "wall_clock_hours": wall_h,
        },
        "reproduce_cmd": args.reproduce,
        "notes": args.notes,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }

    args.run_dir.mkdir(parents=True, exist_ok=True)
    out = args.run_dir / "metadata.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  commit={commit_sha[:12]} dirty={commit_dirty} seed={args.seed} wall_h={wall_h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
