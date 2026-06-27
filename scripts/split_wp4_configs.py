"""WP4 OOD test-config generator.

Reads configs/wp1_gate.yaml (the training condition) and produces 6 test
configs varying one axis at a time:

  Dynamics:
    wp4_dynamics_static.yaml       vehicle_speed_ms: 0 (stationary target)
    wp4_dynamics_maneuver.yaml     vehicle_speed_ms: 60 (aggressive maneuver)

  Geometry:
    wp4_geom_tight_baseline.yaml   min_radar_baseline_m: 2000 (sub-optimal)
    wp4_geom_wide_baseline.yaml    min_radar_baseline_m: 8000 (over-spread)

  EW:
    wp4_ew_jam.yaml                sensing_noise.jam_gain: 1.0
    wp4_ew_exposure.yaml           sensing_noise.jam_gain: 1.0, exposure_gain: 1.0

Each output is a STANDALONE config (full override, no inheritance from
wp1_gate). Run any one via:
    python -m training.train --config configs/wp4_dynamics_static.yaml

Or evaluate a trained policy on all of them via:
    python scripts/wp4_generalization_eval.py \\
        --checkpoint-dir checkpoints/wp1_gate_seed42 \\
        --test-configs configs/wp4_*.yaml
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


# (filename, description, mutator fn)
VARIATIONS = [
    (
        "wp4_dynamics_static",
        "OOD: target is stationary (vehicle_speed_ms=0)",
        lambda cfg: _set(cfg, "env.vehicle_speed_ms", 0.0),
    ),
    (
        "wp4_dynamics_maneuver",
        "OOD: target aggressively maneuvers (vehicle_speed_ms=60)",
        lambda cfg: _set(cfg, "env.vehicle_speed_ms", 60.0),
    ),
    (
        "wp4_geom_tight_baseline",
        "OOD: tight radar baseline (2 km, sub-optimal geometry)",
        lambda cfg: _set(cfg, "env.min_radar_baseline_m", 2000.0),
    ),
    (
        "wp4_geom_wide_baseline",
        "OOD: wide radar baseline (8 km, over-spread)",
        lambda cfg: _set(cfg, "env.min_radar_baseline_m", 8000.0),
    ),
    (
        "wp4_ew_jam",
        "OOD: enemy active jammer (jam_gain=1.0, no exposure game)",
        lambda cfg: _set(cfg, "sensing_noise.jam_gain", 1.0),
    ),
    (
        "wp4_ew_exposure",
        "OOD: home-on-jam exposure game (jam_gain=1.0, exposure_gain=1.0)",
        lambda cfg: (_set(cfg, "sensing_noise.jam_gain", 1.0),
                     _set(cfg, "sensing_noise.exposure_gain", 1.0)),
    ),
]


def _set(cfg: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=Path("configs/wp1_gate.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("configs"))
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing files (default: skip)")
    args = ap.parse_args()

    with open(args.base) as f:
        base_cfg = yaml.safe_load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for name, desc, mutate in VARIATIONS:
        out_path = args.out_dir / f"{name}.yaml"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path} (exists; pass --overwrite to regenerate)")
            continue

        cfg = copy.deepcopy(base_cfg)
        result = mutate(cfg)
        # mutate may return a tuple of results — discard, we just want side-effect

        # Inject description as top-level comment
        header = (
            f"# ============================================================\n"
            f"# {name}.yaml — auto-generated WP4 OOD test config\n"
            f"# {desc}\n"
            f"# Base: {args.base}\n"
            f"# ============================================================\n\n"
        )
        body = yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)
        out_path.write_text(header + body)
        print(f"[wrote] {out_path}")
        written.append(out_path)

    print(f"\nGenerated {len(written)} test configs.")
    print("Evaluate a trained policy on all of them via:")
    print(f"  python scripts/wp4_generalization_eval.py \\")
    print(f"      --checkpoint-dir checkpoints/wp1_gate_seed42 \\")
    print(f"      --train-config {args.base} \\")
    print(f"      --test-configs configs/wp4_*.yaml")


if __name__ == "__main__":
    main()
