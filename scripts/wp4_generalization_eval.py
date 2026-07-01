"""WP4 generalization matrix evaluator.

EAAI Q1 expects demonstration that the trained policy generalizes across
unseen deployment conditions (PAPER_PLAN_EAAI.md §WP4):
  - Target dynamics: static / uniform / maneuvering
  - Deployment geometry: baseline / angle / n_radars
  - EW conditions: none / jam / exposure

This harness loads a wp1_gate-trained policy and evaluates it on a list of
"test configs" — each one a standalone YAML describing a different deployment
condition. Reports a train×test generalization matrix.

Test configs can be auto-generated via scripts/split_wp4_configs.py or
hand-authored. Format: same as wp1_gate.yaml with the relevant field
overridden (e.g. vehicle_speed_ms=0 for static target).

Usage:
    python scripts/wp4_generalization_eval.py \\
        --checkpoint-dir checkpoints/wp1_gate_seed42 \\
        --train-config configs/wp1_gate.yaml \\
        --test-configs configs/wp4_*.yaml \\
        --n-eval-games 50 --max-steps 500

Outputs:
    logs/wp4_generalization.json     structured metrics
    logs/wp4_generalization.log     human-readable table
    figures/wp4_generalization.pdf  bar chart of retention per condition
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from algo._shared.train import compute_env_params
from scripts.wp3_robustness_eval import (
    find_latest_checkpoint,
    load_trained_policy,
    evaluate_cell,
    resolve_kill_radius,
)


DEFAULT_RETENTION_THRESHOLD = 0.70


def _detect_variation_axis(test_config: dict, train_config: dict) -> str:
    """Identify which dimension this test config varies vs train.

    Returns a short label like 'dynamics', 'geometry', 'ew', or 'mixed'.
    """
    env_t = train_config.get("env", {})
    env_x = test_config.get("env", {})
    deltas = []
    if env_x.get("vehicle_speed_ms", env_t.get("vehicle_speed_ms")) != \
            env_t.get("vehicle_speed_ms"):
        deltas.append("dynamics")
    if env_x.get("n_radars", env_t.get("n_radars")) != env_t.get("n_radars"):
        deltas.append("n_radars")
    if env_x.get("map_size", env_t.get("map_size")) != env_t.get("map_size"):
        deltas.append("map")
    if env_x.get("min_radar_baseline_m",
                  env_t.get("min_radar_baseline_m")) != \
            env_t.get("min_radar_baseline_m"):
        deltas.append("baseline")
    # Sensing noise / EW
    sn_x = test_config.get("sensing_noise", {})
    sn_t = train_config.get("sensing_noise", {})
    for k in ("jam_gain", "exposure_gain", "jam_level"):
        if sn_x.get(k, sn_t.get(k, 0.0)) != sn_t.get(k, 0.0):
            deltas.append("ew")
            break

    if not deltas:
        return "identity"
    if len(deltas) == 1:
        return deltas[0]
    return "+".join(deltas)


def plot_retention_bar(
    out_path: Path,
    conditions: List[str],
    kill_retention: List[float],
    illum_retention: List[float],
    threshold: float,
):
    """Grouped bar chart of retention per generalization condition."""
    if not conditions:
        return
    n = len(conditions)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10.0, 1.6 * n), 5.5))
    bars1 = ax.bar(x - width / 2, kill_retention, width,
                   label='kill_rate retention', color='steelblue')
    bars2 = ax.bar(x + width / 2, illum_retention, width,
                   label='illum_progress retention', color='darkorange')

    ax.axhline(threshold, color='r', linestyle='--', linewidth=1.2,
               label=f'PASS threshold ({threshold:.0%})')
    ax.axhline(1.0, color='g', linestyle=':', alpha=0.4,
               label='Ideal (100%)')

    for bar, val in zip(bars1, kill_retention):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, illum_retention):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha='right')
    ax.set_ylabel('Retention vs train condition')
    ax.set_title('WP4 Generalization — Performance Retention on OOD Conditions')
    ax.set_ylim(0, max(1.2, max(max(kill_retention), max(illum_retention)) + 0.1))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--train-config", type=Path,
                    default=Path("configs/wp1_gate.yaml"),
                    help="The training-time config (defines baseline retention)")
    ap.add_argument("--test-configs", nargs="+", required=True,
                    help="List of test config YAMLs (OOD conditions)")
    ap.add_argument("--n-eval-games", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--retention-threshold", type=float,
                    default=DEFAULT_RETENTION_THRESHOLD,
                    help="PASS threshold for kill_rate retention (default 0.70)")
    ap.add_argument("--output-json", type=Path,
                    default=Path("logs/wp4_generalization.json"))
    ap.add_argument("--output-log", type=Path,
                    default=Path("logs/wp4_generalization.log"))
    ap.add_argument("--output-fig", type=Path,
                    default=Path("figures/wp4_generalization.pdf"))
    ap.add_argument("--kill-radius-override", type=float, default=None,
                    help="Test kr override (m). If unset, auto-detect from training log.")
    args = ap.parse_args()

    # Load training config (= the baseline condition)
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)
    if args.device:
        train_config["env"]["device"] = args.device
    if args.num_envs is not None:
        train_config["env"]["num_envs"] = args.num_envs

    # Resolve kr to match policy's training-time operating point
    resolved_kr = resolve_kill_radius(
        train_config, args.checkpoint_dir, args.kill_radius_override,
    )
    train_config["env"]["kill_radius_m"] = resolved_kr

    # Locate trained checkpoint
    ckpt_red = find_latest_checkpoint(args.checkpoint_dir, role="main", team=0)
    ckpt_blue = find_latest_checkpoint(args.checkpoint_dir, role="main", team=1)
    if ckpt_red is None or ckpt_blue is None:
        print(f"ERROR: no main_team*_gen*.pt in {args.checkpoint_dir}",
              file=sys.stderr)
        sys.exit(2)

    env_params = compute_env_params(train_config)
    device = train_config["env"].get("device", "cuda")

    print("=" * 72)
    print("WP4 Generalization Matrix Evaluator")
    print("=" * 72)
    print(f"Checkpoint      : {ckpt_red}")
    print(f"Train config    : {args.train_config}")
    print(f"Test configs    : {len(args.test_configs)} OOD condition(s)")
    for tc in args.test_configs:
        print(f"  - {tc}")
    print(f"Eval            : {args.n_eval_games} games × {args.max_steps} steps")
    print(f"PASS threshold  : kill_rate retention ≥ {args.retention_threshold:.0%}")
    print()

    # Load trained policy
    red_trainer = load_trained_policy(ckpt_red, train_config, env_params,
                                       team=0, device=device)
    blue_trainer = load_trained_policy(ckpt_blue, train_config, env_params,
                                        team=1, device=device)

    # Baseline (train condition)
    print("\n--- Baseline (train condition) ---")
    t0 = time.time()
    baseline = evaluate_cell(
        train_config, red_trainer, blue_trainer,
        args.n_eval_games, args.max_steps, "baseline",
    )
    baseline["elapsed_s"] = time.time() - t0
    print(f"  kill_rate={baseline['kill_rate']:.3f}  "
          f"illum={baseline['mean_illumination_progress']:.4f}  "
          f"({baseline['elapsed_s']:.1f}s)")

    baseline_kr = baseline["kill_rate"]
    baseline_illum = baseline["mean_illumination_progress"]

    # OOD conditions
    results: Dict[str, dict] = {"baseline": baseline}
    conditions: List[str] = []
    kr_retentions: List[float] = []
    illum_retentions: List[float] = []

    for cfg_path in args.test_configs:
        cfg_p = Path(cfg_path)
        with open(cfg_p) as f:
            test_config = yaml.safe_load(f)
        if args.device:
            test_config["env"]["device"] = args.device
        if args.num_envs is not None:
            test_config["env"]["num_envs"] = args.num_envs
        # Force the resolved kr on OOD configs too (they inherit wp1_gate.yaml's
        # 0.2m which is wrong for partial-training checkpoints).
        test_config["env"]["kill_radius_m"] = resolved_kr

        axis = _detect_variation_axis(test_config, train_config)
        label = f"{cfg_p.stem}\n({axis})"
        print(f"\n--- {cfg_p.name} (varies: {axis}) ---")

        t0 = time.time()
        metrics = evaluate_cell(
            test_config, red_trainer, blue_trainer,
            args.n_eval_games, args.max_steps, cfg_p.stem,
        )
        metrics["elapsed_s"] = time.time() - t0
        metrics["variation_axis"] = axis

        kr_ret = (metrics["kill_rate"] / baseline_kr) if baseline_kr > 1e-6 else 0.0
        il_ret = (metrics["mean_illumination_progress"] / baseline_illum) \
            if baseline_illum > 1e-6 else 0.0
        metrics["kill_rate_retention"] = kr_ret
        metrics["illum_retention"] = il_ret
        metrics["pass"] = bool(kr_ret >= args.retention_threshold)

        results[cfg_p.stem] = metrics
        conditions.append(label)
        kr_retentions.append(kr_ret)
        illum_retentions.append(il_ret)

        print(f"  kill_rate={metrics['kill_rate']:.3f} (retention {kr_ret:.1%})  "
              f"illum={metrics['mean_illumination_progress']:.4f} "
              f"(retention {il_ret:.1%})  "
              f"{'PASS' if metrics['pass'] else 'FAIL'}  "
              f"({metrics['elapsed_s']:.1f}s)")

    # Summary
    print()
    print("=" * 72)
    print("WP4 Generalization Summary")
    print("=" * 72)
    print(f"{'Condition':<40} {'kill_rate':>10} {'retention':>10} "
          f"{'illum':>8} {'retention':>10} {'PASS':>6}")
    print("-" * 72)
    print(f"{'baseline (train)':<40} "
          f"{baseline_kr:>10.3f} {'—':>10} "
          f"{baseline_illum:>8.4f} {'—':>10} {'—':>6}")
    n_pass = 0
    for cfg_path in args.test_configs:
        stem = Path(cfg_path).stem
        m = results[stem]
        if m["pass"]:
            n_pass += 1
        print(f"{stem:<40} "
              f"{m['kill_rate']:>10.3f} {m['kill_rate_retention']:>9.1%} "
              f"{m['mean_illumination_progress']:>8.4f} {m['illum_retention']:>9.1%} "
              f"{'PASS' if m['pass'] else 'FAIL':>6}")
    print("-" * 72)
    overall = n_pass == len(args.test_configs)
    print(f"OVERALL: {n_pass}/{len(args.test_configs)} conditions pass "
          f"(threshold {args.retention_threshold:.0%} kill_rate retention)")

    # JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "train_config": str(args.train_config),
        "test_configs": [str(p) for p in args.test_configs],
        "n_eval_games": args.n_eval_games,
        "max_steps": args.max_steps,
        "retention_threshold": args.retention_threshold,
        "results": results,
        "overall_pass": overall,
        "n_conditions_pass": n_pass,
        "n_conditions_total": len(args.test_configs),
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[JSON] wrote {args.output_json}")

    # Plot
    if conditions:
        args.output_fig.parent.mkdir(parents=True, exist_ok=True)
        plot_retention_bar(
            args.output_fig, conditions,
            kr_retentions, illum_retentions,
            args.retention_threshold,
        )
        print(f"[Fig ] wrote {args.output_fig}")

    # Log
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_log, "a") as f:
        f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"checkpoint_dir={args.checkpoint_dir}\n")
        f.write(f"baseline_kr={baseline_kr:.4f} baseline_illum={baseline_illum:.4f}\n")
        for cfg_path in args.test_configs:
            stem = Path(cfg_path).stem
            m = results[stem]
            f.write(f"  {stem}: axis={m['variation_axis']} "
                    f"kr={m['kill_rate']:.4f} (ret {m['kill_rate_retention']:.1%}) "
                    f"illum={m['mean_illumination_progress']:.4f} "
                    f"(ret {m['illum_retention']:.1%}) "
                    f"{'PASS' if m['pass'] else 'FAIL'}\n")
        f.write(f"OVERALL: {n_pass}/{len(args.test_configs)} pass\n")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
