"""WP3.1 Fig C — Achieved localization RMSE vs CRLB ratio.

The "physical anchor" measurement for EAAI: proves the trained policy's
multi-static fused sensing actually approaches the Cramér-Rao Lower Bound
in deployment, not just in theory. This is the headline result for §3.1.

Pipeline:
  1. Load trained wp1_gate policy (latest checkpoint, both teams).
  2. Run N episodes of self-play under the wp1_gate config (deterministic).
  3. After warmup (track_burnin steps), record per-step:
       - Kalman-fused enemy estimate (red_trainer.kalman_tracker._trk_x)
       - True enemy radar position (env.radar_pos)
  4. Compute achieved RMSE = sqrt(mean((estimate - truth)^2)).
  5. Compute theoretical CRLB at the runtime geometry (from wp3_crlb_anchor).
  6. Report: achieved RMSE, CRLB, ratio (target ≤ 1.3× per PAPER_PLAN_EAAI.md §5).
  7. Plot Fig C: achieved vs CRLB vs step.

Usage:
    python scripts/wp3_crlb_achieved.py \\
        --checkpoint-dir checkpoints/wp1_gate_seed42 \\
        --baseline-config configs/wp1_gate.yaml \\
        --n-episodes 20 --max-steps 300

Outputs:
    figures/wp3_crlb_achieved.pdf    (Fig C)
    logs/wp3_crlb_achieved.json      (structured metrics)
    logs/wp3_crlb_achieved.log       (appended human-readable report)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from radar_sim.gpu.vec_mfar_env import MFARVecEnv
from training.laser.episode import LaserEpisodeRunner
from training.laser.sensing import enforce_radar_baseline
from training.train import compute_env_params

# Reuse policy loader from wp3_robustness_eval (same architecture)
from scripts.wp3_robustness_eval import (
    find_latest_checkpoint,
    load_trained_policy,
    make_env,
    resolve_kill_radius,
)
# Reuse CRLB math from wp3_crlb_anchor
from scripts.wp3_crlb_anchor import (
    crlb_for_geometry,
    RANGE_SIGMA_M,
    CROSSRANGE_FACTOR,
    TRACK_BURNIN,
)


# ---------------------------------------------------------------------------
# Episode runner with per-step RMSE logging
# ---------------------------------------------------------------------------

def run_episodes_with_rmse_log(
    config: dict,
    red_trainer,
    blue_trainer,
    n_episodes: int,
    max_steps: int,
    warmup_steps: int,
) -> dict:
    """Run N episodes of self-play, log per-step fused estimate + truth.

    Returns dict with:
      per_step_errors: list of [n_logged_steps, E] tensors (red team's enemy-0 error)
      per_step_cr_sigmas: list of crossrange sigma at each step (for CRLB scaling)
      per_step_baselines: list of actual radar baseline (m) at each step
      final_baseline_m: deployment baseline averaged over logged steps
    """
    env = make_env(config)
    pulses_per_control = config["env"].get("pulses_per_control", 5)
    runner = LaserEpisodeRunner(env, pulses_per_control=pulses_per_control,
                                device=config["env"].get("device", "cuda"))

    min_baseline = float(config["env"].get("min_radar_baseline_m", 5000.0))
    half_x = float(config["env"].get("map_size", [20000.0, 20000.0])[0]) / 2.0
    half_y = float(config["env"].get("map_size", [20000.0, 20000.0])[1]) / 2.0

    E = env.num_envs
    n_teams = env.n_teams
    enemy_idx_red = int(env.battlefield.team_radar_indices[1][0])  # red's enemy-0

    per_step_sq_errors: List[torch.Tensor] = []   # squared error norm
    per_step_cr_sigmas: List[float] = []
    per_step_baselines: List[float] = []

    episodes_done = 0
    batch_episodes_target = n_episodes
    n_batches = max(1, math.ceil(batch_episodes_target / E))

    print(f"[RMSE eval] {n_batches} batches × {E} envs = up to "
          f"{n_batches * E} episodes, {max_steps} steps each "
          f"(warmup {warmup_steps} steps excluded)")

    for batch in range(n_batches):
        runner.reset(red_trainer=red_trainer, blue_trainer=blue_trainer)
        if min_baseline > 0:
            enforce_radar_baseline(env, min_baseline)

        red_trainer.kalman_tracker.reset()
        red_trainer.kalman_tracker.ensure_alloc(E, n_teams, torch.device(env.device))
        red_trainer.kalman_tracker._initialized = True

        for step in range(max_steps):
            out = runner.step_control(red_trainer, blue_trainer, deterministic=True)
            result = out["result"]
            if result is None:
                break

            # Log error only AFTER warmup (Kalman has converged)
            if step >= warmup_steps and red_trainer.kalman_tracker._trk_x is not None:
                # Fused estimate of red team's enemy-0
                est_x = red_trainer.kalman_tracker._trk_x[:, 0, 0, 0]  # [E]
                est_y = red_trainer.kalman_tracker._trk_x[:, 0, 0, 1]  # [E]
                # True enemy position (xy only; tracker is 2D)
                true_xy = env.radar_pos[:, enemy_idx_red, :2]  # [E, 2]
                err_x = est_x - true_xy[:, 0]
                err_y = est_y - true_xy[:, 1]
                sq_err = err_x * err_x + err_y * err_y   # [E]
                per_step_sq_errors.append(sq_err.detach().cpu().clone())

                # Per-step crossrange sigma (scales with range)
                # Use mean range from own team radars to enemy
                own_idx = env.battlefield.team_radar_indices[0]
                own_pos = env.radar_pos[:, own_idx, :2]   # [E, 2, 2]
                ranges = (true_xy.unsqueeze(1) - own_pos).norm(dim=-1)   # [E, n_own]
                mean_range = ranges.mean().item()
                per_step_cr_sigmas.append(CROSSRANGE_FACTOR * mean_range)

                # Per-step actual baseline
                a, b = int(own_idx[0]), int(own_idx[1])
                baseline = (env.radar_pos[:, a, :2] - env.radar_pos[:, b, :2]
                            ).norm(dim=-1).mean().item()
                per_step_baselines.append(baseline)

            if result["dones"].any():
                # Some envs finished — keep counting only the unfinished ones
                episodes_done += int((~result["dones"]).sum().item())
                if result["dones"].all():
                    break

    env.destroy()
    del env
    torch.cuda.empty_cache()

    return {
        "per_step_sq_errors": per_step_sq_errors,   # list of [E] tensors
        "per_step_cr_sigmas": per_step_cr_sigmas,
        "per_step_baselines": per_step_baselines,
        "episodes_observed": episodes_done,
        "n_steps_logged": len(per_step_sq_errors),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_achieved_rmse(log_data: dict) -> dict:
    """Compute achieved RMSE statistics from logged squared errors."""
    sq_errors = log_data["per_step_sq_errors"]
    if not sq_errors:
        return {"rmse_m": float("inf"), "n_samples": 0}

    all_sq = torch.cat([s.flatten() for s in sq_errors])
    n = all_sq.numel()
    mean_sq = float(all_sq.mean().item())
    rmse = math.sqrt(mean_sq)

    # Per-step RMSE (for the convergence plot)
    per_step_rmse = [math.sqrt(float(s.mean().item())) for s in sq_errors]

    return {
        "rmse_m": rmse,
        "mean_sq_err_m2": mean_sq,
        "n_samples": n,
        "per_step_rmse_m": per_step_rmse,
    }


def compute_crlb_at_runtime(log_data: dict, track_burnin: int = TRACK_BURNIN) -> dict:
    """Compute theoretical CRLB at the runtime-observed geometry.

    Uses mean radar baseline + target range over logged steps.
    """
    baselines = log_data["per_step_baselines"]
    cr_sigmas = log_data["per_step_cr_sigmas"]
    if not baselines:
        return {"rmse_tracked_m": float("inf"), "rmse_static_m": float("inf")}

    mean_baseline_m = float(np.mean(baselines))
    mean_cr_sigma = float(np.mean(cr_sigmas))

    # 2-radar deployment along x-axis from origin
    radars = np.array([
        [0.0, 0.0],
        [mean_baseline_m, 0.0],
    ])
    # Target range from CRLB anchor: use mean crossrange sigma to back out R
    # σ_cr = CROSSRANGE_FACTOR × R → R = σ_cr / CROSSRANGE_FACTOR
    target_range_m = mean_cr_sigma / CROSSRANGE_FACTOR
    target = np.array([target_range_m * 0.6, target_range_m * 0.8])  # 3-4-5 triangle

    res_static = crlb_for_geometry(radars, target, n_effective=1)
    res_tracked = crlb_for_geometry(radars, target, n_effective=track_burnin)

    return {
        "rmse_static_m": res_static["rmse_static_m"],
        "rmse_tracked_m": res_tracked["rmse_tracked_m"],
        "mean_baseline_m": mean_baseline_m,
        "mean_target_range_m": target_range_m,
        "mean_cr_sigma_m": mean_cr_sigma,
        "gdop": res_tracked["gdop"],
        "fim_det": res_tracked["fim_det"],
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_fig_c(
    out_path: Path,
    achieved_per_step: List[float],
    crlb_tracked: float,
    crlb_static: float,
    warmup_steps: int,
):
    """Fig C: achieved RMSE vs step, with CRLB lines."""
    if not achieved_per_step:
        print("[Fig C] no data to plot")
        return

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    steps = np.arange(len(achieved_per_step)) + warmup_steps
    rmse_arr = np.array(achieved_per_step)

    ax.plot(steps, rmse_arr, 'b-', linewidth=1.5, label='Achieved RMSE (Kalman-fused)')
    ax.axhline(crlb_tracked, color='g', linestyle='--', linewidth=1.5,
               label=f'Tracked CRLB (N={TRACK_BURNIN}): {crlb_tracked:.3f} m')
    ax.axhline(crlb_static, color='r', linestyle=':', linewidth=1.5,
               label=f'Static CRLB (N=1): {crlb_static:.3f} m')
    ax.axhline(0.2, color='orange', linestyle=':', alpha=0.5,
               label='kill_radius (0.2 m)')

    ax.set_xlabel('Control step (post-warmup)')
    ax.set_ylabel('Localization RMSE (m)')
    title_ratio = (np.mean(rmse_arr) / crlb_tracked) if crlb_tracked > 0 else float('inf')
    ax.set_title(f'Fig C — Achieved RMSE vs CRLB '
                 f'(mean ratio = {title_ratio:.2f}×)')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--baseline-config", type=Path, default=Path("configs/wp1_gate.yaml"))
    ap.add_argument("--n-episodes", type=int, default=20,
                    help="Number of episodes to run (more = tighter estimate)")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="Steps per episode (must exceed warmup)")
    ap.add_argument("--warmup-steps", type=int, default=TRACK_BURNIN,
                    help="Steps to skip before logging (Kalman convergence)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-envs", type=int, default=4,
                    help="Parallel envs (reduced from 12 to fit alongside WP1)")
    ap.add_argument("--output-json", type=Path,
                    default=Path("logs/wp3_crlb_achieved.json"))
    ap.add_argument("--output-log", type=Path,
                    default=Path("logs/wp3_crlb_achieved.log"))
    ap.add_argument("--output-fig", type=Path,
                    default=Path("figures/wp3_crlb_achieved.pdf"))
    ap.add_argument("--ratio-target", type=float, default=1.3,
                    help="PASS threshold for achieved/CRLB ratio (default 1.3×)")
    ap.add_argument("--kill-radius-override", type=float, default=None,
                    help="Test kr override (m). If unset, auto-detect from training log.")
    args = ap.parse_args()

    if args.max_steps <= args.warmup_steps:
        print(f"ERROR: max-steps ({args.max_steps}) must exceed warmup-steps "
              f"({args.warmup_steps})", file=sys.stderr)
        sys.exit(2)

    with open(args.baseline_config) as f:
        config = yaml.safe_load(f)
    if args.device:
        config["env"]["device"] = args.device
    if args.num_envs is not None:
        config["env"]["num_envs"] = args.num_envs

    # Resolve kr to match policy's training-time operating point
    resolved_kr = resolve_kill_radius(
        config, args.checkpoint_dir, args.kill_radius_override,
    )
    config["env"]["kill_radius_m"] = resolved_kr

    ckpt_red = find_latest_checkpoint(args.checkpoint_dir, role="main", team=0)
    ckpt_blue = find_latest_checkpoint(args.checkpoint_dir, role="main", team=1)
    if ckpt_red is None or ckpt_blue is None:
        print(f"ERROR: no main_team*_gen*.pt in {args.checkpoint_dir}",
              file=sys.stderr)
        sys.exit(2)

    env_params = compute_env_params(config)
    device = config["env"].get("device", "cuda")

    print("=" * 72)
    print("WP3.1 Fig C — Achieved RMSE / CRLB Ratio")
    print("=" * 72)
    print(f"Checkpoint    : {ckpt_red} / {ckpt_blue}")
    print(f"Episodes      : {args.n_episodes} × {args.max_steps} steps "
          f"(warmup {args.warmup_steps})")
    print(f"PASS target   : ratio ≤ {args.ratio_target}×")
    print()

    red_trainer = load_trained_policy(ckpt_red, config, env_params, team=0, device=device)
    blue_trainer = load_trained_policy(ckpt_blue, config, env_params, team=1, device=device)

    t0 = time.time()
    log_data = run_episodes_with_rmse_log(
        config, red_trainer, blue_trainer,
        args.n_episodes, args.max_steps, args.warmup_steps,
    )
    elapsed = time.time() - t0

    achieved = compute_achieved_rmse(log_data)
    crlb = compute_crlb_at_runtime(log_data)

    ratio = achieved["rmse_m"] / crlb["rmse_tracked_m"] if crlb["rmse_tracked_m"] > 0 else float("inf")
    pass_status = ratio <= args.ratio_target

    print()
    print(f"Achieved RMSE  = {achieved['rmse_m']:.4f} m  "
          f"({achieved['n_samples']} samples)")
    print(f"CRLB tracked   = {crlb['rmse_tracked_m']:.4f} m  "
          f"(baseline {crlb['mean_baseline_m']/1000:.2f} km, "
          f"target {crlb['mean_target_range_m']/1000:.2f} km, "
          f"N={TRACK_BURNIN})")
    print(f"CRLB static    = {crlb['rmse_static_m']:.4f} m")
    print(f"Ratio          = {ratio:.3f}×   "
          f"({'PASS' if pass_status else 'FAIL'}, target ≤ {args.ratio_target}×)")
    print(f"GDOP           = {crlb['gdop']:.2f}")
    print(f"Elapsed        = {elapsed:.1f}s")

    # Plot
    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    plot_fig_c(
        args.output_fig,
        achieved["per_step_rmse_m"],
        crlb["rmse_tracked_m"],
        crlb["rmse_static_m"],
        args.warmup_steps,
    )
    print(f"[Fig C] wrote {args.output_fig}")

    # JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "n_episodes_target": args.n_episodes,
        "episodes_observed": log_data["episodes_observed"],
        "n_steps_logged": log_data["n_steps_logged"],
        "achieved": {k: v for k, v in achieved.items() if k != "per_step_rmse_m"},
        "crlb": crlb,
        "ratio": ratio,
        "ratio_target": args.ratio_target,
        "pass": bool(pass_status),
        "elapsed_s": elapsed,
        "per_step_rmse_m": achieved["per_step_rmse_m"],
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[JSON ] wrote {args.output_json}")

    # Human log
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_log, "a") as f:
        f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"checkpoint_dir={args.checkpoint_dir}\n")
        f.write(f"episodes={log_data['episodes_observed']} "
                f"steps_logged={log_data['n_steps_logged']}\n")
        f.write(f"achieved_rmse_m={achieved['rmse_m']:.6f}\n")
        f.write(f"crlb_tracked_m={crlb['rmse_tracked_m']:.6f}\n")
        f.write(f"crlb_static_m={crlb['rmse_static_m']:.6f}\n")
        f.write(f"ratio={ratio:.4f} ({'PASS' if pass_status else 'FAIL'}, "
                f"target ≤ {args.ratio_target})\n")
        f.write(f"baseline_m={crlb['mean_baseline_m']:.1f} "
                f"target_range_m={crlb['mean_target_range_m']:.1f} "
                f"gdop={crlb['gdop']:.2f}\n")

    sys.exit(0 if pass_status else 1)


if __name__ == "__main__":
    main()
