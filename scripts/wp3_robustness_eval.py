"""WP3.2 robustness sweep evaluator.

Loads a wp1_gate-trained policy checkpoint, evaluates it under each of the
5 damage cells defined in PAPER_PLAN_EAAI.md §WP3.2, computes performance
retention vs the undamaged baseline, and emits a structured report +
JSON results file for paper figure generation.

PASS criterion (per cell): kill_rate retention ≥ 70% relative to baseline.
kill_rate = fraction of games that ended in a decisive outcome (someone died)
under the success-gated kill_radius curriculum in force at training time.

Usage:
    # Default: evaluate latest wp1_gate checkpoint on all 5 cells + baseline
    python scripts/wp3_robustness_eval.py \\
        --checkpoint-dir checkpoints/wp1_gate_seed42 \\
        --baseline-config configs/wp1_gate.yaml \\
        --n-eval-games 50 --max-steps 500

    # One cell only (quick smoke)
    python scripts/wp3_robustness_eval.py \\
        --checkpoint-dir checkpoints/wp1_gate_seed42 \\
        --baseline-config configs/wp1_gate.yaml \\
        --cells wp3_clutter_weibull_neg10 --n-eval-games 8 --max-steps 200

Outputs:
    logs/wp3_robustness_eval.json   structured metrics for figure pipeline
    logs/wp3_robustness_eval.log    appended human-readable report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Kill-radius auto-detection (critical for partial-training checkpoints)
# ---------------------------------------------------------------------------

_KR_ANNEAL_RE = re.compile(r"kill_radius anneal:\s*([\d.]+)m\s*→\s*([\d.]+)m")
_KR_INIT_RE = re.compile(r"kill_radius initialized at\s+([\d.]+)m")


def detect_current_kill_radius(log_path: Path) -> Optional[float]:
    """Read training log to find the policy's current kill_radius.

    Training anneals kill_radius from init (e.g., 100m) toward final (0.2m)
    gated on kill_rate ≥ threshold. A gen-N checkpoint was trained at whatever
    kr was in effect at iteration N. Testing it at a different kr is a
    train/test mismatch that yields meaningless results.

    Returns:
        The latest kr value seen in the log (init if no anneal yet),
        or None if log not found / unparseable.
    """
    if not log_path.exists():
        return None
    last_init = None
    last_anneal_to = None
    try:
        with open(log_path) as f:
            for line in f:
                m = _KR_INIT_RE.search(line)
                if m:
                    last_init = float(m.group(1))
                m = _KR_ANNEAL_RE.search(line)
                if m:
                    last_anneal_to = float(m.group(2))
    except Exception:
        return None
    if last_anneal_to is not None:
        return last_anneal_to
    return last_init


def resolve_kill_radius(
    config: dict,
    checkpoint_dir: Path,
    override: Optional[float] = None,
    log_name_tmpl: str = "wp1_gate_seed*.log",
) -> float:
    """Determine the kr to test at.

    Priority:
      1. Explicit --kill-radius-override (most reliable)
      2. Auto-detect from training log in logs/ dir
      3. Fall back to config's env.kill_radius_m (only correct for fully-trained policy)
    """
    cfg_kr = float(config.get("env", {}).get("kill_radius_m", 0.2))
    if override is not None and override > 0:
        if abs(override - cfg_kr) > 1e-3:
            print(f"[kr-override] using {override}m (config says {cfg_kr}m)")
        return float(override)

    # Try auto-detect from training logs
    logs_dir = Path("logs")
    candidates = sorted(logs_dir.glob(log_name_tmpl))
    # Find a log whose stem matches the checkpoint dir's parent name
    ckpt_name = checkpoint_dir.stem  # e.g., wp1_gate_seed42
    matched = [p for p in candidates if ckpt_name in p.stem]
    if matched:
        kr = detect_current_kill_radius(matched[-1])
        if kr is not None and abs(kr - cfg_kr) > 1e-3:
            print(f"[kr-auto] training log {matched[-1].name} shows kr={kr}m "
                  f"(config says {cfg_kr}m). Using {kr}m.")
            return kr

    return cfg_kr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from radar_sim.gpu.vec_mfar_env import MFARVecEnv
from radar_sim.config import DEFAULT_ROWS, DEFAULT_COLS
from training.flux_league import FluxLeague  # noqa: F401  (validates import path)
from training.laser.episode import LaserEpisodeRunner
from training.laser.sensing import enforce_radar_baseline
from training.ppo.actor_critic import create_team_policy
from training.ppo.ppo_trainer import TeamPPOTrainer
from training.train import compute_env_params


# ---------------------------------------------------------------------------
# Damage cells from PAPER_PLAN_EAAI.md §WP3.2
# ---------------------------------------------------------------------------

DEFAULT_CELLS = [
    "wp3_clutter_weibull_neg10",
    "wp3_multipath_2ray",
    "wp3_bias_5m_delay_3step",
    "wp3_slew_limit_60degs",
    "wp3_comm_rate_1kbps",
]

# PASS thresholds (EAAI reviewers expect these to be met for all 5 cells)
RETENTION_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir: Path, role: str = "main",
                           team: int = 0) -> Optional[Path]:
    """Find the highest-generation checkpoint for a given role+team.

    Files are named like 'main_team0_gen5.pt'. Returns None if none found.
    """
    pattern = str(checkpoint_dir / f"{role}_team{team}_gen*.pt")
    paths = glob.glob(pattern)
    if not paths:
        return None

    def gen_num(p: str) -> int:
        m = re.search(r"_gen(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    paths.sort(key=gen_num)
    return Path(paths[-1])


def load_trained_policy(
    checkpoint_path: Path,
    config: dict,
    env_params: dict,
    team: int,
    device: str = "cuda",
) -> TeamPPOTrainer:
    """Build a TeamPPOTrainer with architecture matching the training run
    and load weights from checkpoint_path.
    """
    policy_dict = create_team_policy(
        team=team,
        n_elem=env_params["n_elem"],
        n_pulses=env_params["n_pulses"],
        n_bins=env_params["n_bins"],
        num_output_length=env_params["num_output_length"],
        device=device,
        sub_array_size=config.get("sub_array_size", 0),
        hybrid_fire=config.get("training", {}).get("hybrid_fire", False),
        decouple_value=config.get("training", {}).get("decouple_value", False),
    )

    league_cfg = config.get("league", {})
    ppo_cfg = config.get("ppo", {})
    shared = ppo_cfg.get("shared", {})
    cmd = ppo_cfg.get("commander", {})
    radar = ppo_cfg.get("radar", {})
    ppo_config = dict(
        commander_lr=cmd.get("lr", 3e-4),
        radar_lr=radar.get("lr", 1e-4),
        gamma=shared.get("gamma", 0.99),
        gae_lambda=shared.get("gae_lambda", 0.95),
        commander_clip=cmd.get("clip_range", 0.2),
        radar_clip=radar.get("clip_range", 0.1),
        commander_entropy=cmd.get("entropy_coef", 0.01),
        radar_entropy=radar.get("entropy_coef", 0.02),
        value_coef=shared.get("value_coef", 0.5),
        max_grad_norm=shared.get("max_grad_norm", 0.5),
        n_epochs=shared.get("n_epochs", 3),
        batch_size=shared.get("batch_size", 256),
        buffer_size=shared.get("buffer_size", 2048),
        buffer_size_commander=shared.get("buffer_size_commander", 2048),
        buffer_size_radar=shared.get("buffer_size_radar", 512),
        device=device,
        stealth_weight=config.get("reward_shaping", {}).get("stealth_weight", 0.1),
        reward_shaping_config=config.get("reward_shaping", {}),
    )

    trainer = TeamPPOTrainer(
        commander=policy_dict["commander"],
        radar=policy_dict["radar"],
        **ppo_config,
        task_type=config.get("task_type", "generic"),
        laser_cfg={
            "kill_radius_init": config.get("training", {}).get("kill_radius_init", 50.0),
            "kill_radius_final": config["env"].get("kill_radius_m", 0.2),
            "kill_rate_threshold": config.get("training", {}).get("kill_rate_threshold", 0.5),
            "kill_radius_decay": config.get("training", {}).get("kill_radius_decay", 0.5),
            "residual_scale_m": config.get("training", {}).get("residual_scale_m", 6.0),
            "reward_shaping": config.get("reward_shaping", {}),
            "hybrid_fire": config.get("training", {}).get("hybrid_fire", False),
            "decouple_value": config.get("training", {}).get("decouple_value", False),
            "residual_aim": config.get("training", {}).get("residual_aim", False),
            "min_radar_baseline_m": config.get("env", {}).get("min_radar_baseline_m", 0.0),
        },
        sensing_cfg={
            **config.get("sensing_noise", {}),
            "sensing_bias_m": config.get("env", {}).get("sensing_bias_m", 0.0),
            "control_delay_steps": int(config.get("env", {}).get("control_delay_steps", 0)),
        },
    )

    # init_buffers needs state/action dims from the env, but we don't want
    # to hold a full env just for this. Compute dims analytically.
    n_elem = env_params["n_elem"]
    state_dim = (n_elem * (env_params["n_pulses"] * env_params["n_bins"] + 2 + 4)
                 + 5 + 12 + env_params["num_output_length"])
    action_dim = n_elem * 22
    trainer.init_buffers(state_dim, action_dim, commander_act_dim=5)

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    trainer.commander_trainer.ac.load_state_dict(ckpt["commander"])
    trainer.radar_trainer.ac.load_state_dict(ckpt["radar"])

    # Set kill_radius to the FINAL annealed value, not init (we're testing
    # the converged policy at its trained operating point).
    if hasattr(trainer, "laser_cfg") and trainer.laser_cfg:
        kr_final = float(trainer.laser_cfg.get("kill_radius_final", 0.2))
        trainer.kill_radius = kr_final  # informational; env owns the physics

    trainer.commander_trainer.ac.eval()
    trainer.radar_trainer.ac.eval()
    return trainer


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(config: dict) -> MFARVecEnv:
    """Mirror of training.train.create_env, including damage_config extraction."""
    env_cfg = config.get("env", {})
    kwargs = dict(
        num_envs=env_cfg.get("num_envs", 4),
        n_radars=env_cfg.get("n_radars", 4),
        rows=env_cfg.get("rows", DEFAULT_ROWS),
        cols=env_cfg.get("cols", DEFAULT_COLS),
        pulses_per_cpi=env_cfg.get("pulses_per_cpi", 4),
        fft_size=env_cfg.get("fft_size", 64),
        device=env_cfg.get("device", "cuda"),
        tx_power_w=env_cfg.get("tx_power_w", 1.0),
        n_teams=env_cfg.get("n_teams", 2),
        bandwidth=env_cfg.get("bandwidth", 200e6),
        prf=env_cfg.get("prf", 10e3),
        vehicle_speed_ms=env_cfg.get("vehicle_speed_ms", 20.0),
    )
    for k in ("kill_radius_m", "illumination_time_s", "drone_altitude_m", "map_size"):
        if k in env_cfg:
            kwargs[k] = env_cfg[k]
    damage_keys = (
        "clutter_model", "clutter_shape_k", "clutter_scale_lambda", "clutter_cnr_db",
        "multipath_model", "multipath_delay_spread_ns", "multipath_attenuation_db",
        "max_slew_rate_deg_per_s", "duty_cycle_max",
        "control_delay_steps", "comm_rate_bps", "comm_encoding",
    )
    damage_config = {k: env_cfg[k] for k in damage_keys if k in env_cfg}
    sensing_cfg = config.get("sensing_noise", {})
    for k in ("comm_rate_bps", "comm_encoding"):
        if k not in damage_config and k in sensing_cfg:
            damage_config[k] = sensing_cfg[k]
    if damage_config:
        kwargs["damage_config"] = damage_config
    return MFARVecEnv(**kwargs)


# ---------------------------------------------------------------------------
# Cell evaluation
# ---------------------------------------------------------------------------

def evaluate_cell(
    config: dict,
    red_trainer: TeamPPOTrainer,
    blue_trainer: TeamPPOTrainer,
    n_eval_games: int,
    max_steps: int,
    cell_name: str,
) -> dict:
    """Run n_eval_games of self-play under one damage config.

    Returns dict with: red_win_rate, blue_win_rate, draw_rate, kill_rate,
    mean_illumination_progress, n_games, elapsed_s.
    """
    env = make_env(config)

    # Force kill_radius to the policy's trained operating point. The env
    # constructor reads kill_radius_m from config (which is the *training*
    # final value), but training anneals it down over iters; the loaded
    # policy expects that annealed value. config already has env.kill_radius_m
    # = final value, so no override needed.

    pulses_per_control = config["env"].get("pulses_per_control", 5)
    runner = LaserEpisodeRunner(env, pulses_per_control=pulses_per_control,
                                device=config["env"].get("device", "cuda"))

    min_baseline = float(config["env"].get("min_radar_baseline_m", 5000.0))
    half_map = float(config["env"].get("map_size", [20000.0, 20000.0])[0]) / 2.0

    red_wins = 0.0
    blue_wins = 0.0
    draws = 0.0
    n_red_kills = 0
    n_blue_kills = 0
    total_progress = 0.0
    n_decisive = 0
    n_step_cap = 0

    E = env.num_envs
    games_per_batch = E
    n_batches = max(1, n_eval_games // games_per_batch)
    print(f"[{cell_name}] {n_batches} batches × {games_per_batch} envs = "
          f"{n_batches * games_per_batch} games, {max_steps} steps each")

    for batch in range(n_batches):
        runner.reset(red_trainer=red_trainer, blue_trainer=blue_trainer)
        if min_baseline > 0:
            enforce_radar_baseline(env, min_baseline)

        last_result = None
        for step in range(max_steps):
            out = runner.step_control(red_trainer, blue_trainer, deterministic=True)
            last_result = out["result"]
            if last_result is None:
                break
            if last_result["dones"].any():
                break

        if last_result is None:
            draws += games_per_batch
            n_step_cap += games_per_batch
            continue

        # Tally wins
        dones = last_result["dones"]
        winners = last_result["winners"]
        for e in range(games_per_batch):
            if dones[e]:
                w = int(winners[e].item())
                if w == 0:
                    red_wins += 1.0
                    n_decisive += 1
                elif w == 1:
                    blue_wins += 1.0
                    n_decisive += 1
                else:
                    draws += 1.0
                    n_step_cap += 1
            else:
                draws += 1.0
                n_step_cap += 1

        if "kills" in last_result:
            kills = last_result["kills"]
            n_red_kills += int(kills[:, 0, :].any(dim=-1).sum().item())
            n_blue_kills += int(kills[:, 1, :].any(dim=-1).sum().item())

        progress = env.battlefield.laser.get_illumination_progress()
        total_progress += float(progress.sum().item())

        if (batch + 1) % 5 == 0 or batch == n_batches - 1:
            total_done = red_wins + blue_wins + draws
            print(f"  batch {batch + 1}/{n_batches}  "
                  f"red={red_wins:.0f}  blue={blue_wins:.0f}  "
                  f"draw={draws:.0f}  (total {total_done:.0f})")

    total = max(1, red_wins + blue_wins + draws)
    env.destroy()
    del env
    torch.cuda.empty_cache()

    return {
        "red_win_rate": red_wins / total,
        "blue_win_rate": blue_wins / total,
        "draw_rate": draws / total,
        "kill_rate": n_decisive / total,
        "step_cap_rate": n_step_cap / total,
        "red_kill_count": n_red_kills,
        "blue_kill_count": n_blue_kills,
        "mean_illumination_progress": total_progress / total,
        "n_games": int(total),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_cell_config(cell_name: str, baseline_config: dict) -> dict:
    """Load a wp3_<cell>.yaml and overlay it on baseline_config semantics.

    The cell configs are standalone (task_type + full league/ppo/env), so we
    just load them directly. Returns the merged dict.
    """
    cell_path = Path("configs") / f"{cell_name}.yaml"
    if not cell_path.exists():
        raise FileNotFoundError(f"Cell config not found: {cell_path}")
    with open(cell_path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, required=True,
                    help="Directory containing main_team{0,1}_gen*.pt checkpoints")
    ap.add_argument("--baseline-config", type=Path, default=Path("configs/wp1_gate.yaml"),
                    help="Baseline (no-damage) config path")
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS,
                    help="Damage cells to evaluate (default: all 5)")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip the no-damage baseline run (use if cached)")
    ap.add_argument("--n-eval-games", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-envs", type=int, default=None,
                    help="Override config env.num_envs (8 is a good smoke value)")
    ap.add_argument("--output-json", type=Path,
                    default=Path("logs/wp3_robustness_eval.json"))
    ap.add_argument("--output-log", type=Path,
                    default=Path("logs/wp3_robustness_eval.log"))
    ap.add_argument("--retention-threshold", type=float, default=RETENTION_THRESHOLD,
                    help="PASS threshold for kill_rate retention (default 0.70)")
    ap.add_argument("--kill-radius-override", type=float, default=None,
                    help="Test kr override (m). If unset, auto-detect from training log. "
                         "CRITICAL: gen-N checkpoint was trained at iter-N's kr, not the "
                         "config's final kr. Testing at wrong kr → meaningless results.")
    args = ap.parse_args()

    # Load baseline config
    with open(args.baseline_config) as f:
        baseline_config = yaml.safe_load(f)
    if args.device:
        baseline_config["env"]["device"] = args.device
    if args.num_envs is not None:
        baseline_config["env"]["num_envs"] = args.num_envs

    # Resolve kill_radius: must match what the policy was trained at.
    resolved_kr = resolve_kill_radius(
        baseline_config, args.checkpoint_dir, args.kill_radius_override,
    )
    baseline_config["env"]["kill_radius_m"] = resolved_kr

    # Locate latest checkpoint per team
    ckpt_red = find_latest_checkpoint(args.checkpoint_dir, role="main", team=0)
    ckpt_blue = find_latest_checkpoint(args.checkpoint_dir, role="main", team=1)
    if ckpt_red is None or ckpt_blue is None:
        print(f"ERROR: no main_team*_gen*.pt checkpoints in {args.checkpoint_dir}",
              file=sys.stderr)
        sys.exit(2)
    print(f"Loading team-0 policy: {ckpt_red}")
    print(f"Loading team-1 policy: {ckpt_blue}")

    # Compute env params from baseline config (cells share the same shape)
    env_params = compute_env_params(baseline_config)
    device = baseline_config["env"].get("device", "cuda")

    print("=" * 72)
    print("WP3.2 Robustness Sweep Evaluator")
    print("=" * 72)
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Baseline config: {args.baseline_config}")
    print(f"Cells          : {args.cells}")
    print(f"Eval           : {args.n_eval_games} games × {args.max_steps} steps each")
    print(f"PASS threshold : kill_rate retention ≥ {args.retention_threshold:.2f}")
    print(f"Environment    : E={baseline_config['env'].get('num_envs', 4)}, "
          f"N={env_params['n_elem']}, R={env_params['n_radars']}, "
          f"bins={env_params['n_bins']}")
    print()

    # Load trained policies (one per team — they're distinct checkpoints)
    red_trainer = load_trained_policy(ckpt_red, baseline_config, env_params,
                                       team=0, device=device)
    blue_trainer = load_trained_policy(ckpt_blue, baseline_config, env_params,
                                        team=1, device=device)

    results: Dict[str, dict] = {}

    # Baseline (no damage)
    if not args.skip_baseline:
        print("\n--- Baseline (no damage) ---")
        t0 = time.time()
        baseline_metrics = evaluate_cell(
            baseline_config, red_trainer, blue_trainer,
            args.n_eval_games, args.max_steps, "baseline",
        )
        baseline_metrics["elapsed_s"] = time.time() - t0
        results["baseline"] = baseline_metrics
        print(f"  kill_rate={baseline_metrics['kill_rate']:.3f}  "
              f"illum={baseline_metrics['mean_illumination_progress']:.4f}  "
              f"red_wr={baseline_metrics['red_win_rate']:.3f}  "
              f"({baseline_metrics['elapsed_s']:.1f}s)")
    else:
        if "baseline" not in results:
            print("ERROR: --skip-baseline given but no cached baseline. "
                  "Run without --skip-baseline first.", file=sys.stderr)
            sys.exit(2)

    baseline_kr = results["baseline"]["kill_rate"]
    baseline_illum = results["baseline"]["mean_illumination_progress"]

    # Damage cells
    for cell in args.cells:
        print(f"\n--- {cell} ---")
        cell_config = load_cell_config(cell, baseline_config)
        if args.device:
            cell_config["env"]["device"] = args.device
        if args.num_envs is not None:
            cell_config["env"]["num_envs"] = args.num_envs
        # Force the resolved kr (cell configs inherit wp1_gate.yaml which has
        # kill_radius_m=0.2, but partial-training policy needs the training kr).
        cell_config["env"]["kill_radius_m"] = resolved_kr

        t0 = time.time()
        metrics = evaluate_cell(
            cell_config, red_trainer, blue_trainer,
            args.n_eval_games, args.max_steps, cell,
        )
        metrics["elapsed_s"] = time.time() - t0

        # Retention vs baseline (guard against div-by-zero)
        kr_retention = (metrics["kill_rate"] / baseline_kr) if baseline_kr > 1e-6 else 0.0
        illum_retention = (metrics["mean_illumination_progress"] / baseline_illum) \
            if baseline_illum > 1e-6 else 0.0
        metrics["kill_rate_retention"] = kr_retention
        metrics["illum_retention"] = illum_retention
        metrics["pass"] = bool(kr_retention >= args.retention_threshold)

        results[cell] = metrics
        print(f"  kill_rate={metrics['kill_rate']:.3f} (retention {kr_retention:.2%})  "
              f"illum={metrics['mean_illumination_progress']:.4f} "
              f"(retention {illum_retention:.2%})  "
              f"red_wr={metrics['red_win_rate']:.3f}  "
              f"{'PASS' if metrics['pass'] else 'FAIL'}  "
              f"({metrics['elapsed_s']:.1f}s)")

    # Summary
    print()
    print("=" * 72)
    print("WP3.2 Robustness Summary")
    print("=" * 72)
    print(f"{'Cell':<32} {'kill_rate':>10} {'retention':>10} "
          f"{'illum':>8} {'retention':>10} {'PASS':>6}")
    print("-" * 72)
    print(f"{'baseline (no damage)':<32} "
          f"{results['baseline']['kill_rate']:>10.3f} {'—':>10} "
          f"{results['baseline']['mean_illumination_progress']:>8.4f} {'—':>10} {'—':>6}")
    n_pass = 0
    for cell in args.cells:
        m = results[cell]
        status = "PASS" if m["pass"] else "FAIL"
        if m["pass"]:
            n_pass += 1
        print(f"{cell:<32} "
              f"{m['kill_rate']:>10.3f} {m['kill_rate_retention']:>9.1%} "
              f"{m['mean_illumination_progress']:>8.4f} {m['illum_retention']:>9.1%} "
              f"{status:>6}")
    print("-" * 72)
    overall = n_pass == len(args.cells)
    print(f"OVERALL: {n_pass}/{len(args.cells)} cells pass "
          f"(threshold {args.retention_threshold:.0%} kill_rate retention)")
    if overall:
        print("  → WP3.2 gate cleared. Robustness table ready for paper.")
    else:
        print("  → Some cells failed. Investigate before EAAI submission.")

    # Save JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_dir": str(args.checkpoint_dir),
        "baseline_config": str(args.baseline_config),
        "n_eval_games": args.n_eval_games,
        "max_steps": args.max_steps,
        "retention_threshold": args.retention_threshold,
        "results": results,
        "overall_pass": overall,
        "n_cells_pass": n_pass,
        "n_cells_total": len(args.cells),
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote JSON: {args.output_json}")

    # Append human-readable log
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_log, "a") as f:
        f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"checkpoint_dir={args.checkpoint_dir}\n")
        f.write(f"baseline_kill_rate={baseline_kr:.4f}\n")
        f.write(f"baseline_illum={baseline_illum:.4f}\n")
        for cell, m in results.items():
            if cell == "baseline":
                continue
            f.write(f"  {cell}: kill_rate={m['kill_rate']:.4f} "
                    f"(retention {m['kill_rate_retention']:.2%}) "
                    f"illum={m['mean_illumination_progress']:.4f} "
                    f"(retention {m['illum_retention']:.2%}) "
                    f"{'PASS' if m['pass'] else 'FAIL'}\n")
        f.write(f"OVERALL: {n_pass}/{len(args.cells)} cells pass\n")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
