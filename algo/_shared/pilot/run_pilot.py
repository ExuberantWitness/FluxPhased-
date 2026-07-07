"""Concerto-RRM pilot harness (A11).

Runs the full pilot matrix:
  methods     × difficulties × seeds
  {classical, mappo, concerto_v1, concerto_v2} × {L0, L1, L3} × {42..46}
  = 4 × 3 × 5 = 60 cells

Each cell runs ConcertoPilotDriver.run() which evaluates n_eval_episodes=50
episodes and returns aggregate QoS metrics. Results are written incrementally
to a CSV to allow partial-failure recovery.

Usage:
    python -m algo._shared.pilot.run_pilot \\
        --methods classical mappo concerto_v1 concerto_v2 \\
        --difficulties L0 L1 L3 \\
        --seeds 42 43 44 45 46 \\
        --mppo-checkpoint checkpoints/laser_mappo/iter_019.pt \\
        --max-steps 50 \\
        --n-eval-episodes 10 \\
        --out experiments/concerto_pilot_results.csv

For the full pilot (per plan §B7): --max-steps 100 --n-eval-episodes 50.
For quick smoke: --methods classical --difficulties L0 --seeds 42 --n-eval-episodes 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Allow running as `python -m` from repo root
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from env.gpu.vec_mfar_env import MFARVecEnv
from env.gpu.qos_rrm import QoSRRMEnv, make_jammer
from algo._shared.baselines.classical_qos_rrm import ClassicalQoSRRM
from algo._shared.concerto import (
    ComposerV1, ComposerV2, ConcertoPilotDriver, make_composer,
)


# ---------------------------------------------------------------------------
# Build per-cell components
# ---------------------------------------------------------------------------

def build_env(num_envs: int = 2, device: str = "cuda", tx_power_w: float = 100.0) -> MFARVecEnv:
    return MFARVecEnv(
        num_envs=num_envs, n_radars=4, n_teams=2, qos_rrm_mode=True,
        rows=5, cols=5, pulses_per_cpi=4, device=device, fft_size=64,
        tx_power_w=tx_power_w,
    )


def build_qos_env(env: MFARVecEnv) -> QoSRRMEnv:
    return QoSRRMEnv(
        env, jam_gain=8.0, exposure_gain=50.0,
        qos_thresholds=dict(pd_thresh=0.3, trace_thresh=0.6,
                             crc_thresh=0.2, jsr_target_db=6.0),
    )


def build_scheduler(env: MFARVecEnv, team: int = 0) -> ClassicalQoSRRM:
    sched = ClassicalQoSRRM(
        env, team=team, qos_floor_per_fn=3,
        jam_gain=8.0, exposure_gain=50.0,
    )
    sched.jam_level = torch.zeros(env.num_envs, env.n_teams, device=env.device)
    return sched


def build_jammer(difficulty: str):
    if difficulty.upper() == "L0":
        return make_jammer("L0", jam_level=0.1)
    elif difficulty.upper() == "L1":
        return make_jammer("L1", tau=4, base_jam=0.3, max_jam=1.0, adaptivity=0.7)
    elif difficulty.upper() == "L3":
        return make_jammer("L3", base_jam=0.5, hidden=64, device="cuda")
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")


def build_composer(variant: str):
    if variant == "v1":
        return make_composer("v1", n_classical_per_rl=3)
    elif variant == "v2":
        return make_composer("v2", theta1_jsr_db=5.0, theta2_trace=0.6,
                              epsilon_margin=0.2, k_commitment=5)
    else:
        raise ValueError(f"Unknown composer variant: {variant}")


def build_rl_trainer(env: MFARVecEnv, team: int, checkpoint_path: Optional[str]):
    """Load a pre-trained MAPPO policy as a trainer-like object.

    Uses SimpleMAPPOTrainer — an inference-only adapter that exposes the
    `get_own_actions(env, team, deterministic, spectrum, events)` API the
    LaserEpisodeRunner expects. Falls back to ClassicalQoSRRM if the
    checkpoint is missing or load fails (degenerate but won't crash).
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        sys.stderr.write(
            f"[WARN] No MAPPO checkpoint; using classical scheduler as RL fallback. "
            f"This makes concerto cells degenerate (classical vs classical).\n")
        return build_scheduler(env, team)
    try:
        from algo._shared.pilot.simple_mappo import SimpleMAPPOTrainer
        trainer = SimpleMAPPOTrainer(
            env, team=team, checkpoint_path=checkpoint_path,
            jam_gain=8.0, exposure_gain=50.0,
        )
        trainer.jam_level = torch.zeros(env.num_envs, env.n_teams, device=env.device)
        return trainer
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to load MAPPO checkpoint {checkpoint_path}: {e}\n")
        sys.stderr.write("       Falling back to classical scheduler as RL.\n")
        return build_scheduler(env, team)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "method", "difficulty", "seed",
    "qos_satisfaction_mean", "qos_satisfaction_std",
    "qos_detect", "qos_track", "qos_comm", "qos_jam",
    "dwell_detect", "dwell_track", "dwell_comm", "dwell_jam",
    "min_dwell_frac",
    "n_episodes", "n_rl_steps_total", "n_classical_steps_total",
    "rl_frac", "wallclock_s",
    "timestamp",
]


def write_csv_row(csv_path: str, metrics: dict):
    """Append one row to CSV. Create with header if new."""
    is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    row = {
        "method": metrics["method"],
        "difficulty": metrics["difficulty"],
        "seed": metrics["seed"],
        "qos_satisfaction_mean": f"{metrics['qos_satisfaction_mean']:.6f}",
        "qos_satisfaction_std": f"{metrics['qos_satisfaction_std']:.6f}",
        "qos_detect": f"{metrics['qos_per_function_mean']['detect']:.6f}",
        "qos_track":  f"{metrics['qos_per_function_mean']['track']:.6f}",
        "qos_comm":   f"{metrics['qos_per_function_mean']['comm']:.6f}",
        "qos_jam":    f"{metrics['qos_per_function_mean']['jam']:.6f}",
        "dwell_detect": f"{metrics['dwell_frac_mean']['detect']:.6f}",
        "dwell_track":  f"{metrics['dwell_frac_mean']['track']:.6f}",
        "dwell_comm":   f"{metrics['dwell_frac_mean']['comm']:.6f}",
        "dwell_jam":    f"{metrics['dwell_frac_mean']['jam']:.6f}",
        "min_dwell_frac": f"{metrics['min_dwell_frac']:.6f}",
        "n_episodes": metrics["n_episodes"],
        "n_rl_steps_total": metrics["n_rl_steps_total"],
        "n_classical_steps_total": metrics["n_classical_steps_total"],
        "rl_frac": f"{metrics['n_rl_steps_total'] / max(1, metrics['n_rl_steps_total'] + metrics['n_classical_steps_total']):.4f}",
        "wallclock_s": f"{metrics['wallclock_s']:.2f}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)
    return row


# ---------------------------------------------------------------------------
# Main pilot loop
# ---------------------------------------------------------------------------

def run_pilot(
    methods: List[str],
    difficulties: List[str],
    seeds: List[int],
    mppo_checkpoint: Optional[str],
    max_steps: int,
    n_eval_episodes: int,
    out_csv: str,
    num_envs: int = 2,
    device: str = "cuda",
    tx_power_w: float = 100.0,
):
    """Run all (method × difficulty × seed) cells, write CSV rows incrementally."""
    print(f"[pilot] {len(methods)}×{len(difficulties)}×{len(seeds)} = "
          f"{len(methods) * len(difficulties) * len(seeds)} cells")
    print(f"[pilot] writing to {out_csv}")

    total_cells = len(methods) * len(difficulties) * len(seeds)
    done_cells = 0
    t_pilot_start = time.perf_counter()

    for method in methods:
        for difficulty in difficulties:
            for seed in seeds:
                t_cell_start = time.perf_counter()
                print(f"\n[pilot {done_cells+1}/{total_cells}] "
                      f"method={method} difficulty={difficulty} seed={seed}")

                # Build env fresh per cell (avoids state leakage)
                env = build_env(num_envs=num_envs, device=device, tx_power_w=tx_power_w)
                qenv = build_qos_env(env)
                jammer = build_jammer(difficulty)
                scheduler = build_scheduler(env, team=0)
                composer = build_composer(method.split("_")[-1]) if method.startswith("concerto") else None
                rl_trainer = (build_rl_trainer(env, team=0, checkpoint_path=mppo_checkpoint)
                              if method != "classical" else None)

                drv = ConcertoPilotDriver(
                    qenv, method=method, difficulty=difficulty, seed=seed,
                    red_rl_trainer=rl_trainer,
                    classical_scheduler=scheduler,
                    composer=composer,
                    jammer=jammer,
                    max_steps=max_steps,
                    n_eval_episodes=n_eval_episodes,
                )
                try:
                    metrics = drv.run()
                    row = write_csv_row(out_csv, metrics)
                    elapsed_cell = time.perf_counter() - t_cell_start
                    done_cells += 1
                    print(f"  qos_agg={row['qos_satisfaction_mean']} "
                          f"per_fn=[d={row['qos_detect']}, t={row['qos_track']}, "
                          f"c={row['qos_comm']}, j={row['qos_jam']}] "
                          f"rl_frac={row['rl_frac']} "
                          f"wall={elapsed_cell:.1f}s "
                          f"({done_cells}/{total_cells} done)")
                except Exception as e:
                    sys.stderr.write(f"[ERROR] cell failed: {e}\n")
                    import traceback; traceback.print_exc()
                finally:
                    env.destroy()
                    del env
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    pilot_elapsed = time.perf_counter() - t_pilot_start
    print(f"\n[pilot] DONE. {done_cells}/{total_cells} cells in {pilot_elapsed:.1f}s")
    print(f"[pilot] results: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["classical", "mappo", "concerto_v1", "concerto_v2"])
    ap.add_argument("--difficulties", nargs="+", default=["L0", "L1", "L3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--mppo-checkpoint",
                    default="checkpoints/laser_mappo/iter_019.pt")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--n-eval-episodes", type=int, default=10)
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tx-power-w", type=float, default=100.0,
                    help="TX power override (default 100W for detectability)")
    ap.add_argument("--out", default="experiments/concerto_pilot_results.csv")
    args = ap.parse_args()

    run_pilot(
        methods=args.methods,
        difficulties=args.difficulties,
        seeds=args.seeds,
        mppo_checkpoint=args.mppo_checkpoint,
        max_steps=args.max_steps,
        n_eval_episodes=args.n_eval_episodes,
        out_csv=args.out,
        num_envs=args.num_envs,
        device=args.device,
        tx_power_w=args.tx_power_w,
    )


if __name__ == "__main__":
    main()
