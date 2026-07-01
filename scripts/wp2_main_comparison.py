"""WP2 main comparison harness — round-robin win-rate matrix.

EAAI Q1 requires comparison against external baselines (PAPER_PLAN_EAAI.md §WP2):
  {IPPO, QMIX, MAPPO, Classical controller, PSRO-lite, FluxLeague-full} × 5 seeds

This harness takes any list of "entries" (method_name + checkpoint_path or rule)
and produces the N×N win-rate matrix + per-method aggregate metrics. Designed
to be checkpoint-agnostic: once each baseline has a trained checkpoint, drop
them in and run.

Usage:
    # Compare FluxLeague-full vs Classical MPC (the only two available now)
    python scripts/wp2_main_comparison.py \\
        --entries fluxleague=ckpt:checkpoints/wp1_gate_seed42/main_team0_gen1.pt \\
                  fluxleague_blue=ckpt:checkpoints/wp1_gate_seed42/main_team1_gen1.pt \\
        --entries mpc=rule:classical_mpc \\
        --n-eval-games 50 --max-steps 500

    # Future: add IPPO/MAPPO/QMIX once trained
    python scripts/wp2_main_comparison.py \\
        --entries fluxleague=ckpt:checkpoints/wp1_gate_seed42/main_team0_gen1.pt \\
                  ippo=ckpt:checkpoints/ippo_seed42/best.pt \\
                  mappo=ckpt:checkpoints/mappo_seed42/best.pt \\
                  mpc=rule:classical_mpc \\
        --n-eval-games 50

Entry format:
    name=ckpt:<path>     → load TeamPPOTrainer from checkpoint path
    name=ckpt:<dir>      → auto-find latest main_team0_gen*.pt in dir (red side)
                           + main_team1_gen*.pt for blue side
    name=rule:classical_mpc → instantiate ClassicalMPC

Outputs:
    logs/wp2_main_comparison.json     structured win-rate matrix + metrics
    logs/wp2_main_comparison.log      appended human-readable table
    figures/wp2_winrate_heatmap.pdf   N×N heatmap (Fig WP2-main)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from env.gpu.vec_mfar_env import MFARVecEnv
from algo._shared.laser.episode import LaserEpisodeRunner
from algo._shared.laser.sensing import enforce_radar_baseline
from algo._shared.train import compute_env_params
from scripts.wp3_robustness_eval import (
    find_latest_checkpoint,
    load_trained_policy,
    make_env,
    resolve_kill_radius,
)


# ---------------------------------------------------------------------------
# Entry registry
# ---------------------------------------------------------------------------

class PolicyEntry:
    """One comparable method. Holds either a TeamPPOTrainer or a rule-based
    controller, plus metadata for reporting.
    """

    def __init__(self, name: str, kind: str, spec: str):
        self.name = name
        self.kind = kind          # "ckpt" or "rule"
        self.spec = spec          # path or rule name
        self.red_trainer = None
        self.blue_trainer = None

    def load(self, config: dict, env_params: dict, device: str):
        """Instantiate red/blue trainers for this entry."""
        if self.kind == "ckpt":
            if Path(self.spec).is_dir():
                ckpt_red = find_latest_checkpoint(Path(self.spec), role="main", team=0)
                ckpt_blue = find_latest_checkpoint(Path(self.spec), role="main", team=1)
                if ckpt_red is None or ckpt_blue is None:
                    raise FileNotFoundError(
                        f"No main_team*_gen*.pt in {self.spec}")
            else:
                ckpt_red = Path(self.spec)
                # If blue is a separate file, expect a convention; else reuse red.
                # For the wp1_gate layout, dir-based lookup is preferred.
                ckpt_blue = ckpt_red

            self.red_trainer = load_trained_policy(
                ckpt_red, config, env_params, team=0, device=device)
            self.blue_trainer = load_trained_policy(
                ckpt_blue, config, env_params, team=1, device=device)
        elif self.kind == "rule":
            self._load_rule(config, env_params, device)
        else:
            raise ValueError(f"Unknown entry kind: {self.kind}")

    def _load_rule(self, config: dict, env_params: dict, device: str):
        """Instantiate rule-based policy."""
        if self.spec == "classical_mpc":
            from algo._shared.baselines.classical_mpc import ClassicalMPC
            env_cfg = config.get("env", {})
            min_baseline = float(env_cfg.get("min_radar_baseline_m", 5000.0))
            range_sigma = float(config.get("sensing_noise", {})
                                .get("range_sigma_m", 0.05))
            crossrange_factor = float(config.get("sensing_noise", {})
                                      .get("crossrange_factor", 7.4e-5))
            track_q = float(config.get("sensing_noise", {}).get("track_q_m", 0.02))
            track_burnin = int(config.get("sensing_noise", {}).get("track_burnin", 120))
            half_map = float(env_cfg.get("map_size", [20000.0, 20000.0])[0]) / 2.0

            # Placeholder env just for init; we re-init at each eval batch
            tmp_env = make_env(config)
            self.red_trainer = ClassicalMPC(
                tmp_env, team=0, min_radar_baseline_m=min_baseline,
                range_sigma_m=range_sigma, crossrange_factor=crossrange_factor,
                track_q_m=track_q, track_burnin=track_burnin, half_map_m=half_map,
            )
            self.blue_trainer = ClassicalMPC(
                tmp_env, team=1, min_radar_baseline_m=min_baseline,
                range_sigma_m=range_sigma, crossrange_factor=crossrange_factor,
                track_q_m=track_q, track_burnin=track_burnin, half_map_m=half_map,
            )
            # NOTE: red/blue hold a reference to tmp_env; each evaluate_pair
            # call creates a fresh env, but the controller's env field is only
            # used for shape info (n_radars, n_elem, device), so this is safe.
        else:
            raise ValueError(f"Unknown rule: {self.spec}")

    def release(self):
        """Free GPU memory between pair evals (important for OOM avoidance)."""
        self.red_trainer = None
        self.blue_trainer = None
        torch.cuda.empty_cache()


def parse_entry(spec: str) -> PolicyEntry:
    """Parse 'name=kind:spec' into a PolicyEntry."""
    if "=" not in spec:
        raise ValueError(f"Entry must be name=kind:spec, got: {spec}")
    name, rest = spec.split("=", 1)
    if ":" not in rest:
        raise ValueError(f"Entry kind:spec missing ':', got: {rest}")
    kind, value = rest.split(":", 1)
    kind = kind.strip()
    if kind not in ("ckpt", "rule"):
        raise ValueError(f"Unknown kind '{kind}' (must be ckpt or rule)")
    return PolicyEntry(name=name.strip(), kind=kind, spec=value.strip())


# ---------------------------------------------------------------------------
# Pair evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(
    config: dict,
    red: PolicyEntry,
    blue: PolicyEntry,
    n_eval_games: int,
    max_steps: int,
) -> dict:
    """Run N games of red vs blue. Returns metrics dict."""
    env = make_env(config)
    pulses_per_control = config["env"].get("pulses_per_control", 5)
    runner = LaserEpisodeRunner(env, pulses_per_control=pulses_per_control,
                                device=config["env"].get("device", "cuda"))

    min_baseline = float(config["env"].get("min_radar_baseline_m", 5000.0))

    red_wins = 0.0
    blue_wins = 0.0
    draws = 0.0
    n_red_kills = 0
    n_blue_kills = 0
    total_progress = 0.0
    n_decisive = 0

    E = env.num_envs
    games_per_batch = E
    n_batches = max(1, math.ceil(n_eval_games / games_per_batch))

    for batch in range(n_batches):
        runner.reset(red_trainer=red.red_trainer, blue_trainer=blue.blue_trainer)
        if min_baseline > 0:
            enforce_radar_baseline(env, min_baseline)

        last_result = None
        for step in range(max_steps):
            out = runner.step_control(red.red_trainer, blue.blue_trainer,
                                       deterministic=True)
            last_result = out["result"]
            if last_result is None:
                break
            if last_result["dones"].any():
                break

        if last_result is None:
            draws += games_per_batch
            continue

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
            else:
                draws += 1.0

        if "kills" in last_result:
            kills = last_result["kills"]
            n_red_kills += int(kills[:, 0, :].any(dim=-1).sum().item())
            n_blue_kills += int(kills[:, 1, :].any(dim=-1).sum().item())

        progress = env.battlefield.laser.get_illumination_progress()
        total_progress += float(progress.sum().item())

    total = max(1, red_wins + blue_wins + draws)
    env.destroy()
    del env
    torch.cuda.empty_cache()

    return {
        "red_win_rate": red_wins / total,
        "blue_win_rate": blue_wins / total,
        "draw_rate": draws / total,
        "kill_rate": n_decisive / total,
        "red_kill_count": n_red_kills,
        "blue_kill_count": n_blue_kills,
        "mean_illumination_progress": total_progress / total,
        "n_games": int(total),
    }


# ---------------------------------------------------------------------------
# Round-robin main
# ---------------------------------------------------------------------------

def plot_winrate_heatmap(out_path: Path, matrix: np.ndarray, names: List[str]):
    """N×N heatmap of red_win_rate (red's perspective)."""
    n = len(names)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.0, vmax=1.0, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_yticklabels(names)
    ax.set_xlabel('Blue (opponent)')
    ax.set_ylabel('Red (perspective)')
    ax.set_title('WP2 Main Comparison — Win Rate Matrix\n'
                 '(cell[i,j] = P(red=i beats blue=j))')

    for i in range(n):
        for j in range(n):
            if i == j:
                txt = "—"
            else:
                txt = f"{matrix[i, j]:.2f}"
            color = "white" if abs(matrix[i, j] - 0.5) > 0.3 else "black"
            ax.text(j, i, txt, ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    fig.colorbar(im, ax=ax, label='Red win rate')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", nargs="+", required=True,
                    help="List of name=kind:spec entries "
                         "(kind=ckpt|rule, spec=path|classical_mpc)")
    ap.add_argument("--config", type=Path, default=Path("configs/wp1_gate.yaml"))
    ap.add_argument("--n-eval-games", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--include-self-play", action="store_true",
                    help="Also eval each entry vs itself (diagonal cells). "
                         "Off by default since self-play win-rate is meaningless.")
    ap.add_argument("--output-json", type=Path,
                    default=Path("logs/wp2_main_comparison.json"))
    ap.add_argument("--output-log", type=Path,
                    default=Path("logs/wp2_main_comparison.log"))
    ap.add_argument("--output-fig", type=Path,
                    default=Path("figures/wp2_winrate_heatmap.pdf"))
    ap.add_argument("--kill-radius-override", type=float, default=None,
                    help="Test kr override (m). If unset, auto-detect from training log.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.device:
        config["env"]["device"] = args.device
    if args.num_envs is not None:
        config["env"]["num_envs"] = args.num_envs

    # Resolve kr from training log (or override)
    # WP2 entries may include both ckpt-based and rule-based; the kr applies
    # to the env, so it's a global setting here.
    ckpt_dir_for_kr = args.entries[0].split("=ckpt:", 1)[-1] if "ckpt:" in args.entries[0] else None
    if ckpt_dir_for_kr:
        resolved_kr = resolve_kill_radius(
            config, Path(ckpt_dir_for_kr), args.kill_radius_override,
        )
    else:
        resolved_kr = config["env"].get("kill_radius_m", 0.2)
    config["env"]["kill_radius_m"] = resolved_kr

    env_params = compute_env_params(config)
    device = config["env"].get("device", "cuda")

    entries: List[PolicyEntry] = [parse_entry(s) for s in args.entries]
    n = len(entries)
    if n < 2:
        print("ERROR: need at least 2 entries for comparison", file=sys.stderr)
        sys.exit(2)

    print("=" * 72)
    print("WP2 Main Comparison — Round-Robin Win-Rate Matrix")
    print("=" * 72)
    print(f"Entries ({n}):")
    for e in entries:
        print(f"  - {e.name:<24} ({e.kind}:{e.spec})")
    print(f"\nGames per pair : {args.n_eval_games}")
    print(f"Max steps      : {args.max_steps}")
    print(f"Num envs       : {args.num_envs}")
    print()

    # Load all entries upfront (fail-fast)
    for e in entries:
        print(f"[load] {e.name} ...")
        e.load(config, env_params, device)
    print()

    # Win-rate matrix [red, blue]
    matrix = np.full((n, n), 0.5)
    pair_metrics: Dict[str, dict] = {}

    for i, red_entry in enumerate(entries):
        for j, blue_entry in enumerate(entries):
            if i == j and not args.include_self_play:
                continue
            print(f"[eval] {red_entry.name} (red) vs {blue_entry.name} (blue) ...")
            t0 = time.time()
            metrics = evaluate_pair(
                config, red_entry, blue_entry,
                args.n_eval_games, args.max_steps,
            )
            metrics["elapsed_s"] = time.time() - t0
            matrix[i, j] = metrics["red_win_rate"]
            pair_metrics[f"{red_entry.name}__vs__{blue_entry.name}"] = metrics
            print(f"  red_wr={metrics['red_win_rate']:.3f}  "
                  f"blue_wr={metrics['blue_win_rate']:.3f}  "
                  f"draw={metrics['draw_rate']:.3f}  "
                  f"kr={metrics['kill_rate']:.3f}  "
                  f"({metrics['elapsed_s']:.1f}s)")

    # Aggregate per-entry metrics (as red, vs avg opponent)
    per_entry: Dict[str, dict] = {}
    for i, e in enumerate(entries):
        opp_wrs = [matrix[i, j] for j in range(n) if j != i]
        if opp_wrs:
            avg_red_wr = float(np.mean(opp_wrs))
        else:
            avg_red_wr = 0.5
        # Mean kill rate when this entry is red
        kill_rates = []
        for j, e2 in enumerate(entries):
            if j == i:
                continue
            key = f"{e.name}__vs__{e2.name}"
            if key in pair_metrics:
                kill_rates.append(pair_metrics[key]["kill_rate"])
        per_entry[e.name] = {
            "avg_red_win_rate": avg_red_wr,
            "mean_kill_rate_as_red": float(np.mean(kill_rates)) if kill_rates else 0.0,
        }

    # Release GPU memory
    for e in entries:
        e.release()

    # Report
    print()
    print("=" * 72)
    print("Win-Rate Matrix (red row vs blue col)")
    print("=" * 72)
    header = "  " + " ".join(f"{n:>10}" for n in [e.name for e in entries])
    print(header)
    for i, e in enumerate(entries):
        row = "  " + f"{e.name:<10}"
        for j, e2 in enumerate(entries):
            if i == j and not args.include_self_play:
                row += f"{'—':>10}"
            else:
                row += f"{matrix[i, j]:>10.3f}"
        print(row)
    print()

    print("Per-entry aggregates:")
    for name, m in per_entry.items():
        print(f"  {name:<24} avg_red_wr={m['avg_red_win_rate']:.3f}  "
              f"mean_kr={m['mean_kill_rate_as_red']:.3f}")

    # Save JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entries": [{"name": e.name, "kind": e.kind, "spec": e.spec}
                    for e in entries],
        "config": str(args.config),
        "n_eval_games": args.n_eval_games,
        "max_steps": args.max_steps,
        "matrix": matrix.tolist(),
        "entry_names": [e.name for e in entries],
        "pair_metrics": pair_metrics,
        "per_entry": per_entry,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[JSON] wrote {args.output_json}")

    # Plot
    if n >= 2:
        args.output_fig.parent.mkdir(parents=True, exist_ok=True)
        plot_winrate_heatmap(args.output_fig, matrix, [e.name for e in entries])
        print(f"[Fig ] wrote {args.output_fig}")

    # Append log
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_log, "a") as f:
        f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        for name, m in per_entry.items():
            f.write(f"  {name}: avg_red_wr={m['avg_red_win_rate']:.4f} "
                    f"mean_kr={m['mean_kill_rate_as_red']:.4f}\n")


if __name__ == "__main__":
    main()
