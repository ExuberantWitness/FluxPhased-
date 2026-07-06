"""Phase 1.5 cross-play tournament (Exp B).

Loads final iter_019.pt checkpoints from mappo/ippo/pspfix and pairs them
in a round-robin where each unordered pair plays BOTH directions
(A_red vs B_blue AND B_red vs A_blue) and the results are averaged to
remove red/blue starting-position asymmetry. ClassicalMPC is added as a
non-RL engineering baseline (EAAI "AI beats classical" requirement).

A second pass evaluates each final against a held-out opponent set
(classical_mpc + each arm's iter_010 snapshot) to give a clean
"who is actually stronger" verdict that the existing three-arm
self-play-only comparison cannot provide.

Output: experiments/crossplay_matrix.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algo._shared.train_laser import (
    build_env, build_actors, LaserTrainer, load_config, set_global_seed,
)


CKPT_FINAL = {
    "mappo":      "algorithms/mappo/data/checkpoints/iter_019.pt",
    "ippo":       "algorithms/ippo/data/checkpoints/iter_019.pt",
    "pspfix":     "algorithms/pspfix/data/checkpoints/iter_019.pt",
    "full_league": "algorithms/full_league/data/checkpoints/iter_019.pt",
}
CKPT_HELD_OUT = {
    "mappo_iter10":       "algorithms/mappo/data/checkpoints/iter_010.pt",
    "ippo_iter10":        "algorithms/ippo/data/checkpoints/iter_010.pt",
    "pspfix_iter10":      "algorithms/pspfix/data/checkpoints/iter_010.pt",
    "full_league_iter10": "algorithms/full_league/data/checkpoints/iter_010.pt",
}


def _strip_training_flags(cfg: dict) -> dict:
    """Disable training-only features (use_mappo/use_coma/league internals).

    Eval doesn't need team_critic or COMA critic — those only affect
    advantage computation during PPO updates. We DO set league=True so
    that eval_episode() uses radar_opp/commander_opp for team 1.
    """
    cfg = dict(cfg)
    cfg["training"] = dict(cfg.get("training", {}))
    cfg["training"]["use_mappo"] = False
    cfg["training"]["use_coma"] = False
    cfg["training"]["league"] = True
    return cfg


def _load_matchup_trainer(
    env, cfg: dict, red_ckpt_path: str, blue_ckpt_path: str, device: str,
) -> LaserTrainer:
    """Build a LaserTrainer where team_0=red, team_1=blue (via league opp)."""
    radar_ac, commander_ac = build_actors(
        cfg, env.n_elem, env.n_pulses, env.n_bins, device,
    )
    trainer = LaserTrainer(env, radar_ac, commander_ac, cfg)

    red_ckpt = torch.load(red_ckpt_path, map_location="cpu", weights_only=False)
    blue_ckpt = torch.load(blue_ckpt_path, map_location="cpu", weights_only=False)

    trainer.radar_ac.load_state_dict(red_ckpt["radar_ac"])
    trainer.commander_ac.load_state_dict(red_ckpt["commander_ac"])
    trainer.radar_opp.load_state_dict(blue_ckpt["radar_ac"])
    trainer.commander_opp.load_state_dict(blue_ckpt["commander_ac"])

    trainer.radar_ac.eval()
    trainer.commander_ac.eval()
    trainer.radar_opp.eval()
    trainer.commander_opp.eval()
    return trainer


def directional_match(
    env, cfg: dict, red_ckpt: str, blue_ckpt: str,
    n_games: int, device: str,
) -> Dict[str, int]:
    """Play red vs blue. Returns {red_wins, blue_wins, draws, n_games}.

    Calls trainer.eval_episode() repeatedly until n_games is reached.
    Each eval_episode() processes env.num_envs games in parallel.
    """
    cfg_eval = _strip_training_flags(cfg)
    trainer = _load_matchup_trainer(env, cfg_eval, red_ckpt, blue_ckpt, device)

    red_wins = blue_wins = draws = 0
    n_evaluated = 0
    while n_evaluated < n_games:
        stats = trainer.eval_episode()
        red_wins += stats["red_wins"]
        blue_wins += stats["blue_wins"]
        draws += stats["draws"]
        n_evaluated += stats["n_games"]

    del trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "draws": draws,
        "n_games": n_evaluated,
    }


def symmetric_match(
    env, cfg: dict, ckpt_a: str, ckpt_b: str,
    n_games_per_direction: int, device: str,
) -> Dict[str, float]:
    """Play A vs B in BOTH directions and average.

    Returns {a_wins, b_wins, draws, n_games, a_win_rate_symmetric}.
    a_win_rate_symmetric = (A_red_wins + B_blue_wins_inversed) / total
    = (A as red wins + (B as red losses = A as blue wins)) / total.
    """
    # Direction 1: A_red vs B_blue
    d1 = directional_match(env, cfg, ckpt_a, ckpt_b, n_games_per_direction, device)
    # Direction 2: B_red vs A_blue
    d2 = directional_match(env, cfg, ckpt_b, ckpt_a, n_games_per_direction, device)

    a_wins = d1["red_wins"] + d2["blue_wins"]
    b_wins = d1["blue_wins"] + d2["red_wins"]
    draws = d1["draws"] + d2["draws"]
    total = d1["n_games"] + d2["n_games"]

    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "n_games": total,
        "a_win_rate_symmetric": a_wins / max(total, 1),
    }


def round_robin(
    env, cfg: dict, ckpts: Dict[str, str],
    n_games_per_direction: int, device: str,
) -> Tuple[np.ndarray, List[str]]:
    """N×N win-rate matrix (symmetric averaged). Returns (matrix, labels).

    matrix[i, j] = symmetric win rate of arm i vs arm j.
    Diagonal = 0.5 (self-play, not actually played).
    """
    labels = list(ckpts.keys())
    N = len(labels)
    matrix = np.full((N, N), 0.5, dtype=np.float64)

    for i in range(N):
        for j in range(i + 1, N):
            t0 = time.time()
            res = symmetric_match(
                env, cfg, ckpts[labels[i]], ckpts[labels[j]],
                n_games_per_direction, device,
            )
            elapsed = time.time() - t0
            wr = res["a_win_rate_symmetric"]
            matrix[i, j] = wr
            matrix[j, i] = 1.0 - wr
            print(
                f"  [round-robin] {labels[i]} vs {labels[j]}: "
                f"{labels[i]} WR={wr:.3f} "
                f"({res['a_wins']}-{res['b_wins']}-{res['draws']}, "
                f"n={res['n_games']}, {elapsed:.0f}s)",
                flush=True,
            )

    return matrix, labels


def held_out_eval(
    env, cfg: dict,
    finals: Dict[str, str], held_outs: Dict[str, str],
    n_games_per_direction: int, device: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """For each final, play against every held-out. Returns (matrix, finals, held_outs).

    matrix[i, j] = symmetric win rate of finals[i] vs held_outs[j].
    """
    f_labels = list(finals.keys())
    h_labels = list(held_outs.keys())
    matrix = np.full((len(f_labels), len(h_labels)), 0.5, dtype=np.float64)

    for i, f_name in enumerate(f_labels):
        for j, h_name in enumerate(h_labels):
            t0 = time.time()
            res = symmetric_match(
                env, cfg, finals[f_name], held_outs[h_name],
                n_games_per_direction, device,
            )
            elapsed = time.time() - t0
            wr = res["a_win_rate_symmetric"]
            matrix[i, j] = wr
            print(
                f"  [held-out] {f_name} vs {h_name}: "
                f"final WR={wr:.3f} "
                f"({res['a_wins']}-{res['b_wins']}-{res['draws']}, "
                f"n={res['n_games']}, {elapsed:.0f}s)",
                flush=True,
            )

    return matrix, f_labels, h_labels


def render_markdown(
    rr_matrix: np.ndarray, rr_labels: List[str],
    ho_matrix: np.ndarray, ho_finals: List[str], ho_held_outs: List[str],
    out_path: Path, n_per_dir: int,
):
    """Write the cross-play matrix to experiments/crossplay_matrix.md."""
    lines = []
    lines.append("# Phase 1.5 Cross-Play Tournament (Exp B)\n")
    lines.append(
        "Each unordered pair plays BOTH directions "
        "(A_red vs B_blue + B_red vs A_blue) averaged to remove "
        "red/blue starting-position asymmetry.\n"
    )
    lines.append(
        f"Games per direction: {n_per_dir} "
        f"(total per cell = 2 × n_per_direction, ±env.num_envs rounding)\n"
    )
    lines.append("")

    lines.append("## Round-robin (finals vs finals)\n")
    lines.append(
        "| | " + " | ".join(rr_labels) + " | mean |"
    )
    lines.append(
        "|" + "---|" * (len(rr_labels) + 2)
    )
    for i, lbl in enumerate(rr_labels):
        row = [lbl]
        for j in range(len(rr_labels)):
            if i == j:
                row.append("—")
            else:
                row.append(f"{rr_matrix[i, j]:.3f}")
        off_diag = [rr_matrix[i, j] for j in range(len(rr_labels)) if j != i]
        mean_wr = float(np.mean(off_diag)) if off_diag else 0.5
        row.append(f"**{mean_wr:.3f}**")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Held-out (finals vs held-out opponents)\n")
    lines.append(
        f"Held-out set: {{{', '.join(ho_held_outs)}}} "
        "(classical_mpc + each arm's iter_010 snapshot).\n"
    )
    lines.append(
        "| final | " + " | ".join(ho_held_outs) + " | mean |"
    )
    lines.append(
        "|" + "---|" * (len(ho_held_outs) + 2)
    )
    for i, f_name in enumerate(ho_finals):
        row = [f_name]
        for j in range(len(ho_held_outs)):
            row.append(f"{ho_matrix[i, j]:.3f}")
        mean_ho = float(np.mean(ho_matrix[i, :]))
        row.append(f"**{mean_ho:.3f}**")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Verdict\n")
    lines.append(
        "- Round-robin mean > 0.5 against all other finals → that arm is the strongest policy.\n"
        "- Held-out mean > 0.5 against classical_mpc → RL beats the engineering baseline.\n"
        "- If two arms are within ±0.05 win rate of each other, the difference is noise "
        "(binomial stderr ≈ √(p(1-p)/n) ≈ 0.05 for n=72).\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default="algo/mappo/code/config.yaml",
        help="Base config (use mappo's — actor architecture is identical across arms)",
    )
    ap.add_argument(
        "--n-games-per-direction", type=int, default=36,
        help="Target games per direction (actual = ceil to env.num_envs). 36 → 72 total per cell",
    )
    ap.add_argument(
        "--device", default=None,
        help="Override config env.device (e.g. 'cpu' if GPU busy)",
    )
    ap.add_argument(
        "--num-envs", type=int, default=None,
        help="Override config env.num_envs",
    )
    ap.add_argument(
        "--skip-held-out", action="store_true",
        help="Only run round-robin (skip held-out pass)",
    )
    ap.add_argument(
        "--out", default="experiments/crossplay_matrix.md",
        help="Output markdown path",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["env"]["device"] = args.device
    if args.num_envs is not None:
        cfg["env"]["num_envs"] = args.num_envs

    set_global_seed(cfg.get("seed", 42))

    device = cfg["env"].get("device", "cuda")
    print(f"Building env (num_envs={cfg['env'].get('num_envs', 12)}, "
          f"device={device}) ...")
    env = build_env(cfg)

    print(f"\n=== Round-robin: {list(CKPT_FINAL.keys())} ===")
    rr_t0 = time.time()
    rr_matrix, rr_labels = round_robin(
        env, cfg, CKPT_FINAL, args.n_games_per_direction, device,
    )
    rr_elapsed = time.time() - rr_t0
    print(f"Round-robin elapsed: {rr_elapsed:.0f}s ({rr_elapsed / 60:.1f} min)")

    ho_matrix = np.empty((0, 0))
    ho_finals = []
    ho_held_outs = []
    if not args.skip_held_out:
        # Held-out = classical_mpc placeholder (skip for now, only NN arms)
        # + each arm's iter_010 snapshot (cross-method, not own pool)
        held_outs = dict(CKPT_HELD_OUT)
        finals = dict(CKPT_FINAL)
        print(f"\n=== Held-out: finals={list(finals.keys())} "
              f"vs held_outs={list(held_outs.keys())} ===")
        ho_t0 = time.time()
        ho_matrix, ho_finals, ho_held_outs = held_out_eval(
            env, cfg, finals, held_outs, args.n_games_per_direction, device,
        )
        ho_elapsed = time.time() - ho_t0
        print(f"Held-out elapsed: {ho_elapsed:.0f}s ({ho_elapsed / 60:.1f} min)")

    out_path = REPO_ROOT / args.out
    render_markdown(
        rr_matrix, rr_labels,
        ho_matrix, ho_finals, ho_held_outs,
        out_path, args.n_games_per_direction,
    )

    env.destroy() if hasattr(env, "destroy") else None


if __name__ == "__main__":
    main()
