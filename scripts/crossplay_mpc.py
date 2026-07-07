"""Phase 1.5 cross-play vs ClassicalMPC (Exp B supplement).

Plays each NN final (mappo/ippo/pspfix iter_019) against ClassicalMPC
in BOTH directions (NN_red vs MPC_blue + MPC_red vs NN_blue), averaged.
This is the EAAI "AI beats classical" requirement: a non-RL engineering
baseline that shares the same env + sensing frontend.

Uses LaserEpisodeRunner + a thin NNPolicyAdapter (mirrors ClassicalMPC's
structure but queries the trained actor-critics). NN and MPC each play
team 0 in one direction; results averaged to remove red/blue asymmetry.

Output: experiments/crossplay_mpc.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algo._shared.train_laser import (
    build_env, build_actors, load_config, set_global_seed,
)
from algo._shared.laser.episode import LaserEpisodeRunner
from algo._shared.laser.sensing import (
    enforce_radar_baseline, fused_sensing, KalmanTracker,
)
from algo._shared.baselines.classical_mpc import ClassicalMPC


CKPT_FINAL = {
    "mappo":      "algorithms/mappo/data/checkpoints/iter_019.pt",
    "ippo":       "algorithms/ippo/data/checkpoints/iter_019.pt",
    "pspfix":     "algorithms/pspfix/data/checkpoints/iter_019.pt",
    "full_league": "algorithms/full_league/data/checkpoints/iter_019.pt",
    "ew_mappo":   "algo/ew_mappo/data/checkpoints/iter_019.pt",
    "ew_ippo":    "algo/ew_ippo/data/checkpoints/iter_019.pt",
}


class NNPolicyAdapter:
    """Wrap (radar_ac, commander_ac) to play any team in LaserEpisodeRunner.

    Mirrors ClassicalMPC's contract: get_own_actions(env, team, ...) returns
    {r_start, r_end, radar_actions[E, R_team, action_dim], commander_action[E, 5]}.

    Per-team observations are built fresh each call (mirrors LaserTrainer's
    _build_radar_obs / _build_commander_obs). Residual aim is applied to the
    commander action so the env receives absolute aim, not raw residual.
    """

    def __init__(
        self, radar_ac, commander_ac, env, team: int,
        residual_aim: bool, residual_scale_m: float,
        sensing_mode: str, range_sigma_m: float,
        crossrange_factor: float, track_q_m: float, track_burnin: int,
        min_radar_baseline_m: float,
        jam_gain: float = 0.0, exposure_gain: float = 0.0,
        hybrid_fire: bool = True,
    ):
        self.radar_ac = radar_ac
        self.commander_ac = commander_ac
        self.env = env
        self.team = int(team)
        self.residual_aim = bool(residual_aim)
        self.residual_scale_m = float(residual_scale_m)
        self.sensing_mode = sensing_mode
        self.range_sigma_m = float(range_sigma_m)
        self.crossrange_factor = float(crossrange_factor)
        self.min_radar_baseline_m = float(min_radar_baseline_m)
        self.jam_gain = float(jam_gain)
        self.exposure_gain = float(exposure_gain)
        self.hybrid_fire = bool(hybrid_fire)
        self.jam_level = None  # set externally via set_jam_level

        E = env.num_envs
        R_team = env.n_radars // env.n_teams
        self.r_start = self.team * R_team
        self.r_end = (self.team + 1) * R_team
        self.R_team = R_team

        # Kalman tracker for fused sensing (one per adapter — fresh per episode)
        self.kalman_tracker = KalmanTracker(
            track_q_m=track_q_m, track_burnin=track_burnin,
        )
        self.kalman_tracker.ensure_alloc(E, env.n_teams, torch.device(env.device))
        self.kalman_tracker._initialized = True

    def reset_episode(self, E: int, n_teams: int):
        self.kalman_tracker.reset()
        self.kalman_tracker.ensure_alloc(E, n_teams, torch.device(self.env.device))
        self.kalman_tracker._initialized = True

    def get_own_actions(
        self, env, team: int = None, deterministic: bool = True,
        spectrum: torch.Tensor = None, events: dict = None,
    ) -> Dict[str, torch.Tensor]:
        if team is None:
            team = self.team

        dev = torch.device(env.device)
        E = env.num_envs
        R_team = self.R_team
        N = env.n_elem
        ACTION_PER_ELEM = 22

        # Build commander obs from env, then apply fused sensing
        radar_latents = torch.zeros(E, env.n_radars, 32, device=dev)
        cmd_obs_all = env.battlefield.get_commander_observation(
            env.radar_pos, radar_latents,
        )  # [E, n_teams, 76]        # fused_sensing writes Kalman-tracked enemy xy into cmd_obs[..., 68:72]
        # in place. Operates on full [E, n_teams, 76] for both teams at once.
        if self.sensing_mode in ("fused", "tracked"):
            half_x = env.map_size[0] / 2.0
            half_y = env.map_size[1] / 2.0
            fused_sensing(
                cmd_obs_all,
                half_x=half_x, half_y=half_y,
                range_sigma_m=self.range_sigma_m,
                crossrange_factor=self.crossrange_factor,
                tracker=self.kalman_tracker,
                jam_gain=self.jam_gain,
                exposure_gain=self.exposure_gain,
                jam_level=self.jam_level,
            )

        team_cmd_obs = cmd_obs_all[:, team, :]  # [E, 76]

        # Build radar obs (mirrors LaserEpisodeRunner._build_radar_obs layout)
        if spectrum is not None:
            spec_flat = spectrum.reshape(E, env.n_radars, -1)
        else:
            spec_flat = torch.zeros(E, env.n_radars, N * 4 * 64, device=dev)
        comm_flat = torch.zeros(E, env.n_radars, N * 2, device=dev)
        recon_flat = torch.zeros(E, env.n_radars, N * 4, device=dev)
        vehicle = torch.zeros(E, env.n_radars, 5, device=dev)
        laser_state = torch.zeros(E, env.n_radars, 12, device=dev)
        cmd_instr = torch.zeros(E, env.n_radars, 16, device=dev)
        if events is not None and "radar_pos" in events:
            vehicle[:, :, 0] = events["radar_pos"][:, :, 0]
            vehicle[:, :, 1] = events["radar_pos"][:, :, 1]
        radar_obs_all = torch.cat(
            [spec_flat, comm_flat, recon_flat, vehicle, laser_state, cmd_instr],
            dim=-1,
        )
        team_radar_obs = radar_obs_all[:, self.r_start:self.r_end, :]  # [E, R_team, obs_dim]

        with torch.no_grad():
            r_flat = team_radar_obs.reshape(E * R_team, -1)
            r_action, _, _, _ = self.radar_ac.get_action(
                r_flat, deterministic=deterministic,
            )
            r_action = r_action.reshape(E, R_team, -1)
            c_action, _, _, _ = self.commander_ac.get_action(
                team_cmd_obs, deterministic=deterministic,
            )  # [E, 5]

        # LaserEpisodeRunner expects radar_actions as a LIST of [E, action_dim]
        # tensors (one per radar in the team), per TeamPPOTrainer.get_own_actions.
        radar_actions_list = [r_action[:, i, :] for i in range(R_team)]

        if self.residual_aim:
            half_x = env.map_size[0] / 2.0
            half_y = env.map_size[1] / 2.0
            anchor = team_cmd_obs[:, 68:70]  # [E, 2] normalized enemy pos
            c_env = c_action.clone()
            c_env[:, 1] = anchor[:, 0] + c_action[:, 1] * (self.residual_scale_m / half_x)
            c_env[:, 2] = anchor[:, 1] + c_action[:, 2] * (self.residual_scale_m / half_y)
            c_env[:, 3] = c_action[:, 3] * (self.residual_scale_m / 1000.0)
            c_action = c_env

        # Update shared jam_level AFTER policy query: commander_action[4] = jam logit.
        # Map [-1,1] → [0,1] (matches train_laser.py:1276 + 1372).
        if self.jam_level is not None:
            self.jam_level[:, team] = ((c_action[:, 4] + 1.0) * 0.5).clamp(0.0, 1.0).detach()

        return {
            "r_start": self.r_start,
            "r_end": self.r_end,
            "radar_actions": radar_actions_list,
            "commander_action": c_action,
            "transition": None,
        }


def make_nn_adapter(env, cfg, ckpt_path, team, device):
    radar_ac, commander_ac = build_actors(
        cfg, env.n_elem, env.n_pulses, env.n_bins, device,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    radar_ac.load_state_dict(ckpt["radar_ac"])
    commander_ac.load_state_dict(ckpt["commander_ac"])
    radar_ac.eval(); commander_ac.eval()

    scfg = cfg.get("sensing_noise", {})
    tcfg = cfg.get("training", {})
    rcfg = cfg.get("reward_shaping", {})
    return NNPolicyAdapter(
        radar_ac, commander_ac, env, team=team,
        residual_aim=tcfg.get("residual_aim", True),
        residual_scale_m=tcfg.get("residual_scale_m", 6.0),
        sensing_mode=scfg.get("mode", "tracked"),
        range_sigma_m=scfg.get("range_sigma_m", 0.05),
        crossrange_factor=scfg.get("crossrange_factor", 7.4e-5),
        track_q_m=scfg.get("track_q_m", 0.02),
        track_burnin=scfg.get("track_burnin", 120),
        min_radar_baseline_m=cfg.get("env", {}).get("min_radar_baseline_m", 5000.0),
        jam_gain=rcfg.get("jam_gain", 0.0),
        exposure_gain=rcfg.get("exposure_gain", 0.0),
        hybrid_fire=tcfg.get("hybrid_fire", True),
    )


def make_mpc(env, cfg, team):
    scfg = cfg.get("sensing_noise", {})
    rcfg = cfg.get("reward_shaping", {})
    half_map = float(cfg["env"].get("map_size", [20000.0, 20000.0])[0]) / 2.0
    return ClassicalMPC(
        env, team=team,
        min_radar_baseline_m=cfg["env"].get("min_radar_baseline_m", 5000.0),
        range_sigma_m=scfg.get("range_sigma_m", 0.05),
        crossrange_factor=scfg.get("crossrange_factor", 7.4e-5),
        track_q_m=scfg.get("track_q_m", 0.02),
        track_burnin=scfg.get("track_burnin", 120),
        half_map_m=half_map,
        jam_gain=rcfg.get("jam_gain", 0.0),
        exposure_gain=rcfg.get("exposure_gain", 0.0),
    )


def directional_match(
    env, cfg, ckpt_path, nn_team, n_games, max_steps, device,
) -> Dict[str, int]:
    """Play one direction: NN as team `nn_team`, MPC as the other.

    Mirrors PayoffMatrix.evaluate_pair's loop: keeps stepping until ALL envs
    finish (or step cap), scoring each env as it ends. Don't break on the
    first done — that would abandon the other parallel envs mid-game.
    """
    other = 1 - nn_team
    nn = make_nn_adapter(env, cfg, ckpt_path, team=nn_team, device=device)
    mpc = make_mpc(env, cfg, team=other)

    # Shared jam_level tensor: both adapters read enemy_jam from this and
    # write their own team's jam back into it. MPC's slot stays 0 (no jam).
    # Semantics match train_laser.py:1276 — (c_action[:, 4]+1)*0.5 in [0,1].
    jam_level = torch.zeros(env.num_envs, env.n_teams, device=device)
    nn.jam_level = jam_level
    mpc.jam_level = jam_level

    if nn_team == 0:
        red, blue = nn, mpc
    else:
        red, blue = mpc, nn

    runner = LaserEpisodeRunner(
        env, pulses_per_control=cfg["env"].get("pulses_per_control", 5),
        device=device,
    )

    nn_wins = 0
    mpc_wins = 0
    draws = 0
    n_evaluated = 0
    while n_evaluated < n_games:
        runner.reset(red_trainer=red, blue_trainer=blue)
        nn.reset_episode(env.num_envs, env.n_teams)
        mpc.reset_episode(env.num_envs, env.n_teams)
        jam_level.zero_()  # cold-start: no jam until first action
        if nn.min_radar_baseline_m > 0:
            enforce_radar_baseline(env, nn.min_radar_baseline_m)

        live_envs = set(range(env.num_envs))
        last_progress = None
        for step in range(max_steps):
            if not live_envs:
                break
            out = runner.step_control(red, blue, deterministic=True)
            result = out["result"]
            if result is None:
                break
            # MPC never jams — keep its slot pinned to 0 even after step.
            jam_level[:, other].zero_()
            # Track illumination_progress for timeout tiebreaker
            last_progress = result.get("illumination_progress")
            if result["dones"].any():
                dones = result["dones"]
                winners = result["winners"]
                for e in sorted(live_envs):
                    if dones[e]:
                        live_envs.discard(e)
                        w = int(winners[e].item())
                        if w == nn_team:
                            nn_wins += 1
                        elif w == other:
                            mpc_wins += 1
                        else:
                            draws += 1

        # Score remaining live envs (step-cap draws; use progress tiebreaker)
        for e in sorted(live_envs):
            draws += 1  # default: timeout = draw
            # Tiebreaker: team closer to a kill (higher progress) wins
            if last_progress is not None and e < last_progress.shape[0]:
                p_self = float(last_progress[e, nn_team])
                p_other = float(last_progress[e, other])
                if p_self - p_other > 0.01:
                    # Replace the draw we just added with an NN win
                    draws -= 1
                    nn_wins += 1
                elif p_other - p_self > 0.01:
                    draws -= 1
                    mpc_wins += 1
        n_evaluated += env.num_envs

    return {
        "nn_wins": nn_wins,
        "mpc_wins": mpc_wins,
        "draws": draws,
        "n_games": n_evaluated,
    }


def symmetric_match(
    env, cfg, ckpt_path, n_games_per_direction, max_steps, device,
) -> Dict[str, float]:
    """NN vs MPC, both directions averaged."""
    # Direction 1: NN as team 0 (red)
    d1 = directional_match(env, cfg, ckpt_path, nn_team=0,
                           n_games=n_games_per_direction, max_steps=max_steps,
                           device=device)
    # Direction 2: NN as team 1 (blue)
    d2 = directional_match(env, cfg, ckpt_path, nn_team=1,
                           n_games=n_games_per_direction, max_steps=max_steps,
                           device=device)
    nn_wins = d1["nn_wins"] + d2["nn_wins"]
    mpc_wins = d1["mpc_wins"] + d2["mpc_wins"]
    draws = d1["draws"] + d2["draws"]
    total = d1["n_games"] + d2["n_games"]
    return {
        "nn_wins": nn_wins,
        "mpc_wins": mpc_wins,
        "draws": draws,
        "n_games": total,
        "nn_win_rate": nn_wins / max(total, 1),
    }


def render_markdown(rows, out_path: Path, n_per_dir: int):
    lines = []
    lines.append("# Phase 1.5 Cross-Play: NN finals vs ClassicalMPC (Exp B supplement)\n")
    lines.append(
        "Each NN final plays ClassicalMPC in BOTH directions "
        "(NN_red vs MPC_blue + MPC_red vs NN_blue) averaged to remove "
        "red/blue asymmetry. ClassicalMPC = rule-based beam-steer to fused "
        "enemy anchor + always-fire (no learning, no waveform agility).\n"
    )
    lines.append(
        f"Games per direction: {n_per_dir} (total per row ≈ 2 × n_per_direction)\n"
    )
    lines.append("")
    lines.append("| NN final | NN wins | MPC wins | draws | NN win rate |")
    lines.append("|---|---|---|---|---|")
    for name, res in rows.items():
        lines.append(
            f"| {name} | {res['nn_wins']} | {res['mpc_wins']} | "
            f"{res['draws']} | **{res['nn_win_rate']:.3f}** |"
        )
    lines.append("")
    lines.append("## Verdict\n")
    lines.append(
        "- NN win rate > 0.5 → RL beats the classical engineering baseline.\n"
        "- All three arms should beat ClassicalMPC; if any loses, the sensing "
        "frontend (not learning) is doing the work (EAAI: 'AI beats classical').\n"
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
        help="Target games per direction (actual = ceil to env.num_envs)",
    )
    ap.add_argument("--max-steps", type=int, default=500,
                    help="Max pulses per episode")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--out", default="experiments/crossplay_mpc.md")
    ap.add_argument(
        "--arms", nargs="+", default=list(CKPT_FINAL.keys()),
        choices=list(CKPT_FINAL.keys()),
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

    rows = {}
    for arm in args.arms:
        ckpt = CKPT_FINAL[arm]
        t0 = time.time()
        res = symmetric_match(
            env, cfg, ckpt, args.n_games_per_direction, args.max_steps, device,
        )
        elapsed = time.time() - t0
        rows[arm] = res
        print(
            f"  [vs-MPC] {arm}: NN WR={res['nn_win_rate']:.3f} "
            f"({res['nn_wins']}-{res['mpc_wins']}-{res['draws']}, "
            f"n={res['n_games']}, {elapsed:.0f}s)",
            flush=True,
        )

    out_path = REPO_ROOT / args.out
    render_markdown(rows, out_path, args.n_games_per_direction)

    if hasattr(env, "destroy"):
        env.destroy()


if __name__ == "__main__":
    main()
