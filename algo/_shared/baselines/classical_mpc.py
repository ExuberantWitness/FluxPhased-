"""WP2 BASELINE — Classical MPC controller (no RL).

EAAI requires a non-RL engineering baseline to justify "AI beats classical".
This controller uses the same env + same multi-static Kalman fused sensing
as the RL policies, but replaces the learned actor with a fixed rule:

  - Each team's radars beam-steer toward the fused enemy anchor (cmd_obs[68:70]).
  - Commander fires whenever ≥1 own radar has line-of-sight to the anchor
    (always-on strategy; optimal when kill_radius_m > 0).
  - No waveform agility, no jamming, no comm — pure beam-follow.

This isolates the value of RL from the value of the sensing frontend. If RL
doesn't beat this, the contribution is the sensing, not the learning.

Run:
    python -m training.baselines.classical_mpc --config configs/wp1_gate.yaml \
        --n-eval-games 50 --max-steps 500
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from radar_sim.gpu.vec_mfar_env import MFARVecEnv
from radar_sim.config import DEFAULT_ROWS, DEFAULT_COLS
from training.laser.episode import LaserEpisodeRunner
from training.laser.sensing import enforce_radar_baseline


def _create_env(config: dict) -> MFARVecEnv:
    env_cfg = config["env"]
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
        vehicle_speed_ms=env_cfg.get("vehicle_speed_ms", 244.4),
    )
    for k in ("kill_radius_m", "illumination_time_s", "drone_altitude_m", "map_size"):
        if k in env_cfg:
            kwargs[k] = env_cfg[k]
    # WP3.2 damage injection (mirror training/train.py)
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


class ClassicalMPC:
    """Rule-based beam-follow controller. Same API subset as TeamPPOTrainer.

    get_own_actions(env, team, ...) returns a dict with radar_actions and
    commander_action derived analytically from the fused enemy anchor.
    """

    def __init__(
        self,
        env: MFARVecEnv,
        team: int,
        min_radar_baseline_m: float = 5000.0,
        range_sigma_m: float = 0.05,
        crossrange_factor: float = 7.4e-5,
        track_q_m: float = 0.02,
        track_burnin: int = 120,
        half_map_m: float = 10000.0,
    ):
        self.env = env
        self.team = int(team)
        self.min_radar_baseline_m = float(min_radar_baseline_m)
        self.range_sigma_m = float(range_sigma_m)
        self.crossrange_factor = float(crossrange_factor)
        self.track_q_m = float(track_q_m)
        self.track_burnin = int(track_burnin)
        self.half_map_m = float(half_map_m)

        E = env.num_envs
        R_team = env.n_radars // env.n_teams
        self.r_start = team * R_team
        self.r_end = (team + 1) * R_team
        self.R_team = R_team

        from training.laser.sensing import KalmanTracker
        self.kalman_tracker = KalmanTracker(
            track_q_m=self.track_q_m, track_burnin=self.track_burnin,
        )
        self.kalman_tracker.ensure_alloc(E, env.n_teams,
                                          torch.device(env.device))
        self.kalman_tracker._initialized = True

    def reset_episode(self, E: int, n_teams: int):
        self.kalman_tracker.reset()
        self.kalman_tracker.ensure_alloc(E, n_teams, torch.device(self.env.device))
        self.kalman_tracker._initialized = True

    def get_own_actions(
        self,
        env: MFARVecEnv,
        team: int = None,
        deterministic: bool = True,
        spectrum: torch.Tensor = None,
        events: dict = None,
    ) -> Dict[str, torch.Tensor]:
        """Analytical policy: beam-steer to fused enemy anchor + always fire.

        Layout matches the LaserEpisodeRunner contract — radar_actions are
        in [-1, 1] per-element action space; commander_action[0]=fire (1.0),
        commander_action[1:4]=aim in [-1, 1] (absolute tanh scale).
        """
        if team is None:
            team = self.team

        dev = torch.device(env.device)
        E = env.num_envs
        R_team = self.R_team
        n_elem = env.n_elem
        ACTION_PER_ELEM = 22

        # Build commander obs to extract fused enemy anchor (cmd_obs[68:70])
        # NOTE: radar_latents must be 32-dim so off = 4 + 2*32 = 68 matches
        # the layout in vec_drone.py:get_commander_obs and fused_sensing().
        radar_latents = torch.zeros(E, env.n_radars, 32, device=dev)
        cmd_obs = env.battlefield.get_commander_observation(
            env.radar_pos, radar_latents,
        )  # [E, n_teams, 76]

        # Apply fused sensing in-place (writes fused enemy xy into cmd_obs[68:72])
        from training.laser.sensing import fused_sensing
        fused_sensing(
            cmd_obs,
            half_x=self.half_map_m, half_y=self.half_map_m,
            range_sigma_m=self.range_sigma_m,
            crossrange_factor=self.crossrange_factor,
            tracker=self.kalman_tracker,
        )

        team_obs = cmd_obs[:, team, :]  # [E, 76]
        enemy_x_norm = team_obs[:, 68]   # in [-1, 1], absolute map coords
        enemy_y_norm = team_obs[:, 69]

        # Radar actions: all zeros except beam_az=enemy_x, beam_el=enemy_y
        # Layout per element: [task_id(4), beam_az(1), beam_el(1), waveform(8), jam(4), comm(4)]
        # beam_az at index 4, beam_el at index 5. Scale by enemy map-normalized pos.
        radar_actions = torch.zeros(E, R_team, env.action_dim, device=dev)
        elem_actions = radar_actions[:, :, :n_elem * ACTION_PER_ELEM].reshape(
            E, R_team, n_elem, ACTION_PER_ELEM,
        )
        # Beam steer: convert enemy map-pos to az/el via simple ratio
        # (real beamforming would use true LOS geometry; this is the same
        # approximation the RL policy's action[4,5] uses)
        elem_actions[:, :, :, 4] = enemy_x_norm.view(E, 1, 1)  # beam_az
        elem_actions[:, :, :, 5] = enemy_y_norm.view(E, 1, 1)  # beam_el

        # Commander action: fire=1.0 (always), aim=enemy anchor, residual=0
        # action[0]=fire_on_off, action[1:4]=aim_xyz (abs scale)
        commander_action = torch.zeros(E, 5, device=dev)
        commander_action[:, 0] = 1.0                                # fire
        commander_action[:, 1] = enemy_x_norm                       # aim x
        commander_action[:, 2] = enemy_y_norm                       # aim y
        commander_action[:, 3] = 0.0                                # aim z (drone alt is fixed)

        return {
            "r_start": self.r_start,
            "r_end": self.r_end,
            "radar_actions": radar_actions,
            "commander_action": commander_action,
            "transition": None,   # no transitions collected
        }


def evaluate_mpc(config: dict, n_eval_games: int, max_steps: int) -> dict:
    """Run N episodes of Classical MPC vs Classical MPC (self-play baseline).

    Returns aggregate metrics: red_win_rate, blue_win_rate, draw_rate,
    kill_rate, mean_illumination_progress.
    """
    env = _create_env(config)
    runner = LaserEpisodeRunner(env, pulses_per_control=config["env"].get("pulses_per_control", 5))

    min_baseline = float(config["env"].get("min_radar_baseline_m", 5000.0))
    range_sigma = float(config.get("sensing_noise", {}).get("range_sigma_m", 0.05))
    crossrange_factor = float(config.get("sensing_noise", {}).get("crossrange_factor", 7.4e-5))
    track_q = float(config.get("sensing_noise", {}).get("track_q_m", 0.02))
    track_burnin = int(config.get("sensing_noise", {}).get("track_burnin", 120))
    half_map = float(config["env"].get("map_size", [20000.0, 20000.0])[0]) / 2.0

    red = ClassicalMPC(env, team=0, min_radar_baseline_m=min_baseline,
                        range_sigma_m=range_sigma, crossrange_factor=crossrange_factor,
                        track_q_m=track_q, track_burnin=track_burnin, half_map_m=half_map)
    blue = ClassicalMPC(env, team=1, min_radar_baseline_m=min_baseline,
                         range_sigma_m=range_sigma, crossrange_factor=crossrange_factor,
                         track_q_m=track_q, track_burnin=track_burnin, half_map_m=half_map)

    red_wins = 0.0
    blue_wins = 0.0
    draws = 0.0
    n_red_kills = 0
    n_blue_kills = 0
    total_progress = 0.0

    E = env.num_envs
    games_per_batch = E
    n_batches = max(1, n_eval_games // games_per_batch)

    print(f"[MPC eval] {n_batches} batches × {games_per_batch} envs = "
          f"{n_batches * games_per_batch} games, {max_steps} steps each")

    for batch in range(n_batches):
        runner.reset(red_trainer=red, blue_trainer=blue)
        if min_baseline > 0:
            enforce_radar_baseline(env, min_baseline)

        last_result = None
        for step in range(max_steps):
            out = runner.step_control(red, blue, deterministic=True)
            last_result = out["result"]
            if last_result is None:
                break
            if last_result["dones"].any():
                break

        if last_result is None:
            draws += games_per_batch
            continue

        # Tally wins from winners tensor (0=red, 1=blue, -1=no winner/draw)
        winners = last_result["winners"]  # [E]
        for e in range(games_per_batch):
            w = int(winners[e].item())
            if w == 0:
                red_wins += 1.0
            elif w == 1:
                blue_wins += 1.0
            else:
                draws += 1.0

        # Kill stats
        if "kills" in last_result:
            kills = last_result["kills"]  # [E, n_teams, n_enemy_radars]
            red_killed = kills[:, 0, :].any(dim=-1).sum().item()
            blue_killed = kills[:, 1, :].any(dim=-1).sum().item()
            n_red_kills += red_killed
            n_blue_kills += blue_killed

        # Progress
        progress = env.battlefield.laser.get_illumination_progress()  # [E, T]
        total_progress += float(progress.sum().item())

        if (batch + 1) % 5 == 0 or batch == n_batches - 1:
            total = red_wins + blue_wins + draws
            print(f"  batch {batch + 1}/{n_batches}  "
                  f"red_wr={red_wins / total:.3f}  blue_wr={blue_wins / total:.3f}  "
                  f"draw={draws / total:.3f}")

    total = max(1, red_wins + blue_wins + draws)
    env.destroy()
    del env

    return {
        "red_win_rate": red_wins / total,
        "blue_win_rate": blue_wins / total,
        "draw_rate": draws / total,
        "red_kill_count": n_red_kills,
        "blue_kill_count": n_blue_kills,
        "mean_illumination_progress": total_progress / total,
        "n_games": int(total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wp1_gate.yaml")
    ap.add_argument("--n-eval-games", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-envs", type=int, default=None,
                    help="Override config env.num_envs (useful when GPU is busy)")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.device:
        config["env"]["device"] = args.device
    if args.num_envs is not None:
        config["env"]["num_envs"] = args.num_envs

    print("=" * 72)
    print("WP2 Classical MPC Baseline — beam-follow + always-fire")
    print("=" * 72)
    print(f"Config: {args.config}")
    print(f"Eval:   {args.n_eval_games} games × {args.max_steps} steps")
    print()

    t0 = time.time()
    metrics = evaluate_mpc(config, args.n_eval_games, args.max_steps)
    elapsed = time.time() - t0

    print()
    print("=" * 72)
    print(f"Classical MPC results ({elapsed:.1f}s):")
    print(f"  red_win_rate    = {metrics['red_win_rate']:.3f}")
    print(f"  blue_win_rate   = {metrics['blue_win_rate']:.3f}")
    print(f"  draw_rate       = {metrics['draw_rate']:.3f}")
    print(f"  red kills       = {metrics['red_kill_count']}")
    print(f"  blue kills      = {metrics['blue_kill_count']}")
    print(f"  mean progress   = {metrics['mean_illumination_progress']:.4f}")
    print(f"  total games     = {metrics['n_games']}")
    print("=" * 72)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "wp2_classical_mpc.log"
    with open(log_path, "a") as f:
        f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        for k, v in metrics.items():
            f.write(f"  {k} = {v}\n")
        f.write(f"  elapsed_s = {elapsed:.1f}\n\n")
    print(f"(appended to {log_path})")


if __name__ == "__main__":
    main()
