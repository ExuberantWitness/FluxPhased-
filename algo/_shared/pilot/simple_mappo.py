"""Minimal MAPPO inference adapter for Concerto pilot (A11 helper).

Loads a pre-trained checkpoint (radar_ac + commander_ac state dicts) and
exposes the `get_own_actions(env, team, ...)` API that LaserEpisodeRunner
expects from a trainer.

This mirrors LaserTrainer's `_nn_step` observation/action path bit-for-bit
(same obs dims, same residual_aim transform, same Kalman-fused anchor) so
the loaded policy produces actions in the same distribution it was trained
on. No PPO, no buffer, no reward shaping — pure inference.
"""

from __future__ import annotations

import torch
from typing import Dict, Optional

from algo._shared.train_laser import build_actors
from algo._shared.laser.sensing import (
    KalmanTracker, fused_sensing, enforce_radar_baseline,
)


class SimpleMAPPOTrainer:
    """Inference-only MAPPO trainer (no PPO, no buffer).

    For pilot cells that need an RL policy ("mappo" / "concerto_v1/v2"),
    load the checkpoint into actor-critic modules and provide get_own_actions.
    """

    def __init__(
        self,
        env,
        team: int,
        checkpoint_path: str,
        residual_scale_m: float = 100.0,
        min_radar_baseline_m: float = 5000.0,
        range_sigma_m: float = 0.05,
        crossrange_factor: float = 7.4e-5,
        track_q_m: float = 0.02,
        track_burnin: int = 120,
        jam_gain: float = 8.0,
        exposure_gain: float = 50.0,
        sensing_mode: str = "fused",
    ):
        self.env = env
        self.team = int(team)
        self.residual_scale_m = float(residual_scale_m)
        self.min_radar_baseline_m = float(min_radar_baseline_m)
        self.range_sigma_m = float(range_sigma_m)
        self.crossrange_factor = float(crossrange_factor)
        self.track_q_m = float(track_q_m)
        self.track_burnin = int(track_burnin)
        self.jam_gain = float(jam_gain)
        self.exposure_gain = float(exposure_gain)
        self.sensing_mode = sensing_mode

        # Env shape constants (mirrors LaserTrainer)
        self.E = env.num_envs
        self.R = env.n_radars
        self.N = env.n_elem
        self.P = env.n_pulses
        self.S = env.n_samples
        self.n_bins = env.n_bins
        self.device = env.device

        R_team = env.n_radars // env.n_teams
        self.r_start = team * R_team
        self.r_end = (team + 1) * R_team
        self.R_team = R_team

        # Build actor-critic (mirrors LaserTrainer cfg)
        cfg = dict(sub_array_size=5)
        self.radar_ac, self.commander_ac = build_actors(
            cfg, n_elem=env.n_elem, n_pulses=env.n_pulses,
            n_bins=env.n_bins, device=env.device,
        )

        # Load checkpoint
        sd = torch.load(checkpoint_path, map_location=env.device, weights_only=False)
        if "radar_ac" in sd:
            self.radar_ac.load_state_dict(sd["radar_ac"])
            self.commander_ac.load_state_dict(sd["commander_ac"])
        else:
            self.radar_ac.load_state_dict(sd)
        self.radar_ac.eval()
        self.commander_ac.eval()

        # Kalman tracker (mirrors LaserTrainer)
        self.kalman_tracker = KalmanTracker(
            track_q_m=self.track_q_m, track_burnin=self.track_burnin,
        )
        self.kalman_tracker.ensure_alloc(self.E, env.n_teams, torch.device(env.device))
        self.kalman_tracker._initialized = True

        # State cached between calls (mirrors LaserTrainer attributes the runner may touch)
        self._spectrum = None
        self._last_radar_obs = None
        # jam_level passthrough (set externally by runner / driver)
        self.jam_level = None
        # Reward shaper attribute (so runner.reset's reward_shaper check skips cleanly)
        self.reward_shaper = None

    def reset_episode(self, E: int, n_teams: int):
        self.kalman_tracker.reset()
        self.kalman_tracker.ensure_alloc(E, n_teams, torch.device(self.env.device))
        self.kalman_tracker._initialized = True
        self._spectrum = None

    # ------------------------------------------------------------------
    # Observation builders (mirror LaserTrainer exactly)
    # ------------------------------------------------------------------
    def _build_radar_obs(self, events: dict) -> torch.Tensor:
        """Build per-radar obs: [spec_flat, comm_flat(N*2), recon_flat(N*4),
        vehicle(5), laser_state(12), cmd_instr(16)] — matches LaserTrainer.
        """
        env = self.env
        dev = torch.device(env.device)
        E, R, N = self.E, self.R, self.N
        if self._spectrum is not None:
            spec_flat = self._spectrum.reshape(E, R, -1)
        else:
            spec_flat = torch.zeros(E, R, N * self.P * self.n_bins, device=dev)
        comm_flat = torch.zeros(E, R, N * 2, device=dev)
        recon_flat = torch.zeros(E, R, N * 4, device=dev)
        vehicle = torch.zeros(E, R, 5, device=dev)
        laser_state = torch.zeros(E, R, 12, device=dev)
        cmd_instr = torch.zeros(E, R, 16, device=dev)
        if "radar_pos" in events:
            vehicle[:, :, 0] = events["radar_pos"][:, :, 0]
            vehicle[:, :, 1] = events["radar_pos"][:, :, 1]
        return torch.cat([spec_flat, comm_flat, recon_flat, vehicle, laser_state, cmd_instr], dim=-1)

    def _build_commander_obs(self, events: dict) -> torch.Tensor:
        """Build [E, n_teams, 76] commander obs from env battlefield + fused_sensing."""
        env = self.env
        dev = torch.device(env.device)
        radar_latents = torch.zeros(self.E, env.n_radars, 32, device=dev)
        obs = env.battlefield.get_commander_observation(env.radar_pos, radar_latents)
        fused_sensing(
            obs,
            half_x=env.map_size[0] / 2.0, half_y=env.map_size[1] / 2.0,
            range_sigma_m=self.range_sigma_m,
            crossrange_factor=self.crossrange_factor,
            tracker=self.kalman_tracker,
            jam_gain=self.jam_gain,
            exposure_gain=self.exposure_gain,
            jam_level=self.jam_level,
        )
        return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _to_env_cmd_action(self, raw: torch.Tensor, cmd_obs: torch.Tensor) -> torch.Tensor:
        """Mirrors LaserTrainer._to_env_cmd_action: aim = anchor + residual × scale."""
        env_a = raw.clone()
        half_x = self.env.map_size[0] / 2.0
        half_y = self.env.map_size[1] / 2.0
        anchor = cmd_obs[..., 68:70]  # normalized enemy pos
        env_a[..., 1] = anchor[..., 0] + raw[..., 1] * (self.residual_scale_m / half_x)
        env_a[..., 2] = anchor[..., 1] + raw[..., 2] * (self.residual_scale_m / half_y)
        env_a[..., 3] = raw[..., 3] * (self.residual_scale_m / 1000.0)
        return env_a

    # ------------------------------------------------------------------
    # API: get_own_actions
    # ------------------------------------------------------------------
    def get_own_actions(
        self,
        env,
        team: int = None,
        deterministic: bool = True,
        spectrum: torch.Tensor = None,
        events: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        if team is None:
            team = self.team
        events = events or {}

        # Stash spectrum from runner (if provided) — same convention as LaserTrainer
        if spectrum is not None:
            self._spectrum = spectrum

        # Enforce radar baseline (mirrors LaserEpisodeRunner.reset flow)
        if self.min_radar_baseline_m > 0:
            enforce_radar_baseline(env, self.min_radar_baseline_m)

        # Build observations
        radar_obs = self._build_radar_obs(events)  # [E, R, state_dim]
        cmd_obs = self._build_commander_obs(events)  # [E, n_teams, 76]

        # Inference
        with torch.no_grad():
            E = self.E
            R = self.R
            T = env.n_teams
            radar_flat = radar_obs.reshape(E * R, -1)
            r_action, _, _, _ = self.radar_ac.get_action(radar_flat, deterministic=deterministic)
            r_action = r_action.reshape(E, R, -1)

            cmd_flat = cmd_obs.reshape(E * T, -1)
            c_action, _, _, _ = self.commander_ac.get_action(cmd_flat, deterministic=deterministic)
            c_action = c_action.reshape(E, T, -1)

            # Apply residual aim transformation (laser-specific)
            c_action_env = self._to_env_cmd_action(c_action, cmd_obs)

        # Slice own team's actions
        radar_actions_list = [r_action[:, r, :] for r in range(self.r_start, self.r_end)]
        commander_action_team = c_action_env[:, team, :]  # [E, 5]

        return {
            "r_start": self.r_start,
            "r_end": self.r_end,
            "radar_actions": radar_actions_list,
            "commander_action": commander_action_team,
            "transition": None,  # eval-only; no buffer writes
        }
