"""Vectorized cruise missile physics for E parallel environments.

One missile per team (2 total). Straight-line flight at constant speed
with real-time course correction toward current target coordinates.

All tensors shape [E, n_teams, ...] on GPU.
"""

import torch
import numpy as np


class VecMissile:
    """GPU-vectorized cruise missile: launch, flight, kill check."""

    def __init__(
        self,
        num_envs: int,
        n_teams: int = 2,
        speed_ms: float = 244.4,
        kill_radius_m: float = 500.0,
        rcs_dbsm: float = 10.0,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.n_teams = n_teams
        self.speed_ms = speed_ms
        self.kill_radius_m = kill_radius_m
        self.kill_radius_sq = kill_radius_m ** 2
        self.rcs_dbsm = rcs_dbsm
        self.device = device

        dev = torch.device(device)
        self.missile_pos = torch.zeros(num_envs, n_teams, 3, device=dev)
        self.missile_vel = torch.zeros(num_envs, n_teams, 3, device=dev)
        self.target_pos = torch.zeros(num_envs, n_teams, 3, device=dev)
        self.in_flight = torch.zeros(num_envs, n_teams, dtype=torch.bool, device=dev)
        self.launched = torch.zeros(num_envs, n_teams, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self.missile_pos[env_ids] = 0.0
        self.missile_vel[env_ids] = 0.0
        self.target_pos[env_ids] = 0.0
        self.in_flight[env_ids] = False
        self.launched[env_ids] = False

    def launch(
        self,
        env_ids: torch.Tensor,
        team_idx: int,
        start_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ):
        """Launch missile. No-op if already in flight for given envs."""
        dev = torch.device(self.device)
        env_ids = env_ids.to(dev)
        already = self.in_flight[env_ids, team_idx]
        new_envs = env_ids[~already]
        if new_envs.numel() == 0:
            return
        self.missile_pos[new_envs, team_idx] = start_pos[:new_envs.shape[0]].to(dev)
        self.target_pos[new_envs, team_idx] = target_pos[:new_envs.shape[0]].to(dev)
        self.missile_vel[new_envs, team_idx] = 0.0
        self.in_flight[new_envs, team_idx] = True
        self.launched[new_envs, team_idx] = True

    def update_target(
        self,
        env_ids: torch.Tensor,
        team_idx: int,
        new_target: torch.Tensor,
    ):
        """Update missile target coordinates (real-time course correction)."""
        dev = torch.device(self.device)
        env_ids = env_ids.to(dev)
        active = self.in_flight[env_ids, team_idx]
        valid = env_ids[active]
        if valid.numel() == 0:
            return
        self.target_pos[valid, team_idx] = new_target[:valid.shape[0]].to(dev)

    def step(self, dt: float):
        """Advance all in-flight missiles by dt seconds."""
        active = self.in_flight.unsqueeze(-1)  # [E, T, 1]
        diff = self.target_pos - self.missile_pos  # [E, T, 3]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [E, T, 1]
        direction = diff / dist  # [E, T, 3]
        self.missile_vel = direction * self.speed_ms * active.float()
        self.missile_pos = self.missile_pos + self.missile_vel * dt

    def check_kill(
        self,
        enemy_radar_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Check if any in-flight missile is within kill radius of enemy radars.

        Args:
            enemy_radar_pos: [E, n_enemy_radars, 3]
        Returns:
            kills: [E, n_teams, n_enemy_radars] bool
        """
        # missile_pos: [E, T, 3] → [E, T, 1, 3]
        # enemy_pos:   [E, E_r, 3] → [E, 1, E_r, 3]
        m = self.missile_pos.unsqueeze(2)  # [E, T, 1, 3]
        e = enemy_radar_pos.unsqueeze(1)   # [E, 1, E_r, 3]
        dist_sq = ((m - e) ** 2).sum(dim=-1)  # [E, T, E_r]
        in_range = dist_sq < self.kill_radius_sq  # [E, T, E_r]
        flying = self.in_flight.unsqueeze(-1)  # [E, T, 1]
        return in_range & flying
