"""Vectorized cruise missile physics for E parallel environments.

One missile per team (2 total). Straight-line flight at constant speed
with real-time course correction toward current target coordinates.

All tensors shape [E, n_teams, ...] on GPU.
"""

import torch
import numpy as np


def swerling_gain_multiplier(shape, model, device):
    """Generate Swerling RCS fluctuation gain multiplier sqrt(sigma/sigma_avg).

    Args:
        shape: output tensor shape, e.g. [E, R]
        model: 0=none, 1=slow exp, 2=fast exp, 3=slow chi2(4), 4=fast chi2(4)
        device: torch device
    Returns:
        [shape] float32 voltage-ratio multiplier
    """
    if model == 0:
        return torch.ones(shape, device=device)
    if model in (1, 2):
        u = torch.rand(shape, device=device).clamp(min=1e-10)
        return torch.sqrt(-torch.log(u))
    if model in (3, 4):
        u1 = torch.rand(shape, device=device).clamp(min=1e-10)
        u2 = torch.rand(shape, device=device).clamp(min=1e-10)
        return torch.sqrt((-torch.log(u1) - torch.log(u2)) / 2.0)
    return torch.ones(shape, device=device)


class VecMissile:
    """GPU-vectorized cruise missile: launch, flight, kill check."""

    def __init__(
        self,
        num_envs: int,
        n_teams: int = 2,
        speed_ms: float = 244.4,
        kill_radius_m: float = 500.0,
        rcs_dbsm: float = 10.0,
        rcs_nose_dbsm: float = -5.0,
        rcs_side_dbsm: float = 12.0,
        rcs_tail_dbsm: float = 3.0,
        swerling_model: int = 3,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.n_teams = n_teams
        self.speed_ms = speed_ms
        self.kill_radius_m = kill_radius_m
        self.kill_radius_sq = kill_radius_m ** 2
        self.rcs_dbsm = rcs_dbsm
        self.rcs_nose_dbsm = rcs_nose_dbsm
        self.rcs_side_dbsm = rcs_side_dbsm
        self.rcs_tail_dbsm = rcs_tail_dbsm
        self.swerling_model = swerling_model
        self.device = device

        # Quadratic interpolation coefficients for RCS(aspect_angle)
        # c = cos_aspect ∈ [-1, 1]: -1=nose-on, 0=broadside, +1=tail-on
        # RCS(c) = a*c² + b*c + d  (in dBsm)
        nd, sd, td = rcs_nose_dbsm, rcs_side_dbsm, rcs_tail_dbsm
        self._rcs_a = (nd + td) / 2.0 - sd
        self._rcs_b = (td - nd) / 2.0
        self._rcs_d = sd

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

    def compute_aspect_rcs(self, radar_pos: torch.Tensor) -> torch.Tensor:
        """Compute aspect-angle dependent RCS for each radar-missile pair.

        Args:
            radar_pos: [E, R, 3]
        Returns:
            rcs_dbsm: [E, n_teams, R] float32 — per-radar RCS in dBsm
        """
        dev = torch.device(self.device)
        E = self.num_envs
        R = radar_pos.shape[1]

        # LOS vector: radar → missile = missile_pos - radar_pos
        m_pos = self.missile_pos                              # [E, T, 3]
        los = m_pos.unsqueeze(2) - radar_pos.unsqueeze(1)    # [E, T, R, 3]
        los_dist = los.norm(dim=-1).clamp(min=1.0)           # [E, T, R]
        los_hat = los / los_dist.unsqueeze(-1)                # [E, T, R, 3]

        # Missile heading: normalize velocity (zero vel → nose direction)
        m_vel = self.missile_vel                              # [E, T, 3]
        vel_norm = m_vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        heading = m_vel / vel_norm                             # [E, T, 3]

        # Aspect cosine: cos_aspect = dot(los_hat, heading)
        cos_aspect = (los_hat * heading.unsqueeze(2)).sum(dim=-1)  # [E, T, R]
        cos_aspect = cos_aspect.clamp(-1.0, 1.0)

        # Quadratic RCS model: RCS(c) = a*c² + b*c + d
        c = cos_aspect
        rcs = self._rcs_a * c ** 2 + self._rcs_b * c + self._rcs_d  # [E, T, R]

        # Zero out non-flying missiles
        flying = self.in_flight.unsqueeze(-1)  # [E, T, 1]
        rcs = rcs * flying.float()

        return rcs

    def compute_aspect_rcs_correction(self, radar_pos: torch.Tensor) -> torch.Tensor:
        """Compute gain correction factor for aspect RCS.

        Returns:
            correction: [E, n_teams, R] float32 — multiply with base gain
        """
        rcs_aspect = self.compute_aspect_rcs(radar_pos)  # [E, T, R] dBsm
        # gain ∝ sqrt(σ), so correction = 10^((rcs_aspect - rcs_base) / 20)
        correction_db = rcs_aspect - self.rcs_dbsm
        return 10.0 ** (correction_db / 20.0)
