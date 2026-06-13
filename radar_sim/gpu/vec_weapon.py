"""Vectorized space-based laser weapon on drone platform.

Continuous illumination kill model: laser must remain within kill_radius
of target for illumination_time_s seconds without interruption.

One drone per team, fixed at team altitude above team area.
All tensors shape [E, n_teams, ...] on GPU.
"""

import torch


class VecLaser:
    """GPU-vectorized laser weapon with continuous illumination kill model.

    Kill condition: aim_pos within kill_radius of actual_pos for
    illumination_time_s continuously. Any interruption resets the timer.
    """

    def __init__(
        self,
        num_envs: int,
        n_teams: int = 2,
        kill_radius_m: float = 0.2,
        illumination_time_s: float = 0.002,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.n_teams = n_teams
        self.kill_radius_m = kill_radius_m
        self.illumination_time_s = illumination_time_s
        self.device = device

        dev = torch.device(device)
        # [E, n_teams] — accumulated continuous illumination time
        self.illumination_time = torch.zeros(num_envs, n_teams, device=dev)
        # [E, n_teams] — whether currently on target
        self.on_target = torch.zeros(num_envs, n_teams, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self.illumination_time[env_ids] = 0.0
        self.on_target[env_ids] = False

    def step(
        self,
        dt: float,
        aim_pos: torch.Tensor,
        actual_pos: torch.Tensor,
        fire_on: torch.Tensor,
    ) -> torch.Tensor:
        """Check illumination per pulse.

        Args:
            dt: time step (seconds), typically pri = 100μs
            aim_pos: [E, n_teams, 3] — laser aim position
            actual_pos: [E, n_teams, n_enemy_radars, 3] — enemy radar positions
            fire_on: [E, n_teams] bool — whether laser is firing

        Returns:
            kills: [E, n_teams, n_enemy_radars] bool
        """
        # aim_pos: [E, T, 3] → [E, T, 1, 3]
        # actual_pos: [E, T, Er, 3]
        dist = (aim_pos.unsqueeze(2) - actual_pos).norm(dim=-1)  # [E, T, Er]
        # On target if ANY enemy radar within kill_radius
        any_on_target = (dist < self.kill_radius_m).any(dim=-1)  # [E, T]
        # Must also be firing
        self.on_target = any_on_target & fire_on

        # Continuous illumination: on_target → accumulate, else → reset
        self.illumination_time = torch.where(
            self.on_target,
            self.illumination_time + dt,
            torch.zeros_like(self.illumination_time),
        )

        # Kill if illumination time >= required AND currently on target
        illuminated = self.illumination_time >= self.illumination_time_s
        kill_eligible = illuminated & self.on_target  # [E, T]

        # Which specific enemy radars are hit
        in_range = dist < self.kill_radius_m  # [E, T, Er]
        kills = in_range & kill_eligible.unsqueeze(-1)  # [E, T, Er]

        # Reset timer after kill to prevent repeated kills
        self.illumination_time = torch.where(
            kills.any(dim=-1),
            torch.zeros_like(self.illumination_time),
            self.illumination_time,
        )

        return kills

    def get_illumination_progress(self) -> torch.Tensor:
        """Return illumination progress [E, n_teams] in [0, 1]."""
        return (self.illumination_time / self.illumination_time_s).clamp(0.0, 1.0)
