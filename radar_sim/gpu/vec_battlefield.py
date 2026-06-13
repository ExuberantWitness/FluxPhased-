"""Vectorized battlefield game state for E parallel environments.

Manages team structure, drone laser weapon, BPSK comm from radar to drone,
commander perfect link, illumination kill model, and win conditions.
All tensors on GPU.
"""

import torch

from .vec_weapon import VecLaser
from .vec_drone import VecDrone

TEAM_RED = 0
TEAM_BLUE = 1


class VecBattlefield:
    """Vectorized battlefield: team structure + drone laser combat + win conditions.

    Team layout (for R=4 radars):
      radar 0,1 → team 0 (Red)
      radar 2,3 → team 1 (Blue)

    Commander obs: 4 + 2 * num_input_length + 8 = 76
    Commander action: 5 [fire_on_off, aim_x, aim_y, aim_z, reserved]
    """

    def __init__(
        self,
        num_envs: int,
        n_radars: int = 4,
        n_teams: int = 2,
        map_size=(20000.0, 20000.0),
        kill_radius_m: float = 0.2,
        illumination_time_s: float = 0.002,
        drone_altitude_m: float = 3000.0,
        fs: float = 200e6,
        symbol_rate: float = 1e6,
        num_input_length: int = 32,
        num_output_length: int = 16,
        device: str = "cuda",
        reward_config: dict = None,
    ):
        self.num_envs = num_envs
        self.n_radars = n_radars
        self.n_teams = n_teams
        self.map_size = map_size
        self.fs = fs
        self.symbol_rate = symbol_rate
        self.device = device
        self.num_input_length = num_input_length
        self.num_output_length = num_output_length
        self.commander_obs_dim = 4 + 2 * num_input_length + 8
        self.commander_action_dim = 5  # fire, aim_x, aim_y, aim_z, reserved
        self.reward_config = reward_config or {}

        dev = torch.device(device)

        # Team mapping: radar i → team_id[i]
        r_per_team = n_radars // n_teams
        self.team_id = torch.tensor(
            [i // r_per_team for i in range(n_radars)],
            dtype=torch.long, device=dev,
        )
        self.team_radar_indices = []
        for t in range(n_teams):
            self.team_radar_indices.append(
                torch.tensor(
                    [i for i in range(n_radars) if i // r_per_team == t],
                    dtype=torch.long, device=dev,
                )
            )

        # Drone + Laser subsystems
        self.drone = VecDrone(
            num_envs=num_envs, n_teams=n_teams, n_radars=n_radars,
            altitude_m=drone_altitude_m, map_size=map_size,
            fs=fs, symbol_rate=symbol_rate, device=device,
        )
        self.laser = VecLaser(
            num_envs=num_envs, n_teams=n_teams,
            kill_radius_m=kill_radius_m,
            illumination_time_s=illumination_time_s,
            device=device,
        )

        # Game state
        self.alive = torch.ones(num_envs, n_radars, dtype=torch.bool, device=dev)
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self.winners = torch.full((num_envs,), -1, dtype=torch.long, device=dev)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=dev)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self.drone.reset(env_ids)
        self.laser.reset(env_ids)
        self.alive[env_ids] = True
        self.dones[env_ids] = False
        self.winners[env_ids] = -1
        self.step_count[env_ids] = 0

    # ------------------------------------------------------------------
    # Commander actions (perfect link to drone)
    # ------------------------------------------------------------------

    def process_commander_actions(
        self,
        commander_actions: torch.Tensor,
    ):
        """Process commander fire/aim via perfect data link to drone.

        Args:
            commander_actions: [E, n_teams, 5]
                [..., 0] = fire_on_off (>0.5 → fire)
                [..., 1] = aim_x (normalized -1..1)
                [..., 2] = aim_y (normalized -1..1)
                [..., 3] = aim_z (normalized -1..1)
                [..., 4] = reserved
        """
        self.drone.process_commander_actions(commander_actions)

    # ------------------------------------------------------------------
    # Radar BPSK comm to drone
    # ------------------------------------------------------------------

    def process_radar_comm(
        self,
        tx_signal: torch.Tensor,
        radar_pos: torch.Tensor,
        channel,
    ):
        """BPSK comm from radar TX to drone (physical channel).

        Full-signal matched filter: drone receives sum of all elements'
        TX, applies channel, BPSK demodulates. No task_ids needed.

        Args:
            tx_signal: [E, R, N, S] complex64
            radar_pos: [E, R, 3]
            channel: VecChannel instance
        """
        self.drone.process_radar_comm(tx_signal, radar_pos, channel)

    # ------------------------------------------------------------------
    # Laser kill check
    # ------------------------------------------------------------------

    def step_lasers(self, dt: float, radar_pos: torch.Tensor) -> torch.Tensor:
        """Advance laser illumination and check kills.

        Args:
            dt: time step (seconds)
            radar_pos: [E, R, 3]
        Returns:
            kills: [E, n_teams, n_enemy_radars] bool
        """
        self.step_count += 1

        # Resolve aim: commander > radar
        self.drone.update_aim()

        # Build enemy positions per team
        dev = torch.device(self.device)
        kills = torch.zeros(
            self.num_envs, self.n_teams, self.n_radars // self.n_teams,
            dtype=torch.bool, device=dev,
        )

        for t in range(self.n_teams):
            enemy_team = 1 - t
            enemy_idx = self.team_radar_indices[enemy_team]
            enemy_pos = radar_pos[:, enemy_idx, :]  # [E, R/2, 3]

            # Only alive enemies
            enemy_alive = self.alive[:, enemy_idx]  # [E, R/2]
            enemy_pos = enemy_pos * enemy_alive.unsqueeze(-1).float()

            # Fire status
            fire_on = self.drone.fire_on[:, t]

            # Build per-team inputs for laser
            # aim_pos needs [E, n_teams, 3] → we process per team
            # actual_pos needs [E, n_teams, n_enemy, 3]
            actual_pos = enemy_pos.unsqueeze(1)  # [E, 1, R/2, 3]

            # Expand aim and fire for single-team check
            aim_expanded = self.drone.laser_aim[:, t:t+1, :].unsqueeze(2).expand(
                -1, -1, enemy_idx.shape[0], -1)  # [E, 1, R/2, 3]

            # Temporarily set laser state for this team's check
            # Use laser.step with expanded inputs
            dist = (aim_expanded - actual_pos).norm(dim=-1)  # [E, 1, R/2]
            on_target = (dist < self.laser.kill_radius_m).any(dim=-1)  # [E, 1]
            on_target = on_target & fire_on.unsqueeze(-1)  # [E, 1]

            # Continuous illumination
            illum = self.laser.illumination_time[:, t:t+1]
            illum = torch.where(
                on_target, illum + dt, torch.zeros_like(illum))
            kill_eligible = (illum >= self.laser.illumination_time_s) & on_target

            in_range = dist < self.laser.kill_radius_m  # [E, 1, R/2]
            team_kills = in_range & kill_eligible.unsqueeze(-1)  # [E, 1, R/2]
            kills[:, t, :] = team_kills[:, 0, :]

            # Update laser illumination state
            self.laser.illumination_time[:, t] = illum[:, 0]
            self.laser.on_target[:, t] = on_target[:, 0]

            # Reset timer after kill
            if team_kills.any():
                killed_envs = team_kills.any(dim=-1)[:, 0]
                self.laser.illumination_time[killed_envs, t] = 0.0

        return kills

    def update_alive(self, kills: torch.Tensor):
        """Update radar alive status from kills [E, n_teams, n_enemy_radars]."""
        for t in range(self.n_teams):
            enemy_team = 1 - t
            enemy_idx = self.team_radar_indices[enemy_team]
            team_kills = kills[:, t, :]
            for j, r_idx in enumerate(enemy_idx):
                self.alive[:, r_idx] = self.alive[:, r_idx] & ~team_kills[:, j]

    def check_win(self):
        """Check win conditions: destroy any enemy radar.

        Returns:
            dones: [E] bool
            winners: [E] long (0=red, 1=blue, -1=ongoing)
        """
        for t in range(self.n_teams):
            enemy_team = 1 - t
            enemy_idx = self.team_radar_indices[enemy_team]
            any_dead = (~self.alive[:, enemy_idx]).any(dim=-1)
            newly_won = any_dead & ~self.dones
            self.winners[newly_won] = t
            self.dones = self.dones | newly_won

        return self.dones.clone(), self.winners.clone()

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def get_commander_observation(
        self,
        radar_pos: torch.Tensor,
        radar_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Build commander observation via drone.

        Args:
            radar_pos: [E, R, 3]
            radar_latents: [E, R, num_input_length]
        Returns:
            obs: [E, n_teams, commander_obs_dim]
        """
        return self.drone.get_commander_obs(radar_pos, radar_latents)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def compute_rewards(
        self,
        kills: torch.Tensor,
        dones: torch.Tensor,
        winners: torch.Tensor,
    ) -> dict:
        """Compute per-radar and per-commander rewards.

        Returns:
            {"radar_rewards": [E, R], "commander_rewards": [E, n_teams]}
        """
        dev = torch.device(self.device)
        radar_rewards = torch.zeros(self.num_envs, self.n_radars, device=dev)
        commander_rewards = torch.zeros(self.num_envs, self.n_teams, device=dev)

        for t in range(self.n_teams):
            enemy_team = 1 - t
            own_idx = self.team_radar_indices[t]
            enemy_idx = self.team_radar_indices[enemy_team]

            team_kills = kills[:, t, :].any(dim=-1)  # [E]
            commander_rewards[:, t] += team_kills.float() * self.reward_config.get('kill_bonus', 100.0)

            own_killed = kills[:, enemy_team, :].any(dim=-1)
            commander_rewards[:, t] += own_killed.float() * self.reward_config.get('death_penalty', -10.0)

            for ri in own_idx:
                radar_rewards[:, ri] += team_kills.float() * self.reward_config.get('radar_kill_share', 5.0)
                radar_rewards[:, ri] += own_killed.float() * self.reward_config.get('radar_death_share', -1.0)
                radar_rewards[:, ri] += self.reward_config.get('emission_cost', -0.001)

            # Illumination progress reward
            progress = self.laser.get_illumination_progress()  # [E, T]
            commander_rewards[:, t] += progress[:, t] * self.reward_config.get('illumination_reward', 1.0)

        return {
            "radar_rewards": radar_rewards,
            "commander_rewards": commander_rewards,
        }
