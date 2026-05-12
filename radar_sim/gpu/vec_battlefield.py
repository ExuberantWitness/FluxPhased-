"""Vectorized battlefield game state for E parallel environments.

Manages team structure, commander actions, missile BPSK comm link,
kill conditions, and win checks. All tensors on GPU.
"""

import torch
import numpy as np

from .vec_missile import VecMissile
from .waveform_gpu import (
    encode_bpsk, modulate_bpsk,
    demodulate_bpsk_batch, decode_bpsk_batch,
)

TEAM_RED = 0
TEAM_BLUE = 1

# Task IDs (must match vec_element_processor)
TASK_COMM = 3


class VecBattlefield:
    """Vectorized battlefield: team structure + missile combat + win conditions.

    Team layout (for R=4 radars):
      radar 0,1 → team 0 (Red)
      radar 2,3 → team 1 (Blue)

    Commander obs: 4 + 2 * num_input_length
      [0:2]            own radar 0 position (x, y) / half_map
      [2:4]            own radar 1 position (x, y) / half_map
      [4:4+N_in]       radar 0 latent (from radar NN encoder)
      [4+N_in:4+2*N_in] radar 1 latent

    Commander action: 3 + 2 * num_output_length
      [0]              launch_flag (>0.5 triggers launch)
      [1]              target_x (normalized -1..1)
      [2]              target_y (normalized -1..1)
      [3:3+N_out]      instruction to radar 0
      [3+N_out:3+2*N_out] instruction to radar 1
    """

    def __init__(
        self,
        num_envs: int,
        n_radars: int = 4,
        n_teams: int = 2,
        map_size=(20000.0, 20000.0),
        speed_ms: float = 244.4,
        kill_radius_m: float = 500.0,
        missile_rcs_dbsm: float = 10.0,
        red_launch_pos=(0.0, -10000.0),
        blue_launch_pos=(0.0, 10000.0),
        fs: float = 200e6,
        symbol_rate: float = 1e6,
        num_input_length: int = 32,
        num_output_length: int = 16,
        device: str = "cuda",
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
        self.commander_obs_dim = 4 + 2 * num_input_length
        self.commander_action_dim = 3 + 2 * num_output_length

        dev = torch.device(device)

        # Team mapping: radar i → team_id[i]
        r_per_team = n_radars // n_teams
        self.team_id = torch.tensor(
            [i // r_per_team for i in range(n_radars)],
            dtype=torch.long, device=dev,
        )
        # team → list of radar indices
        self.team_radar_indices = []
        for t in range(n_teams):
            self.team_radar_indices.append(
                torch.tensor(
                    [i for i in range(n_radars) if i // r_per_team == t],
                    dtype=torch.long, device=dev,
                )
            )

        # Launch positions per team
        self.launch_pos = torch.zeros(n_teams, 3, device=dev)
        self.launch_pos[TEAM_RED] = torch.tensor([red_launch_pos[0], red_launch_pos[1], 0.0])
        self.launch_pos[TEAM_BLUE] = torch.tensor([blue_launch_pos[0], blue_launch_pos[1], 0.0])

        # Missile subsystem
        self.missile = VecMissile(
            num_envs=num_envs, n_teams=n_teams,
            speed_ms=speed_ms, kill_radius_m=kill_radius_m,
            rcs_dbsm=missile_rcs_dbsm, device=device,
        )

        # Game state
        self.alive = torch.ones(num_envs, n_radars, dtype=torch.bool, device=dev)
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self.winners = torch.full((num_envs,), -1, dtype=torch.long, device=dev)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=dev)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self.missile.reset(env_ids)
        self.alive[env_ids] = True
        self.dones[env_ids] = False
        self.winners[env_ids] = -1
        self.step_count[env_ids] = 0

    # ------------------------------------------------------------------
    # Commander actions
    # ------------------------------------------------------------------

    def process_commander_actions(
        self,
        commander_actions: torch.Tensor,
        radar_pos: torch.Tensor,
    ):
        """Process commander launch decisions (first 3 dims of commander action).

        Args:
            commander_actions: [E, n_teams, commander_action_dim]
                [..., 0] = launch_flag (>0.5 triggers launch)
                [..., 1] = target_x (normalized -1..1 → map coords)
                [..., 2] = target_y (normalized -1..1 → map coords)
            radar_pos: [E, R, 3]
        """
        for t in range(self.n_teams):
            launch_flag = commander_actions[:, t, 0]  # [E]
            target_x = commander_actions[:, t, 1]  # [E] normalized
            target_y = commander_actions[:, t, 2]

            # Find envs that want to launch and don't have missile in flight
            want_launch = launch_flag > 0.5
            not_flying = ~self.missile.in_flight[:, t]
            can_launch = want_launch & not_flying & ~self.dones
            env_ids = torch.where(can_launch)[0]

            if env_ids.numel() == 0:
                continue

            n = env_ids.numel()
            start = self.launch_pos[t].unsqueeze(0).expand(n, 3).clone()

            # Denormalize target coordinates
            half_x = self.map_size[0] / 2.0
            half_y = self.map_size[1] / 2.0
            target = torch.zeros(n, 3, device=torch.device(self.device))
            target[:, 0] = target_x[env_ids] * half_x
            target[:, 1] = target_y[env_ids] * half_y
            target[:, 2] = 0.0

            self.missile.launch(env_ids, t, start, target)

    def extract_radar_instructions(
        self,
        commander_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-radar instruction vectors from commander actions.

        Args:
            commander_actions: [E, n_teams, commander_action_dim]
        Returns:
            instructions: [E, R, num_output_length]
        """
        E = self.num_envs
        dev = torch.device(self.device)
        N_out = self.num_output_length
        result = torch.zeros(E, self.n_radars, N_out, device=dev)

        for t in range(self.n_teams):
            own_idx = self.team_radar_indices[t]
            n_own = own_idx.shape[0]
            inst_start = 3
            inst_end = inst_start + N_out

            result[:, own_idx[0]] = commander_actions[:, t, inst_start:inst_end]
            if n_own > 1:
                result[:, own_idx[1]] = commander_actions[:, t, inst_end:inst_end + N_out]

        return result

    # ------------------------------------------------------------------
    # Missile BPSK communication
    # ------------------------------------------------------------------

    def process_missile_comm(
        self,
        radar_pos: torch.Tensor,
        task_ids: torch.Tensor,
        comm_params: torch.Tensor,
        tx_signal: torch.Tensor,
        channel,
        processor,
    ):
        """BPSK comm from radar comm elements to missile.

        Pipeline per team:
        1. Collect comm element TX signals across team's radars
        2. Sum (coherent combining approximation)
        3. Compute one-way channel: radar_mean_pos → missile_pos
        4. Apply channel (gain + noise)
        5. BPSK demodulate → decode (X, Y)
        6. CRC pass → update missile target

        Args:
            radar_pos: [E, R, 3]
            task_ids: [E, R, N] int
            comm_params: [E, R, N, 3]
            tx_signal: [E, R, N, S] complex64
            channel: VecChannel instance
            processor: VecElementProcessor instance
        """
        dev = torch.device(self.device)
        E = self.num_envs
        S = tx_signal.shape[-1]

        for t in range(self.n_teams):
            flying = self.missile.in_flight[:, t]  # [E]
            if not flying.any():
                continue

            flying_idx = torch.where(flying)[0]
            team_r = self.team_radar_indices[t]  # [R/2]

            # Comm mask: [E, R, N] bool → select team radars
            team_tasks = task_ids[:, team_r, :]  # [E, R/2, N]
            comm_mask = (team_tasks == TASK_COMM)  # [E, R/2, N]

            # Check if any comm elements exist for this team
            has_comm = comm_mask.any(dim=-1).any(dim=-1)  # [E]
            active = flying & has_comm
            if not active.any():
                continue
            active_idx = torch.where(active)[0]

            # Sum comm element TX signals per env
            team_tx = tx_signal[:, team_r, :, :]  # [E, R/2, N, S]
            masked = team_tx * comm_mask.unsqueeze(-1)  # [E, R/2, N, S]
            combined = masked.sum(dim=(1, 2))  # [E, S]

            # Only process active envs
            sig = combined[active_idx]  # [n_active, S]

            # One-way channel: mean radar pos → missile pos
            team_pos = radar_pos[:, team_r, :]  # [E, R/2, 3]
            radar_mean = team_pos.mean(dim=1)  # [E, 3]
            m_pos = self.missile.missile_pos[:, t, :]  # [E, 3]

            tx_p = radar_mean[active_idx]  # [n_active, 3]
            rx_p = m_pos[active_idx]

            _, _, gain = channel.compute_params_one_way(
                tx_p, rx_p,
                tx_power_w=1.0,
                directivity_db=44.0,
                system_loss_db=3.0,
            )

            # Apply channel
            rx = channel.apply_one_way(sig, gain)  # [n_active, S]

            # Add noise
            noise = torch.randn_like(rx) * (channel.noise_std / np.sqrt(2.0))
            rx = rx + noise

            # BPSK demodulate
            bits = demodulate_bpsk_batch(rx, self.symbol_rate, self.fs, n_bits=32)
            data_x, data_y, crc_ok = decode_bpsk_batch(bits)

            # Update missile target where CRC passed
            valid_mask = crc_ok
            if valid_mask.any():
                valid_envs = active_idx[valid_mask]
                n_valid = valid_envs.numel()
                half_x = self.map_size[0] / 2.0
                half_y = self.map_size[1] / 2.0
                new_target = torch.zeros(n_valid, 3, device=dev)
                new_target[:, 0] = data_x[valid_mask] * half_x
                new_target[:, 1] = data_y[valid_mask] * half_y
                new_target[:, 2] = 0.0
                self.missile.update_target(valid_envs, t, new_target)

    # ------------------------------------------------------------------
    # Missile physics + kill check
    # ------------------------------------------------------------------

    def step_missiles(self, dt: float, radar_pos: torch.Tensor):
        """Advance missile physics and check kills.

        Args:
            dt: time step (seconds)
            radar_pos: [E, R, 3]
        Returns:
            kills: [E, n_teams, n_enemy_radars] bool
        """
        self.missile.step(dt)
        self.step_count += 1

        kills = torch.zeros(
            self.num_envs, self.n_teams, self.n_radars // self.n_teams,
            dtype=torch.bool, device=torch.device(self.device),
        )

        for t in range(self.n_teams):
            enemy_team = 1 - t
            enemy_idx = self.team_radar_indices[enemy_team]  # [R/2]
            enemy_pos = radar_pos[:, enemy_idx, :]  # [E, R/2, 3]

            # Only check alive enemies
            enemy_alive = self.alive[:, enemy_idx]  # [E, R/2]
            enemy_pos = enemy_pos * enemy_alive.unsqueeze(-1).float()

            k = self.missile.check_kill(enemy_pos)  # [E, T, R/2]
            kills[:, t, :] = k[:, t, :]

        return kills

    def update_alive(self, kills: torch.Tensor):
        """Update radar alive status from kills [E, n_teams, n_enemy_radars]."""
        for t in range(self.n_teams):
            enemy_team = 1 - t
            enemy_idx = self.team_radar_indices[enemy_team]
            # kills[:, t, :] → which enemy radars were killed by team t's missile
            team_kills = kills[:, t, :]  # [E, R/2]
            for j, r_idx in enumerate(enemy_idx):
                killed = team_kills[:, j]  # [E]
                self.alive[:, r_idx] = self.alive[:, r_idx] & ~killed

    def check_win(self):
        """Check win conditions: destroy any enemy radar.

        Returns:
            dones: [E] bool
            winners: [E] long (0=red, 1=blue, -1=ongoing)
        """
        for t in range(self.n_teams):
            enemy_team = 1 - t
            enemy_idx = self.team_radar_indices[enemy_team]
            any_dead = (~self.alive[:, enemy_idx]).any(dim=-1)  # [E]
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
        """Build commander observation [E, n_teams, 4 + 2*N_in].

        Layout per team:
          [0:2]            own radar 0 position (x, y) / half_map
          [2:4]            own radar 1 position (x, y) / half_map
          [4:4+N_in]       radar 0 latent
          [4+N_in:4+2*N_in] radar 1 latent

        Args:
            radar_pos: [E, R, 3]
            radar_latents: [E, R, num_input_length] from radar NN encoder
        """
        E = self.num_envs
        dev = torch.device(self.device)
        N_in = self.num_input_length
        half_x = self.map_size[0] / 2.0
        half_y = self.map_size[1] / 2.0

        obs = torch.zeros(E, self.n_teams, self.commander_obs_dim, device=dev)

        for t in range(self.n_teams):
            own_idx = self.team_radar_indices[t]
            n_own = own_idx.shape[0]

            # Own radar positions [0:4]
            obs[:, t, 0] = radar_pos[:, own_idx[0], 0] / half_x
            obs[:, t, 1] = radar_pos[:, own_idx[0], 1] / half_y
            if n_own > 1:
                obs[:, t, 2] = radar_pos[:, own_idx[1], 0] / half_x
                obs[:, t, 3] = radar_pos[:, own_idx[1], 1] / half_y
            else:
                obs[:, t, 2] = obs[:, t, 0]
                obs[:, t, 3] = obs[:, t, 1]

            # Radar latents [4:4+2*N_in]
            obs[:, t, 4:4 + N_in] = radar_latents[:, own_idx[0]]
            if n_own > 1:
                obs[:, t, 4 + N_in:4 + 2 * N_in] = radar_latents[:, own_idx[1]]

        return obs

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

            # Any kill by this team's missile?
            team_kills = kills[:, t, :].any(dim=-1)  # [E]

            # Commander: +10 for kill, -10 for own radar killed
            commander_rewards[:, t] += team_kills.float() * 10.0

            # Any own radar killed by enemy?
            own_killed = kills[:, enemy_team, :].any(dim=-1)
            commander_rewards[:, t] -= own_killed.float() * 10.0

            # Radar agents on this team
            for ri in own_idx:
                radar_rewards[:, ri] += team_kills.float() * 1.0
                radar_rewards[:, ri] -= own_killed.float() * 1.0

            # Emission cost per step
            for ri in own_idx:
                radar_rewards[:, ri] -= 0.001

            # Urgency: commander penalized for not launching
            not_launched = ~self.missile.launched[:, t] & ~self.dones
            commander_rewards[:, t] -= not_launched.float() * 0.01

        return {
            "radar_rewards": radar_rewards,
            "commander_rewards": commander_rewards,
        }
