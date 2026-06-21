"""Vectorized drone platform for laser weapon.

One drone per team, fixed at altitude above team area.
Receives aim commands from:
  1. radar BPSK comm (physical channel, ~1.22m precision)
  2. commander perfect link (float precision, higher priority)
Laser aim position determined by: commander > radar.

All tensors on GPU.
"""

import torch

from .waveform_gpu import (
    demodulate_bpsk_batch,
    decode_bpsk_batch,
)


class VecDrone:
    """GPU-vectorized drone: fixed position, receives comm, outputs laser aim.

    BPSK receiver extracts position estimate from full TX signal
    (matched filtering, no task_ids needed).
    Commander perfect link overrides radar comm.
    """

    def __init__(
        self,
        num_envs: int,
        n_teams: int = 2,
        n_radars: int = 4,
        altitude_m: float = 3000.0,
        map_size=(20000.0, 20000.0),
        fs: float = 200e6,
        symbol_rate: float = 1e6,
        device: str = "cuda",
        comm_rate_bps: float = 0.0,
        pri: float = 1e-4,
    ):
        self.num_envs = num_envs
        self.n_teams = n_teams
        self.n_radars = n_radars
        self.altitude_m = altitude_m
        self.map_size = map_size
        self.fs = fs
        self.symbol_rate = symbol_rate
        self.device = device
        # WP3.2 ISAC uplink rate cap: 0 → no cap. Bits beyond rate*per-pulse-dt
        # are dropped (CRC will fail naturally → position not updated).
        self.comm_rate_bps = float(comm_rate_bps)
        self.comm_bits_per_pulse = max(0, int(self.comm_rate_bps * pri)) \
            if self.comm_rate_bps > 0 else 0

        dev = torch.device(device)
        r_per_team = n_radars // n_teams

        # Drone positions: one per team, fixed at altitude
        # Red team at (0, -map_y/4), Blue at (0, +map_y/4)
        self.pos = torch.zeros(num_envs, n_teams, 3, device=dev)
        half_y = map_size[1] / 2.0
        self.pos[:, 0, :] = torch.tensor([0.0, -half_y / 2.0, altitude_m])
        self.pos[:, 1, :] = torch.tensor([0.0, half_y / 2.0, altitude_m])

        # Team radar indices
        self.team_radar_indices = []
        for t in range(n_teams):
            self.team_radar_indices.append(
                torch.tensor(
                    [i for i in range(n_radars) if i // r_per_team == t],
                    dtype=torch.long, device=dev,
                )
            )

        # Laser aim position: [E, n_teams, 3]
        self.laser_aim = torch.zeros(num_envs, n_teams, 3, device=dev)

        # Fire status: [E, n_teams]
        self.fire_on = torch.zeros(num_envs, n_teams, dtype=torch.bool, device=dev)

        # Commander aim (perfect link): [E, n_teams, 3], None if not set
        self._commander_aim = torch.zeros(num_envs, n_teams, 3, device=dev)
        self._commander_fire = torch.zeros(num_envs, n_teams, dtype=torch.bool, device=dev)

        # BPSK decoded position from radar comm: [E, n_teams, 3]
        self._radar_comm_pos = torch.zeros(num_envs, n_teams, 3, device=dev)
        self._radar_comm_valid = torch.zeros(num_envs, n_teams, dtype=torch.bool, device=dev)

        # CRC status for reward shaping: [E, n_teams]
        self._last_comm_crc_ok = torch.zeros(num_envs, n_teams, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        self.laser_aim[env_ids] = 0.0
        self.fire_on[env_ids] = False
        self._commander_aim[env_ids] = 0.0
        self._commander_fire[env_ids] = False
        self._radar_comm_pos[env_ids] = 0.0
        self._radar_comm_valid[env_ids] = False
        self._last_comm_crc_ok[env_ids] = False

    def process_radar_comm(
        self,
        tx_signal: torch.Tensor,
        radar_pos: torch.Tensor,
        channel,
    ):
        """BPSK comm from radar TX → drone receiver.

        Full-signal matched filter: drone receives sum of all elements'
        TX (after channel), then BPSK demodulates. No task_ids needed.

        Args:
            tx_signal: [E, R, N, S] complex64 — full TX signal
            radar_pos: [E, R, 3]
            channel: VecChannel instance
        """
        dev = torch.device(self.device)
        E = self.num_envs
        S = tx_signal.shape[-1]
        half_x = self.map_size[0] / 2.0
        half_y = self.map_size[1] / 2.0

        self._last_comm_crc_ok.zero_()

        for t in range(self.n_teams):
            team_r = self.team_radar_indices[t]

            # Sum all team elements' TX into single signal per env
            team_tx = tx_signal[:, team_r, :, :]  # [E, R/2, N, S]
            combined = team_tx.reshape(E, -1, S).sum(dim=1)  # [E, S]

            # One-way channel: mean radar pos → drone pos
            radar_mean = radar_pos[:, team_r, :].mean(dim=1)  # [E, 3]
            drone_p = self.pos[:, t, :]  # [E, 3]

            _, _, gain = channel.compute_params_one_way(
                radar_mean, drone_p,
                tx_power_w=1.0,
                directivity_db=44.0,
                system_loss_db=3.0,
            )

            # Apply channel + noise
            rx = channel.apply_one_way(combined, gain)  # [E, S]
            noise = torch.randn_like(rx) * channel.noise_std
            rx = rx + noise

            # BPSK demodulate + decode
            bits = demodulate_bpsk_batch(rx, self.symbol_rate, self.fs, n_bits=32)
            # WP3.2 ISAC uplink rate cap: if configured, only the first
            # N bits survive (where N = comm_rate_bps × pulse duration).
            # Remaining bits are zeroed (CRC will fail → no position update).
            if self.comm_bits_per_pulse > 0 and self.comm_bits_per_pulse < 32:
                bits[..., self.comm_bits_per_pulse:] = 0
            data_x, data_y, crc_ok = decode_bpsk_batch(bits)

            # Store CRC status
            self._last_comm_crc_ok[:, t] = crc_ok

            # Update radar comm position where CRC passes
            if crc_ok.any():
                valid = crc_ok
                self._radar_comm_pos[:, t, 0] = torch.where(
                    valid, data_x * half_x, self._radar_comm_pos[:, t, 0])
                self._radar_comm_pos[:, t, 1] = torch.where(
                    valid, data_y * half_y, self._radar_comm_pos[:, t, 1])
                self._radar_comm_pos[:, t, 2] = 0.0
                self._radar_comm_valid[:, t] = self._radar_comm_valid[:, t] | valid

    def process_commander_actions(
        self,
        commander_actions: torch.Tensor,
    ):
        """Perfect data link from commander → drone.

        Commander has priority over radar comm.

        Args:
            commander_actions: [E, n_teams, 5]
                [..., 0] = fire_on (>0.5 → fire)
                [..., 1] = aim_x (normalized -1..1)
                [..., 2] = aim_y (normalized -1..1)
                [..., 3] = aim_z (normalized -1..1)
                [..., 4] = reserved
        """
        dev = torch.device(self.device)
        half_x = self.map_size[0] / 2.0
        half_y = self.map_size[1] / 2.0

        self._commander_fire = commander_actions[..., 0] > 0.5  # [E, T]
        self._commander_aim[..., 0] = commander_actions[..., 1] * half_x
        self._commander_aim[..., 1] = commander_actions[..., 2] * half_y
        self._commander_aim[..., 2] = commander_actions[..., 3] * 1000.0  # z scale

    def update_aim(self):
        """Resolve aim position: commander > radar. Update laser_aim and fire_on."""
        # Commander has priority: if commander is firing, use commander aim
        # Otherwise fall back to radar comm aim (if valid)
        for t in range(self.n_teams):
            cmd_active = self._commander_fire[:, t]  # [E]
            radar_valid = self._radar_comm_valid[:, t]  # [E]

            # Use commander aim where commander is active, else radar comm
            self.laser_aim[:, t, :] = torch.where(
                cmd_active.unsqueeze(-1),
                self._commander_aim[:, t, :],
                torch.where(
                    radar_valid.unsqueeze(-1),
                    self._radar_comm_pos[:, t, :],
                    self.laser_aim[:, t, :],
                )
            )

            # Fire if commander says fire OR radar comm has valid target
            self.fire_on[:, t] = cmd_active | radar_valid

    def get_commander_obs(
        self,
        radar_pos: torch.Tensor,
        radar_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Build commander observation.

        Layout per team:
          [0:2]       own radar 0 position (x, y) / half_map
          [2:4]       own radar 1 position (x, y) / half_map
          [4:36]      radar 0 latent (32-dim)
          [36:68]     radar 1 latent (32-dim)
          [68:70]     enemy radar 0 position (x, y) / half_map
          [70:72]     enemy radar 1 position (x, y) / half_map
          [72:74]     drone laser aim (x, y) / half_map
          [74:76]     [fire_on, illumination_progress]

        Args:
            radar_pos: [E, R, 3]
            radar_latents: [E, R, 32]
        Returns:
            obs: [E, n_teams, 76]
        """
        E = self.num_envs
        dev = torch.device(self.device)
        N_in = radar_latents.shape[-1]
        half_x = self.map_size[0] / 2.0
        half_y = self.map_size[1] / 2.0
        obs_dim = 4 + 2 * N_in + 8  # 76

        obs = torch.zeros(E, self.n_teams, obs_dim, device=dev)

        for t in range(self.n_teams):
            own_idx = self.team_radar_indices[t]
            enemy_t = 1 - t
            enemy_idx = self.team_radar_indices[enemy_t]
            n_own = own_idx.shape[0]
            n_enemy = enemy_idx.shape[0]

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

            # Enemy positions
            off = 4 + 2 * N_in
            obs[:, t, off] = radar_pos[:, enemy_idx[0], 0] / half_x
            obs[:, t, off + 1] = radar_pos[:, enemy_idx[0], 1] / half_y
            if n_enemy > 1:
                obs[:, t, off + 2] = radar_pos[:, enemy_idx[1], 0] / half_x
                obs[:, t, off + 3] = radar_pos[:, enemy_idx[1], 1] / half_y
            else:
                obs[:, t, off + 2] = obs[:, t, off]
                obs[:, t, off + 3] = obs[:, t, off + 1]

            # Drone status
            obs[:, t, off + 4] = self.laser_aim[:, t, 0] / half_x
            obs[:, t, off + 5] = self.laser_aim[:, t, 1] / half_y
            obs[:, t, off + 6] = self.fire_on[:, t].float()
            obs[:, t, off + 7] = self._last_comm_crc_ok[:, t].float()

        return obs
