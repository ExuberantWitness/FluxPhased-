"""MFAR (Multi-Function Phased Array Radar) vectorized environment.

Orchestrates per-element independent control for 4 tasks:
reconnaissance, detection, jamming, communication.

Architecture: E parallel environments × R radars × N elements × 4 tasks.
Each element independently chooses its task, beam direction, and waveform.

State  = [E, R, 625×(32×N_bins+2) + 5 + 12 + L]  (FFT spectra + comm + vehicle + missile)
Action = [E, R, 13753]                              (per-element 22-dim + vehicle 3-dim)
Commander action = [E, n_teams, 3]                   (launch_flag, target_x, target_y)
"""

import time
import numpy as np
import torch

from .vec_array import VecArray
from .vec_channel import VecChannel
from .vec_interference import VecInterference
from .vec_element_processor import VecElementProcessor
from .vec_battlefield import VecBattlefield

SPEED_OF_LIGHT = 299792458.0
DEG2RAD = np.pi / 180.0

# Action layout per element (22 dims)
# [0:4]   task fractions (recon, detect, jam, comm) — argmax selects task
# [4:12]  beam steering (az, el) × 4 tasks
# [12:15] detect TX params (carrier_freq, BW, pulse_width)
# [15:18] jam TX params (BW, power, freq_shift)
# [18:22] comm TX params (carrier_freq, symbol_rate, data_X, data_Y)
ACTION_PER_ELEM = 22
# Vehicle action: [speed, heading_change, array_rotation]
ACTION_VEHICLE = 3
ACTION_TOTAL_PER_RADAR = 625 * ACTION_PER_ELEM + ACTION_VEHICLE  # 13753


class MFARVecEnv:
    """Vectorized MFAR environment with per-element independent control."""

    def __init__(
        self, num_envs: int = 2, n_radars: int = 4,
        rows: int = 25, cols: int = 25,
        fc: float = 10e9, bandwidth: float = 200e6,
        prf: float = 10e3, pulses_per_cpi: int = 32,
        n_targets: int = 1, tx_power_w: float = 1.0,
        target_rcs_dbsm: float = 20.0,
        fft_size: int = 0, symbol_rate: float = 1e6,
        commander_latent_dim: int = 0,
        n_teams: int = 2,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.n_radars = n_radars
        self.n_teams = n_teams
        self.rows = rows
        self.cols = cols
        self.n_elem = rows * cols
        self.n_pulses = pulses_per_cpi
        self.n_targets = n_targets
        self.fc = fc
        self.bandwidth = bandwidth
        self.fs = bandwidth
        self.prf = prf
        self.tx_power_w = tx_power_w
        self.target_rcs_dbsm = target_rcs_dbsm
        self.commander_latent_dim = commander_latent_dim
        self.device = device

        self.pri = 1.0 / prf
        self.n_samples = max(1, int(self.pri * self.fs))
        E, R, N = num_envs, n_radars, self.n_elem
        S = self.n_samples

        dev_torch = torch.device(device)

        # --- Subsystems ---
        self.array = VecArray(
            rows=rows, cols=cols, fc=fc,
            num_envs=num_envs, n_radars=n_radars, device=device,
        )
        self.channel = VecChannel(
            fc=fc, bandwidth=bandwidth,
            num_envs=num_envs, n_radars=n_radars,
            n_elem=N, n_samples=S, device=device,
        )
        self.interference = VecInterference(
            fc=fc, bandwidth=bandwidth, rows=rows, cols=cols,
            num_envs=num_envs, n_radars=n_radars,
            n_elem=N, device=device,
        )
        self.processor = VecElementProcessor(
            fs=self.fs, n_samples=S,
            pulses_per_cpi=pulses_per_cpi,
            fft_size=fft_size, symbol_rate=symbol_rate,
            device=device,
        )

        self.n_bins = self.processor.n_bins

        # --- Pre-allocated GPU buffers ---
        # Per-element per-pulse IQ: [E, R, N, P, S]
        self._buf_rx_signal = torch.zeros(
            E, R, N, S, dtype=torch.complex64, device=dev_torch,
        )
        self._buf_tx = torch.zeros(
            E, R, N, S, dtype=torch.complex64, device=dev_torch,
        )
        self._buf_noise = torch.zeros(
            E, R, N, S, dtype=torch.complex64, device=dev_torch,
        )
        self._buf_intf = torch.zeros(
            E, R, N, S, dtype=torch.complex64, device=dev_torch,
        )
        # CPI buffer: per-element per-pulse IQ [E, R, N, P, S]
        self._buf_cpi = torch.zeros(
            E, R, N, self.n_pulses, S, dtype=torch.complex64, device=dev_torch,
        )

        # --- State tensors ---
        self.radar_pos = torch.zeros(E, R, 3, device=dev_torch)
        self.radar_vel = torch.zeros(E, R, 3, device=dev_torch)
        self.radar_heading = torch.zeros(E, R, device=dev_torch)
        self.radar_speed = torch.zeros(E, R, device=dev_torch)
        self.array_rotation = torch.zeros(E, R, device=dev_torch)

        self.target_pos = torch.zeros(E, n_targets, 3, device=dev_torch)
        self.target_vel = torch.zeros(E, n_targets, 3, device=dev_torch)

        # Commander latent (set externally)
        self.commander_latent = torch.zeros(
            E, R, commander_latent_dim, device=dev_torch,
        )

        # Element positions (shared, computed from array geometry)
        dx_m = 0.5 * self.array.wavelength
        dy_m = 0.5 * self.array.wavelength
        x_pos = (np.arange(cols) - (cols - 1) / 2.0) * dx_m
        y_pos = (np.arange(rows) - (rows - 1) / 2.0) * dy_m
        X, Y = np.meshgrid(x_pos, y_pos)
        self.elem_x = torch.tensor(
            X.ravel().astype(np.float32), device=dev_torch,
        )
        self.elem_y = torch.tensor(
            Y.ravel().astype(np.float32), device=dev_torch,
        )

        # --- Battlefield (missile combat + game state) ---
        self.battlefield = VecBattlefield(
            num_envs=num_envs, n_radars=n_radars, n_teams=n_teams,
            fs=self.fs, symbol_rate=symbol_rate, device=device,
        )

    @property
    def state_dim(self) -> int:
        missile_dims = 6 + self.n_teams * 3  # own missile 6 + all missiles awareness 6
        return self.n_elem * (self.n_pulses * self.n_bins + 2) + 5 + missile_dims + self.commander_latent_dim

    @property
    def action_dim(self) -> int:
        return ACTION_TOTAL_PER_RADAR

    def reset(self, env_ids=None):
        """Randomize positions for specified envs (or all).

        Team layout (20km map, origin at center):
          Red (radars 0,1): y < 0 half
          Blue (radars 2,3): y > 0 half
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        E = len(env_ids)
        dev = torch.device(self.device)

        # Team-based positions: Red in y<0, Blue in y>0
        r_per_team = self.n_radars // self.n_teams
        for t in range(self.n_teams):
            r_start = t * r_per_team
            r_end = r_start + r_per_team
            n_r = r_end - r_start
            y_center = (t * 2 - 1) * 5000.0  # Red: -5000, Blue: +5000
            self.radar_pos[env_ids, r_start:r_end, 0] = (torch.rand(E, n_r, device=dev) - 0.5) * 8000.0
            self.radar_pos[env_ids, r_start:r_end, 1] = y_center + (torch.rand(E, n_r, device=dev) - 0.5) * 6000.0
            self.radar_pos[env_ids, r_start:r_end, 2] = 0.0

        self.radar_vel[env_ids] = (torch.rand(E, self.n_radars, 3, device=dev) - 0.5) * 20.0
        self.radar_heading[env_ids] = torch.rand(E, self.n_radars, device=dev) * 360.0
        self.radar_speed[env_ids] = torch.rand(E, self.n_radars, device=dev) * 8.0
        self.array_rotation[env_ids] = (torch.rand(E, self.n_radars, device=dev) - 0.5) * 120.0

        r = 5000.0 + torch.rand(E, self.n_targets, device=dev) * 10000.0
        ang = torch.rand(E, self.n_targets, device=dev) * 2 * np.pi
        self.target_pos[env_ids, :, 0] = r * torch.cos(ang)
        self.target_pos[env_ids, :, 1] = r * torch.sin(ang)
        self.target_pos[env_ids, :, 2] = 0.0
        self.target_vel[env_ids] = (torch.rand(E, self.n_targets, 3, device=dev) - 0.5) * 30.0

        self.battlefield.reset(env_ids)

    def step(self, actions: torch.Tensor = None,
             commander_actions: torch.Tensor = None) -> dict:
        """Run one CPI for all envs.

        Args:
            actions: [E, R, action_dim] float32. If None, uses default (all detect).
            commander_actions: [E, n_teams, 3] float32. If None, no missile launch.
        Returns:
            dict with keys: state, spectrum, comm_data, task_ids, timing, tx_signal,
                            commander_obs, radar_rewards, commander_rewards, dones, winners,
                            missile_pos, kills
        """
        E, R, N = self.num_envs, self.n_radars, self.n_elem
        S, P = self.n_samples, self.n_pulses
        dev = torch.device(self.device)

        t0 = time.perf_counter()

        # --- Phase 0: Commander actions (missile launch) ---
        if commander_actions is not None:
            self.battlefield.process_commander_actions(
                commander_actions, self.radar_pos,
            )

        # --- Phase 1: Decode radar actions ---
        if actions is not None:
            task_ids, beam_az, beam_el, wf_types, detect_params, jam_params, comm_params = (
                self._decode_actions(actions)
            )
            self._apply_vehicle_actions(actions)
        else:
            # Default: all elements detect with LFM, beam at 0°
            task_ids = torch.ones(E, R, N, dtype=torch.long, device=dev)
            beam_az = torch.zeros(E, R, N, device=dev)
            beam_el = torch.zeros(E, R, N, device=dev)
            wf_types = torch.zeros(E, R, N, dtype=torch.long, device=dev)
            detect_params = torch.zeros(E, R, N, 3, device=dev)
            jam_params = torch.zeros(E, R, N, 3, device=dev)
            comm_params = torch.zeros(E, R, N, 3, device=dev)

        t_action = time.perf_counter()

        # --- Phase 2: Assemble TX signals ---
        ex = self._get_elem_x()
        ey = self._get_elem_y()

        tx_signal = self.processor.assemble_tx_per_element(
            task_ids, beam_az, beam_el, wf_types,
            detect_params, jam_params, comm_params,
            ex, ey, self.array.wavelength, S,
        )
        t_tx = time.perf_counter()

        # --- Interference ---
        avg_az = beam_az.float().mean(dim=-1)
        avg_el = beam_el.float().mean(dim=-1)
        baseband = tx_signal.mean(dim=2)
        self._buf_intf.zero_()
        weights_for_intf = self.array.steer_all(avg_az, avg_el)
        self.interference.compute(
            self.radar_pos, avg_az, avg_el,
            weights_for_intf, baseband[:, 0, :],
            out=self._buf_intf,
        )

        t_intf = time.perf_counter()

        # --- Phase 3: CPI pulse loop (with missile targets) ---
        waveform_refs = self._build_waveform_refs(task_ids, wf_types, detect_params, comm_params)

        missile = self.battlefield.missile
        for p in range(P):
            self._buf_rx_signal.zero_()

            # Static targets
            for t_idx in range(self.n_targets):
                delay_s, doppler_hz, gain = self.channel.compute_params_batch(
                    self.radar_pos, self.radar_vel,
                    self.target_pos[:, t_idx], self.target_vel[:, t_idx],
                    tx_power_w=self.tx_power_w,
                    rcs_dbsm=self.target_rcs_dbsm,
                    array_directivity_db=self.array.directivity_db,
                )
                target_return = self.channel.apply_batch(
                    tx_signal, delay_s, doppler_hz, gain,
                )
                self._buf_rx_signal += target_return

            # Missile targets (in-flight missiles visible to all radars)
            for team_idx in range(self.n_teams):
                flying = missile.in_flight[:, team_idx]
                if not flying.any():
                    continue
                m_pos = missile.missile_pos[:, team_idx]
                m_vel = missile.missile_vel[:, team_idx]
                delay_s, doppler_hz, gain = self.channel.compute_params_batch(
                    self.radar_pos, self.radar_vel, m_pos, m_vel,
                    tx_power_w=self.tx_power_w,
                    rcs_dbsm=missile.rcs_dbsm,
                    array_directivity_db=self.array.directivity_db,
                )
                gain = gain * flying.float().unsqueeze(1)
                target_return = self.channel.apply_batch(
                    tx_signal, delay_s, doppler_hz, gain,
                )
                self._buf_rx_signal += target_return

            self._buf_rx_signal += self._buf_intf
            self.channel.generate_noise(out=self._buf_noise)
            self._buf_rx_signal += self._buf_noise
            self._buf_cpi[:, :, :, p, :] = self._buf_rx_signal

        t_pulses = time.perf_counter()

        # --- Phase 4: RX processing (unchanged) ---
        spectrum = torch.zeros(
            E, R, N, P, self.n_bins, dtype=torch.float32, device=dev,
        )
        comm_data = torch.zeros(E, R, N, 2, dtype=torch.float32, device=dev)

        recon_mask = (task_ids == 0)
        if recon_mask.any():
            recon_spec = self.processor.process_rx_cpi(
                self._buf_cpi, waveform_ref=None,
            )
            spectrum = torch.where(
                recon_mask.unsqueeze(-1).unsqueeze(-1).expand_as(spectrum),
                recon_spec, spectrum,
            )

        detect_mask = (task_ids == 1)
        if detect_mask.any():
            det_ref = self._build_detect_ref(wf_types, detect_params)
            det_spec = self.processor.process_rx_cpi(
                self._buf_cpi, waveform_ref=det_ref,
            )
            spectrum = torch.where(
                detect_mask.unsqueeze(-1).unsqueeze(-1).expand_as(spectrum),
                det_spec, spectrum,
            )

        comm_mask = (task_ids == 3)
        if comm_mask.any():
            comm_ref = self._build_comm_ref(comm_params)
            comm_spec = self.processor.process_rx_cpi(
                self._buf_cpi, waveform_ref=comm_ref,
            )
            spectrum = torch.where(
                comm_mask.unsqueeze(-1).unsqueeze(-1).expand_as(spectrum),
                comm_spec, spectrum,
            )
            comm_xy = self.processor.process_rx_comm(self._buf_cpi, comm_ref)
            comm_data = torch.where(
                comm_mask.unsqueeze(-1).expand_as(comm_data),
                comm_xy, comm_data,
            )

        t_rx = time.perf_counter()

        # --- Phase 5: Radar→missile BPSK communication ---
        self.battlefield.process_missile_comm(
            self.radar_pos, task_ids, comm_params, tx_signal,
            self.channel, self.processor,
        )

        # --- Phase 6: Missile physics + kill check ---
        dt = self.pri * self.n_pulses
        kills = self.battlefield.step_missiles(dt, self.radar_pos)
        self.battlefield.update_alive(kills)
        dones, winners = self.battlefield.check_win()
        rewards = self.battlefield.compute_rewards(kills, dones, winners)

        t_missile = time.perf_counter()

        # --- Phase 7: Assemble state ---
        state = self._assemble_state(spectrum, comm_data)
        commander_obs = self.battlefield.get_commander_observation(
            spectrum, comm_data, self.radar_pos, self.radar_vel,
            self.radar_heading, self.radar_speed,
        )

        timing = {
            "action_ms": (t_action - t0) * 1000,
            "tx_ms": (t_tx - t_action) * 1000,
            "interference_ms": (t_intf - t_tx) * 1000,
            "pulses_ms": (t_pulses - t_intf) * 1000,
            "rx_ms": (t_rx - t_pulses) * 1000,
            "missile_ms": (t_missile - t_rx) * 1000,
            "total_ms": (t_missile - t0) * 1000,
        }

        return {
            "state": state,
            "spectrum": spectrum,
            "comm_data": comm_data,
            "task_ids": task_ids,
            "commander_obs": commander_obs,          # [E, n_teams, 31]
            "radar_rewards": rewards["radar_rewards"],       # [E, R]
            "commander_rewards": rewards["commander_rewards"],  # [E, n_teams]
            "dones": dones,                           # [E] bool
            "winners": winners,                       # [E] long
            "missile_pos": missile.missile_pos,       # [E, n_teams, 3]
            "kills": kills,                           # [E, n_teams, n_enemy]
            "timing": timing,
            "tx_signal": tx_signal,
        }

    def _decode_actions(self, actions: torch.Tensor):
        """Decode [E, R, 13753] actions into per-element parameters."""
        E, R = actions.shape[:2]
        N = self.n_elem
        dev = torch.device(self.device)

        # Per-element actions: [E, R, 13753] → [E, R, N, 22]
        elem_actions = actions[:, :, :N * ACTION_PER_ELEM].reshape(E, R, N, ACTION_PER_ELEM)

        # Task assignment: argmax of first 4 dims
        task_ids = elem_actions[..., 0:4].argmax(dim=-1)  # [E, R, N]

        # Beam steering per element: select (az, el) for the assigned task
        # [4:12] = (az, el) × 4 tasks — pick the one matching task_id
        all_az = elem_actions[..., 4:12:2]  # [E, R, N, 4]
        all_el = elem_actions[..., 5:12:2]  # [E, R, N, 4]
        task_idx = task_ids.unsqueeze(-1)     # [E, R, N, 1]
        beam_az = all_az.gather(-1, task_idx).squeeze(-1) * 60.0   # scale to ±60°
        beam_el = all_el.gather(-1, task_idx).squeeze(-1) * 45.0   # scale to ±45°

        # Waveform type (shared across tasks, used as index)
        wf_types = (elem_actions[..., 4:6].argmax(dim=-1)).long()  # simplified

        # Task-specific params
        detect_params = elem_actions[..., 12:15]  # [E, R, N, 3]
        jam_params = elem_actions[..., 15:18]     # [E, R, N, 3]
        comm_params = elem_actions[..., 18:22]    # [E, R, N, 4] → take first 3

        return task_ids, beam_az, beam_el, wf_types, detect_params, jam_params, comm_params[..., :3]

    def _apply_vehicle_actions(self, actions: torch.Tensor):
        """Update vehicle state from action's last 3 dims."""
        E, R = actions.shape[:2]
        vehicle_action = actions[:, :, -ACTION_VEHICLE:]  # [E, R, 3]
        self.radar_speed = vehicle_action[..., 0].clamp(0, 8.33) * 8.33
        self.radar_heading = (self.radar_heading + vehicle_action[..., 1] * 60.0) % 360.0
        self.array_rotation = (self.array_rotation + vehicle_action[..., 2] * 60.0) % 360.0

    def _build_waveform_refs(self, task_ids, wf_types, detect_params, comm_params):
        """Build waveform reference tensors for matched filtering."""
        # Returns dict mapping task_id → reference waveform tensor
        # For simplicity, use LFM for all detect elements
        E, R, N = task_ids.shape
        dev = torch.device(self.device)

        detect_mask = (task_ids == 1)
        if detect_mask.any():
            # Generate a default LFM reference
            from .waveform_gpu import generate_lfm
            ref = generate_lfm(
                self.pri * 0.1, self.bandwidth, self.fs, dev, "up",
            )
            # Pad to n_samples
            if ref.shape[0] < self.n_samples:
                pad = torch.zeros(
                    self.n_samples - ref.shape[0],
                    dtype=torch.complex64, device=dev,
                )
                ref = torch.cat([ref, pad])
            ref = ref[:self.n_samples]
            return ref
        return None

    def _build_detect_ref(self, wf_types, detect_params):
        """Build matched filter reference for detection elements."""
        E, R, N = wf_types.shape
        dev = torch.device(self.device)

        from .waveform_gpu import generate_lfm
        ref = generate_lfm(self.pri * 0.1, self.bandwidth, self.fs, dev, "up")
        if ref.shape[0] < self.n_samples:
            pad = torch.zeros(
                self.n_samples - ref.shape[0],
                dtype=torch.complex64, device=dev,
            )
            ref = torch.cat([ref, pad])
        return ref[:self.n_samples]

    def _build_comm_ref(self, comm_params):
        """Build BPSK reference for communication elements."""
        E, R, N = comm_params.shape[:3]
        dev = torch.device(self.device)

        from .waveform_gpu import encode_bpsk, modulate_bpsk
        bits = encode_bpsk(0.0, 0.0)
        ref = modulate_bpsk(bits, self.n_samples, self.fs, self.processor.symbol_rate, dev)
        return ref

    def _assemble_state(self, spectrum, comm_data):
        """Assemble state vector from spectrum, comm data, vehicle, and missile awareness.

        Args:
            spectrum: [E, R, N, P, n_bins] float32
            comm_data: [E, R, N, 2] float32
        Returns:
            [E, R, state_dim] float32
        """
        E, R, N, P, B = spectrum.shape

        spec_flat = spectrum.reshape(E, R, N * P * B)
        comm_flat = comm_data.reshape(E, R, N * 2)

        vehicle = torch.stack([
            self.radar_pos[..., 0],
            self.radar_pos[..., 1],
            self.radar_heading,
            self.radar_speed,
            self.array_rotation,
        ], dim=-1)

        missile_state = self._build_missile_state_per_radar()

        parts = [spec_flat, comm_flat, vehicle, missile_state]
        if self.commander_latent_dim > 0:
            parts.append(self.commander_latent)

        return torch.cat(parts, dim=-1)

    def _build_missile_state_per_radar(self) -> torch.Tensor:
        """Build per-radar missile awareness tensor [E, R, 6 + n_teams*3].

        Per radar:
          [0:6] own team missile: pos_x, pos_y, pos_z, in_flight, target_x, target_y
          [6:6+n_teams*3] all missiles: pos_x, pos_y, in_flight per team
        """
        E, R = self.num_envs, self.n_radars
        n_t = self.n_teams
        dev = torch.device(self.device)
        m = self.battlefield.missile

        result = torch.zeros(E, R, 6 + n_t * 3, device=dev)

        for r in range(R):
            team = self.battlefield.team_id[r].item()
            # Own team missile
            result[:, r, 0] = m.missile_pos[:, team, 0]
            result[:, r, 1] = m.missile_pos[:, team, 1]
            result[:, r, 2] = m.missile_pos[:, team, 2]
            result[:, r, 3] = m.in_flight[:, team].float()
            result[:, r, 4] = m.target_pos[:, team, 0]
            result[:, r, 5] = m.target_pos[:, team, 1]
            # All missiles awareness
            for t in range(n_t):
                base = 6 + t * 3
                result[:, r, base] = m.missile_pos[:, t, 0]
                result[:, r, base + 1] = m.missile_pos[:, t, 1]
                result[:, r, base + 2] = m.in_flight[:, t].float()

        return result

    def _get_elem_x(self):
        return self.elem_x

    def _get_elem_y(self):
        return self.elem_y
