"""MFAR (Multi-Function Phased Array Radar) vectorized environment.

Orchestrates per-element independent control for 4 tasks:
reconnaissance, detection, jamming, communication.

Architecture: E parallel environments × R radars × N elements × 4 tasks.
Each element independently chooses its task, beam direction, and waveform.

State  = [E, R, 625×(32×N_bins+2) + 5 + L]  (FFT spectra + comm data + vehicle)
Action = [E, R, 13753]                        (per-element 22-dim + vehicle 3-dim)
"""

import time
import numpy as np
import torch

from .vec_array import VecArray
from .vec_channel import VecChannel
from .vec_interference import VecInterference
from .vec_element_processor import VecElementProcessor

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
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.n_radars = n_radars
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

    @property
    def state_dim(self) -> int:
        return self.n_elem * (self.n_pulses * self.n_bins + 2) + 5 + self.commander_latent_dim

    @property
    def action_dim(self) -> int:
        return ACTION_TOTAL_PER_RADAR

    def reset(self, env_ids=None):
        """Randomize positions for specified envs (or all)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        E = len(env_ids)
        dev = torch.device(self.device)

        self.radar_pos[env_ids] = torch.rand(E, self.n_radars, 3, device=dev) * 10000.0
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

    def step(self, actions: torch.Tensor = None) -> dict:
        """Run one CPI for all envs.

        Args:
            actions: [E, R, action_dim] float32. If None, uses default (all detect).
        Returns:
            dict with keys: state, timing, info
        """
        E, R, N = self.num_envs, self.n_radars, self.n_elem
        S, P = self.n_samples, self.n_pulses
        dev = torch.device(self.device)

        t0 = time.perf_counter()

        # --- Decode actions ---
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

        # --- Assemble TX signals ---
        # Get element positions as torch tensors
        ex = self._get_elem_x()
        ey = self._get_elem_y()

        tx_signal = self.processor.assemble_tx_per_element(
            task_ids, beam_az, beam_el, wf_types,
            detect_params, jam_params, comm_params,
            ex, ey, self.array.wavelength, S,
        )
        t_tx = time.perf_counter()

        # --- Interference ---
        # Simplified: use averaged beam directions for interference calc
        avg_az = beam_az.float().mean(dim=-1)  # [E, R]
        avg_el = beam_el.float().mean(dim=-1)
        # Use averaged TX signal per radar for interference model
        baseband = tx_signal.mean(dim=2)  # [E, R, S]
        self._buf_intf.zero_()
        # Interference computation (simplified for per-element version)
        weights_for_intf = self.array.steer_all(avg_az, avg_el)
        self.interference.compute(
            self.radar_pos, avg_az, avg_el,
            weights_for_intf, baseband[:, 0, :],
            out=self._buf_intf,
        )

        t_intf = time.perf_counter()

        # --- CPI pulse loop ---
        # Store waveform references for matched filtering
        # (elements with same task+wf_type share the same reference)
        waveform_refs = self._build_waveform_refs(task_ids, wf_types, detect_params, comm_params)

        for p in range(P):
            self._buf_rx_signal.zero_()

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

            self._buf_rx_signal += self._buf_intf
            self.channel.generate_noise(out=self._buf_noise)
            self._buf_rx_signal += self._buf_noise

            # Store per-element RX for this pulse
            self._buf_cpi[:, :, :, p, :] = self._buf_rx_signal

        t_pulses = time.perf_counter()

        # --- RX processing: FFT magnitude spectrum ---
        # Process per task: detect/comm get matched filter, recon doesn't, jam gets zeros
        spectrum = torch.zeros(
            E, R, N, P, self.n_bins, dtype=torch.float32, device=dev,
        )
        comm_data = torch.zeros(E, R, N, 2, dtype=torch.float32, device=dev)

        # Recon: direct FFT (no matched filter)
        recon_mask = (task_ids == 0)
        if recon_mask.any():
            recon_iq = self._buf_cpi[recon_mask]  # [?, P, S]
            recon_spec = self.processor.process_rx_cpi(
                self._buf_cpi, waveform_ref=None,
            )
            spectrum = torch.where(
                recon_mask.unsqueeze(-1).unsqueeze(-1).expand_as(spectrum),
                recon_spec, spectrum,
            )

        # Detect: FFT with matched filter
        detect_mask = (task_ids == 1)
        if detect_mask.any():
            # Build per-element waveform reference for detection
            det_ref = self._build_detect_ref(wf_types, detect_params)
            det_spec = self.processor.process_rx_cpi(
                self._buf_cpi, waveform_ref=det_ref,
            )
            spectrum = torch.where(
                detect_mask.unsqueeze(-1).unsqueeze(-1).expand_as(spectrum),
                det_spec, spectrum,
            )

        # Jam: zeros (TX-only)

        # Comm: FFT + BPSK demod
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
            # BPSK demod
            comm_xy = self.processor.process_rx_comm(self._buf_cpi, comm_ref)
            comm_data = torch.where(
                comm_mask.unsqueeze(-1).expand_as(comm_data),
                comm_xy, comm_data,
            )

        t_rx = time.perf_counter()

        # --- Assemble state vector ---
        state = self._assemble_state(spectrum, comm_data)

        timing = {
            "action_ms": (t_action - t0) * 1000,
            "tx_ms": (t_tx - t_action) * 1000,
            "interference_ms": (t_intf - t_tx) * 1000,
            "pulses_ms": (t_pulses - t_intf) * 1000,
            "rx_ms": (t_rx - t_pulses) * 1000,
            "total_ms": (t_rx - t0) * 1000,
        }

        return {
            "state": state,
            "spectrum": spectrum,       # [E, R, N, P, n_bins]
            "comm_data": comm_data,     # [E, R, N, 2]
            "task_ids": task_ids,       # [E, R, N]
            "timing": timing,
            "tx_signal": tx_signal,     # [E, R, N, S]
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
        """Assemble state vector from spectrum and comm data.

        Args:
            spectrum: [E, R, N, P, n_bins] float32
            comm_data: [E, R, N, 2] float32
        Returns:
            [E, R, state_dim] float32
        """
        E, R, N, P, B = spectrum.shape

        # Flatten spectrum: [E, R, N*P*B]
        spec_flat = spectrum.reshape(E, R, N * P * B)

        # Flatten comm data: [E, R, N*2]
        comm_flat = comm_data.reshape(E, R, N * 2)

        # Vehicle state: [E, R, 5]
        vehicle = torch.stack([
            self.radar_pos[..., 0],   # x
            self.radar_pos[..., 1],   # y
            self.radar_heading,       # heading
            self.radar_speed,         # speed
            self.array_rotation,      # array_rotation
        ], dim=-1)

        # Commander latent: [E, R, L]
        parts = [spec_flat, comm_flat, vehicle]
        if self.commander_latent_dim > 0:
            parts.append(self.commander_latent)

        return torch.cat(parts, dim=-1)

    def _get_elem_x(self):
        return self.elem_x

    def _get_elem_y(self):
        return self.elem_y
