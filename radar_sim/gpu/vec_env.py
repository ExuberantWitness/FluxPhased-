"""Vectorized radar simulation environment — num_envs parallel CPI simulations.

Orchestrates: Waveform → Beam Steering (batched Warp) → Channel (batched Warp)
→ Interference (torch) → RX Beamforming (torch) → Receiver (torch FFT + batched Warp CFAR).

Pre-allocates all GPU buffers in __init__; step() overwrites them in-place.
"""

import time
import numpy as np
import warp as wp
import torch

from .vec_array import VecArray
from .vec_channel import VecChannel
from .vec_receiver import VecReceiver
from .vec_interference import VecInterference
from .waveform_gpu import WaveformGeneratorGPU

SPEED_OF_LIGHT = 299792458.0


class RadarSimVecEnv:
    """Batched radar simulation with num_envs parallel environments."""

    def __init__(
        self, num_envs: int = 10, n_radars: int = 4,
        rows: int = 25, cols: int = 25,
        fc: float = 10e9, bandwidth: float = 200e6,
        prf: float = 10e3, pulses_per_cpi: int = 32,
        n_targets: int = 1, tx_power_w: float = 1.0,
        target_rcs_dbsm: float = 20.0, device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.n_radars = n_radars
        self.n_elem = rows * cols
        self.n_pulses = pulses_per_cpi
        self.n_targets = n_targets
        self.fc = fc
        self.bandwidth = bandwidth
        self.fs = bandwidth
        self.prf = prf
        self.tx_power_w = tx_power_w
        self.target_rcs_dbsm = target_rcs_dbsm
        self.device = device

        self.pri = 1.0 / prf
        self.n_samples = max(1, int(self.pri * self.fs))
        S = self.n_samples
        E, R, N, P = num_envs, n_radars, rows * cols, pulses_per_cpi

        # VRAM check: rx_signal + tx + noise + intf + pulse_train + overhead
        vram_bytes = E * R * S * 8 * (N * 4 + P * 2)
        vram_available = torch.cuda.get_device_properties(0).total_memory * 0.85
        if vram_bytes > vram_available:
            raise MemoryError(
                f"Need ~{vram_bytes/1e9:.1f} GB, have {vram_available/1e9:.1f} GB. "
                f"Reduce num_envs, rows, cols, or pulses_per_cpi."
            )

        dev_torch = torch.device(device)

        # Pre-allocated buffers
        self._buf_rx_signal = torch.zeros(
            E, R, N, S, dtype=torch.complex64, device=dev_torch,
        )
        self._buf_pulse_train = torch.zeros(
            E, R, P, S, dtype=torch.complex64, device=dev_torch,
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

        # State tensors
        self.radar_pos = torch.zeros(E, R, 3, device=dev_torch)
        self.radar_vel = torch.zeros(E, R, 3, device=dev_torch)
        self.target_pos = torch.zeros(E, n_targets, 3, device=dev_torch)
        self.target_vel = torch.zeros(E, n_targets, 3, device=dev_torch)
        self.beam_az = torch.zeros(E, R, device=dev_torch)
        self.beam_el = torch.zeros(E, R, device=dev_torch)

        # Subsystems
        self.array = VecArray(
            rows=rows, cols=cols, fc=fc,
            num_envs=num_envs, n_radars=n_radars, device=device,
        )
        self.channel = VecChannel(
            fc=fc, bandwidth=bandwidth,
            num_envs=num_envs, n_radars=n_radars,
            n_elem=N, n_samples=S, device=device,
        )
        self.receiver = VecReceiver(
            fc=fc, bandwidth=bandwidth, prf=prf,
            pulses_per_cpi=pulses_per_cpi,
            num_envs=num_envs, n_radars=n_radars, device=device,
        )
        self.interference = VecInterference(
            fc=fc, bandwidth=bandwidth, rows=rows, cols=cols,
            num_envs=num_envs, n_radars=n_radars,
            n_elem=N, device=device,
        )
        rf_cfg = type('RFConfig', (), {'fs': self.fs, 'bandwidth': self.bandwidth})()
        wf_cfg = type('WFConfig', (), {})()
        cpi_cfg = type('CPIConfig', (), {'prf': self.prf, 'pulses_per_cpi': self.n_pulses})()
        self.waveform_gen = WaveformGeneratorGPU(rf_cfg, wf_cfg, cpi_cfg, dev_torch)

    def reset(self, env_ids=None):
        """Randomize positions for specified envs (or all if None).

        Radar positions: random within 10km×10km.
        Target positions: random within 5-15km from origin.
        Beam directions: random within ±60°.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        E = len(env_ids)

        # Random radar positions in 10km box
        self.radar_pos[env_ids] = torch.rand(E, self.n_radars, 3, device=torch.device(self.device)) * 10000.0
        self.radar_vel[env_ids] = (torch.rand(E, self.n_radars, 3, device=torch.device(self.device)) - 0.5) * 20.0

        # Random targets at 5-15km
        r = 5000.0 + torch.rand(E, self.n_targets, device=torch.device(self.device)) * 10000.0
        ang = torch.rand(E, self.n_targets, device=torch.device(self.device)) * 2 * np.pi
        self.target_pos[env_ids, :, 0] = r * torch.cos(ang)
        self.target_pos[env_ids, :, 1] = r * torch.sin(ang)
        self.target_pos[env_ids, :, 2] = 0.0
        self.target_vel[env_ids] = (torch.rand(E, self.n_targets, 3, device=torch.device(self.device)) - 0.5) * 30.0

        # Random beam directions
        self.beam_az[env_ids] = (torch.rand(E, self.n_radars, device=torch.device(self.device)) - 0.5) * 120.0
        self.beam_el[env_ids] = (torch.rand(E, self.n_radars, device=torch.device(self.device)) - 0.5) * 60.0

    def step(self, actions=None):
        """Run one CPI for all envs.

        Args:
            actions: [E, R, action_dim] (optional, overrides beam_az/el)
        Returns:
            dict with keys: detections, rd_power, pulse_train, timing
        """
        E, R, N, S, P = (
            self.num_envs, self.n_radars, self.n_elem,
            self.n_samples, self.n_pulses,
        )
        t0 = time.perf_counter()

        # Decode actions → beam directions
        if actions is not None:
            self.beam_az.copy_(actions[..., 0].clamp(-60, 60))
            self.beam_el.copy_(actions[..., 1].clamp(-45, 45))

        # Generate shared baseband waveform. Keep the compact pulse for
        # matched filtering, but zero-pad to full PRI for pulse simulation.
        pulse_compact = self.waveform_gen.generate("lfm_up", {
            "pulse_width": self.pri * 0.1, "bandwidth": self.bandwidth,
        })
        n_pulse = pulse_compact.shape[0]
        if n_pulse < S:
            pad = torch.zeros(S - n_pulse, dtype=torch.complex64,
                              device=torch.device(self.device))
            baseband = torch.cat([pulse_compact, pad])
        else:
            baseband = pulse_compact[:S]
        # Use compact pulse for matched filter reference (much smaller n_fft)
        mf_ref_waveform = pulse_compact[:min(n_pulse, S)]

        t_wf = time.perf_counter()

        # Beam steering (batched Warp kernel)
        weights = self.array.steer_all(self.beam_az, self.beam_el)  # [E, R, N] complex64
        t_steer = time.perf_counter()

        # Interference (vectorized torch, writes into pre-allocated buffer)
        self.interference.compute(
            self.radar_pos, self.beam_az, self.beam_el, weights, baseband,
            out=self._buf_intf,
        )
        t_intf = time.perf_counter()

        # Pre-compute baseband broadcast view (no allocation)
        bb_view = baseband.view(1, 1, 1, -1)

        # CPI pulse loop
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
                # TX beamforming: write into pre-allocated _buf_tx
                torch.multiply(
                    weights.unsqueeze(-1), bb_view, out=self._buf_tx,
                )
                # Channel: apply delay/Doppler/gain
                target_return = self.channel.apply_batch(
                    self._buf_tx, delay_s, doppler_hz, gain,
                )
                self._buf_rx_signal += target_return

            self._buf_rx_signal += self._buf_intf
            # Noise: write in-place into pre-allocated buffer
            self.channel.generate_noise(out=self._buf_noise)
            self._buf_rx_signal += self._buf_noise
            # RX beamforming: [E,R,N,S] → [E,R,S]
            self._buf_pulse_train[:, :, p, :] = self.array.beamform_rx(
                weights, self._buf_rx_signal,
            )

        t_pulses = time.perf_counter()

        # Receiver processing (batched FFT + batched Warp CFAR).
        # Pass the compact LFM pulse so n_fft stays small.
        detections = self.receiver.process_all(self._buf_pulse_train, mf_ref_waveform)
        t_rx = time.perf_counter()

        timing = {
            "waveform_ms": (t_wf - t0) * 1000,
            "steer_ms": (t_steer - t_wf) * 1000,
            "interference_ms": (t_intf - t_steer) * 1000,
            "pulses_ms": (t_pulses - t_intf) * 1000,
            "receiver_ms": (t_rx - t_pulses) * 1000,
            "total_ms": (t_rx - t0) * 1000,
        }

        return {
            "detections": detections,
            "pulse_train": self._buf_pulse_train,
            "timing": timing,
            "weights": weights,
        }
