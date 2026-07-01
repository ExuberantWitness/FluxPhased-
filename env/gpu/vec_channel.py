"""Batched per-element channel simulation for num_envs parallel environments.

Warp kernel for delay/Doppler/gain (batched across env×radar×element)
+ torch noise generation. Uses wp.from_torch for zero-copy.
"""

import numpy as np
import warp as wp
import torch
from typing import Optional

SPEED_OF_LIGHT = 299792458.0


@wp.kernel
def _delay_doppler_batched(
    signal_in: wp.array2d(dtype=wp.float32),       # [E*R*N, 2*S]
    delay_samples: wp.array1d(dtype=wp.float32),   # [E*R]
    doppler_phase_step: wp.array1d(dtype=wp.float32),  # [E*R]
    path_gain: wp.array1d(dtype=wp.float32),       # [E*R]
    n_elem: wp.int32,
    n_samples: wp.int32,
    signal_out: wp.array2d(dtype=wp.float32),      # [E*R*N, 2*S]
):
    """Apply delay, Doppler, and gain to per-element IQ signal.

    flat = wp.tid() in [0, E*R*N).  er_idx = flat / N identifies the env-radar pair.
    Each thread processes all samples for one element.
    """
    flat = wp.tid()
    er_idx = flat // n_elem

    g = path_gain[er_idx]
    dstep = doppler_phase_step[er_idx]
    d_int = wp.int32(delay_samples[er_idx])

    for s in range(n_samples):
        src = s - d_int
        if src >= 0 and src < n_samples:
            s_re = signal_in[flat, 2 * src]
            s_im = signal_in[flat, 2 * src + 1]
        else:
            s_re = wp.float32(0.0)
            s_im = wp.float32(0.0)

        phase = wp.float32(s) * dstep
        cos_p = wp.cos(phase)
        sin_p = wp.sin(phase)

        out_re = g * (s_re * cos_p - s_im * sin_p)
        out_im = g * (s_re * sin_p + s_im * cos_p)

        signal_out[flat, 2 * s] = out_re
        signal_out[flat, 2 * s + 1] = out_im


class VecChannel:
    """Batched channel for E parallel environments, each with R radars.

    Pre-allocates output buffer; applies delay/Doppler/gain via batched Warp kernel.
    Noise generated via torch.randn directly on GPU.
    """

    def __init__(
        self, fc: float = 10e9, bandwidth: float = 200e6,
        num_envs: int = 10, n_radars: int = 4,
        n_elem: int = 625, n_samples: int = 20000,
        noise_figure_db: float = 5.0, device: str = "cuda",
    ):
        self.fc = fc
        self.bandwidth = bandwidth
        self.fs = bandwidth
        self.wavelength = SPEED_OF_LIGHT / fc
        self.num_envs = num_envs
        self.n_radars = n_radars
        self.n_elem = n_elem
        self.n_samples = n_samples
        self.device = device

        noise_power_dbm = (
            10 * np.log10(1.380649e-23 * 290 * 1000)
            + 10 * np.log10(bandwidth)
            + noise_figure_db
        )
        self.noise_power_linear = 10.0 ** (noise_power_dbm / 10.0) * 1e-3
        self.noise_std = float(np.sqrt(self.noise_power_linear / 2.0))

        ER = num_envs * n_radars
        # Pre-allocated scalar parameters for Warp kernels
        self._delay_buf = torch.zeros(ER, dtype=torch.float32, device=torch.device(device))
        self._doppler_phase_buf = torch.zeros(ER, dtype=torch.float32, device=torch.device(device))
        self._gain_buf = torch.zeros(ER, dtype=torch.float32, device=torch.device(device))
        # Pre-allocated output buffer (float32 interleaved)
        self._out_buf = torch.zeros(
            ER * n_elem, 2 * n_samples, dtype=torch.float32,
            device=torch.device(device),
        )

    def compute_params_batch(
        self, radar_pos: torch.Tensor, radar_vel: torch.Tensor,
        tgt_pos: torch.Tensor, tgt_vel: torch.Tensor,
        tx_power_w: float = 1.0, rcs_dbsm: float = 20.0,
        array_directivity_db: float = 44.0, system_loss_db: float = 3.0,
        n_elem: int = 625,
        beam_az: Optional[torch.Tensor] = None,
        beam_el: Optional[torch.Tensor] = None,
        array_rotation: Optional[torch.Tensor] = None,
        bw_az_deg: float = 4.06, bw_el_deg: float = 4.06,
    ):
        """Compute two-way channel parameters via the standard radar equation.

        Standard monostatic radar equation (dBm):
          Pr = Pt + 2G + σ + 20·log10(λ) - 30·log10(4π) - 40·log10(R) - Lsys

        Per-element gain for ELDA (voltage amplitude):
          gain = sqrt(Pr_beamformed / N_elem)

        The per-element gain produces correct beamformed-level SNR when the
        RL agent processes all N element spectra.

        Optional beam-aware gain: if beam_az/beam_el are provided, the
        array_directivity_db is reduced by an off-boresight loss based on
        angular separation between world-space beam direction and target.

        Args:
            radar_pos:       [E, R, 3]
            radar_vel:       [E, R, 3]
            tgt_pos:         [E, 3] single target position per env
            tgt_vel:         [E, 3]
            tx_power_w, rcs_dbsm, array_directivity_db, system_loss_db: scalars
            n_elem:          number of array elements (for per-element gain division)
            beam_az:         [E, R] mean array-local azimuth (deg) or None
            beam_el:         [E, R] mean array-local elevation (deg) or None
            array_rotation:  [E, R] array rotation angle (deg) or None
            bw_az_deg:       3dB beamwidth azimuth (deg) for off-boresight loss
            bw_el_deg:       3dB beamwidth elevation (deg) for off-boresight loss
        Returns:
            delay_samples: [E, R] float32 (round-trip)
            doppler_hz:    [E, R] float32 (two-way)
            gain_linear:   [E, R] float32 (per-element voltage amplitude)
        """
        rel = tgt_pos.unsqueeze(1) - radar_pos           # [E, R, 3]
        distance = rel.norm(dim=-1).clamp(min=1.0)       # [E, R]

        delay_s = 2.0 * distance / SPEED_OF_LIGHT
        delay_samples = delay_s * self.fs

        rel_vel = radar_vel - tgt_vel.unsqueeze(1)       # [E, R, 3]
        radial_vel = (rel_vel * rel).sum(dim=-1) / distance
        doppler_hz = 2.0 * radial_vel * self.fc / SPEED_OF_LIGHT

        # Compute beam-dependent gain if beam direction is provided
        tx_gain_db = torch.full_like(distance, array_directivity_db)
        if beam_az is not None and beam_el is not None:
            # Target world-frame direction from radar
            tgt_world_az_rad = torch.atan2(rel[..., 1], rel[..., 0])  # [E, R]
            tgt_world_el_rad = torch.atan2(
                rel[..., 2], torch.sqrt(rel[..., 0]**2 + rel[..., 1]**2).clamp(min=1.0),
            )  # [E, R]

            # Beam world-frame direction (array-local + array rotation)
            rot = array_rotation * (np.pi / 180.0) if array_rotation is not None else 0.0
            world_beam_az_rad = beam_az * (np.pi / 180.0) + rot
            world_beam_el_rad = beam_el * (np.pi / 180.0)

            # Off-boresight angles (wrap azimuth to ±π)
            d_az_rad = world_beam_az_rad - tgt_world_az_rad
            d_az_rad = torch.atan2(torch.sin(d_az_rad), torch.cos(d_az_rad))
            d_el_rad = world_beam_el_rad - tgt_world_el_rad

            d_az_deg = d_az_rad * (180.0 / np.pi)
            d_el_deg = d_el_rad * (180.0 / np.pi)

            # Gaussian beam pattern: -3 dB at BW edge
            loss_db = (-3.0 * (d_az_deg / bw_az_deg)**2
                       - 3.0 * (d_el_deg / bw_el_deg)**2)
            tx_gain_db = array_directivity_db + loss_db

        # Standard monostatic radar equation in dB form:
        # Pr = Pt + 2G + σ + 20·log10(λ) - 30·log10(4π) - 40·log10(R) - Lsys
        rx_power_dbm = (
            10.0 * np.log10(tx_power_w * 1000.0)         # Pt (dBm)
            + 2.0 * tx_gain_db                              # 2G (beam-dependent)
            + rcs_dbsm                                     # σ
            + 20.0 * np.log10(self.wavelength)             # λ² term
            - 30.0 * np.log10(4.0 * np.pi)                 # (4π)³ term
            - 40.0 * torch.log10(distance)                  # R⁴ term
            - system_loss_db                                # Lsys
        )
        rx_power_w = 10.0 ** ((rx_power_dbm - 30.0) / 10.0)
        # Per-element voltage gain for ELDA: sqrt(Pr_beam / N_elem)
        gain_linear = torch.sqrt((rx_power_w / n_elem).clamp(min=0.0))

        return delay_samples, doppler_hz, gain_linear

    def apply_batch(
        self, signal: torch.Tensor,
        delay_samples: torch.Tensor, doppler_hz: torch.Tensor,
        gain_linear: torch.Tensor,
    ) -> torch.Tensor:
        """Apply channel effects to batched per-element TX signal.

        Args:
            signal:        [E, R, N, S] complex64
            delay_samples: [E, R] float32
            doppler_hz:    [E, R] float32
            gain_linear:   [E, R] float32
        Returns:
            [E, R, N, S] complex64 (view into pre-allocated output buffer)
        """
        E, R, N, S = self.num_envs, self.n_radars, self.n_elem, signal.shape[-1]

        # Copy scalar params into pre-allocated buffers
        self._delay_buf.copy_(delay_samples.reshape(-1))
        self._doppler_phase_buf.copy_(
            (2.0 * np.pi * doppler_hz / self.fs).reshape(-1),
        )
        self._gain_buf.copy_(gain_linear.reshape(-1))

        # Convert signal to interleaved float32 (zero-copy views)
        signal_2d = signal.reshape(E * R * N, S)
        signal_real = torch.view_as_real(signal_2d)           # [E*R*N, S, 2]
        signal_interleaved = signal_real.reshape(E * R * N, 2 * S)
        # signal comes from contiguous buffers; reshape preserves contiguity
        assert signal_interleaved.is_contiguous()

        # Zero-copy wrap with Warp
        signal_in_wp = wp.from_torch(signal_interleaved)
        delay_wp = wp.from_torch(self._delay_buf)
        doppler_wp = wp.from_torch(self._doppler_phase_buf)
        gain_wp = wp.from_torch(self._gain_buf)
        out_wp = wp.from_torch(self._out_buf)

        wp.launch(
            _delay_doppler_batched,
            dim=E * R * N,
            inputs=[
                signal_in_wp, delay_wp, doppler_wp, gain_wp,
                N, S,
                out_wp,
            ],
            device=self.device,
        )

        # Convert output buffer back to complex view
        out_3d = self._out_buf.reshape(E * R * N, S, 2)
        # _out_buf is pre-allocated and contiguous; reshape preserves contiguity
        assert out_3d.is_contiguous()
        out_complex = torch.view_as_complex(out_3d)
        return out_complex.reshape(E, R, N, S)

    def generate_noise(self, out: torch.Tensor):
        """Generate complex AWGN in-place into pre-allocated tensor [E, R, N, S] complex64.

        Each quadrature gets variance noise_std**2, so total complex power
        = 2 * noise_std**2 = noise_power_linear, matching the standard RF
        noise power kB*T*B*F.
        """
        noise_view = torch.view_as_real(out)
        # torch.normal_() gives N(0,1); scale by noise_std so each
        # quadrature has variance noise_std**2 — total complex variance
        # 2 * noise_std**2 = noise_power_linear.
        noise_view[..., 0].normal_()
        noise_view[..., 1].normal_()
        noise_view.mul_(self.noise_std)

    def compute_params_one_way(
        self,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        tx_vel: torch.Tensor = None,
        rx_vel: torch.Tensor = None,
        tx_power_w: float = 1.0,
        directivity_db: float = 44.0,
        system_loss_db: float = 3.0,
        beam_az: Optional[torch.Tensor] = None,
        beam_el: Optional[torch.Tensor] = None,
        array_rotation: Optional[torch.Tensor] = None,
        bw_az_deg: float = 4.06,
        bw_el_deg: float = 4.06,
    ):
        """One-way channel parameters (radar → missile comm link).

        Args:
            tx_pos: [E, 3] transmitter position
            rx_pos: [E, 3] receiver position
            tx_vel: [E, 3] (optional)
            rx_vel: [E, 3] (optional)
            tx_power_w: transmit power in watts
            directivity_db: TX antenna directivity (dB)
            system_loss_db: system losses (dB)
            beam_az: [E] mean array-local azimuth (deg) or None
            beam_el: [E] mean array-local elevation (deg) or None
            array_rotation: [E] array rotation (deg) or None
            bw_az_deg, bw_el_deg: 3dB beamwidth (deg)
        Returns:
            delay_samples: [E] float32 (one-way)
            doppler_hz:    [E] float32 (one-way)
            gain_linear:   [E] float32 (one-way path gain)
        """
        rel = rx_pos - tx_pos  # [E, 3]
        distance = rel.norm(dim=-1).clamp(min=1.0)  # [E]

        # One-way delay
        delay_s = distance / SPEED_OF_LIGHT
        delay_samples = delay_s * self.fs

        # One-way Doppler (no factor of 2)
        if tx_vel is not None and rx_vel is not None:
            rel_vel = tx_vel - rx_vel
            radial_vel = (rel_vel * rel).sum(dim=-1) / distance
            doppler_hz = radial_vel * self.fc / SPEED_OF_LIGHT
        else:
            doppler_hz = torch.zeros_like(distance)

        # Beam-dependent directivity (same off-boresight model as two-way)
        tx_gain_db = torch.full_like(distance, directivity_db)
        if beam_az is not None and beam_el is not None:
            tgt_az_rad = torch.atan2(rel[..., 1], rel[..., 0])
            tgt_el_rad = torch.atan2(
                rel[..., 2], torch.sqrt(rel[..., 0]**2 + rel[..., 1]**2).clamp(min=1.0),
            )
            rot = array_rotation * (np.pi / 180.0) if array_rotation is not None else 0.0
            world_beam_az_rad = beam_az * (np.pi / 180.0) + rot
            world_beam_el_rad = beam_el * (np.pi / 180.0)
            d_az_rad = world_beam_az_rad - tgt_az_rad
            d_az_rad = torch.atan2(torch.sin(d_az_rad), torch.cos(d_az_rad))
            d_el_rad = world_beam_el_rad - tgt_el_rad
            d_az_deg = d_az_rad * (180.0 / np.pi)
            d_el_deg = d_el_rad * (180.0 / np.pi)
            loss_db = (-3.0 * (d_az_deg / bw_az_deg)**2
                       - 3.0 * (d_el_deg / bw_el_deg)**2)
            tx_gain_db = directivity_db + loss_db

        # One-way path loss
        one_way_pl_db = 20.0 * torch.log10(
            4.0 * np.pi * distance / self.wavelength + 1e-10,
        )

        rx_power_dbm = (
            10.0 * np.log10(tx_power_w * 1000.0)
            + tx_gain_db
            - one_way_pl_db
            - system_loss_db
        )
        rx_power_w = 10.0 ** ((rx_power_dbm - 30.0) / 10.0)
        gain_linear = torch.sqrt(rx_power_w.clamp(min=0.0))

        return delay_samples, doppler_hz, gain_linear

    def apply_one_way(
        self,
        signal: torch.Tensor,
        gain_linear: torch.Tensor,
        doppler_hz: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply one-way channel to [E, S] signal (pure torch, no Warp).

        Simplified: gain scaling + optional Doppler phase. Delay omitted
        because BPSK matched filter handles timing.

        Args:
            signal: [E, S] complex64
            gain_linear: [E] float32
            doppler_hz: [E] float32 (optional)
        Returns:
            [E, S] complex64
        """
        result = signal * gain_linear.unsqueeze(-1)
        if doppler_hz is not None and doppler_hz.abs().max() > 0:
            S = signal.shape[-1]
            t = torch.arange(S, dtype=torch.float32, device=signal.device) / self.fs
            phase = 2.0 * np.pi * doppler_hz.unsqueeze(-1) * t.unsqueeze(0)
            result = result * torch.exp(1j * phase)
        return result
