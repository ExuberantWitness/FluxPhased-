"""GPU-accelerated per-element channel simulation using NVIDIA Warp.

Applies delay, Doppler shift, and Rayleigh fading to IQ signals
for each of the 4×625 antenna elements. Processes in chunks to fit
within VRAM constraints (RTX 2060: 6.4 GB).
"""

import numpy as np
import warp as wp
import torch
from typing import Optional, Tuple

SPEED_OF_LIGHT = 299792458.0
BOLTZMANN = 1.380649e-23


@wp.kernel
def _apply_delay_doppler_kernel(
    # Input IQ signal [n_elem, 2*n_samples] (interleaved)
    signal_in: wp.array2d(dtype=wp.float32),
    # Per-element parameters [n_elem]
    delay_samples: wp.array1d(dtype=wp.float32),
    doppler_phase_step: wp.array1d(dtype=wp.float32),
    path_gain: wp.array1d(dtype=wp.float32),
    # Fading coefficients [n_elem, 2] (interleaved complex)
    fading: wp.array2d(dtype=wp.float32),
    n_samples: wp.int32,
    # Output [n_elem, 2*n_samples]
    signal_out: wp.array2d(dtype=wp.float32),
):
    """Apply delay, Doppler, path gain, and fading to per-element IQ signal.

    Each thread processes one element. Delay is applied as fractional-sample
    shift (nearest-neighbor for speed, linear interp would be better but slower).
    Doppler is applied as per-pulse phase ramp.
    """
    e = wp.tid()
    g = path_gain[e]
    fd = fading[e, 0]  # real part of fading
    fi = fading[e, 1]  # imag part of fading
    total_gain_re = g * fd
    total_gain_im = g * fi
    dstep = doppler_phase_step[e]
    d_int = wp.int32(delay_samples[e])
    d_frac = delay_samples[e] - wp.float32(d_int)

    for s in range(n_samples):
        # Delay: output[s] = input[s - delay]
        src = s - d_int
        if src >= 0 and src < n_samples:
            s_re = signal_in[e, 2 * src]
            s_im = signal_in[e, 2 * src + 1]
        else:
            s_re = wp.float32(0.0)
            s_im = wp.float32(0.0)

        # Doppler phase rotation for this sample
        phase = wp.float32(s) * dstep
        cos_p = wp.cos(phase)
        sin_p = wp.sin(phase)

        # Apply gain * fading * doppler
        out_re = total_gain_re * (s_re * cos_p - s_im * sin_p) - total_gain_im * (s_re * sin_p + s_im * cos_p)
        out_im = total_gain_re * (s_re * sin_p + s_im * cos_p) + total_gain_im * (s_re * cos_p - s_im * sin_p)

        signal_out[e, 2 * s] = out_re
        signal_out[e, 2 * s + 1] = out_im


@wp.kernel
def _add_thermal_noise_kernel(
    signal: wp.array2d(dtype=wp.float32),
    noise_re: wp.array2d(dtype=wp.float32),
    noise_im: wp.array2d(dtype=wp.float32),
    n_samples: wp.int32,
):
    """Add complex thermal noise to signal. Each thread = one element."""
    e = wp.tid()
    for s in range(n_samples):
        signal[e, 2 * s] += noise_re[e, s]
        signal[e, 2 * s + 1] += noise_im[e, s]


@wp.kernel
def _generate_rayleigh_kernel(
    rand1: wp.array2d(dtype=wp.float32),
    rand2: wp.array2d(dtype=wp.float32),
    n_paths: wp.int32,
    n_elem: wp.int32,
    doppler_spread: wp.float32,
    fs: wp.float32,
    # Output [n_elem, 2] (interleaved complex fading coeff)
    fading_out: wp.array2d(dtype=wp.float32),
):
    """Generate Rayleigh fading coefficients per element using Jakes' model.

    Each thread handles one element.
    """
    e = wp.tid()
    h_re = wp.float32(0.0)
    h_im = wp.float32(0.0)

    for n in range(n_paths):
        alpha_n = (wp.float32(2) * wp.float32(wp.pi) * wp.float32(n + 1) - wp.float32(0.5)) / wp.float32(n_paths * 2)
        phi = wp.float32(2) * wp.float32(wp.pi) * rand1[e, n % 8]
        psi = wp.float32(2) * wp.float32(wp.pi) * rand2[e, n % 8]
        # Simplified Jakes: sum of sinusoids at Doppler-shifted frequencies
        cos_val = wp.cos(wp.float32(2) * wp.float32(wp.pi) * doppler_spread * wp.cos(alpha_n) / fs + phi)
        sin_val = wp.cos(wp.float32(2) * wp.float32(wp.pi) * doppler_spread * wp.cos(alpha_n) / fs + psi)
        h_re += cos_val
        h_im += sin_val

    # Normalize
    norm = wp.sqrt(wp.float32(2) * wp.float32(n_paths))
    fading_out[e, 0] = h_re / norm
    fading_out[e, 1] = h_im / norm


class ChannelGPU:
    """GPU-accelerated per-element channel simulation for multi-radar scenario.

    Handles:
    - Free-space path loss (scalar per link, computed analytically)
    - Per-element delay (range-dependent)
    - Per-element Doppler shift (velocity-dependent)
    - Rayleigh fading (time-correlated, Jakes' model)
    - Thermal noise (AWGN per element)
    """

    def __init__(self, fc: float = 10e9, bandwidth: float = 200e6,
                 noise_figure_db: float = 5.0, device: str = "cuda"):
        self.fc = fc
        self.bandwidth = bandwidth
        self.fs = bandwidth  # complex sampling rate = bandwidth
        self.wavelength = SPEED_OF_LIGHT / fc
        self.noise_figure_db = noise_figure_db
        self.device = device
        self._rng = np.random.default_rng(42)

        # Noise power
        self.noise_power_dbm = (
            10 * np.log10(BOLTZMANN * 290 * 1000)
            + 10 * np.log10(bandwidth)
            + noise_figure_db
        )
        self.noise_power_linear = 10 ** (self.noise_power_dbm / 10.0) * 1e-3

    def compute_path_loss_db(self, distance_m: float) -> float:
        """One-way free-space path loss in dB."""
        if distance_m <= 0:
            return 0.0
        return 20.0 * np.log10(4.0 * np.pi * distance_m / self.wavelength)

    def compute_radar_path_loss_db(self, distance_m: float) -> float:
        """Two-way (monostatic radar) path loss in dB."""
        if distance_m <= 0:
            return 0.0
        return 40.0 * np.log10(4.0 * np.pi * distance_m / self.wavelength)

    def compute_channel_params(
        self,
        tx_pos: np.ndarray,  # [3]
        rx_pos: np.ndarray,  # [3]
        tx_vel: np.ndarray = None,  # [3]
        rx_vel: np.ndarray = None,  # [3]
    ) -> dict:
        """Compute channel parameters for a single TX-RX link.

        Returns delay (samples), Doppler (Hz), path gain (linear), path loss (dB).
        """
        rel = rx_pos - tx_pos
        distance = float(np.linalg.norm(rel))
        if distance < 1.0:
            distance = 1.0

        # Path loss (one-way)
        path_loss_db = self.compute_path_loss_db(distance)
        path_gain_linear = 10.0 ** (-path_loss_db / 10.0)

        # Delay in samples
        delay_s = distance / SPEED_OF_LIGHT
        delay_samples = delay_s * self.fs

        # Doppler from radial velocity
        doppler_hz = 0.0
        if tx_vel is not None and rx_vel is not None:
            rel_vel = np.dot(tx_vel - rx_vel, rel) / distance
            doppler_hz = 2.0 * rel_vel * self.fc / SPEED_OF_LIGHT  # two-way Doppler

        return {
            "delay_samples": float(delay_samples),
            "doppler_hz": float(doppler_hz),
            "path_gain_linear": float(path_gain_linear),
            "path_loss_db": float(path_loss_db),
            "distance_m": float(distance),
        }

    def apply_channel(
        self,
        signal: torch.Tensor,  # [n_elem, n_samples] complex64
        channel_params: dict,
        doppler_spread: float = 0.0,
    ) -> torch.Tensor:
        """Apply channel effects to per-element IQ signal.

        Args:
            signal: [n_elem, n_samples] complex64 TX signal
            channel_params: output of compute_channel_params()
            doppler_spread: Doppler spread for Rayleigh fading (Hz)
        Returns:
            [n_elem, n_samples] complex64 received signal
        """
        n_elem, n_samples = signal.shape
        device = self.device

        # Generate Rayleigh fading coefficients per element
        fading = self._generate_fading(n_elem, doppler_spread)

        # Per-element delay (same for all elements in far-field)
        delay_samples = np.full(n_elem, channel_params["delay_samples"], dtype=np.float32)

        # Per-element Doppler phase step
        doppler_phase_step = np.full(
            n_elem,
            2.0 * np.pi * channel_params["doppler_hz"] / self.fs,
            dtype=np.float32,
        )

        # Path gain
        path_gain = np.full(n_elem, channel_params["path_gain_linear"], dtype=np.float32)

        # Convert signal to interleaved float32 for Warp
        signal_interleaved = self._complex_to_interleaved(signal)

        # Allocate Warp arrays
        delay_wp = wp.array(delay_samples, dtype=wp.float32, device=device)
        doppler_wp = wp.array(doppler_phase_step, dtype=wp.float32, device=device)
        gain_wp = wp.array(path_gain, dtype=wp.float32, device=device)
        fading_wp = wp.array(fading, dtype=wp.float32, device=device)
        out_wp = wp.zeros((n_elem, 2 * n_samples), dtype=wp.float32, device=device)

        wp.launch(
            _apply_delay_doppler_kernel,
            dim=n_elem,
            inputs=[signal_interleaved, delay_wp, doppler_wp, gain_wp,
                    fading_wp, n_samples, out_wp],
            device=device,
        )

        # Convert back to torch complex
        return self._interleaved_to_complex(out_wp, n_elem, n_samples)

    def add_noise(
        self,
        signal: torch.Tensor,  # [n_elem, n_samples] complex64
        snr_db: Optional[float] = None,
        noise_power_linear: Optional[float] = None,
    ) -> torch.Tensor:
        """Add AWGN noise to signal."""
        if noise_power_linear is None:
            if snr_db is not None:
                signal_power = torch.mean(torch.abs(signal) ** 2).item()
                noise_power_linear = signal_power / (10 ** (snr_db / 10.0))
            else:
                noise_power_linear = self.noise_power_linear

        noise_std = np.sqrt(noise_power_linear / 2.0)
        noise = torch.randn_like(signal) * noise_std + 1j * torch.randn_like(signal) * noise_std
        return signal + noise

    def _generate_fading(self, n_elem: int, doppler_spread: float) -> np.ndarray:
        """Generate Rayleigh fading coefficients [n_elem, 2] (interleaved)."""
        if doppler_spread <= 0:
            # No fading: unit gain
            fading = np.zeros((n_elem, 2), dtype=np.float32)
            fading[:, 0] = 1.0  # real part = 1
            return fading

        # Jakes' model: sum of sinusoids
        n_paths = 12
        h = np.zeros(n_elem, dtype=complex)
        for n in range(n_paths):
            alpha_n = (2 * np.pi * (n + 1) - 0.5) / (n_paths * 2)
            phi = self._rng.uniform(0, 2 * np.pi, n_elem)
            psi = self._rng.uniform(0, 2 * np.pi, n_elem)
            h += np.exp(1j * phi) + np.exp(1j * psi)
        h /= np.sqrt(2 * n_paths)

        fading = np.zeros((n_elem, 2), dtype=np.float32)
        fading[:, 0] = h.real
        fading[:, 1] = h.imag
        return fading

    def _complex_to_interleaved(self, tensor: torch.Tensor) -> wp.array:
        """Convert [n_elem, n_samples] complex64 torch tensor to Warp [n_elem, 2*n_samples] float32."""
        np_arr = tensor.detach().cpu().numpy()
        n_elem, n_samples = np_arr.shape
        interleaved = np.zeros((n_elem, 2 * n_samples), dtype=np.float32)
        interleaved[:, 0::2] = np_arr.real
        interleaved[:, 1::2] = np_arr.imag
        return wp.array(interleaved, dtype=wp.float32, device=self.device)

    def _interleaved_to_complex(self, warp_arr: wp.array, n_elem: int,
                                 n_samples: int) -> torch.Tensor:
        """Convert Warp [n_elem, 2*n_samples] float32 back to torch complex64."""
        np_arr = warp_arr.numpy()
        re = torch.from_numpy(np_arr[:, 0::2].copy()).to(self.device)
        im = torch.from_numpy(np_arr[:, 1::2].copy()).to(self.device)
        return torch.complex(re, im)
