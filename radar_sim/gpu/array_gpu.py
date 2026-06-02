"""GPU-accelerated phased array operations using NVIDIA Warp.

Batched array factor computation, beam steering, and per-element
signal generation for multiple 25x25 arrays simultaneously.

Complex numbers represented as wp.vec2f (real, imag) since Warp lacks
native complex64. FFTs delegated to torch.fft (cuFFT backend).
"""

import numpy as np
import warp as wp
import torch
from typing import Tuple, Optional

from ..config import DEFAULT_ROWS, DEFAULT_COLS

SPEED_OF_LIGHT = 299792458.0
DEG2RAD = np.pi / 180.0


@wp.kernel
def _steer_beam_kernel(
    elem_x: wp.array1d(dtype=wp.float32),
    elem_y: wp.array1d(dtype=wp.float32),
    k: wp.float32,
    az_rad: wp.float32,
    el_rad: wp.float32,
    taper: wp.array1d(dtype=wp.float32),
    # Output: interleaved complex [re0, im0, re1, im1, ...]
    weights_out: wp.array1d(dtype=wp.float32),
):
    """Compute beam steering weights for one array. Each thread handles one element."""
    i = wp.tid()
    u0 = wp.sin(az_rad) * wp.cos(el_rad)
    v0 = wp.sin(el_rad)
    phase = -k * (elem_x[i] * u0 + elem_y[i] * v0)
    weights_out[2 * i] = taper[i] * wp.cos(phase)  # re = taper * cos(-k*pos·u0)
    weights_out[2 * i + 1] = taper[i] * wp.sin(phase)  # im = taper * sin(-k*pos·u0)


@wp.kernel
def _array_factor_kernel(
    elem_x: wp.array1d(dtype=wp.float32),
    elem_y: wp.array1d(dtype=wp.float32),
    k: wp.float32,
    weights_re: wp.array1d(dtype=wp.float32),
    weights_im: wp.array1d(dtype=wp.float32),
    az_rad: wp.array1d(dtype=wp.float32),
    el_rad: wp.array1d(dtype=wp.float32),
    # Output [n_angles]: interleaved complex
    af_out: wp.array2d(dtype=wp.float32),
):
    """Compute array factor at multiple angles for one array.

    Launch with dim = (n_elements, n_angles). Each thread computes one
    element's contribution to one angle, then relies on atomic add.
    """
    elem_idx, angle_idx = wp.tid()

    u = wp.sin(az_rad[angle_idx]) * wp.cos(el_rad[angle_idx])
    v = wp.sin(el_rad[angle_idx])

    phase = k * (elem_x[elem_idx] * u + elem_y[elem_idx] * v)
    w_re = weights_re[elem_idx]
    w_im = weights_im[elem_idx]

    contrib_re = w_re * wp.cos(phase) - w_im * wp.sin(phase)
    contrib_im = w_re * wp.sin(phase) + w_im * wp.cos(phase)

    wp.atomic_add(af_out, angle_idx, 0, contrib_re)
    wp.atomic_add(af_out, angle_idx, 1, contrib_im)


@wp.kernel
def _apply_weights_to_signal_kernel(
    # Per-element weights [2*n_elem] (interleaved complex)
    weights: wp.array1d(dtype=wp.float32),
    # Input signal [n_elem, 2*n_samples] (interleaved complex per element)
    signal_in: wp.array2d(dtype=wp.float32),
    # Output signal [n_elem, 2*n_samples]
    signal_out: wp.array2d(dtype=wp.float32),
    n_samples: wp.int32,
):
    """Apply per-element beamforming weights to IQ signal.

    Each thread handles one element, iterates over all samples.
    Complex multiply: (w_re + j*w_im) * (s_re + j*s_im)
    """
    elem_idx = wp.tid()
    w_re = weights[2 * elem_idx]
    w_im = weights[2 * elem_idx + 1]

    for s in range(n_samples):
        s_re = signal_in[elem_idx, 2 * s]
        s_im = signal_in[elem_idx, 2 * s + 1]
        signal_out[elem_idx, 2 * s] = w_re * s_re - w_im * s_im
        signal_out[elem_idx, 2 * s + 1] = w_re * s_im + w_im * s_re


@wp.kernel
def _sum_elements_kernel(
    # Input [n_elem, 2*n_samples] (interleaved complex)
    signal_in: wp.array2d(dtype=wp.float32),
    # Output [2*n_samples] (summed across elements)
    signal_out: wp.array1d(dtype=wp.float32),
    n_samples: wp.int32,
):
    """Sum signals across all antenna elements (digital beamforming).

    Each thread handles one sample index, sums across elements.
    """
    s = wp.tid()
    re_sum = wp.float32(0.0)
    im_sum = wp.float32(0.0)

    for e in range(signal_in.shape[0]):
        re_sum += signal_in[e, 2 * s]
        im_sum += signal_in[e, 2 * s + 1]

    signal_out[2 * s] = re_sum
    signal_out[2 * s + 1] = im_sum


class PhasedArrayGPU:
    """25x25 planar phased array with Warp-accelerated beam steering and signal processing.

    Manages multiple arrays simultaneously for the 4-radar simulation.
    """

    def __init__(self, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS, dx_wl: float = 0.5,
                 dy_wl: float = 0.5, fc: float = 10e9, device: str = "cuda"):
        self.rows = rows
        self.cols = cols
        self.n_elem = rows * cols  # 625
        self.fc = fc
        self.wavelength = SPEED_OF_LIGHT / fc
        self.k = 2.0 * np.pi / self.wavelength
        self.device = device

        # Build element grid (centered at origin)
        dx_m = dx_wl * self.wavelength
        dy_m = dy_wl * self.wavelength
        x_pos = (np.arange(cols) - (cols - 1) / 2.0) * dx_m
        y_pos = (np.arange(rows) - (rows - 1) / 2.0) * dy_m
        X, Y = np.meshgrid(x_pos, y_pos)
        elem_x = X.ravel().astype(np.float32)
        elem_y = Y.ravel().astype(np.float32)

        self.elem_x = wp.array(elem_x, dtype=wp.float32, device=device)
        self.elem_y = wp.array(elem_y, dtype=wp.float32, device=device)
        self.elem_x_np = elem_x
        self.elem_y_np = elem_y

        # Uniform taper
        self._taper = wp.array(
            np.ones(self.n_elem, dtype=np.float32) / self.n_elem,
            dtype=wp.float32, device=device,
        )

        # Beam steering state per radar (up to 4)
        self._weights = {}  # radar_id -> wp.array(2*n_elem, float32)

    def steer_beam(self, radar_id: int, az_deg: float, el_deg: float = 0.0):
        """Set beam steering for a specific radar. Computes per-element complex weights."""
        az_rad = np.float32(np.clip(az_deg, -90, 90) * DEG2RAD)
        el_rad = np.float32(np.clip(el_deg, -90, 90) * DEG2RAD)
        k = wp.float32(self.k)

        weights_buf = wp.zeros(2 * self.n_elem, dtype=wp.float32, device=self.device)
        wp.launch(
            _steer_beam_kernel,
            dim=self.n_elem,
            inputs=[self.elem_x, self.elem_y, k, az_rad, el_rad, self._taper, weights_buf],
            device=self.device,
        )
        self._weights[radar_id] = weights_buf

    def get_weights(self, radar_id: int) -> wp.array:
        """Get current steering weights for a radar. Always returns interleaved [2*n_elem]."""
        if radar_id not in self._weights:
            # Create uniform weights in interleaved format
            uniform = np.zeros(2 * self.n_elem, dtype=np.float32)
            uniform[0::2] = 1.0 / self.n_elem  # real part = taper
            uniform[1::2] = 0.0  # imag part = 0
            self._weights[radar_id] = wp.array(uniform, dtype=wp.float32, device=self.device)
        return self._weights[radar_id]

    def get_weights_torch(self, radar_id: int) -> torch.Tensor:
        """Get weights as PyTorch complex64 tensor on GPU."""
        w = self.get_weights(radar_id)
        w_np = w.numpy()
        w_re = torch.from_numpy(w_np[0::2]).to(self.device)
        w_im = torch.from_numpy(w_np[1::2]).to(self.device)
        return torch.complex(w_re, w_im)

    def beamform_tx(self, radar_id: int, baseband: torch.Tensor) -> torch.Tensor:
        """Apply TX beamforming weights to a baseband waveform.

        Args:
            baseband: [n_samples] complex64 baseband waveform
        Returns:
            [n_elem, n_samples] complex64 per-element TX signals
        """
        weights = self.get_weights_torch(radar_id)  # [n_elem] complex64
        # Broadcast: each element gets the baseband multiplied by its weight
        return weights.unsqueeze(1) * baseband.unsqueeze(0)  # [n_elem, n_samples]

    def beamform_rx(self, radar_id: int, rx_signals: torch.Tensor) -> torch.Tensor:
        """Apply RX digital beamforming: weight and sum across elements.

        Args:
            rx_signals: [n_elem, n_samples] complex64 per-element received signals
        Returns:
            [n_samples] complex64 beamformed output
        """
        weights = self.get_weights_torch(radar_id)  # [n_elem] complex64
        # Weighted sum across elements
        return torch.sum(weights.unsqueeze(1) * rx_signals, dim=0)  # [n_samples]

    def compute_pattern(self, radar_id: int, az_angles: np.ndarray,
                        el_deg: float = 0.0) -> np.ndarray:
        """Compute antenna gain pattern (dB) for a radar.

        Args:
            az_angles: azimuth angles in degrees
            el_deg: elevation angle (scalar)
        Returns:
            gain pattern in dB, normalized to peak=0
        """
        az_rad = (az_angles.astype(np.float32) * DEG2RAD).astype(np.float32)
        el_rad_arr = np.full_like(az_rad, el_deg * DEG2RAD, dtype=np.float32)
        n_angles = len(az_angles)

        az_wp = wp.array(az_rad, dtype=wp.float32, device=self.device)
        el_wp = wp.array(el_rad_arr, dtype=wp.float32, device=self.device)

        w = self.get_weights(radar_id)
        w_re = wp.array(w.numpy()[0::2].copy(), dtype=wp.float32, device=self.device)
        w_im = wp.array(w.numpy()[1::2].copy(), dtype=wp.float32, device=self.device)

        af_out = wp.zeros((n_angles, 2), dtype=wp.float32, device=self.device)

        wp.launch(
            _array_factor_kernel,
            dim=(self.n_elem, n_angles),
            inputs=[self.elem_x, self.elem_y, wp.float32(self.k),
                    w_re, w_im, az_wp, el_wp, af_out],
            device=self.device,
        )

        af_np = af_out.numpy()
        af_complex = af_np[:, 0] + 1j * af_np[:, 1]
        magnitude = np.abs(af_complex)

        # Element pattern (cosine model)
        theta = np.abs(az_angles) * DEG2RAD
        el_pat = np.cos(np.clip(theta, -np.pi / 2, np.pi / 2)) ** 1.5
        magnitude *= el_pat

        # Normalize to peak = 0 dB
        peak = np.max(magnitude)
        if peak > 0:
            pattern_db = 20.0 * np.log10(np.maximum(magnitude / peak, 1e-15))
        else:
            pattern_db = np.full_like(magnitude, -200.0)

        return pattern_db

    def get_gain_at_angle(self, radar_id: int, az_deg: float,
                          el_deg: float = 0.0) -> float:
        """Get antenna gain (dB) at a specific angle for a radar.

        Uses the same normalization as compute_pattern: peak = directivity_db.
        """
        w = self.get_weights(radar_id)
        w_np = w.numpy()
        w_complex = w_np[0::2] + 1j * w_np[1::2]

        az_rad = az_deg * DEG2RAD
        el_rad = el_deg * DEG2RAD
        u = np.sin(az_rad) * np.cos(el_rad)
        v = np.sin(el_rad)

        phase = self.k * (self.elem_x_np * u + self.elem_y_np * v)
        af = np.abs(np.sum(w_complex * np.exp(1j * phase)))

        el_pat = np.cos(np.clip(abs(az_rad), -np.pi / 2, np.pi / 2)) ** 1.5

        gain_linear = af * el_pat * np.sqrt(4.0 * np.pi * self.rows * self.cols * 0.5 * 0.5)
        if gain_linear > 0:
            return 20.0 * np.log10(gain_linear)
        return -200.0

    @property
    def directivity_db(self) -> float:
        """Approximate directivity in dB."""
        dx_wl = 0.5  # half-wavelength spacing
        dy_wl = 0.5
        area_wl2 = self.rows * self.cols * dx_wl * dy_wl
        return 10.0 * np.log10(4.0 * np.pi * area_wl2)

    @property
    def beamwidth_3db(self) -> Tuple[float, float]:
        """3dB beamwidth (az, el) in degrees."""
        bw_az = np.degrees(0.886 / (self.cols * 0.5))
        bw_el = np.degrees(0.886 / (self.rows * 0.5))
        return bw_az, bw_el
