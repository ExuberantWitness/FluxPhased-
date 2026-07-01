"""Phased array antenna modeling: 25x25 element planar array.

Computes array factor, beam steering, 3D gain patterns, and supports
amplitude tapering (uniform, Taylor, Chebyshev) and multi-beam operation.

API-compatible with RadarSimPy Transmitter/Receiver channel antenna patterns:
    az_angles, az_pattern (dB), el_angles, el_pattern (dB)
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from ..config import ArrayGeometry


SPEED_OF_LIGHT = 299792458.0
DEG2RAD = np.pi / 180.0


class PhasedArray:
    """25x25 planar phased array with electronic beam steering."""

    def __init__(self, geom: ArrayGeometry, fc: float):
        self.geom = geom
        self.fc = fc
        self._wavelength = SPEED_OF_LIGHT / fc
        self._k = 2.0 * np.pi / self._wavelength

        # Element positions in meters
        self._build_element_grid()

        # Amplitude taper weights
        self._taper = self._build_taper(geom.taper, geom.taper_param)

        # Current beam steering state
        self._beam_az: float = 0.0
        self._beam_el: float = 0.0
        self._phase_weights: Optional[NDArray] = None

        # Cached 3D pattern for current steering
        self._cached_az_angles: Optional[NDArray] = None
        self._cached_el_angles: Optional[NDArray] = None
        self._cached_pattern_2d: Optional[NDArray] = None

    def _build_element_grid(self):
        """Build 625-element grid positions in meters."""
        rows, cols = self.geom.rows, self.geom.cols
        dx_m = self.geom.dx_wl * self._wavelength
        dy_m = self.geom.dy_wl * self._wavelength

        # Center the array at origin
        x_center = (cols - 1) * dx_m / 2.0
        y_center = (rows - 1) * dy_m / 2.0

        x_pos = np.arange(cols) * dx_m - x_center
        y_pos = np.arange(rows) * dy_m - y_center
        X, Y = np.meshgrid(x_pos, y_pos)
        self._elem_x = X.ravel()  # 625 elements
        self._elem_y = Y.ravel()
        self._elem_z = np.zeros_like(self._elem_x)

    def _build_taper(self, taper_type: str, param: float) -> NDArray:
        """Build 2D amplitude taper weights."""
        n = self.geom.num_elements
        if taper_type == "uniform":
            return np.ones(n) / n

        rows, cols = self.geom.rows, self.geom.cols
        # 1D Taylor window
        if taper_type == "taylor":
            nbar = min(5, rows - 1)
            row_win = self._taylor_window(rows, param, nbar)
            col_win = self._taylor_window(cols, param, nbar)
        elif taper_type == "chebyshev":
            row_win = self._chebyshev_window(rows, param)
            col_win = self._chebyshev_window(cols, param)
        else:
            return np.ones(n) / n

        win_2d = np.outer(row_win, col_win).ravel()
        return win_2d / np.sum(win_2d)

    @staticmethod
    def _taylor_window(n: int, sll_db: float, nbar: int) -> NDArray:
        """Taylor amplitude taper (approximate)."""
        a = np.arccosh(10 ** (sll_db / 20.0)) / np.pi
        w = np.ones(n)
        for i in range(1, n):
            x = (i - (n - 1) / 2.0) / ((n - 1) / 2.0)
            if abs(x) < 1.0:
                # Taylor pattern approximation using Kaiser
                w[i] = np.i0(np.pi * a * np.sqrt(1.0 - x * x)) / np.i0(np.pi * a)
        return w

    @staticmethod
    def _chebyshev_window(n: int, sll_db: float) -> NDArray:
        """Dolph-Chebyshev amplitude taper."""
        r = 10.0 ** (sll_db / 20.0)
        x0 = np.cosh(np.arccosh(r) / (n - 1))
        w = np.zeros(n)
        for i in range(n):
            x = x0 * np.cos(np.pi * i / (n - 1))
            if abs(x) <= 1.0:
                w[i] = np.cos((n - 1) * np.arccos(x))
            else:
                w[i] = np.cosh((n - 1) * np.arccosh(x))
            if i > 0 and i < n - 1:
                w[i] = w[i] * (-1) ** i
        w = np.abs(w)
        return w / np.max(w)

    @property
    def wavelength(self) -> float:
        return self._wavelength

    @property
    def directivity_linear(self) -> float:
        """Approximate directivity (linear). 4πA/λ² for uniform, reduced for tapered."""
        dx_wl = self.geom.dx_wl
        dy_wl = self.geom.dy_wl
        area_wl2 = self.geom.rows * self.geom.cols * dx_wl * dy_wl
        # Taper efficiency: ~0.85 for Taylor -30dB, ~0.65 for Chebyshev -40dB
        if self.geom.taper == "uniform":
            eff = 1.0
        elif self.geom.taper == "taylor":
            eff = 0.85
        else:
            eff = 0.65
        return 4.0 * np.pi * area_wl2 * eff

    @property
    def directivity_db(self) -> float:
        return 10.0 * np.log10(self.directivity_linear)

    def steer_beam(self, az_deg: float, el_deg: float):
        """Set beam steering angles and compute element phase weights."""
        self._beam_az = np.clip(az_deg, -90.0, 90.0)
        self._beam_el = np.clip(el_deg, -90.0, 90.0)

        az_rad = self._beam_az * DEG2RAD
        el_rad = self._beam_el * DEG2RAD

        # Steering vector: desired phase progression
        u0 = np.sin(az_rad) * np.cos(el_rad)
        v0 = np.sin(el_rad)

        # Phase for each element (conjugate for transmit beamforming)
        phase = -self._k * (self._elem_x * u0 + self._elem_y * v0)
        self._phase_weights = np.exp(1j * phase) * self._taper

        # Invalidate cache
        self._cached_pattern_2d = None

    def steer_multi_beam(self, beams: list[Tuple[float, float, float]]):
        """Compute combined weights for multiple simultaneous beams.

        Each beam is (az_deg, el_deg, power_fraction). Sum of power_fraction <= 1.0.
        """
        total_w = np.zeros(self.geom.num_elements, dtype=complex)
        for az, el, frac in beams:
            az_rad = az * DEG2RAD
            el_rad = el * DEG2RAD
            u0 = np.sin(az_rad) * np.cos(el_rad)
            v0 = np.sin(el_rad)
            phase = -self._k * (self._elem_x * u0 + self._elem_y * v0)
            total_w += np.sqrt(frac) * np.exp(1j * phase) * self._taper
        self._phase_weights = total_w
        self._phase_weights /= np.sqrt(np.sum(np.abs(self._phase_weights) ** 2))
        self._cached_pattern_2d = None

    def element_pattern(self, az_deg: NDArray, el_deg: NDArray = None) -> NDArray:
        """Single element pattern (cosine model for patch antenna).

        Supports broadcasting: az_deg [n_el, n_az], el_deg [n_el, n_az] or scalars.
        """
        theta = np.abs(np.atleast_1d(az_deg)) * DEG2RAD
        pattern = np.cos(np.clip(theta, -np.pi / 2, np.pi / 2)) ** 1.5
        pattern[np.abs(np.asarray(az_deg)) > 90] = 0.0
        return pattern

    def array_factor(self, az_deg: NDArray, el_deg: NDArray = None) -> NDArray:
        """Compute array factor (vectorized). Supports broadcasting.

        az_deg: [n_angles] or scalar
        el_deg: [n_angles] or scalar (default 0)
        Returns: complex array factor [n_angles]
        """
        az_arr = np.atleast_1d(np.asarray(az_deg, dtype=np.float64))
        if el_deg is None:
            el_arr = np.zeros_like(az_arr)
        else:
            el_arr = np.atleast_1d(np.asarray(el_deg, dtype=np.float64))

        az_rad = az_arr * DEG2RAD
        el_rad = el_arr * DEG2RAD

        u = np.sin(az_rad) * np.cos(el_rad)  # [n_angles]
        v = np.sin(el_rad)                    # [n_angles]

        # Vectorized: [n_elem, n_angles] = [n_elem, 1] * [n_angles] + [n_elem, 1] * [n_angles]
        phase = self._k * (
            self._elem_x[:, np.newaxis] * u[np.newaxis, :]
            + self._elem_y[:, np.newaxis] * v[np.newaxis, :]
        )  # [n_elem, n_angles]

        weights = self._phase_weights if self._phase_weights is not None else self._taper
        # [n_elem, 1] * exp(j * phase) -> sum over elements
        af = np.sum(weights[:, np.newaxis] * np.exp(1j * phase), axis=0)

        return af

    def compute_pattern(self, az_deg: NDArray, el_deg: NDArray = None) -> NDArray:
        """Compute 2D gain pattern (dB). Vectorized.

        Returns: pattern [len(el), len(az)] or [len(az)] in dB, normalized to peak.
        """
        az_arr = np.atleast_1d(np.asarray(az_deg, dtype=np.float64))
        if el_deg is None:
            el_arr = np.array([0.0])
        else:
            el_arr = np.atleast_1d(np.asarray(el_deg, dtype=np.float64))

        n_el = len(el_arr)
        n_az = len(az_arr)

        # Create grids for broadcasting
        az_grid = np.broadcast_to(az_arr[np.newaxis, :], (n_el, n_az)).ravel()
        el_grid = np.broadcast_to(el_arr[:, np.newaxis], (n_el, n_az)).ravel()

        # Compute array factor for all (az, el) pairs at once
        af = self.array_factor(az_grid, el_grid)  # [n_el * n_az]
        af = af.reshape(n_el, n_az)

        # Element pattern
        az_rad = az_arr * DEG2RAD
        el_pat = np.cos(np.clip(az_rad, -np.pi / 2, np.pi / 2)) ** 1.5
        el_pat = np.broadcast_to(el_pat[np.newaxis, :], (n_el, n_az))

        pattern_linear = np.abs(af) * el_pat

        # Normalize
        peak = np.max(pattern_linear)
        if peak > 0:
            pattern_linear = pattern_linear / peak * np.sqrt(self.directivity_linear)

        pattern_db = 20.0 * np.log10(np.maximum(pattern_linear, 1e-15))
        pattern_db -= np.max(pattern_db)

        self._cached_az_angles = az_arr
        self._cached_el_angles = el_arr
        self._cached_pattern_2d = pattern_db

        if n_el == 1:
            return pattern_db[0, :]
        return pattern_db

    def get_antenna_gain(self, az_deg: float, el_deg: float = 0.0) -> float:
        """Get antenna gain (dB) at a specific angle. Fast single-point."""
        az_rad = float(az_deg) * DEG2RAD
        el_rad = float(el_deg) * DEG2RAD

        u = np.sin(az_rad) * np.cos(el_rad)
        v = np.sin(el_rad)

        # Vectorized single point
        phase = self._k * (self._elem_x * u + self._elem_y * v)
        weights = self._phase_weights if self._phase_weights is not None else self._taper
        af = np.sum(weights * np.exp(1j * phase))

        # Element pattern
        el_pat = np.cos(np.clip(az_rad, -np.pi / 2, np.pi / 2)) ** 1.5

        gain_linear = np.abs(af) * el_pat
        if gain_linear > 0:
            gain_db = 20.0 * np.log10(gain_linear)
        else:
            gain_db = -200.0

        return float(gain_db)

    def beamwidth_3db(self) -> Tuple[float, float]:
        """Estimate 3dB beamwidth (az, el) in degrees.

        3dB beamwidth ≈ 0.886 * λ / (N * d) for uniform linear array.
        For planar array, same formula per dimension.
        """
        bw_az_rad = 0.886 / (self.geom.cols * self.geom.dx_wl)
        bw_el_rad = 0.886 / (self.geom.rows * self.geom.dy_wl)
        bw_az_deg = np.degrees(bw_az_rad)
        bw_el_deg = np.degrees(bw_el_rad)

        # Beam broadening for scanned beams
        scan_az_rad = abs(self._beam_az * DEG2RAD)
        scan_factor = 1.0 / max(np.cos(scan_az_rad), 0.1)
        return bw_az_deg * scan_factor, bw_el_deg * scan_factor

    def get_radarsimpy_pattern(self, az_res: float = 1.0, el_res: float = 1.0):
        """Return antenna pattern dict compatible with RadarSimPy channel spec.

        Returns dict with keys:
            azimuth_angle, azimuth_pattern, elevation_angle, elevation_pattern
        Each pattern is in dB, normalized to peak=0 dB.
        """
        az_angles = np.arange(-90, 90 + az_res, az_res)
        el_angles = np.arange(-90, 90 + el_res, el_res)

        # Azimuth cut at el=0
        az_pat = np.zeros(len(az_angles))
        for i, az in enumerate(az_angles):
            az_pat[i] = self.get_antenna_gain(float(az), 0.0)

        # Elevation cut at az=0
        el_pat = np.zeros(len(el_angles))
        for i, el in enumerate(el_angles):
            el_pat[i] = self.get_antenna_gain(0.0, float(el))

        return {
            "azimuth_angle": az_angles,
            "azimuth_pattern": az_pat,
            "elevation_angle": el_angles,
            "elevation_pattern": el_pat,
        }

    def compute_phase_shifts(self) -> NDArray:
        """Return current per-element phase shifts (degrees)."""
        if self._phase_weights is None:
            return np.zeros(self.geom.num_elements)
        return np.angle(self._phase_weights, deg=True)

    @property
    def beam_az(self) -> float:
        return self._beam_az

    @property
    def beam_el(self) -> float:
        return self._beam_el
