"""Batched phased array for num_envs parallel environments.

Warp kernel for beam steering (batched across env×radar) + per-element steering.
Uses wp.from_torch for zero-copy GPU memory sharing.
"""

import numpy as np
import warp as wp
import torch

SPEED_OF_LIGHT = 299792458.0
DEG2RAD = np.pi / 180.0


@wp.kernel
def _steer_beam_batched(
    elem_x: wp.array1d(dtype=wp.float32),
    elem_y: wp.array1d(dtype=wp.float32),
    k: wp.float32,
    az_rad: wp.array1d(dtype=wp.float32),       # [E*R]
    el_rad: wp.array1d(dtype=wp.float32),       # [E*R]
    taper: wp.array1d(dtype=wp.float32),        # [N]
    n_elem: wp.int32,
    weights_out: wp.array2d(dtype=wp.float32),  # [E*R, 2*N]
):
    """Compute beam steering weights for all env×radar pairs.

    flat = wp.tid() in [0, E*R*N).  er_idx = flat / N, i = flat % N.
    """
    flat = wp.tid()
    er_idx = flat // n_elem
    i = flat % n_elem

    u0 = wp.sin(az_rad[er_idx]) * wp.cos(el_rad[er_idx])
    v0 = wp.sin(el_rad[er_idx])
    phase = -k * (elem_x[i] * u0 + elem_y[i] * v0)
    weights_out[er_idx, 2 * i] = taper[i] * wp.cos(phase)
    weights_out[er_idx, 2 * i + 1] = taper[i] * wp.sin(phase)


@wp.kernel
def _steer_per_element(
    elem_x: wp.array1d(dtype=wp.float32),
    elem_y: wp.array1d(dtype=wp.float32),
    k: wp.float32,
    az_rad: wp.array1d(dtype=wp.float32),       # [E*R*N]
    el_rad: wp.array1d(dtype=wp.float32),       # [E*R*N]
    taper: wp.array1d(dtype=wp.float32),        # [N]
    n_elem: wp.int32,
    weights_out: wp.array2d(dtype=wp.float32),  # [E*R, 2*N]
):
    """Per-element beam steering — each element gets independent (az, el).

    flat = wp.tid() in [0, E*R*N).
    """
    flat = wp.tid()
    er_idx = flat // n_elem
    i = flat % n_elem

    u0 = wp.sin(az_rad[flat]) * wp.cos(el_rad[flat])
    v0 = wp.sin(el_rad[flat])
    phase = -k * (elem_x[i] * u0 + elem_y[i] * v0)
    weights_out[er_idx, 2 * i] = taper[i] * wp.cos(phase)
    weights_out[er_idx, 2 * i + 1] = taper[i] * wp.sin(phase)


class VecArray:
    """Batched phased array for E parallel environments, each with R radars.

    Pre-allocates all GPU buffers in __init__; steer_all() writes weights
    in-place and returns a complex64 view (zero-copy via wp.from_torch).
    """

    def __init__(
        self, rows: int = 25, cols: int = 25, fc: float = 10e9,
        num_envs: int = 10, n_radars: int = 4,
        dx_wl: float = 0.5, dy_wl: float = 0.5, device: str = "cuda",
    ):
        self.rows = rows
        self.cols = cols
        self.n_elem = rows * cols
        self.num_envs = num_envs
        self.n_radars = n_radars
        self.fc = fc
        self.dx_wl = dx_wl
        self.dy_wl = dy_wl
        self.wavelength = SPEED_OF_LIGHT / fc
        self.k = 2.0 * np.pi / self.wavelength
        self.device = device

        # Element grid (shared across all envs and radars)
        dx_m = dx_wl * self.wavelength
        dy_m = dy_wl * self.wavelength
        x_pos = (np.arange(cols) - (cols - 1) / 2.0) * dx_m
        y_pos = (np.arange(rows) - (rows - 1) / 2.0) * dy_m
        X, Y = np.meshgrid(x_pos, y_pos)
        elem_x = X.ravel().astype(np.float32)
        elem_y = Y.ravel().astype(np.float32)

        self._elem_x = wp.array(elem_x, dtype=wp.float32, device=device)
        self._elem_y = wp.array(elem_y, dtype=wp.float32, device=device)
        taper_np = np.ones(self.n_elem, dtype=np.float32) / self.n_elem
        self._taper = wp.array(taper_np, dtype=wp.float32, device=device)

        ER = num_envs * n_radars
        # Pre-allocated: interleaved float32 weights [E*R, 2*N]
        self._weights_buf = torch.zeros(
            ER, 2 * self.n_elem, dtype=torch.float32,
            device=torch.device(device),
        )
        # Pre-allocated: flattened az/el in radians
        self._az_flat = torch.zeros(ER, dtype=torch.float32, device=torch.device(device))
        self._el_flat = torch.zeros(ER, dtype=torch.float32, device=torch.device(device))

        # Per-element steering buffers
        self._az_per_elem = torch.zeros(
            ER * self.n_elem, dtype=torch.float32, device=torch.device(device),
        )
        self._el_per_elem = torch.zeros(
            ER * self.n_elem, dtype=torch.float32, device=torch.device(device),
        )

    def steer_all(self, az_deg: torch.Tensor, el_deg: torch.Tensor) -> torch.Tensor:
        """Compute beam steering weights for all envs and radars.

        Args:
            az_deg: [E, R] azimuth in degrees
            el_deg: [E, R] elevation in degrees
        Returns:
            weights: [E, R, N] complex64 (view into pre-allocated buffer)
        """
        E, R = self.num_envs, self.n_radars

        az_rad = torch.clamp(az_deg, -90.0, 90.0) * DEG2RAD
        el_rad = torch.clamp(el_deg, -90.0, 90.0) * DEG2RAD

        self._az_flat.copy_(az_rad.reshape(-1))
        self._el_flat.copy_(el_rad.reshape(-1))
        self._weights_buf.zero_()

        az_wp = wp.from_torch(self._az_flat)
        el_wp = wp.from_torch(self._el_flat)
        weights_wp = wp.from_torch(self._weights_buf)

        wp.launch(
            _steer_beam_batched,
            dim=E * R * self.n_elem,
            inputs=[
                self._elem_x, self._elem_y,
                wp.float32(self.k),
                az_wp, el_wp,
                self._taper,
                self.n_elem,
                weights_wp,
            ],
            device=self.device,
        )

        # [E*R, 2*N] → [E*R, N, 2] → complex [E*R, N] → [E, R, N]
        weights_3d = self._weights_buf.reshape(E * R, self.n_elem, 2)
        assert weights_3d.is_contiguous()
        weights_complex = torch.view_as_complex(weights_3d)
        return weights_complex.reshape(E, R, self.n_elem)

    def steer_per_element(
        self, az_deg: torch.Tensor, el_deg: torch.Tensor,
    ) -> torch.Tensor:
        """Per-element beam steering — each of N elements gets independent (az, el).

        Args:
            az_deg: [E, R, N] azimuth in degrees per element
            el_deg: [E, R, N] elevation in degrees per element
        Returns:
            weights: [E, R, N] complex64 (view into pre-allocated buffer)
        """
        E, R = self.num_envs, self.n_radars
        N = self.n_elem

        az_rad = torch.clamp(az_deg, -90.0, 90.0) * DEG2RAD
        el_rad = torch.clamp(el_deg, -90.0, 90.0) * DEG2RAD

        self._az_per_elem.copy_(az_rad.reshape(-1))
        self._el_per_elem.copy_(el_rad.reshape(-1))
        self._weights_buf.zero_()

        az_wp = wp.from_torch(self._az_per_elem)
        el_wp = wp.from_torch(self._el_per_elem)
        weights_wp = wp.from_torch(self._weights_buf)

        wp.launch(
            _steer_per_element,
            dim=E * R * N,
            inputs=[
                self._elem_x, self._elem_y,
                wp.float32(self.k),
                az_wp, el_wp,
                self._taper,
                N,
                weights_wp,
            ],
            device=self.device,
        )

        weights_3d = self._weights_buf.reshape(E * R, N, 2)
        assert weights_3d.is_contiguous()
        weights_complex = torch.view_as_complex(weights_3d)
        return weights_complex.reshape(E, R, N)

    def beamform_tx(
        self, weights: torch.Tensor, baseband: torch.Tensor,
    ) -> torch.Tensor:
        """TX beamforming: broadcast baseband to all elements with weights.

        Args:
            weights: [E, R, N] complex64
            baseband: [S] complex64
        Returns:
            [E, R, N, S] complex64
        """
        return weights.unsqueeze(-1) * baseband.view(1, 1, 1, -1)

    def beamform_rx(
        self, weights: torch.Tensor, rx_signal: torch.Tensor,
    ) -> torch.Tensor:
        """RX beamforming: conjugate weights, multiply, sum across elements.

        Args:
            weights: [E, R, N] complex64
            rx_signal: [E, R, N, S] complex64
        Returns:
            [E, R, S] complex64
        """
        return (weights.conj().unsqueeze(-1) * rx_signal).sum(dim=2)

    @property
    def directivity_db(self) -> float:
        area_wl2 = self.rows * self.cols * self.dx_wl * self.dy_wl
        return 10.0 * np.log10(4.0 * np.pi * area_wl2)
