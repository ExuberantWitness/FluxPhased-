"""Batched cross-radar interference for num_envs parallel environments.

Pure-torch vectorized link budget: pairwise distances, angles, path loss,
and array gain → IQ-level interference. Uses torch broadcasting; no Warp.
"""

import numpy as np
import torch

SPEED_OF_LIGHT = 299792458.0
DEG2RAD = np.pi / 180.0
DEFAULT_POLARIZATION_LOSS_DB = 3.0


def _wrapped_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angular difference wrapped to [-180, 180]."""
    d = a - b
    return (d + 180.0) % 360.0 - 180.0


class VecInterference:
    """Vectorized mutual interference across R radars in E parallel environments."""

    def __init__(
        self, fc: float = 10e9, bandwidth: float = 200e6,
        rows: int = 25, cols: int = 25,
        num_envs: int = 10, n_radars: int = 4,
        n_elem: int = 625, device: str = "cuda",
        polarization_loss_db: float = DEFAULT_POLARIZATION_LOSS_DB,
        tx_power_dbm: float = 30.0,
        dx_wl: float = 0.5, dy_wl: float = 0.5,
    ):
        self.fc = fc
        self.bandwidth = bandwidth
        self.fs = bandwidth
        self.wavelength = SPEED_OF_LIGHT / fc
        self.k = 2.0 * np.pi / self.wavelength
        self.num_envs = num_envs
        self.n_radars = n_radars
        self.n_elem = n_elem
        self.device = device
        self.polarization_loss_db = polarization_loss_db
        self.tx_power_dbm = tx_power_dbm

        dx_m = dx_wl * self.wavelength
        dy_m = dy_wl * self.wavelength
        x_pos = (np.arange(cols) - (cols - 1) / 2.0) * dx_m
        y_pos = (np.arange(rows) - (rows - 1) / 2.0) * dy_m
        X, Y = np.meshgrid(x_pos, y_pos)
        self.elem_x = torch.tensor(
            X.ravel().astype(np.float32), device=torch.device(device),
        )
        self.elem_y = torch.tensor(
            Y.ravel().astype(np.float32), device=torch.device(device),
        )
        self._directivity_linear = float(
            np.sqrt(4.0 * np.pi * self.n_elem * dx_wl * dy_wl),
        )

    def _array_gain_db_batch(
        self, weights: torch.Tensor, az_deg: torch.Tensor, el_deg: torch.Tensor,
    ) -> torch.Tensor:
        """Array gain (dBi) at query angles given steering weights.

        Args:
            weights: [*, N] complex64 steering weights
            az_deg:  [*] azimuth in degrees
            el_deg:  [*] elevation in degrees
        Returns:
            gain_db: [*] array gain in dBi
        """
        az_rad = torch.clamp(az_deg, -90.0, 90.0) * DEG2RAD
        el_rad = torch.clamp(el_deg, -90.0, 90.0) * DEG2RAD
        u = torch.sin(az_rad) * torch.cos(el_rad)
        v = torch.sin(el_rad)

        phase = self.k * (
            self.elem_x.view(*([1] * (weights.dim() - 1)), -1) * u.unsqueeze(-1)
            + self.elem_y.view(*([1] * (weights.dim() - 1)), -1) * v.unsqueeze(-1)
        )
        af = torch.abs((weights * torch.exp(1j * phase)).sum(dim=-1))

        # Float32 cos(pi/2) can be slightly negative; clamp before fractional
        # exponent to avoid NaN at exactly +/-90 deg azimuth (corner geometry).
        theta = torch.abs(az_deg) * DEG2RAD
        cos_theta = torch.cos(torch.clamp(theta, 0.0, np.pi / 2)).clamp(min=0.0)
        el_pat = cos_theta ** 1.5

        gain_linear = af * el_pat * self._directivity_linear
        return 20.0 * torch.log10(gain_linear.clamp(min=1e-15))

    def compute(
        self, radar_pos: torch.Tensor, beam_az: torch.Tensor, beam_el: torch.Tensor,
        weights: torch.Tensor, baseband: torch.Tensor, out: torch.Tensor,
    ):
        """Compute IQ-level interference at each radar from all others. Writes into out.

        Args:
            radar_pos: [E, R, 3] radar positions (m)
            beam_az:   [E, R] current beam azimuth (deg)
            beam_el:   [E, R] current beam elevation (deg)
            weights:   [E, R, N] complex64 steering weights
            baseband:  [S] complex64 shared waveform
            out:       [E, R, N, S] complex64 pre-allocated output buffer
            interference: [E, R, N, S] complex64
        """
        E, R, N = self.num_envs, self.n_radars, self.n_elem
        S = baseband.shape[0]
        dev = torch.device(self.device)

        # Pairwise position differences: [E, R, R, 3]
        rel = radar_pos.unsqueeze(2) - radar_pos.unsqueeze(1)
        dist = rel.norm(dim=-1).clamp(min=1.0)  # [E, R, R]

        # Global azimuth/elevation from TX to RX
        az_tx_to_rx = torch.atan2(rel[..., 1], rel[..., 0]).rad2deg()
        el_tx_to_rx = torch.atan2(rel[..., 2], dist).rad2deg()

        # Relative angles: TX boresight → RX direction
        tx_rel_az = _wrapped_diff(
            az_tx_to_rx, beam_az.unsqueeze(1),
        )  # [E, R_tx, R_rx]
        tx_rel_el = el_tx_to_rx - beam_el.unsqueeze(1)

        # Relative angles: RX boresight → TX direction (opposite)
        rx_rel_az = _wrapped_diff(
            az_tx_to_rx + 180.0, beam_az.unsqueeze(2),
        )  # [E, R_tx, R_rx] → RX j seeing TX i
        rx_rel_el = -el_tx_to_rx - beam_el.unsqueeze(2)

        # TX gain: evaluate each TX radar's pattern toward every RX radar
        tx_gain_db = torch.zeros(E, R, R, device=dev)
        rx_gain_db = torch.zeros(E, R, R, device=dev)
        for i in range(R):
            w_i = weights[:, i, :].unsqueeze(1).expand(E, R, N)
            tx_gain_db[:, i, :] = self._array_gain_db_batch(
                w_i, tx_rel_az[:, i, :], tx_rel_el[:, i, :],
            )
        for j in range(R):
            w_j = weights[:, j, :].unsqueeze(1).expand(E, R, N)
            rx_gain_db[:, :, j] = self._array_gain_db_batch(
                w_j, rx_rel_az[:, :, j], rx_rel_el[:, :, j],
            )

        # Path loss: [E, R, R]
        path_loss_db = 20.0 * torch.log10(
            4.0 * np.pi * dist / self.wavelength + 1e-10,
        )

        tx_power_dbm = self.tx_power_dbm

        # Received interference power per (TX i, RX j) pair
        rx_power_dbm = (
            tx_power_dbm + tx_gain_db + rx_gain_db
            - path_loss_db - self.polarization_loss_db
        )
        rx_power_w = 10.0 ** ((rx_power_dbm - 30.0) / 10.0)
        rx_amplitude = torch.sqrt(rx_power_w.clamp(min=0.0))  # [E, R_tx, R_rx]

        # Build IQ interference signal into pre-allocated buffer
        out.zero_()
        for i in range(R):
            for j in range(R):
                if i == j:
                    continue
                delay_samples = (dist[:, i, j] / SPEED_OF_LIGHT * self.fs).long()
                amp = rx_amplitude[:, i, j]  # [E]

                for e in range(E):
                    d = int(delay_samples[e].item())
                    if d >= S:
                        continue
                    a = float(amp[e].item())
                    if a < 1e-15:
                        continue
                    out[e, j, :, d:] += a * baseband[:S - d].unsqueeze(0)
