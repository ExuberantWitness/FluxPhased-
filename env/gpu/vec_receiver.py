"""Batched radar receiver processing for num_envs parallel environments.

torch.fft for multi-dim batched matched filter + Doppler FFT (cuFFT backend).
Warp kernel for batched 2D CA-CFAR on range-Doppler power maps.
Uses wp.from_torch for zero-copy GPU memory sharing.
"""

import numpy as np
import warp as wp
import torch

SPEED_OF_LIGHT = 299792458.0


@wp.kernel
def _ca_cfar_batched(
    rd_power: wp.array3d(dtype=wp.float32),   # [E*R, n_doppler, n_range]
    guard_cells: wp.int32,
    train_cells: wp.int32,
    alpha: wp.float32,
    detections: wp.array3d(dtype=wp.float32),  # [E*R, n_doppler, n_range]
):
    """2D CA-CFAR on range-Doppler power maps, batched across env×radar.

    3D launch: (E*R, n_doppler, n_range).  er = env-radar pair, d = doppler, r = range.
    """
    er, d, r = wp.tid()
    n_doppler = rd_power.shape[1]
    n_range = rd_power.shape[2]

    half = guard_cells + train_cells
    if d < half or d >= n_doppler - half:
        return
    if r < half or r >= n_range - half:
        return

    noise_sum = wp.float32(0.0)
    count = wp.int32(0)

    for dd in range(d - half, d + half + 1):
        for rr in range(r - half, r + half + 1):
            if wp.abs(dd - d) <= guard_cells and wp.abs(rr - r) <= guard_cells:
                continue
            noise_sum += rd_power[er, dd, rr]
            count += 1

    if count > 0:
        noise_est = noise_sum / wp.float32(count)
        threshold = alpha * noise_est
        if rd_power[er, d, r] > threshold and rd_power[er, d, r] > wp.float32(1e-15):
            detections[er, d, r] = wp.float32(1.0)


class VecReceiver:
    """Batched receiver for E parallel environments, each with R radars.

    Pre-allocates GPU buffers; process_all() runs MF + Doppler + CFAR in one call.
    """

    def __init__(
        self, fc: float = 10e9, bandwidth: float = 200e6,
        prf: float = 10e3, pulses_per_cpi: int = 32,
        num_envs: int = 10, n_radars: int = 4,
        pfa: float = 1e-6, device: str = "cuda",
    ):
        self.fc = fc
        self.bandwidth = bandwidth
        self.fs = bandwidth
        self.prf = prf
        self.pulses_per_cpi = pulses_per_cpi
        self.num_envs = num_envs
        self.n_radars = n_radars
        self.device = device

        self.range_res = SPEED_OF_LIGHT / (2.0 * self.fs)
        self.doppler_bin_hz = prf / pulses_per_cpi
        self.doppler_res = self.doppler_bin_hz * SPEED_OF_LIGHT / (2.0 * fc)

        # CFAR parameters
        guard_cells = 4
        train_cells = 16
        n_train = 2 * train_cells
        self._cfar_alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
        self._cfar_guard = guard_cells
        self._cfar_train = train_cells

        ER = num_envs * n_radars
        # Pre-allocated CFAR detections buffer [E*R, n_doppler, n_range]
        # Size depends on FFT padding; allocate max expected size
        self._det_buf = None  # allocated on first call (depends on FFT size)

    def process_all(
        self, pulse_train: torch.Tensor, waveform: torch.Tensor,
    ) -> list:
        """Process CPI for all envs and radars simultaneously.

        Args:
            pulse_train: [E, R, n_pulses, n_samples] complex64
            waveform:    [n_filter] complex64 baseband reference
        Returns:
            List of lists: detections[env_idx][radar_idx] = list of detection dicts
        """
        E, R, n_pulses, n_samples = pulse_train.shape
        n_filter = waveform.shape[0]

        # 1. Batched matched filter (frequency domain, FFT on last dim)
        n_fft = 1
        while n_fft < n_samples + n_filter - 1:
            n_fft *= 2

        mf_ref = torch.fft.fft(waveform.conj().flip(0), n=n_fft)
        # pulse_train: [E, R, n_pulses, n_samples]
        sig_fft = torch.fft.fft(pulse_train, n=n_fft, dim=-1)
        range_profile = torch.fft.ifft(
            sig_fft * mf_ref.view(1, 1, 1, -1), dim=-1,
        )
        range_profile = range_profile[..., :n_samples]  # trim

        # 2. Doppler FFT across pulses (dim=-2)
        n_doppler = 1
        while n_doppler < n_pulses:
            n_doppler *= 2
        rd_map = torch.fft.fft(range_profile, n=n_doppler, dim=-2)
        rd_map = torch.fft.fftshift(rd_map, dim=-2)

        rd_power = torch.abs(rd_map) ** 2  # [E, R, n_doppler, n_samples]

        # 3. Batched CFAR via Warp kernel
        detections = self._cfar_detect_batched(rd_power, n_doppler, n_samples)

        # 4. Extract detections per env-radar (numpy on CPU, ~4ms for 10 envs)
        return self._extract_all(rd_power, detections)

    def _cfar_detect_batched(
        self, rd_power: torch.Tensor, n_doppler: int, n_range: int,
    ) -> np.ndarray:
        """Run batched 2D CA-CFAR. Returns [E*R, n_doppler, n_range] bool."""
        E, R = self.num_envs, self.n_radars
        ER = E * R

        # Flatten env×radar into first dim
        rd_flat = rd_power.reshape(ER, n_doppler, n_range)
        # Ensure contiguous float32
        rd_contig = rd_flat.contiguous()

        # Allocate or reuse detection buffer
        if self._det_buf is None or self._det_buf.shape != (ER, n_doppler, n_range):
            self._det_buf = torch.zeros(
                ER, n_doppler, n_range, dtype=torch.float32,
                device=torch.device(self.device),
            )
        else:
            self._det_buf.zero_()

        rd_wp = wp.from_torch(rd_contig)
        det_wp = wp.from_torch(self._det_buf)

        wp.launch(
            _ca_cfar_batched,
            dim=(ER, n_doppler, n_range),
            inputs=[
                rd_wp,
                self._cfar_guard, self._cfar_train,
                wp.float32(self._cfar_alpha),
                det_wp,
            ],
            device=self.device,
        )

        # Copy to CPU numpy for extraction
        return self._det_buf.cpu().numpy()

    def _extract_all(
        self, rd_power: torch.Tensor, detections: np.ndarray,
    ) -> list:
        """Extract target detections for all env-radar pairs (vectorized).

        Iterates only over CFAR-flagged cells (typically sparse) rather than
        the full (n_doppler x n_range) grid. Local-max check uses tensor ops.
        """
        E, R = self.num_envs, self.n_radars
        _, _, n_doppler, n_range = rd_power.shape

        rd_np = rd_power.cpu().numpy()
        doppler_center = n_doppler // 2

        # Edge-based noise floor: [E, R]
        edge_top = rd_np[:, :, :4, :].reshape(E, R, -1)
        edge_bot = rd_np[:, :, -4:, :].reshape(E, R, -1)
        edge_left = rd_np[:, :, :, :4].reshape(E, R, -1)
        edge_right = rd_np[:, :, :, -4:].reshape(E, R, -1)
        edges = np.concatenate([edge_top, edge_bot, edge_left, edge_right], axis=-1)
        noise_floor = edges.mean(axis=-1)
        noise_floor = np.maximum(noise_floor, 1e-15)  # [E, R]

        all_detections = []
        for e in range(E):
            env_dets = []
            for r_idx in range(R):
                er = e * R + r_idx
                det_mask = detections[er]
                rd = rd_np[e, r_idx]
                nf = noise_floor[e, r_idx]

                # Indices of CFAR-flagged cells (excluding border)
                d_idx, r_idx_arr = np.where(det_mask >= 0.5)
                interior = (
                    (d_idx >= 1) & (d_idx < n_doppler - 1)
                    & (r_idx_arr >= 1) & (r_idx_arr < n_range - 1)
                )
                d_idx = d_idx[interior]
                r_idx_arr = r_idx_arr[interior]

                dets = []
                if d_idx.size > 0:
                    # Local-max check: compare cell to 8-neighbours via stack
                    center = rd[d_idx, r_idx_arr]
                    is_max = np.ones_like(center, dtype=bool)
                    for dd in (-1, 0, 1):
                        for rr in (-1, 0, 1):
                            if dd == 0 and rr == 0:
                                continue
                            neighbor = rd[d_idx + dd, r_idx_arr + rr]
                            is_max &= center >= neighbor

                    d_idx = d_idx[is_max]
                    r_idx_arr = r_idx_arr[is_max]
                    center = center[is_max]

                    snr_db = 10.0 * np.log10(center / nf)
                    range_m = r_idx_arr * self.range_res
                    doppler_bin = d_idx - doppler_center
                    doppler_hz = doppler_bin * self.doppler_bin_hz
                    velocity_mps = doppler_hz * SPEED_OF_LIGHT / (2.0 * self.fc)

                    # Sort by SNR descending
                    order = np.argsort(-snr_db)[:50]
                    for k in order:
                        dets.append({
                            "range_m": float(range_m[k]),
                            "velocity_mps": float(velocity_mps[k]),
                            "snr_db": float(snr_db[k]),
                            "range_bin": int(r_idx_arr[k]),
                            "doppler_bin": int(d_idx[k]),
                            "doppler_hz": float(doppler_hz[k]),
                        })

                env_dets.append(dets)
            all_detections.append(env_dets)

        return all_detections
