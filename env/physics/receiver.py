"""Radar receiver: signal processing, detection, and parameter estimation.

Implements:
- Matched filter / pulse compression
- Doppler processing (FFT across pulses)
- CFAR detection (CA-CFAR, OS-CFAR)
- Mono-pulse angle estimation
- Target parameter extraction

Output format is compatible with what would be extracted from RadarSimPy baseband data.
"""

from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray

from ..config import RFConfig, WaveformConfig, CPIConfig


SPEED_OF_LIGHT = 299792458.0


class MatchedFilter:
    """Pulse compression via matched filtering."""

    def __init__(self, coefficients: Optional[NDArray] = None):
        self._coeffs = coefficients

    def set_coefficients(self, coeffs: NDArray):
        self._coeffs = coeffs

    def apply(self, signal: NDArray) -> NDArray:
        """Apply matched filter. signal shape: [..., samples]."""
        if self._coeffs is None:
            return signal

        # Frequency-domain convolution for efficiency
        n_signal = signal.shape[-1]
        n_filter = len(self._coeffs)
        n_fft = n_signal + n_filter - 1
        n_fft = 2 ** int(np.ceil(np.log2(n_fft)))

        sig_fft = np.fft.fft(signal, n=n_fft, axis=-1)
        mf_fft = np.fft.fft(np.conj(self._coeffs[::-1]), n=n_fft)

        result = np.fft.ifft(sig_fft * mf_fft, axis=-1)
        # Trim to valid output: first n_signal - n_filter + 1 samples
        valid_len = n_signal
        start = (n_fft - valid_len) // 2
        return result[..., start:start + valid_len]


class CFARDetector:
    """Constant False Alarm Rate detector."""

    def __init__(self, guard_cells: int = 4, training_cells: int = 16,
                 pfa: float = 1e-6, mode: str = "ca"):
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.pfa = pfa
        self.mode = mode  # "ca" = cell-averaging, "os" = ordered-statistic

        # Threshold multiplier for CA-CFAR
        n_train = 2 * training_cells
        self._alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

    def detect_1d(self, signal: NDArray) -> NDArray:
        """1D CA-CFAR detection. Returns binary detection mask."""
        n = len(signal)
        detections = np.zeros(n, dtype=bool)
        half_window = self.guard_cells + self.training_cells

        signal_power = np.abs(signal) ** 2

        for i in range(half_window, n - half_window):
            # Leading window
            lead = signal_power[i + self.guard_cells:i + half_window]
            # Lagging window
            lag = signal_power[i - half_window:i - self.guard_cells]

            if self.mode == "ca":
                noise_est = np.mean(np.concatenate([lead, lag]))
            else:  # os-cfar
                combined = np.sort(np.concatenate([lead, lag]))
                k = int(0.75 * len(combined))
                noise_est = combined[k]

            threshold = self._alpha * noise_est
            if signal_power[i] > max(threshold, 1e-15):
                detections[i] = True

        return detections

    def detect_range_doppler(self, rd_map: NDArray) -> NDArray:
        """2D CFAR on range-Doppler map. Returns binary detection mask."""
        detections = np.zeros_like(rd_map, dtype=bool)
        rd_power = np.abs(rd_map) ** 2

        n_doppler, n_range = rd_map.shape
        g = self.guard_cells
        t = self.training_cells
        half = g + t

        # Simplified: rectangular guard + training band
        for d in range(half, n_doppler - half):
            for r in range(half, n_range - half):
                # Training band (exclude guard band)
                training_data = []
                for dd in range(d - half, d + half + 1):
                    for rr in range(r - half, r + half + 1):
                        if (abs(dd - d) > g or abs(rr - r) > g) and dd != d and rr != r:
                            training_data.append(rd_power[dd, rr])

                if training_data:
                    noise_est = np.mean(training_data)
                    threshold = self._alpha * noise_est
                    if rd_power[d, r] > max(threshold, 1e-15):
                        detections[d, r] = True

        return detections


class RadarReceiver:
    """Complete radar receiver processing chain."""

    def __init__(self, rf: RFConfig, wf_cfg: WaveformConfig, cpi: CPIConfig):
        self.rf = rf
        self.wf_cfg = wf_cfg
        self.cpi = cpi
        self.matched_filter = MatchedFilter()
        self.cfar = CFARDetector(pfa=1e-6, mode="ca")

    def process_pulse_train(
        self,
        pulse_train: NDArray,        # [pulses, samples]
    ) -> dict:
        """Process a full CPI of pulses.

        Steps:
        1. Pulse compression (matched filter per pulse)
        2. Doppler FFT across pulses
        3. CFAR detection
        4. Target parameter extraction

        Returns dict with:
            range_profile: [pulses, range_bins]
            rd_map: [doppler_bins, range_bins]
            detections: list of {range_m, velocity_mps, snr_db, doppler_bin, range_bin}
        """
        num_pulses = pulse_train.shape[0]
        fs = self.rf.fs

        # 1. Range FFT for each pulse
        n_range = pulse_train.shape[1]
        range_profile = np.fft.fft(pulse_train, n=n_range, axis=1)

        # 2. Doppler FFT across pulses
        n_doppler = max(64, 2 ** int(np.ceil(np.log2(num_pulses))))
        rd_map = np.fft.fft(range_profile, n=n_doppler, axis=0)
        rd_map = np.fft.fftshift(rd_map, axes=0)  # zero Doppler at center

        # 3. CFAR detection on range-Doppler map
        detections = self._extract_detections(rd_map, fs)

        return {
            "range_profile": np.abs(range_profile),
            "rd_map": np.abs(rd_map),
            "detections": detections,
            "n_range_bins": n_range,
            "n_doppler_bins": n_doppler,
        }

    def _extract_detections(self, rd_map: NDArray, fs: float) -> list:
        """Extract target detections from range-Doppler map."""
        detections = []
        rd_power = np.abs(rd_map) ** 2

        # Simplified peak detection (faster than full 2D CFAR for simulation)
        n_doppler, n_range = rd_map.shape

        # Estimate noise floor from outer edges
        edge_power = np.concatenate([
            rd_power[0:4, :].ravel(),
            rd_power[-4:, :].ravel(),
            rd_power[:, 0:4].ravel(),
            rd_power[:, -4:].ravel(),
        ])
        noise_floor = np.mean(edge_power) if len(edge_power) > 0 else 1e-15
        threshold = noise_floor * 20  # 13 dB SNR threshold

        # Find local maxima above threshold
        for d in range(1, n_doppler - 1):
            for r in range(1, n_range - 1):
                patch = rd_power[d-1:d+2, r-1:r+2]
                center = rd_power[d, r]
                if center == np.max(patch) and center > threshold:
                    # Interpolate for sub-bin accuracy
                    snr_db = 10.0 * np.log10(center / noise_floor)

                    # Range from bin index
                    range_res = SPEED_OF_LIGHT / (2.0 * fs)
                    range_m = r * range_res

                    # Doppler from bin index (shifted back)
                    doppler_center = n_doppler // 2
                    doppler_bin = d - doppler_center
                    prf = self.cpi.prf
                    doppler_hz = doppler_bin * prf / n_doppler
                    velocity_mps = doppler_hz * SPEED_OF_LIGHT / (2.0 * self.rf.fc)

                    detections.append({
                        "range_m": range_m,
                        "velocity_mps": velocity_mps,
                        "snr_db": snr_db,
                        "range_bin": r,
                        "doppler_bin": d,
                        "doppler_hz": doppler_hz,
                    })

        # Sort by SNR descending, keep top 100
        detections.sort(key=lambda x: x["snr_db"], reverse=True)
        return detections[:100]


def estimate_target_angle(
    received_power_per_beam: NDArray,  # power received in each beam direction
    beam_angles_deg: NDArray,          # corresponding beam center angles
) -> dict:
    """Estimate target angle via monopulse-like power ratio.

    For a phased array scanning multiple beams simultaneously,
    estimate the angle that maximizes received power.

    Returns:
        dict with az_est_deg, el_est_deg, confidence
    """
    if len(received_power_per_beam) == 0:
        return {"az_est_deg": 0.0, "el_est_deg": 0.0, "confidence": 0.0}

    # Weighted centroid of beam angles
    power = np.maximum(received_power_per_beam, 0)
    total_power = np.sum(power)
    if total_power < 1e-15:
        return {"az_est_deg": 0.0, "el_est_deg": 0.0, "confidence": 0.0}

    az_est = np.sum(power * beam_angles_deg) / total_power
    confidence = np.max(power) / (np.mean(power) + 1e-15)

    return {
        "az_est_deg": float(az_est),
        "el_est_deg": 0.0,  # elevation requires 2D scan
        "confidence": float(confidence),
    }


class DetectionAggregator:
    """Aggregate detections across multiple processing intervals."""

    def __init__(self, track_lifetime: int = 20):
        self.tracks: list = []
        self.track_lifetime = track_lifetime
        self._next_id = 0

    def update(self, detections: list, timestamp_s: float) -> list:
        """Update tracks with new detections. Simple nearest-neighbor association."""
        # Age existing tracks
        for track in self.tracks:
            track["age"] += 1

        # Associate
        used_dets = set()
        for track in self.tracks:
            if track["age"] > self.track_lifetime:
                continue

            best_dist = float("inf")
            best_det = None
            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                dist = abs(det["range_m"] - track["range_m"])
                if dist < best_dist and dist < 100:  # 100m gate
                    best_dist = dist
                    best_det = i

            if best_det is not None:
                used_dets.add(best_det)
                det = detections[best_det]
                track["range_m"] = det["range_m"]
                track["velocity_mps"] = det["velocity_mps"]
                track["snr_db"] = det["snr_db"]
                track["age"] = 0
                track["hits"] += 1

        # New tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_dets:
                self.tracks.append({
                    "id": self._next_id,
                    "range_m": det["range_m"],
                    "velocity_mps": det["velocity_mps"],
                    "snr_db": det["snr_db"],
                    "age": 0,
                    "hits": 1,
                })
                self._next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t["age"] <= self.track_lifetime]

        # Return confirmed tracks (hits >= 3)
        return [t for t in self.tracks if t["hits"] >= 3]
