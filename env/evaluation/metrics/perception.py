"""Radar perception effectiveness metrics.

Evaluates detection quality by comparing spectrum peaks against expected
range/doppler from the simulation's channel model. Simulation IS ground truth.

Maps to document "感知效能评估":
  - Accuracy: range_accuracy, doppler_accuracy
  - Completeness: target_coverage, false_alarm_rate
  - Real-time: timing_metrics
  - Robustness: per_snr_accuracy (SNR-stratified accuracy)
"""

import torch
import numpy as np
from typing import Optional


class PerceptionMetrics:
    """Computes radar detection performance metrics from per-step data."""

    def __init__(self, env):
        self.env = env
        self.n_bins = env.n_bins
        self.n_pulses = env.n_pulses
        self.prf = env.prf
        self.fs = env.fs

    def beamformed_spectrum(
        self,
        spectrum: torch.Tensor,
        task_ids: torch.Tensor,
        task: int = 1,
    ) -> torch.Tensor:
        """Sum spectra across elements assigned to a given task.

        Args:
            spectrum: [E, R, N, P, n_bins] or [T, E, R, N, P, n_bins]
            task_ids: [E, R, N] or [T, E, R, N]
            task: task ID to select (1=detect, 0=recon)
        Returns:
            [E, R, P, n_bins] or [T, E, R, P, n_bins] beamformed RDM
        """
        mask = (task_ids == task).float()
        # Expand mask to match spectrum dims
        while mask.dim() < spectrum.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(spectrum)
        return (spectrum * mask).sum(dim=2)  # sum over N (element dim)

    def spectrum_peaks(
        self,
        spectrum: torch.Tensor,
        noise_floor_pct: float = 10.0,
    ) -> dict:
        """Extract peaks from per-element spectra.

        Args:
            spectrum: [..., n_bins] (any leading dims)
        Returns:
            dict with:
              peak_bin: [...] long tensor of peak bin index
              peak_power: [...] float tensor of peak power
              noise_floor: [...] float tensor of estimated noise floor
              snr_est_db: [...] float tensor of estimated SNR in dB
        """
        # Noise floor: use bottom percentile of bins
        sorted_spec, _ = spectrum.sort(dim=-1)
        n_pct = max(1, int(spectrum.shape[-1] * noise_floor_pct / 100))
        noise_floor = sorted_spec[..., :n_pct].mean(dim=-1)
        peak_power, peak_bin = spectrum.max(dim=-1)
        noise_clamped = noise_floor.clamp(min=1e-30)
        snr_est_db = 10.0 * torch.log10((peak_power / noise_clamped).clamp(min=1e-30))
        return {
            "peak_bin": peak_bin,
            "peak_power": peak_power,
            "noise_floor": noise_floor,
            "snr_est_db": snr_est_db,
        }

    def detection_accuracy(
        self,
        spectrum: torch.Tensor,
        expected_range_bin: torch.Tensor,
        expected_doppler_bin: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        range_tol_bins: int = 5,
    ) -> dict:
        """Fraction of (env, radar) pairs with spectrum peak near expected range.

        Args:
            spectrum: [E, R, N, P, n_bins] per-element spectra
            expected_range_bin: [E, R, n_targets] expected peak location
            expected_doppler_bin: [E, R, n_targets] optional doppler check
            task_ids: [E, R, N] optional, for beamforming
            range_tol_bins: tolerance in bins
        Returns:
            dict with range_accuracy, target_coverage, per_snr_accuracy
        """
        # Beamform detect elements if task_ids provided
        if task_ids is not None:
            rdm = self.beamformed_spectrum(spectrum, task_ids, task=1)
        else:
            rdm = spectrum.float().mean(dim=2)  # average over elements

        # rdm: [E, R, P, n_bins] — average over pulses for range-only
        range_profile = rdm.mean(dim=-2)  # [E, R, n_bins]
        _, peak_bin = range_profile.max(dim=-1)  # [E, R]

        # For first target only (most common case)
        exp_bin = expected_range_bin[:, :, 0]  # [E, R]

        range_correct = (peak_bin - exp_bin).abs() <= range_tol_bins
        range_accuracy = range_correct.float().mean().item()

        # Target coverage: fraction of (env, radar) that detect the target
        target_coverage = range_correct.float().mean().item()

        # Doppler accuracy if provided
        doppler_accuracy = None
        if expected_doppler_bin is not None:
            # Average over range bins near the peak
            doppler_profile = rdm.mean(dim=-1)  # [E, R, P]
            _, peak_pulse = doppler_profile.max(dim=-1)  # [E, R]
            exp_dop = expected_doppler_bin[:, :, 0]
            n_pulses = rdm.shape[-2]
            doppler_center = n_pulses // 2
            peak_dop_bin = peak_pulse - doppler_center
            dop_correct = (peak_dop_bin - exp_dop).abs() <= 3
            doppler_accuracy = dop_correct.float().mean().item()

        return {
            "range_accuracy": range_accuracy,
            "target_coverage": target_coverage,
            "doppler_accuracy": doppler_accuracy,
            "peak_bin_mean": peak_bin.float().mean().item(),
            "expected_bin_mean": exp_bin.float().mean().item(),
        }

    def false_alarm_rate(
        self,
        spectrum: torch.Tensor,
        expected_range_bin: torch.Tensor,
        guard_bins: int = 3,
    ) -> dict:
        """Fraction of false peaks (peak far from expected target).

        Args:
            spectrum: [E, R, N, P, n_bins]
            expected_range_bin: [E, R, n_targets]
            guard_bins: bins around expected to consider as target region
        Returns:
            dict with false_alarm_rate, peak_to_sidelobe_ratio
        """
        rdm = spectrum.float().mean(dim=2)  # average over elements [E, R, P, n_bins]
        range_profile = rdm.mean(dim=-2)  # [E, R, n_bins]
        peak_power, peak_bin = range_profile.max(dim=-1)

        # Expected peak for first target
        exp_bin = expected_range_bin[:, :, 0]

        # False alarm: peak is not near expected target
        is_false = (peak_bin - exp_bin).abs() > guard_bins
        false_alarm_rate = is_false.float().mean().item()

        # Peak-to-sidelobe ratio
        sorted_powers, _ = range_profile.sort(dim=-1, descending=True)
        if sorted_powers.shape[-1] > 1:
            sidelobe = sorted_powers[..., 1]
            psr = (peak_power / sidelobe.clamp(min=1e-30)).mean().item()
        else:
            psr = float("inf")

        return {
            "false_alarm_rate": false_alarm_rate,
            "peak_to_sidelobe_ratio": psr,
        }

    def per_snr_accuracy(
        self,
        spectrum: torch.Tensor,
        expected_range_bin: torch.Tensor,
        expected_snr_db: torch.Tensor,
        snr_bins: list = (-10, 0, 10, 20, 30),
        range_tol_bins: int = 5,
    ) -> dict:
        """Detection accuracy stratified by expected SNR.

        Args:
            spectrum: [E, R, N, P, n_bins]
            expected_range_bin: [E, R, n_targets]
            expected_snr_db: [E, R, n_targets]
            snr_bins: SNR boundaries in dB
        Returns:
            dict mapping SNR range label → accuracy
        """
        rdm = spectrum.float().mean(dim=2).mean(dim=-2)  # [E, R, n_bins]
        _, peak_bin = rdm.max(dim=-1)  # [E, R]

        exp_bin = expected_range_bin[:, :, 0]  # [E, R]
        snr = expected_snr_db[:, :, 0]  # [E, R]

        correct = (peak_bin - exp_bin).abs() <= range_tol_bins

        result = {}
        for i in range(len(snr_bins) - 1):
            lo, hi = snr_bins[i], snr_bins[i + 1]
            mask = (snr >= lo) & (snr < hi)
            if mask.any():
                acc = correct[mask].float().mean().item()
            else:
                acc = float("nan")
            label = f"{lo}to{hi}dB"
            result[label] = acc

        # Also add overall
        if len(snr_bins) > 0:
            below = snr < snr_bins[0]
            if below.any():
                result[f"below{snr_bins[0]}dB"] = correct[below].float().mean().item()
            above = snr >= snr_bins[-1]
            if above.any():
                result[f"above{snr_bins[-1]}dB"] = correct[above].float().mean().item()

        return result

    def timing_metrics(self, timing_list: list) -> dict:
        """Aggregate timing data from episode.

        Args:
            timing_list: list of timing dicts from step() results
        Returns:
            dict with per-phase mean/max/p95 in milliseconds
        """
        if not timing_list:
            return {}

        keys = list(timing_list[0].keys())
        result = {}
        for key in keys:
            vals = [t.get(key, 0) for t in timing_list]
            arr = np.array(vals)
            result[key] = {
                "mean_ms": float(arr.mean()),
                "max_ms": float(arr.max()),
                "p95_ms": float(np.percentile(arr, 95)),
                "total_ms": float(arr.sum()),
            }
        return result
