"""Dense reward shaping from MFARVecEnv step outputs.

Computes per-task intermediate rewards from existing simulation data:
- Detection SNR and coverage (from spectrum + matched filtering)
- Jamming effectiveness (from interference power)
- Communication reliability (from BPSK CRC results)
- Reconnaissance intelligence (from spectrum energy detection)

All rewards are shaped to [0, 1] range, then scaled by configurable weights.
"""

import torch
import numpy as np


TASK_RECON = 0
TASK_DETECT = 1
TASK_JAM = 2
TASK_COMM = 3


class DenseRewardShaper:
    """Compute dense intermediate rewards from MFARVecEnv step() output dict."""

    def __init__(
        self,
        detect_snr_weight: float = 0.1,
        detect_coverage_weight: float = 0.05,
        jam_effectiveness_weight: float = 0.1,
        comm_reliability_weight: float = 0.05,
        recon_intel_weight: float = 0.03,
        beam_accuracy_weight: float = 0.02,
        snr_threshold_db: float = 10.0,
        device: str = "cuda",
    ):
        self.detect_snr_weight = detect_snr_weight
        self.detect_coverage_weight = detect_coverage_weight
        self.jam_effectiveness_weight = jam_effectiveness_weight
        self.comm_reliability_weight = comm_reliability_weight
        self.recon_intel_weight = recon_intel_weight
        self.beam_accuracy_weight = beam_accuracy_weight
        self.snr_threshold_db = snr_threshold_db
        self.device = device

    def __call__(self, step_output: dict) -> dict:
        """Compute shaped rewards from one env step.

        Args:
            step_output: dict from MFARVecEnv.step(), must contain:
                - spectrum: [E, R, N, P, n_bins] float32
                - comm_data: [E, R, N, 2] float32
                - task_ids: [E, R, N] int64
                - comm_crc_ok: [E, n_teams] bool (optional)
                - channel_params: tuple (delay, doppler, gain) (optional)
                - detect_params: [E, R, N, 3] (optional)
                - jam_params: [E, R, N, 3] (optional)
        Returns:
            dict with per-radar shaped rewards [E, R] and individual components.
        """
        spectrum = step_output["spectrum"]       # [E, R, N, P, n_bins]
        task_ids = step_output["task_ids"]       # [E, R, N]
        E, R, N = task_ids.shape
        dev = spectrum.device

        # --- Detection reward ---
        detect_reward = self._detect_reward(spectrum, task_ids)

        # --- Jamming reward ---
        jam_reward = self._jam_reward(spectrum, task_ids)

        # --- Communication reward ---
        comm_reward = self._comm_reward(step_output, task_ids)

        # --- Reconnaissance reward ---
        recon_reward = self._recon_reward(spectrum, task_ids)

        total = (
            detect_reward * self.detect_snr_weight
            + jam_reward * self.jam_effectiveness_weight
            + comm_reward * self.comm_reliability_weight
            + recon_reward * self.recon_intel_weight
        )

        return {
            "detect_reward": detect_reward,
            "jam_reward": jam_reward,
            "comm_reward": comm_reward,
            "recon_reward": recon_reward,
            "total_shaped": total,
        }

    def _detect_reward(self, spectrum: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Detection quality: SNR above threshold, normalized by number of detect elements."""
        E, R, N, P, B = spectrum.shape
        detect_mask = (task_ids == TASK_DETECT)  # [E, R, N]
        n_detect = detect_mask.sum(dim=-1).clamp(min=1).float()  # [E, R]

        # Peak power across frequency bins for detect elements
        detect_spectrum = spectrum * detect_mask.unsqueeze(-1).unsqueeze(-1).float()
        peak_power = detect_spectrum.amax(dim=-1).amax(dim=-1)  # [E, R, N]

        # Noise floor: median of non-peak bins
        noise_floor = spectrum.median(dim=-1).values.median(dim=-1).values.clamp(min=1e-30)

        # SNR in dB for detect elements
        snr_db = 10.0 * torch.log10(peak_power.clamp(min=1e-30) / noise_floor.clamp(min=1e-30))
        snr_db = snr_db * detect_mask.float()

        # Reward: fraction of detect elements above threshold
        above_thresh = (snr_db > self.snr_threshold_db).float()
        coverage = above_thresh.sum(dim=-1) / n_detect  # [E, R]

        # Average SNR above threshold
        avg_snr = (snr_db - self.snr_threshold_db).clamp(min=0).sum(dim=-1) / (n_detect * 20.0)

        return coverage * 0.5 + avg_snr * 0.5

    def _jam_reward(self, spectrum: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Jam effectiveness: total power emitted by jam elements, normalized."""
        E, R, N, P, B = spectrum.shape
        jam_mask = (task_ids == TASK_JAM).float()  # [E, R, N]
        n_jam = jam_mask.sum(dim=-1).clamp(min=1)  # [E, R]

        # Jam reward is based on proportion of elements allocated to jamming
        # (proxy for jamming investment — actual effectiveness requires cross-team measurement)
        jam_fraction = jam_mask.sum(dim=-1) / N  # [E, R]

        # Energy in spectrum for jam elements (transmitted power proxy)
        jam_energy = (spectrum.mean(dim=-1).mean(dim=-1) * jam_mask).sum(dim=-1)
        jam_energy_norm = jam_energy / (jam_energy.amax() + 1e-10)

        return 0.3 * jam_fraction + 0.7 * jam_energy_norm

    def _comm_reward(self, step_output: dict, task_ids: torch.Tensor) -> torch.Tensor:
        """Communication reliability: CRC pass rate for own team's comm link."""
        E, R, N = task_ids.shape
        dev = task_ids.device

        crc_ok = step_output.get("comm_crc_ok")
        if crc_ok is None:
            return torch.zeros(E, R, device=dev)

        # crc_ok: [E, n_teams] bool — expand to [E, R]
        team_id = torch.tensor(
            [i // (R // 2) for i in range(R)], device=dev,
        )  # [R]
        crc_per_radar = crc_ok[:, team_id].float()  # [E, R]

        # Weight by comm element fraction
        comm_mask = (task_ids == TASK_COMM).float()
        comm_fraction = comm_mask.sum(dim=-1) / task_ids.shape[-1]  # [E, R]

        return crc_per_radar * (0.5 + 0.5 * comm_fraction)

    def _recon_reward(self, spectrum: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Reconnaissance intelligence: energy detected by recon elements."""
        E, R, N, P, B = spectrum.shape
        recon_mask = (task_ids == TASK_RECON).float()  # [E, R, N]
        n_recon = recon_mask.sum(dim=-1).clamp(min=1)  # [E, R]

        # Total energy received by recon elements
        recon_energy = (spectrum.mean(dim=-1).mean(dim=-1) * recon_mask).sum(dim=-1)
        max_energy = recon_energy.amax() + 1e-10

        return (recon_energy / max_energy).clamp(0, 1)
