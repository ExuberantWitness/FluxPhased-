"""BPSK communication quality metrics.

Evaluates the radar-to-missile BPSK communication link quality.

Maps to document "通信中断率" and communication reliability metrics.
"""

import torch
import numpy as np
from typing import Optional

from ..collectors.episode_collector import EpisodeData


class CommMetrics:
    """BPSK communication link quality metrics."""

    def comm_accuracy(
        self,
        comm_data: torch.Tensor,
        commander_target: Optional[torch.Tensor] = None,
    ) -> dict:
        """Evaluate BPSK decoded data quality.

        Args:
            comm_data: [..., N, 2] decoded (X, Y) per comm element
            commander_target: [..., 2] intended target (X, Y) for comparison
        Returns:
            dict with comm_error, active_frac
        """
        # Active comm elements (non-zero data)
        active = (comm_data.abs().sum(dim=-1) > 1e-6)  # [..., N]
        active_frac = active.float().mean().item()

        if commander_target is not None and active_frac > 0:
            # Compare decoded vs intended
            error = (comm_data - commander_target.unsqueeze(-2)).norm(dim=-1)
            # Only count active elements
            if active.any():
                mean_error = error[active].mean().item()
                max_error = error[active].max().item()
            else:
                mean_error = float("inf")
                max_error = float("inf")
        else:
            mean_error = float("nan")
            max_error = float("nan")

        return {
            "active_comm_frac": active_frac,
            "comm_error_mean": mean_error,
            "comm_error_max": max_error,
            "crc_pass_rate_est": active_frac,  # proxy: non-zero ≈ CRC passed
        }

    def comm_accuracy_over_episode(
        self,
        episode_data: EpisodeData,
    ) -> dict:
        """Track comm accuracy across episode steps.

        Args:
            episode_data: collected trajectory data
        Returns:
            dict with per-step and aggregate comm metrics
        """
        if episode_data.comm_data is None or episode_data.comm_data.shape[0] == 0:
            return {"mean_active_frac": 0.0}

        T = episode_data.comm_data.shape[0]
        per_step = []
        for t in range(T):
            step_metrics = self.comm_accuracy(episode_data.comm_data[t])
            per_step.append(step_metrics)

        active_fracs = [s["active_comm_frac"] for s in per_step]

        return {
            "per_step": per_step,
            "mean_active_frac": float(np.mean(active_fracs)),
            "std_active_frac": float(np.std(active_fracs)),
            "min_active_frac": float(np.min(active_fracs)),
        }

    def comm_vs_task_allocation(
        self,
        episode_data: EpisodeData,
    ) -> dict:
        """Correlate comm element allocation with comm data quality.

        Args:
            episode_data: collected trajectory data
        Returns:
            dict with comm element fraction and corresponding data quality
        """
        if episode_data.task_ids is None or episode_data.comm_data is None:
            return {}

        if episode_data.task_ids.shape[0] == 0:
            return {}

        comm_frac = (episode_data.task_ids == 3).float().mean().item()
        comm_metrics = self.comm_accuracy_over_episode(episode_data)

        return {
            "comm_element_fraction": comm_frac,
            "comm_data_quality": comm_metrics,
        }
