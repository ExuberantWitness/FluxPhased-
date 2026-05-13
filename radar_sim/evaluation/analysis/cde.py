"""CDE (Countermeasure Decision Effectiveness) composite metric.

Implements formulas (15)-(25) from the evaluation document.
Combines kill effectiveness, resource efficiency, decision quality,
and generalization into a single [0, 1] score.
"""

import numpy as np
from typing import Optional

from ..collectors.episode_collector import EpisodeData


class CDEMetric:
    """对抗决策有效率 (Countermeasure Decision Effectiveness).

    CDE = w1 * kill_effectiveness
        + w2 * resource_efficiency
        + w3 * decision_quality
        + w4 * generalization_score

    All components normalized to [0, 1].
    """

    def __init__(
        self,
        weights: tuple = (0.3, 0.2, 0.3, 0.2),
        kill_radius_excellent: float = 200.0,
        kill_radius_poor: float = 900.0,
        optimal_launch_steps: int = 5,
    ):
        self.w = weights
        self.kill_radius_excellent = kill_radius_excellent
        self.kill_radius_poor = kill_radius_poor
        self.optimal_launch_steps = optimal_launch_steps

    def _kill_effectiveness(self, episode_data: EpisodeData) -> float:
        """Kill rate + miss distance → [0, 1].

        kill_rate: 1.0 if any kill, 0.0 otherwise
        miss_distance: mapped from [excellent, poor] → [1.0, 0.0]
        """
        kills = episode_data.kills
        if kills is None or kills.shape[0] == 0:
            return 0.0

        kill_rate = float(kills.any().item())
        return kill_rate

    def _resource_efficiency(self, episode_data: EpisodeData) -> float:
        """Task allocation entropy → [0, 1].

        Maximum efficiency when task allocation has reasonable entropy
        (not all elements on one task).
        """
        task_ids = episode_data.task_ids
        if task_ids is None or task_ids.shape[0] == 0:
            return 0.5

        N = task_ids.shape[-1]
        max_entropy = np.log(4)

        counts = np.array([
            (task_ids == t).float().mean().item() for t in range(4)
        ])
        counts = np.clip(counts, 1e-10, None)
        entropy = -np.sum(counts * np.log(counts))
        normalized = entropy / max_entropy

        return float(normalized)

    def _decision_quality(self, episode_data: EpisodeData) -> float:
        """Launch timing score → [0, 1].

        Earlier launch is better. Optimal = 1.0 at optimal_launch_steps,
        decays linearly to 0.0 at max_steps.
        """
        missile_pos = episode_data.missile_pos
        if missile_pos is None or missile_pos.shape[0] == 0:
            return 0.0

        # Find launch step
        launched = (missile_pos.abs().sum(dim=-1) > 0)  # [T, E, n_teams]
        if not launched.any():
            return 0.0

        first_launch = launched.float().argmax(dim=0).min().item()
        if not launched[first_launch].any():
            return 0.0

        # Score: 1.0 at optimal, decaying
        max_steps = episode_data.n_steps
        if max_steps <= 0:
            return 0.0

        score = 1.0 - (first_launch - self.optimal_launch_steps) / max(1, max_steps)
        return float(np.clip(score, 0.0, 1.0))

    def _generalization_score(
        self,
        game_metrics: Optional[dict] = None,
    ) -> float:
        """Win rate as generalization proxy → [0, 1].

        Defaults to 0.5 (neutral) if no game metrics provided.
        """
        if game_metrics is None:
            return 0.5

        wr = game_metrics.get("win_rate_red", 0.5)
        return float(np.clip(wr, 0.0, 1.0))

    def compute(
        self,
        episode_data: EpisodeData,
        game_metrics: Optional[dict] = None,
    ) -> dict:
        """Compute CDE and sub-metrics.

        Args:
            episode_data: collected trajectory data
            game_metrics: optional multi-episode game outcomes
        Returns:
            dict with cde (0-1) and sub-metrics
        """
        kill_eff = self._kill_effectiveness(episode_data)
        resource_eff = self._resource_efficiency(episode_data)
        decision_q = self._decision_quality(episode_data)
        gen_score = self._generalization_score(game_metrics)

        cde = (
            self.w[0] * kill_eff
            + self.w[1] * resource_eff
            + self.w[2] * decision_q
            + self.w[3] * gen_score
        )

        return {
            "cde": float(np.clip(cde, 0.0, 1.0)),
            "kill_effectiveness": kill_eff,
            "resource_efficiency": resource_eff,
            "decision_quality": decision_q,
            "generalization_score": gen_score,
            "weights": self.w,
        }

    def fitness(
        self,
        episode_data: EpisodeData,
        tolerance_bounds: Optional[dict] = None,
    ) -> float:
        """Fitness function for accelerated evaluation (document formula 25).

        Fitness = sum(|metric_i - tolerance_i|) for each sub-metric.
        Lower fitness = closer to tolerance boundary = higher scenario value.

        Args:
            episode_data: collected trajectory data
            tolerance_bounds: dict of metric → tolerance value
        Returns:
            fitness value (lower = more valuable scenario)
        """
        metrics = self.compute(episode_data)

        if tolerance_bounds is None:
            # Default tolerances: all metrics should be > 0.5
            tolerance_bounds = {
                "kill_effectiveness": 0.5,
                "resource_efficiency": 0.3,
                "decision_quality": 0.5,
                "generalization_score": 0.5,
            }

        fitness = 0.0
        for key, bound in tolerance_bounds.items():
            if key in metrics:
                fitness += abs(metrics[key] - bound)

        return fitness
