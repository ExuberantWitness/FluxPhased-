"""Adversarial game metrics (multi-episode aggregation).

Maps to document "干扰博弈效能评估":
  - Game outcomes: win rates, episode lengths, kill balance
  - Strategy stability: reward CV, win rate convergence
  - Generalization: cross-opponent and cross-scenario performance
"""

import numpy as np
from typing import Optional

from ..collectors.episode_collector import EpisodeData


class GameMetrics:
    """Multi-episode adversarial game evaluation."""

    def game_outcomes(self, episodes: list) -> dict:
        """Aggregate game-level outcomes across episodes.

        Args:
            episodes: list of EpisodeData
        Returns:
            dict with win rates, avg length, kill balance
        """
        n = len(episodes)
        if n == 0:
            return {}

        winners = []
        lengths = []
        red_kills = 0
        blue_kills = 0

        for ep in episodes:
            winners.append(ep.final_winner)
            lengths.append(ep.n_steps)
            if ep.kills is not None and ep.kills.shape[0] > 0:
                # kills: [T, E, n_teams, n_enemy]
                red_kills += ep.kills[:, :, 0].sum().item()  # Red team's kills
                blue_kills += ep.kills[:, :, 1].sum().item()

        red_wins = sum(1 for w in winners if w == 0)
        blue_wins = sum(1 for w in winners if w == 1)
        draws = sum(1 for w in winners if w == -1 or w not in (0, 1))

        return {
            "n_episodes": n,
            "win_rate_red": red_wins / n,
            "win_rate_blue": blue_wins / n,
            "draw_rate": draws / n,
            "avg_episode_length": float(np.mean(lengths)),
            "std_episode_length": float(np.std(lengths)),
            "total_red_kills": red_kills,
            "total_blue_kills": blue_kills,
            "kill_balance": (red_kills - blue_kills) / max(1, red_kills + blue_kills),
            "first_blood_rate_red": red_kills > 0 and red_kills >= blue_kills,
        }

    def strategy_stability(self, episodes: list) -> dict:
        """Measure strategy consistency across episodes.

        Args:
            episodes: list of EpisodeData
        Returns:
            dict with reward CV and win rate convergence metrics
        """
        n = len(episodes)
        if n == 0:
            return {}

        # Reward statistics
        red_rewards = []
        blue_rewards = []
        for ep in episodes:
            if ep.radar_rewards is not None and ep.radar_rewards.shape[0] > 0:
                # Sum rewards over episode for first radar (red team)
                red_total = ep.radar_rewards[:, 0, 0].sum().item()  # [T] → scalar
                blue_total = ep.radar_rewards[:, 0, 2].sum().item() if ep.radar_rewards.shape[2] > 2 else 0
                red_rewards.append(red_total)
                blue_rewards.append(blue_total)

        red_arr = np.array(red_rewards) if red_rewards else np.array([0.0])
        blue_arr = np.array(blue_rewards) if blue_rewards else np.array([0.0])

        def cv(arr):
            return float(np.std(arr) / max(1e-10, abs(np.mean(arr))))

        # Win rate convergence: rolling variance
        winners = np.array([ep.final_winner for ep in episodes])
        win_rate_rolling = np.cumsum(winners == 0) / np.arange(1, n + 1)
        win_rate_var = float(win_rate_rolling[-min(10, n):].var()) if n > 0 else 0.0

        return {
            "red_reward_mean": float(red_arr.mean()),
            "red_reward_std": float(red_arr.std()),
            "red_reward_cv": cv(red_arr),
            "blue_reward_mean": float(blue_arr.mean()),
            "blue_reward_cv": cv(blue_arr),
            "win_rate_convergence_var": win_rate_var,
        }

    def generalization(
        self,
        baseline_win_rate: float = 0.5,
        adversarial_win_rate: float = 0.5,
        scene_win_rates: Optional[list] = None,
        opponent_win_rates: Optional[list] = None,
    ) -> dict:
        """Generalization across opponents and scenarios.

        Args:
            baseline_win_rate: win rate against reference opponent
            adversarial_win_rate: win rate against strongest opponent
            scene_win_rates: list of win rates across different scenarios
            opponent_win_rates: list of win rates against different opponents
        Returns:
            dict with generalization metrics
        """
        result = {
            "adversarial_drop": baseline_win_rate - adversarial_win_rate,
            "baseline_win_rate": baseline_win_rate,
            "adversarial_win_rate": adversarial_win_rate,
        }

        if scene_win_rates:
            arr = np.array(scene_win_rates)
            result["scene_generalization_mean"] = float(arr.mean())
            result["scene_generalization_std"] = float(arr.std())
            result["scene_generalization_min"] = float(arr.min())

        if opponent_win_rates:
            arr = np.array(opponent_win_rates)
            result["opponent_generalization_mean"] = float(arr.mean())
            result["opponent_generalization_std"] = float(arr.std())
            result["opponent_generalization_min"] = float(arr.min())

        return result
