"""Combat decision quality metrics.

Evaluates resource allocation, missile effectiveness, and commander decisions.

Maps to document "态势分析效能评估" + "干扰博弈效能评估" (partial):
  - Resource efficiency: task allocation fractions, entropy
  - Decision quality: target error, launch timing
  - Missile effectiveness: kill rate, miss distance
  - Threat assessment: targeting accuracy
"""

import torch
import numpy as np
from typing import Optional

from ..collectors.episode_collector import EpisodeData


class CombatMetrics:
    """Combat decision quality metrics from episode-level data."""

    TASK_NAMES = {0: "recon", 1: "detect", 2: "jam", 3: "comm"}

    def resource_allocation(self, task_ids: torch.Tensor) -> dict:
        """Compute element task distribution.

        Args:
            task_ids: [..., N] (any leading dims: [E, R, N] or [T, E, R, N])
        Returns:
            dict with per-task fractions and allocation entropy
        """
        N = task_ids.shape[-1]
        result = {}
        for task_id, name in self.TASK_NAMES.items():
            frac = (task_ids == task_id).float().mean().item()
            result[f"{name}_frac"] = frac

        # Allocation entropy: H = -sum(p * log(p))
        counts = torch.stack([
            (task_ids == t).float().sum(dim=-1) / N
            for t in range(4)
        ], dim=-1)  # [..., 4]
        counts = counts.clamp(min=1e-10)
        entropy = -(counts * counts.log()).sum(dim=-1).mean().item()
        max_entropy = np.log(4)
        result["task_entropy"] = entropy
        result["task_entropy_normalized"] = entropy / max_entropy if max_entropy > 0 else 0.0

        return result

    def resource_allocation_over_time(
        self,
        task_ids: torch.Tensor,
    ) -> dict:
        """Track task allocation changes over episode steps.

        Args:
            task_ids: [T, E, R, N]
        Returns:
            dict with per-step allocation fractions and stability score
        """
        T = task_ids.shape[0]
        per_step = []
        for t in range(T):
            step_alloc = self.resource_allocation(task_ids[t])
            per_step.append(step_alloc)

        # Stability: variance of detect fraction over time
        detect_fracs = [s["detect_frac"] for s in per_step]
        stability = float(np.std(detect_fracs))

        return {
            "per_step": per_step,
            "detect_stability_std": stability,
        }

    def missile_efficiency(
        self,
        episode_data: EpisodeData,
        radar_pos: Optional[torch.Tensor] = None,
        enemy_radar_pos: Optional[torch.Tensor] = None,
    ) -> dict:
        """Missile combat effectiveness metrics.

        Args:
            episode_data: collected trajectory data
            radar_pos: [E, R, 3] if None, uses final step positions
            enemy_radar_pos: [E, R_enemy, 3] derived from radar_pos
        Returns:
            dict with kill_rate, time_to_kill, missile stats
        """
        kills = episode_data.kills  # [T, E, n_teams, n_enemy]
        if kills is None or kills.shape[0] == 0:
            return {
                "kill_rate": 0.0,
                "time_to_kill_steps": -1,
                "any_kill": False,
                "kills_by_team": {},
            }

        T = kills.shape[0]

        # Any kill happened?
        any_kill = kills.any().item()

        # Time to first kill
        kill_steps = kills.any(dim=-1).any(dim=-1)  # [T, E]
        time_to_kill = -1
        if kill_steps.any():
            first_kill_step = kill_steps.float().argmax(dim=0).min().item()
            if kill_steps[first_kill_step].any():
                time_to_kill = first_kill_step

        # Kill rate per team
        n_teams = kills.shape[2]
        kills_by_team = {}
        for t in range(n_teams):
            team_kills = kills[:, :, t]  # [T, E, n_enemy]
            kills_by_team[f"team_{t}"] = {
                "total_kills": team_kills.sum().item(),
                "any_kill": team_kills.any().item(),
            }

        # Kill rate = episodes with at least one kill / total episodes
        ep_kills = kills.reshape(T, -1).any(dim=0)  # [E * n_teams * n_enemy] per step
        kill_rate = float(kills.any().item())

        return {
            "kill_rate": kill_rate,
            "time_to_kill_steps": time_to_kill,
            "any_kill": any_kill,
            "kills_by_team": kills_by_team,
            "total_kills": kills.sum().item(),
        }

    def commander_decision_quality(
        self,
        episode_data: EpisodeData,
        radar_pos: torch.Tensor,
    ) -> dict:
        """Evaluate commander target selection.

        Since we don't store commander_actions in EpisodeData,
        this analyzes the missile_pos trajectory to infer targeting quality.

        Args:
            episode_data: collected trajectory data
            radar_pos: [E, R, 3] radar positions
        Returns:
            dict with missile launch stats
        """
        missile_pos = episode_data.missile_pos  # [T, E, n_teams, 3]
        if missile_pos is None or missile_pos.shape[0] == 0:
            return {"launched": False}

        # Check if missile was launched (non-zero position)
        launched = (missile_pos.abs().sum(dim=-1) > 0)  # [T, E, n_teams]
        any_launched = launched.any().item()

        # Time to launch
        launch_step = -1
        if any_launched:
            first_launch = launched.float().argmax(dim=0).min().item()
            if launched[first_launch].any():
                launch_step = first_launch

        return {
            "launched": any_launched,
            "launch_step": launch_step,
            "launch_steps_per_team": {
                f"team_{t}": int(launched[:, 0, t].float().argmax().item())
                if launched[:, 0, t].any() else -1
                for t in range(missile_pos.shape[2])
            },
        }

    def threat_assessment(
        self,
        episode_data: EpisodeData,
        radar_pos: torch.Tensor,
    ) -> dict:
        """Evaluate threat targeting quality.

        Measures how close the missile trajectory gets to enemy radars.

        Args:
            episode_data: collected trajectory data
            radar_pos: [E, R, 3] radar positions
        Returns:
            dict with miss distances to enemy radars
        """
        missile_pos = episode_data.missile_pos  # [T, E, n_teams, 3]
        if missile_pos is None or missile_pos.shape[0] == 0:
            return {"min_miss_distance_m": float("inf")}

        T, E, n_teams = missile_pos.shape[:3]
        R = radar_pos.shape[1]
        r_per_team = R // n_teams

        min_distances = {}
        for team in range(n_teams):
            # This team's missile targets the other team's radars
            enemy_start = (1 - team) * r_per_team
            enemy_end = enemy_start + r_per_team
            enemy_pos = radar_pos[:, enemy_start:enemy_end, :]  # [E, r_per_team, 3]

            m_pos = missile_pos[:, :, team, :]  # [T, E, 3]
            if m_pos.abs().sum() == 0:
                continue

            # Distance from each missile position to each enemy radar
            # [T, E, 1, 3] vs [1, E, r_per_team, 3]
            dist = (m_pos.unsqueeze(2) - enemy_pos.unsqueeze(0)).norm(dim=-1)
            min_dist = dist.min().item()
            min_distances[f"team_{team}_min_miss_m"] = min_dist

        overall_min = min(min_distances.values()) if min_distances else float("inf")
        return {
            "min_miss_distance_m": overall_min,
            "per_team": min_distances,
        }
