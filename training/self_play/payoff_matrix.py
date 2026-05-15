"""Payoff matrix computation for PSRO.

Evaluates win rates between all pairs of policies by running games
in the vectorized MFARVecEnv. Supports batch evaluation using
parallel environments.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

from .opponent_pool import OpponentPool


class PayoffMatrix:
    """Compute and store empirical payoff matrix between policy populations."""

    def __init__(
        self,
        opponent_pool: OpponentPool,
        n_eval_games: int = 50,
        device: str = "cuda",
    ):
        self.pool = opponent_pool
        self.n_eval_games = n_eval_games
        self.device = device
        # matrix[i][j] = win rate of policy i against policy j
        self.matrix: Dict[Tuple[str, str], float] = {}

    def evaluate_pair(
        self,
        red_policy_id: str,
        blue_policy_id: str,
        env,  # MFARVecEnv
        red_trainer,  # TeamPPOTrainer
        blue_trainer,  # TeamPPOTrainer
    ) -> float:
        """Evaluate win rate of red vs blue over multiple games.

        Uses deterministic policy inference for both sides.
        """
        red_wins = 0
        total = 0
        remaining = self.n_eval_games
        E = env.num_envs

        while remaining > 0:
            batch = min(E, remaining)
            env.reset()

            for step in range(10000):
                with torch.no_grad():
                    # Red team (team 0): deterministic policy
                    red = red_trainer.get_own_actions(env, team=0, deterministic=True)
                    # Blue team (team 1): deterministic policy
                    blue = blue_trainer.get_own_actions(env, team=1, deterministic=True)

                    actions = torch.zeros(batch, env.n_radars, env.action_dim, device=self.device)
                    for i, r in enumerate(range(red["r_start"], red["r_end"])):
                        actions[:, r, :] = red["radar_actions"][i]
                    for i, r in enumerate(range(blue["r_start"], blue["r_end"])):
                        actions[:, r, :] = blue["radar_actions"][i]

                    commander_actions = torch.zeros(
                        batch, env.n_teams, env.battlefield.commander_action_dim,
                        device=self.device,
                    )
                    commander_actions[:, 0, :] = red["commander_action"]
                    commander_actions[:, 1, :] = blue["commander_action"]

                result = env.step(actions=actions, commander_actions=commander_actions)

                if result["dones"].any():
                    for e in range(batch):
                        if result["dones"][e]:
                            if result["winners"][e] == 0:
                                red_wins += 1
                            total += 1
                    break

            remaining -= batch

        win_rate = red_wins / max(total, 1)
        self.matrix[(red_policy_id, blue_policy_id)] = win_rate
        self.matrix[(blue_policy_id, red_policy_id)] = 1.0 - win_rate

        # Update pool win rates
        self.pool.update_win_rate(red_policy_id, blue_policy_id, win_rate >= 0.5)
        self.pool.update_win_rate(blue_policy_id, red_policy_id, win_rate < 0.5)

        return win_rate

    def evaluate_all(self, env, trainers: dict):
        """Evaluate all cross-team pairs.

        Args:
            env: MFARVecEnv
            trainers: {policy_id: TeamPPOTrainer}
        """
        red_policies = [
            pid for pid, rec in self.pool.policies.items()
            if rec.team == 0 and rec.is_active
        ]
        blue_policies = [
            pid for pid, rec in self.pool.policies.items()
            if rec.team == 1 and rec.is_active
        ]

        for r_id in red_policies:
            for b_id in blue_policies:
                key = (r_id, b_id)
                if key not in self.matrix:
                    r_trainer = trainers.get(r_id)
                    b_trainer = trainers.get(b_id)
                    if r_trainer and b_trainer:
                        self.evaluate_pair(r_id, b_id, env, r_trainer, b_trainer)

    def get_submatrix(self, team: int) -> np.ndarray:
        """Get payoff submatrix for one team's perspective.

        Args:
            team: 0 (red) or 1 (blue)
        Returns:
            [K, K_opponent] numpy array where K = policies for this team
        """
        own_policies = [
            pid for pid, rec in self.pool.policies.items()
            if rec.team == team and rec.is_active
        ]
        opp_policies = [
            pid for pid, rec in self.pool.policies.items()
            if rec.team != team and rec.is_active
        ]

        n_own = len(own_policies)
        n_opp = len(opp_policies)
        payoff = np.full((n_own, n_opp), 0.5)

        for i, own_id in enumerate(own_policies):
            for j, opp_id in enumerate(opp_policies):
                payoff[i, j] = self.matrix.get((own_id, opp_id), 0.5)

        return payoff, own_policies, opp_policies

    def to_array(self) -> np.ndarray:
        """Export full payoff matrix as numpy array."""
        active = [
            pid for pid, rec in self.pool.policies.items() if rec.is_active
        ]
        n = len(active)
        mat = np.full((n, n), 0.5)
        for i, p1 in enumerate(active):
            for j, p2 in enumerate(active):
                if p1 != p2:
                    mat[i, j] = self.matrix.get((p1, p2), 0.5)
        return mat, active
