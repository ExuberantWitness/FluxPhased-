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
        red_trainer,  # TeamPPOTrainer or policy loader
        blue_trainer,
    ) -> float:
        """Evaluate win rate of red vs blue over multiple games.

        Args:
            red_policy_id, blue_policy_id: policy IDs from opponent pool
            env: MFARVecEnv instance (E must be > 0)
            red_trainer, blue_trainer: TeamPPOTrainer instances with loaded policies
        Returns:
            win rate of red (0.0 - 1.0)
        """
        red_wins = 0
        total = 0
        remaining = self.n_eval_games
        E = env.num_envs

        while remaining > 0:
            batch = min(E, remaining)
            env.reset()

            for step in range(env.max_steps if hasattr(env, 'max_steps') else 10000):
                # Get observations
                obs = env._assemble_state(
                    env._buf_spectrum.zero_(),
                    env._buf_comm_data.zero_(),
                ) if step == 0 else None

                # Use deterministic policies for evaluation
                with torch.no_grad():
                    # Red team actions
                    red_radar_obs = obs[:, :2] if obs is not None else None  # radars 0,1
                    blue_radar_obs = obs[:, 2:] if obs is not None else None  # radars 2,3

                    # Simplified: use zero actions for now (placeholder for actual policy inference)
                    actions = torch.rand(batch, env.n_radars, env.action_dim, device=self.device)
                    commander_actions = torch.zeros(
                        batch, env.n_teams, env.battlefield.commander_action_dim,
                        device=self.device,
                    )

                result = env.step(actions=actions, commander_actions=commander_actions)

                if result["dones"].any():
                    for e in range(batch):
                        if result["dones"][e]:
                            if result["winners"][e] == 0:  # Red wins
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
