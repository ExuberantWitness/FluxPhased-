"""Opponent pool with PFSP (Prioritized Fictitious Self-Play) sampling.

Manages a collection of policy checkpoints with win-rate tracking.
Supports three sampling modes:
  - PFSP: prioritize opponents the current agent loses to
  - Uniform: equal probability
  - Uniform_old: uniform over past checkpoints only
"""

import os
import torch
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PolicyRecord:
    """Metadata for one policy in the pool."""
    policy_id: str
    team: int              # 0=red, 1=blue
    role: str              # "main", "main_exploiter", "league_exploiter"
    checkpoint_path: str
    generation: int        # PSRO iteration when created
    win_rates: dict = field(default_factory=dict)  # {opponent_id: win_rate}
    parent_id: Optional[str] = None
    is_active: bool = True


class OpponentPool:
    """Pool of past policy checkpoints for self-play training."""

    def __init__(
        self,
        pool_dir: str = "checkpoints/pool",
        population_cap: int = 20,
        pfsp_temperature: float = 1.0,
    ):
        self.pool_dir = pool_dir
        self.population_cap = population_cap
        self.pfsp_temperature = pfsp_temperature
        self.policies: dict = {}  # {policy_id: PolicyRecord}
        self._next_id = 0

        os.makedirs(pool_dir, exist_ok=True)

    def add_policy(
        self,
        team: int,
        role: str,
        checkpoint_path: str,
        generation: int,
        parent_id: str = None,
    ) -> str:
        """Add a policy checkpoint to the pool.

        Returns:
            policy_id string
        """
        policy_id = f"p{self._next_id:04d}"
        self._next_id += 1

        record = PolicyRecord(
            policy_id=policy_id,
            team=team,
            role=role,
            checkpoint_path=checkpoint_path,
            generation=generation,
            parent_id=parent_id,
        )
        self.policies[policy_id] = record

        # Evict oldest inactive if over cap
        if len(self.policies) > self.population_cap:
            self._evict_oldest()

        return policy_id

    def get_opponent_team(self, team: int) -> List[str]:
        """Get all policy IDs for the opposing team."""
        return [
            pid for pid, rec in self.policies.items()
            if rec.team != team and rec.is_active
        ]

    def get_team_policies(self, team: int, role: str = None) -> List[str]:
        """Get policy IDs for a team, optionally filtered by role."""
        return [
            pid for pid, rec in self.policies.items()
            if rec.team == team and rec.is_active
            and (role is None or rec.role == role)
        ]

    def sample_pfsp(
        self,
        current_policy_id: str,
        n_samples: int = 1,
    ) -> List[str]:
        """PFSP sampling: prioritize opponents the current agent loses to.

        Args:
            current_policy_id: the policy that will train against the sampled opponent
            n_samples: number of opponents to sample
        Returns:
            list of opponent policy_ids
        """
        record = self.policies[current_policy_id]
        opponents = self.get_opponent_team(record.team)

        if not opponents:
            return []

        # Get win rates vs each opponent
        win_rates = np.array([
            record.win_rates.get(opp, 0.5) for opp in opponents
        ])

        # PFSP: probability ∝ (1 - win_rate)
        loss_rates = 1.0 - win_rates
        if loss_rates.sum() < 1e-8:
            # All win rates unknown or 1.0: uniform
            probs = np.ones(len(opponents)) / len(opponents)
        else:
            # Softmax with temperature
            logits = loss_rates / self.pfsp_temperature
            logits = logits - logits.max()  # numerical stability
            probs = np.exp(logits)
            probs = probs / probs.sum()

        indices = np.random.choice(
            len(opponents), size=min(n_samples, len(opponents)),
            replace=False, p=probs,
        )
        return [opponents[i] for i in indices]

    def sample_uniform(
        self,
        current_policy_id: str,
        n_samples: int = 1,
    ) -> List[str]:
        """Uniform random sampling over opposing team."""
        record = self.policies[current_policy_id]
        opponents = self.get_opponent_team(record.team)
        if not opponents:
            return []
        indices = np.random.choice(
            len(opponents), size=min(n_samples, len(opponents)),
            replace=False,
        )
        return [opponents[i] for i in indices]

    def update_win_rate(self, policy_id: str, opponent_id: str, win: bool):
        """Update win rate between two policies."""
        record = self.policies[policy_id]
        key = opponent_id
        if key not in record.win_rates:
            record.win_rates[key] = 0.5
        # Exponential moving average
        alpha = 0.1
        record.win_rates[key] = (1 - alpha) * record.win_rates[key] + alpha * float(win)

    def get_active_main(self, team: int) -> Optional[str]:
        """Get the current main agent for a team."""
        mains = [
            pid for pid, rec in self.policies.items()
            if rec.team == team and rec.role == "main" and rec.is_active
        ]
        if not mains:
            return None
        # Return latest generation
        return max(mains, key=lambda pid: self.policies[pid].generation)

    def load_policy(self, policy_id: str) -> dict:
        """Load checkpoint for a policy."""
        path = self.policies[policy_id].checkpoint_path
        return torch.load(path, map_location="cpu", weights_only=False)

    def _evict_oldest(self):
        """Evict the oldest non-main policy to stay under cap."""
        candidates = [
            (pid, rec) for pid, rec in self.policies.items()
            if rec.role != "main"
        ]
        if not candidates:
            return
        oldest = min(candidates, key=lambda x: x[1].generation)
        self.policies[oldest[0]].is_active = False

    def save_metadata(self):
        """Save pool metadata to disk."""
        import json
        meta = {}
        for pid, rec in self.policies.items():
            meta[pid] = {
                "team": rec.team,
                "role": rec.role,
                "checkpoint_path": rec.checkpoint_path,
                "generation": rec.generation,
                "parent_id": rec.parent_id,
                "is_active": rec.is_active,
                "win_rates": {k: float(v) for k, v in rec.win_rates.items()},
            }
        path = os.path.join(self.pool_dir, "pool_metadata.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_metadata(self):
        """Load pool metadata from disk."""
        import json
        path = os.path.join(self.pool_dir, "pool_metadata.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            meta = json.load(f)
        for pid, data in meta.items():
            rec = PolicyRecord(
                policy_id=pid,
                team=data["team"],
                role=data["role"],
                checkpoint_path=data["checkpoint_path"],
                generation=data["generation"],
                parent_id=data.get("parent_id"),
                is_active=data.get("is_active", True),
                win_rates=data.get("win_rates", {}),
            )
            self.policies[pid] = rec
            self._next_id = max(self._next_id, int(pid[1:]) + 1)
