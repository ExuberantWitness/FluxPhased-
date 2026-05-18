"""Elo-band Prioritized Fictitious Self-Play (PFSP) sampler.

Wraps `OpponentPool.sample_pfsp` with an Elo rating system so that during
early training, the agent samples opponents near its own skill band rather
than blindly across the full population. The band width anneals over PSRO
iterations: wide early (exploration) -> narrow late (hard exploitation).

Design:
  - Each policy_id has a scalar Elo (default 1500).
  - Elo updates from payoff matrix entries with K-factor=24 after each
    PSRO iteration.
  - sample_elo_band(policy_id, n_samples, band) restricts the PFSP
    softmax to opponents with |Elo_opp - Elo_self| <= band; falls back
    to full PFSP if the band is empty.

This is wired into FluxLeague behind a config flag so we can ablate
{Nash, TC-DAMS} x {PFSP, Elo-band PFSP} cleanly.
"""

from __future__ import annotations

import os
import json
from typing import Dict, List

import numpy as np

from .opponent_pool import OpponentPool


DEFAULT_ELO = 1500.0
DEFAULT_K = 24.0


class EloBandSampler:
    """Elo-banded PFSP wrapper on top of OpponentPool."""

    def __init__(
        self,
        pool: OpponentPool,
        initial_elo: float = DEFAULT_ELO,
        k_factor: float = DEFAULT_K,
        band_init: float = 400.0,
        band_final: float = 100.0,
        anneal_iters: int = 15,
    ):
        self.pool = pool
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.band_init = band_init
        self.band_final = band_final
        self.anneal_iters = max(1, anneal_iters)
        self.elo: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Rating bookkeeping
    # ------------------------------------------------------------------
    def _get(self, policy_id: str) -> float:
        if policy_id not in self.elo:
            self.elo[policy_id] = self.initial_elo
        return self.elo[policy_id]

    def update_from_match(
        self,
        winner_id: str,
        loser_id: str,
        draw: bool = False,
    ) -> None:
        ra = self._get(winner_id)
        rb = self._get(loser_id)
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        sa = 0.5 if draw else 1.0
        sb = 0.5 if draw else 0.0
        self.elo[winner_id] = ra + self.k_factor * (sa - ea)
        self.elo[loser_id] = rb + self.k_factor * (sb - eb)

    def update_from_payoff_matrix(self, payoff: dict) -> None:
        """Refresh Elo from a {(p1, p2): win_rate} dict.

        Uses win_rate as a soft outcome score (PSRO payoffs are already
        averaged over many games, so this is a low-noise update). Each
        ordered pair contributes one update.
        """
        for (a, b), w in payoff.items():
            if a not in self.pool.policies or b not in self.pool.policies:
                continue
            ra = self._get(a)
            rb = self._get(b)
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            self.elo[a] = ra + self.k_factor * (w - ea)
            # Symmetric: b's expected = 1 - ea, actual = 1 - w
            self.elo[b] = rb + self.k_factor * ((1.0 - w) - (1.0 - ea))

    def current_band(self, iteration: int) -> float:
        """Linearly anneal band width from band_init to band_final."""
        if iteration >= self.anneal_iters:
            return self.band_final
        alpha = iteration / self.anneal_iters
        return (1.0 - alpha) * self.band_init + alpha * self.band_final

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(
        self,
        current_policy_id: str,
        iteration: int,
        n_samples: int = 1,
    ) -> List[str]:
        """Elo-banded PFSP sample.

        1. Restrict opponents to {opp : |Elo(opp) - Elo(self)| <= band(iter)}
        2. Within that band, apply PFSP softmax on loss-rate vs current agent.
        3. If band is empty, fall back to pool.sample_pfsp.
        """
        if current_policy_id not in self.pool.policies:
            return []
        record = self.pool.policies[current_policy_id]
        all_opponents = self.pool.get_opponent_team(record.team)
        if not all_opponents:
            return []

        self_elo = self._get(current_policy_id)
        band = self.current_band(iteration)
        in_band = [
            opp for opp in all_opponents
            if abs(self._get(opp) - self_elo) <= band
        ]
        if not in_band:
            return self.pool.sample_pfsp(current_policy_id, n_samples=n_samples)

        win_rates = np.array([
            record.win_rates.get(opp, 0.5) for opp in in_band
        ])
        loss_rates = 1.0 - win_rates
        if loss_rates.sum() < 1e-8:
            probs = np.ones(len(in_band)) / len(in_band)
        else:
            logits = loss_rates / max(self.pool.pfsp_temperature, 1e-6)
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()

        n = min(n_samples, len(in_band))
        idx = np.random.choice(len(in_band), size=n, replace=False, p=probs)
        return [in_band[i] for i in idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({k: float(v) for k, v in self.elo.items()}, f, indent=2)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.elo = {k: float(v) for k, v in data.items()}
