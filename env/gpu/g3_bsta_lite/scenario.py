"""Scenario table for G3-BSTA-lite debug env (F1 §5 RNG + §8 metric).

A Scenario is a pre-generated, policy-independent exogenous table:

  arrivals: bool[H, n_services] — arrivals per (step, service)
  baseline_snr_db_per_service: float[n_services] — RF budget per service

The arrivals table is FIXED for the episode before any jammer action.
Different seeds produce different tables; the same seed reproduces.

Eligibility rule: a service is "eligible" in a scenario if at least one
arrival appears in the table. Scenarios with zero eligible arrivals are
excluded from the manifest and marked NA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Scenario:
    seed: int
    arrivals: torch.Tensor           # [H, n_services] bool
    baseline_snr_db: torch.Tensor    # [n_services] float32

    @property
    def horizon(self) -> int:
        return int(self.arrivals.shape[0])

    @property
    def n_services(self) -> int:
        return int(self.arrivals.shape[1])

    def eligible(self) -> bool:
        return bool(self.arrivals.any().item())

    def to(self, device: str) -> "Scenario":
        return Scenario(
            seed=self.seed,
            arrivals=self.arrivals.to(device),
            baseline_snr_db=self.baseline_snr_db.to(device),
        )


def generate_scenario(
    *,
    seed: int,
    horizon: int,
    n_services: int,
    arrival_rate_per_service: float,
    baseline_snr_db: float,
    device: str = "cpu",
) -> Scenario:
    """Bernoulli arrivals per service per step.

    The arrivals table is the canonical policy-independent exogenous input
    to mission_drop_ratio. Its RNG is the dedicated environment-event RNG,
    isolated from detector and action RNG.
    """
    g = torch.Generator(device=device).manual_seed(int(seed))
    p = torch.tensor(arrival_rate_per_service, device=device)
    arrivals = torch.rand(horizon, n_services, generator=g, device=device) < p
    baseline = torch.full((n_services,), float(baseline_snr_db), device=device)
    return Scenario(seed=int(seed), arrivals=arrivals, baseline_snr_db=baseline)


def generate_paired_manifest(
    *,
    base_seed: int,
    n_scenarios: int,
    horizon: int,
    n_services: int,
    arrival_rate_per_service: float,
    baseline_snr_db: float,
    device: str = "cpu",
) -> list[Scenario]:
    """Generate a list of scenarios, excluding non-eligible ones.

    Returns exactly n_scenarios eligible scenarios. The search continues
    past non-eligible seeds until enough eligible scenarios are found, so
    the manifest size is deterministic.
    """
    out: list[Scenario] = []
    seed = int(base_seed)
    while len(out) < n_scenarios:
        s = generate_scenario(
            seed=seed,
            horizon=horizon,
            n_services=n_services,
            arrival_rate_per_service=arrival_rate_per_service,
            baseline_snr_db=baseline_snr_db,
            device=device,
        )
        if s.eligible():
            out.append(s)
        seed += 1
    return out
