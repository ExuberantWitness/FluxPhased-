"""Frozen rule radar opponent for G3-BSTA-lite debug env (F1 §2).

A single radar opponent scans the two services in a fixed round-robin
pattern. On each step the radar probes exactly one service: the detection
succeeds with the service-specific probability degraded by the jammer's
service-specific JNR. The radar's choice is independent of the jammer
action (the radar cannot observe the jammer's emission pattern); the
choice only depends on the step index modulo 2.

Per DEBUG_CONTRACT.md §6 "next rule-radar action" is hidden from the
jammer; the actor sees only delayed detection outcomes and a delayed
urgency proxy, never the next radar slot.
"""

from __future__ import annotations

import math

import torch


class FrozenRuleRadar:
    """Deterministic 2-service round-robin radar opponent."""

    def __init__(self, *, n_envs: int, n_services: int = 2, device: str = "cpu"):
        if n_services != 2:
            raise ValueError(f"debug profile pins n_services=2; got {n_services}")
        self.n_envs = n_envs
        self.n_services = n_services
        self.device = device

    def service_at_step(self, step_idx: int) -> int:
        """Round-robin: even step -> service 0, odd step -> service 1."""
        return step_idx % 2

    def service_at_step_batch(self, step_idx: int) -> torch.Tensor:
        """[E] int64 service index per env at this step."""
        svc = self.service_at_step(step_idx)
        return torch.full((self.n_envs,), svc, dtype=torch.int64, device=self.device)

    def detector_rng_draw(
        self,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """[E] uniform draw in [0,1) using the dedicated detector RNG."""
        return torch.rand(self.n_envs, generator=generator, device=self.device)
