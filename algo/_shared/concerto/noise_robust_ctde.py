"""Noise-robust CTDE: trace(P)-weighted team-advantage blend.

Per Concerto-RRM plan §3.3 and EAAI_RESEARCH_PLAN.md §3.3:

  α_eff = α_max · exp(-β · trace_P_norm)        # high uncertainty → low α
  adv   = (1 - α_eff) · A_agent  +  α_eff · A_team

Mechanism (from CTDE-reversal diagnostic): under heavy EW, the centralized
critic learns a mis-calibrated credit assignment because its target comms
observations are corrupted. The fix is to *de-weight* the team advantage in
proportion to sensing uncertainty (proxied by trace(P_norm)) — i.e. trust the
centralized critic only when fusion is tight.

The same trace(P_norm) signal also drives the V2 composer's θ2 trigger — one
signal, two consumers. Mechanism-consistent design.

API:
    blend_advantages(A_agent, A_team, trace_P_norm, alpha_max, beta) → adv

`trace_P_norm` is shape [E, T] (per-env per-team), broadcast across the time
and agent axes of the advantage tensors. Out-of-band (trace_P_norm=None) →
fall back to pure team advantage (bit-exact MAPPO).
"""

from __future__ import annotations

import torch
from typing import Optional


def blend_advantages(
    A_agent: torch.Tensor,
    A_team: torch.Tensor,
    trace_P_norm: Optional[torch.Tensor] = None,
    alpha_max: float = 0.7,
    beta: float = 2.0,
) -> torch.Tensor:
    """Blend per-agent and team advantages via trace(P)-driven α_eff.

    Args:
        A_agent: [..., D_agent] per-agent GAE advantage.
        A_team:  [..., D_team] team GAE advantage (broadcastable to A_agent).
        trace_P_norm: [E, T] normalized trace(P) in [0, 1] for each team, or None.
            When None, returns A_team unchanged (bit-exact MAPPO fallback).
        alpha_max: max α when trace_P_norm → 0 (clean sensing → trust team).
        beta: exponential decay rate; α_eff halves at trace_P_norm ≈ ln(2)/β.
    Returns:
        adv: blended advantage with same shape as A_agent (after broadcast).
    """
    if trace_P_norm is None:
        # Bit-exact fallback to MAPPO
        return A_team.expand_as(A_agent) if A_team.shape != A_agent.shape else A_team

    # α_eff per env-team, in [0, alpha_max]
    alpha_eff = alpha_max * torch.exp(-beta * trace_P_norm.clamp(min=0.0, max=1.0))

    # Broadcast α_eff to match A_agent's shape.
    # A_agent is typically [T, B, D] where B = E * n_agents_per_team * time_unroll.
    # The caller is responsible for reshaping trace_P_norm to match — we just
    # broadcast any leading dims.
    while alpha_eff.dim() < A_agent.dim():
        alpha_eff = alpha_eff.unsqueeze(-1)
    alpha_eff = alpha_eff.expand_as(A_agent)

    # If A_team has fewer dims than A_agent, broadcast.
    A_team_b = A_team.expand_as(A_agent) if A_team.shape != A_agent.shape else A_team

    return (1.0 - alpha_eff) * A_agent + alpha_eff * A_team_b


def alpha_eff_scalar(
    trace_P_norm: float,
    alpha_max: float = 0.7,
    beta: float = 2.0,
) -> float:
    """Convenience: compute α_eff for a scalar trace_P_norm.

    Useful for instrumentation / logging.
    """
    return float(alpha_max * __import__("math").exp(-beta * max(0.0, min(1.0, trace_P_norm))))
