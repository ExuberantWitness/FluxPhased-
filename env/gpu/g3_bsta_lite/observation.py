"""Causal observation builder for G3-BSTA-lite debug env (F1 §6, R1B profiles).

Two profiles are exposed (PREREGISTRATION.md §2):

  - ``mdp_sanity_v1``: a genuine fully-observed MDP. Observation explicitly
    contains the exact per-service pending count and the current radar
    service one-hot. Purpose: prove PPO can learn **at all**. NOT a POMDP,
    NOT a paper claim about active perception.

  - ``pomdp_v1``: a genuine POMDP. Actor / critic / witness share the
    same information set, which excludes the true pending count, the
    radar's hidden phase, and the future arrivals. Activity proxy is
    a **non-invertible saturating function** of pending count; the
    radar service is hidden; the observation channels are delayed
    (``obs_delay_steps >= 1``).

Both profiles share OBS_DIM = 11 so downstream plumbing (actor/critic
input size, action_mask code) is uniform; the SEMANTICS of the channels
differ between profiles.

Channel layout for both profiles (slots 0..10):

  mdp_sanity_v1:
    [0]    rem_E
    [1]    rem_t
    [2..3] pending_per_service (exact, float)
    [4..5] radar_service_onehot (current radar slot, exact)
    [6]    intercept_confidence
    [7]    intercept_age
    [8..10] prev_action_onehot

  pomdp_v1:
    [0]    rem_E
    [1]    rem_t
    [2..3] delayed_detect (per service, EMA of past detections, delay>=1)
    [4..5] delayed_urgency_proxy (per service, NON-INVERTIBLE saturating
           function of pending count, delay>=1)
    [6]    intercept_confidence
    [7]    intercept_age
    [8..10] prev_action_onehot

The mask is derived only from actor-visible state and own resource state.
``privileged`` (for critic only) adds true pending count and true track
health so the central critic can verify the headroom gap; this is logged
as a separately registered experiment variable and is NEVER an actor
input.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


OBS_DIM = 11
PRIVILEGED_DIM = 4  # pending_per_service[2] + track_health_per_service[2]

PROFILE_MDP_SANITY = "mdp_sanity_v1"
PROFILE_POMDP = "pomdp_v1"
PROFILES = (PROFILE_MDP_SANITY, PROFILE_POMDP)

# Saturating constant for the POMDP urgency proxy. With K=3.0 the proxy
# takes values 0.000, 0.283, 0.487, 0.632, 0.736, 0.811, 0.865 for
# n_pending = 0..6 — strictly monotone but bounded in [0, 1). Even with
# perfect knowledge of K and the proxy value, n_pending can only be
# recovered up to a wide interval once n >= 4 (proxy deltas < 0.07).
POMDP_URGENCY_SAT_K = 3.0


def pomdp_urgency_proxy(
    n_pending: torch.Tensor,
    *,
    K: float = POMDP_URGENCY_SAT_K,
) -> torch.Tensor:
    """Non-invertible saturating activity proxy.

    f(n) = 1 - exp(-n / K).

    Strictly monotone in n, bounded in [0, 1). The inverse image of a
    proxy value is an interval of width ~K at large n; the exact
    pending count cannot be recovered from the proxy. This is the
    fix for the leak flagged in POST_AUDIT_CORRECTION.md §4.1.
    """
    return 1.0 - torch.exp(-n_pending.float() / float(K))


def build_observation(
    *,
    energy: torch.Tensor,           # [E] float32
    initial_energy: torch.Tensor,   # [E] float32
    step_idx: int,
    horizon: int,
    delayed_detect: Optional[torch.Tensor] = None,   # [E, n_services] pomdp
    delayed_urgency: Optional[torch.Tensor] = None,  # [E, n_services] pomdp
    intercept_confidence: torch.Tensor,              # [E] float32
    intercept_age: torch.Tensor,                     # [E] float32
    prev_action_onehot: torch.Tensor,                # [E, 3] float32
    profile: str = PROFILE_POMDP,
    pending_per_service: Optional[torch.Tensor] = None,    # [E, n_services] mdp
    radar_service_onehot: Optional[torch.Tensor] = None,   # [E, n_services] mdp
) -> torch.Tensor:
    """[E, OBS_DIM] observation for the requested profile."""
    E = energy.shape[0]
    rem_E = (energy / initial_energy.clamp(min=1e-6)).clamp(0.0, 1.0)
    rem_t = torch.full((E,), float(horizon - step_idx) / float(max(horizon, 1)),
                       device=energy.device, dtype=energy.dtype)

    if profile == PROFILE_POMDP:
        if delayed_detect is None or delayed_urgency is None:
            raise ValueError(
                "pomdp_v1 profile requires delayed_detect and delayed_urgency"
            )
        return torch.cat(
            [
                rem_E.unsqueeze(-1),
                rem_t.unsqueeze(-1),
                delayed_detect.to(energy.dtype),
                delayed_urgency.to(energy.dtype),
                intercept_confidence.unsqueeze(-1),
                intercept_age.unsqueeze(-1),
                prev_action_onehot,
            ],
            dim=-1,
        )

    if profile == PROFILE_MDP_SANITY:
        if pending_per_service is None or radar_service_onehot is None:
            raise ValueError(
                "mdp_sanity_v1 profile requires pending_per_service and "
                "radar_service_onehot"
            )
        return torch.cat(
            [
                rem_E.unsqueeze(-1),
                rem_t.unsqueeze(-1),
                pending_per_service.to(energy.dtype),
                radar_service_onehot.to(energy.dtype),
                intercept_confidence.unsqueeze(-1),
                intercept_age.unsqueeze(-1),
                prev_action_onehot,
            ],
            dim=-1,
        )

    raise ValueError(f"unknown profile: {profile!r}")


def build_privileged(
    *,
    pending_per_service: torch.Tensor,    # [E, n_services] int64
    track_health_per_service: torch.Tensor,  # [E, n_services] float32
) -> torch.Tensor:
    """[E, PRIVILEGED_DIM] privileged critic facts (NOT actor input)."""
    return torch.cat(
        [
            pending_per_service.float(),
            track_health_per_service.float(),
        ],
        dim=-1,
    )
