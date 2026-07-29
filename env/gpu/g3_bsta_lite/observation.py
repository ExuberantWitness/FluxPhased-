"""Causal observation builder for G3-BSTA-lite debug env (F1 §6).

Per DEBUG_CONTRACT.md §6, the actor may only observe:

  - remaining_energy / initial_energy           (1 scalar)
  - remaining_time / horizon                    (1 scalar)
  - delayed/noisy service activity              (2 scalars, one per service)
  - delayed urgency proxy                       (2 scalars, one per service)
  - intercept confidence and age               (2 scalars: conf, age)
  - previous executed action one-hot            (3 scalars)

Total: 11 scalars. The observation does NOT contain:

  - exact pending queue length
  - exact progress / deadline
  - true target slot/id
  - future arrivals
  - post-action detector outcome
  - next radar action
  - environment RNG state

The mask is derived only from actor-visible state and own resource state.
``privileged`` (for critic only) adds true pending count and true track
health so the central critic can verify the headroom gap; this is logged
as a separately registered experiment variable.
"""

from __future__ import annotations

import torch


OBS_DIM = 11
PRIVILEGED_DIM = 4  # pending_per_service[2] + track_health_per_service[2]


def build_observation(
    *,
    energy: torch.Tensor,           # [E] float32
    initial_energy: torch.Tensor,   # [E] float32
    step_idx: int,
    horizon: int,
    delayed_detect: torch.Tensor,   # [E, n_services] float32, recent detect rate
    delayed_urgency: torch.Tensor,  # [E, n_services] float32, recent track health
    intercept_confidence: torch.Tensor,  # [E] float32
    intercept_age: torch.Tensor,         # [E] float32 (steps since last intercept)
    prev_action_onehot: torch.Tensor,    # [E, 3] float32
) -> torch.Tensor:
    """[E, OBS_DIM] causal observation."""
    E = energy.shape[0]
    rem_E = (energy / initial_energy.clamp(min=1e-6)).clamp(0.0, 1.0)
    rem_t = torch.full((E,), float(horizon - step_idx) / float(max(horizon, 1)),
                       device=energy.device, dtype=energy.dtype)
    obs = torch.cat(
        [
            rem_E.unsqueeze(-1),
            rem_t.unsqueeze(-1),
            delayed_detect,
            delayed_urgency,
            intercept_confidence.unsqueeze(-1),
            intercept_age.unsqueeze(-1),
            prev_action_onehot,
        ],
        dim=-1,
    )
    return obs


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
