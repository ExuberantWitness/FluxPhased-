"""S2 action contract: MultiDiscrete([3, 5]) = base + beam.

Extends lite's Categorical(3) contract by adding a beam_az head:
  action_base: [E] int64 ∈ {0, 1, 2}  (idle, jam_svc_0, jam_svc_1)
  action_beam: [E] int64 ∈ {0, 1, 2, 3, 4}  (jammer ULA beam_az index)

Combined action is a tuple (base, beam). When base = 0 (idle), beam is unused
(no Tx); the jammer still picks one but it has no physical effect.

Both heads are Categorical; mask is a tuple (mask_base[E,3], mask_beam[E,5]).
mask_beam is always all-True (no constraint on beam selection).
mask_base follows the same energy-budget rule as lite/S1.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from env.gpu.g3_bsta_lite.action_contract import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    ACTION_JAM_SERVICE_1,
    SERVICE_FOR_ACTION,
    ContractViolation,
)


N_ACTIONS_BASE: int = 3
N_ACTIONS_BEAM: int = 5
N_ACTIONS_TOTAL: int = N_ACTIONS_BASE + N_ACTIONS_BEAM  # for actor flat param count, unused in step


@dataclass
class MultiDiscreteTransitionTrace:
    """Per-step diagnostic record for S2 MultiDiscrete actions.

    Mirrors lite's TransitionTrace but with separate base/beam action tensors.
    """
    observation_state_version: int
    mask_base: torch.Tensor              # [E, N_ACTIONS_BASE=3] bool
    mask_beam: torch.Tensor              # [E, N_ACTIONS_BEAM=5] bool
    requested_base: torch.Tensor         # [E] int64
    requested_beam: torch.Tensor         # [E] int64
    executed_base: torch.Tensor          # [E] int64
    executed_beam: torch.Tensor          # [E] int64
    selected_service: torch.Tensor       # [E] int64, -1 for idle
    energy_before: torch.Tensor          # [E] float32
    energy_after: torch.Tensor           # [E] float32
    legal: torch.Tensor                  # [E] bool


def validate_actions(
    action_base: torch.Tensor,
    action_beam: torch.Tensor,
    *,
    E: int,
    device,
) -> None:
    """Raises ContractViolation on shape/dtype/range violations. Does NOT check energy."""
    if action_base.shape != (E,) or action_base.dtype != torch.int64:
        raise ContractViolation(
            f"action_base must be [E={E}] int64, got shape={tuple(action_base.shape)} dtype={action_base.dtype}"
        )
    if action_beam.shape != (E,) or action_beam.dtype != torch.int64:
        raise ContractViolation(
            f"action_beam must be [E={E}] int64, got shape={tuple(action_beam.shape)} dtype={action_beam.dtype}"
        )
    if (action_base < 0).any() or (action_base >= N_ACTIONS_BASE).any():
        raise ContractViolation(
            f"action_base must be in 0..{N_ACTIONS_BASE-1}, got min={int(action_base.min().item())} max={int(action_base.max().item())}"
        )
    if (action_beam < 0).any() or (action_beam >= N_ACTIONS_BEAM).any():
        raise ContractViolation(
            f"action_beam must be in 0..{N_ACTIONS_BEAM-1}, got min={int(action_beam.min().item())} max={int(action_beam.max().item())}"
        )
