"""Action contract for the G3-BSTA-lite debug env (F1).

Frozen per docs/g3-bsta-lite/DEBUG_CONTRACT.md §3.

Action semantics:
  0 = idle (always legal)
  1 = jam_service_0
  2 = jam_service_1

One masked categorical distribution. At most one service jammed per step.
The action mask is a deterministic function of actor-visible observation
and own resource state; it never reveals whether a service is currently
active or valuable.

Illegal actions produce an explicit ContractViolation; they are never
silently substituted. ``requested_action == executed_action`` for legal
actions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


N_ACTIONS = 3
ACTION_IDLE = 0
ACTION_JAM_SERVICE_0 = 1
ACTION_JAM_SERVICE_1 = 2
SERVICE_FOR_ACTION = {
    ACTION_IDLE: None,
    ACTION_JAM_SERVICE_0: 0,
    ACTION_JAM_SERVICE_1: 1,
}


class ContractViolation(ValueError):
    """Raised when the env observes an illegal action that cannot be honored.

    Per DEBUG_CONTRACT.md §3, illegal actions are an explicit contract
    violation and never silently substituted. The env step does not advance
    when this is raised.
    """


@dataclass
class TransitionTrace:
    """Per-step diagnostic record required by DEBUG_CONTRACT.md §3.

    Every transition logs mask, requested/executed action, selected service,
    and energy before/after. The trace is the canonical evidence used by
    test_runtime_contract, test_resource_contract and test_transition_order.
    """

    observation_state_version: int
    mask: torch.Tensor             # [E, N_ACTIONS] bool
    requested_action: torch.Tensor # [E] int64
    executed_action: torch.Tensor  # [E] int64
    selected_service: torch.Tensor # [E] int64, -1 for idle
    energy_before: torch.Tensor    # [E] float32
    energy_after: torch.Tensor     # [E] float32
    legal: torch.Tensor            # [E] bool
