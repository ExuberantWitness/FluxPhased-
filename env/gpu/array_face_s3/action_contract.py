"""S3 action contract: MultiDiscrete([3, 5]) + Bernoulli(5) cell binding.

Extends S2's MultiDiscrete([3, 5]) = base + beam with a third head:
  action_base: [E] int64 ∈ {0, 1, 2}   (idle, jam_svc_0, jam_svc_1)
  action_beam: [E] int64 ∈ {0, 1, 2, 3, 4}  (jammer ULA beam_az index)
  action_cell: [E, N_CELLS] float ∈ {0., 1.}  (per-cell on/off binding)

Cell binding semantics (per confirmed design decisions):
  - Energy cost: each jamming step consumes Σ(active cells) tokens.
    Idle (base=0) consumes 0 tokens regardless of cell mask.
  - Zero-cell clamp: if base=jam but cell_mask is all-zero, env forces >=1
    cell on (the highest-logit cell) before applying physics. This prevents
    the "free idle via all-zero cells" degenerate local optimum and is handled
    in env.step, not in validation (validate only checks shape/dtype/binary).

mask is a 3-tuple (mask_base[E,3], mask_beam[E,5], mask_cell[E,N_CELLS]).
mask_cell is always all-True (no per-cell constraint at sample time; the
zero-cell clamp is applied post-sample in env.step).
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
from env.gpu.array_face_s3.array_factor import N_CELLS


N_ACTIONS_BASE: int = 3
N_ACTIONS_BEAM: int = 5
# N_CELLS imported from array_factor (asserted == JammerULAConfig().n_cells)


@dataclass
class BernoulliTransitionTrace:
    """Per-step diagnostic record for S3 three-head actions.

    Mirrors S2's MultiDiscreteTransitionTrace + cell-binding fields.
    requested_cell / executed_cell may differ when the zero-cell clamp fires
    (base=jam + all-zero requested -> executed has >=1 cell forced on).
    """
    observation_state_version: int
    mask_base: torch.Tensor              # [E, N_ACTIONS_BASE] bool
    mask_beam: torch.Tensor              # [E, N_ACTIONS_BEAM] bool
    mask_cell: torch.Tensor              # [E, N_CELLS] bool
    requested_base: torch.Tensor         # [E] int64
    requested_beam: torch.Tensor         # [E] int64
    requested_cell: torch.Tensor         # [E, N_CELLS] float32
    executed_base: torch.Tensor          # [E] int64
    executed_beam: torch.Tensor          # [E] int64
    executed_cell: torch.Tensor          # [E, N_CELLS] float32 (post-clamp)
    selected_service: torch.Tensor       # [E] int64, -1 for idle
    n_active_cells: torch.Tensor         # [E] int64 (post-clamp, executed)
    energy_before: torch.Tensor          # [E] float32
    energy_after: torch.Tensor           # [E] float32
    tokens_consumed: torch.Tensor        # [E] int64 (Σ executed_cell for jam, 0 for idle)
    legal: torch.Tensor                  # [E] bool


def validate_actions(
    action_base: torch.Tensor,
    action_beam: torch.Tensor,
    action_cell: torch.Tensor,
    *,
    E: int,
    device,
) -> None:
    """Raises ContractViolation on shape/dtype/range/binary violations.

    Does NOT check energy (env handles via mask) and does NOT enforce >=1
    cell when base=jam (env applies the zero-cell clamp post-sample).
    """
    if action_base.shape != (E,) or action_base.dtype != torch.int64:
        raise ContractViolation(
            f"action_base must be [E={E}] int64, got shape={tuple(action_base.shape)} dtype={action_base.dtype}"
        )
    if action_beam.shape != (E,) or action_beam.dtype != torch.int64:
        raise ContractViolation(
            f"action_beam must be [E={E}] int64, got shape={tuple(action_beam.shape)} dtype={action_beam.dtype}"
        )
    if action_cell.shape != (E, N_CELLS) or action_cell.dtype not in (torch.float32, torch.float64):
        raise ContractViolation(
            f"action_cell must be [E={E}, N_CELLS={N_CELLS}] float, got shape={tuple(action_cell.shape)} dtype={action_cell.dtype}"
        )
    if (action_base < 0).any() or (action_base >= N_ACTIONS_BASE).any():
        raise ContractViolation(
            f"action_base must be in 0..{N_ACTIONS_BASE-1}, got min={int(action_base.min().item())} max={int(action_base.max().item())}"
        )
    if (action_beam < 0).any() or (action_beam >= N_ACTIONS_BEAM).any():
        raise ContractViolation(
            f"action_beam must be in 0..{N_ACTIONS_BEAM-1}, got min={int(action_beam.min().item())} max={int(action_beam.max().item())}"
        )
    # cell must be binary {0., 1.}
    unique_vals = action_cell.unique()
    for v in unique_vals.tolist():
        if v not in (0.0, 1.0):
            raise ContractViolation(
                f"action_cell must be binary {{0., 1.}}, got value {v}"
            )
