"""S4 action contract: Bernoulli(25) cell binding + Categorical(25) 2D beam.

HANDOFF §11.2 removes S3's base head entirely:
  - idle/jam semantics are ABSORBED by cell binding: all-zero cells = idle
    (no Tx), >=1 cell on = jam.
  - service selection is removed: the jammer always jams the radar's current
    service (ESM reports it; the decision problem is 2D beam + cells).

Action is a 2-tuple:
  action_cell: [E, 25] float ∈ {0., 1.}  (per-cell on/off, 5×5 UPA)
  action_beam: [E] int64 ∈ 0..24         (flat 2D beam index, az-fastest raster)

mask is a 2-tuple (mask_cell[E,25], mask_beam[E,25]):
  - mask_cell is all-False when energy_tokens == 0 (no jamming possible),
    else all-True (per-cell cost is handled post-sample by the env's
    over-budget top-k clamp, mirroring S3).
  - mask_beam is always all-True (beam choice costs nothing; unused when idle).

Unlike S3 there is NO zero-cell clamp: all-zero cells is a legitimate idle
action in S4 (it is the only way to express idle without a base head).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from env.gpu.g3_bsta_lite.action_contract import ContractViolation
from env.gpu.array_face_s4.array_factor import N_CELLS_S4, N_BEAM_DIRS_S4


N_ACTIONS_CELL: int = N_CELLS_S4      # 25
N_ACTIONS_BEAM: int = N_BEAM_DIRS_S4  # 25


@dataclass
class UPATransitionTrace:
    """Per-step diagnostic record for S4 two-head actions.

    requested_cell / executed_cell may differ when the over-budget top-k
    clamp fires (Σ requested cells > remaining tokens).
    """
    observation_state_version: int
    mask_cell: torch.Tensor              # [E, N_ACTIONS_CELL] bool
    mask_beam: torch.Tensor              # [E, N_ACTIONS_BEAM] bool
    requested_cell: torch.Tensor         # [E, N_ACTIONS_CELL] float32
    requested_beam: torch.Tensor         # [E] int64
    executed_cell: torch.Tensor          # [E, N_ACTIONS_CELL] float32 (post-clamp)
    executed_beam: torch.Tensor          # [E] int64
    is_jam: torch.Tensor                 # [E] bool (executed_cell.sum(-1) > 0)
    n_active_cells: torch.Tensor         # [E] int64 (post-clamp, executed)
    energy_before: torch.Tensor          # [E] float32
    energy_after: torch.Tensor           # [E] float32
    tokens_consumed: torch.Tensor        # [E] int64 (Σ executed_cell for jam, 0 for idle)
    legal: torch.Tensor                  # [E] bool


def validate_actions(
    action_cell: torch.Tensor,
    action_beam: torch.Tensor,
    *,
    E: int,
    device,
) -> None:
    """Raises ContractViolation on shape/dtype/range/binary violations.

    Does NOT check energy (env handles via mask + over-budget clamp).
    """
    if action_cell.shape != (E, N_ACTIONS_CELL) or action_cell.dtype not in (torch.float32, torch.float64):
        raise ContractViolation(
            f"action_cell must be [E={E}, N_CELLS={N_ACTIONS_CELL}] float, "
            f"got shape={tuple(action_cell.shape)} dtype={action_cell.dtype}"
        )
    if action_beam.shape != (E,) or action_beam.dtype != torch.int64:
        raise ContractViolation(
            f"action_beam must be [E={E}] int64, got shape={tuple(action_beam.shape)} dtype={action_beam.dtype}"
        )
    if (action_beam < 0).any() or (action_beam >= N_ACTIONS_BEAM).any():
        raise ContractViolation(
            f"action_beam must be in 0..{N_ACTIONS_BEAM-1}, "
            f"got min={int(action_beam.min().item())} max={int(action_beam.max().item())}"
        )
    unique_vals = action_cell.unique()
    for v in unique_vals.tolist():
        if v not in (0.0, 1.0):
            raise ContractViolation(
                f"action_cell must be binary {{0., 1.}}, got value {v}"
            )
