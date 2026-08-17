"""S5 action contract: per-jammer (Bernoulli(25) cell + Categorical(25) beam) × K=2.

Same head structure as S4 (HANDOFF §11.2 semantics), stacked on a jammer axis:
  action_cell: [E, K, 25] float ∈ {0., 1.}   (per-cell on/off per jammer)
  action_beam: [E, K] int64 ∈ 0..24          (flat 2D beam index per jammer)

Idle semantics per jammer: all-zero cells = that jammer idles this step.
mask is a 2-tuple (mask_cell[E,K,25], mask_beam[E,K,25]); mask_cell[k] is
all-False when jammer k's energy_tokens == 0, else all-True (over-budget
top-k clamp handled post-sample per jammer, mirroring S3/S4).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from env.gpu.g3_bsta_lite.action_contract import ContractViolation
from env.gpu.array_face_s5.array_factor import (
    N_CELLS_S5, N_BEAM_DIRS_S5, N_JAMMERS,
)


N_ACTIONS_CELL: int = N_CELLS_S5      # 25
N_ACTIONS_BEAM: int = N_BEAM_DIRS_S5  # 25
K_JAMMERS: int = N_JAMMERS            # 2


@dataclass
class S5TransitionTrace:
    """Per-step diagnostic record for S5 two-jammer actions."""
    observation_state_version: int
    mask_cell: torch.Tensor              # [E, K, N_ACTIONS_CELL] bool
    mask_beam: torch.Tensor              # [E, K, N_ACTIONS_BEAM] bool
    requested_cell: torch.Tensor         # [E, K, N_ACTIONS_CELL] float32
    requested_beam: torch.Tensor         # [E, K] int64
    executed_cell: torch.Tensor          # [E, K, N_ACTIONS_CELL] float32 (post-clamp)
    executed_beam: torch.Tensor          # [E, K] int64
    is_jam: torch.Tensor                 # [E, K] bool (per-jammer activity)
    n_active_cells: torch.Tensor         # [E, K] int64 (post-clamp, executed)
    energy_before: torch.Tensor          # [E, K] float32
    energy_after: torch.Tensor           # [E, K] float32
    tokens_consumed: torch.Tensor        # [E, K] int64
    legal: torch.Tensor                  # [E] bool


def validate_actions(
    action_cell: torch.Tensor,
    action_beam: torch.Tensor,
    *,
    E: int,
    device,
) -> None:
    """Raises ContractViolation on shape/dtype/range/binary violations.

    Does NOT check energy (env handles via mask + per-jammer over-budget clamp).
    """
    if action_cell.shape != (E, K_JAMMERS, N_ACTIONS_CELL) or action_cell.dtype not in (torch.float32, torch.float64):
        raise ContractViolation(
            f"action_cell must be [E={E}, K={K_JAMMERS}, N_CELLS={N_ACTIONS_CELL}] float, "
            f"got shape={tuple(action_cell.shape)} dtype={action_cell.dtype}"
        )
    if action_beam.shape != (E, K_JAMMERS) or action_beam.dtype != torch.int64:
        raise ContractViolation(
            f"action_beam must be [E={E}, K={K_JAMMERS}] int64, "
            f"got shape={tuple(action_beam.shape)} dtype={action_beam.dtype}"
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
