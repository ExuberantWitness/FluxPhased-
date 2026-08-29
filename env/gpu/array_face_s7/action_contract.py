"""S7 action contract — jammer TEAM (K=2) vs radar team (R=2).

Jammer side (K=2 agents, parameter-shared, S4 semantics each):
  jammer_cell: [E, K, 25] float {0,1}   (all-zero slice = that jammer idle)
  jammer_beam: [E, K] int64 0..24

Radar side (R=2 agents, parameter-shared):
  radar_beam: [E, R] int64 0..24     (where to look)
  radar_svc:  [E, R] int64 0..1      (which service to attend)

Masks: each jammer's cells gated by ITS OWN energy budget; all radar heads
always legal.
"""
from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite.action_contract import ContractViolation
from env.gpu.array_face_s7.array_factor import (
    N_CELLS_S7, N_BEAM_DIRS_S7, N_JAMMERS, N_RADARS,
)

N_JAM_CELLS: int = N_CELLS_S7
N_JAM_BEAMS: int = N_BEAM_DIRS_S7
N_RAD_BEAMS: int = N_BEAM_DIRS_S7
N_RAD_SVCS: int = 2
K_JAMMERS: int = N_JAMMERS
K_RADARS: int = N_RADARS


def validate_actions(
    jammer_cell: torch.Tensor,
    jammer_beam: torch.Tensor,
    radar_beam: torch.Tensor,
    radar_svc: torch.Tensor,
    *,
    E: int,
    device,
    K: int | None = None,
) -> None:
    K = K or K_JAMMERS  # attacker-count scaling: env passes its own K
    if jammer_cell.shape != (E, K, N_JAM_CELLS) or \
            jammer_cell.dtype not in (torch.float32, torch.float64):
        raise ContractViolation(
            f"jammer_cell must be [E={E}, K={K}, {N_JAM_CELLS}] float, "
            f"got {tuple(jammer_cell.shape)} {jammer_cell.dtype}")
    if jammer_beam.shape != (E, K) or jammer_beam.dtype != torch.int64:
        raise ContractViolation(
            f"jammer_beam must be [E={E}, K={K}] int64, got "
            f"{tuple(jammer_beam.shape)} {jammer_beam.dtype}")
    if radar_beam.shape != (E, K_RADARS) or radar_beam.dtype != torch.int64:
        raise ContractViolation(
            f"radar_beam must be [E={E}, R={K_RADARS}] int64, got "
            f"{tuple(radar_beam.shape)} {radar_beam.dtype}")
    if radar_svc.shape != (E, K_RADARS) or radar_svc.dtype != torch.int64:
        raise ContractViolation(
            f"radar_svc must be [E={E}, R={K_RADARS}] int64, got "
            f"{tuple(radar_svc.shape)} {radar_svc.dtype}")
    if (jammer_beam < 0).any() or (jammer_beam >= N_JAM_BEAMS).any():
        raise ContractViolation("jammer_beam out of range")
    if (radar_beam < 0).any() or (radar_beam >= N_RAD_BEAMS).any():
        raise ContractViolation("radar_beam out of range")
    if (radar_svc < 0).any() or (radar_svc >= N_RAD_SVCS).any():
        raise ContractViolation("radar_svc out of range")
    for v in jammer_cell.unique().tolist():
        if v not in (0.0, 1.0):
            raise ContractViolation(f"jammer_cell must be binary, got {v}")
