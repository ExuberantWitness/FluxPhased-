"""S6 action contract — jammer (S4-style) vs radar team (beam + service).

Jammer side (1 agent, S4 semantics):
  jammer_cell: [E, 25] float {0,1}   (all-zero = idle)
  jammer_beam: [E] int64 0..24

Radar side (R=2 agents, parameter-shared):
  radar_beam: [E, R] int64 0..24     (where to look)
  radar_svc:  [E, R] int64 0..1      (which service to attend)

Masks: jammer cell gated by energy (S4); all radar heads always legal.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from env.gpu.g3_bsta_lite.action_contract import ContractViolation
from env.gpu.array_face_s6.array_factor import N_CELLS_S6, N_BEAM_DIRS_S6, N_RADARS

N_JAM_CELLS: int = N_CELLS_S6
N_JAM_BEAMS: int = N_BEAM_DIRS_S6
N_RAD_BEAMS: int = N_BEAM_DIRS_S6
N_RAD_SVCS: int = 2
K_RADARS: int = N_RADARS


def validate_actions(
    jammer_cell: torch.Tensor,
    jammer_beam: torch.Tensor,
    radar_beam: torch.Tensor,
    radar_svc: torch.Tensor,
    *,
    E: int,
    device,
) -> None:
    if jammer_cell.shape != (E, N_JAM_CELLS) or jammer_cell.dtype not in (torch.float32, torch.float64):
        raise ContractViolation(
            f"jammer_cell must be [E={E}, {N_JAM_CELLS}] float, got {tuple(jammer_cell.shape)} {jammer_cell.dtype}")
    if jammer_beam.shape != (E,) or jammer_beam.dtype != torch.int64:
        raise ContractViolation(
            f"jammer_beam must be [E={E}] int64, got {tuple(jammer_beam.shape)} {jammer_beam.dtype}")
    if radar_beam.shape != (E, K_RADARS) or radar_beam.dtype != torch.int64:
        raise ContractViolation(
            f"radar_beam must be [E={E}, R={K_RADARS}] int64, got {tuple(radar_beam.shape)} {radar_beam.dtype}")
    if radar_svc.shape != (E, K_RADARS) or radar_svc.dtype != torch.int64:
        raise ContractViolation(
            f"radar_svc must be [E={E}, R={K_RADARS}] int64, got {tuple(radar_svc.shape)} {radar_svc.dtype}")
    if (jammer_beam < 0).any() or (jammer_beam >= N_JAM_BEAMS).any():
        raise ContractViolation("jammer_beam out of range")
    if (radar_beam < 0).any() or (radar_beam >= N_RAD_BEAMS).any():
        raise ContractViolation("radar_beam out of range")
    if (radar_svc < 0).any() or (radar_svc >= N_RAD_SVCS).any():
        raise ContractViolation("radar_svc out of range")
    for v in jammer_cell.unique().tolist():
        if v not in (0.0, 1.0):
            raise ContractViolation(f"jammer_cell must be binary, got {v}")
