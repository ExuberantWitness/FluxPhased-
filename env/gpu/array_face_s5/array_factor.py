"""S5 array factor — re-exports S4's 2D UPA geometry for both jammers.

S5 (HANDOFF §11.3) puts TWO identical jammers against the radar. Each jammer
is the same 5×5 UPA as S4 (same beam grid, same cell layout), so this module
re-exports S4's symbols unchanged; only the physics (power summation across
jammers) differs in S5.
"""
from __future__ import annotations

from env.gpu.array_face_s4.array_factor import (
    UPAConfig,
    BEAM_AZ_DEG_S4,
    BEAM_EL_DEG_S4,
    N_AZ,
    N_EL,
    N_BEAM_DIRS_S4,
    N_CELLS_S4,
    beam_idx_to_az_el,
    compute_upa_af_db,
    compute_upa_af_db_flat,
    upa_af_table,
)

# S5 aliases (same grids, new names for module coherence)
BEAM_AZ_DEG_S5: tuple[float, ...] = BEAM_AZ_DEG_S4
BEAM_EL_DEG_S5: tuple[float, ...] = BEAM_EL_DEG_S4
N_BEAM_DIRS_S5: int = N_BEAM_DIRS_S4
N_CELLS_S5: int = N_CELLS_S4
N_JAMMERS: int = 2  # K: the S5 cooperation dimension

__all__ = [
    "UPAConfig", "BEAM_AZ_DEG_S4", "BEAM_EL_DEG_S4",
    "N_AZ", "N_EL", "N_BEAM_DIRS_S4", "N_CELLS_S4",
    "beam_idx_to_az_el", "compute_upa_af_db", "compute_upa_af_db_flat", "upa_af_table",
    "BEAM_AZ_DEG_S5", "BEAM_EL_DEG_S5", "N_BEAM_DIRS_S5", "N_CELLS_S5", "N_JAMMERS",
]
