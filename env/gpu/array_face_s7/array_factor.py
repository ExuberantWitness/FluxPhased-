"""S7 array factor — reuses S6's generalized UPA response unchanged.

S7 keeps the same 5×5 UPA grids and beam raster (25 beams, 25 cells). What
changes is the LINK TOPOLOGY (2 jammers × 2 radars, per-pair bearings), which
lives in geometry.py / physics.py — not in the AF evaluation itself.
"""
from __future__ import annotations

from env.gpu.array_face_s6.array_factor import (
    UPAConfig,
    BEAM_AZ_DEG_S6,
    BEAM_EL_DEG_S6,
    N_AZ,
    N_EL,
    N_BEAM_DIRS_S6,
    N_CELLS_S6,
    beam_idx_to_az_el,
    az_el_to_uv,
    compute_upa_af_db_toward,
    upa_af_table,
)

# S7 aliases (same grids; counts now describe a 2v2 game)
BEAM_AZ_DEG_S7: tuple[float, ...] = BEAM_AZ_DEG_S6
BEAM_EL_DEG_S7: tuple[float, ...] = BEAM_EL_DEG_S6
N_BEAM_DIRS_S7: int = N_BEAM_DIRS_S6
N_CELLS_S7: int = N_CELLS_S6
N_JAMMERS: int = 2  # K: two cooperating jammers (S5's cooperation dimension)
N_RADARS: int = 2   # R: two radars, spatially separated again (roadmap S7)


__all__ = [
    "UPAConfig", "BEAM_AZ_DEG_S7", "BEAM_EL_DEG_S7", "N_AZ", "N_EL",
    "N_BEAM_DIRS_S7", "N_CELLS_S7", "N_JAMMERS", "N_RADARS",
    "beam_idx_to_az_el", "az_el_to_uv", "compute_upa_af_db_toward", "upa_af_table",
]
