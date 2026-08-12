"""S3 array factor geometry — identical to S2 (5-cell 1D ULA).

S3 adds cell binding (per-cell on/off) but does NOT change the array
geometry. The array factor (AF) formulas are unchanged from S2; cell binding
only modulates the coherent combining gain via N_active (see physics.py).

This module re-exports S2's geometry symbols so S3 code can import them from
a local namespace, and defines N_CELLS (asserted to match JammerULAConfig).

For S4 (2D 5x5 UPA), this module will be replaced with a 2D version; the
physics.py compute_jnr_db_s3 accepts injectable AF callables so S4 only needs
to provide a 2D AF function without rewriting the JNR link budget.
"""
from __future__ import annotations

# Re-export S2 geometry (1D ULA, 5 cells, beam_az {-60,-30,0,30,60}).
# S3's jammer uses the same 5-cell 1D ULA as S2; only the per-cell on/off
# control is new (cell binding).
from env.gpu.array_face_s2.array_factor import (  # noqa: F401
    RadarULAConfig,
    JammerULAConfig,
    BEAM_AZ_DEG_S1,
    N_BEAM_DIRS_S1,
    BEAM_AZ_DEG_S2,
    N_BEAM_DIRS_S2,
    compute_radar_af_db,
    compute_radar_af_db_all,
    compute_jammer_af_db,
    compute_jammer_af_db_all,
)

# Number of independently controllable cells in the jammer ULA.
# Must match JammerULAConfig().n_cells so physics and action space agree.
N_CELLS: int = JammerULAConfig().n_cells
assert N_CELLS == 5, f"S3 expects 5-cell jammer ULA, got n_cells={N_CELLS}"
