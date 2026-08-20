"""S6 array factor — GENERALIZED UPA response toward an arbitrary target.

S2-S5 evaluate a special case: the gain toward BROADSIDE when steered to a
grid direction. S6 has TWO radars at distinct off-broadside bearings, so the
jammer's Tx gain toward radar i and radar i's Rx gain toward the jammer must
be evaluated at arbitrary (target − steering) direction cosines:

    AF(u, v; u_b, v_b) = AF_az(u − u_b) · AF_el(v − v_b)

with u = sin(az)·cos(el), v = sin(el) (same convention as S4) and the same
separable |AF|² peak-normalized dB form. By even symmetry this equals the
S4 function exactly when the target is broadside (u = v = 0) — the M0
consistency gate.
"""
from __future__ import annotations

import torch

from env.gpu.array_face_s4.array_factor import (
    UPAConfig,
    BEAM_AZ_DEG_S4,
    BEAM_EL_DEG_S4,
    N_AZ,
    N_EL,
    N_BEAM_DIRS_S4,
    N_CELLS_S4,
    beam_idx_to_az_el,
    _ula_af_sq_db,
    upa_af_table,
)

# S6 aliases (same grids)
BEAM_AZ_DEG_S6: tuple[float, ...] = BEAM_AZ_DEG_S4
BEAM_EL_DEG_S6: tuple[float, ...] = BEAM_EL_DEG_S4
N_BEAM_DIRS_S6: int = N_BEAM_DIRS_S4
N_CELLS_S6: int = N_CELLS_S4
N_RADARS: int = 2  # R: the S6 defender count


def az_el_to_uv(az_rad: torch.Tensor, el_rad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(az, el) radians -> direction cosines (u, v) = (sin az·cos el, sin el)."""
    return torch.sin(az_rad) * torch.cos(el_rad), torch.sin(el_rad)


def compute_upa_af_db_toward(
    cfg: UPAConfig,
    *,
    beam_az_idx: torch.Tensor,   # [E] int64 in 0..N_AZ-1 (steering)
    beam_el_idx: torch.Tensor,   # [E] int64 in 0..N_EL-1 (steering)
    target_az_rad: torch.Tensor, # [E] float (target bearing)
    target_el_rad: torch.Tensor, # [E] float
) -> torch.Tensor:
    """Peak-normalized |AF|² (dB) toward an arbitrary target given steering.

    Evaluates the separable UPA pattern at (u_t − u_b, v_t − v_b). When the
    target is broadside (u_t = v_t = 0) this reproduces compute_upa_af_db
    exactly (even-symmetric AF) — the S6↔S4 consistency gate.
    """
    device = beam_az_idx.device
    az_table = torch.tensor(
        [torch.deg2rad(torch.tensor(float(a))).item() for a in cfg.beam_az_deg],
        device=device, dtype=torch.float32,
    )
    el_table = torch.tensor(
        [torch.deg2rad(torch.tensor(float(e))).item() for e in cfg.beam_el_deg],
        device=device, dtype=torch.float32,
    )
    az_b = az_table.gather(0, beam_az_idx.long())
    el_b = el_table.gather(0, beam_el_idx.long())
    u_b, v_b = az_el_to_uv(az_b, el_b)
    u_t, v_t = az_el_to_uv(target_az_rad.float(), target_el_rad.float())
    af_az_db = _ula_af_sq_db(cfg.n_cells_az, u_t - u_b)
    af_el_db = _ula_af_sq_db(cfg.n_cells_el, v_t - v_b)
    return (af_az_db + af_el_db).to(torch.float32)


__all__ = [
    "UPAConfig", "BEAM_AZ_DEG_S6", "BEAM_EL_DEG_S6", "N_AZ", "N_EL",
    "N_BEAM_DIRS_S6", "N_CELLS_S6", "N_RADARS", "beam_idx_to_az_el",
    "az_el_to_uv", "compute_upa_af_db_toward", "upa_af_table",
]
