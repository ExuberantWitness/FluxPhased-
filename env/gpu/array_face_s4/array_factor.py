"""S4 2D array factor — 5×5 UPA with separable azimuth/elevation AF.

Both radar and jammer upgrade from 1D ULA (S1-S3) to a planar 5×5 UPA
(HANDOFF §11.2). Beam grid:
  az ∈ {-60, -30, 0, 30, 60}°  (same as S1-S3)
  el ∈ {-30, -15, 0, 15, 30}°  (new elevation dimension)
  => 25 beam directions, indexed as beam_idx = el_idx * 5 + az_idx
     (az fastest — consistent with S1-S3 radar az sweep `step % 5`).

Geometry (unchanged): jammer at (0,0,0), radar at (0, 8000, 0). Each sees
the other at az=0, el=0 (broadside). The UPA lies in the x-z plane facing
+y. For a direction with azimuth az and elevation el the direction cosines
are u = sin(az)·cos(el), v = sin(el).

Separable UPA AF (uniform weights, d = λ/2):
  AF(u, v) = Σ_m Σ_n exp(j·π·(m·u + n·v))
           = AF_az(u) · AF_el(v)
  |AF_az(u)|² = sin²(N_az·π·u/2) / sin²(π·u/2)
  (same for el with v), peak-normalized by N_az² · N_el² so the 2D peak = 0 dB.

By even symmetry, the gain TOWARD the radar (at broadside) when the beam is
steered to (az_b, el_b) equals the pattern value at (az_b, el_b) — the same
convention as S2's 1D AF, so when el_b = 0 this reduces exactly to S2's
compute_jammer_af_db (continuity check in tests).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


BEAM_AZ_DEG_S4: tuple[float, ...] = (-60.0, -30.0, 0.0, 30.0, 60.0)
BEAM_EL_DEG_S4: tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)
N_AZ: int = len(BEAM_AZ_DEG_S4)
N_EL: int = len(BEAM_EL_DEG_S4)
N_BEAM_DIRS_S4: int = N_AZ * N_EL  # 25
N_CELLS_S4: int = N_AZ * N_EL      # 25 (5×5 array)


@dataclass(frozen=True)
class UPAConfig:
    """5×5 planar array, λ/2 spacing in both axes (HANDOFF §11.2)."""
    n_cells_az: int = 5
    n_cells_el: int = 5
    spacing_lambda: float = 0.5
    fc_hz: float = 10.0e9
    beam_az_deg: tuple[float, ...] = BEAM_AZ_DEG_S4
    beam_el_deg: tuple[float, ...] = BEAM_EL_DEG_S4

    @property
    def n_cells(self) -> int:
        return self.n_cells_az * self.n_cells_el

    @property
    def n_beam_dirs(self) -> int:
        return len(self.beam_az_deg) * len(self.beam_el_deg)


def beam_idx_to_az_el(beam_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split flat beam index into (az_idx, el_idx). az fastest raster."""
    beam_idx = beam_idx.long()
    return beam_idx % N_AZ, beam_idx // N_AZ


def _ula_af_sq_db(n_cells: int, s: torch.Tensor) -> torch.Tensor:
    """|AF|² peak-normalized in dB for a 1D ULA at spatial frequency s.

    Same numerical pattern as S2's compute_jammer_af_db (L'Hôpital branch at
    s≈0 returns N²). s here is the direction cosine (sin-based), not an angle.
    """
    N = float(n_cells)
    num = torch.sin(N * torch.pi * s / 2.0) ** 2
    den = torch.sin(torch.pi * s / 2.0) ** 2
    af_sq = torch.where(
        den > 1e-10,
        num / den.clamp(min=1e-12),
        torch.full_like(s, N * N),
    )
    af_norm_sq = (af_sq / (N * N)).clamp(min=1e-12)
    return 10.0 * torch.log10(af_norm_sq)


def compute_upa_af_db(
    cfg: UPAConfig,
    *,
    beam_az_idx: torch.Tensor,
    beam_el_idx: torch.Tensor,
) -> torch.Tensor:
    """2D UPA AF² (dB, peak-normalized) evaluated at the steered direction.

    Args:
        cfg: UPAConfig
        beam_az_idx: [E] int64 in 0..N_AZ-1
        beam_el_idx: [E] int64 in 0..N_EL-1

    Returns:
        [E] float32 dB. Peak (az=0, el=0) = 0 dB. Separable:
        AF_2d_db = AF_az(u) + AF_el(v), u = sin(az)·cos(el), v = sin(el).
    """
    if beam_az_idx.dim() != 1 or beam_el_idx.dim() != 1:
        raise ValueError("beam_az_idx / beam_el_idx must be [E] 1-D")
    if beam_az_idx.shape != beam_el_idx.shape:
        raise ValueError(
            f"beam_az_idx {tuple(beam_az_idx.shape)} and beam_el_idx "
            f"{tuple(beam_el_idx.shape)} must have the same shape"
        )
    device = beam_az_idx.device
    az_table = torch.tensor(
        [torch.deg2rad(torch.tensor(float(a))).item() for a in cfg.beam_az_deg],
        device=device, dtype=torch.float32,
    )
    el_table = torch.tensor(
        [torch.deg2rad(torch.tensor(float(e))).item() for e in cfg.beam_el_deg],
        device=device, dtype=torch.float32,
    )
    az_rad = az_table.gather(0, beam_az_idx.long())
    el_rad = el_table.gather(0, beam_el_idx.long())
    u = torch.sin(az_rad) * torch.cos(el_rad)
    v = torch.sin(el_rad)
    af_az_db = _ula_af_sq_db(cfg.n_cells_az, u)
    af_el_db = _ula_af_sq_db(cfg.n_cells_el, v)
    return (af_az_db + af_el_db).to(torch.float32)


def compute_upa_af_db_flat(
    cfg: UPAConfig,
    *,
    beam_idx: torch.Tensor,
) -> torch.Tensor:
    """Flat-index version: beam_idx in 0..24 (az fastest raster)."""
    az_idx, el_idx = beam_idx_to_az_el(beam_idx)
    return compute_upa_af_db(cfg, beam_az_idx=az_idx, beam_el_idx=el_idx)


def upa_af_table(cfg: UPAConfig, *, device) -> torch.Tensor:
    """Precomputed [N_AZ, N_EL] AF dB lookup table (az-major: [az_idx, el_idx])."""
    az_idx = torch.arange(N_AZ, device=device).unsqueeze(1).expand(N_AZ, N_EL).reshape(-1)
    el_idx = torch.arange(N_EL, device=device).unsqueeze(0).expand(N_AZ, N_EL).reshape(-1)
    flat = compute_upa_af_db(cfg, beam_az_idx=az_idx, beam_el_idx=el_idx)
    return flat.reshape(N_AZ, N_EL)
