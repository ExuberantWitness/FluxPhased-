"""S2 array factor physics — extends S1 with jammer 1D ULA + Tx AF.

Compared to S1 (`env/gpu/array_face_s1/array_factor.py`):
  - Adds JammerULAConfig (5-cell, λ/2, beam_az grid {-60,-30,0,30,60})
  - Adds compute_jammer_af_db(jammer_beam_az_idx): Tx AF^2 at radar's broadside
  - Radar ULA + compute_radar_af_db unchanged from S1

Geometry (re-emphasized):
  jammer at (0,0,0), radar at (0, 8000, 0).
  From jammer's perspective, radar is at az=0 (broadside).
  From radar's perspective, jammer is at az=0 (broadside).
  So:
    - jammer steers its Tx main lobe to jammer_beam_az[idx] → AF_tx peaks when idx=0
    - radar steers its Rx main lobe to radar_beam_az[idx] → AF_rx peaks when idx=0
  Both AFs are 0 dB at idx=0 and roll off symmetrically.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from env.gpu.array_face_s1.array_factor import (
    RadarULAConfig, BEAM_AZ_DEG_S1, N_BEAM_DIRS_S1,
    compute_radar_af_db as _s1_compute_radar_af_db,
    beam_az_sin_table,
)

# Re-export radar-side from S1 unchanged
BEAM_AZ_DEG_S2: tuple[float, ...] = BEAM_AZ_DEG_S1
N_BEAM_DIRS_S2: int = N_BEAM_DIRS_S1


@dataclass(frozen=True)
class JammerULAConfig:
    n_cells: int = 5
    spacing_lambda: float = 0.5
    fc_hz: float = 10.0e9
    beam_az_deg: tuple[float, ...] = BEAM_AZ_DEG_S2

    @property
    def n_beam_dirs(self) -> int:
        return len(self.beam_az_deg)


def compute_jammer_af_db(
    cfg: JammerULAConfig,
    *,
    jammer_beam_az_idx: torch.Tensor,
) -> torch.Tensor:
    """Tx AF^2 (dB, peak-normalized) at the radar's broadside direction.

    Identical formula to radar AF (ULA, λ/2 spacing, uniform weights), just
    interpreted as the jammer's transmission gain toward the radar.

    Args:
        cfg: JammerULAConfig
        jammer_beam_az_idx: [E] int64 in 0..n_beam_dirs-1

    Returns:
        [E] float32, AF^2 peak-normalized in dB. Peak (idx for theta_0=0) = 0 dB;
        all other directions < 0 dB.
    """
    if jammer_beam_az_idx.dim() != 1:
        raise ValueError(f"jammer_beam_az_idx must be [E] 1-D, got shape {tuple(jammer_beam_az_idx.shape)}")
    device = jammer_beam_az_idx.device
    s0_table = beam_az_sin_table(cfg, device=device, dtype=torch.float32)
    s0 = s0_table.gather(0, jammer_beam_az_idx.long())

    N = float(cfg.n_cells)
    num = torch.sin(N * torch.pi * s0 / 2.0) ** 2
    den = torch.sin(torch.pi * s0 / 2.0) ** 2

    af_sq = torch.where(
        den > 1e-10,
        num / den.clamp(min=1e-12),
        torch.full_like(s0, N * N),
    )
    af_norm_sq = (af_sq / (N * N)).clamp(min=1e-12)
    af_norm_db = 10.0 * torch.log10(af_norm_sq)
    return af_norm_db.to(torch.float32)


def compute_jammer_af_db_all(cfg: JammerULAConfig, *, device) -> torch.Tensor:
    idx_all = torch.arange(cfg.n_beam_dirs, device=device, dtype=torch.int64)
    return compute_jammer_af_db(cfg, jammer_beam_az_idx=idx_all)


def compute_radar_af_db(
    cfg: RadarULAConfig,
    *,
    radar_beam_az_idx: torch.Tensor,
) -> torch.Tensor:
    """Wrapper around S1's radar AF for S2 module coherence."""
    return _s1_compute_radar_af_db(cfg, radar_beam_az_idx=radar_beam_az_idx)


def compute_radar_af_db_all(cfg: RadarULAConfig, *, device) -> torch.Tensor:
    idx_all = torch.arange(cfg.n_beam_dirs, device=device, dtype=torch.int64)
    return _s1_compute_radar_af_db(cfg, radar_beam_az_idx=idx_all)
