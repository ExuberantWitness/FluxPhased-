"""S1 radar array factor physics — 1D ULA, 5 cells, lambda/2 spacing, beam steering.

Geometry: jammer at (0,0,0), radar at (0, 8000, 0). Jammer lies at broadside (theta=0)
from radar's perspective. Radar steers its main lobe to theta_0 = beam_az[idx];
the array factor at the jammer's direction is AF(0; theta_0), which equals N^2 (peak)
when theta_0=0 and rolls off as theta_0 moves away from broadside.

Formula (uniform linear array, uniform weights, d=lambda/2, beam steer theta_0):
    AF(0; theta_0) = Sum_{n=0..N-1} exp(-j * pi * n * sin(theta_0))
    |AF|^2 = sin^2(N*pi*sin(theta_0)/2) / sin^2(pi*sin(theta_0)/2)
    Peak-normalized (theta_0=0):  |AF|^2 / N^2
    In dB:                        10 * log10(|AF|^2 / N^2)

At sin(theta_0) -> 0, |AF|^2 -> N^2 by L'Hopital, so the normalized form -> 1 (= 0 dB).

Frozen beam grid: 5 azimuths {-60, -30, 0, +30, +60} degrees.
Jammer is always at 0 degrees (broadside); only the radar main lobe scans.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

SPEED_OF_LIGHT = 299_792_458.0

BEAM_AZ_DEG_S1: tuple[float, ...] = (-60.0, -30.0, 0.0, 30.0, 60.0)
N_BEAM_DIRS_S1: int = len(BEAM_AZ_DEG_S1)


@dataclass(frozen=True)
class RadarULAConfig:
    n_cells: int = 5
    spacing_lambda: float = 0.5
    fc_hz: float = 10.0e9
    beam_az_deg: tuple[float, ...] = BEAM_AZ_DEG_S1

    @property
    def n_beam_dirs(self) -> int:
        return len(self.beam_az_deg)


def beam_az_sin_table(cfg: RadarULAConfig, device, dtype=torch.float32) -> torch.Tensor:
    """Returns [n_beam_dirs] tensor of sin(theta_0) for each beam direction."""
    angles_rad = torch.tensor(cfg.beam_az_deg, device=device, dtype=dtype) * (torch.pi / 180.0)
    return torch.sin(angles_rad)


def compute_radar_af_db(
    cfg: RadarULAConfig,
    *,
    radar_beam_az_idx: torch.Tensor,
) -> torch.Tensor:
    """AF^2 (dB, peak-normalized) at the jammer's broadside direction, given radar's beam steering.

    Args:
        cfg: RadarULAConfig
        radar_beam_az_idx: [E] int64 in 0..n_beam_dirs-1

    Returns:
        [E] float32, AF^2 peak-normalized in dB. Peak (idx for theta_0=0) = 0 dB;
        all other directions < 0 dB.
    """
    if radar_beam_az_idx.dim() != 1:
        raise ValueError(f"radar_beam_az_idx must be [E] 1-D, got shape {tuple(radar_beam_az_idx.shape)}")
    device = radar_beam_az_idx.device
    s0_table = beam_az_sin_table(cfg, device=device, dtype=torch.float32)  # [n_beam_dirs]
    s0 = s0_table.gather(0, radar_beam_az_idx.long())                       # [E]

    N = float(cfg.n_cells)
    num = torch.sin(N * torch.pi * s0 / 2.0) ** 2                            # [E]
    den = torch.sin(torch.pi * s0 / 2.0) ** 2                                # [E]

    # L'Hopital at s0 -> 0: limit of |AF|^2 = N^2
    af_sq = torch.where(
        den > 1e-10,
        num / den.clamp(min=1e-12),
        torch.full_like(s0, N * N),
    )
    af_norm_sq = (af_sq / (N * N)).clamp(min=1e-12)
    af_norm_db = 10.0 * torch.log10(af_norm_sq)
    return af_norm_db.to(torch.float32)


def compute_radar_af_db_all(cfg: RadarULAConfig, *, device) -> torch.Tensor:
    """AF^2 (dB, peak-normalized) for all beam directions. Useful for plotting / debugging.

    Returns: [n_beam_dirs] float32
    """
    idx_all = torch.arange(cfg.n_beam_dirs, device=device, dtype=torch.int64)
    return compute_radar_af_db(cfg, radar_beam_az_idx=idx_all)
