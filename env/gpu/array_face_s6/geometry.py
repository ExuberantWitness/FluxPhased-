"""S6 geometry — a co-located two-head radar at az = +20° (S6b rebalance).

v1 placed radars at ±20°; oracle diagnostics showed that configuration is
structurally defense-dominant (one radar always evades the single jammer,
and the 27 dB detection cushion makes evasion free). The S6b rebalance:

  - CO-LOCATED heads: both radar beams sit at az = +20° (one site, two
    beam/service heads) — a single jammer beam suppresses both, so "the
    clean second radar" rescue is gone. Spatial separation returns in S7
    (2 jammers vs 2 radars).
  - Rebalanced link budget (EnvConfig): baseline_snr_db 22→12, P_jam_W
    2.0→0.1. Best-response sweep places every mission azimuth in a
    contestable band (p ∈ [0.12, 0.99], contested middle 0.57-0.67) — the
    scan-vs-stare hinge now binds (see REPORT: contestability sweep).

The off-broadside +20° bearing keeps the jammer's aiming non-trivial (its
beam grid has no +20° point; nearest grid aim is ±10° off).
"""
from __future__ import annotations

import math
import torch

from env.gpu.array_face_s6.array_factor import az_el_to_uv

RADAR_RANGE_M: float = 8000.0
RADAR_AZ_DEG: tuple[float, float] = (+20.0, +20.0)  # co-located two-head site
RADAR_EL_DEG: tuple[float, float] = (0.0, 0.0)
N_RADARS: int = 2


def radar_directions(device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-radar (az, el) radians tensors [R] for AF target evaluation."""
    az = torch.tensor([math.radians(a) for a in RADAR_AZ_DEG], device=device)
    el = torch.tensor([math.radians(e) for e in RADAR_EL_DEG], device=device)
    return az, el


def radar_uv(device) -> torch.Tensor:
    """[R, 2] direction cosines (u, v) of each radar as seen from the jammer."""
    az, el = radar_directions(device)
    u, v = az_el_to_uv(az, el)
    return torch.stack([u, v], dim=-1)
