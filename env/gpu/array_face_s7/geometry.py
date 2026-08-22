"""S7 geometry — spatial separation returns: 2 radars at ±20°, 2 jammers at ±60°.

Roadmap note (S6 geometry.py): S6b co-located the radars at +20° because the
v1 ±20° layout was structurally defense-dominant against a SINGLE jammer
(one radar always evaded). S7 gives the offense two jammers, so the radars
spread out again — and the interesting question becomes whether the jammer
team can cross-assign: jammer k suppresses radar k while its partner takes
the other. The relative bearings that fall out:

    jammer +60°: radar +20° → −40°, radar −20° → −80°
    jammer −60°: radar +20° → +80°, radar −20° → +40°

All four pair bearings are OFF the beam grid (−60/−30/0/+30/+60): no pair is
covered for free, so single-beam suppression of both radars is impossible and
the two jammers must coordinate. (Symmetric +40° jammer placement was
rejected: it would put one pair bearing exactly on the −60° grid point.)
"""
from __future__ import annotations

import math
import torch

from env.gpu.array_face_s6.array_factor import az_el_to_uv

RADAR_RANGE_M: float = 8000.0
RADAR_AZ_DEG: tuple[float, float] = (+20.0, -20.0)  # spatial separation returns
RADAR_EL_DEG: tuple[float, float] = (0.0, 0.0)
JAMMER_AZ_DEG: tuple[float, float] = (+60.0, -60.0)  # symmetric cross-fire sites
JAMMER_EL_DEG: tuple[float, float] = (0.0, 0.0)
N_RADARS: int = 2
N_JAMMERS: int = 2


def _dirs(az_deg, el_deg, device) -> tuple[torch.Tensor, torch.Tensor]:
    az = torch.tensor([math.radians(a) for a in az_deg], device=device)
    el = torch.tensor([math.radians(e) for e in el_deg], device=device)
    return az, el


def radar_directions(device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-radar (az, el) radians tensors [R] (global frame, site-centered)."""
    return _dirs(RADAR_AZ_DEG, RADAR_EL_DEG, device)


def jammer_directions(device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-jammer (az, el) radians tensors [K] (global frame, site-centered)."""
    return _dirs(JAMMER_AZ_DEG, JAMMER_EL_DEG, device)


def pair_bearings(device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(jammer k, radar r) relative bearing [K, R] in radians.

    The angular frame is centered on the radar site. The bearing of radar r
    as seen from jammer k is radar_az[r] − jammer_az[k]. By |AF| even
    symmetry the same pair bearing serves BOTH link ends: jammer k's Tx gain
    toward radar r and radar r's Rx gain toward jammer k (S6 physics.py
    makes the identical argument for the single-jammer case).
    """
    az_k, el_k = jammer_directions(device)
    az_r, el_r = radar_directions(device)
    rel_az = az_r.unsqueeze(0) - az_k.unsqueeze(1)   # [K, R]
    rel_el = el_r.unsqueeze(0) - el_k.unsqueeze(1)   # [K, R]
    return rel_az, rel_el


def pair_uv(device) -> torch.Tensor:
    """[K, R, 2] direction cosines of each radar as seen from each jammer."""
    rel_az, rel_el = pair_bearings(device)
    u, v = az_el_to_uv(rel_az, rel_el)
    return torch.stack([u, v], dim=-1)
