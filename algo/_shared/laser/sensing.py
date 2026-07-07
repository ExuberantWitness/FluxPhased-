"""Laser precise-kill sensing: anisotropic noise + multi-radar fusion + Kalman tracking.

Migrated from `training/train_laser.py` lines 377-599 to make laser sensing logic
reusable by TeamPPOTrainer / FluxLeague without duplicating the 270 lines of KF math.

Three modes (per `sensing_noise.mode` in config):
  - "single":  anisotropic noise from one radar only. Cross-range wall at
               σ_cross = R × crossrange_factor (~700m at 3km range).
  - "fused":   multi-static range triangulation across own radars (information
               filter). Both axes become ~range-precise where the angular
               baseline is good.
  - "tracked": "fused" + Kalman filter over time. Moving radars (20 m/s)
               diversify geometry; the KF collapses bad-collinear (GDOP-limited)
               directions that floor pure spatial fusion at ~10m.

The fused enemy_xy is written IN-PLACE into obs[..., 68:72] (two enemies at
obs[68:70] and obs[70:72], each in normalized [-1, 1]^2 = ±half_map metres).

Home-on-jam / ESM beacon term: when jamming is enabled, the target's own
emission feeds an isotropic info term ∝ exposure_gain · emission². The
policy must learn an intermediate (timed) jam — too low gives no denial,
too high makes it a beacon.
"""

from __future__ import annotations

import torch
from typing import List, Tuple, Optional


__all__ = [
    "KalmanTracker",
    "fused_sensing",
    "add_sensing_noise",
    "enforce_radar_baseline",
]


# ---------------------------------------------------------------------------
# Single-radar anisotropic noise (S0 baseline)
# ---------------------------------------------------------------------------

def add_sensing_noise(
    obs: torch.Tensor,
    range_sigma_m: float,
    crossrange_factor: float,
    half_x: float,
    half_y: float,
) -> torch.Tensor:
    """Replace exact enemy positions in obs[68:72] with anisotropic-noisy estimate
    measured from the team's own radar-0.

    Args:
        obs: [E, T, 76] commander observation. obs[68:70] and obs[70:72] are
             the two enemy radars' positions in normalized [-1, 1] coords.
        range_sigma_m: range noise standard deviation (bandwidth-limited, cm).
        crossrange_factor: cross-range σ = R × factor (diffraction-limited).
        half_x, half_y: map half-extents in metres (for normalized ↔ physical).
    Returns:
        obs with obs[68:72] replaced by noisy estimate (in-place modified).
    """
    if range_sigma_m <= 0.0 and crossrange_factor <= 0.0:
        return obs
    ox = obs[..., 0] * half_x
    oy = obs[..., 1] * half_y
    for k in range(2):
        off = 68 + 2 * k
        ex = obs[..., off] * half_x
        ey = obs[..., off + 1] * half_y
        dx, dy = ex - ox, ey - oy
        R = torch.sqrt(dx * dx + dy * dy).clamp(min=1.0)
        rx, ry = dx / R, dy / R          # along-range unit
        cx, cy = -ry, rx                 # cross-range unit
        nr = torch.randn_like(R) * range_sigma_m
        nc = torch.randn_like(R) * (R * crossrange_factor)
        obs[..., off] = (ex + nr * rx + nc * cx) / half_x
        obs[..., off + 1] = (ey + nr * ry + nc * cy) / half_y
    return obs


# ---------------------------------------------------------------------------
# Multi-radar information-filter fusion (S2)
# ---------------------------------------------------------------------------

def _fuse_one(
    ex: torch.Tensor,
    ey: torch.Tensor,
    own: List[Tuple[torch.Tensor, torch.Tensor]],
    sr: float,
    cf: float,
    jam_mul: torch.Tensor = None,
    emission: torch.Tensor = None,
    exposure_gain: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One information-filter-fused measurement of (ex, ey) from the own radars.

    Each radar contributes an anisotropic measurement (range-precise, cross-range
    poor from its angle). The information filter intersects the error ellipses.

    Args:
        ex, ey: true enemy position (metres) [E, T]
        own: list of (ox, oy) own radar positions in metres
        sr: range σ (already jam-scaled if applicable)
        cf: crossrange factor (already jam-scaled)
        jam_mul: [E, T] noise multiplier from enemy jamming. If None, treated as 1.
        emission: [E, T] target's own jam level (for home-on-jam beacon).
        exposure_gain: home-on-jam info scaling.
    Returns:
        zx, zy: fused position estimate (metres)
        R00, R01, R11: measurement covariance (2x2, 3 unique entries)
    """
    if jam_mul is not None:
        sr = sr * jam_mul
        cf = cf * jam_mul
    L00 = torch.zeros_like(ex); L01 = torch.zeros_like(ex); L11 = torch.zeros_like(ex)
    e0 = torch.zeros_like(ex);  e1 = torch.zeros_like(ex)
    for (ox, oy) in own:
        dx, dy = ex - ox, ey - oy
        R = torch.sqrt(dx * dx + dy * dy).clamp(min=1.0)
        rx, ry = dx / R, dy / R
        cx, cy = -ry, rx
        sc2 = (R * cf) ** 2 + 1e-6
        sr2_eff = sr ** 2 + 1e-9
        nr = torch.randn_like(R) * sr
        nc = torch.randn_like(R) * (R * cf)
        mx = ex + nr * rx + nc * cx
        my = ey + nr * ry + nc * cy
        a, b = 1.0 / sr2_eff, 1.0 / sc2
        i00 = a * rx * rx + b * cx * cx
        i01 = a * rx * ry + b * cx * cy
        i11 = a * ry * ry + b * cy * cy
        L00 += i00; L01 += i01; L11 += i11
        e0 += i00 * mx + i01 * my
        e1 += i01 * mx + i11 * my
    # Home-on-jam / ESM beacon: target's own emission feeds an isotropic info term.
    # info ∝ exposure_gain · emission² (received jammer power → ESM bearing SNR).
    if emission is not None and exposure_gain > 0.0:
        binfo = (exposure_gain * emission * emission).clamp(min=0.0) + 1e-9
        bstd = 1.0 / torch.sqrt(binfo)
        bx = ex + torch.randn_like(ex) * bstd
        by = ey + torch.randn_like(ey) * bstd
        L00 = L00 + binfo; L11 = L11 + binfo
        e0 = e0 + binfo * bx
        e1 = e1 + binfo * by
    det = (L00 * L11 - L01 * L01).clamp(min=1e-9)
    zx = (L11 * e0 - L01 * e1) / det
    zy = (-L01 * e0 + L00 * e1) / det
    return zx, zy, L11 / det, -L01 / det, L00 / det


# ---------------------------------------------------------------------------
# Kalman filter (2x2 closed form)
# ---------------------------------------------------------------------------

def _kalman_step(
    x0: torch.Tensor, x1: torch.Tensor,
    P00: torch.Tensor, P01: torch.Tensor, P11: torch.Tensor,
    zx: torch.Tensor, zy: torch.Tensor,
    R00: torch.Tensor, R01: torch.Tensor, R11: torch.Tensor,
    q: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Kalman predict (random walk, +q) + update with measurement (z, R). 2x2 closed form."""
    P00 = P00 + q; P11 = P11 + q
    S00 = P00 + R00; S01 = P01 + R01; S11 = P11 + R11
    sdet = (S00 * S11 - S01 * S01).clamp(min=1e-9)
    Si00 = S11 / sdet; Si01 = -S01 / sdet; Si11 = S00 / sdet
    K00 = P00 * Si00 + P01 * Si01; K01 = P00 * Si01 + P01 * Si11
    K10 = P01 * Si00 + P11 * Si01; K11 = P01 * Si01 + P11 * Si11
    yx = zx - x0; yy = zy - x1
    nx0 = x0 + K00 * yx + K01 * yy
    nx1 = x1 + K10 * yx + K11 * yy
    nP00 = (1 - K00) * P00 - K01 * P01
    nP01 = (1 - K00) * P01 - K01 * P11
    nP11 = -K10 * P01 + (1 - K11) * P11
    return nx0, nx1, nP00, nP01, nP11


# ---------------------------------------------------------------------------
# KalmanTracker — encapsulates per-team per-enemy track state
# ---------------------------------------------------------------------------

class KalmanTracker:
    """Per-episode Kalman track state for 2 enemies × n_teams.

    Lives inside TeamPPOTrainer; reset at episode start via `reset()`.
    The fused measurement is computed externally via `fused_sensing(track=True)`,
    which mutates the tracker in place.
    """

    def __init__(self, track_q_m: float = 0.05, track_burnin: int = 30,
                 acq_baseline_m: float = 0.0):
        self.track_q_m = float(track_q_m)
        self.track_burnin = int(track_burnin)
        self.acq_baseline_m = float(acq_baseline_m)
        self._initialized = False
        self._trk_x: Optional[torch.Tensor] = None
        self._trk_P: Optional[torch.Tensor] = None

    def reset(self):
        """Clear track state at episode start."""
        self._initialized = False
        self._trk_x = None
        self._trk_P = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def trace_P(self) -> Optional[torch.Tensor]:
        """Per-env per-team trace of the 2×2 enemy-track covariance.

        Returns [E, T, 2] tensor (one trace per enemy) or None if uninitialized.
        Consumed by Concerto composer (event trigger θ2) and noise-robust CTDE
        (α_eff weighting) — both need a scalar uncertainty signal per team.
        """
        if not self._initialized or self._trk_P is None:
            return None
        # _trk_P shape: [E, T, 2_enemies, 2, 2]; trace each 2×2 → [E, T, 2]
        return self._trk_P[..., 0, 0] + self._trk_P[..., 1, 1]

    def ensure_alloc(self, E: int, T: int, device: torch.device):
        if not self._initialized:
            self._trk_x = torch.zeros(E, T, 2, 2, device=device)
            self._trk_P = torch.zeros(E, T, 2, 2, 2, device=device)


# ---------------------------------------------------------------------------
# fused_sensing — top-level entry used by get_own_actions
# ---------------------------------------------------------------------------

def fused_sensing(
    obs: torch.Tensor,
    half_x: float,
    half_y: float,
    range_sigma_m: float,
    crossrange_factor: float,
    tracker: Optional[KalmanTracker] = None,
    jam_gain: float = 0.0,
    exposure_gain: float = 0.0,
    jam_level: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-static range triangulation (+ optional Kalman tracking).

    Information-filter fusion of the own radars' anisotropic measurements makes
    both axes ~range-precise where the angular baseline is good. If `tracker`
    is provided, additionally runs a 2x2 Kalman filter over time (track=True path).

    Writes the fused enemy_xy IN-PLACE into obs[68:72]. Each episode the track
    is WARM-STARTED with track_burnin pre-convergence updates so the anchor is
    tight at step 0 (otherwise within-episode convergence makes obs non-stationary
    and destabilises PPO).

    Args:
        obs: [E, T, 76] commander observation.
        half_x, half_y: map half-extents in metres.
        range_sigma_m: range σ.
        crossrange_factor: crossrange σ / R.
        tracker: KalmanTracker instance (track=True) or None (fused-only).
        jam_gain: enemy jam noise multiplier (0 disables EW coupling).
        exposure_gain: home-on-jam beacon strength (0 disables).
        jam_level: [E, n_teams] current jam level per team (for jam_gain + emission).
    Returns:
        obs (modified in-place).
    """
    sr2 = range_sigma_m ** 2
    q = (tracker.track_q_m if tracker else 0.05) ** 2
    own = [(obs[..., 0] * half_x, obs[..., 1] * half_y),    # own radar-0 (sensor A)
           (obs[..., 2] * half_x, obs[..., 3] * half_y)]    # own radar-1 (sensor B)
    # Jam coupling: enemy team's jam multiplies my sensing noise floor.
    # enemy_jam[E,T] = the jam level of team t's enemy (flip swaps team↔enemy).
    if jam_level is not None and jam_gain > 0.0:
        enemy_jam = jam_level.flip(-1)
        jam_mul = 1.0 + jam_gain * enemy_jam
    else:
        enemy_jam = jam_level.flip(-1) if jam_level is not None else None
        jam_mul = None
    emission = enemy_jam if (exposure_gain > 0.0 and enemy_jam is not None) else None

    track = tracker is not None
    if track and not tracker.is_initialized:
        E, T = obs.shape[0], obs.shape[1]
        tracker.ensure_alloc(E, T, obs.device)

    for e in range(2):  # two enemy radars
        off = 68 + 2 * e
        ex = obs[..., off] * half_x      # true enemy position (m) [E, T]
        ey = obs[..., off + 1] * half_y
        zx, zy, R00, R01, R11 = _fuse_one(
            ex, ey, own, range_sigma_m, crossrange_factor,
            jam_mul=jam_mul, emission=emission, exposure_gain=exposure_gain,
        )
        zx = zx.clamp(-half_x, half_x); zy = zy.clamp(-half_y, half_y)
        if not track:
            obs[..., off] = zx / half_x
            obs[..., off + 1] = zy / half_y
            continue
        if not tracker.is_initialized:
            # Warm-start: pre-converge with track_burnin fused measurements.
            # If acq_baseline_m>0, the radars sweep perpendicular to their LOS
            # (opposite senses) to widen the angular baseline → geometry diversity
            # that collapses bad-collinear GDOP.
            x0, x1, P00, P01, P11 = zx, zy, R00, R01, R11
            K = max(tracker.track_burnin, 1)
            for k in range(tracker.track_burnin):
                own_k = own
                if tracker.acq_baseline_m > 0.0:
                    d = tracker.acq_baseline_m * ((k + 1) / K - 0.5)
                    own_k = []
                    for ri, (ox, oy) in enumerate(own):
                        dxr, dyr = ex - ox, ey - oy
                        Rr = torch.sqrt(dxr * dxr + dyr * dyr).clamp(min=1.0)
                        sgn = 1.0 if ri == 0 else -1.0
                        own_k.append((ox - sgn * d * dyr / Rr, oy + sgn * d * dxr / Rr))
                bzx, bzy, BR00, BR01, BR11 = _fuse_one(
                    ex, ey, own_k, range_sigma_m, crossrange_factor,
                    jam_mul=jam_mul, emission=emission, exposure_gain=exposure_gain,
                )
                bzx = bzx.clamp(-half_x, half_x); bzy = bzy.clamp(-half_y, half_y)
                x0, x1, P00, P01, P11 = _kalman_step(
                    x0, x1, P00, P01, P11, bzx, bzy, BR00, BR01, BR11, q)
            # [ANCHOR-AB] Diagnostic: warm-start final state for enemy e.
            # Fires once per enemy per first-call (use module-level flag).
            global _ANCHOR_AB_PRINTED
            if "_ANCHOR_AB_PRINTED" not in globals() or len(_ANCHOR_AB_PRINTED) < 2:
                _ANCHOR_AB_PRINTED = globals().get("_ANCHOR_AB_PRINTED", set())
                if e not in _ANCHOR_AB_PRINTED:
                    _ANCHOR_AB_PRINTED.add(e)
                    err_x = (x0 - ex).abs().max().item()
                    err_y = (x1 - ey).abs().max().item()
                    print(f"[ANCHOR-AB] enemy={e} burnin={tracker.track_burnin} "
                          f"max_err_x={err_x:.3f}m max_err_y={err_y:.3f}m "
                          f"(true=({ex[0,0].item():.1f},{ey[0,0].item():.1f}) "
                          f"est=({x0[0,0].item():.1f},{x1[0,0].item():.1f}))",
                          flush=True)
        else:
            x0 = tracker._trk_x[..., e, 0]; x1 = tracker._trk_x[..., e, 1]
            P00 = tracker._trk_P[..., e, 0, 0]; P01 = tracker._trk_P[..., e, 0, 1]
            P11 = tracker._trk_P[..., e, 1, 1]
            x0, x1, P00, P01, P11 = _kalman_step(
                x0, x1, P00, P01, P11, zx, zy, R00, R01, R11, q)
        x0 = x0.clamp(-half_x, half_x); x1 = x1.clamp(-half_y, half_y)
        tracker._trk_x[..., e, 0] = x0; tracker._trk_x[..., e, 1] = x1
        tracker._trk_P[..., e, 0, 0] = P00; tracker._trk_P[..., e, 0, 1] = P01
        tracker._trk_P[..., e, 1, 0] = P01; tracker._trk_P[..., e, 1, 1] = P11
        obs[..., off] = x0 / half_x
        obs[..., off + 1] = x1 / half_y

    if track:
        tracker._initialized = True
    return obs


# ---------------------------------------------------------------------------
# Radar baseline enforcement — guarantees a triangulation crossing angle
# ---------------------------------------------------------------------------

def enforce_radar_baseline(env, min_baseline_m: float):
    """Push each team's two radars apart to at least min_baseline_m, keeping midpoint.

    Guarantees a good triangulation crossing angle so cm-range fusion localises
    the target to sub-0.2m without tracking. Random placement sometimes lands the
    two radars near-collinear; this removes those bad geometries (mirrors a real
    SAM battery deliberately spreading radar vehicles).

    No-op if min_baseline_m <= 0.
    """
    if min_baseline_m <= 0.0:
        return
    rp = env.radar_pos  # [E, R, 3]
    for t in range(env.n_teams):
        idx = env.battlefield.team_radar_indices[t]
        if len(idx) < 2:
            continue
        a, b = int(idx[0]), int(idx[1])
        pa = rp[:, a, :2]
        pb = rp[:, b, :2]
        mid = 0.5 * (pa + pb)
        d = pb - pa
        dist = d.norm(dim=-1, keepdim=True).clamp(min=1.0)
        unit = d / dist
        half = 0.5 * dist.clamp(min=min_baseline_m)
        rp[:, a, :2] = mid - unit * half
        rp[:, b, :2] = mid + unit * half
