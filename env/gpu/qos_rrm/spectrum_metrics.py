"""QoS-RRM metrics: derive per-function QoS signals from env step output.

Four quantities feed (a) the Concerto composer's event triggers and (b) the
pilot's QoS-satisfaction scoring:

  - Pd_at_pfa          : per-radar probability of detection at Pfa=1e-4 (detect fn)
  - trace_P_norm       : normalized Kalman cov trace [E, T] in [0, 1] (track fn)
  - crc_pass_rate      : per-team comm reliability (comm fn)
  - jam_power_on_victim: per-team received jam power in dB (jam effectiveness)

All are computed from existing env tensors; no new physics. The functions are
pure (no env mutation) so they can be unit-tested with synthetic inputs.

Conventions:
  - E = num envs, R = num radars, T = num teams, N = num elements
  - All metrics returned as torch.Tensor on the input device.
"""

from __future__ import annotations

import math
import torch
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# Pd@Pfa — detection probability from spectrum
# ---------------------------------------------------------------------------

def pd_at_pfa(
    spectrum: torch.Tensor,
    task_ids: torch.Tensor,
    pfa: float = 1e-4,
) -> torch.Tensor:
    """Probability of detection at given Pfa, per radar.

    Uses a simple CFAR-like threshold: noise floor = median across frequency
    bins, signal = peak across frequency bins, for elements allocated to
    TASK_DETECT (task_id == 1). Pd = fraction of detect elements whose
    SNR (dB) exceeds the theoretical threshold for Pfa (sigma × sqrt(-2 ln Pfa)).

    Args:
        spectrum: [E, R, N, P, n_bins] complex spectrum (magnitude^2 if real).
        task_ids: [E, R, N] long tensor (0=recon, 1=detect, 2=jam, 3=comm).
        pfa: false-alarm probability (default 1e-4 → ~4.0 sigma).
    Returns:
        [E, R] float in [0, 1].
    """
    E, R, N, P, B = spectrum.shape
    TASK_DETECT = 1
    detect_mask = (task_ids == TASK_DETECT).float().unsqueeze(-1).unsqueeze(-1)  # [E, R, N, 1, 1]

    # Power: magnitude^2 across pulses & bins, masked to detect elems
    power = (spectrum.abs() ** 2) * detect_mask  # [E, R, N, P, B]
    peak = power.amax(dim=-1).amax(dim=-1).amax(dim=-1)  # [E, R]
    noise = power.median(dim=-1).values.median(dim=-1).values.median(dim=-1).values.clamp(min=1e-30)  # [E, R]
    snr_linear = (peak / noise).clamp(min=1e-30)
    snr_db = 10.0 * torch.log10(snr_linear)

    # Theoretical threshold for Pfa (Gaussian noise): sigma^2 * sqrt(-2 ln Pfa)
    # Linear scale: ~4.29 for Pfa=1e-4 → 12.3 dB above noise median.
    thresh_db = 10.0 * math.log10(max(-2.0 * math.log(pfa), 1e-10))

    # Pd proxy: sigmoid around the detection threshold (4 dB transition width).
    pd = torch.sigmoid((snr_db - thresh_db) / 4.0)
    return pd.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# trace_P_norm — normalized Kalman covariance trace
# ---------------------------------------------------------------------------

def trace_P_norm(
    trace_P: Optional[torch.Tensor],
    init_max: float = 1e4,
) -> Optional[torch.Tensor]:
    """Normalize trace(P) to [0, 1] for composer trigger + α_eff weighting.

    trace_P is sum of diagonal entries of the 2×2 enemy-track covariance per
    team, returned by KalmanTracker.trace_P (shape [E, T, 2_enemies]). We take
    the max across enemies (worst-case) and normalize by an init_max constant
    (default 1e4 m^2 — corresponds to ~100m sigma per axis at filter init).

    Args:
        trace_P: [E, T, 2] tensor from KalmanTracker.trace_P, or None.
        init_max: normalization constant (m^2).
    Returns:
        [E, T] float in [0, 1], or None if trace_P is None.
    """
    if trace_P is None:
        return None
    worst = trace_P.max(dim=-1).values  # [E, T]
    return (worst / max(float(init_max), 1e-6)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# crc_pass_rate — comm reliability
# ---------------------------------------------------------------------------

def crc_pass_rate(
    comm_crc_ok: torch.Tensor,
    team_radar_indices = None,
    n_radars: Optional[int] = None,
) -> torch.Tensor:
    """Per-team CRC pass rate.

    Args:
        comm_crc_ok: [E, n_teams] bool tensor from env step output.
        team_radar_indices: optional list of tensors for per-radar expansion.
        n_radars: optional total radars (used to infer per-team count).
    Returns:
        [E, n_teams] float in [0, 1] (already in [0,1] since comm_crc_ok is bool).
    """
    return comm_crc_ok.float().clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# jam_power_on_victim — JSR proxy
# ---------------------------------------------------------------------------

def jam_power_on_victim_db(
    jam_level: torch.Tensor,
    jam_gain: float = 8.0,
    tx_power_db: float = 30.0,
) -> torch.Tensor:
    """JSR proxy in dB at the victim, per team.

    The env couples enemy jam multiplicatively into MY sensing noise floor via
    jam_mul = 1 + jam_gain × enemy_jam (sensing.py:259-264). We invert this to
    an effective JSR in dB at the victim's receiver:
        JSR_dB ≈ 10 * log10(jam_gain × enemy_jam)
                = noise_floor_db + 10*log10(jam_gain * enemy_jam / 1)

    For the composer's θ1=10dB trigger to fire at moderate jam:
        jam=0.5, jam_gain=8 → jam_mul=5 → +7 dB above noise.
        jam=1.0, jam_gain=8 → jam_mul=9 → +9.5 dB.

    We add tx_power_db (default 30 dBm = 1W) as the signal reference for a
    proper JSR (jammer-to-signal ratio), though for the pilot the relative
    ordering matters more than the absolute calibration.

    Args:
        jam_level: [E, n_teams] jam level per team (each team's own jam).
        jam_gain: env jam gain (default 8.0 from ew_mappo config).
        tx_power_db: victim TX power in dB for JSR normalization.
    Returns:
        [E, n_teams] float — JSR in dB at each team's receiver from its enemy.
    """
    enemy_jam = jam_level.flip(-1)  # team t receives team (1-t)'s jam
    jsr_linear = (jam_gain * enemy_jam).clamp(min=1e-10)
    jsr_db = 10.0 * torch.log10(jsr_linear)
    return jsr_db  # relative to noise floor; positive = jam dominates


# ---------------------------------------------------------------------------
# QoS satisfaction aggregate
# ---------------------------------------------------------------------------

def qos_satisfaction(
    pd: torch.Tensor,
    trace_norm: Optional[torch.Tensor],
    crc_rate: torch.Tensor,
    jsr_db: torch.Tensor,
    pd_thresh: float = 0.9,
    trace_thresh: float = 0.6,
    crc_thresh: float = 0.7,
    jsr_target_db: float = 6.0,
    team_radar_indices = None,
    n_teams: int = 2,
    ew_degradation: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Aggregate QoS satisfaction per team in [0, 1].

    Each function contributes a [0,1] score; satisfaction = average across the
    four functions. The Concerto composer uses this for the ε trigger (any fn
    below threshold) and as the headline metric for pilot verdict.

    Args:
        pd: [E, R] per-radar detection probability.
        trace_norm: [E, T] normalized trace(P) per team, or None (track→0).
        crc_rate: [E, T] per-team CRC pass rate.
        jsr_db: [E, T] per-team JSR at receiver.
        *thresh: per-function satisfaction thresholds.
        ew_degradation: optional [E, T] in [0,1] — fraction of sensing/comm
            denied by enemy jamming. When provided, degrades Pd and CRC
            multiplicatively (pd_eff = pd * (1 - 0.7*ew), crc_eff = crc*(1-0.9*ew))
            to model real EW impact on radar/comm receivers.
    Returns:
        dict with keys: detect, track, comm, jam, aggregate (all [E, T]).
    """
    E = pd.shape[0]
    n_teams_local = crc_rate.shape[1] if crc_rate.dim() > 1 else n_teams
    dev = pd.device

    # Apply EW degradation to Pd and CRC (if provided) — models the physical
    # reality that heavy jamming denies radar detection and comm decoding.
    pd_eff = pd
    crc_eff = crc_rate
    if ew_degradation is not None:
        # Reshape pd_eff per-team for per-team degradation
        # pd is [E, R]; we'll degrade each team's radars by its team-level ew.
        n_teams_local2 = ew_degradation.shape[1]
        R = pd.shape[1]
        r_per = R // n_teams_local2
        pd_degrade = torch.ones(E, R, device=dev)
        for t in range(n_teams_local2):
            pd_degrade[:, t * r_per:(t + 1) * r_per] = (1.0 - 0.7 * ew_degradation[:, t]).unsqueeze(-1)
        pd_eff = pd * pd_degrade
        crc_eff = crc_rate * (1.0 - 0.9 * ew_degradation)

    # Per-radar → per-team reduction (mean over team's radars)
    if team_radar_indices is None:
        # Assume symmetric: R // n_teams radars per team
        R = pd_eff.shape[1]
        r_per = R // n_teams_local
        pd_team = torch.zeros(E, n_teams_local, device=dev)
        for t in range(n_teams_local):
            pd_team[:, t] = pd_eff[:, t * r_per:(t + 1) * r_per].mean(dim=-1)
    else:
        pd_team = torch.stack([
            pd_eff[:, idx].mean(dim=-1) for idx in team_radar_indices
        ], dim=-1)

    # Detect: sigmoid around pd_thresh
    detect_score = torch.sigmoid((pd_team - pd_thresh) / 0.05)

    # Track: sigmoid around (1 - trace_thresh) — lower trace is better.
    # When ew_degradation is provided, also degrade track (Kalman diverges under jam).
    if trace_norm is not None:
        tr_eff = trace_norm
        if ew_degradation is not None:
            tr_eff = (trace_norm + 1.5 * ew_degradation).clamp(0.0, 1.0)
        track_score = torch.sigmoid(((1.0 - tr_eff) - (1.0 - trace_thresh)) / 0.05)
    else:
        track_score = torch.zeros(E, n_teams_local, device=dev)

    # Comm: sigmoid around crc_thresh (after EW degradation)
    comm_score = torch.sigmoid((crc_eff - crc_thresh) / 0.05)

    # Jam (defensive): how well MY sensing survives enemy jam.
    # jsr_db[t] already = JSR at team t's receiver from its enemy (see
    # jam_power_on_victim_db). Lower received JSR → higher score.
    jam_score = torch.sigmoid((jsr_target_db - jsr_db) / 3.0)

    aggregate = (detect_score + track_score + comm_score + jam_score) / 4.0
    return {
        "detect": detect_score,
        "track": track_score,
        "comm": comm_score,
        "jam": jam_score,
        "aggregate": aggregate,
    }
