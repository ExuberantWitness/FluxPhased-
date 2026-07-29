"""Isolated IQ adapter for G3-BSTA-lite debug env (F1 §6 route b).

This module does NOT modify the shared ``env/gpu/twoteam/iq_interference.py``
kernel. It implements a deliberately small, single-jammer / single-radar
physics path tuned for the two-service debug profile. The legacy 4-node
kernel assumes two 2-radar teams in mirror-symmetric configuration; that
topology is wrong for the 1+jammer / 1+radar debug env.

Physics chain (per DEBUG_CONTRACT.md §7):

    executed service action
      -> jammer power/frequency
      -> frequency overlap and path/receiver gain
      -> service-specific receiver JNR/SINR
      -> detector or measurement quality
      -> mission task outcome

The two services differ by center frequency and receiver antenna gain.
Service 0: 10.0 GHz, narrowbeam (high gain, smaller BW)
Service 1: 10.5 GHz, widebeam (lower gain, larger BW)

The radar opponent's detector on service k is a sigmoid on
``snr_eff_db - detect_threshold_db``. ``snr_eff_db`` degrades with JNR
applied to service k. The same jammer power at a non-matching frequency
produces overlap dilution (per legacy iq_interference._rect_overlap_frac).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


SPEED_OF_LIGHT = 299_792_458.0
BOLTZMANN_K = 1.38e-23
T0_KELVIN = 290.0


@dataclass
class ServiceChannel:
    fc_hz: float
    bw_hz: float
    rx_gain_db: float


@dataclass(frozen=True)
class DebugPhysicsConfig:
    service_0: ServiceChannel
    service_1: ServiceChannel
    # Jammer
    P_jam_W: float
    jam_antenna_gain_db: float
    # Geometry (m), one jammer vs one radar opponent
    distance_jm: float
    # Radar receiver / processing
    noise_figure_db: float
    coherent_gain_db: float
    detect_threshold_db: float
    detect_width_db: float
    polarization_loss_db: float = 3.0
    distance_floor_m: float = 100.0


def default_debug_physics_config(*, P_jam_W: float) -> DebugPhysicsConfig:
    return DebugPhysicsConfig(
        service_0=ServiceChannel(fc_hz=10.0e9, bw_hz=10e6, rx_gain_db=35.0),
        service_1=ServiceChannel(fc_hz=10.5e9, bw_hz=20e6, rx_gain_db=30.0),
        P_jam_W=P_jam_W,
        jam_antenna_gain_db=20.0,
        distance_jm=8_000.0,
        noise_figure_db=5.0,
        coherent_gain_db=20.0,
        detect_threshold_db=15.0,
        detect_width_db=3.0,
    )


def _rect_overlap_frac(f_i: float, f_j: float, bw_i: float, bw_j: float) -> float:
    """Fraction of victim band overlapped by interferer band (symmetric rect)."""
    half_i = bw_i / 2.0
    half_j = bw_j / 2.0
    lo_i, hi_i = f_i - half_i, f_i + half_i
    lo_j, hi_j = f_j - half_j, f_j + half_j
    lo = max(lo_i, lo_j)
    hi = min(hi_i, hi_j)
    if hi <= lo:
        return 0.0
    return (hi - lo) / bw_j


def compute_service_jnr_db(
    cfg: DebugPhysicsConfig,
    *,
    jammer_active: bool,
    jammer_service_id: int,
    victim_service_id: int,
) -> float:
    """One-way JNR (dB) at victim service receiver from a single jammer.

    Returns -inf when jammer is inactive. JNR = J / N where
        J_dBm = P_tx_dBm + G_tx_dB + G_rx_dB - L_path_dB - L_pol
                + 10 log10(overlap)
        N_dBm = -174 + 10 log10(B) + NF
    """
    if not jammer_active:
        return float("-inf")
    jam_svc = cfg.service_0 if jammer_service_id == 0 else cfg.service_1
    vic_svc = cfg.service_0 if victim_service_id == 0 else cfg.service_1

    d = max(cfg.distance_jm, cfg.distance_floor_m)
    lambda_m = SPEED_OF_LIGHT / jam_svc.fc_hz
    L_path_db = 20.0 * math.log10(4.0 * math.pi * d / lambda_m)
    overlap = _rect_overlap_frac(jam_svc.fc_hz, vic_svc.fc_hz, jam_svc.bw_hz, vic_svc.bw_hz)
    overlap_db = 10.0 * math.log10(overlap + 1e-15)
    P_tx_dBm = 10.0 * math.log10(cfg.P_jam_W * 1000.0)
    J_dBm = (
        P_tx_dBm
        + cfg.jam_antenna_gain_db
        + vic_svc.rx_gain_db
        - L_path_db
        - cfg.polarization_loss_db
        + overlap_db
    )
    N_dBm = -174.0 + 10.0 * math.log10(vic_svc.bw_hz) + cfg.noise_figure_db
    return J_dBm - N_dBm


def compute_detection_probability(
    cfg: DebugPhysicsConfig,
    *,
    baseline_snr_db: float,
    jnr_db: float,
) -> float:
    """P_detect for a victim service given baseline SNR and JNR.

    Effective SNR = baseline_snr / (1 + JNR_lin) — closed-form for noise-
    limited receiver with additive Gaussian jammer (matches legacy
    iq_interference.compute_meas_sigma form).
    """
    jnr_lin = 0.0 if not math.isfinite(jnr_db) else 10.0 ** (jnr_db / 10.0)
    snr_lin = 10.0 ** (baseline_snr_db / 10.0)
    snr_eff_lin = snr_lin / (1.0 + jnr_lin)
    snr_eff_db = 10.0 * math.log10(max(snr_eff_lin, 1e-12))
    # Apply coherent processing gain.
    snr_eff_db += cfg.coherent_gain_db
    p = 1.0 / (1.0 + math.exp(-(snr_eff_db - cfg.detect_threshold_db) / cfg.detect_width_db))
    return p


def batch_p_detect_for_service(
    cfg: DebugPhysicsConfig,
    *,
    baseline_snr_db: float,
    jnr_db_per_env: torch.Tensor,
) -> torch.Tensor:
    """Vectorized P_detect for one service across a batch.

    Args:
      baseline_snr_db: scalar baseline SNR (dB) for the service being probed.
      jnr_db_per_env: [E] tensor of JNR(dB) per env at that service.

    Returns: [E] detection probability.
    """
    jnr_lin = torch.where(
        torch.isfinite(jnr_db_per_env),
        10.0 ** (jnr_db_per_env / 10.0),
        torch.zeros_like(jnr_db_per_env),
    )
    snr_lin = torch.tensor(10.0 ** (baseline_snr_db / 10.0), dtype=jnr_lin.dtype)
    snr_eff_lin = snr_lin / (1.0 + jnr_lin)
    snr_eff_db = 10.0 * torch.log10(snr_eff_lin.clamp(min=1e-12))
    snr_eff_db = snr_eff_db + cfg.coherent_gain_db
    return torch.sigmoid((snr_eff_db - cfg.detect_threshold_db) / cfg.detect_width_db)
