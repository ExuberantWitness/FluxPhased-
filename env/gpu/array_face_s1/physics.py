"""S1 link budget with radar array factor — vectorized over env batch.

Adapts lite's scalar link budget (`compute_service_jnr_db`) by adding a per-env
radar array factor term that depends on the radar's current beam_az.

    JNR_db = P_tx_dBm + jam_antenna_gain_db + (vic_rx_gain_db + AF_rx_db)
             - L_path_db - L_pol_db + overlap_db - N_dBm

Where `AF_rx_db[E]` is the radar ULA's response at the jammer's broadside
direction given the radar's current beam_az. Peak (beam_az=0) = 0 dB; all
other directions < 0 dB.

Compared to lite: `vic_svc.rx_gain_db` is replaced by
`(vic_svc.rx_gain_db + AF_rx_db)`. Everything else (P_jam, path loss,
overlap, noise, detection sigmoid) is identical to lite.
"""
from __future__ import annotations
import torch

from env.gpu.g3_bsta_lite.physics import (
    SPEED_OF_LIGHT, DebugPhysicsConfig,
)
from env.gpu.array_face_s1.array_factor import (
    RadarULAConfig, compute_radar_af_db,
)


def compute_jnr_db_s1(
    physics: DebugPhysicsConfig,
    radar: RadarULAConfig,
    *,
    jammer_active: torch.Tensor,
    jammer_service_id: torch.Tensor,
    victim_service_id: torch.Tensor,
    radar_beam_az_idx: torch.Tensor,
) -> torch.Tensor:
    """Vectorized S1 JNR.

    Args:
        physics: lite DebugPhysicsConfig (services, P_jam_W, geometry, etc.)
        radar: RadarULAConfig
        jammer_active: [E] bool
        jammer_service_id: [E] int64 in {0, 1} (meaningful only where jammer_active)
        victim_service_id: [E] int64 in {0, 1} (radar's current svc to detect)
        radar_beam_az_idx: [E] int64 in 0..n_beam_dirs-1

    Returns:
        [E] float32 JNR in dB. Returns -inf where not jammer_active.
        Cross-svc jamming (jam_svc != vic_svc) is handled by overlap_frac→0
        (same behavior as lite scalar version).
    """
    if jammer_active.dim() != 1:
        raise ValueError(f"jammer_active must be [E] 1-D, got {tuple(jammer_active.shape)}")
    E = jammer_active.shape[0]
    device = jammer_active.device

    services = (physics.service_0, physics.service_1)
    if any(s.fc_hz <= 0 for s in services):
        raise ValueError(f"S1 physics requires positive fc_hz for both services")

    fc_hz_table = torch.tensor([s.fc_hz for s in services], device=device, dtype=torch.float32)
    bw_hz_table = torch.tensor([s.bw_hz for s in services], device=device, dtype=torch.float32)
    rx_gain_table = torch.tensor([s.rx_gain_db for s in services], device=device, dtype=torch.float32)

    vic_fc = fc_hz_table.gather(0, victim_service_id.long())
    vic_bw = bw_hz_table.gather(0, victim_service_id.long())
    vic_rx_gain = rx_gain_table.gather(0, victim_service_id.long())
    jam_fc = fc_hz_table.gather(0, jammer_service_id.long())
    jam_bw = bw_hz_table.gather(0, jammer_service_id.long())

    # Spectral overlap (rectangular windows) — vectorized form of lite's _rect_overlap_frac
    lo_i = jam_fc - jam_bw / 2.0
    hi_i = jam_fc + jam_bw / 2.0
    lo_j = vic_fc - vic_bw / 2.0
    hi_j = vic_fc + vic_bw / 2.0
    overlap = torch.clamp(torch.minimum(hi_i, hi_j) - torch.maximum(lo_i, lo_j), min=0.0) / vic_bw
    overlap_db = 10.0 * torch.log10(overlap + 1e-15)

    # Path loss (using victim frequency)
    d = max(float(physics.distance_jm), float(physics.distance_floor_m))
    lambda_m = SPEED_OF_LIGHT / vic_fc
    L_path_db = 20.0 * torch.log10(4.0 * torch.pi * d / lambda_m)

    # Tx power (scalar; same P_jam for all envs)
    P_tx_dBm = 10.0 * torch.log10(torch.tensor(float(physics.P_jam_W) * 1000.0, device=device))

    # Noise per env (depends on vic_bw)
    N_dBm = -174.0 + 10.0 * torch.log10(vic_bw) + float(physics.noise_figure_db)

    # Radar ULA array factor
    af_rx_db = compute_radar_af_db(radar, radar_beam_az_idx=radar_beam_az_idx)  # [E]

    # Assemble JNR
    J_dBm = (P_tx_dBm
             + float(physics.jam_antenna_gain_db)
             + vic_rx_gain
             + af_rx_db
             - L_path_db
             - float(physics.polarization_loss_db)
             + overlap_db)
    jnr = J_dBm - N_dBm

    jnr = torch.where(jammer_active, jnr, torch.full_like(jnr, float("-inf")))
    return jnr.to(torch.float32)


def compute_p_detect_s1(
    physics: DebugPhysicsConfig,
    *,
    baseline_snr_db: float,
    jnr_db: torch.Tensor,
) -> torch.Tensor:
    """Sigmoid P_detect (vectorized, same form as lite's batch_p_detect_for_service).

    P_detect = sigmoid((10*log10(snr_eff_lin) + coh_gain - thr) / width)
    where snr_eff_lin = snr_lin / (1 + jnr_lin).
    """
    jnr_lin = torch.where(
        torch.isfinite(jnr_db),
        10.0 ** (jnr_db / 10.0),
        torch.zeros_like(jnr_db),
    )
    snr_lin = 10.0 ** (baseline_snr_db / 10.0)
    snr_eff_lin = snr_lin / (1.0 + jnr_lin)
    snr_eff_db = 10.0 * torch.log10(snr_eff_lin.clamp(min=1e-12)) + float(physics.coherent_gain_db)
    p = torch.sigmoid((snr_eff_db - float(physics.detect_threshold_db)) / float(physics.detect_width_db))
    return p.to(torch.float32)
