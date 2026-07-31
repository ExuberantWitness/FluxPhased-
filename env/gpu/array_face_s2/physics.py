"""S2 link budget with BOTH jammer Tx AF and radar Rx AF.

    JNR_db = P_tx_dBm + jam_antenna_gain_db + AF_tx_db
             + vic_rx_gain_db + AF_rx_db
             - L_path_db - L_pol_db + overlap_db - N_dBm

Compared to S1: adds the `AF_tx_db` term from the jammer ULA. Both AFs peak at
0 dB when their respective beam_az idx = 0, and roll off symmetrically.

S2 introduces **per-cell power budget**: the jammer has 5 cells, each at
P_cell_W (default 2.0 W), so total radiated power = N_cells * P_cell_W = 10 W
when all cells active. This is encoded by the physics config's P_jam_W field
(which the env config sets to N_cells * P_cell_W).

Default P_jam_W for S2 = 5 * 2.0 = 10.0 W (vs S1 / lite's 50.0 W).
Net effect: JNR is similar because the +AF_tx (main lobe gain) compensates for
lower total power.
"""
from __future__ import annotations
import torch

from env.gpu.g3_bsta_lite.physics import (
    SPEED_OF_LIGHT, DebugPhysicsConfig,
)
from env.gpu.array_face_s2.array_factor import (
    RadarULAConfig, JammerULAConfig,
    compute_radar_af_db, compute_jammer_af_db,
)


def compute_jnr_db_s2(
    physics: DebugPhysicsConfig,
    radar: RadarULAConfig,
    jammer: JammerULAConfig,
    *,
    jammer_active: torch.Tensor,
    jammer_service_id: torch.Tensor,
    victim_service_id: torch.Tensor,
    radar_beam_az_idx: torch.Tensor,
    jammer_beam_az_idx: torch.Tensor,
) -> torch.Tensor:
    """Vectorized S2 JNR with both AFs.

    Args:
        physics: lite DebugPhysicsConfig (services, P_jam_W, geometry, etc.)
        radar: RadarULAConfig (S1 radar ULA, unchanged)
        jammer: JammerULAConfig (new S2 jammer ULA)
        jammer_active: [E] bool
        jammer_service_id: [E] int64 in {0, 1}
        victim_service_id: [E] int64 in {0, 1} (radar's current svc to detect)
        radar_beam_az_idx: [E] int64 in 0..radar.n_beam_dirs-1
        jammer_beam_az_idx: [E] int64 in 0..jammer.n_beam_dirs-1

    Returns:
        [E] float32 JNR in dB. -inf where not jammer_active.
    """
    if jammer_active.dim() != 1:
        raise ValueError(f"jammer_active must be [E] 1-D, got {tuple(jammer_active.shape)}")
    E = jammer_active.shape[0]
    device = jammer_active.device

    services = (physics.service_0, physics.service_1)
    if any(s.fc_hz <= 0 for s in services):
        raise ValueError(f"S2 physics requires positive fc_hz for both services")

    fc_hz_table = torch.tensor([s.fc_hz for s in services], device=device, dtype=torch.float32)
    bw_hz_table = torch.tensor([s.bw_hz for s in services], device=device, dtype=torch.float32)
    rx_gain_table = torch.tensor([s.rx_gain_db for s in services], device=device, dtype=torch.float32)

    vic_fc = fc_hz_table.gather(0, victim_service_id.long())
    vic_bw = bw_hz_table.gather(0, victim_service_id.long())
    vic_rx_gain = rx_gain_table.gather(0, victim_service_id.long())
    jam_fc = fc_hz_table.gather(0, jammer_service_id.long())
    jam_bw = bw_hz_table.gather(0, jammer_service_id.long())

    # Spectral overlap
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

    # Tx peak EIRP: P_cell × N² (coherent combining at main lobe).
    # physics.P_jam_W is per-cell power (P_cell_W). JammerULAConfig.n_cells gives N.
    # Peak EIRP_dBm = P_cell_dBm + 20log10(N). AF_db (added below) modulates off-peak.
    N = float(jammer.n_cells)
    P_cell_dBm = 10.0 * torch.log10(torch.tensor(float(physics.P_jam_W) * 1000.0, device=device))
    P_peak_dBm = P_cell_dBm + 20.0 * torch.log10(torch.tensor(N, device=device))

    # Noise
    N_dBm = -174.0 + 10.0 * torch.log10(vic_bw) + float(physics.noise_figure_db)

    # Both AFs
    af_rx_db = compute_radar_af_db(radar, radar_beam_az_idx=radar_beam_az_idx)
    af_tx_db = compute_jammer_af_db(jammer, jammer_beam_az_idx=jammer_beam_az_idx)

    J_dBm = (P_peak_dBm
             + float(physics.jam_antenna_gain_db)
             + af_tx_db
             + vic_rx_gain
             + af_rx_db
             - L_path_db
             - float(physics.polarization_loss_db)
             + overlap_db)
    jnr = J_dBm - N_dBm

    jnr = torch.where(jammer_active, jnr, torch.full_like(jnr, float("-inf")))
    return jnr.to(torch.float32)


def compute_p_detect_s2(
    physics: DebugPhysicsConfig,
    *,
    baseline_snr_db: float,
    jnr_db: torch.Tensor,
) -> torch.Tensor:
    """Sigmoid P_detect (identical formula to S1 / lite, kept for module symmetry)."""
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
