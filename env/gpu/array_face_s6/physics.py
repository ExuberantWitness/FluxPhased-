"""S6 link budget — per-radar JNR with generalized (off-broadside) AFs.

Each radar i sits at its own bearing (az_i, el_i); the jammer's Tx gain
toward radar i and radar i's Rx gain toward the jammer are evaluated with
compute_upa_af_db_toward (arbitrary target), replacing S4's broadside-only
special case. The per-radar chain is the S3 budget (P_cell + 20·log10(N_active)
+ gains + AFs − losses − noise) with two S6 simplifications:

  (1) The jammer is service-agnostic (broadband noise): spectral overlap is
      exactly 1.0 (0 dB) against BOTH radars regardless of their service
      choice (S4 had one radar; with two potentially on different services
      a service head would be needed — deliberately out of scope).
  (2) Path loss uses each radar's victim-service fc (as S3).

M0 gates:
  - target at broadside (az=el=0) + radar steered identically == S4 JNR
  - jammer idle → -inf for both radars
  - a beam optimal toward +20° is suboptimal toward −20° (asymmetry)
"""
from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite.physics import SPEED_OF_LIGHT, DebugPhysicsConfig
from env.gpu.array_face_s4.array_factor import UPAConfig
from env.gpu.array_face_s6.array_factor import compute_upa_af_db_toward


def compute_jnr_db_s6(
    physics: DebugPhysicsConfig,
    radar: UPAConfig,
    jammer: UPAConfig,
    *,
    jammer_active: torch.Tensor,      # [E] bool
    radar_beam_az_idx: torch.Tensor,  # [E, R] int64
    radar_beam_el_idx: torch.Tensor,  # [E, R] int64
    jammer_beam_az_idx: torch.Tensor, # [E] int64
    jammer_beam_el_idx: torch.Tensor, # [E] int64
    cell_mask: torch.Tensor,          # [E, 25] float
    radar_az_rad: torch.Tensor,       # [R] float (bearings of the radars)
    radar_el_rad: torch.Tensor,       # [R] float
    victim_service_id: torch.Tensor,  # [E, R] int64 (each radar's current svc)
) -> torch.Tensor:
    """Returns [E, R] per-radar JNR (dB); -inf where the jammer is idle."""
    E = jammer_active.shape[0]
    R = radar_az_rad.shape[0]
    device = jammer_active.device

    services = (physics.service_0, physics.service_1)
    if any(s.fc_hz <= 0 for s in services):
        raise ValueError("S6 physics requires positive fc_hz for both services")
    fc_table = torch.tensor([s.fc_hz for s in services], device=device, dtype=torch.float32)
    bw_table = torch.tensor([s.bw_hz for s in services], device=device, dtype=torch.float32)
    rx_gain_table = torch.tensor([s.rx_gain_db for s in services], device=device, dtype=torch.float32)

    n_active = cell_mask.sum(dim=-1).clamp(min=1).to(torch.float32)  # [E]
    P_cell_dBm = 10.0 * torch.log10(torch.tensor(float(physics.P_jam_W) * 1000.0, device=device))
    P_peak_dBm = P_cell_dBm + 20.0 * torch.log10(n_active.clamp(min=1e-12))

    jnr = torch.zeros(E, R, device=device, dtype=torch.float32)
    for i in range(R):
        vic_fc = fc_table.gather(0, victim_service_id[:, i].long())
        vic_bw = bw_table.gather(0, victim_service_id[:, i].long())
        vic_rx_gain = rx_gain_table.gather(0, victim_service_id[:, i].long())

        d = max(float(physics.distance_jm), float(physics.distance_floor_m))
        lambda_m = SPEED_OF_LIGHT / vic_fc
        L_path_db = 20.0 * torch.log10(4.0 * torch.pi * d / lambda_m)
        N_dBm = -174.0 + 10.0 * torch.log10(vic_bw) + float(physics.noise_figure_db)

        tgt_az = radar_az_rad[i].expand(E)
        tgt_el = radar_el_rad[i].expand(E)
        af_tx_db = compute_upa_af_db_toward(
            jammer, beam_az_idx=jammer_beam_az_idx, beam_el_idx=jammer_beam_el_idx,
            target_az_rad=tgt_az, target_el_rad=tgt_el,
        )
        # Rx gain of radar_i toward the jammer: by even symmetry the bearing
        # from radar_i to the jammer equals (−az_i, −el_i) ≡ (az_i, el_i)
        # for |AF|, so the same target serves both link ends.
        af_rx_db = compute_upa_af_db_toward(
            radar, beam_az_idx=radar_beam_az_idx[:, i], beam_el_idx=radar_beam_el_idx[:, i],
            target_az_rad=tgt_az, target_el_rad=tgt_el,
        )

        J_dBm = (P_peak_dBm
                 + float(physics.jam_antenna_gain_db) + af_tx_db
                 + vic_rx_gain + af_rx_db
                 - L_path_db - float(physics.polarization_loss_db))  # overlap = 0 dB (broadband)
        jnr[:, i] = J_dBm - N_dBm

    jnr = torch.where(jammer_active.unsqueeze(1), jnr, torch.full_like(jnr, float("-inf")))
    return jnr.to(torch.float32)


def compute_p_detect_s6(
    physics: DebugPhysicsConfig,
    *,
    baseline_snr_db: float,
    jnr_db: torch.Tensor,             # [E, R]
) -> torch.Tensor:
    """Per-radar sigmoid P_detect — same formula as S1-S5, shape [E, R]."""
    from env.gpu.array_face_s3.physics import compute_p_detect_s3
    return compute_p_detect_s3(
        physics, baseline_snr_db=baseline_snr_db, jnr_db=jnr_db.reshape(-1),
    ).reshape(jnr_db.shape)


def compute_snr_eff_db_s6(
    physics: DebugPhysicsConfig,
    *,
    baseline_snr_db: float,
    jnr_db: torch.Tensor,             # [E, R]
) -> torch.Tensor:
    """Pre-sigmoid effective SNR (dB) per radar — the bearing-model hinge.

    p_detect(mission m at radar i) = sigmoid((snr_eff_db[i]
                                            + target_gain_db(i, m) − thr)/width)
    target_gain = 0 dB when radar i stares at m's bearing (S1-S5 semantics).
    """
    jnr_lin = torch.where(
        torch.isfinite(jnr_db), 10.0 ** (jnr_db / 10.0), torch.zeros_like(jnr_db))
    snr_lin = 10.0 ** (baseline_snr_db / 10.0)
    snr_eff_lin = snr_lin / (1.0 + jnr_lin)
    return 10.0 * torch.log10(snr_eff_lin.clamp(min=1e-12)) + float(physics.coherent_gain_db)


def target_gain_db(
    radar_cfg: UPAConfig,
    *,
    beam_az_idx: torch.Tensor,        # [E] or [E, R]
    beam_el_idx: torch.Tensor,
    mission_az_idx: torch.Tensor,     # same leading shape as beam_az_idx
) -> torch.Tensor:
    """Radar's normalized Rx gain (dB) toward a mission bearing (el = 0).

    0 dB when the beam's az matches the mission az and el_idx = 2 (el=0
    plane); negative off-axis. This is the scan-vs-stare coupling: staring
    at a mission maximizes detection gain, but the same pointing fixes the
    Rx pattern toward the jammer (handled separately in JNR).
    """
    import math
    flat_ba = beam_az_idx.reshape(-1).long()
    flat_be = beam_el_idx.reshape(-1).long()
    flat_ma = mission_az_idx.reshape(-1).long()
    az_grid_rad = torch.tensor(
        [math.radians(float(a)) for a in radar_cfg.beam_az_deg],
        device=beam_az_idx.device, dtype=torch.float32)
    tgt_az = az_grid_rad.gather(0, flat_ma)
    tgt_el = torch.zeros_like(tgt_az)
    out = compute_upa_af_db_toward(
        radar_cfg, beam_az_idx=flat_ba, beam_el_idx=flat_be,
        target_az_rad=tgt_az, target_el_rad=tgt_el,
    )
    return out.reshape(beam_az_idx.shape)
