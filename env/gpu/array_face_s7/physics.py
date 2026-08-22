"""S7 link budget — per-(jammer, radar) pair JNR, combined per radar in linear power.

S7 = S6's generalized per-radar chain (AF toward an arbitrary bearing) applied
to K=2 jammers, then combined with S5's incoherent two-source rule:

    JNR_7(e, r) = 10·log10( Σ_k 10^(JNR_kr / 10) )      [N_dBm cancels]

Each pair (k, r) uses its own relative bearing (geometry.pair_bearings):
jammer k's Tx AF toward radar r and radar r's Rx AF toward jammer k are both
evaluated at that bearing (|AF| even symmetry). Per-jammer idle → 0 mW
contribution; all idle → −inf.

M0 gates:
  - one jammer fully idle → combined == that single jammer's S6-style chain
    (per radar, at the same pair bearing)
  - two identical active jammers (same beam/cells/bearing) → combined ==
    single + 10·log10(2) = +3.0103 dB
  - both idle → −inf for every radar
"""
from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite.physics import SPEED_OF_LIGHT, DebugPhysicsConfig
from env.gpu.array_face_s4.array_factor import UPAConfig
from env.gpu.array_face_s6.array_factor import compute_upa_af_db_toward
from env.gpu.array_face_s6.physics import compute_snr_eff_db_s6, target_gain_db


def compute_jnr_db_s7(
    physics: DebugPhysicsConfig,
    radar: UPAConfig,
    jammer: UPAConfig,
    *,
    jammer_active: torch.Tensor,        # [E, K] bool
    radar_beam_az_idx: torch.Tensor,    # [E, R] int64
    radar_beam_el_idx: torch.Tensor,    # [E, R] int64
    jammer_beam_az_idx: torch.Tensor,   # [E, K] int64
    jammer_beam_el_idx: torch.Tensor,   # [E, K] int64
    cell_mask: torch.Tensor,            # [E, K, 25] float in {0, 1}
    pair_az_rad: torch.Tensor,          # [K, R] float (relative bearings)
    pair_el_rad: torch.Tensor,          # [K, R] float
    victim_service_id: torch.Tensor,    # [E, R] int64 (each radar's current svc)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (jnr [E, R], jnr_per [E, K, R]) combined / per-pair JNR in dB.

    jnr_per is -inf where jammer k is idle. jnr is -inf where NO jammer is
    active. Both jammers are service-agnostic broadband sources (spectral
    overlap 1.0, 0 dB) exactly as S6.
    """
    if jammer_active.dim() != 2:
        raise ValueError(f"jammer_active must be [E, K], got {tuple(jammer_active.shape)}")
    E, K = jammer_active.shape
    R = radar_beam_az_idx.shape[1]
    if cell_mask.shape != (E, K, cell_mask.shape[-1]):
        raise ValueError(f"cell_mask must be [E={E}, K={K}, N_CELLS], got {tuple(cell_mask.shape)}")
    if pair_az_rad.shape != (K, R) or pair_el_rad.shape != (K, R):
        raise ValueError(
            f"pair bearings must be [K={K}, R={R}], got "
            f"{tuple(pair_az_rad.shape)} / {tuple(pair_el_rad.shape)}")
    device = jammer_active.device

    services = (physics.service_0, physics.service_1)
    if any(s.fc_hz <= 0 for s in services):
        raise ValueError("S7 physics requires positive fc_hz for both services")
    fc_table = torch.tensor([s.fc_hz for s in services], device=device, dtype=torch.float32)
    bw_table = torch.tensor([s.bw_hz for s in services], device=device, dtype=torch.float32)
    rx_gain_table = torch.tensor([s.rx_gain_db for s in services], device=device, dtype=torch.float32)

    n_active = cell_mask.sum(dim=-1).clamp(min=1).to(torch.float32)   # [E, K]
    P_cell_dBm = 10.0 * torch.log10(torch.tensor(float(physics.P_jam_W) * 1000.0, device=device))
    P_peak_dBm = P_cell_dBm + 20.0 * torch.log10(n_active.clamp(min=1e-12))  # [E, K]

    jnr_per = torch.full((E, K, R), float("-inf"), device=device, dtype=torch.float32)
    for k in range(K):
        for r in range(R):
            vic_fc = fc_table.gather(0, victim_service_id[:, r].long())
            vic_bw = bw_table.gather(0, victim_service_id[:, r].long())
            vic_rx_gain = rx_gain_table.gather(0, victim_service_id[:, r].long())

            d = max(float(physics.distance_jm), float(physics.distance_floor_m))
            lambda_m = SPEED_OF_LIGHT / vic_fc
            L_path_db = 20.0 * torch.log10(4.0 * torch.pi * d / lambda_m)
            N_dBm = -174.0 + 10.0 * torch.log10(vic_bw) + float(physics.noise_figure_db)

            tgt_az = pair_az_rad[k, r].expand(E)
            tgt_el = pair_el_rad[k, r].expand(E)
            af_tx_db = compute_upa_af_db_toward(
                jammer, beam_az_idx=jammer_beam_az_idx[:, k], beam_el_idx=jammer_beam_el_idx[:, k],
                target_az_rad=tgt_az, target_el_rad=tgt_el,
            )
            af_rx_db = compute_upa_af_db_toward(
                radar, beam_az_idx=radar_beam_az_idx[:, r], beam_el_idx=radar_beam_el_idx[:, r],
                target_az_rad=tgt_az, target_el_rad=tgt_el,
            )

            J_dBm = (P_peak_dBm[:, k]
                     + float(physics.jam_antenna_gain_db) + af_tx_db
                     + vic_rx_gain + af_rx_db
                     - L_path_db - float(physics.polarization_loss_db))  # overlap = 0 dB (broadband)
            jnr_per[:, k, r] = J_dBm - N_dBm

    # per-pair JNR is -inf where jammer k is idle (its contribution is 0 mW)
    jnr_per = torch.where(
        jammer_active.unsqueeze(2), jnr_per,
        torch.full_like(jnr_per, float("-inf")))

    # idle jammers contribute exactly zero power (10^(-inf/10) = 0)
    jnr_lin = torch.where(
        jammer_active.unsqueeze(2) & torch.isfinite(jnr_per),
        10.0 ** (jnr_per / 10.0),
        torch.zeros_like(jnr_per))
    total_lin = jnr_lin.sum(dim=1)  # [E, R]
    any_active = jammer_active.any(dim=1)  # [E]
    jnr = 10.0 * torch.log10(total_lin.clamp(min=1e-30))
    jnr = torch.where(any_active.unsqueeze(1), jnr, torch.full_like(jnr, float("-inf")))
    return jnr.to(torch.float32), jnr_per.to(torch.float32)


def compute_p_detect_s7(
    physics: DebugPhysicsConfig,
    *,
    baseline_snr_db: float,
    jnr_db: torch.Tensor,             # [E, R]
) -> torch.Tensor:
    """Per-radar sigmoid P_detect on the combined JNR — same formula as S6."""
    from env.gpu.array_face_s6.physics import compute_p_detect_s6
    return compute_p_detect_s6(physics, baseline_snr_db=baseline_snr_db, jnr_db=jnr_db)


__all__ = [
    "compute_jnr_db_s7", "compute_p_detect_s7",
    "compute_snr_eff_db_s6", "target_gain_db",  # re-exported unchanged from S6
]
