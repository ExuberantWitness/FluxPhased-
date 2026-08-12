"""S3 link budget — S2 physics + dynamic N_active (cell binding) + injectable AF.

Two changes from S2 (`env/gpu/array_face_s2/physics.py`):

  (1) Coherent array gain scales with the ACTIVE cell count, not the fixed
      total. S2 hard-codes `20*log10(N_cells=5)`; S3 uses
      `20*log10(N_active)` where `N_active = sum(cell_mask)`. When all cells
      are on, N_active == N_cells and S3 reproduces S2 exactly (this is the
      M0 physical gate: all-cells-on JNR ≈ 67.48 dB).

  (2) The array-factor (AF) computation is injectable via `af_rx_fn` /
      `af_tx_fn` callables. S3 defaults to S2's 1D ULA AF; S4 (2D 5x5 UPA)
      injects a 2D AF function without rewriting this link budget. The JNR
      formula (P_cell + 20log10(N_active) + EIRP chain - noise) is identical
      for S3 and S4; only the AF geometry differs.

JNR_db = P_cell_dBm + 20log10(N_active)         # dynamic coherent gain
         + jam_antenna_gain_db + AF_tx_db        # injectable
         + vic_rx_gain_db + AF_rx_db             # injectable
         - L_path_db - L_pol_db + overlap_db - N_dBm

Per-cell power semantics (S2 post-fix): physics.P_jam_W is per-cell power
(P_cell_W). S3 EnvConfig defaults P_jam_W=2.0 (correct), so all-cells-on
gives the same EIRP as S2-fixed.
"""
from __future__ import annotations
from typing import Callable, Optional

import torch

from env.gpu.g3_bsta_lite.physics import (
    SPEED_OF_LIGHT, DebugPhysicsConfig,
)
from env.gpu.array_face_s3.array_factor import (
    RadarULAConfig, JammerULAConfig,
    compute_radar_af_db, compute_jammer_af_db,
)


def compute_jnr_db_s3(
    physics: DebugPhysicsConfig,
    radar: RadarULAConfig,
    jammer: JammerULAConfig,
    *,
    jammer_active: torch.Tensor,
    jammer_service_id: torch.Tensor,
    victim_service_id: torch.Tensor,
    radar_beam_az_idx: torch.Tensor,
    jammer_beam_az_idx: torch.Tensor,
    cell_mask: torch.Tensor,
    af_rx_fn: Optional[Callable[[], torch.Tensor]] = None,
    af_tx_fn: Optional[Callable[[], torch.Tensor]] = None,
) -> torch.Tensor:
    """Vectorized S3 JNR with dynamic N_active and injectable AF.

    Args:
        physics: lite DebugPhysicsConfig (services, P_jam_W, geometry, etc.)
        radar: RadarULAConfig (S1/S2 radar ULA, unchanged)
        jammer: JammerULAConfig (S2 jammer ULA; S4 will pass a 2D UPA config)
        jammer_active: [E] bool
        jammer_service_id: [E] int64 in {0, 1}
        victim_service_id: [E] int64 in {0, 1}
        radar_beam_az_idx: [E] int64 in 0..radar.n_beam_dirs-1
        jammer_beam_az_idx: [E] int64 in 0..jammer.n_beam_dirs-1
        cell_mask: [E, N_CELLS] float in {0., 1.}. Determines N_active (the
            number of cells participating in coherent combining).
        af_rx_fn: callable returning [E] AF_rx in dB. If None, uses S2's 1D
            compute_radar_af_db(radar, radar_beam_az_idx=...).
        af_tx_fn: callable returning [E] AF_tx in dB. If None, uses S2's 1D
            compute_jammer_af_db(jammer, jammer_beam_az_idx=...).
            (S4 injects 2D AF here.)

    Returns:
        [E] float32 JNR in dB. -inf where not jammer_active.
        When cell_mask is all ones, reproduces S2-fixed JNR exactly.
    """
    if jammer_active.dim() != 1:
        raise ValueError(f"jammer_active must be [E] 1-D, got {tuple(jammer_active.shape)}")
    if cell_mask.shape[0] != jammer_active.shape[0]:
        raise ValueError(
            f"cell_mask batch {cell_mask.shape[0]} != jammer_active {jammer_active.shape[0]}"
        )
    E = jammer_active.shape[0]
    device = jammer_active.device

    services = (physics.service_0, physics.service_1)
    if any(s.fc_hz <= 0 for s in services):
        raise ValueError(f"S3 physics requires positive fc_hz for both services")

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

    # --- S3 change (1): dynamic N_active coherent gain ---
    # S2 uses fixed N = jammer.n_cells; S3 uses N_active = sum(active cells).
    # clamp(min=1) guards against all-zero cell masks (env-side also clamps,
    # so when jammer_active=True there is always >=1 active cell).
    n_active = cell_mask.sum(dim=-1).clamp(min=1).to(torch.float32)  # [E]
    P_cell_dBm = 10.0 * torch.log10(torch.tensor(float(physics.P_jam_W) * 1000.0, device=device))
    # Coherent peak EIRP: P_cell + 20*log10(N_active) (N_active^2 in power domain).
    # When n_active == N_cells, this equals S2's fixed formula exactly.
    P_peak_dBm = P_cell_dBm + 20.0 * torch.log10(n_active.clamp(min=1e-12))

    # Noise
    N_dBm = -174.0 + 10.0 * torch.log10(vic_bw) + float(physics.noise_figure_db)

    # --- S3 change (2): injectable AF (defaults to S2's 1D ULA AF) ---
    if af_rx_fn is None:
        af_rx_fn = lambda: compute_radar_af_db(radar, radar_beam_az_idx=radar_beam_az_idx)
    if af_tx_fn is None:
        af_tx_fn = lambda: compute_jammer_af_db(jammer, jammer_beam_az_idx=jammer_beam_az_idx)
    af_rx_db = af_rx_fn()
    af_tx_db = af_tx_fn()

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


def compute_p_detect_s3(
    physics: DebugPhysicsConfig,
    *,
    baseline_snr_db: float,
    jnr_db: torch.Tensor,
) -> torch.Tensor:
    """Sigmoid P_detect (identical formula to S1/S2/lite, kept for module symmetry)."""
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
