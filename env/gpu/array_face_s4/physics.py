"""S4 link budget — S3 physics with 2D UPA AF injected (AF-injectable design).

This is the payoff of S3's `af_rx_fn`/`af_tx_fn` injection hooks: the JNR
formula (P_cell + 20·log10(N_active) + EIRP chain - noise) is NOT rewritten.
S4 only supplies the 2D AF geometry via two callables, so the link budget
body is bit-identical to S3's.

Two semantic notes vs S3:
  (1) No service head in S4 — the jammer always jams the radar's current
      service (jammer_service_id = victim_service_id), so spectral overlap
      is always 1.0 (0 dB). The service-decision dimension is removed
      (HANDOFF §11.2: base head absorbed by cell binding; service choice
      is delegated to the ESM layer that reports the radar's active service).
  (2) N_active = Σ(cell_mask) over 25 UPA cells. All-25-cells-on at
      broadside ⇒ JNR ≈ 81.5 dB (vs S3's 67.5 dB with 5 cells); 5-cells-on
      reproduces S3 exactly (continuity gate in tests).

M0 physics gate:
  - all 25 cells on + both beams broadside (az=0, el=0) → JNR ≈ 81.5 dB
  - 5 cells on + both broadside → JNR ≈ 67.5 dB (== S3/S2 all-on)
"""
from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.array_face_s3.physics import compute_jnr_db_s3, compute_p_detect_s3
from env.gpu.array_face_s4.array_factor import UPAConfig, compute_upa_af_db


def compute_jnr_db_s4(
    physics: DebugPhysicsConfig,
    radar: UPAConfig,
    jammer: UPAConfig,
    *,
    jammer_active: torch.Tensor,
    victim_service_id: torch.Tensor,
    radar_beam_az_idx: torch.Tensor,
    radar_beam_el_idx: torch.Tensor,
    jammer_beam_az_idx: torch.Tensor,
    jammer_beam_el_idx: torch.Tensor,
    cell_mask: torch.Tensor,
) -> torch.Tensor:
    """Vectorized S4 JNR: S3 link budget + injected 2D UPA AFs.

    Args:
        physics: lite DebugPhysicsConfig (unchanged from S1-S3)
        radar: UPAConfig (radar 5×5 UPA)
        jammer: UPAConfig (jammer 5×5 UPA)
        jammer_active: [E] bool (cell_mask.sum(-1) > 0)
        victim_service_id: [E] int64 in {0,1} (radar's current svc; jammer
            always matches it — no service head in S4)
        radar_beam_az_idx:  [E] int64 in 0..4
        radar_beam_el_idx:  [E] int64 in 0..4
        jammer_beam_az_idx: [E] int64 in 0..4
        jammer_beam_el_idx: [E] int64 in 0..4
        cell_mask: [E, 25] float in {0., 1.} (UPA cells)

    Returns:
        [E] float32 JNR in dB. -inf where not jammer_active.
    """
    return compute_jnr_db_s3(
        physics, radar, jammer,
        jammer_active=jammer_active,
        jammer_service_id=victim_service_id,   # S4: always matched
        victim_service_id=victim_service_id,
        radar_beam_az_idx=radar_beam_az_idx,
        jammer_beam_az_idx=jammer_beam_az_idx,
        cell_mask=cell_mask,
        # --- AF injection (S3's hooks, S4's 2D geometry) ---
        af_rx_fn=lambda: compute_upa_af_db(
            radar, beam_az_idx=radar_beam_az_idx, beam_el_idx=radar_beam_el_idx),
        af_tx_fn=lambda: compute_upa_af_db(
            jammer, beam_az_idx=jammer_beam_az_idx, beam_el_idx=jammer_beam_el_idx),
    )


def compute_p_detect_s4(
    physics: DebugPhysicsConfig,
    *,
    baseline_snr_db: float,
    jnr_db: torch.Tensor,
) -> torch.Tensor:
    """Sigmoid P_detect — identical formula to S1/S2/S3."""
    return compute_p_detect_s3(
        physics, baseline_snr_db=baseline_snr_db, jnr_db=jnr_db,
    )
