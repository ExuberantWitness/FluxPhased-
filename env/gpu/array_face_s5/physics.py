"""S5 link budget — two incoherent jammers, powers summed in LINEAR scale.

The S1-S4 chain computes one jammer's JNR in dB. Two independent jammers are
INCOHERENT sources: their received powers add in mW, not dB. The correct S5
combination (each J_k_dBm shares the same noise term N_dBm and path loss):

    JNR_5 = 10·log10( Σ_k 10^(JNR_k / 10) )        [since N_dBm cancels]

M0 gates:
  - jammer 2 fully idle  → JNR_5 == JNR_4 (single-jammer S4 equivalence)
  - two identical active → JNR_5 == JNR_4 + 10·log10(2) = +3.0103 dB
  - both idle            → -inf

Implementation: calls S4's compute_jnr_db_s4 once per jammer (the AF-injection
design paying off again — the per-jammer link budget body is unchanged), then
combines in linear. No service head in S5 either: both jammers always match
the radar's current service (spectral overlap 1.0).
"""
from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.array_face_s4.physics import compute_jnr_db_s4, compute_p_detect_s4
from env.gpu.array_face_s5.array_factor import UPAConfig


def compute_jnr_db_s5(
    physics: DebugPhysicsConfig,
    radar: UPAConfig,
    jammer: UPAConfig,
    *,
    jammer_active: torch.Tensor,        # [E, K] bool
    victim_service_id: torch.Tensor,    # [E] int64 (radar's current svc)
    radar_beam_az_idx: torch.Tensor,    # [E] int64 in 0..4
    radar_beam_el_idx: torch.Tensor,    # [E] int64 in 0..4
    jammer_beam_az_idx: torch.Tensor,   # [E, K] int64 in 0..4
    jammer_beam_el_idx: torch.Tensor,   # [E, K] int64 in 0..4
    cell_mask: torch.Tensor,            # [E, K, 25] float in {0., 1.}
) -> torch.Tensor:
    """Vectorized S5 JNR: linear-scale power sum over K=2 incoherent jammers.

    Returns [E] float32 combined JNR in dB. -inf where NO jammer is active.
    """
    if jammer_active.dim() != 2:
        raise ValueError(
            f"jammer_active must be [E, K] 2-D, got {tuple(jammer_active.shape)}")
    E, K = jammer_active.shape
    if jammer_beam_az_idx.shape != (E, K) or jammer_beam_el_idx.shape != (E, K):
        raise ValueError(
            f"jammer_beam_*_idx must be [E={E}, K={K}], got "
            f"{tuple(jammer_beam_az_idx.shape)} / {tuple(jammer_beam_el_idx.shape)}")
    if cell_mask.shape != (E, K, cell_mask.shape[-1]):
        raise ValueError(
            f"cell_mask must be [E={E}, K={K}, N_CELLS], got {tuple(cell_mask.shape)}")

    total_lin = torch.zeros(E, device=jammer_active.device, dtype=torch.float32)
    any_active = torch.zeros(E, dtype=torch.bool, device=jammer_active.device)
    for k in range(K):
        jnr_k = compute_jnr_db_s4(
            physics, radar, jammer,
            jammer_active=jammer_active[:, k],
            victim_service_id=victim_service_id,
            radar_beam_az_idx=radar_beam_az_idx,
            radar_beam_el_idx=radar_beam_el_idx,
            jammer_beam_az_idx=jammer_beam_az_idx[:, k],
            jammer_beam_el_idx=jammer_beam_el_idx[:, k],
            cell_mask=cell_mask[:, k],
        )
        # 10^(-inf/10) = 0: idle jammers contribute exactly zero power.
        total_lin = total_lin + 10.0 ** (jnr_k / 10.0)
        any_active = any_active | jammer_active[:, k]

    jnr = 10.0 * torch.log10(total_lin.clamp(min=1e-30))
    jnr = torch.where(any_active, jnr, torch.full_like(jnr, float("-inf")))
    return jnr.to(torch.float32)


def compute_p_detect_s5(
    physics: DebugPhysicsConfig,
    *,
    baseline_snr_db: float,
    jnr_db: torch.Tensor,
) -> torch.Tensor:
    """Sigmoid P_detect on the combined JNR — identical formula to S3/S4."""
    return compute_p_detect_s4(
        physics, baseline_snr_db=baseline_snr_db, jnr_db=jnr_db,
    )
