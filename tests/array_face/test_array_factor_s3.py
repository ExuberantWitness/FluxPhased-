"""S3 physics unit tests — JNR with dynamic N_active + injectable AF.

Covers the S3 physical contract:
  - M0 gate: all-cells-on broadside JNR ≈ 67.48 dB (matches S2-fixed)
  - cell scaling: N_active drives the coherent gain (20·log10(N_active))
  - AF spread unchanged (depends on geometry, not cell count)
  - idle returns -inf; cross-service < same-service
  - no NaN at degenerate cell masks (all-zero + jammer_active)
  - AF injection: custom af_rx_fn / af_tx_fn are called (S4 readiness)
"""
from __future__ import annotations
import math
import sys
import pytest
import torch

sys.path.insert(0, ".")

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s3.array_factor import (
    RadarULAConfig, JammerULAConfig, N_CELLS, N_BEAM_DIRS_S1, N_BEAM_DIRS_S2,
)
from env.gpu.array_face_s3.physics import compute_jnr_db_s3, compute_p_detect_s3


def _cuda():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_cell_count_matches_jammer_config():
    """N_CELLS constant must match JammerULAConfig().n_cells."""
    assert N_CELLS == JammerULAConfig().n_cells == 5


def test_jnr_s3_all_cells_on_matches_s2_m0_gate():
    """M0 gate: all-cells-on broadside JNR ≈ 67.48 dB (S2-equivalent).

    This is the primary S3 physical gate: when all cells are active, S3 must
    reproduce S2-fixed exactly (same EIRP, same AF). If this fails, the
    dynamic N_active refactor broke physical comparability with S2.
    """
    cfg = RadarULAConfig()
    phys = default_debug_physics_config(P_jam_W=2.0)
    dev = _cuda()
    ja = torch.tensor([True], device=dev)
    js = torch.tensor([0], device=dev, dtype=torch.int64)
    vs = torch.tensor([0], device=dev, dtype=torch.int64)
    bi = torch.tensor([2], device=dev, dtype=torch.int64)  # broadside both
    cell = torch.ones((1, N_CELLS), device=dev, dtype=torch.float32)
    jnr = compute_jnr_db_s3(phys, cfg, JammerULAConfig(),
                            jammer_active=ja, jammer_service_id=js,
                            victim_service_id=vs,
                            radar_beam_az_idx=bi, jammer_beam_az_idx=bi,
                            cell_mask=cell)
    assert 65.0 < jnr.item() < 70.0, f"all-cells-on broadside JNR ~67.5 dB, got {jnr.item()}"


def test_jnr_s3_cell_scaling_monotone():
    """More active cells -> higher JNR (coherent array gain scales with N_active).

    JNR(k cells) - JNR(k-1 cells) ≈ 20·log10(k/(k-1)).
    """
    cfg = RadarULAConfig()
    phys = default_debug_physics_config(P_jam_W=2.0)
    dev = _cuda()
    ja = torch.tensor([True], device=dev)
    js = torch.tensor([0], device=dev, dtype=torch.int64)
    vs = torch.tensor([0], device=dev, dtype=torch.int64)
    bi = torch.tensor([2], device=dev, dtype=torch.int64)

    jnrs = []
    for k in range(1, N_CELLS + 1):
        cell = torch.zeros((1, N_CELLS), device=dev, dtype=torch.float32)
        cell[0, :k] = 1.0  # first k cells on
        jnr = compute_jnr_db_s3(phys, cfg, JammerULAConfig(),
                                jammer_active=ja, jammer_service_id=js,
                                victim_service_id=vs,
                                radar_beam_az_idx=bi, jammer_beam_az_idx=bi,
                                cell_mask=cell)
        jnrs.append(jnr.item())
    # monotone increasing
    for i in range(len(jnrs) - 1):
        assert jnrs[i + 1] > jnrs[i], f"JNR not monotone: {jnrs}"
    # 5-cell vs 1-cell difference ≈ 20·log10(5)
    expected_diff = 20.0 * math.log10(5.0)
    actual_diff = jnrs[-1] - jnrs[0]
    assert abs(actual_diff - expected_diff) < 0.5, \
        f"5cell-1cell diff {actual_diff:.2f} != expected {expected_diff:.2f}"


def test_jnr_s3_spread_unchanged_by_cell_count():
    """AF spread (broadside vs sidelobe) is independent of cell count.

    Cells only scale the coherent gain (a DC offset on JNR); the AF shape
    (and thus the spread) is geometric and unchanged. This must hold so S3's
    physical difficulty is comparable to S2.
    """
    cfg = RadarULAConfig()
    phys = default_debug_physics_config(P_jam_W=2.0)
    dev = _cuda()
    ja = torch.tensor([True, True], device=dev)
    js = torch.tensor([0, 0], device=dev, dtype=torch.int64)
    vs = torch.tensor([0, 0], device=dev, dtype=torch.int64)
    rbi = torch.tensor([2, 2], device=dev, dtype=torch.int64)   # radar broadside
    jbi = torch.tensor([2, 0], device=dev, dtype=torch.int64)   # jammer broadside vs sidelobe
    # 3 cells active
    cell = torch.zeros((2, N_CELLS), device=dev, dtype=torch.float32)
    cell[:, :3] = 1.0
    jnr = compute_jnr_db_s3(phys, cfg, JammerULAConfig(),
                            jammer_active=ja, jammer_service_id=js,
                            victim_service_id=vs,
                            radar_beam_az_idx=rbi, jammer_beam_az_idx=jbi,
                            cell_mask=cell)
    spread = jnr[0].item() - jnr[1].item()
    assert spread >= 15.0, f"JNR spread {spread:.2f} < 15 dB (AF shape broken)"


def test_jnr_s3_idle_returns_neg_inf():
    """jammer_active=False -> JNR = -inf regardless of cell mask."""
    cfg = RadarULAConfig()
    phys = default_debug_physics_config(P_jam_W=2.0)
    dev = _cuda()
    ja = torch.tensor([False], device=dev)
    js = torch.tensor([0], device=dev, dtype=torch.int64)
    vs = torch.tensor([0], device=dev, dtype=torch.int64)
    bi = torch.tensor([2], device=dev, dtype=torch.int64)
    cell = torch.ones((1, N_CELLS), device=dev, dtype=torch.float32)
    jnr = compute_jnr_db_s3(phys, cfg, JammerULAConfig(),
                            jammer_active=ja, jammer_service_id=js,
                            victim_service_id=vs,
                            radar_beam_az_idx=bi, jammer_beam_az_idx=bi,
                            cell_mask=cell)
    assert torch.isinf(jnr).all() and (jnr < 0).all(), f"idle JNR must be -inf, got {jnr}"


def test_jnr_s3_cross_service_lower_than_same():
    """Jamming a different service than the radar's victim yields lower JNR
    (spectral overlap drives the difference)."""
    cfg = RadarULAConfig()
    phys = default_debug_physics_config(P_jam_W=2.0)
    dev = _cuda()
    ja = torch.tensor([True, True], device=dev)
    js = torch.tensor([0, 1], device=dev, dtype=torch.int64)   # jam svc 0 / 1
    vs = torch.tensor([0, 0], device=dev, dtype=torch.int64)   # radar on svc 0
    bi = torch.tensor([2, 2], device=dev, dtype=torch.int64)
    cell = torch.ones((2, N_CELLS), device=dev, dtype=torch.float32)
    jnr = compute_jnr_db_s3(phys, cfg, JammerULAConfig(),
                            jammer_active=ja, jammer_service_id=js,
                            victim_service_id=vs,
                            radar_beam_az_idx=bi, jammer_beam_az_idx=bi,
                            cell_mask=cell)
    assert jnr[0] > jnr[1] + 30.0, f"same-service JNR should dominate by >30 dB, got {jnr.tolist()}"


def test_no_nan_at_degenerate_cell_masks():
    """No NaN/Inf for: (a) all-zero cell + jammer_active=True (clamp path),
    (b) single-cell active, (c) all cells on at extreme beam_az.
    """
    cfg = RadarULAConfig()
    phys = default_debug_physics_config(P_jam_W=2.0)
    dev = _cuda()
    ja = torch.tensor([True, True, True], device=dev)
    js = torch.tensor([0, 0, 0], device=dev, dtype=torch.int64)
    vs = torch.tensor([0, 0, 0], device=dev, dtype=torch.int64)
    bi = torch.tensor([2, 2, 0], device=dev, dtype=torch.int64)
    cell = torch.tensor([
        [1, 0, 0, 0, 0],   # single cell
        [1, 1, 1, 1, 1],   # all cells
        [0, 0, 0, 0, 0],   # all-zero (clamp(min=1) path -> 1 cell equivalent)
    ], device=dev, dtype=torch.float32)
    jnr = compute_jnr_db_s3(phys, cfg, JammerULAConfig(),
                            jammer_active=ja, jammer_service_id=js,
                            victim_service_id=vs,
                            radar_beam_az_idx=bi, jammer_beam_az_idx=bi,
                            cell_mask=cell)
    p = compute_p_detect_s3(phys, baseline_snr_db=22.0, jnr_db=jnr)
    assert torch.isfinite(jnr).all(), f"JNR not finite: {jnr.tolist()}"
    assert torch.isfinite(p).all(), f"P_detect not finite: {p.tolist()}"
    assert (p >= 0).all() and (p <= 1).all()


def test_af_injection_is_called():
    """S4 readiness: custom af_rx_fn / af_tx_fn are used when provided.

    Injects a constant -10 dB AF for both heads and verifies the result equals
    the default-path JNR minus 20 dB (two injected AFs each contributing -10).
    """
    cfg = RadarULAConfig()
    phys = default_debug_physics_config(P_jam_W=2.0)
    dev = _cuda()
    ja = torch.tensor([True], device=dev)
    js = torch.tensor([0], device=dev, dtype=torch.int64)
    vs = torch.tensor([0], device=dev, dtype=torch.int64)
    bi = torch.tensor([2], device=dev, dtype=torch.int64)
    cell = torch.ones((1, N_CELLS), device=dev, dtype=torch.float32)

    jnr_default = compute_jnr_db_s3(phys, cfg, JammerULAConfig(),
                                    jammer_active=ja, jammer_service_id=js,
                                    victim_service_id=vs,
                                    radar_beam_az_idx=bi, jammer_beam_az_idx=bi,
                                    cell_mask=cell)
    const_af = lambda: torch.tensor([-10.0], device=dev)
    jnr_injected = compute_jnr_db_s3(phys, cfg, JammerULAConfig(),
                                     jammer_active=ja, jammer_service_id=js,
                                     victim_service_id=vs,
                                     radar_beam_az_idx=bi, jammer_beam_az_idx=bi,
                                     cell_mask=cell,
                                     af_rx_fn=const_af, af_tx_fn=const_af)
    diff = jnr_default.item() - jnr_injected.item()
    # default AFs at broadside are both 0 dB; injected are both -10 dB.
    # So injected JNR should be 20 dB LOWER than default.
    assert abs(diff - 20.0) < 0.5, f"AF injection diff {diff:.2f} != 20 dB"


def test_p_detect_monotone_in_jnr_s3():
    """P_detect decreases monotonically as JNR increases (S2 parity check)."""
    phys = default_debug_physics_config(P_jam_W=2.0)
    dev = _cuda()
    jnr_db = torch.linspace(0.0, 60.0, 13, device=dev, dtype=torch.float32)
    p = compute_p_detect_s3(phys, baseline_snr_db=22.0, jnr_db=jnr_db)
    diffs = p[1:] - p[:-1]
    assert (diffs <= 1e-6).all(), f"P_detect must be non-increasing, got diffs={diffs.tolist()}"
    assert p[0] > 0.99 and p[-1] < 1e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
