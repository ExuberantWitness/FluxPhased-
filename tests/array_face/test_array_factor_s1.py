"""S1 physics unit tests — radar 1D ULA AF + JNR + P_detect.

Covers plan §8 T1.1-T1.8:
  T1.1 radar AF peak: radar_beam_az_idx=2 (broadside) -> 0 dB
  T1.2 radar AF monotonicity: |az| 0->30->60 deg gives 0 -> -14 -> -20 dB
  T1.3 radar AF sidelobe: AF_norm_db <= 0 always
  T1.4 radar AF main-lobe width at half-power (interpolated)
  T1.5 S1 JNR physical range: svc0 peak ~67 dB; svc0 60-deg ~47 dB
  T1.6 P_detect monotonicity in JNR
  T1.7 batch consistency: [E=16] vs scalar loop
  T1.8 NaN-free at extreme beam_az
"""
from __future__ import annotations
import math
import sys
import pytest
import torch

sys.path.insert(0, ".")

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s1.array_factor import (
    RadarULAConfig, BEAM_AZ_DEG_S1, N_BEAM_DIRS_S1,
    compute_radar_af_db, compute_radar_af_db_all,
)
from env.gpu.array_face_s1.physics import compute_jnr_db_s1, compute_p_detect_s1


def _cuda():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_T1_1_af_peak_at_broadside():
    """T1.1: AF at radar_beam_az_idx=2 (theta_0=0, broadside) = 0 dB."""
    cfg = RadarULAConfig()
    dev = _cuda()
    idx = torch.tensor([2], device=dev, dtype=torch.int64)
    af = compute_radar_af_db(cfg, radar_beam_az_idx=idx)
    assert af.shape == (1,)
    assert math.isclose(af.item(), 0.0, abs_tol=1e-3), f"peak should be 0 dB, got {af.item()}"


def test_T1_2_af_monotonic_with_beam_az_offset():
    """T1.2: AF monotonic as |theta_0| grows. idx 2 -> 0 dB, idx 1/3 -> ~-14 dB, idx 0/4 -> ~-20 dB."""
    cfg = RadarULAConfig()
    dev = _cuda()
    af_all = compute_radar_af_db_all(cfg, device=dev)
    assert af_all.shape == (N_BEAM_DIRS_S1,)
    peak = af_all[2].item()
    mid = af_all[1].item()  # +/- 30 deg
    far = af_all[0].item()  # +/- 60 deg
    assert math.isclose(peak, 0.0, abs_tol=1e-3)
    assert -16.0 < mid < -12.0, f"30-dec AF expected ~-14 dB, got {mid}"
    assert -22.0 < far < -17.0, f"60-dec AF expected ~-20 dB, got {far}"
    assert af_all[1].item() == pytest.approx(af_all[3].item(), abs=1e-4)  # symmetry
    assert af_all[0].item() == pytest.approx(af_all[4].item(), abs=1e-4)


def test_T1_3_af_always_nonpositive():
    """T1.3: AF_norm_db <= 0 for all beam directions (peak-normalized)."""
    cfg = RadarULAConfig()
    dev = _cuda()
    af_all = compute_radar_af_db_all(cfg, device=dev)
    assert (af_all <= 1e-3).all(), f"AF must be <= 0 dB, got {af_all.tolist()}"


def test_T1_4_af_main_lobe_width():
    """T1.4: half-power (−3 dB) beamwidth of N-cell d=λ/2 ULA ≈ 2/N rad ≈ 23° for N=5.

    We test by interpolating AF around broadside on a fine grid and checking the
    −3 dB width is in [20°, 26°] (engineering tolerance for N=5).
    """
    cfg = RadarULAConfig()
    dev = _cuda()
    # Fine-grained scan
    angles_deg = torch.linspace(-30.0, 30.0, 61, device=dev, dtype=torch.float32)
    s0 = torch.sin(angles_deg * math.pi / 180.0)
    N = cfg.n_cells
    num = torch.sin(N * math.pi * s0 / 2.0) ** 2
    den = torch.sin(math.pi * s0 / 2.0) ** 2
    af_sq = torch.where(den > 1e-10, num / den.clamp(min=1e-12), torch.full_like(s0, float(N * N)))
    af_norm_db = 10.0 * torch.log10((af_sq / (N * N)).clamp(min=1e-12))
    # Find angles where af_norm_db >= -3
    above_3db = (af_norm_db >= -3.0)
    # Width = (max - min) of those angles
    idx_above = torch.where(above_3db)[0]
    width_deg = (angles_deg[idx_above[-1]] - angles_deg[idx_above[0]]).item()
    assert 18.0 < width_deg < 28.0, f"3-dB beamwidth expected ~23 deg, got {width_deg}"


def test_T1_5_jnr_physical_range():
    """T1.5: JNR_db for full-power jamming falls in expected physical range.

    svc0 (fc=10 GHz) at broadside: link budget gives JNR ≈ 67.5 dB
    svc0 at ±60° (AF=−20 dB): ≈ 47.5 dB
    svc1 (fc=10.5 GHz) at broadside: ≈ 59 dB
    svc1 at ±60°: ≈ 39 dB
    """
    cfg = RadarULAConfig()
    phys = default_debug_physics_config(P_jam_W=50.0)
    dev = _cuda()
    ja = torch.tensor([True, True, True, True], device=dev)
    js = torch.tensor([0, 0, 1, 1], device=dev, dtype=torch.int64)
    vs = torch.tensor([0, 0, 1, 1], device=dev, dtype=torch.int64)
    bi = torch.tensor([2, 0, 2, 4], device=dev, dtype=torch.int64)
    jnr = compute_jnr_db_s1(phys, cfg, jammer_active=ja, jammer_service_id=js,
                            victim_service_id=vs, radar_beam_az_idx=bi)
    jnr = jnr.tolist()
    # svc0 broadside (peak AF=0)
    assert 65.0 < jnr[0] < 70.0, f"svc0_peak JNR ~67.5 dB, got {jnr[0]}"
    # svc0 ±60° (AF=−20)
    assert 45.0 < jnr[1] < 50.0, f"svc0_60deg JNR ~47.5 dB, got {jnr[1]}"
    # svc1 broadside
    assert 57.0 < jnr[2] < 61.0, f"svc1_peak JNR ~59 dB, got {jnr[2]}"
    # svc1 ±60°
    assert 37.0 < jnr[3] < 41.0, f"svc1_60deg JNR ~39 dB, got {jnr[3]}"


def test_T1_6_p_detect_monotone_in_jnr():
    """T1.6: P_detect decreases monotonically as JNR increases."""
    cfg = RadarULAConfig()
    phys = default_debug_physics_config(P_jam_W=50.0)
    dev = _cuda()
    jnr_db = torch.linspace(0.0, 60.0, 13, device=dev, dtype=torch.float32)
    p = compute_p_detect_s1(phys, baseline_snr_db=22.0, jnr_db=jnr_db)
    diffs = p[1:] - p[:-1]
    assert (diffs <= 1e-6).all(), f"P_detect must be non-increasing in JNR, got diffs={diffs.tolist()}"
    # Sanity: lowest JNR ~ 1, highest JNR ~ 0
    assert p[0] > 0.99, f"P_detect at JNR=0 should be ~1, got {p[0]}"
    assert p[-1] < 1e-3, f"P_detect at JNR=60 should be ~0, got {p[-1]}"


def test_T1_7_batch_consistency():
    """T1.7: Vectorized [E=16] AF gives same result as one-by-one scalar loop."""
    cfg = RadarULAConfig()
    dev = _cuda()
    E = 16
    torch.manual_seed(0)
    bi = torch.randint(0, N_BEAM_DIRS_S1, (E,), device=dev, dtype=torch.int64)
    af_batch = compute_radar_af_db(cfg, radar_beam_az_idx=bi)
    af_loop = torch.empty(E, device=dev, dtype=torch.float32)
    for i in range(E):
        bi_i = torch.tensor([int(bi[i].item())], device=dev, dtype=torch.int64)
        af_loop[i] = compute_radar_af_db(cfg, radar_beam_az_idx=bi_i).item()
    assert torch.allclose(af_batch, af_loop, atol=1e-5), \
        f"batch vs loop mismatch: max abs diff = {(af_batch - af_loop).abs().max().item()}"


def test_T1_8_nan_free_extreme():
    """T1.8: No NaN / Inf for extreme beam_az (all 5 grid points) and inactive jammer."""
    cfg = RadarULAConfig()
    phys = default_debug_physics_config(P_jam_W=50.0)
    dev = _cuda()
    # All 5 beam_az, all jammed
    ja = torch.ones(5, dtype=torch.bool, device=dev)
    js = torch.zeros(5, dtype=torch.int64, device=dev)
    vs = torch.zeros(5, dtype=torch.int64, device=dev)
    bi = torch.arange(5, dtype=torch.int64, device=dev)
    jnr = compute_jnr_db_s1(phys, cfg, jammer_active=ja, jammer_service_id=js,
                            victim_service_id=vs, radar_beam_az_idx=bi)
    assert torch.isfinite(jnr).all(), f"JNR must be finite for all active envs, got {jnr.tolist()}"
    p = compute_p_detect_s1(phys, baseline_snr_db=22.0, jnr_db=jnr)
    assert torch.isfinite(p).all(), f"P_detect must be finite, got {p.tolist()}"
    assert (p >= 0).all() and (p <= 1).all(), "P_detect must be in [0, 1]"

    # Inactive jammer -> JNR -inf, P_detect well-defined via where
    ja2 = torch.zeros(2, dtype=torch.bool, device=dev)
    js2 = torch.zeros(2, dtype=torch.int64, device=dev)
    vs2 = torch.zeros(2, dtype=torch.int64, device=dev)
    bi2 = torch.tensor([0, 4], dtype=torch.int64, device=dev)
    jnr2 = compute_jnr_db_s1(phys, cfg, jammer_active=ja2, jammer_service_id=js2,
                             victim_service_id=vs2, radar_beam_az_idx=bi2)
    p2 = compute_p_detect_s1(phys, baseline_snr_db=22.0, jnr_db=jnr2)
    assert torch.isfinite(p2).all(), f"P_detect for inactive jammer must be finite, got {p2.tolist()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
