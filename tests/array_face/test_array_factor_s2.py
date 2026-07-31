"""S2 array factor unit tests.

Tests jammer ULA AF (Tx), radar ULA AF (Rx, wrapper around S1), and S2 JNR link
budget that combines both.
"""
from __future__ import annotations
import pytest
import torch

from env.gpu.array_face_s2.array_factor import (
    RadarULAConfig, JammerULAConfig,
    BEAM_AZ_DEG_S2, N_BEAM_DIRS_S2,
    compute_jammer_af_db, compute_jammer_af_db_all,
    compute_radar_af_db, compute_radar_af_db_all,
)
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s2.physics import compute_jnr_db_s2


def test_beam_grid_size():
    assert N_BEAM_DIRS_S2 == 5
    assert len(BEAM_AZ_DEG_S2) == 5
    assert BEAM_AZ_DEG_S2 == (-60.0, -30.0, 0.0, 30.0, 60.0)


def test_jammer_af_peak_at_broadside():
    """Peak (0 dB) at beam_az_idx=2 (0° broadside)."""
    cfg = JammerULAConfig()
    all_af = compute_jammer_af_db_all(cfg, device='cpu')
    assert all_af[2].item() == pytest.approx(0.0, abs=1e-6)
    assert all_af.shape == (5,)


def test_jammer_af_symmetric():
    """AF symmetric around broadside: idx=0 (−60°) == idx=4 (+60°), idx=1 == idx=3."""
    cfg = JammerULAConfig()
    all_af = compute_jammer_af_db_all(cfg, device='cpu')
    assert all_af[0].item() == pytest.approx(all_af[4].item(), abs=1e-6)
    assert all_af[1].item() == pytest.approx(all_af[3].item(), abs=1e-6)


def test_jammer_af_below_zero_db():
    """All non-peak AFs < 0 dB (peak-normalized)."""
    cfg = JammerULAConfig()
    all_af = compute_jammer_af_db_all(cfg, device='cpu')
    for i, v in enumerate(all_af.tolist()):
        if i == 2:  # peak
            continue
        assert v < 0.0, f"beam_az_idx={i} AF={v} should be < 0 dB"


def test_jammer_af_batch_consistency():
    """[E] batch eval matches scalar eval gathered."""
    cfg = JammerULAConfig()
    idx = torch.tensor([0, 1, 2, 3, 4, 0, 4, 2], dtype=torch.int64)
    batch = compute_jammer_af_db(cfg, jammer_beam_az_idx=idx)
    all_af = compute_jammer_af_db_all(cfg, device='cpu')
    expected = all_af.gather(0, idx)
    assert torch.allclose(batch, expected, atol=1e-6)


def test_jammer_af_equals_radar_af_for_same_config():
    """Same formula, same config → same values (both are ULA, λ/2, peak-normalized)."""
    j = compute_jammer_af_db_all(JammerULAConfig(), device='cpu')
    r = compute_radar_af_db_all(RadarULAConfig(), device='cpu')
    assert torch.allclose(j, r, atol=1e-6)


def test_jnr_s2_peaks_when_both_afs_at_broadside():
    """JNR highest when both jammer and radar aim at broadside (idx=2)."""
    physics = default_debug_physics_config(P_jam_W=10.0)
    radar = RadarULAConfig()
    jammer = JammerULAConfig()
    E = 5
    is_jam = torch.ones(E, dtype=torch.bool)
    jam_svc = torch.zeros(E, dtype=torch.int64)
    vic_svc = torch.zeros(E, dtype=torch.int64)
    radar_az = torch.full((E,), 2, dtype=torch.int64)  # radar at broadside
    jammer_az = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    jnr = compute_jnr_db_s2(
        physics, radar, jammer,
        jammer_active=is_jam, jammer_service_id=jam_svc,
        victim_service_id=vic_svc,
        radar_beam_az_idx=radar_az, jammer_beam_az_idx=jammer_az,
    )
    # Peak at idx=2 (both main lobes aligned)
    assert jnr[2].item() == max(jnr.tolist())
    # Symmetric
    assert jnr[0].item() == pytest.approx(jnr[4].item(), abs=1e-3)
    assert jnr[1].item() == pytest.approx(jnr[3].item(), abs=1e-3)
    # Spread > 15 dB between peak and sidelobe (radar_az fixed at broadside;
    # only jammer AF varies → single-AF spread ≈ 19.88 dB)
    spread = jnr[2].item() - jnr[0].item()
    assert spread > 15.0, f"AF should give > 15 dB JNR spread, got {spread}"


def test_jnr_s2_idle_returns_neg_inf():
    """Jammer idle → JNR = -inf."""
    physics = default_debug_physics_config(P_jam_W=10.0)
    radar = RadarULAConfig()
    jammer = JammerULAConfig()
    E = 4
    is_jam = torch.tensor([False, True, False, True], dtype=torch.bool)
    jam_svc = torch.zeros(E, dtype=torch.int64)
    vic_svc = torch.zeros(E, dtype=torch.int64)
    radar_az = torch.full((E,), 2, dtype=torch.int64)
    jammer_az = torch.full((E,), 2, dtype=torch.int64)
    jnr = compute_jnr_db_s2(
        physics, radar, jammer,
        jammer_active=is_jam, jammer_service_id=jam_svc,
        victim_service_id=vic_svc,
        radar_beam_az_idx=radar_az, jammer_beam_az_idx=jammer_az,
    )
    assert not torch.isfinite(jnr[0])
    assert torch.isfinite(jnr[1])
    assert not torch.isfinite(jnr[2])
    assert torch.isfinite(jnr[3])


def test_jnr_s2_cross_service_lower_than_same():
    """Cross-service jamming (jam_svc != vic_svc) gives lower JNR due to overlap=0."""
    physics = default_debug_physics_config(P_jam_W=10.0)
    radar = RadarULAConfig()
    jammer = JammerULAConfig()
    E = 2
    is_jam = torch.ones(E, dtype=torch.bool)
    radar_az = torch.full((E,), 2, dtype=torch.int64)
    jammer_az = torch.full((E,), 2, dtype=torch.int64)
    # Same service
    jnr_same = compute_jnr_db_s2(
        physics, radar, jammer,
        jammer_active=is_jam, jammer_service_id=torch.zeros(E, dtype=torch.int64),
        victim_service_id=torch.zeros(E, dtype=torch.int64),
        radar_beam_az_idx=radar_az, jammer_beam_az_idx=jammer_az,
    )
    # Cross service
    jnr_cross = compute_jnr_db_s2(
        physics, radar, jammer,
        jammer_active=is_jam, jammer_service_id=torch.zeros(E, dtype=torch.int64),
        victim_service_id=torch.ones(E, dtype=torch.int64),
        radar_beam_az_idx=radar_az, jammer_beam_az_idx=jammer_az,
    )
    # Same-service JNR should be much higher (overlap → 1 vs overlap → 0)
    assert jnr_same[0].item() > jnr_cross[0].item() + 30.0


def test_no_nan_at_extreme_beam():
    """No NaN at extreme beam_az (idx=0 or idx=4 → sin(theta)=±0.866)."""
    cfg = JammerULAConfig()
    idx = torch.tensor([0, 4], dtype=torch.int64)
    af = compute_jammer_af_db(cfg, jammer_beam_az_idx=idx)
    assert torch.isfinite(af).all()
