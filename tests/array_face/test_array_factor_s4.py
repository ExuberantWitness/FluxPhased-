"""S4 2D UPA physics tests — M0 gate (HANDOFF §8.5).

Gate values (hand-computed link budget, P_cell = 2 W, svc_0 rx_gain 35 dB):
  - all 25 cells on + both beams broadside → JNR ≈ 81.47 dB
  - 5 cells on + both broadside → JNR ≈ 67.49 dB (== S3/S2 all-on gate)
  - AF continuity: el=0 plane reduces exactly to S2's 1D ULA AF
"""
import math
import pytest
import torch

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s2.array_factor import (
    JammerULAConfig, compute_jammer_af_db,
)
from env.gpu.array_face_s4.array_factor import (
    UPAConfig, N_AZ, N_EL, N_BEAM_DIRS_S4, N_CELLS_S4,
    compute_upa_af_db, compute_upa_af_db_flat, upa_af_table,
    beam_idx_to_az_el,
)
from env.gpu.array_face_s4.physics import compute_jnr_db_s4, compute_p_detect_s4


def _upa():
    return UPAConfig()


def _idx(t, device="cpu"):
    return torch.tensor(t, dtype=torch.int64, device=device)


def _all_cells_on(E, device="cpu"):
    return torch.ones((E, N_CELLS_S4), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Geometry / AF
# ---------------------------------------------------------------------------

def test_upa_config_defaults():
    upa = _upa()
    assert upa.n_cells == 25
    assert upa.n_cells_az == 5 and upa.n_cells_el == 5
    assert upa.n_beam_dirs == N_BEAM_DIRS_S4 == 25
    assert len(upa.beam_az_deg) == N_AZ == 5
    assert len(upa.beam_el_deg) == N_EL == 5


def test_beam_idx_to_az_el_raster():
    az, el = beam_idx_to_az_el(_idx([0, 4, 5, 24]))
    assert az.tolist() == [0, 4, 0, 4]
    assert el.tolist() == [0, 0, 1, 4]


def test_af_broadside_peak_zero_db():
    upa = _upa()
    # broadside = az index 2 (0°), el index 2 (0°) in the ± grids
    af = compute_upa_af_db(upa, beam_az_idx=_idx([2]), beam_el_idx=_idx([2]))
    assert af.shape == (1,)
    assert abs(float(af[0]) - 0.0) < 1e-4


def test_af_el0_matches_s2_ula():
    """Continuity gate: the el=0 plane of the 2D AF reduces to S2's 1D AF."""
    upa = _upa()
    s2_jammer = JammerULAConfig()
    az_idx = torch.arange(N_AZ)
    el0 = torch.full_like(az_idx, 2)  # el index 2 = 0°
    s4_af = compute_upa_af_db(upa, beam_az_idx=az_idx, beam_el_idx=el0)
    s2_af = compute_jammer_af_db(s2_jammer, jammer_beam_az_idx=az_idx)
    assert torch.allclose(s4_af, s2_af, atol=1e-4), f"{s4_af} vs {s2_af}"


def test_af_separability_in_uv():
    """AF_2d(az, el) = AF_az(u) + AF_el(v) in DIRECTION-COSINE space:
    u = sin(az)cos(el), v = sin(el). Note u couples az and el, so the
    factorization does NOT hold over (az, el) directly."""
    from env.gpu.array_face_s4.array_factor import _ula_af_sq_db
    upa = _upa()
    az = _idx([3])  # 30°
    el = _idx([4])  # 30°
    af_2d = compute_upa_af_db(upa, beam_az_idx=az, beam_el_idx=el)
    u = math.sin(math.radians(30.0)) * math.cos(math.radians(30.0))
    v = math.sin(math.radians(30.0))
    expected = _ula_af_sq_db(5, torch.tensor([u])) + _ula_af_sq_db(5, torch.tensor([v]))
    assert torch.allclose(af_2d, expected, atol=1e-4), f"{af_2d} vs {expected}"


def test_af_rolloff_known_values():
    """|AF|² of a 5-element ULA at s=sin(30°)=0.5 is 1/25 → -13.98 dB;
    at s=sin(60°)=0.866 → 0.0102 → -19.89 dB (all normalized by 25)."""
    upa = _upa()
    af = compute_upa_af_db(upa, beam_az_idx=_idx([1, 3]), beam_el_idx=_idx([2, 2]))
    expected = 10.0 * math.log10(1.0 / 25.0)  # -13.979
    assert abs(float(af[0]) - expected) < 1e-2
    assert abs(float(af[1]) - expected) < 1e-2
    af_el = compute_upa_af_db(upa, beam_az_idx=_idx([2]), beam_el_idx=_idx([4]))  # el=30°
    assert abs(float(af_el[0]) - expected) < 1e-2


def test_af_symmetry_and_table():
    upa = _upa()
    table = upa_af_table(upa, device="cpu")
    assert table.shape == (N_AZ, N_EL)
    assert abs(float(table[2, 2])) < 1e-4  # broadside peak
    # even symmetry
    assert torch.allclose(table, table.flip(0).flip(1), atol=1e-4)
    # flat lookup consistency
    flat = compute_upa_af_db_flat(upa, beam_idx=_idx([12, 0, 24]))
    assert abs(float(flat[0]) - float(table[2, 2])) < 1e-4
    assert abs(float(flat[1]) - float(table[0, 0])) < 1e-4
    assert abs(float(flat[2]) - float(table[4, 4])) < 1e-4


# ---------------------------------------------------------------------------
# JNR physics (M0 gate)
# ---------------------------------------------------------------------------

def _jnr_s4(cell_mask, jam_az=2, jam_el=2, rad_az=2, rad_el=2, svc=0):
    """JNR with both beams at broadside by default (az idx 2, el idx 2)."""
    physics = default_debug_physics_config(P_jam_W=2.0)
    upa = _upa()
    E = cell_mask.shape[0]
    return compute_jnr_db_s4(
        physics, upa, upa,
        jammer_active=cell_mask.sum(-1) > 0,
        victim_service_id=torch.full((E,), svc, dtype=torch.int64),
        radar_beam_az_idx=torch.full((E,), rad_az, dtype=torch.int64),
        radar_beam_el_idx=torch.full((E,), rad_el, dtype=torch.int64),
        jammer_beam_az_idx=torch.full((E,), jam_az, dtype=torch.int64),
        jammer_beam_el_idx=torch.full((E,), jam_el, dtype=torch.int64),
        cell_mask=cell_mask,
    )


def test_jnr_s4_all25_broadside_gate():
    """M0 gate: all 25 cells on, both beams broadside → ≈ 81.47 dB."""
    jnr = _jnr_s4(_all_cells_on(1))
    assert torch.isfinite(jnr).all()
    assert 80.0 < float(jnr[0]) < 83.0, f"JNR = {float(jnr[0]):.2f} dB"


def test_jnr_s4_5cells_matches_s3():
    """Continuity gate: 5 cells on → ≈ 67.49 dB (== S3/S2 all-cells-on)."""
    cell = torch.zeros(1, N_CELLS_S4)
    cell[:, 0:5] = 1.0
    jnr = _jnr_s4(cell)
    assert 65.0 < float(jnr[0]) < 70.0, f"JNR = {float(jnr[0]):.2f} dB"
    assert abs(float(jnr[0]) - 67.49) < 0.2


def test_jnr_s4_cell_scaling_monotone():
    """JNR strictly increases with active cell count (N² coherent gain)."""
    jnrs = []
    for k in (1, 5, 10, 25):
        cell = torch.zeros(1, N_CELLS_S4)
        cell[:, :k] = 1.0
        jnrs.append(float(_jnr_s4(cell)[0]))
    for a, b in zip(jnrs, jnrs[1:]):
        assert b > a
    # 20·log10(25/5) ≈ 14 dB between 5 and 25 cells
    assert abs((jnrs[3] - jnrs[1]) - 20.0 * math.log10(5.0)) < 0.1


def test_jnr_s4_beam_rolloff_matches_af():
    """JNR penalty for steering away equals the AF rolloff (az 30° → -13.98)."""
    all_on = _all_cells_on(1)
    broadside = float(_jnr_s4(all_on, jam_az=2, jam_el=2)[0])
    steered = float(_jnr_s4(all_on, jam_az=3, jam_el=2)[0])  # az=30°
    assert abs((broadside - steered) - 13.98) < 0.05
    # elevation rolloff same magnitude
    steered_el = float(_jnr_s4(all_on, jam_az=2, jam_el=4)[0])  # el=30°
    assert abs((broadside - steered_el) - 13.98) < 0.05


def test_jnr_s4_idle_neg_inf():
    cell = torch.zeros(1, N_CELLS_S4)
    jnr = _jnr_s4(cell)
    assert not torch.isfinite(jnr).any()
    assert float(jnr[0]) == float("-inf")


def test_jnr_s4_no_nan_all_beam_combos():
    """25×25 beam grid: finite JNR everywhere active (no NaN)."""
    cell = _all_cells_on(1)
    for az in range(N_AZ):
        for el in range(N_EL):
            jnr = _jnr_s4(cell, jam_az=az, jam_el=el)
            assert torch.isfinite(jnr).all(), f"NaN at az={az} el={el}"
            assert float(jnr[0]) <= 81.5 + 1e-3  # peak bound


def test_p_detect_s4_monotone():
    physics = default_debug_physics_config(P_jam_W=2.0)
    jnrs = torch.tensor([-10.0, 0.0, 20.0, 40.0, 60.0, 81.5])
    p = compute_p_detect_s4(physics, baseline_snr_db=22.0, jnr_db=jnrs)
    for a, b in zip(p.tolist(), p.tolist()[1:]):
        assert b <= a + 1e-6  # higher JNR → lower or equal p_detect
    assert p[-1] < 1e-3  # saturated jamming → detection ~0
