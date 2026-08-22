"""S7 physics gates — two-jammer JNR combination + contestability under 2 jammers."""
from __future__ import annotations
import torch

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import (
    compute_jnr_db_s7, compute_snr_eff_db_s6, target_gain_db, UPAConfig,
)
from env.gpu.array_face_s7.geometry import pair_bearings

PHYS = default_debug_physics_config(P_jam_W=0.1)  # S6b-validated regime
U = UPAConfig()


def _jnr(active, *, jbeam_az, jbeam_el, cell, azp, elp, rb_az, rb_el, svc):
    E = active.shape[0]
    return compute_jnr_db_s7(
        PHYS, U, U,
        jammer_active=active,
        radar_beam_az_idx=rb_az, radar_beam_el_idx=rb_el,
        jammer_beam_az_idx=jbeam_az, jammer_beam_el_idx=jbeam_el,
        cell_mask=cell, pair_az_rad=azp, pair_el_rad=elp,
        victim_service_id=svc,
    )[0]


def test_s7_jnr_generalized_af_matches_s4_broadside():
    """S7 pair chain at broadside target == S4 broadside JNR (M0 consistency).

    Only jammer 0 active: the combined JNR must equal the S4 single-jammer
    reference exactly (the 'one idle → single-jammer chain' M0 gate).
    """
    from env.gpu.array_face_s4.physics import compute_jnr_db_s4
    E = 2
    azp = torch.zeros(2, 2); elp = torch.zeros(2, 2)  # everything broadside
    cell = torch.zeros(E, 2, 25); cell[:, :, 0] = 1.0
    ja = torch.zeros(E, 2, dtype=torch.int64); je = torch.full((E, 2), 2, dtype=torch.int64)
    ra = torch.zeros(E, 2, dtype=torch.int64); re_ = torch.full((E, 2), 2, dtype=torch.int64)
    svc = torch.zeros(E, 2, dtype=torch.int64)
    act = torch.tensor([[True, False], [True, False]])  # jammer 0 only
    j7, jp = compute_jnr_db_s7(PHYS, U, U, jammer_active=act,
                               radar_beam_az_idx=ra, radar_beam_el_idx=re_,
                               jammer_beam_az_idx=ja, jammer_beam_el_idx=je,
                               cell_mask=cell, pair_az_rad=azp, pair_el_rad=elp,
                               victim_service_id=svc)
    # S4 reference: single jammer, one radar at broadside, same steering
    j4 = compute_jnr_db_s4(
        PHYS, U, U,
        jammer_active=act[:, 0], victim_service_id=svc[:, 0],
        radar_beam_az_idx=ra[:, 0], radar_beam_el_idx=re_[:, 0],
        jammer_beam_az_idx=ja[:, 0], jammer_beam_el_idx=je[:, 0],
        cell_mask=cell[:, 0],
    )
    assert torch.allclose(j7, j4, atol=1e-3), f"single-active pair JNR must equal S4: {j7} vs {j4}"
    # the idle jammer's per-pair slice must be -inf
    assert torch.isinf(jp[:, 1]).all()


def test_s7_jnr_twins_plus_3dB():
    """Two identical active jammers (same bearing/steering) == single + 3.0103 dB."""
    E = 2
    az, el = pair_bearings("cpu")
    azp = az[0:1].expand(2, 2).contiguous()
    elp = el[0:1].expand(2, 2).contiguous()
    cell = torch.zeros(E, 2, 25); cell[:, :, 0] = 1.0
    ja = torch.full((E, 2), 3, dtype=torch.int64); je = torch.full((E, 2), 2, dtype=torch.int64)
    ra = torch.full((E, 2), 2, dtype=torch.int64); re_ = torch.full((E, 2), 2, dtype=torch.int64)
    svc = torch.zeros(E, 2, dtype=torch.int64)
    both = _jnr(torch.ones(E, 2, dtype=torch.bool), jbeam_az=ja, jbeam_el=je,
                cell=cell, azp=azp, elp=elp, rb_az=ra, rb_el=re_, svc=svc)
    one = _jnr(torch.tensor([[True, False], [True, False]]), jbeam_az=ja, jbeam_el=je,
               cell=cell, azp=azp, elp=elp, rb_az=ra, rb_el=re_, svc=svc)
    assert torch.allclose(both - one, torch.full_like(both, 3.0103), atol=1e-3)


def test_s7_jnr_idle_neg_inf():
    E = 2
    az, el = pair_bearings("cpu")
    cell = torch.zeros(E, 2, 25)
    ja = torch.zeros(E, 2, dtype=torch.int64); je = torch.full((E, 2), 2, dtype=torch.int64)
    ra = torch.zeros(E, 2, dtype=torch.int64); re_ = torch.full((E, 2), 2, dtype=torch.int64)
    svc = torch.zeros(E, 2, dtype=torch.int64)
    j, jp = compute_jnr_db_s7(PHYS, U, U,
                              jammer_active=torch.zeros(E, 2, dtype=torch.bool),
                              radar_beam_az_idx=ra, radar_beam_el_idx=re_,
                              jammer_beam_az_idx=ja, jammer_beam_el_idx=je,
                              cell_mask=cell, pair_az_rad=az, pair_el_rad=el,
                              victim_service_id=svc)
    assert torch.isinf(j).all() and torch.isinf(jp).all()


def test_s7_cross_beam_aiming_asymmetry():
    """A beam optimal toward one radar's bearing is suboptimal toward the other.

    With jammers at ±60° and radars at ±20°, no single (jammer, radar) pair
    bearing lands on the beam grid — single-beam suppression of BOTH radars is
    structurally impossible (the S7 hypothesis premise).
    """
    from env.gpu.array_face_s6.array_factor import compute_upa_af_db_toward
    az, el = pair_bearings("cpu")
    # best beam toward pair (0,0) (rel -40°): maximize Tx AF over the az grid
    best = None
    for b in range(5):
        tgt = az[0, 0].expand(1)
        af = compute_upa_af_db_toward(U, beam_az_idx=torch.tensor([b]),
                                      beam_el_idx=torch.tensor([2]),
                                      target_az_rad=tgt, target_el_rad=el[0, 0].expand(1))
        if best is None or af.item() > best[1]:
            best = (b, af.item())
    # same beam toward pair (1, 1) (rel +40°, the mirrored bearing) must differ
    # in general, and no beam covers BOTH ±40° pair bearings at once
    gains_00 = [compute_upa_af_db_toward(
        U, beam_az_idx=torch.tensor([b]), beam_el_idx=torch.tensor([2]),
        target_az_rad=az[0, 0].expand(1), target_el_rad=el[0, 0].expand(1)).item()
        for b in range(5)]
    gains_11 = [compute_upa_af_db_toward(
        U, beam_az_idx=torch.tensor([b]), beam_el_idx=torch.tensor([2]),
        target_az_rad=az[1, 1].expand(1), target_el_rad=el[1, 1].expand(1)).item()
        for b in range(5)]
    assert max(gains_00) < -0.5, f"pair (0,0) rel -40° must stay off-grid: {max(gains_00)}"
    assert max(gains_11) < -0.5, f"pair (1,1) rel +40° must stay off-grid: {max(gains_11)}"
    assert max(max(gains_00), max(gains_11)) > -20.0  # not unreachable either


def test_s7_p_detect_shape_and_monotone():
    from env.gpu.array_face_s7.physics import compute_p_detect_s7
    jnr = torch.tensor([[5.0, 12.0], [-3.0, 8.0]])
    p = compute_p_detect_s7(PHYS, baseline_snr_db=12.0, jnr_db=jnr)
    assert p.shape == (2, 2)
    assert (p > 0).all() and (p < 1).all()
    assert (p[:, 1] < p[:, 0]).all()  # monotone decreasing in JNR


def test_s7_contestability_under_two_jammers():
    """CONTESTABILITY gate (S7): under BOTH jammers at full leverage the
    radar's per-mission best-response profile must stay contested — the 2v2
    game must not degenerate into radar blindness before training.

    Full leverage: each jammer transmits 1 cell, best beams chosen
    independently (cross-assignment), radar best-responds per mission azimuth.
    """
    from env.gpu.array_face_s7.geometry import pair_bearings
    E, K, R = 1, 2, 2
    azp, elp = pair_bearings("cpu")
    thr = float(PHYS.detect_threshold_db)
    width = float(PHYS.detect_width_db)

    cell = torch.zeros(E, K, 25); cell[:, :, 0] = 1.0  # 1 cell each
    svc = torch.zeros(E, R, dtype=torch.int64)

    # per-jammer beam: enumerate the full 25-beam grid independently
    jam_beams = [(ba, be) for ba in range(5) for be in range(5)]
    p_best = []
    for m in range(5):  # mission azimuth
        best = 0.0
        for b0 in jam_beams:
            for b1 in jam_beams:
                jb_az = torch.tensor([[b0[0], b1[0]]], dtype=torch.int64)
                jb_el = torch.tensor([[b0[1], b1[1]]], dtype=torch.int64)
                for rb in range(5):
                    rb_az = torch.full((E, R), rb, dtype=torch.int64)
                    rb_el = torch.full((E, R), 2, dtype=torch.int64)
                    jnr, _ = compute_jnr_db_s7(
                        PHYS, U, U,
                        jammer_active=torch.ones(E, K, dtype=torch.bool),
                        radar_beam_az_idx=rb_az, radar_beam_el_idx=rb_el,
                        jammer_beam_az_idx=jb_az, jammer_beam_el_idx=jb_el,
                        cell_mask=cell, pair_az_rad=azp, pair_el_rad=elp,
                        victim_service_id=svc)
                    snr = compute_snr_eff_db_s6(PHYS, baseline_snr_db=12.0, jnr_db=jnr)
                    tg = target_gain_db(U, beam_az_idx=rb_az, beam_el_idx=rb_el,
                                        mission_az_idx=torch.full_like(rb_az, m))
                    p = float(torch.sigmoid((snr[0, 0] + tg[0, 0] - thr) / width).item())
                    best = max(best, p)
        p_best.append(best)

    contestable = sum(1 for p in p_best if 0.05 < p < 0.95)
    assert contestable >= 3, f"2-jammer game must stay contested, profile={p_best}"
    assert min(p_best) > 0.02, f"no azimuth may be unreachable, profile={p_best}"
    assert max(p_best) > 0.8, f"radar must keep a strong sector, profile={p_best}"
