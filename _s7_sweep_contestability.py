"""S7 pre-training contestability sweep — validates the ±60°/±20° geometry.

Prints the per-mission best-response detection profile (radar best-responds
per mission azimuth) under:
  (a) ONE jammer at full leverage   (best beam over the 25-beam grid, 1 cell)
  (b) TWO jammers at full leverage  (each best beam independently, 1 cell each)

Gate (mirrors test_s7_contestability_under_two_jammers):
  - >=3/5 azimuths contestable (0.05 < p < 0.95) under BOTH jammers
  - min p > 0.02, max p > 0.8
  - cross-beam advantage exists: the 2-jammer profile is strictly lower than
    the 1-jammer profile (the second jammer buys real suppression at equal
    per-jammer leverage) — otherwise the 2v2 question is vacuous.

Usage: python _s7_sweep_contestability.py
"""
from __future__ import annotations
import sys
sys.path.insert(0, '.')
import torch

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import (
    compute_jnr_db_s7, compute_snr_eff_db_s6, target_gain_db, UPAConfig,
)
from env.gpu.array_face_s7.geometry import pair_bearings

PHYS = default_debug_physics_config(P_jam_W=0.1)
U = UPAConfig()
thr = float(PHYS.detect_threshold_db)
width = float(PHYS.detect_width_db)
E, K, R = 1, 2, 2
azp, elp = pair_bearings("cpu")
svc = torch.zeros(E, R, dtype=torch.int64)

BEAMS = [(ba, be) for ba in range(5) for be in range(5)]


def best_p_two_jammers(m: int) -> float:
    cell = torch.zeros(E, K, 25); cell[:, :, 0] = 1.0
    best = 0.0
    for b0 in BEAMS:
        for b1 in BEAMS:
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
    return best


def best_p_one_jammer(m: int) -> float:
    """Jammer 0 only (jammer 1 idle) — the S7-env 1v2 reference."""
    cell = torch.zeros(E, K, 25); cell[:, 0, 0] = 1.0
    best = 0.0
    for b0 in BEAMS:
        jb_az = torch.tensor([[b0[0], 0]], dtype=torch.int64)
        jb_el = torch.tensor([[b0[1], 0]], dtype=torch.int64)
        for rb in range(5):
            rb_az = torch.full((E, R), rb, dtype=torch.int64)
            rb_el = torch.full((E, R), 2, dtype=torch.int64)
            jnr, _ = compute_jnr_db_s7(
                PHYS, U, U,
                jammer_active=torch.tensor([[True, False]]),
                radar_beam_az_idx=rb_az, radar_beam_el_idx=rb_el,
                jammer_beam_az_idx=jb_az, jammer_beam_el_idx=jb_el,
                cell_mask=cell, pair_az_rad=azp, pair_el_rad=elp,
                victim_service_id=svc)
            snr = compute_snr_eff_db_s6(PHYS, baseline_snr_db=12.0, jnr_db=jnr)
            tg = target_gain_db(U, beam_az_idx=rb_az, beam_el_idx=rb_el,
                                mission_az_idx=torch.full_like(rb_az, m))
            p = float(torch.sigmoid((snr[0, 0] + tg[0, 0] - thr) / width).item())
            best = max(best, p)
    return best


if __name__ == "__main__":
    p1 = [best_p_one_jammer(m) for m in range(5)]
    p2 = [best_p_two_jammers(m) for m in range(5)]
    print("mission az       :", list(range(5)))
    print("1 jammer profile :", [round(p, 4) for p in p1])
    print("2 jammer profile :", [round(p, 4) for p in p2])
    print("cross-beam drop  :", [round(p1[i] - p2[i], 4) for i in range(5)])

    contestable2 = sum(1 for p in p2 if 0.05 < p < 0.95)
    gates = {
        "contestable_2jam (>=3)": contestable2 >= 3,
        "min_p_2jam (>0.02)": min(p2) > 0.02,
        "max_p_2jam (>0.8)": max(p2) > 0.8,
        "cross_beam_advantage (2jam < 1jam all az)": all(p2[i] < p1[i] for i in range(5)),
    }
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(gates.values())
    print("SWEEP", "PASS — geometry validated" if ok else "FAIL — geometry needs revision")
    sys.exit(0 if ok else 1)
