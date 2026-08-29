"""Analytic stare-vs-hedge separation under the S7 link budget.

For each mission azimuth m and radar r, compute the per-step detection
probability when the radar beam is EXACTLY on the mission bearing (stare)
vs one azimuth step away (hedge), under the worst-case jammer pair (both
jammers choose the beam maximizing received JNR at that radar). Window
success is 1-(1-p)^tau_window with tau_window = 6. Also scans P_jam to
locate the critical jammer power at which exact pointing stops dominating.

Writes experiments/array_face_s7/learning_repair/stare_analysis.json.
"""
import sys, json
sys.path.insert(0, '.')
import torch
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import UPAConfig, N_RADARS
from env.gpu.array_face_s7.geometry import pair_bearings
from env.gpu.array_face_s7.physics import compute_jnr_db_s7
from env.gpu.array_face_s6.physics import compute_snr_eff_db_s6, target_gain_db

OUT = 'experiments/array_face_s7/learning_repair/stare_analysis.json'
THR, WIDTH, TAU_WINDOW = 15.0, 3.0, 6


def worst_jnr(physics, radar_beam_az_idx, n_cells_per_jammer=32):
    """JNR at radar 0 when both jammers pick their best beams against it."""
    az_pairs, el_pairs = pair_bearings('cpu')  # [K, R] default cross-fire
    E, K, R = 1, 2, N_RADARS
    rb = torch.full((E, R), radar_beam_az_idx, dtype=torch.int64)
    rbe = torch.full_like(rb, 2)  # horizon plane
    cell = torch.zeros(E, K, 25)
    # worst case: each jammer concentrates its whole per-jammer budget
    cell[:, :, :n_cells_per_jammer] = 1.0
    best = None
    for k in range(K):
        pass
    # search the joint worst case over both jammers' beams (25x25 grid is
    # separable: JNR is a sum of per-pair powers, so each jammer's best beam
    # is individually optimal)
    jbeam = torch.zeros(E, K, dtype=torch.int64)
    for k in range(K):
        best_p, best_b = -1.0, 0
        for b in range(25):
            jb = jbeam.clone(); jb[0, k] = b
            jnr, _ = compute_jnr_db_s7(
                physics, UPAConfig(), UPAConfig(),
                jammer_active=torch.ones(E, K, dtype=torch.bool),
                radar_beam_az_idx=rb, radar_beam_el_idx=rbe,
                jammer_beam_az_idx=jb % 5, jammer_beam_el_idx=jb // 5,
                cell_mask=cell, pair_az_rad=az_pairs, pair_el_rad=el_pairs,
                victim_service_id=torch.zeros(E, R, dtype=torch.int64))
            v = float(jnr[0, 0])
            if v > best_p:
                best_p, best_b = v, b
        jbeam[0, k] = best_b
    jnr, _ = compute_jnr_db_s7(
        physics, UPAConfig(), UPAConfig(),
        jammer_active=torch.ones(E, K, dtype=torch.bool),
        radar_beam_az_idx=rb, radar_beam_el_idx=rbe,
        jammer_beam_az_idx=jbeam % 5, jammer_beam_el_idx=jbeam // 5,
        cell_mask=cell, pair_az_rad=az_pairs, pair_el_rad=el_pairs,
        victim_service_id=torch.zeros(E, R, dtype=torch.int64))
    return float(jnr[0, 0])


def p_detect(physics, snr0, jnr_db, beam_az, mission_az):
    E, R = 1, N_RADARS
    rb = torch.full((E, R), beam_az, dtype=torch.int64)
    snr = compute_snr_eff_db_s6(physics, baseline_snr_db=snr0,
                                jnr_db=torch.tensor([[jnr_db]]))
    tg = target_gain_db(UPAConfig(), beam_az_idx=rb,
                        beam_el_idx=torch.full_like(rb, 2),
                        mission_az_idx=torch.full((E, R), mission_az))
    return float(torch.sigmoid((snr[0, 0] + tg[0, 0] - THR) / WIDTH))


def window_success(p, tau=TAU_WINDOW):
    return 1.0 - (1.0 - p) ** tau


out = {'per_azimuth': [], 'critical_scan': [], 'concentration_scan': []}
P_JAM = 0.1
physics = default_debug_physics_config(P_jam_W=P_JAM)
# deployed per-step activation: trained jammers use ~1 cell per step
for m in range(5):
    jnr_exact = worst_jnr(physics, radar_beam_az_idx=m, n_cells_per_jammer=1)
    jnr_hedge = worst_jnr(physics, radar_beam_az_idx=(m + 1) % 5, n_cells_per_jammer=1)
    p_ex = p_detect(physics, 12.0, jnr_exact, m, m)
    p_hd = p_detect(physics, 12.0, jnr_hedge, (m + 1) % 5, m)
    out['per_azimuth'].append({
        'mission_az': m, 'jnr_stare_db': round(jnr_exact, 2),
        'p_stare': round(p_ex, 4), 'p_hedge': round(p_hd, 4),
        'win_stare': round(window_success(p_ex), 4),
        'win_hedge': round(window_success(p_hd), 4)})
    print(f"az={m}: jnr={jnr_exact:6.2f}dB  p_stare={p_ex:.4f} "
          f"p_hedge={p_hd:.4f}  win_stare={window_success(p_ex):.4f} "
          f"win_hedge={window_success(p_hd):.4f}")

# budget-concentration scan: a jammer may legally stack cells in ONE step;
# how much concentration does exact pointing survive at the fixed 0.1 W/cell?
for n_cells in (1, 2, 4, 8, 16, 32):
    min_win, min_p = 1.0, 1.0
    for m in range(5):
        p_ex = p_detect(physics, 12.0,
                        worst_jnr(physics, m, n_cells_per_jammer=n_cells), m, m)
        min_p = min(min_p, p_ex)
        min_win = min(min_win, window_success(p_ex))
    out['concentration_scan'].append({
        'cells_per_step': n_cells, 'min_p_stare': round(min_p, 4),
        'min_win_stare': round(min_win, 4)})
    print(f"cells/step={n_cells:2d}  min p_stare={min_p:.4f}  "
          f"min window success={min_win:.4f}")

# critical P_jam scan at the deployed 1-cell activation
for P_jam in [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]:
    physics = default_debug_physics_config(P_jam_W=P_jam)
    min_win, still_dominates = 1.0, True
    for m in range(5):
        p_ex = p_detect(physics, 12.0, worst_jnr(physics, m, n_cells_per_jammer=1), m, m)
        p_hd = p_detect(physics, 12.0, worst_jnr(physics, (m + 1) % 5, n_cells_per_jammer=1),
                        (m + 1) % 5, m)
        min_win = min(min_win, window_success(p_ex))
        if p_ex <= p_hd:
            still_dominates = False
    out['critical_scan'].append({
        'P_jam_W': P_jam, 'min_win_stare': round(min_win, 4),
        'stare_dominates': still_dominates})
    print(f"P_jam={P_jam:5.2f}W  min window success (stare)={min_win:.4f}  "
          f"stare dominates everywhere: {still_dominates}")

with open(OUT, 'w') as f:
    json.dump(out, f, indent=1)
print(f'wrote {OUT}')
