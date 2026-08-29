"""Pre-training contestability profiles for the SNR sweep (9 / 12 / 15 dB).

Mirrors the oracle sweep in _s7_sweep_contestability.py but parameterizes the
baseline SNR, so each sweep regime is gated/logged before its training run.
Writes a JSON next to the chain log; the chain does not hard-fail on profile
shape (profiles are recorded, the 12-dB gate remains the validated regime).
"""
import sys, json
sys.path.insert(0, '.')
import torch
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import EnvConfig, UPAConfig, N_RADARS
from env.gpu.array_face_s7.physics import compute_jnr_db_s7
from env.gpu.array_face_s7.geometry import pair_bearings
from env.gpu.array_face_s6.physics import compute_snr_eff_db_s6, target_gain_db

OUT = 'experiments/array_face_s7/learning_repair/snr_contestability_profiles.json'
profiles = {}
for snr_db in (9.0, 12.0, 15.0):
    physics = default_debug_physics_config(P_jam_W=0.1)
    az_pairs, el_pairs = pair_bearings('cpu')   # [K, R] relative bearings
    E, R, K = 1, N_RADARS, 2
    cell = torch.zeros(E, K, 25)
    cell[:, :, 0] = 1.0                      # both jammers active, one cell
    jbeam = torch.tensor([[13, 13]], dtype=torch.int64)  # toward center first
    svc = torch.zeros(E, R, dtype=torch.int64)
    thr, width = float(physics.detect_threshold_db), float(physics.detect_width_db)
    p_best = []
    for m in range(5):
        best = 0.0
        for b in range(5):
            rb = torch.full((E, R), b, dtype=torch.int64)
            jnr, _ = compute_jnr_db_s7(
                physics, UPAConfig(), UPAConfig(),
                jammer_active=torch.ones(E, K, dtype=torch.bool),
                radar_beam_az_idx=rb, radar_beam_el_idx=torch.full_like(rb, 2),
                jammer_beam_az_idx=jbeam % 5, jammer_beam_el_idx=jbeam // 5,
                cell_mask=cell, pair_az_rad=az_pairs, pair_el_rad=el_pairs,
                victim_service_id=svc,
            )
            snr = compute_snr_eff_db_s6(physics, baseline_snr_db=snr_db, jnr_db=jnr)
            tg = target_gain_db(UPAConfig(), beam_az_idx=rb,
                                 beam_el_idx=torch.full_like(rb, 2),
                                 mission_az_idx=torch.full_like(rb, m))
            best = max(best, float(torch.sigmoid(
                (snr[0, 0] + tg[0, 0] - thr) / width).item()))
        p_best.append(best)
    profiles[f'snr_{snr_db:g}db'] = {
        'profile': [round(p, 4) for p in p_best],
        'n_contestable': sum(1 for p in p_best if 0.05 < p < 0.95),
    }
    print(f"snr={snr_db:g}dB profile={profiles[f'snr_{snr_db:g}db']['profile']} "
          f"contestable={profiles[f'snr_{snr_db:g}db']['n_contestable']}/5")

with open(OUT, 'w') as f:
    json.dump(profiles, f, indent=2)
print(f'wrote {OUT}')
