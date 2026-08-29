"""Pre-training contestability profiles for n = 2/3/4 jammer placements.

Replicates the pre-registered _s7_sweep_contestability.py semantics exactly:
per mission azimuth, the profile is max over radar azimuth beams of the
detection probability when EACH jamMER independently chooses the beam that
minimizes its own received power at the victim radar for that radar beam
(joint upper envelope; separable via the per-pair JNR output).

Sanity anchor: n=2 (+60,-60) must reproduce the published 12 dB cross-fire
profile [0.8367, 0.9115, 0.9885, 0.9928, 0.8552].

Gate (identical to the published gate): >=3/5 azimuths in (0.05, 0.95),
min p > 0.02, max p > 0.8.

Writes experiments/array_face_s7/learning_repair/n_scaling_profiles.json.
"""
import sys, json
sys.path.insert(0, '.')
import torch
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s7 import UPAConfig, N_RADARS
from env.gpu.array_face_s7.geometry import pair_bearings_for
from env.gpu.array_face_s7.physics import compute_jnr_db_s7
from env.gpu.array_face_s6.physics import compute_snr_eff_db_s6, target_gain_db

OUT = 'experiments/array_face_s7/learning_repair/n_scaling_profiles.json'
PHYS = default_debug_physics_config(P_jam_W=0.1)
U = UPAConfig()
THR = float(PHYS.detect_threshold_db)
WIDTH = float(PHYS.detect_width_db)
BEAMS = list(range(25))  # az + 5*el; el row irrelevant at 0-deg plane but kept


def pair_jnr_for(az_pairs, el_pairs, radar_beam_az, jammer_beams):
    """per-pair JNR [K] at radar 0 for the given radar az beam and per-j beams."""
    K = az_pairs.shape[0]
    E, R = 1, N_RADARS
    rb = torch.full((E, R), radar_beam_az, dtype=torch.int64)
    rbe = torch.full_like(rb, 2)
    cell = torch.zeros(E, K, 25); cell[:, :, 0] = 1.0
    jb = torch.tensor([jammer_beams], dtype=torch.int64)
    _, jnr_per = compute_jnr_db_s7(
        PHYS, U, U,
        jammer_active=torch.ones(E, K, dtype=torch.bool),
        radar_beam_az_idx=rb, radar_beam_el_idx=rbe,
        jammer_beam_az_idx=jb % 5, jammer_beam_el_idx=jb // 5,
        cell_mask=cell, pair_az_rad=az_pairs, pair_el_rad=el_pairs,
        victim_service_id=torch.zeros(E, R, dtype=torch.int64))
    return jnr_per[0, :, 0]  # [K] at radar 0


def profile(az_pairs, el_pairs, n):
    out = []
    for m in range(5):
        best = 0.0
        for rb in range(5):
            # each jammer independently picks its least-received-power beam
            # at this radar beam (max-p allocation, separable per pair)
            beams, total = [], 0.0
            for k in range(n):
                best_p, best_b = 1e9, 0
                for b in BEAMS:
                    v = float(pair_jnr_for(az_pairs, el_pairs, rb, [0] * k + [b] + [0] * (n - k - 1))[k])
                    if v < best_p:
                        best_p, best_b = v, b
                beams.append(best_b)
                total += 10.0 ** (best_p / 10.0)
            jnr = 10.0 * torch.log10(torch.tensor(total))
            snr = compute_snr_eff_db_s6(PHYS, baseline_snr_db=12.0, jnr_db=jnr.reshape(1, 1))
            tg = target_gain_db(U, beam_az_idx=torch.full((1, N_RADARS), rb),
                                beam_el_idx=torch.full((1, N_RADARS), 2),
                                mission_az_idx=torch.full((1, N_RADARS), m))
            p = float(torch.sigmoid((snr[0, 0] + tg[0, 0] - THR) / WIDTH))
            best = max(best, p)
        out.append(round(best, 4))
    return out


CONFIGS = {
    'n2': {'n': 2, 'az': (60.0, -60.0)},   # anchor: must match published
    'n3': {'n': 3, 'az': (60.0, 0.0, -60.0)},
    'n4': {'n': 4, 'az': (60.0, 20.0, -20.0, -60.0)},
}

result = {}
for name, c in CONFIGS.items():
    az_pairs, el_pairs = pair_bearings_for(c['az'], (20.0, -20.0), torch.device('cpu'))
    prof = profile(az_pairs, el_pairs, c['n'])
    gate = {
        'n_contestable': sum(1 for p in prof if 0.05 < p < 0.95),
        'min_p': min(prof), 'max_p': max(prof),
        'passes': (sum(1 for p in prof if 0.05 < p < 0.95) >= 3
                   and min(prof) > 0.02 and max(prof) > 0.8),
    }
    result[name] = {'az': c['az'], 'profile': prof, 'gate': gate}
    print(f"{name} az={c['az']} profile={prof} gate={gate}")

with open(OUT, 'w') as f:
    json.dump(result, f, indent=1)
print(f'wrote {OUT}')
