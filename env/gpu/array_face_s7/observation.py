"""S7 observations — asymmetric two-sided views for the 2v2 game.

Jammer obs [E, K, 67] (per jammer k, parameter-shared):
  S6 55-dim layout (own energy/state + BOTH radars' beams/svcs/detected +
  mission-bearing az map), plus a 12-dim partner channel:
  [55..59] other jammer beam az oh, [60..64] other jammer beam el oh,
  [65] other jammer energy ratio, [66] other jammer active flag.

Radar obs [E, R, 60] (per radar r, parameter-shared):
  S6 49-dim layout (own/other radar + ONE jammer ESM), but the jammer ESM
  section is duplicated per jammer:
  [38..42] jam0 DOA az oh, [43..47] jam0 DOA el oh, [48] jam0 active,
  [49..53] jam1 DOA az oh, [54..58] jam1 DOA el oh, [59] jam1 active.
  own_intercept_confidence is now PER-RADAR (each radar has its own snr_eff).

Privileged (central-critic) views — the global public state:
  priv_j [E, 134] = concat of both jammers' full obs (already include radar
                    beams/svcs), used by the jammer team's central critic
  priv_r [E, 120] = concat of both radars' full obs (already include both
                    jammers' DOA/active), used by the radar team's central critic
No oracle information (deadlines, true arrivals) is exposed — centralization
is over the public ESM-visible state only.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from env.gpu.array_face_s6.array_factor import N_AZ, N_EL

OBS_DIM_JAMMER: int = 55 + 12   # S6 layout + partner channel (n_jammers = 2)
OBS_DIM_RADAR: int = 49 + 11    # S6 layout + second jammer ESM section (K = 2)
PRIVILEGED_DIM_JAMMER: int = 2 * OBS_DIM_JAMMER
PRIVILEGED_DIM_RADAR: int = 2 * OBS_DIM_RADAR
N_SERVICES: int = 2
PROFILE_ARRAY_FACE_S7: str = "array_face_s7_v1"

# n-jammer layout dims (attacker-count scaling): the jammer view carries one
# 12-dim partner channel per teammate; the radar view carries one 11-dim ESM
# section per jammer beyond the first (the S6 49-dim layout already includes
# one jammer ESM section).
def obs_dim_jammer(n_jammers: int) -> int:
    return 55 + 12 * (n_jammers - 1)

def obs_dim_radar(n_jammers: int) -> int:
    return 49 + 11 * (n_jammers - 1)

def priv_dim_jammer(n_jammers: int, n_radars: int = 2) -> int:
    return n_jammers * obs_dim_jammer(n_jammers)

def priv_dim_radar(n_jammers: int, n_radars: int = 2) -> int:
    return n_radars * obs_dim_radar(n_jammers)


def _oh(idx: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(idx.long(), num_classes=n).to(torch.float32)


def build_observation_jammer(
    *,
    energy, initial_energy, step_idx, horizon,
    pending_per_service,            # [E, 2]
    pending_az_map,                 # [E, n_services, N_AZ] float
    intercept_confidence,           # [E] (mean over radars, 0 when all idle)
    intercept_age,                  # [E] (own, per jammer)
    prev_active,                    # [E] int64 {0,1}
    radar_beam_az: torch.Tensor,    # [E, R]
    radar_beam_el: torch.Tensor,    # [E, R]
    radar_svc: torch.Tensor,        # [E, R]
    jammer_beam_az: torch.Tensor,   # [E] (own)
    jammer_beam_el: torch.Tensor,   # [E] (own)
    radar_detected_last: torch.Tensor,  # [E, R] float {0,1}
    other_beam_az: torch.Tensor,    # [E] (partner jammer)
    other_beam_el: torch.Tensor,    # [E]
    other_energy_ratio: torch.Tensor,   # [E] float 0..1
    other_active: torch.Tensor,     # [E] float {0,1}
) -> torch.Tensor:
    """Returns [E, 67]: S6 55-dim layout + 12-dim partner coordination channel."""
    return build_observation_jammer_n(
        energy=energy, initial_energy=initial_energy, step_idx=step_idx,
        horizon=horizon, pending_per_service=pending_per_service,
        pending_az_map=pending_az_map,
        intercept_confidence=intercept_confidence, intercept_age=intercept_age,
        prev_active=prev_active, radar_beam_az=radar_beam_az,
        radar_beam_el=radar_beam_el, radar_svc=radar_svc,
        jammer_beam_az=jammer_beam_az, jammer_beam_el=jammer_beam_el,
        radar_detected_last=radar_detected_last,
        others_beam_az=other_beam_az.unsqueeze(-1),
        others_beam_el=other_beam_el.unsqueeze(-1),
        others_energy_ratio=other_energy_ratio.unsqueeze(-1),
        others_active=other_active.unsqueeze(-1),
    )


def build_observation_jammer_n(
    *,
    energy, initial_energy, step_idx, horizon,
    pending_per_service,            # [E, 2]
    pending_az_map,                 # [E, n_services, N_AZ] float
    intercept_confidence,           # [E]
    intercept_age,                  # [E] (own, per jammer)
    prev_active,                    # [E] int64 {0,1}
    radar_beam_az: torch.Tensor,    # [E, R]
    radar_beam_el: torch.Tensor,    # [E, R]
    radar_svc: torch.Tensor,        # [E, R]
    jammer_beam_az: torch.Tensor,   # [E] (own)
    jammer_beam_el: torch.Tensor,   # [E] (own)
    radar_detected_last: torch.Tensor,  # [E, R] float {0,1}
    others_beam_az: torch.Tensor,   # [E, K-1] partners, ascending slot order
    others_beam_el: torch.Tensor,   # [E, K-1]
    others_energy_ratio: torch.Tensor,  # [E, K-1]
    others_active: torch.Tensor,    # [E, K-1]
) -> torch.Tensor:
    """Returns [E, 55 + 12*(K-1)]: S6 prefix + one partner channel per teammate."""
    E = energy.shape[0]
    rem_E = (energy / initial_energy.clamp(min=1e-6)).clamp(0.0, 1.0)
    rem_t = torch.full((E,), float(horizon - step_idx) / float(max(horizon, 1)),
                       device=energy.device, dtype=torch.float32)
    prev_oh = _oh(prev_active, 3)
    cols = [
        rem_E.unsqueeze(-1), rem_t.unsqueeze(-1),
        pending_per_service.float(),
        intercept_confidence.unsqueeze(-1), intercept_age.float().unsqueeze(-1),
        prev_oh,
        _oh(radar_beam_az[:, 0], N_AZ), _oh(radar_beam_el[:, 0], N_EL),
        _oh(radar_beam_az[:, 1], N_AZ), _oh(radar_beam_el[:, 1], N_EL),
        _oh(radar_svc[:, 0], N_SERVICES), _oh(radar_svc[:, 1], N_SERVICES),
        _oh(jammer_beam_az, N_AZ), _oh(jammer_beam_el, N_EL),
        radar_detected_last.float(),
        pending_az_map.float().reshape(E, -1),
    ]
    n_partners = others_beam_az.shape[-1]
    for p in range(n_partners):
        cols += [
            _oh(others_beam_az[:, p], N_AZ), _oh(others_beam_el[:, p], N_EL),
            others_energy_ratio[:, p].float().unsqueeze(-1),
            others_active[:, p].float().unsqueeze(-1),
        ]
    return torch.cat(cols, dim=-1)


def build_observation_radar(
    *,
    step_idx, horizon,
    pending_az_map,                 # [E, n_services, N_AZ] float
    own_intercept_confidence,       # [E] (per radar)
    own_detected_last,              # [E] float {0,1}
    other_detected_last,            # [E]
    own_beam_az: torch.Tensor,      # [E]
    own_beam_el: torch.Tensor,      # [E]
    own_svc: torch.Tensor,          # [E]
    other_beam_az: torch.Tensor,    # [E]
    other_beam_el: torch.Tensor,    # [E]
    other_svc: torch.Tensor,        # [E]
    jammer_beam_az: torch.Tensor,   # [E, K] (ESM DOA estimates, all jammers)
    jammer_beam_el: torch.Tensor,   # [E, K]
    jammer_active: torch.Tensor,    # [E, K] float {0,1}
) -> torch.Tensor:
    """Returns [E, 49 + 11*(K-1)]: S6 49-dim layout with per-jammer ESM sections."""
    return build_observation_radar_n(
        step_idx=step_idx, horizon=horizon, pending_az_map=pending_az_map,
        own_intercept_confidence=own_intercept_confidence,
        own_detected_last=own_detected_last,
        other_detected_last=other_detected_last,
        own_beam_az=own_beam_az, own_beam_el=own_beam_el, own_svc=own_svc,
        other_beam_az=other_beam_az, other_beam_el=other_beam_el,
        other_svc=other_svc,
        jammer_beam_az=jammer_beam_az, jammer_beam_el=jammer_beam_el,
        jammer_active=jammer_active,
    )


def build_observation_radar_n(
    *,
    step_idx, horizon,
    pending_az_map,                 # [E, n_services, N_AZ] float
    own_intercept_confidence,       # [E] (per radar)
    own_detected_last,              # [E] float {0,1}
    other_detected_last,            # [E]
    own_beam_az: torch.Tensor,      # [E]
    own_beam_el: torch.Tensor,      # [E]
    own_svc: torch.Tensor,          # [E]
    other_beam_az: torch.Tensor,    # [E]
    other_beam_el: torch.Tensor,    # [E]
    other_svc: torch.Tensor,        # [E]
    jammer_beam_az: torch.Tensor,   # [E, K] (ESM DOA estimates, all jammers)
    jammer_beam_el: torch.Tensor,   # [E, K]
    jammer_active: torch.Tensor,    # [E, K] float {0,1}
) -> torch.Tensor:
    """Returns [E, 49 + 11*(K-1)]: S6 prefix + one 11-dim ESM section per jammer."""
    E = pending_az_map.shape[0]
    rem_t = torch.full((E,), float(horizon - step_idx) / float(max(horizon, 1)),
                       device=pending_az_map.device, dtype=torch.float32)
    cols = [
        rem_t.unsqueeze(-1),
        pending_az_map.float().reshape(E, -1),
        own_intercept_confidence.unsqueeze(-1),
        own_detected_last.float().unsqueeze(-1),
        other_detected_last.float().unsqueeze(-1),
        _oh(own_beam_az, N_AZ), _oh(own_beam_el, N_EL), _oh(own_svc, N_SERVICES),
        _oh(other_beam_az, N_AZ), _oh(other_beam_el, N_EL), _oh(other_svc, N_SERVICES),
    ]
    K = jammer_beam_az.shape[-1]
    for k in range(K):
        cols += [
            _oh(jammer_beam_az[:, k], N_AZ), _oh(jammer_beam_el[:, k], N_EL),
            jammer_active[:, k].float().unsqueeze(-1),
        ]
    return torch.cat(cols, dim=-1)
