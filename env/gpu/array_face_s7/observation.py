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

OBS_DIM_JAMMER: int = 55 + 12   # S6 layout + partner channel
OBS_DIM_RADAR: int = 49 + 11    # S6 layout + second jammer ESM section
PRIVILEGED_DIM_JAMMER: int = 2 * OBS_DIM_JAMMER
PRIVILEGED_DIM_RADAR: int = 2 * OBS_DIM_RADAR
N_SERVICES: int = 2
PROFILE_ARRAY_FACE_S7: str = "array_face_s7_v1"


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
    E = energy.shape[0]
    rem_E = (energy / initial_energy.clamp(min=1e-6)).clamp(0.0, 1.0)
    rem_t = torch.full((E,), float(horizon - step_idx) / float(max(horizon, 1)),
                       device=energy.device, dtype=torch.float32)
    prev_oh = _oh(prev_active, 3)  # [E, 3] (slot 2 unused, S4 convention)
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
        # partner channel
        _oh(other_beam_az, N_AZ), _oh(other_beam_el, N_EL),
        other_energy_ratio.float().unsqueeze(-1),
        other_active.float().unsqueeze(-1),
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
    jammer_beam_az: torch.Tensor,   # [E, K] (ESM DOA estimates, both jammers)
    jammer_beam_el: torch.Tensor,   # [E, K]
    jammer_active: torch.Tensor,    # [E, K] float {0,1}
) -> torch.Tensor:
    """Returns [E, 60]: S6 49-dim layout with per-jammer ESM sections."""
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
        _oh(jammer_beam_az[:, 0], N_AZ), _oh(jammer_beam_el[:, 0], N_EL),
        jammer_active[:, 0].float().unsqueeze(-1),
        _oh(jammer_beam_az[:, 1], N_AZ), _oh(jammer_beam_el[:, 1], N_EL),
        jammer_active[:, 1].float().unsqueeze(-1),
    ]
    return torch.cat(cols, dim=-1)
