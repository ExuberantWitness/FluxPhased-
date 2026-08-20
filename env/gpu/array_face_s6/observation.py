"""S6 observations — asymmetric two-sided views (jammer vs radar team).

Jammer obs [E, 45] (S4-style ESM view of BOTH radars):
  [0] rem_E, [1] rem_t, [2..3] pending_per_service, [4] intercept_confidence,
  [5] intercept_age, [6..8] prev_activity onehot [idle,jam,unused],
  [9..13] radar0 beam az oh, [14..18] radar0 beam el oh,
  [19..23] radar1 beam az oh, [24..28] radar1 beam el oh,
  [29..30] radar0 svc oh, [31..32] radar1 svc oh,
  [33..37] jammer prev beam az oh, [38..42] jammer prev beam el oh,
  [43..44] per-radar detected-last-step flags.

Radar obs [E, 41] (parameter-shared per radar; scheduling + ESM view):
  [0] rem_t, [1..2] pending_per_service, [3] own intercept_confidence,
  [4] own detected-last-step, [5] other detected-last-step,
  [6..10] own beam az oh, [11..15] own beam el oh, [16..17] own svc oh,
  [18..22] other beam az oh, [23..27] other beam el oh, [28..29] other svc oh,
  [30..34] jammer DOA az oh (jammer prev beam), [35..39] jammer DOA el oh,
  [40] jammer active flag.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from env.gpu.array_face_s6.array_factor import N_AZ, N_EL, N_CELLS_S6, N_RADARS

OBS_DIM_JAMMER: int = 45 + 10  # S6b: + per-service×az mission-bearing map
OBS_DIM_RADAR: int = 41 - 2 + 10  # S6b: per-service pending (2) replaced by the 10-dim map
N_SERVICES: int = 2
PROFILE_ARRAY_FACE_S6: str = "array_face_s6_v1"


def _oh(idx: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(idx.long(), num_classes=n).to(torch.float32)


def build_observation_jammer(
    *,
    energy, initial_energy, step_idx, horizon,
    pending_per_service,            # [E, 2]
    pending_az_map,                 # [E, n_services, N_AZ] float
    intercept_confidence, intercept_age,
    prev_active,                    # [E] int64 {0,1}
    radar_beam_az: torch.Tensor,    # [E, R]
    radar_beam_el: torch.Tensor,    # [E, R]
    radar_svc: torch.Tensor,        # [E, R]
    jammer_beam_az: torch.Tensor,   # [E]
    jammer_beam_el: torch.Tensor,   # [E]
    radar_detected_last: torch.Tensor,  # [E, R] float {0,1}
) -> torch.Tensor:
    """Returns [E, 55]: S6a layout + mission-bearing map [svc0 az 5 | svc1 az 5]."""
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
    ]
    return torch.cat(cols, dim=-1)


def build_observation_radar(
    *,
    step_idx, horizon,
    pending_az_map,                 # [E, n_services, N_AZ] float
    own_intercept_confidence,       # [E]
    own_detected_last,              # [E] float {0,1}
    other_detected_last,            # [E]
    own_beam_az: torch.Tensor,      # [E]
    own_beam_el: torch.Tensor,      # [E]
    own_svc: torch.Tensor,          # [E]
    other_beam_az: torch.Tensor,    # [E]
    other_beam_el: torch.Tensor,    # [E]
    other_svc: torch.Tensor,        # [E]
    jammer_beam_az: torch.Tensor,   # [E] (ESM DOA estimate)
    jammer_beam_el: torch.Tensor,   # [E]
    jammer_active: torch.Tensor,    # [E] float {0,1}
) -> torch.Tensor:
    """Returns [E, 49]: the view of ONE radar (map replaces per-service counts)."""
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
        _oh(jammer_beam_az, N_AZ), _oh(jammer_beam_el, N_EL),
        jammer_active.float().unsqueeze(-1),
    ]
    return torch.cat(cols, dim=-1)
