"""S5 observation: 43 dims — S4 layout (own) + other-jammer coarse state.

Per-jammer observation (jammer k sees its own S4-style 31-dim obs PLUS the
OTHER jammer's coarse state — the explicit coordination channel required for
IPPO parameter sharing to express division of labor):

  [0..30]   own S4 obs (31 dims, identical layout to array_face_s4)
  [31..35]  other_beam_az one-hot (5)
  [36..40]  other_beam_el one-hot (5)
  [41]      other_rem_E (other's energy / initial, clamped [0,1])
  [42]      other_prev_active (0/1)

OBS_DIM_S5 = 31 + 5 + 5 + 1 + 1 = 43.

Privileged (training-only central-critic input, CTDE):
  S4 privileged (49: pending 4 + radar az/el 10 + own az/el 10 + own cell 25)
  + other's az/el one-hots (10) + other's executed cell mask (25)
  PRIVILEGED_DIM_S5 = 49 + 10 + 25 = 74.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from env.gpu.array_face_s4.observation import (
    build_observation_s4,
    build_privileged_s4,
    OBS_DIM_S4,
    PRIVILEGED_DIM_S4,
)
from env.gpu.array_face_s5.array_factor import N_AZ, N_EL, N_CELLS_S5, N_JAMMERS

OBS_DIM_S5: int = OBS_DIM_S4 + N_AZ + N_EL + 1 + 1  # 31 + 12 = 43
# 49 (own: pending 4 + radar 10 + own az/el 10 + own cell 25) + other 10 + other cell 25 = 84
PRIVILEGED_DIM_S5: int = PRIVILEGED_DIM_S4 + N_AZ + N_EL + N_CELLS_S5  # 49 + 10 + 25 = 84
PROFILE_ARRAY_FACE_S5: str = "array_face_s5_v1"


def build_observation_s5(
    *,
    radar_beam_az_idx: torch.Tensor,     # [E]
    radar_beam_el_idx: torch.Tensor,     # [E]
    jammer_beam_az_idx: torch.Tensor,    # [E] (own, jammer k)
    jammer_beam_el_idx: torch.Tensor,    # [E]
    other_beam_az_idx: torch.Tensor,     # [E] (the other jammer)
    other_beam_el_idx: torch.Tensor,     # [E]
    prev_active: torch.Tensor,           # [E] int64 ∈ {0,1} (own)
    other_prev_active: torch.Tensor,     # [E] int64 ∈ {0,1}
    other_energy: torch.Tensor,          # [E] float
    other_initial_energy: torch.Tensor,  # [E] float
    energy, initial_energy, step_idx, horizon,
    delayed_detect=None, delayed_urgency=None,
    intercept_confidence, intercept_age,
    profile,
    pending_per_service=None, radar_service_onehot=None,
) -> torch.Tensor:
    """Returns [E, OBS_DIM_S5] = [E, 43]: own S4 obs + other's coarse state."""
    own = build_observation_s4(
        radar_beam_az_idx=radar_beam_az_idx,
        radar_beam_el_idx=radar_beam_el_idx,
        jammer_beam_az_idx=jammer_beam_az_idx,
        jammer_beam_el_idx=jammer_beam_el_idx,
        prev_active=prev_active,
        energy=energy, initial_energy=initial_energy,
        step_idx=step_idx, horizon=horizon,
        delayed_detect=delayed_detect, delayed_urgency=delayed_urgency,
        intercept_confidence=intercept_confidence, intercept_age=intercept_age,
        profile=profile,
        pending_per_service=pending_per_service,
        radar_service_onehot=radar_service_onehot,
    )
    other_az_oh = F.one_hot(other_beam_az_idx.long(), num_classes=N_AZ).to(torch.float32)
    other_el_oh = F.one_hot(other_beam_el_idx.long(), num_classes=N_EL).to(torch.float32)
    other_rem_E = (other_energy / other_initial_energy.clamp(min=1e-6)).clamp(0.0, 1.0).unsqueeze(-1)
    other_act = other_prev_active.float().unsqueeze(-1)
    return torch.cat([own, other_az_oh, other_el_oh, other_rem_E, other_act], dim=-1)


def build_privileged_s5(
    *,
    radar_beam_az_idx: torch.Tensor,
    radar_beam_el_idx: torch.Tensor,
    jammer_beam_az_idx: torch.Tensor,    # [E] own
    jammer_beam_el_idx: torch.Tensor,
    other_beam_az_idx: torch.Tensor,     # [E] other
    other_beam_el_idx: torch.Tensor,
    pending_per_service: torch.Tensor,
    track_health_per_service: torch.Tensor,
    executed_cell_mask: torch.Tensor,    # [E, 25] own
    other_executed_cell_mask: torch.Tensor,  # [E, 25] other
) -> torch.Tensor:
    """Returns [E, PRIVILEGED_DIM_S5] = [E, 84]. Central-critic (CTDE) input."""
    own_priv = build_privileged_s4(
        radar_beam_az_idx=radar_beam_az_idx,
        radar_beam_el_idx=radar_beam_el_idx,
        jammer_beam_az_idx=jammer_beam_az_idx,
        jammer_beam_el_idx=jammer_beam_el_idx,
        pending_per_service=pending_per_service,
        track_health_per_service=track_health_per_service,
        executed_cell_mask=executed_cell_mask,
    )
    other_az_oh = F.one_hot(other_beam_az_idx.long(), num_classes=N_AZ).to(torch.float32)
    other_el_oh = F.one_hot(other_beam_el_idx.long(), num_classes=N_EL).to(torch.float32)
    return torch.cat([
        own_priv, other_az_oh, other_el_oh,
        other_executed_cell_mask.to(torch.float32),
    ], dim=-1)
