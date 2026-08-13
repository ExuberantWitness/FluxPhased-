"""S4 observation: 31 dims — S3 layout + elevation one-hots on both sides.

Layout (MDP profile shown; POMDP slots mirror lite):
  [0]    rem_E
  [1]    rem_t
  [2..3] pending_per_service (exact, float) / delayed_detect (POMDP)
  [4..5] radar_service_onehot            / delayed_urgency (POMDP)
  [6]    intercept_confidence
  [7]    intercept_age
  [8..10] prev_activity_onehot: [idle, jam, unused] — S4 has no base head,
          so the lite prev-action slot encodes whether the previous step
          transmitted (>=1 cell) or idled (all-zero cells). Dim 2 is always
          0 and kept so the lite 11-dim layout is reused unchanged.
  [11..15] radar_beam_az one-hot (5)
  [16..20] radar_beam_el one-hot (5)     <- new (elevation)
  [21..25] jammer_beam_az one-hot (5)
  [26..30] jammer_beam_el one-hot (5)    <- new (elevation)

OBS_DIM_S4 = 11 + 5 + 5 + 5 + 5 = 31.

The cell mask is NOT in the actor obs (same deliberate choice as S3 §11.1:
the research question is whether PPO learns a cell policy from reward alone).
Privileged (training-only critic input) exposes the executed cell mask:
  PRIVILEGED_DIM_S4 = 4 + 5 + 5 + 5 + 5 + 25 = 49.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from env.gpu.g3_bsta_lite.observation import (
    build_observation as build_lite_obs,
    OBS_DIM as OBS_DIM_LITE,
    PRIVILEGED_DIM as PRIV_DIM_LITE,
    PROFILE_MDP_SANITY, PROFILE_POMDP,
)
from env.gpu.array_face_s4.array_factor import N_AZ, N_EL, N_CELLS_S4

OBS_DIM_S4: int = OBS_DIM_LITE + N_AZ + N_EL + N_AZ + N_EL  # 11 + 20 = 31
PRIVILEGED_DIM_S4: int = PRIV_DIM_LITE + N_AZ + N_EL + N_AZ + N_EL + N_CELLS_S4  # 49
PROFILE_ARRAY_FACE_S4: str = "array_face_s4_v1"


def build_observation_s4(
    *,
    radar_beam_az_idx: torch.Tensor,
    radar_beam_el_idx: torch.Tensor,
    jammer_beam_az_idx: torch.Tensor,
    jammer_beam_el_idx: torch.Tensor,
    prev_active: torch.Tensor,           # [E] int64 ∈ {0,1}
    energy, initial_energy, step_idx, horizon,
    delayed_detect=None, delayed_urgency=None,
    intercept_confidence, intercept_age,
    profile=PROFILE_MDP_SANITY,
    pending_per_service=None, radar_service_onehot=None,
) -> torch.Tensor:
    """Returns [E, OBS_DIM_S4] = [E, 31].

    Layout: [lite 11] [radar az 5] [radar el 5] [jammer az 5] [jammer el 5].
    """
    prev_oh = F.one_hot(prev_active.long(), num_classes=3).to(torch.float32)
    lite_obs = build_lite_obs(
        energy=energy, initial_energy=initial_energy,
        step_idx=step_idx, horizon=horizon,
        delayed_detect=delayed_detect, delayed_urgency=delayed_urgency,
        intercept_confidence=intercept_confidence, intercept_age=intercept_age,
        prev_action_onehot=prev_oh, profile=profile,
        pending_per_service=pending_per_service,
        radar_service_onehot=radar_service_onehot,
    )
    radar_az_oh = F.one_hot(radar_beam_az_idx.long(), num_classes=N_AZ).to(torch.float32)
    radar_el_oh = F.one_hot(radar_beam_el_idx.long(), num_classes=N_EL).to(torch.float32)
    jammer_az_oh = F.one_hot(jammer_beam_az_idx.long(), num_classes=N_AZ).to(torch.float32)
    jammer_el_oh = F.one_hot(jammer_beam_el_idx.long(), num_classes=N_EL).to(torch.float32)
    return torch.cat([lite_obs, radar_az_oh, radar_el_oh, jammer_az_oh, jammer_el_oh], dim=-1)


def build_privileged_s4(
    *,
    radar_beam_az_idx: torch.Tensor,
    radar_beam_el_idx: torch.Tensor,
    jammer_beam_az_idx: torch.Tensor,
    jammer_beam_el_idx: torch.Tensor,
    pending_per_service: torch.Tensor,
    track_health_per_service: torch.Tensor,
    executed_cell_mask: torch.Tensor,    # [E, N_CELLS_S4] float
) -> torch.Tensor:
    """Returns [E, PRIVILEGED_DIM_S4] = [E, 49]."""
    radar_az_oh = F.one_hot(radar_beam_az_idx.long(), num_classes=N_AZ).to(torch.float32)
    radar_el_oh = F.one_hot(radar_beam_el_idx.long(), num_classes=N_EL).to(torch.float32)
    jammer_az_oh = F.one_hot(jammer_beam_az_idx.long(), num_classes=N_AZ).to(torch.float32)
    jammer_el_oh = F.one_hot(jammer_beam_el_idx.long(), num_classes=N_EL).to(torch.float32)
    return torch.cat([
        pending_per_service.float(),
        track_health_per_service.float(),
        radar_az_oh, radar_el_oh, jammer_az_oh, jammer_el_oh,
        executed_cell_mask.to(torch.float32),
    ], dim=-1)
