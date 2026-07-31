"""S2 observation: S1 obs + jammer_beam_az one-hot = 21 dims.

S1 obs was: lite 11 + radar_beam_az one-hot 5 = 16.
S2 adds: jammer_beam_az one-hot 5 (jammer's OWN last chosen beam, observable).

This makes the env MDP (jammer knows what beam it last steered) and gives the
actor the input it needs to learn a beam-selection policy.
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
from env.gpu.array_face_s2.array_factor import (
    N_BEAM_DIRS_S2, N_BEAM_DIRS_S1,
)

# S2 obs = lite 11 + radar_beam_az 5 + jammer_beam_az 5 = 21
OBS_DIM_S2: int = OBS_DIM_LITE + N_BEAM_DIRS_S1 + N_BEAM_DIRS_S2  # 11 + 5 + 5 = 21
PRIVILEGED_DIM_S2: int = PRIV_DIM_LITE + N_BEAM_DIRS_S1 + N_BEAM_DIRS_S2
PROFILE_ARRAY_FACE_S2: str = "array_face_s2_v1"


def build_observation_s2(
    *,
    radar_beam_az_idx: torch.Tensor,
    jammer_beam_az_idx: torch.Tensor,
    energy, initial_energy, step_idx, horizon,
    delayed_detect=None, delayed_urgency=None,
    intercept_confidence, intercept_age,
    prev_action_onehot,
    profile=PROFILE_POMDP,
    pending_per_service=None, radar_service_onehot=None,
) -> torch.Tensor:
    """Returns [E, OBS_DIM_S2] = [E, 21].

    Layout: [lite 11] [radar_beam_az one-hot 5] [jammer_beam_az one-hot 5]
    """
    lite_obs = build_lite_obs(
        energy=energy, initial_energy=initial_energy,
        step_idx=step_idx, horizon=horizon,
        delayed_detect=delayed_detect, delayed_urgency=delayed_urgency,
        intercept_confidence=intercept_confidence, intercept_age=intercept_age,
        prev_action_onehot=prev_action_onehot, profile=profile,
        pending_per_service=pending_per_service,
        radar_service_onehot=radar_service_onehot,
    )
    radar_oh = F.one_hot(radar_beam_az_idx.long(), num_classes=N_BEAM_DIRS_S1).to(torch.float32)
    jammer_oh = F.one_hot(jammer_beam_az_idx.long(), num_classes=N_BEAM_DIRS_S2).to(torch.float32)
    return torch.cat([lite_obs, radar_oh, jammer_oh], dim=-1)


def build_privileged_s2(
    *,
    radar_beam_az_idx: torch.Tensor,
    jammer_beam_az_idx: torch.Tensor,
    pending_per_service: torch.Tensor,
    track_health_per_service: torch.Tensor,
) -> torch.Tensor:
    """Returns [E, PRIVILEGED_DIM_S2] = [E, 14]."""
    radar_oh = F.one_hot(radar_beam_az_idx.long(), num_classes=N_BEAM_DIRS_S1).to(torch.float32)
    jammer_oh = F.one_hot(jammer_beam_az_idx.long(), num_classes=N_BEAM_DIRS_S2).to(torch.float32)
    return torch.cat([
        pending_per_service.float(),
        track_health_per_service.float(),
        radar_oh, jammer_oh,
    ], dim=-1)
