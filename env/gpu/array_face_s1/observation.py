"""S1 observation: lite 11 dims + 5-dim radar_beam_az one-hot = 16 dims.

The radar_beam_az one-hot is added in MDP mode (jammer directly observes
current radar_beam_az). This is a deliberate simplification over a strict
POMDP to make S1's research question well-defined:
    "Can PPO exploit knowledge of radar's beam_az to choose better actions?"

Future phases may obscure beam_az (true POMDP) once S1's baseline is established.
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
from env.gpu.array_face_s1.array_factor import N_BEAM_DIRS_S1

OBS_DIM_S1: int = OBS_DIM_LITE + N_BEAM_DIRS_S1               # 11 + 5 = 16
PRIVILEGED_DIM_S1: int = PRIV_DIM_LITE + N_BEAM_DIRS_S1       # 4 + 5 = 9
PROFILE_ARRAY_FACE_S1: str = "array_face_s1_v1"


def build_observation_s1(
    *,
    radar_beam_az_idx: torch.Tensor,
    energy, initial_energy, step_idx, horizon,
    delayed_detect=None, delayed_urgency=None,
    intercept_confidence, intercept_age,
    prev_action_onehot,
    profile=PROFILE_POMDP,
    pending_per_service=None, radar_service_onehot=None,
) -> torch.Tensor:
    """Returns [E, OBS_DIM_S1] = [E, 16]. Concatenates lite obs with beam_az one-hot."""
    lite_obs = build_lite_obs(
        energy=energy, initial_energy=initial_energy,
        step_idx=step_idx, horizon=horizon,
        delayed_detect=delayed_detect, delayed_urgency=delayed_urgency,
        intercept_confidence=intercept_confidence, intercept_age=intercept_age,
        prev_action_onehot=prev_action_onehot, profile=profile,
        pending_per_service=pending_per_service,
        radar_service_onehot=radar_service_onehot,
    )
    beam_oh = F.one_hot(radar_beam_az_idx.long(), num_classes=N_BEAM_DIRS_S1).to(torch.float32)
    return torch.cat([lite_obs, beam_oh], dim=-1)


def build_privileged_s1(
    *,
    radar_beam_az_idx: torch.Tensor,
    pending_per_service: torch.Tensor,
    track_health_per_service: torch.Tensor,
) -> torch.Tensor:
    """Returns [E, PRIVILEGED_DIM_S1] = [E, 9]. Adds beam_az one-hot to lite privileged."""
    beam_oh = F.one_hot(radar_beam_az_idx.long(), num_classes=N_BEAM_DIRS_S1).to(torch.float32)
    return torch.cat([pending_per_service.float(), track_health_per_service.float(), beam_oh], dim=-1)
