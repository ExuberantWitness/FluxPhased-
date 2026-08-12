"""S3 observation: same as S2 (21 dims). Privileged adds cell mask.

HANDOFF §11.1 specifies the actor observation is unchanged from S2:
  OBS_DIM_S3 = OBS_DIM_S2 = 21 (lite 11 + radar_beam_az 5 + jammer_beam_az 5)

The cell-binding decision is made WITHOUT seeing the previous cell mask in
the actor obs (the env is memoryless w.r.t. cells in the observation). This
is a deliberate design choice: the research question is whether PPO can learn
a useful cell policy from the reward signal alone.

For the privileged critic (optional, B1-style), we DO expose the executed
cell mask (training-only, not deployable). This adds N_CELLS to the privileged
dim without touching the actor obs, preserving S2/S3 obs comparability.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from env.gpu.array_face_s2.observation import (
    build_observation_s2,
    build_privileged_s2,
    OBS_DIM_S2,
    PRIVILEGED_DIM_S2,
)
from env.gpu.array_face_s3.array_factor import (
    N_CELLS, N_BEAM_DIRS_S2, N_BEAM_DIRS_S1,
)


# Actor obs unchanged from S2 (HANDOFF §11.1).
OBS_DIM_S3: int = OBS_DIM_S2  # 21
# Privileged adds the executed cell mask (training-only critic input).
PRIVILEGED_DIM_S3: int = PRIVILEGED_DIM_S2 + N_CELLS  # 14 + 5 = 19
PROFILE_ARRAY_FACE_S3: str = "array_face_s3_v1"


def build_observation_s3(
    *,
    radar_beam_az_idx: torch.Tensor,
    jammer_beam_az_idx: torch.Tensor,
    energy, initial_energy, step_idx, horizon,
    delayed_detect=None, delayed_urgency=None,
    intercept_confidence, intercept_age,
    prev_action_onehot,
    profile,
    pending_per_service=None, radar_service_onehot=None,
) -> torch.Tensor:
    """Returns [E, OBS_DIM_S3] = [E, 21]. Identical to S2 (delegates)."""
    return build_observation_s2(
        radar_beam_az_idx=radar_beam_az_idx,
        jammer_beam_az_idx=jammer_beam_az_idx,
        energy=energy, initial_energy=initial_energy,
        step_idx=step_idx, horizon=horizon,
        delayed_detect=delayed_detect, delayed_urgency=delayed_urgency,
        intercept_confidence=intercept_confidence, intercept_age=intercept_age,
        prev_action_onehot=prev_action_onehot, profile=profile,
        pending_per_service=pending_per_service,
        radar_service_onehot=radar_service_onehot,
    )


def build_privileged_s3(
    *,
    radar_beam_az_idx: torch.Tensor,
    jammer_beam_az_idx: torch.Tensor,
    pending_per_service: torch.Tensor,
    track_health_per_service: torch.Tensor,
    executed_cell_mask: torch.Tensor,   # [E, N_CELLS] float (S3 new)
) -> torch.Tensor:
    """Returns [E, PRIVILEGED_DIM_S3] = [E, 19].

    S2's 14-dim privileged (pending + health + radar_az + jammer_az) + the
    S3 executed cell mask (5 dims). The cell mask gives the privileged critic
    visibility into the resource allocation the actor chose, without leaking
    it into the deployable actor obs.
    """
    s2_priv = build_privileged_s2(
        radar_beam_az_idx=radar_beam_az_idx,
        jammer_beam_az_idx=jammer_beam_az_idx,
        pending_per_service=pending_per_service,
        track_health_per_service=track_health_per_service,
    )
    return torch.cat([s2_priv, executed_cell_mask.to(torch.float32)], dim=-1)
