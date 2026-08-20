"""S6 env: adversarial 1-jammer vs 2-learning-radars."""
from env.gpu.array_face_s6.array_factor import (
    UPAConfig, N_BEAM_DIRS_S6, N_CELLS_S6, N_RADARS,
    compute_upa_af_db_toward, az_el_to_uv,
)
from env.gpu.array_face_s6.geometry import radar_directions, radar_uv, RADAR_AZ_DEG
from env.gpu.array_face_s6.physics import compute_jnr_db_s6, compute_p_detect_s6
from env.gpu.array_face_s6.observation import (
    build_observation_jammer, build_observation_radar,
    OBS_DIM_JAMMER, OBS_DIM_RADAR, PROFILE_ARRAY_FACE_S6,
)
from env.gpu.array_face_s6.action_contract import validate_actions
from env.gpu.array_face_s6.env import EnvConfig, ArrayFaceS6VecEnv

__all__ = [
    "UPAConfig", "N_BEAM_DIRS_S6", "N_CELLS_S6", "N_RADARS",
    "compute_upa_af_db_toward", "az_el_to_uv",
    "radar_directions", "radar_uv", "RADAR_AZ_DEG",
    "compute_jnr_db_s6", "compute_p_detect_s6",
    "build_observation_jammer", "build_observation_radar",
    "OBS_DIM_JAMMER", "OBS_DIM_RADAR", "PROFILE_ARRAY_FACE_S6",
    "validate_actions", "EnvConfig", "ArrayFaceS6VecEnv",
]
