"""S1 module: lite + radar 1D ULA array factor.

Re-exports lite's action_contract / scenario / metrics unchanged, plus the
S1-specific physics (array_factor), env, observation.
"""
from env.gpu.array_face_s1.array_factor import (
    RadarULAConfig, BEAM_AZ_DEG_S1, N_BEAM_DIRS_S1,
    compute_radar_af_db, compute_radar_af_db_all,
)
from env.gpu.array_face_s1.physics import (
    compute_jnr_db_s1, compute_p_detect_s1,
)
from env.gpu.array_face_s1.observation import (
    build_observation_s1, build_privileged_s1,
    OBS_DIM_S1, PRIVILEGED_DIM_S1, PROFILE_ARRAY_FACE_S1,
)
from env.gpu.array_face_s1.env import (
    EnvConfig, ArrayFaceS1VecEnv,
)

__all__ = [
    "RadarULAConfig", "BEAM_AZ_DEG_S1", "N_BEAM_DIRS_S1",
    "compute_radar_af_db", "compute_radar_af_db_all",
    "compute_jnr_db_s1", "compute_p_detect_s1",
    "build_observation_s1", "build_privileged_s1",
    "OBS_DIM_S1", "PRIVILEGED_DIM_S1", "PROFILE_ARRAY_FACE_S1",
    "EnvConfig", "ArrayFaceS1VecEnv",
]
