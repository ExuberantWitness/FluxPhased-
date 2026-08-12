"""S3 env: S2 + jammer cell binding (Bernoulli(5)) + per-cell energy budget.

Public API mirrors S2 plus cell-binding symbols (N_CELLS, compute_jnr_db_s3).
"""
from env.gpu.array_face_s3.array_factor import (
    RadarULAConfig,
    JammerULAConfig,
    BEAM_AZ_DEG_S1,
    N_BEAM_DIRS_S1,
    BEAM_AZ_DEG_S2,
    N_BEAM_DIRS_S2,
    N_CELLS,
    compute_radar_af_db,
    compute_radar_af_db_all,
    compute_jammer_af_db,
    compute_jammer_af_db_all,
)
from env.gpu.array_face_s3.physics import (
    compute_jnr_db_s3,
    compute_p_detect_s3,
)
from env.gpu.array_face_s3.observation import (
    build_observation_s3,
    build_privileged_s3,
    OBS_DIM_S3,
    PRIVILEGED_DIM_S3,
    PROFILE_ARRAY_FACE_S3,
)
from env.gpu.array_face_s3.action_contract import (
    N_ACTIONS_BASE,
    N_ACTIONS_BEAM,
    BernoulliTransitionTrace,
    validate_actions,
)
from env.gpu.array_face_s3.env import (
    EnvConfig,
    ArrayFaceS3VecEnv,
)

__all__ = [
    # geometry
    "RadarULAConfig", "JammerULAConfig",
    "BEAM_AZ_DEG_S1", "N_BEAM_DIRS_S1", "BEAM_AZ_DEG_S2", "N_BEAM_DIRS_S2",
    "N_CELLS",
    "compute_radar_af_db", "compute_radar_af_db_all",
    "compute_jammer_af_db", "compute_jammer_af_db_all",
    # physics
    "compute_jnr_db_s3", "compute_p_detect_s3",
    # observation
    "build_observation_s3", "build_privileged_s3",
    "OBS_DIM_S3", "PRIVILEGED_DIM_S3", "PROFILE_ARRAY_FACE_S3",
    # action contract
    "N_ACTIONS_BASE", "N_ACTIONS_BEAM",
    "BernoulliTransitionTrace", "validate_actions",
    # env
    "EnvConfig", "ArrayFaceS3VecEnv",
]
