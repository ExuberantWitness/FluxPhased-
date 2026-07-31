"""S2 module: S1 + jammer 1D ULA + Tx AF + MultiDiscrete([3,5]) action."""
from env.gpu.array_face_s2.array_factor import (
    RadarULAConfig, JammerULAConfig,
    BEAM_AZ_DEG_S2, N_BEAM_DIRS_S2,
    BEAM_AZ_DEG_S1, N_BEAM_DIRS_S1,
    compute_radar_af_db, compute_radar_af_db_all,
    compute_jammer_af_db, compute_jammer_af_db_all,
)
from env.gpu.array_face_s2.physics import (
    compute_jnr_db_s2, compute_p_detect_s2,
)
from env.gpu.array_face_s2.observation import (
    build_observation_s2, build_privileged_s2,
    OBS_DIM_S2, PRIVILEGED_DIM_S2, PROFILE_ARRAY_FACE_S2,
)
from env.gpu.array_face_s2.action_contract import (
    N_ACTIONS_BASE, N_ACTIONS_BEAM, MultiDiscreteTransitionTrace,
)
from env.gpu.array_face_s2.env import (
    EnvConfig, ArrayFaceS2VecEnv,
)

__all__ = [
    "RadarULAConfig", "JammerULAConfig",
    "BEAM_AZ_DEG_S2", "N_BEAM_DIRS_S2",
    "BEAM_AZ_DEG_S1", "N_BEAM_DIRS_S1",
    "compute_radar_af_db", "compute_radar_af_db_all",
    "compute_jammer_af_db", "compute_jammer_af_db_all",
    "compute_jnr_db_s2", "compute_p_detect_s2",
    "build_observation_s2", "build_privileged_s2",
    "OBS_DIM_S2", "PRIVILEGED_DIM_S2", "PROFILE_ARRAY_FACE_S2",
    "N_ACTIONS_BASE", "N_ACTIONS_BEAM",
    "MultiDiscreteTransitionTrace",
    "EnvConfig", "ArrayFaceS2VecEnv",
]
