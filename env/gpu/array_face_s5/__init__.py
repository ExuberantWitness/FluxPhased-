"""S5 env: two cooperative jammers (K=2) on 2D UPAs vs one radar.

Public API mirrors S4 plus the jammer-axis (K) symbols.
"""
from env.gpu.array_face_s5.array_factor import (
    UPAConfig,
    BEAM_AZ_DEG_S5,
    BEAM_EL_DEG_S5,
    N_AZ,
    N_EL,
    N_BEAM_DIRS_S5,
    N_CELLS_S5,
    N_JAMMERS,
    beam_idx_to_az_el,
    compute_upa_af_db,
)
from env.gpu.array_face_s5.physics import (
    compute_jnr_db_s5,
    compute_p_detect_s5,
)
from env.gpu.array_face_s5.observation import (
    build_observation_s5,
    build_privileged_s5,
    OBS_DIM_S5,
    PRIVILEGED_DIM_S5,
    PROFILE_ARRAY_FACE_S5,
)
from env.gpu.array_face_s5.action_contract import (
    N_ACTIONS_CELL,
    N_ACTIONS_BEAM,
    K_JAMMERS,
    S5TransitionTrace,
    validate_actions,
)
from env.gpu.array_face_s5.env import (
    EnvConfig,
    ArrayFaceS5VecEnv,
)

__all__ = [
    # geometry
    "UPAConfig", "BEAM_AZ_DEG_S5", "BEAM_EL_DEG_S5",
    "N_AZ", "N_EL", "N_BEAM_DIRS_S5", "N_CELLS_S5", "N_JAMMERS",
    "beam_idx_to_az_el", "compute_upa_af_db",
    # physics
    "compute_jnr_db_s5", "compute_p_detect_s5",
    # observation
    "build_observation_s5", "build_privileged_s5",
    "OBS_DIM_S5", "PRIVILEGED_DIM_S5", "PROFILE_ARRAY_FACE_S5",
    # action contract
    "N_ACTIONS_CELL", "N_ACTIONS_BEAM", "K_JAMMERS",
    "S5TransitionTrace", "validate_actions",
    # env
    "EnvConfig", "ArrayFaceS5VecEnv",
]
