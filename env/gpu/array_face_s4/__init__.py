"""S4 env: 2D 5×5 UPA + Bernoulli(25) cell binding + Categorical(25) 2D beam.

Public API mirrors S3 but with UPA geometry symbols (UPAConfig, N_CELLS_S4,
N_BEAM_DIRS_S4, compute_upa_af_db, compute_jnr_db_s4) and a 2-head action
contract (cell + beam; no base head).
"""
from env.gpu.array_face_s4.array_factor import (
    UPAConfig,
    BEAM_AZ_DEG_S4,
    BEAM_EL_DEG_S4,
    N_AZ,
    N_EL,
    N_BEAM_DIRS_S4,
    N_CELLS_S4,
    beam_idx_to_az_el,
    compute_upa_af_db,
    compute_upa_af_db_flat,
    upa_af_table,
)
from env.gpu.array_face_s4.physics import (
    compute_jnr_db_s4,
    compute_p_detect_s4,
)
from env.gpu.array_face_s4.observation import (
    build_observation_s4,
    build_privileged_s4,
    OBS_DIM_S4,
    PRIVILEGED_DIM_S4,
    PROFILE_ARRAY_FACE_S4,
)
from env.gpu.array_face_s4.action_contract import (
    N_ACTIONS_CELL,
    N_ACTIONS_BEAM,
    UPATransitionTrace,
    validate_actions,
)
from env.gpu.array_face_s4.env import (
    EnvConfig,
    ArrayFaceS4VecEnv,
)

__all__ = [
    # geometry
    "UPAConfig",
    "BEAM_AZ_DEG_S4", "BEAM_EL_DEG_S4",
    "N_AZ", "N_EL", "N_BEAM_DIRS_S4", "N_CELLS_S4",
    "beam_idx_to_az_el",
    "compute_upa_af_db", "compute_upa_af_db_flat", "upa_af_table",
    # physics
    "compute_jnr_db_s4", "compute_p_detect_s4",
    # observation
    "build_observation_s4", "build_privileged_s4",
    "OBS_DIM_S4", "PRIVILEGED_DIM_S4", "PROFILE_ARRAY_FACE_S4",
    # action contract
    "N_ACTIONS_CELL", "N_ACTIONS_BEAM",
    "UPATransitionTrace", "validate_actions",
    # env
    "EnvConfig", "ArrayFaceS4VecEnv",
]
