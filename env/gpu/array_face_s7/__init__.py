"""S7 env: 2 learning jammers vs 2 learning radars (full two-team MAPPO)."""
from env.gpu.array_face_s7.array_factor import (
    UPAConfig, N_BEAM_DIRS_S7, N_CELLS_S7, N_JAMMERS, N_RADARS,
    compute_upa_af_db_toward, az_el_to_uv,
)
from env.gpu.array_face_s7.geometry import (
    radar_directions, jammer_directions, pair_bearings, pair_uv,
    RADAR_AZ_DEG, JAMMER_AZ_DEG,
)
from env.gpu.array_face_s7.physics import (
    compute_jnr_db_s7, compute_p_detect_s7,
    compute_snr_eff_db_s6, target_gain_db,
)
from env.gpu.array_face_s7.observation import (
    build_observation_jammer, build_observation_radar,
    OBS_DIM_JAMMER, OBS_DIM_RADAR,
    PRIVILEGED_DIM_JAMMER, PRIVILEGED_DIM_RADAR,
    PROFILE_ARRAY_FACE_S7,
)
from env.gpu.array_face_s7.action_contract import validate_actions
from env.gpu.array_face_s7.env import EnvConfig, ArrayFaceS7VecEnv

__all__ = [
    "UPAConfig", "N_BEAM_DIRS_S7", "N_CELLS_S7", "N_JAMMERS", "N_RADARS",
    "compute_upa_af_db_toward", "az_el_to_uv",
    "radar_directions", "jammer_directions", "pair_bearings", "pair_uv",
    "RADAR_AZ_DEG", "JAMMER_AZ_DEG",
    "compute_jnr_db_s7", "compute_p_detect_s7",
    "compute_snr_eff_db_s6", "target_gain_db",
    "build_observation_jammer", "build_observation_radar",
    "OBS_DIM_JAMMER", "OBS_DIM_RADAR",
    "PRIVILEGED_DIM_JAMMER", "PRIVILEGED_DIM_RADAR",
    "PROFILE_ARRAY_FACE_S7",
    "validate_actions", "EnvConfig", "ArrayFaceS7VecEnv",
]
