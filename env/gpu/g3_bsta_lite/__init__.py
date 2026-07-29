"""G3-BSTA-lite clean implementation line.

New namespace for the budgeted service-target allocation MFR interference
benchmark. This package is authored from the G3-BSTA fast-work specification
(``docs/g3-bsta-lite/DEBUG_CONTRACT.md``) against the verified repository
dependencies in ``env/gpu/twoteam/``.

Status: F1 — minimal causal MFR-lite env. The :class:`G3BstaLiteVecEnv`
implements the frozen debug profile (2 services, horizon=64, duty_budget=0.25).

Provenance note: this clean line is independent of the quarantined
``mfr-orphans-20260728T094154Z`` package. No orphan bytes are imported or
executed here. The orphan package exists only as static defect evidence
under ``evidence/`` on the forensic branches.
"""

from .action_contract import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    ACTION_JAM_SERVICE_1,
    N_ACTIONS,
    ContractViolation,
    TransitionTrace,
)
from .env import EnvConfig, G3BstaLiteVecEnv
from .metrics import (
    DISPO_ADMISSION_REJECT,
    DISPO_HORIZON_FAILURE,
    DISPO_SUCCESS,
    DISPO_TIMEOUT,
    MissionCounterBatch,
    MissionTracker,
)
from .observation import OBS_DIM, PRIVILEGED_DIM
from .physics import (
    DebugPhysicsConfig,
    ServiceChannel,
    compute_detection_probability,
    compute_service_jnr_db,
    default_debug_physics_config,
)
from .radar_opponent import FrozenRuleRadar
from .scenario import Scenario, generate_paired_manifest, generate_scenario

__all__ = [
    "ACTION_IDLE",
    "ACTION_JAM_SERVICE_0",
    "ACTION_JAM_SERVICE_1",
    "ContractViolation",
    "DebugPhysicsConfig",
    "DISPO_ADMISSION_REJECT",
    "DISPO_HORIZON_FAILURE",
    "DISPO_SUCCESS",
    "DISPO_TIMEOUT",
    "EnvConfig",
    "FrozenRuleRadar",
    "G3BstaLiteVecEnv",
    "MissionCounterBatch",
    "MissionTracker",
    "N_ACTIONS",
    "OBS_DIM",
    "PRIVILEGED_DIM",
    "Scenario",
    "ServiceChannel",
    "TransitionTrace",
    "compute_detection_probability",
    "compute_service_jnr_db",
    "default_debug_physics_config",
    "generate_paired_manifest",
    "generate_scenario",
]
