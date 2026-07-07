"""QoS-RRM env layer: Concerto-RRM support.

Modules:
  - spectrum_metrics: per-function QoS signal computation (Pd, trace_P_norm,
    crc_pass_rate, jam_power_on_victim_db, qos_satisfaction).
  - qos_rrm_env: asymmetric-team wrapper around MFARVecEnv (cognitive radar vs
    adaptive jammer) — A1.
  - adversary: L0/L1/L3 adaptive jammer opponents — A3.
"""

from .spectrum_metrics import (
    pd_at_pfa,
    trace_P_norm,
    crc_pass_rate,
    jam_power_on_victim_db,
    qos_satisfaction,
)
from .qos_rrm_env import QoSRRMEnv
from .adversary import (
    StaticJammer,
    ReactiveJammer,
    LearnedJammer,
    make_jammer,
)

__all__ = [
    "pd_at_pfa",
    "trace_P_norm",
    "crc_pass_rate",
    "jam_power_on_victim_db",
    "qos_satisfaction",
    "QoSRRMEnv",
    "StaticJammer",
    "ReactiveJammer",
    "LearnedJammer",
    "make_jammer",
]
