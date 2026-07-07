"""Concerto-RRM: time-interleaved classical + RL cognitive radar RM.

Modules:
  - composer: V1 (fixed N:1 alternation) + V2 (event-triggered) orchestrator.
  - concerto_trainer: ConcertoTrainerAdapter + ConcertoPilotDriver.
  - noise_robust_ctde: trace(P)-weighted team-advantage blend.
"""

from .composer import (
    ComposerV1,
    ComposerV2,
    make_composer,
    OWNER_CLASSICAL,
    OWNER_RL,
)
from .noise_robust_ctde import blend_advantages, alpha_eff_scalar
from .concerto_trainer import ConcertoTrainerAdapter, ConcertoPilotDriver

__all__ = [
    "ComposerV1",
    "ComposerV2",
    "make_composer",
    "OWNER_CLASSICAL",
    "OWNER_RL",
    "blend_advantages",
    "alpha_eff_scalar",
    "ConcertoTrainerAdapter",
    "ConcertoPilotDriver",
]
