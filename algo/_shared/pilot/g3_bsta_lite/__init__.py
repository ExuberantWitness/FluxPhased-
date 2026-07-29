"""G3-BSTA-lite baselines, oracle, imitation and PPO pilot.

Companion package to ``env/gpu/g3_bsta_lite/``. Status: F2 introduces the
first frozen baselines + executable clairvoyant oracle + 128 paired
scenarios + LCB95 evaluation harness.

Provenance note: this clean line is independent of the quarantined
``mfr-orphans-20260728T094154Z`` package.
"""

from .baselines import (
    AlwaysOff,
    Baseline,
    BudgetedBarrage,
    BudgetedRoundRobin,
    CausalReactiveOrEDF,
    FROZEN_BASELINES,
    PeriodicBlink,
    RandomFeasible,
)
from .evaluation import (
    PolicyResult,
    ScenarioResult,
    evaluate_policies,
)
from .oracle import (
    ClairvoyantGreedyOracle,
    ClairvoyantOptimalOracle,
    make_clairvoyant_oracle,
)

__all__ = [
    "AlwaysOff",
    "Baseline",
    "BudgetedBarrage",
    "BudgetedRoundRobin",
    "CausalReactiveOrEDF",
    "ClairvoyantGreedyOracle",
    "ClairvoyantOptimalOracle",
    "FROZEN_BASELINES",
    "PeriodicBlink",
    "PolicyResult",
    "RandomFeasible",
    "ScenarioResult",
    "evaluate_policies",
    "make_clairvoyant_oracle",
]
