"""FluxPhased Effectiveness Evaluation Framework.

Provides metrics, data collection, sensitivity analysis, and reporting
for the GPU-vectorized MFAR radar combat simulation.
"""

from .collectors.ground_truth import GroundTruthComputer
from .collectors.episode_collector import EpisodeCollector, EpisodeData
from .metrics.perception import PerceptionMetrics
from .metrics.combat import CombatMetrics
from .metrics.game import GameMetrics
from .metrics.comm import CommMetrics
from .analysis.cde import CDEMetric
from .reporting.report import EvaluationReport
