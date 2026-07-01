"""Scenario selector for calibration parameter sweeps.

Reuses trigger_sources.py for parameter bounds and scenario_generator.py
for Sobol/random sampling.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class CalibrationParameter:
    """A single calibratable parameter with bounds."""
    name: str
    param_name: str       # MFARVecEnv kwarg name
    bounds: tuple         # (min, max)
    true_value: float
    unit: str = ""


class ScenarioSelector:
    """Generate calibration scenarios by sampling parameter space."""

    def __init__(
        self,
        calibratable_params: List[CalibrationParameter],
    ):
        self.params = calibratable_params
        self._param_names = [p.param_name for p in self.params]

    @classmethod
    def from_trigger_sources(cls, param_names: List[str]) -> "ScenarioSelector":
        """Create from evaluation trigger source definitions."""
        from ..evaluation.analysis.trigger_sources import ALL_TRIGGERS
        trigger_map = {t.param_name: t for t in ALL_TRIGGERS}
        cal_params = []
        for name in param_names:
            if name not in trigger_map:
                raise ValueError(f"Unknown parameter: {name}")
            t = trigger_map[name]
            cal_params.append(CalibrationParameter(
                name=t.name,
                param_name=t.param_name,
                bounds=t.value_range,
                true_value=t.default,
                unit=t.unit,
            ))
        return cls(cal_params)

    def generate_random(
        self, n_samples: int, seed: int = 42,
    ) -> List[Dict]:
        """Generate random parameter combinations within bounds."""
        rng = np.random.RandomState(seed)
        scenarios = []
        for _ in range(n_samples):
            overrides = {}
            for p in self.params:
                val = rng.uniform(p.bounds[0], p.bounds[1])
                overrides[p.param_name] = val
            scenarios.append(overrides)
        return scenarios

    def generate_sobol(
        self, n_samples: int, seed: int = 42,
    ) -> List[Dict]:
        """Generate Sobol quasi-random samples (better coverage)."""
        try:
            from scipy.stats.qmc import Sobol
        except ImportError:
            return self.generate_random(n_samples, seed)

        dim = len(self.params)
        # Sobol requires 2^m samples
        m = max(1, int(np.ceil(np.log2(n_samples))))
        actual_n = 2 ** m

        sampler = Sobol(dim, scramble=True, seed=seed)
        samples = sampler.random_base2(m)  # [actual_n, dim]

        scenarios = []
        for i in range(min(n_samples, actual_n)):
            overrides = {}
            for j, p in enumerate(self.params):
                lo, hi = p.bounds
                val = lo + samples[i, j] * (hi - lo)
                overrides[p.param_name] = val
            scenarios.append(overrides)
        return scenarios

    def generate_grid(
        self, points_per_dim: int = 3,
    ) -> List[Dict]:
        """Generate grid samples (exponential in dimension count)."""
        dim = len(self.params)
        if dim > 5:
            raise ValueError("Grid sampling impractical for >5 dimensions")

        linspaces = []
        for p in self.params:
            ls = np.linspace(p.bounds[0], p.bounds[1], points_per_dim)
            linspaces.append(ls)

        scenarios = []
        for combo in np.meshgrid(*linspaces, indexing='ij'):
            pass  # just setup
        for idx in np.ndindex(*([points_per_dim] * dim)):
            overrides = {}
            for j, p in enumerate(self.params):
                lo, hi = p.bounds
                val = lo + (idx[j] / (points_per_dim - 1)) * (hi - lo)
                overrides[p.param_name] = val
            scenarios.append(overrides)
        return scenarios

    def get_bounds_array(self):
        """Return (lower_bounds, upper_bounds) as numpy arrays."""
        lo = np.array([p.bounds[0] for p in self.params])
        hi = np.array([p.bounds[1] for p in self.params])
        return lo, hi

    def get_true_values(self):
        """Return true parameter values as numpy array."""
        return np.array([p.true_value for p in self.params])
