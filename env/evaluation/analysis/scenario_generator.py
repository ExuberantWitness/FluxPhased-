"""Scenario generator from trigger source configurations."""

from typing import Dict, List, Optional
import numpy as np

from .trigger_sources import TriggerSource, ALL_TRIGGERS


class ScenarioGenerator:
    """Generates evaluation scenarios from trigger source configurations."""

    def __init__(self, triggers: Optional[List[TriggerSource]] = None):
        self.triggers = triggers or ALL_TRIGGERS
        self._name_map = {t.param_name: t for t in self.triggers}

    def from_config(self, trigger_values: Dict[str, float]) -> dict:
        """Map trigger values to env constructor kwargs.

        Args:
            trigger_values: {trigger_name: value}
        Returns:
            dict of env constructor kwargs
        """
        kwargs = {}
        for name, value in trigger_values.items():
            if name in self._name_map:
                t = self._name_map[name]
                lo, hi = t.value_range
                kwargs[t.param_name] = float(np.clip(value, lo, hi))
        return kwargs

    def random_scenario(self, rng: Optional[np.random.RandomState] = None) -> dict:
        """Generate a random scenario within trigger ranges."""
        if rng is None:
            rng = np.random.RandomState()
        config = {}
        for t in self.triggers:
            lo, hi = t.value_range
            if isinstance(lo, int) and isinstance(hi, int):
                config[t.param_name] = int(rng.randint(lo, hi + 1))
            else:
                config[t.param_name] = float(rng.uniform(lo, hi))
        return config

    def sobol_scenarios(
        self,
        triggers: Optional[List[TriggerSource]] = None,
        n_samples: int = 64,
    ) -> List[dict]:
        """Generate Sobol-sampled scenario configs."""
        trig = triggers or self.triggers
        try:
            from SALib.sample import saltelli
            problem = {
                "num_vars": len(trig),
                "names": [t.param_name for t in trig],
                "bounds": [list(t.value_range) for t in trig],
            }
            samples = saltelli.sample(problem, n_samples, calc_second_order=False)
        except ImportError:
            n_total = n_samples * 2
            samples = np.random.uniform(size=(n_total, len(trig)))
            for i, t in enumerate(trig):
                lo, hi = t.value_range
                samples[:, i] = lo + samples[:, i] * (hi - lo)

        scenarios = []
        for row in samples:
            config = {}
            for i, t in enumerate(trig):
                config[t.param_name] = float(row[i])
            scenarios.append(config)
        return scenarios

    def adversarial_pair(self, base_config: dict) -> tuple:
        """Generate asymmetric red/blue configs for adversarial evaluation.

        Returns two configs where one team has advantage.
        """
        config_a = dict(base_config)
        config_b = dict(base_config)

        # Swap some parameters to create asymmetry
        if "tx_power_w" in config_a:
            config_a["tx_power_w"] = base_config["tx_power_w"] * 1.5
            config_b["tx_power_w"] = base_config["tx_power_w"] * 0.7
        if "target_rcs_dbsm" in config_a:
            config_a["target_rcs_dbsm"] = base_config.get("target_rcs_dbsm", 20) + 5
            config_b["target_rcs_dbsm"] = base_config.get("target_rcs_dbsm", 20) - 5

        return config_a, config_b
