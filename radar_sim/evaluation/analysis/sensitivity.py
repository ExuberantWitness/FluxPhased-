"""BN-Sobol sensitivity analysis for trigger source ranking.

Uses SALib for Sobol index computation on CPU. The GPU env generates
the data; Sobol analysis runs on CPU with numpy arrays.
"""

from typing import Callable, List, Optional

import numpy as np

from .trigger_sources import TriggerSource


class SensitivityAnalyzer:
    """BN-Sobol global sensitivity analysis for evaluation triggers.

    Maps each trigger source to a SALib problem variable, generates
    Sobol quasi-random samples, evaluates each sample as an env config,
    and computes first-order and total-order Sobol indices.
    """

    def __init__(
        self,
        triggers: List[TriggerSource],
        metric_fn: Callable,
        env_factory: Callable,
        n_steps: int = 50,
    ):
        """
        Args:
            triggers: list of TriggerSource to analyze
            metric_fn: callable(env) -> float, computes the target metric
            env_factory: callable(**kwargs) -> MFARVecEnv, creates env with overrides
            n_steps: max steps per evaluation episode
        """
        self.triggers = triggers
        self.metric_fn = metric_fn
        self.env_factory = env_factory
        self.n_steps = n_steps

    def _build_problem(self) -> dict:
        """Build SALib problem spec from trigger sources."""
        names = [t.param_name for t in self.triggers]
        bounds = [list(t.value_range) for t in self.triggers]
        return {
            "num_vars": len(self.triggers),
            "names": names,
            "bounds": bounds,
        }

    def _generate_sobol_samples(self, n_base: int = 64) -> np.ndarray:
        """Generate Sobol quasi-random parameter matrix."""
        try:
            from SALib.sample import saltelli
            problem = self._build_problem()
            samples = saltelli.sample(problem, n_base, calc_second_order=False)
            return samples
        except ImportError:
            # Fallback: uniform random sampling
            n_vars = len(self.triggers)
            n_samples = n_base * (2 * n_vars + 2)
            samples = np.random.uniform(size=(n_samples, n_vars))
            for i, t in enumerate(self.triggers):
                lo, hi = t.value_range
                samples[:, i] = lo + samples[:, i] * (hi - lo)
            return samples

    def _evaluate_sample(self, sample: np.ndarray) -> float:
        """Evaluate one parameter sample as an env config."""
        kwargs = {}
        for i, trigger in enumerate(self.triggers):
            kwargs[trigger.param_name] = float(sample[i])

        try:
            env = self.env_factory(**kwargs)
            env.reset()
            result = env.step()
            value = self.metric_fn(result)
            del env
            return value
        except Exception:
            return float("nan")

    def run(self, n_base_samples: int = 64) -> dict:
        """Run full Sobol sensitivity analysis.

        Args:
            n_base_samples: base number of Sobol samples (total = N*(2D+2))
        Returns:
            dict with S1, ST, ranking
        """
        problem = self._build_problem()
        samples = self._generate_sobol_samples(n_base_samples)

        # Evaluate all samples
        outputs = np.array([
            self._evaluate_sample(s) for s in samples
        ])

        # Remove NaN
        valid = ~np.isnan(outputs)
        if valid.sum() < 10:
            return {
                "S1": {},
                "ST": {},
                "ranking": [],
                "n_valid": int(valid.sum()),
                "n_total": len(outputs),
            }

        # Sobol analysis
        try:
            from SALib.analyze import sobol
            samples_v = samples[valid]
            outputs_v = outputs[valid]
            si = sobol.analyze(problem, outputs_v, calc_second_order=False)
            s1 = {name: float(si["S1"][i]) for i, name in enumerate(problem["names"])}
            st = {name: float(si["ST"][i]) for i, name in enumerate(problem["names"])}
        except (ImportError, Exception):
            # Fallback: simple variance-based ranking
            s1 = {}
            st = {}
            for i, t in enumerate(self.triggers):
                vals = outputs[valid]
                var_total = float(np.var(vals))
                if var_total > 0:
                    # Bin by quartiles of this variable
                    feature = samples[valid, i]
                    q33 = np.percentile(feature, 33)
                    q66 = np.percentile(feature, 66)
                    low = vals[feature <= q33]
                    mid = vals[(feature > q33) & (feature <= q66)]
                    high = vals[feature > q66]
                    between_var = float(np.var([low.mean(), mid.mean(), high.mean()]))
                    st[t.param_name] = between_var / var_total
                    s1[t.param_name] = between_var / var_total
                else:
                    st[t.param_name] = 0.0
                    s1[t.param_name] = 0.0

        # Ranking by total-order index
        ranking = sorted(st.keys(), key=lambda k: abs(st.get(k, 0)), reverse=True)

        return {
            "S1": s1,
            "ST": st,
            "ranking": ranking,
            "n_valid": int(valid.sum()),
            "n_total": len(outputs),
            "metric_mean": float(outputs[valid].mean()),
            "metric_std": float(outputs[valid].std()),
        }

    def high_value_scenarios(self, n_base: int = 64, top_k: int = 5) -> list:
        """Return parameter configs that maximize metric variance.

        These are the 'high-value' test scenarios from the document.
        """
        samples = self._generate_sobol_samples(n_base)
        outputs = np.array([self._evaluate_sample(s) for s in samples])

        # Scenarios with extreme metric values (high variance contribution)
        valid = ~np.isnan(outputs)
        if valid.sum() < top_k:
            return []

        samples_v = samples[valid]
        outputs_v = outputs[valid]

        # Find samples farthest from mean
        mean_val = outputs_v.mean()
        deviations = np.abs(outputs_v - mean_val)
        top_indices = np.argsort(deviations)[-top_k:][::-1]

        scenarios = []
        for idx in top_indices:
            config = {}
            for i, trigger in enumerate(self.triggers):
                config[trigger.name] = float(samples_v[idx, i])
            config["_metric_value"] = float(outputs_v[idx])
            scenarios.append(config)

        return scenarios
