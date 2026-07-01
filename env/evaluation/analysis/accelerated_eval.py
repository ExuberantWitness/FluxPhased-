"""Confidence-driven early-stopping accelerated evaluation.

Reduces evaluation cost by stopping once metric confidence interval
narrows below a threshold. Uses scipy for statistical computation.
"""

import numpy as np
from typing import Callable, Optional


class AcceleratedEvaluator:
    """Adaptive evaluation with confidence-based early stopping."""

    def __init__(
        self,
        env_factory: Callable,
        metric_fn: Callable,
        confidence: float = 0.95,
        half_width: float = 0.05,
        max_episodes: int = 500,
        min_episodes: int = 20,
    ):
        """
        Args:
            env_factory: callable() -> MFARVecEnv
            metric_fn: callable(env) -> float, computes target metric
            confidence: target confidence level (e.g. 0.95)
            half_width: target CI half-width (e.g. 0.05)
            max_episodes: maximum episodes to run
            min_episodes: minimum episodes before checking CI
        """
        self.env_factory = env_factory
        self.metric_fn = metric_fn
        self.confidence = confidence
        self.half_width = half_width
        self.max_episodes = max_episodes
        self.min_episodes = min_episodes

    def _compute_ci(self, values: list) -> tuple:
        """Compute confidence interval for current samples."""
        arr = np.array(values)
        n = len(arr)
        if n < 2:
            return (float("nan"), float("nan"))

        mean = arr.mean()
        std = arr.std(ddof=1)

        # t-distribution critical value
        try:
            from scipy.stats import t as t_dist
            t_crit = t_dist.ppf((1 + self.confidence) / 2, df=n - 1)
        except ImportError:
            # Fallback: normal approximation
            from math import erf, sqrt
            z = 1.96  # ~95% CI
            t_crit = z

        margin = t_crit * std / np.sqrt(n)
        return (mean - margin, mean + margin)

    def evaluate(
        self,
        policy_fn: Optional[Callable] = None,
        verbose: bool = False,
    ) -> dict:
        """Run adaptive evaluation with early stopping.

        Args:
            policy_fn: optional callable(env) -> actions, None for default
            verbose: print progress
        Returns:
            dict with metric stats and episode usage
        """
        values = []
        converged_at = -1

        for ep in range(self.max_episodes):
            try:
                env = self.env_factory()
                env.reset()
                result = env.step()
                value = self.metric_fn(result)
                del env
            except Exception:
                value = float("nan")

            if not np.isnan(value):
                values.append(value)

            # Check CI convergence after minimum episodes
            if len(values) >= self.min_episodes:
                ci_lo, ci_hi = self._compute_ci(values)
                if not np.isnan(ci_lo):
                    ci_width = ci_hi - ci_lo
                    if ci_width <= 2 * self.half_width:
                        converged_at = ep + 1
                        if verbose:
                            print(f"  Converged at episode {ep+1}, "
                                  f"CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
                        break

            if verbose and (ep + 1) % 50 == 0:
                mean = np.mean(values) if values else float("nan")
                print(f"  Episode {ep+1}: mean={mean:.4f}, n={len(values)}")

        n_used = len(values)
        ci_lo, ci_hi = self._compute_ci(values) if n_used >= 2 else (float("nan"), float("nan"))

        return {
            "metric_mean": float(np.mean(values)) if values else float("nan"),
            "metric_std": float(np.std(values)) if values else float("nan"),
            "metric_ci": (float(ci_lo), float(ci_hi)),
            "episodes_used": n_used,
            "episodes_attempted": ep + 1 if values else 0,
            "episodes_saved": self.max_episodes - n_used,
            "convergence_step": converged_at,
            "target_confidence": self.confidence,
            "target_half_width": self.half_width,
        }
