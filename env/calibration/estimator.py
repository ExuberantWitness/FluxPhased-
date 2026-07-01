"""Parameter estimator using scipy optimization.

Supports least_squares, differential_evolution, and L-BFGS-B.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable


@dataclass
class EstimationResult:
    """Result of parameter estimation."""
    estimated_params: dict          # param_name -> estimated_value
    true_params: dict               # param_name -> true_value (if known)
    covariance: Optional[np.ndarray] = None
    residuals: List[float] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)
    method: str = ""
    n_evaluations: int = 0
    success: bool = False
    message: str = ""


class ParameterEstimator:
    """Core optimization engine for parameter calibration."""

    def __init__(self, runner, selector):
        """
        Args:
            runner: CalibrationRunner instance
            selector: ScenarioSelector instance
        """
        self.runner = runner
        self.selector = selector

    def estimate(
        self,
        method: str = "de",
        initial_guess: dict = None,
        max_evals: int = 100,
        seed: int = 42,
    ) -> EstimationResult:
        """Run parameter estimation.

        Args:
            method: "ls" (least_squares), "de" (differential_evolution), or "lbfgsb"
            initial_guess: dict of param_name -> initial value (default: bounds midpoint)
            max_evals: maximum function evaluations
            seed: random seed
        Returns:
            EstimationResult
        """
        from scipy.optimize import least_squares, differential_evolution, minimize

        lo, hi = self.selector.get_bounds_array()
        true_vals = self.selector.get_true_values()
        eval_count = [0]
        history = []

        def objective(x):
            eval_count[0] += 1
            val = self.runner.objective(x, self.selector)
            history.append(val)
            return val

        if method == "ls":
            x0 = np.array([initial_guess.get(p.param_name, (l + h) / 2)
                           for p, l, h in zip(self.selector.params, lo, hi)])
            result = least_squares(
                lambda x: np.array([objective(x)]),
                x0,
                bounds=(lo, hi),
                max_nfev=max_evals,
            )
            estimated = result.x
            success = result.success
            msg = result.message if isinstance(result.message, str) else str(result.message)
            cov = None
            try:
                cov = np.linalg.inv(result.jac.T @ result.jac) * (result.cost / max(1, len(result.fun) - len(x0)))
            except Exception:
                pass

        elif method == "de":
            result = differential_evolution(
                objective,
                list(zip(lo, hi)),
                maxiter=max_evals,
                seed=seed,
                polish=True,
            )
            estimated = result.x
            success = result.success
            msg = result.message
            cov = None

        elif method == "lbfgsb":
            x0 = np.array([initial_guess.get(p.param_name, (l + h) / 2)
                           for p, l, h in zip(self.selector.params, lo, hi)])
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=list(zip(lo, hi)),
                options={'maxiter': max_evals},
            )
            estimated = result.x
            success = result.success
            msg = result.message
            cov = None
        else:
            raise ValueError(f"Unknown method: {method}")

        est_dict = {p.param_name: float(v) for p, v in zip(self.selector.params, estimated)}
        true_dict = {p.param_name: float(v) for p, v in zip(self.selector.params, true_vals)}

        return EstimationResult(
            estimated_params=est_dict,
            true_params=true_dict,
            covariance=cov,
            residuals=history,
            convergence_history=history,
            method=method,
            n_evaluations=eval_count[0],
            success=success,
            message=msg,
        )
