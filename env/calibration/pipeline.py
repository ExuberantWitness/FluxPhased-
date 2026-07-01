"""Calibration pipeline orchestrator.

Ties together scenario selection, reference data, simulation runner,
parameter estimation, and reporting.
"""

import os
import torch
from typing import List, Optional

from .scenario_selector import ScenarioSelector, CalibrationParameter
from .reference_data import ReferenceDataLoader, ReferenceDataset
from .runner import CalibrationRunner
from .estimator import ParameterEstimator, EstimationResult
from .report import CalibrationReport


class CalibrationPipeline:
    """Full sim2real calibration workflow.

    Usage:
        pipeline = CalibrationPipeline(
            base_env_kwargs={...},
            calibratable_param_names=["noise_figure_db", "target_rcs_dbsm"],
            method="de",
        )
        result = pipeline.run()
        pipeline.report(result)
    """

    def __init__(
        self,
        base_env_kwargs: dict,
        calibratable_param_names: List[str],
        method: str = "de",
        n_scenarios: int = 10,
        reference_data_path: Optional[str] = None,
        output_dir: str = "calibration_output",
        seed: int = 42,
    ):
        self.base_kwargs = base_env_kwargs
        self.param_names = calibratable_param_names
        self.method = method
        self.n_scenarios = n_scenarios
        self.ref_path = reference_data_path
        self.output_dir = output_dir
        self.seed = seed

        self.selector = ScenarioSelector.from_trigger_sources(calibratable_param_names)
        self.reporter = CalibrationReport()

    def run(self) -> EstimationResult:
        """Execute the full calibration pipeline."""
        print(f"[Calibration] Parameters: {self.param_names}")
        print(f"[Calibration] Method: {self.method}")

        # Step 1: Generate or load reference data
        if self.ref_path and os.path.exists(self.ref_path):
            print(f"[Calibration] Loading reference data from {self.ref_path}")
            ref_data = ReferenceDataLoader.load(self.ref_path)
        else:
            print(f"[Calibration] Generating synthetic reference data ({self.n_scenarios} scenarios)")
            overrides = self.selector.generate_sobol(self.n_scenarios, seed=self.seed)
            ref_data = ReferenceDataLoader.generate_synthetic(
                true_env_kwargs=self.base_kwargs,
                calibratable_overrides=overrides,
                n_scenarios=self.n_scenarios,
                seed=self.seed,
            )
            # Save for reuse
            ref_path = os.path.join(self.output_dir, 'reference_data.pt')
            ReferenceDataLoader.save(ref_data, ref_path)

        # Step 2: Create runner
        runner = CalibrationRunner(
            base_env_kwargs=self.base_kwargs,
            reference_range=ref_data.range_measurements,
            reference_snr=ref_data.snr_measurements,
        )

        # Step 3: Run estimation
        estimator = ParameterEstimator(runner, self.selector)
        result = estimator.estimate(
            method=self.method,
            max_evals=50,  # keep small for GPU time
            seed=self.seed,
        )

        # Step 4: Generate report
        self.reporter.save_report(result, os.path.join(self.output_dir, 'calibration_report.md'))
        try:
            self.reporter.plot_convergence(result, os.path.join(self.output_dir, 'convergence.png'))
        except Exception:
            pass

        return result

    def report(self, result: EstimationResult):
        """Print summary of estimation result."""
        print("\n" + "=" * 60)
        print("Calibration Result")
        print("=" * 60)
        for name in result.estimated_params:
            true_v = result.true_params.get(name, float('nan'))
            est_v = result.estimated_params[name]
            err = abs(est_v - true_v)
            print(f"  {name}: true={true_v:.4f}, est={est_v:.4f}, error={err:.4f}")
        print(f"  Method: {result.method}, Evals: {result.n_evaluations}")
        print(f"  Converged: {result.success}")
        print(f"  Report saved to: {self.output_dir}/")
