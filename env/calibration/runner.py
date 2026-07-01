"""Calibration runner: run simulation with candidate parameters and collect outputs."""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CalibrationResult:
    """Result from a single calibration scenario run."""
    params: dict
    range_error: float
    snr_error: float
    total_error: float
    n_steps: int


class CalibrationRunner:
    """Run simulation with parameter overrides and compute residuals."""

    def __init__(
        self,
        base_env_kwargs: dict,
        reference_range: torch.Tensor = None,
        reference_snr: torch.Tensor = None,
        steps_per_run: int = 3,
        device: str = "cuda",
    ):
        """
        Args:
            base_env_kwargs: base MFARVecEnv kwargs
            reference_range: [n_scenarios, ...] expected range from reference
            reference_snr: [n_scenarios, ...] expected SNR from reference
            steps_per_run: simulation steps per calibration run
            device: torch device
        """
        self.base_kwargs = dict(base_env_kwargs)
        self.reference_range = reference_range
        self.reference_snr = reference_snr
        self.steps_per_run = steps_per_run
        self.device = device

    def run_scenario(
        self,
        param_overrides: dict,
        scenario_idx: int = 0,
    ) -> CalibrationResult:
        """Run one scenario with parameter overrides.

        Args:
            param_overrides: dict of param_name -> value to override
            scenario_idx: which reference scenario to compare against
        Returns:
            CalibrationResult with residuals
        """
        from ..gpu.vec_mfar_env import MFARVecEnv
        from ..evaluation.collectors.ground_truth import GroundTruthComputer

        kwargs = dict(self.base_kwargs)
        kwargs.update(param_overrides)

        env = MFARVecEnv(**kwargs)
        env.reset()
        result = env.step()
        torch.cuda.synchronize()

        gt = GroundTruthComputer(env)
        gt_result = gt.compute()

        sim_range = gt_result['expected_range_m']
        sim_snr = gt_result['expected_snr_db']

        range_error = 0.0
        snr_error = 0.0

        if self.reference_range is not None and scenario_idx < len(self.reference_range):
            ref = self.reference_range[scenario_idx]
            range_error = (sim_range.cpu() - ref.cpu()).abs().mean().item()

        if self.reference_snr is not None and scenario_idx < len(self.reference_snr):
            ref = self.reference_snr[scenario_idx]
            # Clamp SNR to avoid log-space issues
            sim_clamped = sim_snr.cpu().clamp(-100, 100)
            ref_clamped = ref.cpu().clamp(-100, 100)
            snr_error = (sim_clamped - ref_clamped).abs().mean().item()

        total_error = range_error + snr_error

        cal_result = CalibrationResult(
            params=param_overrides,
            range_error=range_error,
            snr_error=snr_error,
            total_error=total_error,
            n_steps=1,
        )

        del env
        torch.cuda.empty_cache()

        return cal_result

    def run_batch(
        self,
        scenarios: List[dict],
    ) -> List[CalibrationResult]:
        """Run multiple scenarios sequentially."""
        results = []
        for i, overrides in enumerate(scenarios):
            r = self.run_scenario(overrides, scenario_idx=i)
            results.append(r)
        return results

    def objective(self, params_vector: np.ndarray, selector) -> float:
        """Objective function for optimizer.

        Args:
            params_vector: numpy array of parameter values (ordered by selector.params)
            selector: ScenarioSelector for mapping vector -> dict
        Returns:
            scalar residual (sum of squared errors)
        """
        overrides = {}
        for j, p in enumerate(selector.params):
            overrides[p.param_name] = float(params_vector[j])

        result = self.run_scenario(overrides, scenario_idx=0)
        return result.range_error ** 2 + result.snr_error ** 2
