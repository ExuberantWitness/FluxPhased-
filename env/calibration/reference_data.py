"""Reference data loader for calibration.

Loads "real" measurement data from files, or generates synthetic ground truth
by running the simulation with known "true" parameters.
"""

import os
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ReferenceDataset:
    """Container for calibration reference measurements."""
    range_measurements: Optional[torch.Tensor] = None   # [n_scenarios, n_radars] expected range (m)
    snr_measurements: Optional[torch.Tensor] = None     # [n_scenarios, n_radars] expected SNR (dB)
    detection_probs: Optional[torch.Tensor] = None       # [n_scenarios, n_radars] P_detect
    doppler_measurements: Optional[torch.Tensor] = None  # [n_scenarios, n_radars] doppler (Hz)
    scenario_params: Optional[list] = None               # list of param dicts per scenario
    metadata: dict = field(default_factory=dict)


class ReferenceDataLoader:
    """Load or generate calibration reference data."""

    @staticmethod
    def load(path: str) -> ReferenceDataset:
        """Load reference data from a .pt or .npz file."""
        if path.endswith('.pt'):
            data = torch.load(path, map_location='cpu')
            return ReferenceDataset(
                range_measurements=data.get('range'),
                snr_measurements=data.get('snr'),
                detection_probs=data.get('detection_probs'),
                doppler_measurements=data.get('doppler'),
                scenario_params=data.get('scenario_params'),
                metadata=data.get('metadata', {}),
            )
        elif path.endswith('.npz'):
            npz = np.load(path, allow_pickle=True)
            def _to_tensor(key):
                arr = npz.get(key)
                return torch.from_numpy(arr) if arr is not None else None
            return ReferenceDataset(
                range_measurements=_to_tensor('range'),
                snr_measurements=_to_tensor('snr'),
                detection_probs=_to_tensor('detection_probs'),
                doppler_measurements=_to_tensor('doppler'),
                scenario_params=npz.get('scenario_params', None),
            )
        else:
            raise ValueError(f"Unsupported format: {path}")

    @staticmethod
    def save(dataset: ReferenceDataset, path: str):
        """Save reference data to a .pt file."""
        data = {}
        if dataset.range_measurements is not None:
            data['range'] = dataset.range_measurements.cpu()
        if dataset.snr_measurements is not None:
            data['snr'] = dataset.snr_measurements.cpu()
        if dataset.detection_probs is not None:
            data['detection_probs'] = dataset.detection_probs.cpu()
        if dataset.doppler_measurements is not None:
            data['doppler'] = dataset.doppler_measurements.cpu()
        if dataset.scenario_params is not None:
            data['scenario_params'] = dataset.scenario_params
        data['metadata'] = dataset.metadata
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(data, path)

    @staticmethod
    def generate_synthetic(
        true_env_kwargs: dict,
        calibratable_overrides: list = None,
        n_scenarios: int = 10,
        steps_per_scenario: int = 3,
        seed: int = 42,
    ) -> ReferenceDataset:
        """Generate synthetic reference data by running simulation with true params.

        Args:
            true_env_kwargs: MFARVecEnv kwargs for the "true" simulation
            calibratable_overrides: list of param dicts to vary per scenario
            n_scenarios: number of calibration scenarios
            steps_per_scenario: simulation steps per scenario
            seed: random seed
        Returns:
            ReferenceDataset with ground-truth measurements
        """
        from ..gpu.vec_mfar_env import MFARVecEnv
        from ..evaluation.collectors.ground_truth import GroundTruthComputer

        torch.manual_seed(seed)
        np.random.seed(seed)

        all_ranges = []
        all_snrs = []

        for i in range(n_scenarios):
            # Override params for this scenario
            kwargs = dict(true_env_kwargs)
            if calibratable_overrides and i < len(calibratable_overrides):
                kwargs.update(calibratable_overrides[i])

            env = MFARVecEnv(**kwargs)
            env.reset()
            result = env.step()
            torch.cuda.synchronize()

            gt = GroundTruthComputer(env)
            gt_result = gt.compute()

            all_ranges.append(gt_result['expected_range_m'].cpu())
            all_snrs.append(gt_result['expected_snr_db'].cpu())

            del env
            torch.cuda.empty_cache()

        dataset = ReferenceDataset(
            range_measurements=torch.stack(all_ranges),
            snr_measurements=torch.stack(all_snrs),
            scenario_params=calibratable_overrides,
            metadata={
                'n_scenarios': n_scenarios,
                'steps_per_scenario': steps_per_scenario,
                'true_env_kwargs': {k: v for k, v in true_env_kwargs.items()
                                    if not isinstance(v, torch.Tensor)},
                'seed': seed,
            },
        )
        return dataset
