"""Compute expected detection parameters from simulation state.

Reuses VecChannel.compute_params_batch for consistency with the simulation.
Simulation IS ground truth — no physical calibration needed.
"""

import numpy as np
import torch

SPEED_OF_LIGHT = 299792458.0


class GroundTruthComputer:
    """Computes expected range/doppler/SNR from radar and target positions.

    All outputs are GPU tensors matching env dimensions.
    """

    def __init__(self, env):
        self.env = env
        self.channel = env.channel
        self.fs = env.fs
        self.prf = env.prf
        self.n_pulses = env.n_pulses
        self.n_bins = env.n_bins
        self.range_res = SPEED_OF_LIGHT / (2.0 * self.fs)

    def compute(
        self,
        radar_pos: torch.Tensor = None,
        radar_vel: torch.Tensor = None,
        target_pos: torch.Tensor = None,
        target_vel: torch.Tensor = None,
    ) -> dict:
        """Compute expected detection parameters for all radar-target pairs.

        Args:
            radar_pos: [E, R, 3] if None, uses env.radar_pos
            radar_vel: [E, R, 3] if None, uses env.radar_vel
            target_pos: [E, n_targets, 3] if None, uses env.target_pos
            target_vel: [E, n_targets, 3] if None, uses env.target_vel
        Returns:
            dict with GPU tensors:
              expected_range_m:    [E, R, n_targets]
              expected_range_bin:  [E, R, n_targets] int
              expected_velocity:   [E, R, n_targets] m/s (radial)
              expected_doppler_hz: [E, R, n_targets]
              expected_snr_db:     [E, R, n_targets]
        """
        env = self.env
        if radar_pos is None:
            radar_pos = env.radar_pos
        if radar_vel is None:
            radar_vel = env.radar_vel
        if target_pos is None:
            target_pos = env.target_pos
        if target_vel is None:
            target_vel = env.target_vel

        E, R = radar_pos.shape[:2]
        n_targets = target_pos.shape[1]
        device = radar_pos.device

        # Per-target: compute channel params
        all_range = []
        all_range_bin = []
        all_velocity = []
        all_doppler = []
        all_snr = []

        for t in range(n_targets):
            delay_s, doppler_hz, gain_linear = self.channel.compute_params_batch(
                radar_pos, radar_vel,
                target_pos[:, t], target_vel[:, t],
                tx_power_w=env.tx_power_w,
                rcs_dbsm=env.target_rcs_dbsm,
                array_directivity_db=env.array.directivity_db,
            )
            # delay_s is [E, R] — these are delay_samples from compute_params_batch
            delay_samples = delay_s  # already in samples
            range_m = delay_samples / (2.0 * self.fs) * SPEED_OF_LIGHT  # two-way → range
            range_bin = (delay_samples / 2.0).long().clamp(0, self.n_bins - 1)

            # Radial velocity from doppler
            velocity = doppler_hz * SPEED_OF_LIGHT / (2.0 * env.fc)

            # SNR: gain_linear includes path gain; noise power from channel
            signal_power = gain_linear ** 2
            noise_pwr = torch.tensor(self.channel.noise_power_linear, device=signal_power.device)
            snr_linear = signal_power / noise_pwr.clamp(min=1e-30)
            snr_db = 10.0 * torch.log10(snr_linear.clamp(min=1e-30))

            all_range.append(range_m)
            all_range_bin.append(range_bin)
            all_velocity.append(velocity)
            all_doppler.append(doppler_hz)
            all_snr.append(snr_db)

        return {
            "expected_range_m": torch.stack(all_range, dim=-1),
            "expected_range_bin": torch.stack(all_range_bin, dim=-1),
            "expected_velocity": torch.stack(all_velocity, dim=-1),
            "expected_doppler_hz": torch.stack(all_doppler, dim=-1),
            "expected_snr_db": torch.stack(all_snr, dim=-1),
        }
