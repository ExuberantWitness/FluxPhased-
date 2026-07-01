"""Dense reward shaping from MFARVecEnv step outputs.

Computes per-task intermediate rewards from existing simulation data:
- Detection SNR and coverage (from spectrum + matched filtering)
- Jamming effectiveness (from interference power)
- Communication reliability (from BPSK CRC results)
- Reconnaissance intelligence (from spectrum energy detection)

All rewards are shaped to [0, 1] range, then scaled by configurable weights.
"""

import torch
import numpy as np


TASK_RECON = 0
TASK_DETECT = 1
TASK_JAM = 2
TASK_COMM = 3


class DenseRewardShaper:
    """Compute dense intermediate rewards from MFARVecEnv step() output dict."""

    def __init__(
        self,
        detect_snr_weight: float = 0.1,
        detect_coverage_weight: float = 0.05,
        jam_effectiveness_weight: float = 0.1,
        comm_reliability_weight: float = 0.05,
        recon_intel_weight: float = 0.03,
        beam_accuracy_weight: float = 0.02,
        stealth_weight: float = 0.1,
        missile_guidance_weight: float = 0.02,
        snr_threshold_db: float = 10.0,
        device: str = "cuda",
    ):
        self.detect_snr_weight = detect_snr_weight
        self.detect_coverage_weight = detect_coverage_weight
        self.jam_effectiveness_weight = jam_effectiveness_weight
        self.comm_reliability_weight = comm_reliability_weight
        self.recon_intel_weight = recon_intel_weight
        self.beam_accuracy_weight = beam_accuracy_weight
        self.stealth_weight = stealth_weight
        self.missile_guidance_weight = missile_guidance_weight
        self.snr_threshold_db = snr_threshold_db
        self.device = device

    def __call__(self, step_output: dict) -> dict:
        """Compute shaped rewards from one env step.

        Args:
            step_output: dict from MFARVecEnv.step(), must contain:
                - spectrum: [E, R, N, P, n_bins] float32
                - comm_data: [E, R, N, 2] float32
                - task_ids: [E, R, N] int64
                - comm_crc_ok: [E, n_teams] bool (optional)
                - channel_params: tuple (delay, doppler, gain) (optional)
                - detect_params: [E, R, N, 3] (optional)
                - jam_params: [E, R, N, 3] (optional)
        Returns:
            dict with per-radar shaped rewards [E, R] and individual components.
        """
        spectrum = step_output["spectrum"]       # [E, R, N, P, n_bins]
        task_ids = step_output["task_ids"]       # [E, R, N]
        E, R, N = task_ids.shape
        dev = spectrum.device

        # --- Detection reward ---
        detect_reward = self._detect_reward(spectrum, task_ids)

        # --- Jamming reward ---
        jam_reward = self._jam_reward(spectrum, task_ids)

        # --- Communication reward ---
        comm_reward = self._comm_reward(step_output, task_ids)

        # --- Reconnaissance reward ---
        recon_reward = self._recon_reward(spectrum, task_ids)

        # --- Stealth penalty (being intercepted by enemy) ---
        stealth_penalty = self._stealth_penalty(step_output)  # [E, R]

        # --- Beam accuracy reward (pointing at target) ---
        beam_accuracy_reward = self._beam_accuracy_reward(step_output)  # [E, R]

        # --- Detect coverage reward (spatial diversity) ---
        detect_coverage_reward = self._detect_coverage_reward(step_output)  # [E, R]

        # --- Missile guidance reward (kill progress) ---
        missile_guidance_reward = self._missile_guidance_reward(step_output)  # [E, R]

        total = (
            detect_reward * self.detect_snr_weight
            + jam_reward * self.jam_effectiveness_weight
            + comm_reward * self.comm_reliability_weight
            + recon_reward * self.recon_intel_weight
            + beam_accuracy_reward * self.beam_accuracy_weight
            + detect_coverage_reward * self.detect_coverage_weight
            + missile_guidance_reward * self.missile_guidance_weight
            - stealth_penalty * self.stealth_weight
        )

        return {
            "detect_reward": detect_reward,
            "jam_reward": jam_reward,
            "comm_reward": comm_reward,
            "recon_reward": recon_reward,
            "stealth_penalty": stealth_penalty,
            "beam_accuracy_reward": beam_accuracy_reward,
            "detect_coverage_reward": detect_coverage_reward,
            "missile_guidance_reward": missile_guidance_reward,
            "total_shaped": total,
        }

    def _detect_reward(self, spectrum: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Detection quality: SNR above threshold, normalized by number of detect elements."""
        E, R, N, P, B = spectrum.shape
        detect_mask = (task_ids == TASK_DETECT)  # [E, R, N]
        n_detect = detect_mask.sum(dim=-1).clamp(min=1).float()  # [E, R]

        # Peak power across frequency bins for detect elements
        detect_spectrum = spectrum * detect_mask.unsqueeze(-1).unsqueeze(-1).float()
        peak_power = detect_spectrum.amax(dim=-1).amax(dim=-1)  # [E, R, N]

        # Noise floor: median of non-peak bins
        noise_floor = spectrum.median(dim=-1).values.median(dim=-1).values.clamp(min=1e-30)

        # SNR in dB for detect elements
        snr_db = 10.0 * torch.log10(peak_power.clamp(min=1e-30) / noise_floor.clamp(min=1e-30))
        snr_db = snr_db * detect_mask.float()

        # Reward: fraction of detect elements above threshold
        above_thresh = (snr_db > self.snr_threshold_db).float()
        coverage = above_thresh.sum(dim=-1) / n_detect  # [E, R]

        # Average SNR above threshold
        avg_snr = (snr_db - self.snr_threshold_db).clamp(min=0).sum(dim=-1) / (n_detect * 20.0)

        return coverage * 0.5 + avg_snr * 0.5

    def _jam_reward(self, spectrum: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Jam effectiveness: total power emitted by jam elements, normalized."""
        E, R, N, P, B = spectrum.shape
        jam_mask = (task_ids == TASK_JAM).float()  # [E, R, N]
        n_jam = jam_mask.sum(dim=-1).clamp(min=1)  # [E, R]

        # Jam reward is based on proportion of elements allocated to jamming
        # (proxy for jamming investment — actual effectiveness requires cross-team measurement)
        jam_fraction = jam_mask.sum(dim=-1) / N  # [E, R]

        # Energy in spectrum for jam elements (transmitted power proxy)
        jam_energy = (spectrum.mean(dim=-1).mean(dim=-1) * jam_mask).sum(dim=-1)
        jam_energy_norm = jam_energy / (jam_energy.amax() + 1e-10)

        return 0.3 * jam_fraction + 0.7 * jam_energy_norm

    def _comm_reward(self, step_output: dict, task_ids: torch.Tensor) -> torch.Tensor:
        """Communication reliability: CRC pass rate for own team's comm link."""
        E, R, N = task_ids.shape
        dev = task_ids.device

        crc_ok = step_output.get("comm_crc_ok")
        if crc_ok is None:
            return torch.zeros(E, R, device=dev)

        # crc_ok: [E, n_teams] bool — expand to [E, R]
        team_id = torch.tensor(
            [i // (R // 2) for i in range(R)], device=dev,
        )  # [R]
        crc_per_radar = crc_ok[:, team_id].float()  # [E, R]

        # Weight by comm element fraction
        comm_mask = (task_ids == TASK_COMM).float()
        comm_fraction = comm_mask.sum(dim=-1) / task_ids.shape[-1]  # [E, R]

        return crc_per_radar * (0.5 + 0.5 * comm_fraction)

    def _recon_reward(self, spectrum: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Reconnaissance intelligence: energy detected by recon elements."""
        E, R, N, P, B = spectrum.shape
        recon_mask = (task_ids == TASK_RECON).float()  # [E, R, N]
        n_recon = recon_mask.sum(dim=-1).clamp(min=1)  # [E, R]

        # Total energy received by recon elements
        recon_energy = (spectrum.mean(dim=-1).mean(dim=-1) * recon_mask).sum(dim=-1)
        max_energy = recon_energy.amax() + 1e-10

        return (recon_energy / max_energy).clamp(0, 1)

    def _stealth_penalty(self, step_output: dict) -> torch.Tensor:
        """Penalty for being intercepted by enemy radars.

        Uses cross_team_intercept from the simulator which estimates SNR
        at enemy receivers per task type. Higher SNR = easier to detect.

        Task-type weights (higher = more easily intercepted):
          - jam: high-power noise, most visible → 0.15
          - comm: BPSK detectable → 0.05
          - detect: LPI waveform, hard to intercept → 0.02
          - recon: purely passive, no penalty → 0.0

        Returns:
            [E, R] penalty per env per radar, in [0, 1].
        """
        intercept = step_output.get("cross_team_intercept")
        if intercept is None:
            return torch.zeros(
                step_output["spectrum"].shape[0],
                step_output["spectrum"].shape[1],
                device=step_output["spectrum"].device,
            )

        E, R = step_output["spectrum"].shape[:2]
        dev = step_output["spectrum"].device

        # Detect number of teams from intercept keys
        n_teams = sum(1 for k in intercept if k.endswith("_intercept_detail"))
        if n_teams < 2:
            return torch.zeros(E, R, device=dev)

        r_per_team = R // n_teams
        weights = torch.tensor([0.02, 0.15, 0.05], device=dev)  # detect, jam, comm

        penalty = torch.zeros(E, R, device=dev)
        for team in range(n_teams):
            detail = intercept.get(f"team{team}_intercept_detail")
            if detail is None:
                continue
            team_penalty = (detail * weights).sum(dim=-1)  # [E]
            r0, r1 = team * r_per_team, (team + 1) * r_per_team
            penalty[:, r0:r1] = team_penalty.unsqueeze(-1).expand(E, r_per_team)

        return torch.sigmoid(penalty)

    def _beam_accuracy_reward(self, step_output: dict) -> torch.Tensor:
        """Reward beam pointing accuracy via actual array gain in target direction.

        Uses `channel_params.gain_linear` which measures post-beamforming
        voltage gain toward target 0. This is physically correct: higher gain
        means the phased array beam is better aligned with the target direction,
        regardless of coordinate system conventions.

        Normalized by sqrt(n_elem) — the theoretical maximum coherent gain
        for an N-element array.

        Returns:
            [E, R] reward per env per radar, in [0, 1].
        """
        # Use detect-specific channel params (computed with detect beam directions)
        # rather than the global-mean channel_params (diluted by all tasks).
        channel_params = (step_output.get("detect_channel_params")
                          or step_output.get("channel_params"))
        E = step_output["spectrum"].shape[0]
        R = step_output["spectrum"].shape[1]
        dev = step_output["spectrum"].device

        if channel_params is None:
            return torch.zeros(E, R, device=dev)

        # channel_params is dict with keys: delay_samples, doppler_hz, gain_linear
        # gain_linear: [E, R] — per-radar voltage gain toward target
        if isinstance(channel_params, dict):
            gain = channel_params.get("gain_linear")
        elif isinstance(channel_params, (tuple, list)) and len(channel_params) >= 3:
            gain = channel_params[2]
        else:
            return torch.zeros(E, R, device=dev)

        if gain is None:
            return torch.zeros(E, R, device=dev)

        # Normalize by sqrt(n_elem) for max theoretical coherent gain
        n_elem = step_output["task_ids"].shape[-1]  # N
        max_gain = max(float(n_elem) ** 0.5, 1.0)

        # gain_linear is [E, R] — per-radar voltage gain toward target 0
        reward = (gain.float().clamp(min=0) / max_gain).clamp(0, 1)

        # Only give reward when there are detect elements active
        task_ids = step_output["task_ids"]  # [E, R, N]
        has_detect = (task_ids == TASK_DETECT).any(dim=-1).float()  # [E, R]
        reward = reward * has_detect

        return reward

    def _detect_coverage_reward(self, step_output: dict) -> torch.Tensor:
        """Reward spatial diversity of detect beam directions.

        Penalizes all detect elements pointing at the same direction.
        Encourages the radar to search across multiple angular sectors.

        Returns:
            [E, R] reward per env per radar, in [0, 1].
        """
        task_ids = step_output.get("task_ids")     # [E, R, N]
        beam_az = step_output.get("beam_az")       # [E, R]
        beam_el = step_output.get("beam_el")       # [E, R]

        if task_ids is None or beam_az is None:
            E = step_output["spectrum"].shape[0]
            R = step_output["spectrum"].shape[1]
            return torch.zeros(E, R, device=step_output["spectrum"].device)

        E, R, N = task_ids.shape
        dev = task_ids.device
        detect_mask = (task_ids == TASK_DETECT).float()  # [E, R, N]
        n_detect = detect_mask.sum(dim=-1).clamp(min=1)  # [E, R]

        # Variance of beam directions across detect elements per radar.
        # We can't directly get per-element beam directions from step_output,
        # so use a proxy: reward detect element count diversity across sub-regions.
        # Higher reward when detect elements are spread across the array
        # (which maps to different beam directions via the phased array).
        detect_positions = detect_mask.sum(dim=-1)       # [E, R] — total detect count
        # Normalize: reward is higher when detect count is in a sweet spot
        # (not all elements, not zero — distributed allocation)
        detect_frac = detect_positions / N               # [E, R]
        # Reward peaks at ~50% allocation (balanced with other tasks)
        coverage = 1.0 - 4.0 * (detect_frac - 0.25) ** 2  # parabola, max=1 at 0.25
        coverage = coverage.clamp(0, 1)

        return coverage

    def _missile_guidance_reward(self, step_output: dict) -> torch.Tensor:
        """Reward progress toward kill: missile is flying and approaching target.

        Checks if own team's missile is in flight and how close it is to the
        target. This bridges the gap between detect success and kill reward.

        Returns:
            [E, R] reward per env per radar, in [0, 1].
        """
        missile_pos = step_output.get("missile_pos")  # [E, n_teams, 3]
        task_fingerprint = step_output.get("task_fingerprint")  # [E, n_teams, 4]

        if missile_pos is None:
            E = step_output["spectrum"].shape[0]
            R = step_output["spectrum"].shape[1]
            return torch.zeros(E, R, device=step_output["spectrum"].device)

        E = missile_pos.shape[0]
        R = step_output["spectrum"].shape[1]
        dev = missile_pos.device
        n_teams = missile_pos.shape[1]
        r_per_team = R // n_teams

        # Missile is "in flight" if its position has significant z component
        # (launched missiles move; pre-launch pos is at launch site)
        in_flight = (missile_pos[:, :, 2].abs() > 100.0).float()  # [E, n_teams]

        # Target direction: missile for team 0 heads toward +y (blue territory),
        # missile for team 1 heads toward -y (red territory).
        # Reward based on missile z-progress (proxy for distance to target):
        # team 0 missile z increases as it flies; team 1 z decreases.
        missile_z = missile_pos[:, :, 2]  # [E, n_teams]
        # Normalize to [0,1]: team 0 → higher z better, team 1 → lower z better
        z_min = missile_z.min()
        z_max = missile_z.max()
        z_range = max(z_max - z_min, 1.0)
        progress_0 = (missile_z[:, 0] - z_min) / z_range  # [E]
        progress_1 = (z_max - missile_z[:, 1]) / z_range  # [E]

        guidance = torch.zeros(E, R, device=dev)
        for t in range(n_teams):
            progress = progress_0 if t == 0 else progress_1
            r0, r1 = t * r_per_team, (t + 1) * r_per_team
            guidance[:, r0:r1] = (in_flight[:, t] * progress).unsqueeze(-1).expand(E, r_per_team)

        return guidance.clamp(0, 1)
