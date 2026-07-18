"""Two-team symmetric multifunction phased-array adversarial env.

Per TWOTEAM_MULTIFUNCTION_PLAN.md (commit 4329bae).

Scenario: two symmetric teams (A=red, B=blue). Each team has:
  - 2 phased-array radars (25 subarrays each, single aperture, 5x5)
  - 1 commander (policy or rule-based)
  - 1 laser

Each radar's 25 subarrays are time-shared per step across 4 functions:
  detect / track / jam / comm   (D2=A: soft fractions via softmax)

Win (D1=B): laser fires at enemy radars. Each kill degrades the enemy team
from multi-baseline fusion to single-station → CRLB worsens → cascading
vulnerability. Commander is NOT a separate target.

Geometry (D3=A+B):
  MIRROR_GEOMETRY = deterministic mirror-symmetric (for WP0 unbiased check)
  RANDOM_GEOMETRY = random but mirror-axis preserving (for training/eval)

Action API: env.step(actions) where actions contains BOTH teams' moves.
Callers generate per-team actions (RL policy, mirror, rule-based, league).

State (per env, E parallel matches):
  radar_pos[E, 2, 2, 2]      team, radar, xy
  radar_alive[E, 2, 2]       bool
  radar_E[E, 2, 2]           laser kill energy per team's radar
  tracker_x[E, 2, 2, 4]      [x,vx,y,vy] per team's tracker of enemy radar
  tracker_P[E, 2, 2, 4, 4]   covariance
  exposure[E, 2]             per-team cumulative emission
  comm_link_ok[E, 2]         per-team fusion link status
  step_idx[E]

Action (dict, both teams):
  task_alloc[E, 2, 2, 4]     softmax fractions per team per aperture
  beam_target[E, 2, 2]       long per team per aperture, 0 or 1 enemy radar
  laser_target[E, 2]         long per team, 0 or 1 enemy radar
  emission_on[E, 2, 2]       bool per team per aperture

Observation: dim=36 per team (see get_obs docstring for layout).
"""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Optional, Dict, Tuple


__all__ = ["TwoTeamVecEnv", "MIRROR_GEOMETRY", "RANDOM_GEOMETRY"]

MIRROR_GEOMETRY = "mirror"
RANDOM_GEOMETRY = "random"


class TwoTeamVecEnv:
    """Two-team symmetric multifunction adversarial testbed (vectorized, GPU)."""

    def __init__(
        self,
        n_envs: int = 8,
        device: str = "cuda",
        dt: float = 0.1,
        episode_steps: int = 600,
        # Geometry
        map_size_m: float = 8000.0,
        team_offset_m: float = 2500.0,
        radar_separation_m: float = 1500.0,
        geometry: str = MIRROR_GEOMETRY,
        # Physics
        sigma_q: float = 2.0,
        # Radar measurements
        range_sigma_m: float = 0.05,
        # Function allocation
        n_subarrays: int = 25,
        comm_threshold: float = 0.10,
        # IQ-native physics constants (X-band nominal, WP-B will recalibrate)
        fc_hz: float = 10e9,
        channel_bw_hz: float = 10e6,
        noise_figure_db: float = 5.0,
        P_per_subarray_W: float = 5.0,
        aperture_D_m: float = 0.4,
        aperture_eta: float = 0.6,
        n_channels: int = 8,
        channel_spacing_hz: float = 50e6,
        # Kill chain
        e_kill: float = 2.0,
        dwell_rate: float = 1.0,
        decay_factor: float = 0.95,
        # WP-2 M0: laser slot-id semantics — energy accumulates on the enemy the
        # slot is *actually* tracking (by belief position), not on the slot index.
        # If slot r's belief is farther than `laser_hit_radius_m` from any alive
        # enemy, the laser misses (blind-fire correct behavior).
        laser_hit_radius_m: float = 50.0,
        # WP-C R4: tau_track relaxed 0.04 → 4.0 (σ_pos = 2m, UAV-class medium track).
        # At 0.04 (σ=0.2m), any f_emit>0 made kills=0, hiding RL kill advantage.
        # trace_P remains the continuous primary metric; kills now a usable auxiliary.
        tau_track: float = 4.0,
        # Exposure / home-on-jam
        exposure_gain: float = 200.0,   # FIX 2: was 50 → 200 (4× more sensitive, breaks duck lock)
        emit_power_per_subarray: float = 0.005,
        # Exposure overload (FIX 2: direct tracker decay when exposure extreme)
        exposure_overload_threshold: float = 50.0,
        exposure_decay_rate: float = 0.5,
        # FIX 1: frequency agility (anti-jam skill dimension)
        freq_hop_max: float = 8.0,   # max frequency-hop rate (N_FREQ_MAX in spec)
        # WP-1 M1: detection chain (replace god-view z=true_pos+noise).
        # See env/gpu/twoteam/detection.py + spec §9 TBD decisions.
        detect_threshold_db: float = 15.0,   # Swerling-0 13.2 / Swerling-I ~18 dB midpoint
        detect_width_db: float = 3.0,        # sigmoid transition width
        p_fa: float = 1e-6,                  # false alarm prob per cell (1e-3 stress)
        k_max: int = 8,                      # detection list cap (16 stress)
        n_search_cells: int = 84,            # azimuth search cells (360° / 4.3°)
        beam_width_rad: float = 0.075,       # 4.3° (0.886·λ/D, λ=0.03, D=0.4)
        sigma_rcs_m2: float = 1.0,           # target RCS (UAV-class Swerling-I median)
        range_max_m: float = 8000.0,         # max detection range (match map size)
        # WP-1 M1: pulse compression + coherent integration gain (dB).
        # 20 dB default = LFM/Barker (13 dB) + multi-pulse CPI (~7 dB) → nominal
        # SNR ~20 dB at 5 km (P_detect ~0.84 at threshold 15 dB).
        coherent_processing_gain_db: float = 20.0,
        # WP-1 M2: proactive-detect exposure bonus. When own aperture actively detects
        # a hidden enemy (frames_since_last_detection > 5), bump team exposure by this amount.
        # Represents the "active reveal" moment — enemy now cues your position too.
        # Small (~1 step of regular emit_increment = 0.005·25·0.1·0.75 ≈ 0.009) to not
        # destabilize home-on-jam dynamics.
        proactive_detect_exposure_bonus: float = 0.05,
        # Rewards
        w_kill: float = 10.0,
        w_survive: float = 1.0,
        w_exposure: float = 1.0,
        w_track: float = 0.1,
        seed: int = 42,
    ):
        self.E = int(n_envs)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.episode_steps = int(episode_steps)
        self.seed = int(seed)

        self.map_size_m = float(map_size_m)
        self.team_offset_m = float(team_offset_m)
        self.radar_separation_m = float(radar_separation_m)
        self.geometry = str(geometry)
        assert self.geometry in (MIRROR_GEOMETRY, RANDOM_GEOMETRY)

        self.sigma_q = float(sigma_q)
        self.range_sigma_m = float(range_sigma_m)
        self.n_subarrays = int(n_subarrays)
        self.comm_threshold = float(comm_threshold)

        self.fc_hz = float(fc_hz)
        self.channel_bw_hz = float(channel_bw_hz)
        self.noise_figure_db = float(noise_figure_db)
        self.P_per_subarray_W = float(P_per_subarray_W)
        self.aperture_D_m = float(aperture_D_m)
        self.aperture_eta = float(aperture_eta)
        self.n_channels = int(n_channels)
        self.channel_spacing_hz = float(channel_spacing_hz)
        self.e_kill = float(e_kill)
        self.dwell_rate = float(dwell_rate)
        self.decay_factor = float(decay_factor)
        self.laser_hit_radius_m = float(laser_hit_radius_m)
        self.tau_track = float(tau_track)

        self.exposure_gain = float(exposure_gain)
        self.emit_power_per_subarray = float(emit_power_per_subarray)
        self.exposure_overload_threshold = float(exposure_overload_threshold)
        self.exposure_decay_rate = float(exposure_decay_rate)
        self.freq_hop_max = float(freq_hop_max)

        # WP-1 M1: detection chain parameters.
        self.detect_threshold_db = float(detect_threshold_db)
        self.detect_width_db = float(detect_width_db)
        self.p_fa = float(p_fa)
        self.k_max = int(k_max)
        self.n_search_cells = int(n_search_cells)
        self.beam_width_rad = float(beam_width_rad)
        self.sigma_rcs_m2 = float(sigma_rcs_m2)
        self.range_max_m = float(range_max_m)
        self.coherent_processing_gain_db = float(coherent_processing_gain_db)
        self.proactive_detect_exposure_bonus = float(proactive_detect_exposure_bonus)

        self.w_kill = float(w_kill)
        self.w_survive = float(w_survive)
        self.w_exposure = float(w_exposure)
        self.w_track = float(w_track)

        self.n_teams = 2
        self.n_radars_per_team = 2
        self.n_fn = 4
        # WP-A: obs_dim 36 → 40 (4 freq-channel slots)
        # WP-1 M2: obs_dim 40 → 44 (4 partial-obs fields: fsld[r]×2, search_cov, n_det)
        self.obs_dim = 44
        self.privileged_dim = 8
        self._reset_count = 0

        self._init_tensors()

        # WP-A: instantiate IQ-native interference physics (stateless, ctor-once)
        from .iq_interference import IqInterference
        self.iq = IqInterference(
            fc_hz=self.fc_hz,
            channel_bw_hz=self.channel_bw_hz,
            noise_figure_db=self.noise_figure_db,
            P_per_subarray_W=self.P_per_subarray_W,
            aperture_D_m=self.aperture_D_m,
            aperture_eta=self.aperture_eta,
            n_subarrays=self.n_subarrays,
        )
        # WP-1 M1: import detection chain (stateless functions; mirror-symmetric RNG inside).
        from .detection import detect as _detect_fn   # noqa: F401  (used in step)
        self._detect_fn = _detect_fn
        # WP-2 M1: batched IMM (CV+CT) + PDAF tracker (replaces σ-gate NN + EKF).
        from .tracker import BatchedIMMPDAF
        self.tracker = BatchedIMMPDAF(self)

    def _init_tensors(self):
        dev = self.device
        E, T, R = self.E, self.n_teams, self.n_radars_per_team
        self.radar_pos = torch.zeros(E, T, R, 2, device=dev)
        self.radar_alive = torch.ones(E, T, R, dtype=torch.bool, device=dev)
        self.radar_E = torch.zeros(E, T, R, device=dev)
        self.tracker_x = torch.zeros(E, T, R, 4, device=dev)
        self.tracker_P = torch.eye(4, device=dev).expand(E, T, R, 4, 4).clone() * 0.5
        self.tracker_initialized = torch.zeros(E, T, R, dtype=torch.bool, device=dev)
        self.exposure = torch.zeros(E, T, device=dev)
        self.comm_link_ok = torch.ones(E, T, dtype=torch.bool, device=dev)
        self.step_idx = torch.zeros(E, dtype=torch.long, device=dev)
        self.team_kills = torch.zeros(E, T, dtype=torch.long, device=dev)
        self.team_alive = torch.ones(E, T, dtype=torch.bool, device=dev)
        self.first_kill_step = torch.full((E,), float(self.episode_steps), device=dev)
        self._last_jam_matrix = torch.zeros(E, T, R, device=dev)
        self._last_task_alloc = torch.full((E, T, R, 4), 0.25, device=dev)
        self._last_freq_hop = torch.ones(E, T, R, device=dev)   # FIX 1: default 1.0 = no hopping
        # WP-1 M1: enemy emit state (True = enemy radiates → passive/active detect possible;
        # False = enemy shuts down → only proactive detect can find it).
        # Default True — commanders can mutate to model LPI duty cycle / shutdown-when-tracked.
        self.enemy_emitting = torch.ones(E, T, R, dtype=torch.bool, device=dev)
        # WP-1 M2 hooks (filled in M2): search coverage + frames since last detection.
        self.search_coverage = torch.zeros(E, T, device=dev)
        self.frames_since_last_detection = torch.zeros(E, T, R, dtype=torch.long, device=dev)
        # WP-1 M2: per-team bitmap of searched azimuth cells. Each step, mark the cell
        # each own aperture's beam_az falls in (when aperture is in detect/track mode).
        # search_coverage = bitmap.mean(dim=-1). Reset to zeros each episode.
        self._searched_cells = torch.zeros(E, T, self.n_search_cells, dtype=torch.bool, device=dev)
        # WP-1 M2: per-team count of real (non-FA) detections this step. Updated by step().
        self.n_detections = torch.zeros(E, T, dtype=torch.long, device=dev)
        # WP-1 M1: last-step detections (for downstream consumers / obs in M2).
        self._last_detections = None   # type: ignore[assignment]
        # WP-A: per-radar absolute frequency + continuous beam azimuth (for IQ-native physics).
        # Filled by reset() with mirror-symmetric defaults.
        self.radar_freq_hz = torch.full((E, T, R), float(self.fc_hz), device=dev)
        self.radar_beam_az = torch.zeros(E, T, R, device=dev)
        # WP-A: per-episode pairwise geometry cache (filled by reset(), geometry is fixed per episode)
        self._pairwise_distance = torch.zeros(E, T * R, T * R, device=dev)

    def reset(self) -> Dict[str, torch.Tensor]:
        dev = self.device
        E = self.E
        self._reset_count += 1

        # WP-A: reset episode state FIRST, then set geometry on top.
        # (Previously _init_tensors was called AFTER setting radar_pos, which
        # zeroed radar_pos — silently degrading all distance-dependent physics
        # to the origin. Existing G0 PASS was only saved by mirror symmetry
        # holding even at pos=0. Fixed as part of WP-A real-geometry rollout.)
        self._init_tensors()

        if self.geometry == MIRROR_GEOMETRY:
            team_A_center = torch.tensor([-self.team_offset_m, 0.0], device=dev)
            team_B_center = torch.tensor([+self.team_offset_m, 0.0], device=dev)
            offsets = torch.tensor([
                [0.0, +self.radar_separation_m / 2.0],
                [0.0, -self.radar_separation_m / 2.0],
            ], device=dev)
            for e in range(E):
                self.radar_pos[e, 0, 0] = team_A_center + offsets[0]
                self.radar_pos[e, 0, 1] = team_A_center + offsets[1]
                self.radar_pos[e, 1, 0] = team_B_center - offsets[0]
                self.radar_pos[e, 1, 1] = team_B_center - offsets[1]
        else:
            rng = np.random.RandomState(self.seed + self._reset_count * 7919)
            for e in range(E):
                theta = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(self.team_offset_m * 0.7, self.team_offset_m * 1.3)
                cx, cy = r * math.cos(theta), r * math.sin(theta)
                team_A = torch.tensor([cx, cy], device=dev)
                team_B = -team_A
                sep_angle = rng.uniform(0, 2 * math.pi)
                dx = (self.radar_separation_m / 2.0) * math.cos(sep_angle)
                dy = (self.radar_separation_m / 2.0) * math.sin(sep_angle)
                offset_a = torch.tensor([dx, dy], device=dev)
                self.radar_pos[e, 0, 0] = team_A + offset_a
                self.radar_pos[e, 0, 1] = team_A - offset_a
                self.radar_pos[e, 1, 0] = team_B - offset_a
                self.radar_pos[e, 1, 1] = team_B + offset_a

        # WP-A: assign per-radar frequency channels, mirror-symmetric by default.
        # Team B's channel[r] = Team A's channel[r] (mirrored geometry already
        # negates positions; matching freq ensures cross-team JNR is symmetric).
        # Default: both radars within a team share channel 0 (worst-case intra-team
        # mutual interference). Tests / WP-D fixtures can override via env setter.
        ch0 = float(self.fc_hz)
        for e in range(E):
            for r in range(self.n_radars_per_team):
                self.radar_freq_hz[e, 0, r] = ch0
                self.radar_freq_hz[e, 1, r] = ch0   # mirror-symmetric

        return self.get_obs()

    def set_radar_freqs(self, freqs_hz: torch.Tensor):
        """Override per-radar frequencies (used by WP-A validation fixtures).

        Args:
            freqs_hz: [E, T, R] or [T, R] tensor of absolute frequencies.
        """
        if freqs_hz.dim() == 2:
            freqs_hz = freqs_hz.unsqueeze(0).expand(self.E, -1, -1)
        assert freqs_hz.shape == self.radar_freq_hz.shape, \
            f"shape mismatch: {freqs_hz.shape} vs {self.radar_freq_hz.shape}"
        self.radar_freq_hz = freqs_hz.to(self.device).float()

    def assert_no_godview(self, tol: float = 1e-5) -> Dict[str, object]:
        """Permutation test: obs must be invariant under enemy truth permutation (WP-1 §2.5).

        Randomly permutes `radar_pos` (the underlying enemy truth) and checks that
        `get_obs()` output is unchanged. Any obs dimension that changes is a god-view
        leak — it depends on enemy truth directly rather than through own sensors.

        Contract:
          - Test runs AFTER step() so all sensor state (tracker_x, jam_matrix, etc.)
            is already computed from past truth; permuting truth now shouldn't
            propagate to obs unless obs reads truth directly.
          - Restore state before returning (no side effects).

        Returns:
          {
            "pass_dims": [int],        # obs dim indices invariant under permutation
            "fail_dims": [int],        # obs dim indices that changed (god-view leaks)
            "max_diff_per_dim": [float],
            "tol": float,
          }
        """
        E, T, R = self.E, self.n_teams, self.n_radars_per_team
        dev = self.device

        # Save state
        pos_orig = self.radar_pos.clone()

        # Baseline obs
        obs_orig = self.get_obs()["obs"]   # [E, T, obs_dim]

        # Random per-env permutation of all radar positions (within each env)
        perm_idx = torch.argsort(torch.rand(E, T * R, device=dev), dim=-1)   # [E, T*R]
        pos_flat = pos_orig.view(E, T * R, 2)
        pos_perm_flat = torch.gather(pos_flat, 1, perm_idx.unsqueeze(-1).expand(-1, -1, 2))
        self.radar_pos = pos_perm_flat.view(E, T, R, 2)

        # Permuted obs
        obs_perm = self.get_obs()["obs"]

        # Restore state
        self.radar_pos = pos_orig

        # Per-dim max abs diff over (E, T)
        diff = (obs_orig - obs_perm).abs()                       # [E, T, obs_dim]
        max_diff_per_dim = diff.max(dim=0)[0].max(dim=0)[0]      # [obs_dim]

        pass_dims = [i for i in range(self.obs_dim)
                     if max_diff_per_dim[i].item() < tol]
        fail_dims = [i for i in range(self.obs_dim)
                     if max_diff_per_dim[i].item() >= tol]

        return {
            "pass_dims": pass_dims,
            "fail_dims": fail_dims,
            "max_diff_per_dim": max_diff_per_dim.tolist(),
            "tol": tol,
        }

    def get_detect_list(self) -> torch.Tensor:
        """Return last step's detection list for RL actor (WP-3 M0).

        Returns: [E, T, K_max, 5] tensor of (z_x, z_y, snr_db, is_fa_float, mask_float).
        - z_x, z_y: cartesian measurement position (m)
        - snr_db: SNR in dB (0 for FA / padding)
        - is_fa_float: 1.0 if false alarm, 0.0 otherwise
        - mask_float: 1.0 if slot holds any detection (real or FA), 0.0 if padding

        Returns zeros if no detections have been generated yet (first call before step).
        """
        E, T = self.E, self.n_teams
        K = self.k_max
        dev = self.device
        out = torch.zeros(E, T, K, 5, device=dev)
        if self._last_detections is None:
            return out
        dets = self._last_detections   # Detections dataclass: z[E,T,K,2], mask[E,T,K], is_false_alarm[E,T,K], snr_db[E,T,K]
        out[..., 0:2] = dets.z
        out[..., 2] = dets.snr_db
        out[..., 3] = dets.is_false_alarm.float()
        out[..., 4] = dets.mask.float()
        return out

    def warm_start_tracker(self, team: int, p: float = 1.0, rng_seed: Optional[int] = None):
        """WP-3.1 Fix D2: reverse curriculum — pre-init tracker for fraction of envs.

        For each env with prob `p`, set tracker_x[team] to true enemy position,
        tracker_P[team] to small value (< tau_track), tracker_initialized[team]=True.
        Used by br_trainer.collect_rollout at episode start when curriculum active.

        Args:
            team: which team's tracker to warm-start (usually learning_team)
            p: probability per env (0..1); 0 = no warm start, 1 = always
            rng_seed: optional seed for deterministic testing
        """
        if p <= 0.0:
            return
        E, R = self.E, self.n_radars_per_team
        et = 1 - team
        if rng_seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(rng_seed)
            mask = torch.rand(E, generator=gen, device=self.device) < p
        else:
            mask = torch.rand(E, device=self.device) < p
        mask = mask & self.radar_alive[:, et, 0]   # only if enemy alive
        # Set tracker_x[team] = true enemy pos (use slot 0 of enemy)
        enemy_pos = self.radar_pos[:, et, 0]   # [E, 2]
        for r in range(R):
            # Only update masked envs
            m = mask
            self.tracker_x[:, team, r, 0] = torch.where(
                m, enemy_pos[:, 0], self.tracker_x[:, team, r, 0])
            self.tracker_x[:, team, r, 2] = torch.where(
                m, enemy_pos[:, 1], self.tracker_x[:, team, r, 2])
            # velocity components left at current value (typically 0)
            # Set tracker_P small (well below tau_track) to mark "tight track"
            tight = torch.tensor(self.tau_track * 0.1, device=self.device)
            self.tracker_P[:, team, r, 0, 0] = torch.where(
                m, tight, self.tracker_P[:, team, r, 0, 0])
            self.tracker_P[:, team, r, 2, 2] = torch.where(
                m, tight, self.tracker_P[:, team, r, 2, 2])
            self.tracker_initialized[:, team, r] = self.tracker_initialized[:, team, r] | mask

    def get_obs(self) -> Dict[str, torch.Tensor]:
        """Per-team obs: {"obs": [E, 2, obs_dim], "privileged": [E, 2, priv_dim]}."""
        E, T, R = self.E, self.n_teams, self.n_radars_per_team
        dev = self.device
        obs = torch.zeros(E, T, self.obs_dim, device=dev)

        trace_P = self.tracker_P[..., 0, 0] + self.tracker_P[..., 2, 2]   # [E, T, R]

        for t in range(T):
            et = 1 - t
            for r in range(R):
                base = r * 8
                x_hat = self.tracker_x[:, t, r].clamp(-1e4, 1e4)
                obs[:, t, base + 0] = x_hat[..., 0] / 1000.0
                obs[:, t, base + 1] = x_hat[..., 1] / 100.0
                obs[:, t, base + 2] = x_hat[..., 2] / 1000.0
                obs[:, t, base + 3] = x_hat[..., 3] / 100.0
                obs[:, t, base + 4] = trace_P[:, t, r].clamp(0.0, 10.0)
                obs[:, t, base + 5] = self.radar_E[:, et, r] / self.e_kill
                obs[:, t, base + 6] = self._last_jam_matrix[:, t, r].clamp(0.0, 1.0)
                obs[:, t, base + 7] = (trace_P[:, t, r] < self.tau_track).float() * self.tracker_initialized[:, t, r].float()

            own_base = R * 8
            obs[:, t, own_base + 0] = (self.exposure[:, t] / 100.0).clamp(0.0, 10.0)
            obs[:, t, own_base + 1] = self.radar_alive[:, t, 0].float()
            obs[:, t, own_base + 2] = self.radar_alive[:, t, 1].float()
            obs[:, t, own_base + 3] = self.comm_link_ok[:, t].float()
            obs[:, t, own_base + 4] = (self.step_idx.float() / self.episode_steps).clamp(0.0, 1.0)
            ta_flat = self._last_task_alloc[:, t].reshape(E, -1)
            obs[:, t, own_base + 5:own_base + 13] = ta_flat
            # FIX 1: freq_hop exposure in obs (own + enemy). Normalized to [0, 1] by freq_hop_max.
            # Own hop rate per aperture (passive observation of own action memory):
            obs[:, t, own_base + 13] = (self._last_freq_hop[:, t, 0] / self.freq_hop_max).clamp(0.0, 1.0)
            obs[:, t, own_base + 14] = (self._last_freq_hop[:, t, 1] / self.freq_hop_max).clamp(0.0, 1.0)
            # Enemy hop rate per aperture (passive sensing: detect emission pattern):
            et = 1 - t
            obs[:, t, own_base + 15] = (self._last_freq_hop[:, et, 0] / self.freq_hop_max).clamp(0.0, 1.0)
            obs[:, t, own_base + 16] = (self._last_freq_hop[:, et, 1] / self.freq_hop_max).clamp(0.0, 1.0)
            # WP-A: frequency channel index per radar (own + enemy), normalized to [0,1].
            # Channel = round((freq - fc) / channel_spacing); clamp to n_channels for safety.
            def _ch_idx_norm(freq_hz_scalar_tensor):
                ch = ((freq_hz_scalar_tensor - self.fc_hz) / self.channel_spacing_hz).round()
                return (ch / max(self.n_channels, 1)).clamp(0.0, 1.0)
            # WP-A freq slots live at indices 36..39 (obs_dim 36 → 40).
            obs[:, t, 36] = _ch_idx_norm(self.radar_freq_hz[:, t, 0])
            obs[:, t, 37] = _ch_idx_norm(self.radar_freq_hz[:, t, 1])
            # WP-1 M2 no-godview: enemy freq only when enemy is emitting (passive sensing).
            # When enemy_emitting=False, obs reads 0 (can't measure enemy freq from silent target).
            enemy_emit_r0 = self.enemy_emitting[:, et, 0].float()
            enemy_emit_r1 = self.enemy_emitting[:, et, 1].float()
            obs[:, t, 38] = _ch_idx_norm(self.radar_freq_hz[:, et, 0]) * enemy_emit_r0
            obs[:, t, 39] = _ch_idx_norm(self.radar_freq_hz[:, et, 1]) * enemy_emit_r1
            # WP-1 M2: partial-obs fields (obs_dim 40 → 44).
            # frames_since_last_detection[r] / 100 (belief aging; ~episode_steps clamp at 1.0)
            obs[:, t, 40] = (self.frames_since_last_detection[:, t, 0].float() / 100.0).clamp(0.0, 1.0)
            obs[:, t, 41] = (self.frames_since_last_detection[:, t, 1].float() / 100.0).clamp(0.0, 1.0)
            # search_coverage: fraction of azimuth cells scanned so far
            obs[:, t, 42] = self.search_coverage[:, t].clamp(0.0, 1.0)
            # n_detections this step / K_max (real + FA)
            obs[:, t, 43] = (self.n_detections[:, t].float() / max(self.k_max, 1)).clamp(0.0, 1.0)

        priv = torch.zeros(E, T, self.privileged_dim, device=dev)
        for t in range(T):
            # Clamp trace_P at 1.0 before averaging: above this, track is effectively
            # lost and α_eff has saturated to 0 anyway (formula: 0.5·exp(-2·priv_4)
            # ≈ 0 once priv_4 > 5). Clamping prevents dead-target process-noise
            # accumulation (P grows via Q on dead targets) from inflating priv[:, 4]
            # to e.g. 100+, which would look like the α_eff bug.
            trace_P_clamped_mean = trace_P[:, t].clamp(0.0, 1.0).mean(dim=-1)
            priv[:, t, 0] = trace_P_clamped_mean
            priv[:, t, 1] = (self.exposure[:, t] / 100.0).clamp(0.0, 10.0)
            priv[:, t, 2] = self.radar_alive[:, t].float().mean(dim=-1)
            priv[:, t, 3] = (self.step_idx.float() / self.episode_steps).clamp(0.0, 1.0)
            # ⚠️ CRITICAL (per user reminder): priv[:, 4] MUST be normalized trace_P.
            # The α_eff bug: raw trace_P ≈ 200 → priv_4 = 5000 → α_eff collapses to 0
            # → MAPPO becomes IPPO silently. The clamp + assert below catch this.
            priv[:, t, 4] = trace_P_clamped_mean / max(self.tau_track, 1e-3)
            priv[:, t, 5] = self.team_alive[:, t].float()
            priv[:, t, 6] = self._last_jam_matrix[:, t].mean(dim=-1).clamp(0.0, 1.0)
            priv[:, t, 7] = self.comm_link_ok[:, t].float()

        # ASSERT (per user reminder): priv[:, 4] must be in normalized range.
        # Healthy tracking: trace_P → 0.005 → priv_4 = 0.125
        # Initial: trace_P = 1.0 (clamped) → priv_4 = 25
        # α_eff bug (raw trace_P ≈ 200): would give priv_4 = 5000 → fires this assert.
        # Threshold 100 = (1.0 clamp + 1.0 clamp) / 0.04 = 50, with 2x margin.
        priv_4_max = float(priv[..., 4].max().item())
        assert priv_4_max < 100.0, (
            f"priv[:, 4] max = {priv_4_max:.1f} — looks like raw trace_P, not normalized. "
            f"This is the α_eff bug. Check tau_track normalization."
        )

        return {"obs": obs, "privileged": priv}

    def step(self, action: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        E, T, R = self.E, self.n_teams, self.n_radars_per_team
        dev = self.device
        dt = self.dt

        task_alloc = action["task_alloc"]
        # WP-2 M2: beam_target optional — BlindClassical emits beam_direction only.
        # When absent, zeros (env still uses beam_direction below; legacy path skipped).
        beam_target = action.get("beam_target", torch.zeros(E, T, R, dtype=torch.long, device=dev))
        laser_target = action["laser_target"]
        emission_on = action["emission_on"].float()
        # FIX 1: freq_hop_rate per aperture; default to 1.0 (no hopping) for backward compat
        freq_hop_rate = action.get("freq_hop_rate",
                                    torch.ones(E, T, R, device=dev)).float().clamp(1.0, self.freq_hop_max)
        self._last_freq_hop = freq_hop_rate

        # WP-C R3: per-radar channel selection (dynamic coordination action).
        # Optional — if absent, env keeps the frequencies set at reset (or by
        # an external wrapper via set_radar_freqs). When present, env updates
        # radar_freq_hz = fc + channel_select * channel_spacing before IQ physics.
        channel_select = action.get("channel_select", None)
        if channel_select is not None:
            cs = channel_select.long().clamp(0, max(self.n_channels - 1, 0))
            self.radar_freq_hz = self.fc_hz + cs.float() * self.channel_spacing_hz

        # Normalize task_alloc
        task_alloc = task_alloc / (task_alloc.sum(dim=-1, keepdim=True) + 1e-8)
        self._last_task_alloc = task_alloc

        # 1. Compute freq_hop_per_tracker (used in both modes for victim-side
        # coherent-integration overhead).
        freq_hop_per_tracker = torch.ones(E, T, R, device=dev)
        for t in range(T):
            for r in range(R):
                hop_contribs = []
                for k in range(R):
                    beam_at = beam_target[:, t, k]
                    contrib = freq_hop_rate[:, t, k] * (beam_at == r).float() * emission_on[:, t, k]
                    hop_contribs.append(contrib)
                stacked = torch.stack(hop_contribs, dim=-1)
                freq_hop_per_tracker[:, t, r] = stacked.max(dim=-1).values.clamp(min=1.0)

        # 2. Comm link
        for t in range(T):
            f_comm_both = task_alloc[:, t, :, 3]
            self.comm_link_ok[:, t] = (f_comm_both >= self.comm_threshold).all(dim=-1)

        # 3a. Compute continuous beam_az per radar.
        # WP-1 M3: prefer `beam_direction` (continuous azimuth, no god-view) when present;
        # fall back to legacy `beam_target` (enemy index → azimuth via true_pos) otherwise.
        # M4 will hard-cut the legacy path. The legacy path is a known god-view leak
        # (action presupposes labeled enemies) — caught by §2.5 contract review.
        beam_az = torch.zeros(E, T, R, device=dev)
        beam_direction = action.get("beam_direction", None)
        if beam_direction is not None:
            # New API: continuous azimuth [-π, π] provided directly by policy.
            beam_az = beam_direction.float()
        else:
            # Legacy API: beam_target indexes enemy by 0/1 — convert via true_pos.
            # ⚠️ god-view leak: presumes knowledge of which physical enemy is "r=0".
            for t in range(T):
                et = 1 - t
                for k in range(R):
                    own_pos = self.radar_pos[:, t, k]                          # [E,2]
                    tgt_r = beam_target[:, t, k]                                # [E]
                    enemy_pos_all = self.radar_pos[:, et]                       # [E,R,2]
                    enemy_pos = torch.gather(
                        enemy_pos_all, 1, tgt_r.view(-1, 1, 1).expand(-1, 1, 2)
                    ).squeeze(1)                                                # [E,2]
                    delta = enemy_pos - own_pos
                    beam_az[:, t, k] = torch.atan2(delta[:, 1], delta[:, 0])
        self.radar_beam_az = beam_az

        # 3b. Per-victim σ computation — IQ-native physics.
        # Compute IQ-native JNR matrix [E, N=4, N=4]
        jnr_mat = self.iq.compute_jnr_matrix(
            pos=self.radar_pos,
            beam_az=beam_az,
            alloc=task_alloc,
            freq_hz=self.radar_freq_hz,
            emission_on=emission_on.bool(),
            hop_rate=freq_hop_rate,
            radar_alive=self.radar_alive,
        )
        # Build f_track_eff_TR and fusion_factor_TR [E,T,R] (loops unavoidable
        # due to f_track's beam_target indexing).
        f_track_eff_TR = torch.zeros(E, T, R, device=dev)
        for t in range(T):
            for r in range(R):
                f_track = torch.zeros(E, device=dev)
                for k in range(R):
                    beam_at = beam_target[:, t, k]
                    f_track += task_alloc[:, t, k, 1] * (beam_at == r).float() * emission_on[:, t, k]
                processing_overhead = 1.0 / freq_hop_per_tracker[:, t, r].pow(0.25).clamp(min=1.0)
                f_track_eff_TR[:, t, r] = f_track * processing_overhead
        fusion_factor_TR = torch.zeros(E, T, R, device=dev)
        for t in range(T):
            fusion_factor_TR[:, t] = torch.where(
                self.comm_link_ok[:, t].unsqueeze(-1),
                torch.ones(E, R, device=dev),
                torch.full((E, R), 1.5, device=dev),
            )
        # σ_meas from IQ physics (kept for σ_range floor + downstream metrics; Kalman
        # now consumes detection-chain σ via _kalman_update_step_external).
        sigma_meas = self.iq.compute_meas_sigma(
            jnr_matrix=jnr_mat,
            f_track_eff=f_track_eff_TR,
            range_sigma=self.range_sigma_m,
            fusion_factor=fusion_factor_TR,
        )
        # WP-1 M1: detection chain — replaces god-view `z = true_pos + noise`.
        # detections.z: [E, T, K_max, 2] cartesian; mask: real-detection slots.
        detections = self._detect_fn(
            radar_pos=self.radar_pos,
            beam_az=beam_az,
            alloc=task_alloc,
            emission_on=emission_on.bool(),
            enemy_emitting=self.enemy_emitting,
            radar_alive=self.radar_alive,
            jnr_matrix=jnr_mat,
            range_max_m=self.range_max_m,
            fc_hz=self.fc_hz,
            channel_bw_hz=self.channel_bw_hz,
            noise_figure_db=self.noise_figure_db,
            P_per_subarray_W=self.P_per_subarray_W,
            n_subarrays=self.n_subarrays,
            aperture_D_m=self.aperture_D_m,
            aperture_eta=self.aperture_eta,
            sigma_rcs_m2=self.sigma_rcs_m2,
            detect_threshold_db=self.detect_threshold_db,
            detect_width_db=self.detect_width_db,
            p_fa=self.p_fa,
            k_max=self.k_max,
            n_search_cells=self.n_search_cells,
            beam_width_rad=self.beam_width_rad,
            coherent_processing_gain_db=self.coherent_processing_gain_db,
            device=dev,
        )
        self._last_detections = detections

        # ---- IMM-PDAF tracker update (WP-2 M1: replaces σ-gate NN + EKF) ----
        # Per spec §3 ③: batched IMM (CV+CT 2-model) + PDAF data association
        # (5σ Mahalanobis gate, probabilistic β_i weights). Tracker reads env
        # state, performs mixing + per-model predict + PDAF update + fusion,
        # writes back tracker_x/P/initialized + last_real_assoc for fsld update.
        self.tracker.update(detections, sigma_meas)
        real_assoc = self.tracker.last_real_assoc                              # [E, T, R]
        # WP-1 M2: save pre-update fsld for proactive-detect bonus computation.
        fsld_pre_update = self.frames_since_last_detection.clone()
        self.frames_since_last_detection = torch.where(
            real_assoc,
            torch.zeros_like(self.frames_since_last_detection),
            self.frames_since_last_detection + 1,
        )

        # WP-1 M2: track search_coverage bitmap (azimuth cells scanned by own beams).
        # Mark cell each own aperture's beam_az falls in when active (detect/track alloc + emitting).
        cell_width = 2.0 * math.pi / self.n_search_cells
        cell_idx_all = ((beam_az + math.pi) / cell_width).long() % self.n_search_cells   # [E, T, R]
        f_dt_all = task_alloc[..., 0] + task_alloc[..., 1]                              # [E, T, R]
        active_search = (f_dt_all > 0.01) & emission_on.bool()                          # [E, T, R]
        for t in range(T):
            for r in range(R):
                cells_to_mark = cell_idx_all[:, t, r]                                   # [E]
                onehot = torch.zeros(E, self.n_search_cells, device=dev, dtype=torch.bool)
                onehot.scatter_(1, cells_to_mark.unsqueeze(1), True)
                active_e = active_search[:, t, r].unsqueeze(-1)                         # [E, 1]
                self._searched_cells[:, t] = self._searched_cells[:, t] | (onehot & active_e)
        self.search_coverage = self._searched_cells.float().mean(dim=-1)                # [E, T]

        # WP-1 M2: count real (non-FA) detections per team for obs.
        # detection.mask is [E, T, K_max]; sum over K_max gives per-team count.
        real_per_team = (detections.mask & ~detections.is_false_alarm).sum(dim=-1)      # [E, T]
        self.n_detections = real_per_team.long()

        # WP-1 M2: proactive detect → exposure bonus.
        # Spec §2.3: when own aperture actively detects a HIDDEN enemy (fsld_pre > threshold),
        # bump team exposure — represents the "active reveal" moment when enemy cues your position.
        # Bonus is small (~1 step of emit_increment) to not destabilize home-on-jam dynamics.
        hidden_threshold_steps = 5
        was_hidden = fsld_pre_update > hidden_threshold_steps                           # [E, T, R]
        proactive_events = (was_hidden & real_assoc).float().sum(dim=-1)                # [E, T]
        proactive_bonus = proactive_events * self.proactive_detect_exposure_bonus       # [E, T]
        # Apply after the regular exposure update below (deferred to section 5).

        # Preserve priv-obs contract: _last_jam_matrix = total JNR per victim (clamped to [0,1])
        self._last_jam_matrix = jnr_mat.sum(dim=1).view(E, T, R).clamp(0.0, 1.0)


        # 4. Laser dwell-kill chain (WP-2 M0: slot-id semantics + belief check)
        # lsr_slot indexes TRACKER SLOT (not enemy). Energy accumulates on the
        # enemy the slot is actually tracking (by belief position). If slot's
        # belief is farther than laser_hit_radius_m from any alive enemy, laser
        # misses. Blind-fire correct: slot mis-tracked → laser misses.
        for t in range(T):
            et = 1 - t
            lsr_slot = laser_target[:, t]                              # [E] slot id
            trace_P_t = self.tracker_P[:, t, :, 0, 0] + self.tracker_P[:, t, :, 2, 2]
            lsr_trace_P = torch.gather(trace_P_t, 1, lsr_slot.unsqueeze(1)).squeeze(1)
            lsr_init = self.tracker_initialized[:, t].gather(1, lsr_slot.unsqueeze(1)).squeeze(1)
            lsr_track_ok = (lsr_trace_P < self.tau_track) & lsr_init
            # Belief position for the fired slot [E, 2] (tracker_x is [x, vx, y, vy])
            lsr_belief_state = torch.gather(
                self.tracker_x[:, t], 1,
                lsr_slot.view(-1, 1, 1).expand(-1, 1, 4),
            ).squeeze(1)                                               # [E, 4]
            lsr_belief_pos = lsr_belief_state[:, [0, 2]]               # [E, 2]
            # Distance from belief to each alive enemy [E, R]
            enemy_pos_all = self.radar_pos[:, et]                      # [E, R, 2]
            d_to_enemies = (enemy_pos_all - lsr_belief_pos.unsqueeze(1)).norm(dim=-1)
            enemy_alive_mask = self.radar_alive[:, et]                 # [E, R]
            d_masked = torch.where(
                enemy_alive_mask, d_to_enemies,
                torch.full_like(d_to_enemies, 1e9),
            )
            nearest_d, nearest_enemy = d_masked.min(dim=-1)            # [E], [E]
            hit_mask = nearest_d < self.laser_hit_radius_m             # [E]
            emitting_t = emission_on[:, t].sum(dim=-1) > 0.5
            accum_mask = lsr_track_ok & hit_mask & emitting_t
            accum_dt = self.dwell_rate * dt * accum_mask.float()
            accum_per_radar = torch.zeros(E, R, device=dev)
            accum_per_radar.scatter_(1, nearest_enemy.unsqueeze(1), accum_dt.unsqueeze(1))
            new_E = self.radar_E[:, et] + accum_per_radar
            track_all = (trace_P_t < self.tau_track) & self.tracker_initialized[:, t]
            decay_mask = (~track_all) & self.radar_alive[:, et]
            new_E = torch.where(decay_mask, new_E * self.decay_factor, new_E)
            self.radar_E[:, et] = new_E

            new_kill = (self.radar_E[:, et] >= self.e_kill) & self.radar_alive[:, et]
            self.radar_alive[:, et] = self.radar_alive[:, et] & (~new_kill)
            n_new_kills = new_kill.long().sum(dim=-1)
            self.team_kills[:, t] += n_new_kills

            any_kill = new_kill.any(dim=-1)
            not_yet = self.first_kill_step >= self.episode_steps
            upd = any_kill & not_yet
            self.first_kill_step = torch.where(upd, (self.step_idx + 1).float(), self.first_kill_step)

        self.team_alive = self.radar_alive.any(dim=-1)

        # 5. Exposure + home-on-jam (FIX 2: exposure_gain 50→200, add overload decay)
        emit_increment = self.emit_power_per_subarray * self.n_subarrays * emission_on * dt
        radiating_fraction = task_alloc[..., 0] + task_alloc[..., 1] + task_alloc[..., 2]
        emit_increment = emit_increment * radiating_fraction
        self.exposure = self.exposure + emit_increment.sum(dim=-1)
        # WP-1 M2: proactive-detect bonus (computed above from fsld_pre_update & real_assoc).
        self.exposure = self.exposure + proactive_bonus

        exposure_norm = self.exposure / 100.0
        p_homejam = 1.0 - torch.exp(-self.exposure_gain * exposure_norm * 0.001)
        # Mirror-symmetric home-on-jam: same roll per env per radar slot across teams.
        # Without this, identical actions under MIRROR_GEOMETRY produce asymmetric
        # deaths (team 0 dies, team 1 lives) → kills/exposure/trace_P all diverge,
        # and the WP0 unbiased check fails on a stochasticity, not a real bias.
        homejam_roll = torch.rand(E, 1, R, device=dev).expand(E, T, R)
        homejam_death = (homejam_roll < p_homejam.unsqueeze(-1)) & self.radar_alive
        self.radar_alive = self.radar_alive & (~homejam_death)

        # FIX 2: exposure overload — extreme exposure directly inflates own tracker_P,
        # representing geolocation backfire (you radiate too much → enemy cues your
        # position → your own tracking suffers from counter-fire EM contamination).
        # This is asymmetric (depends on per-team exposure), so it breaks the mirror
        # duck lock without depending on RNG.
        for t in range(T):
            overloaded = (self.exposure[:, t] > self.exposure_overload_threshold).float()   # [E]
            decay = overloaded * self.exposure_decay_rate * dt   # [E]
            # Inflate diag of own tracker_P (position uncertainty grows)
            for r in range(R):
                # Only inflate if tracker is on alive enemy (dead enemy tracker P grows via Q anyway)
                enemy_alive = self.radar_alive[:, 1 - t, r].float()
                self.tracker_P[:, t, r, 0, 0] = self.tracker_P[:, t, r, 0, 0] + decay * enemy_alive * 0.5
                self.tracker_P[:, t, r, 2, 2] = self.tracker_P[:, t, r, 2, 2] + decay * enemy_alive * 0.5

        self.team_alive = self.radar_alive.any(dim=-1)

        # 6. Reward (zero-sum)
        reward = torch.zeros(E, T, device=dev)
        for t in range(T):
            kill_score = self.team_kills[:, t].float() * self.w_kill
            survive_score = self.radar_alive[:, t].sum(dim=-1).float() * self.w_survive
            exposure_pen = -self.exposure[:, t] * 0.001 * self.w_exposure
            trace_P_t = self.tracker_P[:, t, :, 0, 0] + self.tracker_P[:, t, :, 2, 2]
            track_bonus = ((trace_P_t < self.tau_track) & self.tracker_initialized[:, t]).float().sum(dim=-1) * self.w_track * 0.1
            reward[:, t] = kill_score + survive_score + exposure_pen + track_bonus
        zero_sum_reward = reward - reward.flip(dims=[1])

        # 7. Step + done
        self.step_idx += 1
        done = (self.step_idx >= self.episode_steps) | (~self.team_alive.any(dim=-1))

        info = {
            "team_kills": self.team_kills.clone(),
            "team_alive": self.team_alive.clone(),
            "radar_alive": self.radar_alive.clone(),
            "exposure": self.exposure.clone(),
            # Mean trace_P only over alive enemy radars (dead targets' P grows
            # unboundedly via process noise — masking gives a clean tracking-quality signal).
            "mean_trace_P": self._alive_mean_trace_P(),
            "comm_link_ok": self.comm_link_ok.clone(),
            "step_idx": self.step_idx.clone(),
        }

        return self.get_obs(), zero_sum_reward, done, info

    def _alive_mean_trace_P(self) -> torch.Tensor:
        """Per-team mean trace_P, only over ALIVE enemy radars.

        Returns [E, T]. Dead enemy radars excluded (their P grows via Q).
        If both enemy radars are dead, returns 0 for that team.
        """
        trace_P = self.tracker_P[..., 0, 0] + self.tracker_P[..., 2, 2]   # [E, T, R]
        # Team t tracks enemy team 1-t's radars. Enemy alive mask:
        enemy_alive_per_tracker = torch.stack([self.radar_alive[:, 1], self.radar_alive[:, 0]], dim=1)   # [E, T, R]
        masked = trace_P * enemy_alive_per_tracker.float()
        n_alive = enemy_alive_per_tracker.float().sum(dim=-1).clamp(min=1.0)
        return (masked.sum(dim=-1) / n_alive).clone()
