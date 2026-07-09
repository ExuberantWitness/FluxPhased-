"""TAES asymmetric engagement environment (vectorized, GPU).

Scenario:
  Red team: 2 phased-array radars + 1 commander + 1 laser, defending against
            N incoming targets (missiles/drones).
  Blue team: 1 adaptive jammer (L0/L1/L3 from adversary.py) disrupting red's
             tracking to prevent laser kills, with home-on-jam capability.

The laser-kill chain is the core coupling: per-target energy E_i accumulates
only while track quality is sufficient (trace_P_i < tau_track); track loss
(jamming-induced σ inflation) decays E_i. This forces the commander to maintain
firm track on the lased target throughout the dwell.

State vector (per env):
  radar_pos[E, 2, 2]         red radar xy (fixed)
  target_pos[E, N_max, 2]    true target xy
  target_vel[E, N_max, 2]    true target velocity
  target_alive[E, N_max]     bool, real+alive targets
  target_n_actual[E]         int, real target count (≤ N_max)
  target_E[E, N_max]         kill energy accumulator
  tracker_x[E, N_max, 4]     [x, vx, y, vy] per target
  tracker_P[E, N_max, 4, 4]  covariance per target
  exposure[E]                cumulative emission (home-on-jam bait)
  own_alive[E]               bool
  step_idx[E]                int

Action (red commander, dict of tensors):
  task_alloc[E, 4]           subarray fractions (detect/track/jam/comm)
  beam_target_idx[E]         which target to point main beam at
  laser_target_idx[E]        which target to fire laser at
  emission_on[E]             bool (False = passive, drops exposure growth
                            but inflates track σ via missed measurements)

Observation (E, 95):
  per-target block × 8 (88 dim): [x̂(2), v̂(2), trace_P(1), E_i(1),
                                  JSR_i(4), track_ok(1)]
  global (7): [exposure_norm, task_alloc(4), own_alive, step_norm]

All physics run on GPU; jammer is the only external object (CPU-side calls).
"""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Optional, Dict, Any


__all__ = ["TAESVecEnv"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _angle_wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap radians to [-pi, pi]."""
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


# ----------------------------------------------------------------------------
# Main env
# ----------------------------------------------------------------------------

class TAESVecEnv:
    """Asymmetric TAES engagement testbed.

    Vectors over E parallel envs. Targets padded to N_max=8; targets beyond
    n_actual are masked (alive_mask=False) and don't participate in physics.
    """

    def __init__(
        self,
        n_envs: int = 8,
        n_targets: int = 4,
        n_targets_max: int = 8,
        device: str = "cuda",
        dt: float = 0.1,                   # control step (s)
        episode_steps: int = 600,          # 60s
        # Geometry
        map_size_m: float = 8000.0,
        radar_offset_m: float = 2500.0,    # radars at (±offset, 0)
        target_range_km: tuple = (2.5, 3.5),
        target_azimuth_deg: tuple = (-60.0, 60.0),
        target_v_mps: tuple = (150.0, 300.0),
        # Target kinematics
        p_turn_per_step: float = 0.05,
        turn_rate_deg_s: tuple = (5.0, 15.0),
        sigma_q: float = 2.0,              # process noise accel (m/s^2) — match real CV
        # Radar measurements
        range_sigma_m: float = 0.05,       # 5 cm (spec: cm level)
        crossrange_factor: float = 7.4e-5, # σ_cross = R * factor
        bearing_sigma_rad: float = 1e-4,   # alt: bearing accuracy
        residual_scale_m: float = 6.0,
        use_range_bearing: bool = True,    # measurement type
        # Jamming coupling
        jam_gain: float = 8.0,
        # Kill chain
        kr_thresh_m: float = 0.5,
        tau_track_scale: float = 3.0,      # tau_track = scale * steady-state trace_P
        dwell_rate: float = 1.0,           # E_gain per second when track_ok+lasing
        e_kill: float = 2.0,               # energy to kill
        decay_factor: float = 0.5,         # E_i multiplier on track loss
        # Exposure / home-on-jam
        exposure_gain: float = 50.0,
        emit_power_per_subarray: float = 0.005,  # exposure increment per step per active subarray
        n_subarrays: int = 25,
        race_death_penalty: float = 30.0,
        # Reward weights
        w_kill: float = 10.0,
        w_survival: float = 5.0,
        w_exposure: float = 1.0,
        w_ttk: float = 0.1,
        # Seeding
        seed: int = 42,
        # Initial covariance
        p0_pos: float = 100.0,             # initial position uncertainty (m^2)
        p0_vel: float = 500.0,             # initial velocity uncertainty (m^2/s^2)
        # Logging
        log_metrics: bool = True,
    ):
        self.E = int(n_envs)
        self.N = int(n_targets)
        self.N_max = int(n_targets_max)
        self.device = torch.device(device)
        self.dt = float(dt)
        self.episode_steps = int(episode_steps)

        # Bookkeeping constants
        self.map_size_m = float(map_size_m)
        self.radar_offset_m = float(radar_offset_m)
        self.target_range_km = target_range_km
        self.target_azimuth_deg = target_azimuth_deg
        self.target_v_mps = target_v_mps
        self.p_turn_per_step = float(p_turn_per_step)
        self.turn_rate_deg_s = turn_rate_deg_s
        self.sigma_q = float(sigma_q)
        self.range_sigma_m = float(range_sigma_m)
        self.crossrange_factor = float(crossrange_factor)
        self.bearing_sigma_rad = float(bearing_sigma_rad)
        self.residual_scale_m = float(residual_scale_m)
        self.use_range_bearing = bool(use_range_bearing)

        self.jam_gain = float(jam_gain)
        self.kr_thresh_m = float(kr_thresh_m)
        self.tau_track_scale = float(tau_track_scale)
        self.dwell_rate = float(dwell_rate)
        self.e_kill = float(e_kill)
        self.decay_factor = float(decay_factor)

        self.exposure_gain = float(exposure_gain)
        self.emit_power_per_subarray = float(emit_power_per_subarray)
        self.n_subarrays = int(n_subarrays)
        self.race_death_penalty = float(race_death_penalty)

        self.w_kill = float(w_kill)
        self.w_survival = float(w_survival)
        self.w_exposure = float(w_exposure)
        self.w_ttk = float(w_ttk)

        self.p0_pos = float(p0_pos)
        self.p0_vel = float(p0_vel)

        self.seed = int(seed)
        self._g = torch.Generator(device=self.device)
        self._g.manual_seed(self.seed)

        # Observation / action dims (spec)
        self.obs_dim = 11 * self.N_max + 7   # 95 when N_max=8
        self.n_task_alloc = 4

        # State tensors (allocated on reset)
        self.radar_pos: torch.Tensor = None     # [E, 2, 2]
        self.target_pos: torch.Tensor = None    # [E, N_max, 2]
        self.target_vel: torch.Tensor = None    # [E, N_max, 2]
        self.target_alive_mask: torch.Tensor = None  # [E, N_max] bool
        self.target_n_actual: torch.Tensor = None    # [E] long
        self.target_E: torch.Tensor = None      # [E, N_max]
        self.target_killed: torch.Tensor = None # [E, N_max] bool, cumulative
        self.tracker_x: torch.Tensor = None     # [E, N_max, 4]
        self.tracker_P: torch.Tensor = None     # [E, N_max, 4, 4]
        self.exposure: torch.Tensor = None      # [E]
        self.own_alive: torch.Tensor = None     # [E] bool
        self.step_idx: torch.Tensor = None      # [E] long
        self.first_kill_step: torch.Tensor = None  # [E] long, episode_steps if no kill

        # Derived: tau_track (FIXED threshold based on empirical sim observations).
        # The auto-compute via Riccati was numerically unstable (overestimated
        # by 10-50×). Empirical: no-jam trace_P ~0.005-0.03 (transient+post-turn),
        # jammed trace_P ~0.07-0.15. tau=0.04 sits between, allowing no-jam track
        # to mostly succeed while breaking under sustained jam.
        # Override the computed value with the empirical constant.
        _ = self._compute_tau_track()  # warm-up (validates math doesn't crash)
        self.tau_track_nominal: float = 0.04
        # Backward-compat alias
        self.tau_track: float = self.tau_track_nominal
        # Per-target tau disabled: jam scales CRLB, defeats the purpose.
        # Use fixed no-jam threshold instead.
        self.use_per_target_tau: bool = False

        # Step metrics (reset each step)
        self._last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # CV Kalman stationary covariance → tau_track
    # ------------------------------------------------------------------
    def _compute_tau_track(self) -> float:
        """Estimate steady-state trace_P via 200-step Riccati iteration,
        averaged over multiple random target geometries drawn from the spawn
        distribution. tau_track = tau_track_scale × median(trace_P_ss).

        Using a single nominal geometry was unstable: worst-case (symmetric)
        geometry gave 16× larger trace_P_ss than typical off-axis geometry.
        Median over random samples is representative.
        """
        dt = self.dt
        F = torch.tensor([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=self.device)
        q = self.sigma_q ** 2
        Q = torch.tensor([
            [q*dt**4/4, q*dt**3/2, 0, 0],
            [q*dt**3/2, q*dt**2,   0, 0],
            [0, 0, q*dt**4/4, q*dt**3/2],
            [0, 0, q*dt**3/2, q*dt**2],
        ], dtype=torch.float32, device=self.device)

        r0 = torch.tensor([-self.radar_offset_m, 0.0], device=self.device)
        r1 = torch.tensor([+self.radar_offset_m, 0.0], device=self.device)

        # Sample 16 random geometries from spawn distribution
        n_samples = 16
        g = torch.Generator(device=self.device)
        g.manual_seed(0)
        r_low = self.target_range_km[0] * 1000.0
        r_high = self.target_range_km[1] * 1000.0
        az_low = math.radians(self.target_azimuth_deg[0])
        az_high = math.radians(self.target_azimuth_deg[1])
        r_s = torch.rand(n_samples, generator=g, device=self.device) * (r_high - r_low) + r_low
        az_s = torch.rand(n_samples, generator=g, device=self.device) * (az_high - az_low) + az_low
        x_s = (r_s * torch.sin(az_s)).tolist()
        y_s = (r_s * torch.cos(az_s)).tolist()

        trace_P_ss_list = []
        for x_nom_xy in zip(x_s, y_s):
            x_nom = torch.tensor([x_nom_xy[0], 0.0, x_nom_xy[1], -150.0],
                                  device=self.device)
            P = torch.diag(torch.tensor(
                [self.p0_pos, self.p0_vel, self.p0_pos, self.p0_vel],
                device=self.device))
            for _ in range(300):
                P_pred = F @ P @ F.T + Q
                for r_pos in [r0, r1]:
                    dx = x_nom[0] - r_pos[0]
                    dy = x_nom[2] - r_pos[1]
                    R_dist = torch.sqrt(dx*dx + dy*dy + 1.0)
                    if self.use_range_bearing:
                        H = torch.tensor([
                            [dx/R_dist, 0.0, dy/R_dist, 0.0],
                            [-dy/(R_dist*R_dist), 0.0, dx/(R_dist*R_dist), 0.0],
                        ], device=self.device)
                        sig_r = self.range_sigma_m
                        sig_b = self.bearing_sigma_rad
                        R_meas = torch.diag(torch.tensor(
                            [sig_r*sig_r, sig_b*sig_b], device=self.device))
                    else:
                        H = torch.tensor([
                            [dx/R_dist, 0.0, dy/R_dist, 0.0],
                            [-dy/R_dist, 0.0, dx/R_dist, 0.0],
                        ], device=self.device)
                        sig_r = self.range_sigma_m
                        sig_x = R_dist * self.crossrange_factor
                        R_meas = torch.diag(torch.tensor(
                            [sig_r*sig_r, sig_x*sig_x], device=self.device))
                    S = H @ P_pred @ H.T + R_meas
                    K = P_pred @ H.T @ torch.linalg.inv(S)
                    P = (torch.eye(4, device=self.device) - K @ H) @ P_pred
            trace_P_ss_list.append(float(P[0, 0] + P[2, 2]))

        # Use median (robust to outliers)
        trace_P_ss_list.sort()
        median_ss = trace_P_ss_list[len(trace_P_ss_list) // 2]
        tau = self.tau_track_scale * median_ss
        return tau

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, torch.Tensor]:
        E, N_max = self.E, self.N_max
        dev = self.device

        # Radars at fixed positions (mirrored layout)
        self.radar_pos = torch.zeros(E, 2, 2, device=dev)
        self.radar_pos[:, 0, 0] = -self.radar_offset_m
        self.radar_pos[:, 1, 0] = +self.radar_offset_m

        # Targets: spawn in ring [2.5, 3.5] km, azimuth ±60°, moving toward origin
        r_low = self.target_range_km[0] * 1000.0
        r_high = self.target_range_km[1] * 1000.0
        az_low = math.radians(self.target_azimuth_deg[0])
        az_high = math.radians(self.target_azimuth_deg[1])
        v_low, v_high = self.target_v_mps

        # Sample initial positions/velocities
        rng = self._g
        r = torch.rand(E, N_max, generator=rng, device=dev) * (r_high - r_low) + r_low
        az = torch.rand(E, N_max, generator=rng, device=dev) * (az_high - az_low) + az_low
        # Target spawns in upper half (positive y) heading toward origin
        target_x0 = r * torch.sin(az)   # ±azimuth from +y axis
        target_y0 = r * torch.cos(az)
        self.target_pos = torch.stack([target_x0, target_y0], dim=-1)  # [E, N_max, 2]

        # Velocity: heading toward origin (radial inward) + small perturbation
        speed = torch.rand(E, N_max, generator=rng, device=dev) * (v_high - v_low) + v_low
        # Radial inward unit vector
        radial = -self.target_pos / (torch.norm(self.target_pos, dim=-1, keepdim=True) + 1.0)
        # Add small perpendicular perturbation (±15° off-radial)
        perturb_angle = (torch.rand(E, N_max, generator=rng, device=dev) - 0.5) * math.radians(30.0)
        cos_p = torch.cos(perturb_angle)
        sin_p = torch.sin(perturb_angle)
        # Rotate radial by perturb_angle
        rx, ry = radial[..., 0], radial[..., 1]
        vx = cos_p * rx - sin_p * ry
        vy = sin_p * rx + cos_p * ry
        self.target_vel = torch.stack([vx * speed, vy * speed], dim=-1)

        # Target alive mask: first n_actual=N targets real, rest padding
        self.target_alive_mask = torch.zeros(E, N_max, dtype=torch.bool, device=dev)
        self.target_alive_mask[:, :self.N] = True
        self.target_n_actual = torch.full((E,), self.N, dtype=torch.long, device=dev)

        self.target_E = torch.zeros(E, N_max, device=dev)
        self.target_killed = torch.zeros(E, N_max, dtype=torch.bool, device=dev)

        # Tracker: initialize at (radar centroid + measurement) with large covariance
        # For simplicity, init at the true target position (we'll have a separate
        # acquisition model in WP1's IMM-PDAF; for WP0 validation, perfect init
        # is acceptable to isolate kill-chain coupling from acquisition).
        self.tracker_x = torch.zeros(E, N_max, 4, device=dev)
        self.tracker_x[..., 0] = self.target_pos[..., 0]
        self.tracker_x[..., 1] = self.target_vel[..., 0]
        self.tracker_x[..., 2] = self.target_pos[..., 1]
        self.tracker_x[..., 3] = self.target_vel[..., 1]

        self.tracker_P = torch.diag_embed(torch.tensor(
            [self.p0_pos, self.p0_vel, self.p0_pos, self.p0_vel],
            device=dev)).unsqueeze(0).unsqueeze(0).expand(E, N_max, 4, 4).clone()

        self.exposure = torch.zeros(E, device=dev)
        self.own_alive = torch.ones(E, dtype=torch.bool, device=dev)
        self.step_idx = torch.zeros(E, dtype=torch.long, device=dev)
        self.first_kill_step = torch.full((E,), self.episode_steps, dtype=torch.long, device=dev)

        # Build obs
        obs = self._build_obs()
        self._last_info = {}
        return obs

    def _compute_per_target_tau(self, jam_mul: torch.Tensor) -> torch.Tensor:
        """Compute per-target tau_track based on current-geometry CRLB.

        For each alive target i at its current position, compute the single-time
        CRLB with the current jam_mul. tau_track_i = tau_track_scale × CRLB_i.

        Semantics: track_ok_i means "trace_P_i is within scale× of the best
        achievable at this geometry under current jamming". This auto-adapts
        to both geometry and jam level.

        Returns: [E, N_max] tau_track_i
        """
        from algo._shared.laser.crlb import compute_crlb

        E, N_max = self.E, self.N_max
        dev = self.device
        # Per-target sigmas with current jam_mul
        sig_r = (self.range_sigma_m * jam_mul).unsqueeze(-1).expand(E, N_max)
        sig_b = (self.bearing_sigma_rad * jam_mul).unsqueeze(-1).expand(E, N_max)
        crlb = compute_crlb(
            self.target_pos, self.radar_pos, sig_r, sig_b,
            self.target_alive_mask, use_range_bearing=self.use_range_bearing,
            crossrange_factor=self.crossrange_factor,
        )
        # Replace inf (dead targets) with the nominal tau to avoid NaN
        crlb = torch.where(torch.isfinite(crlb), crlb,
                           torch.full_like(crlb, self.tau_track_nominal))
        return self.tau_track_scale * crlb

    # ------------------------------------------------------------------
    # Build observation
    # ------------------------------------------------------------------
    def _build_obs(
        self,
        task_alloc: Optional[torch.Tensor] = None,
        jam_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Construct the 95-dim commander observation."""
        E, N_max = self.E, self.N_max
        dev = self.device

        # Per-target block: [x̂(2), v̂(2), trace_P(1), E_i(1), JSR_i(4), track_ok(1)] = 11
        per_target = torch.zeros(E, N_max, 11, device=dev)
        per_target[..., 0:2] = self.tracker_x[..., 0:3:2] / self.map_size_m   # x, y (skip vx)
        # Wait: tracker_x[..., 0]=x, [1]=vx, [2]=y, [3]=vy
        per_target[..., 0] = self.tracker_x[..., 0] / (self.map_size_m / 2.0)
        per_target[..., 1] = self.tracker_x[..., 2] / (self.map_size_m / 2.0)
        per_target[..., 2] = self.tracker_x[..., 1] / 300.0   # vx norm
        per_target[..., 3] = self.tracker_x[..., 3] / 300.0   # vy norm
        # trace_P
        trace_P = self.tracker_P[..., 0, 0] + self.tracker_P[..., 2, 2]
        per_target[..., 4] = trace_P / max(self.tau_track_nominal, 1e-3)
        # E_i normalized by e_kill
        per_target[..., 5] = (self.target_E / max(self.e_kill, 1e-6)).clamp(0.0, 1.5)
        # JSR_i (4 dim): scalar jam_level broadcast to 4 (placeholder for future multi-band)
        if jam_level is None:
            jam_level = torch.zeros(E, device=dev)
        # jam_level applied to all targets uniformly (no per-target discrimination yet)
        per_target[..., 6:10] = jam_level.unsqueeze(-1).unsqueeze(-1).expand(E, N_max, 4) / 1.0
        # track_ok
        track_ok = (trace_P < self.tau_track) & self.target_alive_mask
        per_target[..., 10] = track_ok.float()

        # Mask padding (zero out non-alive targets in obs)
        mask = self.target_alive_mask.float().unsqueeze(-1)
        per_target = per_target * mask

        # Flatten per-target block
        per_target_flat = per_target.reshape(E, N_max * 11)

        # Global (7): [exposure_norm, task_alloc(4), own_alive, step_norm]
        glob = torch.zeros(E, 7, device=dev)
        glob[:, 0] = (self.exposure / 100.0).clamp(0.0, 10.0)  # normalize
        if task_alloc is not None:
            glob[:, 1:5] = task_alloc
        glob[:, 5] = self.own_alive.float()
        glob[:, 6] = self.step_idx.float() / float(self.episode_steps)

        obs = torch.cat([per_target_flat, glob], dim=-1)  # [E, 95]
        return {"obs": obs}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self,
        action: Dict[str, torch.Tensor],
        jammer=None,
    ) -> tuple:
        """Step the env one control step.

        Args:
            action: dict with keys
                task_alloc[E, 4]        subarray fractions
                beam_target_idx[E]      long, which target main beam points at
                laser_target_idx[E]     long, which target laser fires at
                emission_on[E]          float 0/1
            jammer: optional adversary object with .step(red_task_hist[E,1,4],
                    jam_history[E,1]) → jam_level[E,1]. If None, jam_level=0.

        Returns:
            obs_dict: {"obs": [E, 95]}
            reward: [E]
            done: [E] bool
            info: dict of tensors
        """
        E, N_max = self.E, self.N_max
        dev = self.device
        dt = self.dt

        # ----------------------------------------------------------------
        # 1. Apply target motion (CV + Poisson turn)
        # ----------------------------------------------------------------
        self._update_target_kinematics()

        # ----------------------------------------------------------------
        # 2. Jammer step: compute jam_level applied to currently-lased target
        # ----------------------------------------------------------------
        red_task_hist = action["task_alloc"].unsqueeze(1)  # [E, 1, 4] (single team)
        if jammer is not None:
            jam_history = self._last_jam.unsqueeze(1) if hasattr(self, "_last_jam") and self._last_jam is not None else None
            jam_level = jammer.step(red_task_hist, jam_history).squeeze(1)  # [E]
        else:
            jam_level = torch.zeros(E, device=dev)
        self._last_jam = jam_level.clone()

        # jam_mul inflates measurement σ
        jam_mul = 1.0 + self.jam_gain * jam_level   # [E]

        # ----------------------------------------------------------------
        # 3. Apply measurements: fused Kalman update per target
        # ----------------------------------------------------------------
        self._update_trackers(jam_mul)

        # ----------------------------------------------------------------
        # 4. Laser dwell-kill-chain update
        # ----------------------------------------------------------------
        laser_idx = action["laser_target_idx"]                 # [E] long
        emission_on = action["emission_on"].float()            # [E]
        beam_idx = action["beam_target_idx"]                   # [E] long (for sensing emphasis)

        trace_P = self.tracker_P[..., 0, 0] + self.tracker_P[..., 2, 2]   # [E, N_max]
        # Per-target tau_track based on current geometry CRLB
        if self.use_per_target_tau:
            tau_per_target = self._compute_per_target_tau(jam_mul)  # [E, N_max]
        else:
            tau_per_target = torch.full_like(trace_P, self.tau_track)
        track_ok_all = (trace_P < tau_per_target) & self.target_alive_mask  # [E, N_max]

        # Gather laser target's track status
        laser_track_ok = torch.gather(
            track_ok_all, 1, laser_idx.unsqueeze(1)).squeeze(1)  # [E] bool
        laser_alive = torch.gather(
            self.target_alive_mask, 1, laser_idx.unsqueeze(1)).squeeze(1)  # [E] bool

        # E accumulation: only if lased target is real, alive, and track_ok
        accum_mask = laser_track_ok & laser_alive & (emission_on > 0.5)
        # For targets that are alive but track lost → decay
        # For the lased+track_ok target → accumulate
        delta_E = torch.zeros(E, N_max, device=dev)
        delta_E[laser_idx.unsqueeze(1) == torch.arange(N_max, device=dev).unsqueeze(0)] = 0.0
        # Vectorized: scatter dwell into laser target slot
        accum_dt = self.dwell_rate * dt * accum_mask.float()    # [E]
        # Decay applied to ALL alive targets whose track_ok is False
        decay_mask = (~track_ok_all) & self.target_alive_mask   # [E, N_max]

        # Apply accumulation to laser target
        # Build one-hot and multiply
        laser_onehot = torch.zeros(E, N_max, device=dev)
        laser_onehot.scatter_(1, laser_idx.unsqueeze(1), 1.0)
        accum_per_target = laser_onehot * accum_dt.unsqueeze(1)  # [E, N_max]
        # Add to E (only for alive)
        new_E = self.target_E + accum_per_target
        # Apply decay (only for alive targets with track lost)
        new_E = torch.where(decay_mask, new_E * self.decay_factor, new_E)
        # Zero out dead targets
        new_E = torch.where(self.target_alive_mask, new_E, torch.zeros_like(new_E))
        self.target_E = new_E

        # Kill: E >= e_kill → target dies
        new_kill = (self.target_E >= self.e_kill) & self.target_alive_mask
        # Update alive mask
        self.target_alive_mask = self.target_alive_mask & (~new_kill)
        self.target_killed = self.target_killed | new_kill
        n_new_kills = new_kill.float().sum(dim=1)  # [E]

        # Track first-kill step (only set once per env, when first kill occurs)
        any_new_kill_this_step = new_kill.any(dim=1)
        not_yet_killed = self.first_kill_step >= self.episode_steps
        update_mask = any_new_kill_this_step & not_yet_killed
        self.first_kill_step = torch.where(
            update_mask,
            self.step_idx + torch.ones_like(self.step_idx),  # 1-indexed: step at which kill registered
            self.first_kill_step,
        )

        # ----------------------------------------------------------------
        # 5. Exposure update
        # ----------------------------------------------------------------
        # emit_power scales with active subarrays (n_subarrays * emission_on)
        emit_increment = self.emit_power_per_subarray * self.n_subarrays * emission_on * dt
        self.exposure = self.exposure + emit_increment

        # Stochastic home-on-jam
        exposure_norm = self.exposure / 100.0
        p_homejam = 1.0 - torch.exp(-self.exposure_gain * exposure_norm * 0.001)
        p_homejam = p_homejam.clamp(0.0, 0.99)
        rand_draw = torch.rand(E, generator=self._g, device=dev)
        homejam_death = (rand_draw < p_homejam) & self.own_alive
        # First-time death: set own_alive False
        newly_dead = homejam_death & self.own_alive
        self.own_alive = self.own_alive & (~homejam_death)

        # ----------------------------------------------------------------
        # 6. Reward
        # ----------------------------------------------------------------
        # Per-step reward components
        r_kill = self.w_kill * n_new_kills
        r_surv = self.w_survival * self.own_alive.float() * (1.0 / float(self.episode_steps))
        r_exp = -self.w_exposure * emit_increment
        r_ttk = -self.w_ttk * self.target_alive_mask.float().sum(dim=1) / float(self.episode_steps)
        r_death = -self.race_death_penalty * newly_dead.float()

        reward = r_kill + r_surv + r_exp + r_ttk + r_death

        # ----------------------------------------------------------------
        # 7. Done conditions
        # ----------------------------------------------------------------
        self.step_idx = self.step_idx + 1
        time_up = self.step_idx >= self.episode_steps
        all_killed = (self.target_alive_mask.sum(dim=1) == 0) & (self.target_n_actual > 0)
        own_dead = ~self.own_alive
        done = time_up | all_killed | own_dead

        # ----------------------------------------------------------------
        # 8. Build obs + info
        # ----------------------------------------------------------------
        obs_dict = self._build_obs(
            task_alloc=action["task_alloc"],
            jam_level=jam_level,
        )

        info = {
            "n_kills_step": n_new_kills,
            "n_alive_targets": self.target_alive_mask.float().sum(dim=1),
            "n_total_targets": self.target_n_actual.float(),
            "exposure": self.exposure.clone(),
            "trace_P_mean": (trace_P * self.target_alive_mask.float()).sum(dim=1) /
                            (self.target_alive_mask.float().sum(dim=1).clamp(min=1.0)),
            "track_loss_rate": ((~track_ok_all) & self.target_alive_mask).float().sum(dim=1) /
                               (self.target_alive_mask.float().sum(dim=1).clamp(min=1.0)),
            "tau_per_target_mean": (tau_per_target * self.target_alive_mask.float()).sum(dim=1) /
                                    (self.target_alive_mask.float().sum(dim=1).clamp(min=1.0)),
            "laser_track_ok": laser_track_ok.float(),
            "homejam_death": newly_dead.float(),
            "jam_level": jam_level.clone(),
            "E_progress_mean": (self.target_E * self.target_alive_mask.float()).sum(dim=1) /
                               (self.target_alive_mask.float().sum(dim=1).clamp(min=1.0)) / max(self.e_kill, 1e-6),
            "time_to_kill_first": self.first_kill_step.float().clone(),
            "step_idx": self.step_idx.clone(),
        }
        self._last_info = info
        return obs_dict, reward, done, info

    # ------------------------------------------------------------------
    # Target kinematics update (CV + Poisson turn)
    # ------------------------------------------------------------------
    def _update_target_kinematics(self):
        E, N_max = self.E, self.N_max
        dev = self.device
        dt = self.dt

        # CV propagation
        self.target_pos = self.target_pos + self.target_vel * dt

        # Poisson turn: with p_turn_per_step, rotate velocity by random angle
        turn_event = torch.rand(E, N_max, generator=self._g, device=dev) < self.p_turn_per_step
        # Only turn alive targets
        turn_event = turn_event & self.target_alive_mask
        # Turn angle: ±turn_rate_deg_s * dt
        tr_low = math.radians(self.turn_rate_deg_s[0]) * dt
        tr_high = math.radians(self.turn_rate_deg_s[1]) * dt
        turn_angle = (torch.rand(E, N_max, generator=self._g, device=dev) * 2.0 - 1.0) * tr_high
        # Apply rotation
        cos_t = torch.cos(turn_angle)
        sin_t = torch.sin(turn_angle)
        vx, vy = self.target_vel[..., 0], self.target_vel[..., 1]
        new_vx = cos_t * vx - sin_t * vy
        new_vy = sin_t * vx + cos_t * vy
        # Only apply where turn_event
        self.target_vel[..., 0] = torch.where(turn_event, new_vx, self.target_vel[..., 0])
        self.target_vel[..., 1] = torch.where(turn_event, new_vy, self.target_vel[..., 1])

        # Process noise (small random accel)
        noise_ax = torch.randn(E, N_max, generator=self._g, device=dev) * self.sigma_q * dt
        noise_ay = torch.randn(E, N_max, generator=self._g, device=dev) * self.sigma_q * dt
        # Only apply to alive
        noise_ax = noise_ax * self.target_alive_mask.float()
        noise_ay = noise_ay * self.target_alive_mask.float()
        self.target_vel[..., 0] = self.target_vel[..., 0] + noise_ax
        self.target_vel[..., 1] = self.target_vel[..., 1] + noise_ay

        # Targets that pass the radars (y < -500m) are "missed" — keep them alive
        # for tracking purposes but they don't score kills. For simplicity in WP0,
        # we don't model target scoring; episode ends on time-up or all-killed.

    # ------------------------------------------------------------------
    # Tracker update (fused multi-radar Kalman)
    # ------------------------------------------------------------------
    def _update_trackers(self, jam_mul: torch.Tensor):
        """Fused Kalman update for all alive targets using both radars.

        Args:
            jam_mul: [E] scalar multiplier on measurement σ
        """
        E, N_max = self.E, self.N_max
        dev = self.device
        dt = self.dt

        # Build F, Q (constant)
        F = torch.tensor([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=dev)
        q = (self.sigma_q ** 2)
        Q = torch.tensor([
            [q*dt**4/4, q*dt**3/2, 0, 0],
            [q*dt**3/2, q*dt**2,   0, 0],
            [0, 0, q*dt**4/4, q*dt**3/2],
            [0, 0, q*dt**3/2, q*dt**2],
        ], dtype=torch.float32, device=dev)

        # Predict step (all targets; we'll mask later)
        # tracker_x: [E, N_max, 4]; tracker_P: [E, N_max, 4, 4]
        x_pred = torch.einsum("ij,enj->eni", F, self.tracker_x)
        P_pred = torch.einsum("ij,enjk->enik", F, self.tracker_P)
        P_pred = torch.einsum("enik,kl->enil", P_pred, F.T) + Q.unsqueeze(0).unsqueeze(0)

        # Generate measurements from both radars (with jam_mul inflation)
        # Measurement: (range, bearing) per radar
        # range_r = ||target - radar_r||, bearing_r = atan2(dy, dx)
        for r_idx in range(2):
            r_pos = self.radar_pos[:, r_idx:r_idx+1, :].expand(E, N_max, 2)  # [E, N_max, 2]
            dx = self.target_pos[..., 0] - r_pos[..., 0]
            dy = self.target_pos[..., 1] - r_pos[..., 1]
            R_true = torch.sqrt(dx*dx + dy*dy + 1.0)

            # Measurement noise (with jam_mul)
            sig_r = self.range_sigma_m * jam_mul.unsqueeze(-1)         # [E, N_max]
            if self.use_range_bearing:
                sig_b = self.bearing_sigma_rad * jam_mul.unsqueeze(-1)
            else:
                sig_b = R_true * self.crossrange_factor * jam_mul.unsqueeze(-1)

            # Generate noisy measurement
            noise_r = torch.randn(E, N_max, generator=self._g, device=dev) * sig_r
            noise_b = torch.randn(E, N_max, generator=self._g, device=dev) * sig_b
            range_meas = R_true + noise_r
            bearing_true = torch.atan2(dy, dx)
            bearing_meas = bearing_true + noise_b

            # Jacobian H at predicted state
            dx_p = x_pred[..., 0] - r_pos[..., 0]    # [E, N_max]
            dy_p = x_pred[..., 2] - r_pos[..., 1]
            R_pred = torch.sqrt(dx_p*dx_p + dy_p*dy_p + 1.0)
            if self.use_range_bearing:
                # H = [[dx/R, 0, dy/R, 0],
                #      [-dy/R^2, 0, dx/R^2, 0]]
                H = torch.zeros(E, N_max, 2, 4, device=dev)
                H[..., 0, 0] = dx_p / R_pred
                H[..., 0, 2] = dy_p / R_pred
                H[..., 1, 0] = -dy_p / (R_pred * R_pred)
                H[..., 1, 2] = dx_p / (R_pred * R_pred)
                # R_meas covariance
                R_cov = torch.zeros(E, N_max, 2, 2, device=dev)
                R_cov[..., 0, 0] = sig_r * sig_r
                R_cov[..., 1, 1] = sig_b * sig_b
                # Predicted measurement
                z_pred = torch.stack([
                    R_pred,
                    torch.atan2(dy_p, dx_p),
                ], dim=-1)  # [E, N_max, 2]
                z_meas = torch.stack([range_meas, bearing_meas], dim=-1)
                # Bearing innovation wrap
                innov = z_meas - z_pred
                innov[..., 1] = _angle_wrap(innov[..., 1])
            else:
                # (range, crossrange) — crossrange = perpendicular component
                # crossrange = (-dy * x + dx * y) / R? For 2D use projection on perp axis
                # Simplified: crossrange = (perpendicular offset)
                # Use H = [[dx/R, 0, dy/R, 0], [-dy/R, 0, dx/R, 0]]
                H = torch.zeros(E, N_max, 2, 4, device=dev)
                H[..., 0, 0] = dx_p / R_pred
                H[..., 0, 2] = dy_p / R_pred
                H[..., 1, 0] = -dy_p / R_pred
                H[..., 1, 2] = dx_p / R_pred
                R_cov = torch.zeros(E, N_max, 2, 2, device=dev)
                R_cov[..., 0, 0] = sig_r * sig_r
                R_cov[..., 1, 1] = sig_b * sig_b
                # Predicted measurement
                # crossrange_pred = (-dy * x_pred + dx * y_pred) / R_pred
                cross_pred = (-dy_p * x_pred[..., 0] + dx_p * x_pred[..., 2]) / R_pred
                z_pred = torch.stack([R_pred, cross_pred], dim=-1)
                cross_meas = (-dy * self.target_pos[..., 0] + dx * self.target_pos[..., 1]) / R_true + noise_b
                z_meas = torch.stack([range_meas, cross_meas], dim=-1)
                innov = z_meas - z_pred

            # Kalman update (per-target, vectorized)
            # S = H P H^T + R    [E, N_max, 2, 2]
            HP = torch.einsum("enij,enjk->enik", H, P_pred)        # [E, N_max, 2, 4]
            S = torch.einsum("enij,enkj->enik", HP, H) + R_cov      # [E, N_max, 2, 2]
            try:
                S_inv = torch.linalg.inv(S + 1e-9 * torch.eye(2, device=dev))
            except Exception:
                S_inv = torch.linalg.pinv(S)
            # K = P H^T S^-1   [E, N_max, 4, 2]
            PHt = torch.einsum("enij,enkj->enik", P_pred, H)        # P H^T [E,N,4,2]
            K = torch.einsum("enij,enjk->enik", PHt, S_inv)
            # Update: x += K innov
            x_upd = x_pred + torch.einsum("enij,enj->eni", K, innov)
            # P_upd = (I - K H) P_pred
            KH = torch.einsum("enij,enjk->enik", K, H)              # [E, N_max, 4, 4]
            I4 = torch.eye(4, device=dev).unsqueeze(0).unsqueeze(0).expand(E, N_max, 4, 4)
            P_upd = torch.einsum("enij,enjk->enik", I4 - KH, P_pred)

            # Symmetrize P
            P_upd = 0.5 * (P_upd + torch.einsum("enij->enji", P_upd))

            # Only update alive targets
            alive_mask = self.target_alive_mask.float().unsqueeze(-1)  # [E, N_max, 1]
            x_pred = torch.where(alive_mask.expand_as(x_pred).bool(), x_upd, x_pred)
            P_pred = torch.where(alive_mask.unsqueeze(-1).expand_as(P_pred).bool(), P_upd, P_pred)

        # Commit
        self.tracker_x = x_pred
        self.tracker_P = P_pred

    # ------------------------------------------------------------------
    # Compute time-to-kill (first kill step)
    # ------------------------------------------------------------------
    def _compute_ttk(self) -> torch.Tensor:
        """Return step_idx at which first kill occurred (or episode_steps if none)."""
        # We track this lazily by checking step_idx against target_killed
        # For simplicity, return current step_idx if any killed, else max
        any_killed = self.target_killed.any(dim=1)
        ttk = torch.where(any_killed, self.step_idx,
                          torch.full_like(self.step_idx, self.episode_steps))
        return ttk.float()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_obs(self) -> Dict[str, torch.Tensor]:
        return self._build_obs()

    @property
    def num_envs(self) -> int:
        return self.E

    @property
    def n_targets_actual(self) -> int:
        return self.N
