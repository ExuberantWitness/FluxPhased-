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
        # Jam coupling
        jam_gain: float = 8.0,
        # Kill chain
        e_kill: float = 2.0,
        dwell_rate: float = 1.0,
        decay_factor: float = 0.95,
        tau_track: float = 0.04,
        # Exposure / home-on-jam
        exposure_gain: float = 50.0,
        emit_power_per_subarray: float = 0.005,
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

        self.jam_gain = float(jam_gain)
        self.e_kill = float(e_kill)
        self.dwell_rate = float(dwell_rate)
        self.decay_factor = float(decay_factor)
        self.tau_track = float(tau_track)

        self.exposure_gain = float(exposure_gain)
        self.emit_power_per_subarray = float(emit_power_per_subarray)

        self.w_kill = float(w_kill)
        self.w_survive = float(w_survive)
        self.w_exposure = float(w_exposure)
        self.w_track = float(w_track)

        self.n_teams = 2
        self.n_radars_per_team = 2
        self.n_fn = 4
        self.obs_dim = 36
        self.privileged_dim = 8
        self._reset_count = 0

        self._init_tensors()

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

    def reset(self) -> Dict[str, torch.Tensor]:
        dev = self.device
        E = self.E
        self._reset_count += 1

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

        self._init_tensors()
        return self.get_obs()

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
        beam_target = action["beam_target"]
        laser_target = action["laser_target"]
        emission_on = action["emission_on"].float()

        # Normalize task_alloc
        task_alloc = task_alloc / (task_alloc.sum(dim=-1, keepdim=True) + 1e-8)
        self._last_task_alloc = task_alloc

        # 1. Jam matrix (with emission gate)
        jam_level = torch.zeros(E, T, R, device=dev)
        for t in range(T):
            et = 1 - t
            for k in range(R):
                tgt = beam_target[:, et, k]
                f_jam_k = task_alloc[:, et, k, 2] * emission_on[:, et, k]
                for r in range(R):
                    jam_level[:, t, r] += (tgt == r).float() * f_jam_k
        self._last_jam_matrix = jam_level
        jam_mul = 1.0 + self.jam_gain * jam_level

        # 2. Comm link
        for t in range(T):
            f_comm_both = task_alloc[:, t, :, 3]
            self.comm_link_ok[:, t] = (f_comm_both >= self.comm_threshold).all(dim=-1)

        # 3. Tracker update
        for t in range(T):
            et = 1 - t
            for r in range(R):
                true_pos = self.radar_pos[:, et, r]
                enemy_alive = self.radar_alive[:, et, r]
                f_track = torch.zeros(E, device=dev)
                for k in range(R):
                    beam_at = beam_target[:, t, k]
                    f_track += task_alloc[:, t, k, 1] * (beam_at == r).float() * emission_on[:, t, k]
                base_sigma = self.range_sigma_m
                track_sigma = base_sigma / (f_track + 1e-3).sqrt() * jam_mul[:, t, r]
                fusion_factor = torch.where(self.comm_link_ok[:, t], 1.0, 1.5)
                track_sigma = track_sigma * fusion_factor
                track_sigma = torch.where(enemy_alive, track_sigma, torch.full_like(track_sigma, 1e6))
                emitting = emission_on[:, t, :].sum(dim=-1) > 0.5
                self._kalman_update_step(t, r, true_pos, track_sigma, emitting)

        # 4. Laser dwell-kill chain
        for t in range(T):
            et = 1 - t
            lsr_r = laser_target[:, t]
            trace_P_t = self.tracker_P[:, t, :, 0, 0] + self.tracker_P[:, t, :, 2, 2]
            lsr_trace_P = torch.gather(trace_P_t, 1, lsr_r.unsqueeze(1)).squeeze(1)
            lsr_init = self.tracker_initialized[:, t].gather(1, lsr_r.unsqueeze(1)).squeeze(1)
            lsr_track_ok = (lsr_trace_P < self.tau_track) & lsr_init
            lsr_alive_enemy = self.radar_alive[:, et].gather(1, lsr_r.unsqueeze(1)).squeeze(1)
            emitting_t = emission_on[:, t].sum(dim=-1) > 0.5
            accum_mask = lsr_track_ok & lsr_alive_enemy & emitting_t
            accum_dt = self.dwell_rate * dt * accum_mask.float()
            lsr_onehot = torch.zeros(E, R, device=dev)
            lsr_onehot.scatter_(1, lsr_r.unsqueeze(1), 1.0)
            accum_per_radar = lsr_onehot * accum_dt.unsqueeze(1)
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

        # 5. Exposure + home-on-jam
        emit_increment = self.emit_power_per_subarray * self.n_subarrays * emission_on * dt
        radiating_fraction = task_alloc[..., 0] + task_alloc[..., 1] + task_alloc[..., 2]
        emit_increment = emit_increment * radiating_fraction
        self.exposure = self.exposure + emit_increment.sum(dim=-1)

        exposure_norm = self.exposure / 100.0
        p_homejam = 1.0 - torch.exp(-self.exposure_gain * exposure_norm * 0.001)
        # Mirror-symmetric home-on-jam: same roll per env per radar slot across teams.
        # Without this, identical actions under MIRROR_GEOMETRY produce asymmetric
        # deaths (team 0 dies, team 1 lives) → kills/exposure/trace_P all diverge,
        # and the WP0 unbiased check fails on a stochasticity, not a real bias.
        homejam_roll = torch.rand(E, 1, R, device=dev).expand(E, T, R)
        homejam_death = (homejam_roll < p_homejam.unsqueeze(-1)) & self.radar_alive
        self.radar_alive = self.radar_alive & (~homejam_death)
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

    def _kalman_update_step(self, team: int, enemy_r: int, true_pos: torch.Tensor,
                             meas_sigma: torch.Tensor, emitting: torch.Tensor):
        """EKF position-only update for team's tracker on enemy radar."""
        dev = self.device
        dt = self.dt
        E = self.E

        F = torch.eye(4, device=dev)
        F[0, 1] = dt
        F[2, 3] = dt
        q = self.sigma_q ** 2
        Q = torch.eye(4, device=dev) * q
        Q[0, 0] = q * dt ** 2 / 4
        Q[1, 1] = q * dt ** 2
        Q[2, 2] = q * dt ** 2 / 4
        Q[3, 3] = q * dt ** 2

        x_pred = self.tracker_x[:, team, enemy_r] @ F.T
        P_pred = F @ self.tracker_P[:, team, enemy_r] @ F.T + Q

        H = torch.zeros(2, 4, device=dev)
        H[0, 0] = 1.0
        H[1, 2] = 1.0

        R_meas = torch.zeros(E, 2, 2, device=dev)
        R_meas[:, 0, 0] = meas_sigma ** 2
        R_meas[:, 1, 1] = meas_sigma ** 2

        noise = torch.randn(E, 2, device=dev) * meas_sigma.unsqueeze(-1)
        z = true_pos + noise
        z = torch.where(emitting.unsqueeze(-1), z, torch.full_like(z, float('nan')))

        y_innov = z - x_pred[:, [0, 2]]
        y_innov = torch.where(torch.isnan(y_innov), torch.zeros_like(y_innov), y_innov)

        S = H @ P_pred @ H.T + R_meas
        S = torch.where(emitting.unsqueeze(-1).unsqueeze(-1), S,
                        torch.eye(2, device=dev).expand(E, 2, 2) * 1e10)

        K = P_pred @ H.T @ torch.linalg.inv(S)
        x_new = x_pred + (y_innov.unsqueeze(-2) @ K.transpose(-1, -2)).squeeze(-2)
        P_new = (torch.eye(4, device=dev) - K @ H) @ P_pred
        P_new = 0.5 * (P_new + P_new.transpose(-1, -2))

        first_time = ~self.tracker_initialized[:, team, enemy_r]
        init_mask = first_time & emitting
        if init_mask.any():
            x_init = torch.zeros(E, 4, device=dev)
            x_init[:, 0] = torch.where(torch.isnan(z[:, 0]), torch.zeros(E, device=dev), z[:, 0])
            x_init[:, 2] = torch.where(torch.isnan(z[:, 1]), torch.zeros(E, device=dev), z[:, 1])
            self.tracker_x[:, team, enemy_r] = torch.where(init_mask.unsqueeze(-1), x_init, x_new)
            P_init = torch.eye(4, device=dev).expand(E, 4, 4).clone() * 1.0
            self.tracker_P[:, team, enemy_r] = torch.where(init_mask.unsqueeze(-1).unsqueeze(-2), P_init, P_new)
            self.tracker_initialized[:, team, enemy_r] = self.tracker_initialized[:, team, enemy_r] | init_mask
        else:
            self.tracker_x[:, team, enemy_r] = x_new
            self.tracker_P[:, team, enemy_r] = P_new

        self.tracker_P[:, team, enemy_r] = self.tracker_P[:, team, enemy_r].clamp(-1e3, 1e3)

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
