"""Laser precise-kill reward shaper.

Migrated from `training/train_laser.py` lines 228-280, 640-707 to make laser
reward logic reusable by TeamPPOTrainer / FluxLeague.

Reward components (per team t, per env e):
  1. Beam guidance:    beam_guidance_weight × log(r_ref / max(r, r_floor))
                       — monotone, non-saturating potential on intended aim.
  2. Fire-lock bonus:  + fire_lock_bonus if firing AND within kill_radius.
  3. Misfire penalty:  - misfire_penalty if firing AND outside kill_radius.
  4. Sustained dwell:  + illum_reward_weight × (t / t_max)² when fire-locked.
                       t accumulates only while fire-gated within kill_radius;
                       resets if drifts out or stops firing.
  5. Kill bonus:       + kill_bonus (already in env.commander_rewards).
  6. Death penalty:    - death_penalty (already in env.commander_rewards).
  7. Emission cost:    - |emission_cost| per radar (per-step resource use).
  8. EW race terms:    - race_time_cost per step (game undecided)
                       - race_death_penalty the step own radar dies
                       - jam_cost × jam_level (jamming is not free)

Duck-typed to match `DenseRewardShaper.__call__(step_output)` signature, but
returns full reward override (radar_rewards + commander_rewards) since the
laser task replaces both. The TeamPPOTrainer must check task_type and use the
override path instead of `total_shaped + result["radar_rewards"]`.
"""

from __future__ import annotations

import torch
from typing import Optional


__all__ = ["LaserRewardShaper"]


class LaserRewardShaper:
    """Laser precise-kill reward shaper.

    Replaces DenseRewardShaper when `task_type == "laser"`. Returns full reward
    override (radar_rewards, commander_rewards, total_shaped).

    Lifetime:
      - Constructed once per trainer (config params from laser_cfg).
      - `reset_episode()` called at the start of each episode (clears per-episode
        state: _beam_hit_time, _jam_level).
      - `update_jam(commander_action)` called per step to extract jam from
        action dim 4 (if EW is enabled and commander outputs jam).
      - `__call__(result)` called per step to compute rewards.
    """

    def __init__(
        self,
        laser_cfg: dict,
        env=None,
        device: str = "cuda",
    ):
        rc = laser_cfg.get("reward_shaping", {})
        # Beam guidance (log-distance potential, non-saturating to r_floor)
        self.beam_guidance_weight = float(rc.get("beam_guidance_weight", 5.0))
        self.beam_r_ref_m = float(rc.get("beam_r_ref_m", 3000.0))
        self.beam_r_floor_m = float(rc.get("beam_r_floor_m", 0.2))

        # Fire-gated illumination (the actual kill mechanic — 20 cont. locked pulses)
        self.illum_reward_weight = float(rc.get("illum_reward_weight", 50.0))
        self.fire_lock_bonus = float(rc.get("fire_lock_bonus", 5.0))
        self.misfire_penalty = float(rc.get("misfire_penalty", 0.5))

        # Tier 1.2: Fire-commitment reward (independent of aim quality).
        # Grow superlinearly with consecutive fire_on steps to strongly encourage
        # the policy to commit fire once it starts. This breaks the "50% Bernoulli
        # init → never sustains 4 control steps" failure mode.
        # Reward per step: fire_commitment_weight × (streak/streak_cap)^p
        self.fire_commitment_weight = float(rc.get("fire_commitment_weight", 8.0))
        self.fire_commitment_cap = float(rc.get("fire_commitment_cap", 4.0))  # ~4 control steps = 20 pulses
        self.fire_commitment_exp = float(rc.get("fire_commitment_exp", 2.0))
        # Tier 1.2: Make sustained-dwell reward superlinear exponent configurable
        # (higher exponent → more reward for nearly-complete dwell → stronger pull to finish).
        self.illum_dwell_exp = float(rc.get("illum_dwell_exp", 2.0))

        # Aim-progress reward ("dart-board" dense signal, v2 ablation).
        # Per-step reward proportional to (prev_dist - cur_dist) — positive when
        # aim converges toward enemy, negative when it diverges. This is a much
        # stronger learning signal than log-distance guidance alone because it
        # directly reinforces the policy's temporal improvement, not just the
        # instantaneous aim quality.
        #   reward = aim_progress_weight × (prev_min_dist - cur_min_dist)
        # Rationale: kill_bonus is sparse (needs 4 cont. locked pulses inside
        # kill_radius); pure log-distance guidance saturates near r_floor and
        # gives no signal once aim is "good enough". The progress signal keeps
        # giving gradient as long as aim is improving, which is exactly the
        # "darts converging on bullseye" behavior we want to see in the metrics.
        self.aim_progress_weight = float(rc.get("aim_progress_weight", 0.0))
        # Clip the per-step delta so a single wildly off aim doesn't dominate.
        self.aim_progress_clip_m = float(rc.get("aim_progress_clip_m", 500.0))

        # Phase 3 v5: exponential precision ("dart-board") reward.
        # Per-step reward = dartboard_weight × exp(-min_dist / dartboard_scale_m),
        # applied only when fire_on (so policy is rewarded for firing AT the
        # target, not just being close). Compared to log-distance guidance,
        # exp(-d/scale) has much stronger gradient near d→0:
        #   d=270m, scale=5m  → exp(-54) ≈ 0     (no reward, far from target)
        #   d=10m,  scale=5m  → exp(-2)  ≈ 0.135
        #   d=1m,   scale=5m  → exp(-0.2)≈ 0.82  (near-max reward)
        # This gives PPO a clear signal: aim closer → exponentially more reward.
        # Default weight 0 = disabled (backward compat with v3/v4 configs).
        self.dartboard_weight = float(rc.get("dartboard_weight", 0.0))
        self.dartboard_scale_m = float(rc.get("dartboard_scale_m", 5.0))

        # Kill / death (env already includes these in commander_rewards; kept here
        # for back-compat if laser_cfg chooses to override).
        self.kill_bonus = float(rc.get("kill_bonus", 100.0))
        self.death_penalty = float(rc.get("death_penalty", -10.0))

        # Emission cost (per-radar, per-step)
        self.emission_cost = float(rc.get("emission_cost", -0.001))

        # EW race terms
        self.race_time_cost = float(rc.get("race_time_cost", 0.0))
        self.race_death_penalty = float(rc.get("race_death_penalty", 0.0))
        self.jam_cost = float(rc.get("jam_cost", 0.0))

        # t_max: 2ms dwell required for a kill (laser illumination_time_s)
        self.t_max = (
            float(env.battlefield.laser.illumination_time_s)
            if env is not None and hasattr(env, "battlefield")
            else 0.002
        )

        self.device = torch.device(device)
        self.env = env

        # Per-episode state
        self._beam_hit_time: Optional[torch.Tensor] = None  # [E, n_teams]
        self._jam_level: Optional[torch.Tensor] = None      # [E, n_teams]
        # Tier 1.2: consecutive fire_on streak (resets when fire_on=False)
        self._fire_streak: Optional[torch.Tensor] = None    # [E, n_teams]
        # v2 ablation: previous-step min_dist per team (for aim-progress reward)
        self._prev_min_dist: Optional[torch.Tensor] = None  # [E, n_teams]
        self.E: Optional[int] = None
        self.n_teams: Optional[int] = None

    # ------------------------------------------------------------------
    # Per-episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self, E: int, n_teams: int):
        """Call at episode start to clear per-episode state."""
        self.E = E
        self.n_teams = n_teams
        self._beam_hit_time = torch.zeros(E, n_teams, device=self.device)
        self._jam_level = torch.zeros(E, n_teams, device=self.device)
        self._fire_streak = torch.zeros(E, n_teams, device=self.device)
        # Initialize prev_min_dist to NaN so the first step's progress reward
        # is zero (no baseline to compare against); replaced on first call.
        self._prev_min_dist = torch.full(
            (E, n_teams), float("nan"), device=self.device,
        )

    def update_jam(self, commander_action: torch.Tensor, team: int):
        """Extract jam level from commander action dim 4 for the given team.

        Args:
            commander_action: [E, n_teams, 5] or [E, 5] (single team).
            team: which team's jam to update.
        """
        if commander_action.dim() == 3 and commander_action.shape[-1] >= 5:
            self._jam_level[:, team] = commander_action[:, team, 4].clamp(0.0, 1.0)
        elif commander_action.dim() == 2 and commander_action.shape[-1] >= 5:
            self._jam_level[:, team] = commander_action[:, 4].clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def __call__(self, step_output: dict) -> dict:
        """Compute laser rewards for one env step.

        Args:
            step_output: dict from env.step(), expected to contain:
                - radar_rewards: [E, R] base radar rewards (will be augmented)
                - commander_rewards: [E, n_teams] base commander rewards
                - dones: [E] bool
                - alive: [E, R] (from env.battlefield, attached by caller)
                - commander_obs: [E, n_teams, 76] for aim info
                Or — if env is attached to the shaper, we read from env.battlefield
                directly (cleaner; avoids needing to thread everything through step_output).
        Returns:
            dict with:
                - radar_rewards: [E, R] shaped
                - commander_rewards: [E, n_teams] shaped
                - total_shaped: [E, R] (sum of additional radar shaping,
                  for compatibility with TeamPPOTrainer.store_transition)
        """
        dev = self.device
        env = self.env
        if env is None:
            raise RuntimeError(
                "LaserRewardShaper.env not set — pass env at construction "
                "or call `shaper.env = env` before first use."
            )

        E = env.num_envs
        n_teams = env.n_teams
        if self._beam_hit_time is None or self._beam_hit_time.shape[0] != E:
            self.reset_episode(E, n_teams)

        dt = float(env.pri)
        bf = env.battlefield
        drone = bf.drone
        radar_pos = env.radar_pos

        # Phase 1 diagnostic: per-step dart-board metrics (read-only).
        # Collected across teams for episode-level logging in _train_against.
        min_dist_per_team = torch.full((E, n_teams), float("nan"), device=dev)
        fire_on_per_team = torch.zeros(E, n_teams, dtype=torch.bool, device=dev)

        # Start from env's base rewards (preserve kill/death that env already credited)
        radar_rewards = step_output.get(
            "radar_rewards", torch.zeros(E, env.n_radars, device=dev),
        ).clone()
        cmd_rewards = step_output.get(
            "commander_rewards", torch.zeros(E, n_teams, device=dev),
        ).clone()

        for t in range(n_teams):
            enemy_t = 1 - t
            enemy_idx = bf.team_radar_indices[enemy_t]
            enemy_alive = bf.alive[:, enemy_idx]  # [E, R/2]

            if not enemy_alive.any():
                self._beam_hit_time[:, t] = 0.0
                continue

            enemy_pos = radar_pos[:, enemy_idx, :]  # [E, R/2, 3]
            # Use the commander's INTENDED aim (set every step regardless of fire)
            # so the guidance gradient exists even before the policy learns to fire.
            aim = drone._commander_aim[:, t, :].unsqueeze(1)  # [E, 1, 3]
            dist_all = (aim - enemy_pos).norm(dim=-1)  # [E, R/2]
            dist_all = dist_all + (~enemy_alive).float() * 1e6
            min_dist = dist_all.min(dim=-1).values  # [E]
            # [ANCHOR-E] Diagnostic: aim vs enemy_pos vs min_dist.
            # Fires for first 8 invocations per team — covers dart_min_dist_init
            # window (steps 1-8 after first_step guard) so we see what the shaper
            # actually computes when min_dist_init=139m is reported.
            cnt = getattr(self, f'_anchor_e_count_t{t}', 0)
            if cnt < 8:
                setattr(self, f'_anchor_e_count_t{t}', cnt + 1)
                aim0 = aim[0, 0].tolist()
                ep0 = enemy_pos[0, 0].tolist()
                d0 = dist_all[0, 0].item()
                md0 = min_dist[0].item()
                line = (f"[ANCHOR-E] n={cnt} team={t} aim[0]=({aim0[0]:.1f},{aim0[1]:.1f},{aim0[2]:.1f}) "
                        f"enemy0=({ep0[0]:.1f},{ep0[1]:.1f}) dist0={d0:.1f}m min_dist[0]={md0:.1f}m")
                if enemy_pos.shape[1] > 1:
                    ep1 = enemy_pos[0, 1].tolist()
                    d1 = dist_all[0, 1].item()
                    line += f" enemy1=({ep1[0]:.1f},{ep1[1]:.1f}) dist1={d1:.1f}m"
                # Show min_dist for ALL envs to find which envs pull avg up.
                md_all = min_dist.tolist()
                line += f" | all_envs min_dist={[f'{x:.1f}' for x in md_all]}"
                # And the mean for this team (compare to dart_min_dist_avg which
                # averages over both teams).
                line += f" | team_mean={min_dist.mean().item():.2f}m max={min_dist.max().item():.1f}m"
                print(line, flush=True)

            # (A) Guidance: log-distance potential, monotone & non-saturating to r_floor
            r_eff = min_dist.clamp(min=self.beam_r_floor_m)
            guidance = torch.log(self.beam_r_ref_m / r_eff).clamp(min=0.0)
            beam_reward = guidance * self.beam_guidance_weight

            # (A2) Aim-progress reward ("dart-board" signal, v2 ablation).
            # Per-step delta of min_dist: positive when aim improves, negative
            # when aim diverges. Stronger temporal gradient than (A) alone.
            if self.aim_progress_weight > 0.0:
                prev = self._prev_min_dist[:, t]  # [E], NaN on first step
                # Use nan_to_num so first-step (NaN prev) → 0 progress reward.
                progress = (prev - min_dist).clamp(
                    -self.aim_progress_clip_m, self.aim_progress_clip_m,
                )
                progress = torch.nan_to_num(progress, nan=0.0)
                beam_reward = beam_reward + self.aim_progress_weight * progress
                # Update prev for next step (only for alive envs to avoid bleed)
                self._prev_min_dist[:, t] = torch.where(
                    torch.isnan(prev),
                    min_dist,  # first step: just record, no reward next step
                    min_dist,
                )

            # (C) Fire-gated illumination
            kill_radius = float(bf.laser.kill_radius_m)  # live (curriculum)
            fire_on = drone._commander_fire[:, t]  # [E] bool
            locked = (min_dist < kill_radius) & fire_on

            # Phase 1 diagnostic: record per-team metrics.
            min_dist_per_team[:, t] = min_dist
            fire_on_per_team[:, t] = fire_on

            # Phase 1.0 改动 2b (plan T1): 全场稠密 dartboard reward.
            # 之前: 只在 fire_on 时给 (line 303 旧版),PPO init 时 fire 50% Bernoulli
            # → dartboard 信号稀疏,PPO 学不会瞄准。
            # 现在: per-step 都给 (不管 fire_on),fire_on 时额外加权 0.5×,
            # 鼓励"瞄准时开火"而非"只靠近不开火"。
            # 量级: weight=50 同 guidance,per-step max 贡献 ~50 (近距离),
            # 远距离 →0; kill_bonus=100 仍稀疏奖励。总 reward magnitude ~100-200/step。
            if self.dartboard_weight > 0.0:
                dart_r = torch.exp(-min_dist / max(self.dartboard_scale_m, 1e-3))
                # 全场稠密项(per-step 都给,不管 fire_on)
                beam_reward = beam_reward + self.dartboard_weight * dart_r
                # fire_on 时额外加权(鼓励"瞄准时开火")
                beam_reward = beam_reward + fire_on.float() * self.dartboard_weight * 0.5 * dart_r
            # (C1) Immediate dense reward for firing while locked
            beam_reward = beam_reward + locked.float() * self.fire_lock_bonus
            # (C2) Small penalty for firing while NOT locked
            misfire = fire_on & (min_dist >= kill_radius)
            beam_reward = beam_reward - misfire.float() * self.misfire_penalty
            # (C3) Sustained-dwell illumination (superlinear → strong finish pull)
            self._beam_hit_time[:, t] = torch.where(
                locked,
                self._beam_hit_time[:, t] + dt,
                torch.zeros_like(self._beam_hit_time[:, t]),
            )
            t_norm = (self._beam_hit_time[:, t] / self.t_max).clamp(0.0, 1.0)
            beam_reward = beam_reward + locked.float() * (t_norm ** self.illum_dwell_exp) * self.illum_reward_weight

            # (D) Tier 1.2: Fire-commitment reward — grows with consecutive fire_on
            # steps, independent of aim quality. This directly attacks the
            # "fire head ~50% Bernoulli at init → 0.5^4 = 6% chance of 4 cont.
            # fire_on" failure mode by giving a superlinear incentive to sustain.
            self._fire_streak[:, t] = torch.where(
                fire_on,
                self._fire_streak[:, t] + 1.0,
                torch.zeros_like(self._fire_streak[:, t]),
            )
            streak_norm = (self._fire_streak[:, t] / self.fire_commitment_cap).clamp(0.0, 1.0)
            commit_reward = (streak_norm ** self.fire_commitment_exp) * self.fire_commitment_weight
            beam_reward = beam_reward + fire_on.float() * commit_reward

            cmd_rewards[:, t] += beam_reward

            # Share a fraction of beam reward with own radars (team reward)
            own_idx = bf.team_radar_indices[t]
            for ri in own_idx:
                radar_rewards[:, ri] += beam_reward * 0.1

        # Emission cost
        radar_rewards += self.emission_cost

        # EW race terms: time cost + death penalty + jam cost
        for t in range(n_teams):
            # Fast kill: per-step time cost while game undecided
            cmd_rewards[:, t] -= self.race_time_cost * (~bf.dones).float()
            # Survive: extra penalty the step own radar dies
            own_idx = bf.team_radar_indices[t]
            own_dead = (~bf.alive[:, own_idx]).any(dim=-1).float()
            cmd_rewards[:, t] -= self.race_death_penalty * own_dead
            # Jamming costs emission (and exposes — handled in sensing)
            cmd_rewards[:, t] -= self.jam_cost * self._jam_level[:, t]

        # total_shaped: the extra radar shaping beyond env's radar_rewards.
        # TeamPPOTrainer.store_transition does `total_shaped + result["radar_rewards"]`,
        # so we return the full override here as the delta to keep that contract.
        # The cleanest path is for TeamPPOTrainer to check task_type and use
        # the full override when laser. We still populate total_shaped for compat.
        # When task_type=laser, store_transition should use:
        #     radar_reward = shaped["radar_rewards"][e, r_start:r_start+r_per_team].sum()
        # instead of `total_shaped + result["radar_rewards"]`.
        total_shaped = radar_rewards  # full override (not just delta)

        return {
            "radar_rewards": radar_rewards,
            "commander_rewards": cmd_rewards,
            "total_shaped": total_shaped,
            "beam_hit_time": self._beam_hit_time.clone(),
            "jam_level": self._jam_level.clone(),
            "fire_streak": self._fire_streak.clone(),
            # Phase 1 dart-board metrics (read-only diagnostic).
            "dart_min_dist_per_team": min_dist_per_team,            # [E, n_teams], m
            "dart_fire_on_per_team": fire_on_per_team,              # [E, n_teams], bool
            "dart_min_dist_avg":    float(min_dist_per_team[~torch.isnan(min_dist_per_team)].mean().item()) if not torch.isnan(min_dist_per_team).all() else float("nan"),
            "dart_min_dist_min":    float(min_dist_per_team[~torch.isnan(min_dist_per_team)].min().item())  if not torch.isnan(min_dist_per_team).all() else float("nan"),
            "dart_fire_rate":       float(fire_on_per_team.float().mean().item()),
            # Phase 1.0 改动 2 (plan §1.0): 离群-resistant 版本,排 2% alive-flag/zero-init
            # outlier (425-439m)。avg 被 outlier 拉到 139m 是统计假象 (见
            # EXPERIMENTAL_PHENOMENA_REPORT.md 现象 2),用 median / trim_mean(10%) 看真值。
            "dart_min_dist_median":    float(min_dist_per_team[~torch.isnan(min_dist_per_team)].median().item()) if not torch.isnan(min_dist_per_team).all() else float("nan"),
            "dart_min_dist_trim_mean": float(self._trim_mean(min_dist_per_team[~torch.isnan(min_dist_per_team)], p=0.1)) if not torch.isnan(min_dist_per_team).all() else float("nan"),
        }

    @staticmethod
    def _trim_mean(t: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Trimmed mean: drop fraction p from each tail (default p=0.1 → drop 10% each side).

        Robust to outliers (e.g., 425-439m min_dist from zero-init policy early steps).
        Falls back to median for tensors too small to trim.
        """
        if t.numel() == 0:
            return torch.tensor(float("nan"), device=t.device)
        k = int(t.numel() * p)
        if 2 * k >= t.numel():
            return t.median()
        s, _ = torch.sort(t.flatten())
        return s[k:-k].mean()
