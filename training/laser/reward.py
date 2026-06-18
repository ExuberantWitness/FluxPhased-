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

            # (A) Guidance: log-distance potential, monotone & non-saturating to r_floor
            r_eff = min_dist.clamp(min=self.beam_r_floor_m)
            guidance = torch.log(self.beam_r_ref_m / r_eff).clamp(min=0.0)
            beam_reward = guidance * self.beam_guidance_weight

            # (C) Fire-gated illumination
            kill_radius = float(bf.laser.kill_radius_m)  # live (curriculum)
            fire_on = drone._commander_fire[:, t]  # [E] bool
            locked = (min_dist < kill_radius) & fire_on
            # (C1) Immediate dense reward for firing while locked
            beam_reward = beam_reward + locked.float() * self.fire_lock_bonus
            # (C2) Small penalty for firing while NOT locked
            misfire = fire_on & (min_dist >= kill_radius)
            beam_reward = beam_reward - misfire.float() * self.misfire_penalty
            # (C3) Sustained-dwell illumination
            self._beam_hit_time[:, t] = torch.where(
                locked,
                self._beam_hit_time[:, t] + dt,
                torch.zeros_like(self._beam_hit_time[:, t]),
            )
            t_norm = (self._beam_hit_time[:, t] / self.t_max).clamp(0.0, 1.0)
            beam_reward = beam_reward + locked.float() * (t_norm ** 2) * self.illum_reward_weight

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
        }
