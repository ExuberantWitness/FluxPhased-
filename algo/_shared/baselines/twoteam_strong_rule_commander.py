"""Strong rule-based multifunction commander for two-team env (WP1 anti-strawman baseline).

Per TWOTEAM_MULTIFUNCTION_PLAN.md §WP1.1: anti-strawman strong rule commander.
Required properties:
  - Strong enough that "G0 exploitability PASS" is meaningful (not beating a strawman)
  - Per-aperture adaptive 4-function allocation (NOT a fixed vector)
  - Tracks own kill progress, defensive jam when enemy tracks me, comm floor, exposure duck

Strategy:
  1. laser target = argmax over alive+trackable enemy radars of E_progress
  2. task_alloc per aperture adapts:
     - boost track if laser target's trace_P > tau_track
     - boost jam if enemy tracking me well (their trace_P on me < jamreact_tau)
     - comm floor maintained
  3. beam = laser target (focus fire)
  4. emission_on = False iff exposure high AND no enemy near kill (duck home-on-jam)

API matches ExtremeCommander (per-team slice via get_action(env, team)).
"""

from __future__ import annotations
import torch
from typing import Dict


class TwoTeamStrongRuleCommander:
    """Anti-strawman strong rule-based multifunction commander."""

    def __init__(
        self,
        sharpness: float = 4.0,
        exposure_duck_threshold: float = 30.0,
        near_kill_threshold: float = 0.5,
        jamreact_tau: float = 0.04,
        comm_floor: float = 0.10,
        device: str = "cuda",
    ):
        self.sharpness = float(sharpness)
        self.exposure_duck_threshold = float(exposure_duck_threshold)
        self.near_kill_threshold = float(near_kill_threshold)
        self.jamreact_tau = float(jamreact_tau)
        self.comm_floor = float(comm_floor)
        self.device = device

    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        """Per-team action slice. Same API as ExtremeCommander.get_action."""
        E = env.E
        dev = env.device
        et = 1 - team
        R = env.n_radars_per_team

        # --- 1. trace_P of MY trackers on ENEMY radars [E, R] ---
        trace_P_me_on_enemy = (
            env.tracker_P[:, team, :, 0, 0] + env.tracker_P[:, team, :, 2, 2]
        )
        # --- trace_P of ENEMY trackers on MY radars [E, R] ---
        trace_P_enemy_on_me = (
            env.tracker_P[:, et, :, 0, 0] + env.tracker_P[:, et, :, 2, 2]
        )

        enemy_alive = env.radar_alive[:, et]   # [E, R]
        my_init = env.tracker_initialized[:, team]   # [E, R]
        E_progress = env.radar_E[:, et] / env.e_kill   # [E, R]

        # --- 2. laser target: trackable + highest E_progress ---
        trackable = (trace_P_me_on_enemy < env.tau_track) & my_init & enemy_alive   # [E, R]
        laser_score = E_progress * trackable.float() + trackable.float() * 0.01
        # If nothing trackable: fall back to highest E_progress alive
        fallback_score = E_progress * enemy_alive.float() + enemy_alive.float() * 1e-3
        any_trackable = trackable.any(dim=-1)   # [E] bool
        final_score = torch.where(
            any_trackable.unsqueeze(-1), laser_score, fallback_score)
        lt_idx = final_score.argmax(dim=-1)   # [E]

        # --- 3. per-aperture adaptive task_alloc ---
        # Base allocation: detect/track/jam/comm
        base = torch.tensor(
            [0.10, 0.45, 0.30, 0.15], device=dev).expand(E, R, 4).clone()

        # Boost track if laser target's trace_P high
        trace_P_lt = trace_P_me_on_enemy.gather(
            1, lt_idx.unsqueeze(1)).squeeze(1)   # [E]
        low_track = (trace_P_lt > env.tau_track).float()   # [E]
        low_track_b = low_track.unsqueeze(-1).expand(E, R)   # [E, R]
        base[..., 1] = base[..., 1] + 0.20 * low_track_b
        base[..., 2] = base[..., 2] - 0.10 * low_track_b

        # Boost jam if enemy tracking me
        enemy_tracking_me = (
            trace_P_enemy_on_me.min(dim=-1).values < self.jamreact_tau
        ).float()   # [E]
        etm_b = enemy_tracking_me.unsqueeze(-1).expand(E, R)   # [E, R]
        base[..., 2] = base[..., 2] + 0.20 * etm_b
        base[..., 0] = base[..., 0] - 0.10 * etm_b

        # Comm floor (clamp)
        base[..., 3] = torch.clamp(base[..., 3], min=self.comm_floor)

        # Clip negatives, then softmax with sharpness
        base = torch.clamp(base, min=1e-3)
        task_alloc = torch.softmax(base * self.sharpness, dim=-1)   # [E, R, 4]

        # Renormalize (softmax already sums to 1, but be safe)
        task_alloc = task_alloc / (task_alloc.sum(dim=-1, keepdim=True) + 1e-8)

        # --- 4. beam target: both apertures point at laser target (focus fire) ---
        beam_target = lt_idx.unsqueeze(-1).expand(E, R).clone()   # [E, R]

        # --- 5. emission_on: duck if exposure high and no near-kill ---
        E_max = E_progress.max(dim=-1).values   # [E]
        duck_mask = (
            (env.exposure[:, team] > self.exposure_duck_threshold)
            & (E_max < self.near_kill_threshold)
        )   # [E] bool
        emit = (~duck_mask).float()   # [E]
        emission_on = emit.unsqueeze(-1).expand(E, R)   # [E, R] both apertures same

        return {
            "task_alloc": task_alloc,           # [E, R, 4]
            "beam_target": beam_target,         # [E, R] long
            "laser_target": lt_idx,             # [E] long
            "emission_on": emission_on,         # [E, R] float
        }
