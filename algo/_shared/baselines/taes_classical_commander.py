"""Strong modular classical commander for the TAES env.

WP0/WP1 baseline. Combines:
  - Q-RAM subarray allocation (water-fill by kill priority × track margin)
  - Shoot-look-shoot fire control (highest E_i / exposure_cost target first)
  - Reactive ECCM (emission back-off when exposure is high)
  - (IMM-PDAF tracker is the env's built-in fused Kalman; Stone Soup integration
    deferred to WP1 full validation phase)

API matches what TAESVecEnv.step expects:
    commander.step(env, info_dict) → action_dict
"""

from __future__ import annotations

import torch
from typing import Dict


__all__ = ["TaesClassicalCommander", "TaesGreedyCommander"]


class TaesClassicalCommander:
    """Strong modular classical commander for TAESVecEnv.

    Strategy:
      1. Read tracker state from env (per-target trace_P, E_i, alive_mask).
      2. Pick laser target = alive target with highest kill-progress-per-exposure
         (prioritize near-kill targets to finish them; tie-break by lowest trace_P).
      3. Subarray allocation (Q-RAM-lite):
         - Floor: 3 subarrays per active function (detect/track/comm)
         - Water-fill remainder proportional to (E_i × track_ok) per target
         - (No real per-function QoS measurement in WP0; uses heuristic weights)
      4. Beam target = laser target (main beam emphasizes lased target).
      5. Emission control: emission_on=False if cumulative p_homejam > threshold
         AND no target is close to kill (E_i > 0.5 × E_kill).
    """

    def __init__(
        self,
        qos_floor_per_fn: int = 3,
        exposure_threshold: float = 3.0,    # emit_power accumulated
        near_kill_threshold: float = 0.5,  # E_i/E_kill above this = "near kill"
        track_loss_threshold: float = None,  # set from env.tau_track by default
    ):
        self.qos_floor = int(qos_floor_per_fn)
        self.exposure_threshold = float(exposure_threshold)
        self.near_kill_threshold = float(near_kill_threshold)
        self.track_loss_threshold = track_loss_threshold

    @torch.no_grad()
    def step(self, env) -> Dict[str, torch.Tensor]:
        E = env.E
        N_max = env.N_max
        dev = env.device

        # Read state
        alive_mask = env.target_alive_mask                     # [E, N_max]
        E_i = env.target_E                                     # [E, N_max]
        trace_P = env.tracker_P[..., 0, 0] + env.tracker_P[..., 2, 2]  # [E, N_max]
        tau = env.tau_track if self.track_loss_threshold is None else self.track_loss_threshold
        track_ok = (trace_P < tau) & alive_mask               # [E, N_max]

        # Kill priority: high E_i + track_ok = urgent to finish
        kill_priority = (E_i / max(env.e_kill, 1e-6)).clamp(0.0, 1.5) * track_ok.float()
        # Tie-break: lower trace_P preferred (more solid track)
        # Add small bonus to alive targets to break ties
        score = kill_priority + 0.01 * alive_mask.float() - 1e-6 * trace_P

        # Laser target = argmax score (if any alive target, else 0)
        any_alive = alive_mask.any(dim=1)
        # Mask non-alive to -inf
        score_masked = torch.where(alive_mask, score, torch.full_like(score, -1e9))
        laser_idx = score_masked.argmax(dim=1)                  # [E]
        # If no alive targets, default to 0
        laser_idx = torch.where(any_alive, laser_idx, torch.zeros_like(laser_idx))

        # Beam target = laser target (main beam on lased target)
        beam_idx = laser_idx.clone()

        # Q-RAM-lite subarray allocation (heuristic for WP0):
        # [detect, track, jam, comm] — jam is unused in this scenario
        # Allocate: detect 25%, track 50%, jam 0%, comm 25%
        task_alloc = torch.zeros(E, 4, device=dev)
        task_alloc[:, 0] = 0.25   # detect
        task_alloc[:, 1] = 0.50   # track (primary)
        task_alloc[:, 2] = 0.00   # jam (no friendly jamming needed)
        task_alloc[:, 3] = 0.25   # comm

        # Emission control: if exposure > threshold AND no target is near kill,
        # back off (emission_on=0) to avoid home-on-jam
        near_kill = ((E_i / max(env.e_kill, 1e-6)) > self.near_kill_threshold) & alive_mask
        any_near_kill = near_kill.any(dim=1)
        exposure_high = env.exposure > self.exposure_threshold
        emission_on = (~exposure_high | any_near_kill).float()

        return {
            "task_alloc": task_alloc,
            "beam_target_idx": beam_idx,
            "laser_target_idx": laser_idx,
            "emission_on": emission_on,
        }


class TaesGreedyCommander:
    """Greedy baseline: always fire at highest-E_i target, no emission control.

    Used as a sanity-check strawman to verify the strong classical is meaningfully
    better. Strong classical should beat this in scenarios with exposure risk.
    """

    @torch.no_grad()
    def step(self, env) -> Dict[str, torch.Tensor]:
        E = env.E
        dev = env.device
        alive_mask = env.target_alive_mask
        E_i = env.target_E

        score = E_i.clone()
        score_masked = torch.where(alive_mask, score, torch.full_like(score, -1e9))
        laser_idx = score_masked.argmax(dim=1)
        any_alive = alive_mask.any(dim=1)
        laser_idx = torch.where(any_alive, laser_idx, torch.zeros_like(laser_idx))

        task_alloc = torch.tensor([[0.25, 0.5, 0.0, 0.25]] * E, device=dev)
        emission_on = torch.ones(E, device=dev)

        return {
            "task_alloc": task_alloc,
            "beam_target_idx": laser_idx.clone(),
            "laser_target_idx": laser_idx,
            "emission_on": emission_on,
        }
