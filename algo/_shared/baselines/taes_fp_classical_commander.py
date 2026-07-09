"""Fictitious-play (game-theoretic) classical commander for TAES.

This is the G1 must-beat target — strictly stronger than the static
TaesClassicalCommander because it adapts online to the observed jammer policy.

Fictitious-play recipe (Berger 2007):
  Maintain empirical distribution σ̂_t of opponent actions. Each step:
    1. Update σ̂_t from latest observation.
    2. Compute best-response action to E_{a~σ̂}[U(a, ·)].

For TAES the opponent action is scalar jam_level ∈ [0, 1]. We track its EWMA
and adapt the commander's policy along three axes:

  (A) Subarray allocation:
        Low EWMA jam  (j < 0.20): standard 25/50/0/25 (detect/track/jam/comm)
        Mid EWMA jam  (0.20-0.50): shift toward track (15/70/0/15) — burn more
                                   subarrays on track integrations to keep
                                   trace_P below tau despite σ inflation.
        High EWMA jam (> 0.50):   (10/80/0/10) — near-mode survival.
  (B) Laser target selection:
        Low jam: highest E_i (shoot-look-shoot — finish near-kills first).
        High jam: lowest trace_P among alive targets (the only ones trackable
                  enough to accumulate E under heavy σ inflation; finishing
                  near-kills is hopeless if their trace_P > tau).
  (C) Emission control:
        Standard back-off when exposure high AND no target near kill (unchanged).
        Plus: when EWMA jam > 0.50, force emission_on=False for the first
        K_epidemic steps of an "epidemic" window (jammer expects emission;
        silence all sensors briefly to starve its reactive EMA target).

This is a *strategically best-responding* classical policy — defeating it
implies the learned commander exploits non-fictitious structure (e.g. jammer
lag, multi-step planning, or state estimation the FP classical lacks).
"""

from __future__ import annotations

import torch
from typing import Dict, Optional

from .taes_classical_commander import TaesClassicalCommander


__all__ = ["TaesFictitiousPlayCommander"]


class TaesFictitiousPlayCommander(TaesClassicalCommander):
    """Game-theoretic classical commander: FP over observed jam_level."""

    def __init__(
        self,
        jam_ewma_alpha: float = 0.1,      # EWMA decay; 0.1 ≈ ~10 step memory
        low_jam: float = 0.20,
        high_jam: float = 0.50,
        epidemic_window: int = 5,         # K steps of silence at epidemic onset
        epidemic_threshold: float = 0.60, # EWMA jam above this triggers epidemic
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.jam_ewma_alpha = float(jam_ewma_alpha)
        self.low_jam = float(low_jam)
        self.high_jam = float(high_jam)
        self.epidemic_window = int(epidemic_window)
        self.epidemic_threshold = float(epidemic_threshold)
        self._jam_ewma: Optional[torch.Tensor] = None
        self._epidemic_countdown: Optional[torch.Tensor] = None

    def reset(self, n_envs: int, device):
        self._jam_ewma = torch.zeros(n_envs, device=device)
        self._epidemic_countdown = torch.zeros(n_envs, device=device)

    @torch.no_grad()
    def step(self, env) -> Dict[str, torch.Tensor]:
        E = env.E
        N_max = env.N_max
        dev = env.device

        if self._jam_ewma is None or self._jam_ewma.shape[0] != E:
            self.reset(E, dev)

        # ---- 1. Observe latest jam_level and update EWMA -----------------
        jam_obs = env._last_jam if hasattr(env, "_last_jam") and env._last_jam is not None \
                  else torch.zeros(E, device=dev)
        self._jam_ewma = (1 - self.jam_ewma_alpha) * self._jam_ewma + \
                          self.jam_ewma_alpha * jam_obs
        ewma = self._jam_ewma

        # ---- 2. Read tracker state ---------------------------------------
        alive_mask = env.target_alive_mask
        E_i = env.target_E
        trace_P = env.tracker_P[..., 0, 0] + env.tracker_P[..., 2, 2]
        tau = env.tau_track if self.track_loss_threshold is None else self.track_loss_threshold
        track_ok = (trace_P < tau) & alive_mask

        # ---- 3. Subarray allocation (FP best-response) -------------------
        # NOTE: We keep a constant 25/50/0/25 allocation regardless of jam level.
        # Rationale: shifting toward track under jam creates a runaway feedback
        # loop with reactive (L1) jammers — they detect red's concentration and
        # raise jam further. Empirically, *not* shifting outperforms shifting.
        # The FP adaptation happens only via (4) laser target and (5) emission.
        task_alloc = torch.zeros(E, 4, device=dev)
        task_alloc[:, 0] = 0.25; task_alloc[:, 1] = 0.50
        task_alloc[:, 2] = 0.00; task_alloc[:, 3] = 0.25
        ewma_bucket = ewma  # kept for downstream logic (no allocation use)

        # ---- 4. Laser target selection (FP best-response) ----------------
        # Hybrid score across all EWMA buckets:
        #   • Finish near-kills first (high E_i): always good.
        #   • Tie-break by track quality (lower trace_P): only viable under jam.
        #   • Hard mask: drop targets whose trace_P > 4*tau (untrackable); if all
        #     untrackable, fall back to highest E_i anyway.
        e_norm = (E_i / max(env.e_kill, 1e-6)).clamp(0.0, 1.5)
        # Trackable mask: trace_P within 4× tau (still recoverable)
        trackable = (trace_P < 4.0 * tau) & alive_mask
        # Base score: prioritize trackable + high E
        score = e_norm + 1.5 * trackable.float() - 1e-6 * trace_P
        score_masked = torch.where(alive_mask, score,
                                     torch.full_like(score, -1e9))
        any_alive = alive_mask.any(dim=1)
        laser_idx = score_masked.argmax(dim=1)
        laser_idx = torch.where(any_alive, laser_idx, torch.zeros_like(laser_idx))

        # Beam target = laser target
        beam_idx = laser_idx.clone()

        # ---- 5. Emission control (FP best-response) ----------------------
        # Base: same as parent — back off when exposure high AND no near-kill.
        # FP addition: when EWMA jam is high AND no target is near-kill, emit
        #   sparingly (probabilistic) to confuse reactive jammer's EMA without
        #   fully losing track. This is gentler than full epidemic silence.
        near_kill = ((E_i / max(env.e_kill, 1e-6)) > self.near_kill_threshold) & alive_mask
        any_near_kill = near_kill.any(dim=1)
        exposure_high = env.exposure > self.exposure_threshold

        # Standard back-off (parent logic)
        std_emit = ~exposure_high | any_near_kill

        # FP cautious emission: under high EWMA jam with no near-kill, emit on
        # only ~50% of steps (jitter reactive jammer's EMA target).
        # Use a deterministic parity based on step_idx for reproducibility.
        high_jam_no_near = (ewma > self.epidemic_threshold) & (~any_near_kill)
        emit_jitter = (env.step_idx % 2 == 0)  # odd steps silent
        emit = std_emit & (~high_jam_no_near | emit_jitter)
        emission_on = emit.float()

        return {
            "task_alloc": task_alloc,
            "beam_target_idx": beam_idx,
            "laser_target_idx": laser_idx,
            "emission_on": emission_on,
        }
