"""Reactive jammer commander for two-team env (WP-C R2 dynamic enemy).

Per WP-C handoff: "敌方动态干扰(非固定 action jammer)逼迫己方换信道 →
制造'没有 trivial 固定解'的协同难度".

Strategy — "狠但合理"的 reactive enemy:
  1. Each step, identify the victim radar with the best track quality
     (min trace_P < tau_track AND alive AND initialized) — that's where the
     victim's coordination is paying off; jam that radar specifically.
  2. Set jam beam_target at that radar (focus fire).
  3. Tune aggressiveness via jam_fraction (fraction of aperture power to jam).
  4. Channel-following: enemy channel_select = victim's best-tracked radar's
     channel index (derived from env.radar_freq_hz). Implemented as a per-team
     action slice that env.step consumes — no wrapper-side env mutation needed.

The point: a StrongRule+orth baseline with FIXED ch0/ch1 assignment gets
tracked by this reactive jammer within a few steps, forcing the victim to
DYNAMICALLY re-allocate channels — the WP-C coordination skill RL must learn.

API matches StrongRule.get_action (env, team) -> dict.
"""

from __future__ import annotations
import torch
from typing import Dict


class ReactiveJammerCommander:
    """Reactive channel-following jammer commander (dynamic enemy)."""

    def __init__(
        self,
        jam_fraction: float = 0.6,
        comm_floor: float = 0.10,
        track_victim_min_trace_P: float = -1.0,   # any trace_P, ranked low→high
        device: str = "cuda",
    ):
        """
        jam_fraction: fraction of aperture power to jam (rest to comm).
        """
        self.jam_fraction = float(jam_fraction)
        self.comm_floor = float(comm_floor)
        self.track_victim_min_trace_P = float(track_victim_min_trace_P)
        self.device = device

    def _pick_target_victim_radar(self, env, et: int) -> torch.Tensor:
        """Returns [E] target victim-radar index per env (best-tracked = lowest trace_P)."""
        victim_trace_P = (
            env.tracker_P[:, et, :, 0, 0] + env.tracker_P[:, et, :, 2, 2]
        )
        victim_alive = env.radar_alive[:, et]
        victim_init = env.tracker_initialized[:, et]

        big = torch.full_like(victim_trace_P, 1e6)
        score_track = torch.where(victim_init & victim_alive, victim_trace_P, big)
        alive_big = torch.where(victim_alive, torch.zeros_like(victim_trace_P), big)
        any_track = (score_track < 1e5).any(dim=-1)
        target_idx_track = score_track.argmin(dim=-1)
        target_idx_any = alive_big.argmin(dim=-1)
        return torch.where(any_track, target_idx_track, target_idx_any)

    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        """Per-team action slice. Enemy strategy: jam victim's best-tracked radar
        AND follow that radar's channel."""
        E = env.E
        dev = env.device
        et = 1 - team
        R = env.n_radars_per_team

        target_idx = self._pick_target_victim_radar(env, et)   # [E]

        # task_alloc: jam-only (f_emit == jam_fraction for apple-to-apple).
        # NOTE: do NOT clamp base at 1e-3 — that would raise jam_fraction < 1e-3
        # to 1e-3, breaking apple-to-apple with fixed jammer. Just normalize.
        base = torch.zeros(E, R, 4, device=dev)
        base[..., 2] = max(float(self.jam_fraction), 1e-9)
        base[..., 3] = 1.0 - base[..., 2]
        task_alloc = base.clone()   # already sums to 1 per aperture

        beam_target = target_idx.unsqueeze(-1).expand(E, R).clone()
        laser_target = torch.zeros(E, dtype=torch.long, device=dev)
        emission_on = torch.ones(E, R, device=dev)
        freq_hop_rate = torch.ones(E, R, device=dev)

        # Channel-following: enemy channel_select = victim target's channel idx.
        # All enemy radars share that channel (concentrate jam on the tracked band).
        victim_ch_idx = ((env.radar_freq_hz[:, et, :] - env.fc_hz)
                         / env.channel_spacing_hz).round().long().clamp(0, env.n_channels - 1)   # [E, R]
        target_ch_idx = victim_ch_idx.gather(1, target_idx.unsqueeze(1)).squeeze(1)   # [E]
        channel_select = target_ch_idx.unsqueeze(-1).expand(E, R).clone()

        return {
            "task_alloc": task_alloc,
            "beam_target": beam_target,
            "laser_target": laser_target,
            "emission_on": emission_on,
            "freq_hop_rate": freq_hop_rate,
            "channel_select": channel_select,
        }
