"""Adaptive spectrum jammer for two-team env (WP-C D1 dynamic enemy).

Per WP-C D1 (post-R2 0/3 fail diagnosis):
  R2 failed because the "fixed" comparison also channel-followed (it read
  env.radar_freq_hz[:, team] for channel_select, so under orth mirror config
  the fixed enemy landed on the same [ch0, ch1] as the victim). Both enemies
  ended up on the victim's channels → no apple-to-apple difference.

This module provides the TWO enemies needed for a clean D1:
  1. TRUE FIXED jammer: both enemy radars stay on one constant channel
     (default ch0) regardless of victim. Victim radar on the OTHER channel
     is free → baseline can still track with one radar.
  2. ADAPTIVE SPECTRUM jammer (channel-split follower): enemy radar i →
     victim radar i's channel + beam. Both victim channels get jammed
     simultaneously → victim cannot escape by re-allocating one radar.
     This is the truly harder dynamic enemy.

Apple-to-apple: same jam_fraction (same total emit power), only strategy
differs. If adaptive > fixed in trace_P degradation, D1 PASSES.

API matches StrongRule.get_action (env, team) -> dict.
"""

from __future__ import annotations
import torch
from typing import Dict


class TrueFixedJammer:
    """Channel-constant jammer. Both enemy radars stay on `fixed_channel`.

    Used as the apple-to-apple baseline against AdaptiveSpectrumJammer:
    same jam_fraction, only strategy differs (constant vs follow).
    """

    def __init__(
        self,
        jam_fraction: float = 1e-4,
        fixed_channel: int = 0,
        device: str = "cuda",
    ):
        self.jam_fraction = float(jam_fraction)
        self.fixed_channel = int(fixed_channel)
        self.device = device

    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        E = env.E
        dev = env.device
        R = env.n_radars_per_team

        base = torch.zeros(E, R, 4, device=dev)
        jf = max(float(self.jam_fraction), 1e-9)
        base[..., 2] = jf
        base[..., 3] = 1.0 - jf
        task_alloc = base.clone()

        beam_target = torch.zeros(E, R, dtype=torch.long, device=dev)
        laser_target = torch.zeros(E, dtype=torch.long, device=dev)
        emission_on = torch.ones(E, R, device=dev)
        freq_hop_rate = torch.ones(E, R, device=dev)
        channel_select = torch.full(
            (E, R), self.fixed_channel, dtype=torch.long, device=dev,
        )

        return {
            "task_alloc": task_alloc,
            "beam_target": beam_target,
            "laser_target": laser_target,
            "emission_on": emission_on,
            "freq_hop_rate": freq_hop_rate,
            "channel_select": channel_select,
        }


class AdaptiveSpectrumJammer:
    """Channel-split follower. Enemy radar i → victim radar i's channel+beam.

    Strategy:
      - Enemy radar i tracks victim radar i's current channel
      - Enemy radar i beams at victim radar i (focus fire per pair)
      - jam_fraction apple-to-apple with TrueFixedJammer

    Why this is harder than fixed:
      Victim on orth (ch0 + ch1):
        - Fixed (both on ch0): victim radar 1 free → still 1 good tracker
        - Adaptive split: both victim channels jammed → no escape
      Victim dynamically re-allocates (RL):
        - Adaptive follows each victim radar → continuous pressure
        - Fixed stays on ch0 → trivially dodged by moving off ch0

    Parameters:
      jam_fraction: fraction of aperture power to jam (rest to comm).
      probe_eps: with prob ε, probe a random channel (anti-dodge exploration).
      device: torch device.
    """

    def __init__(
        self,
        jam_fraction: float = 1e-4,
        probe_eps: float = 0.0,
        comm_floor: float = 0.10,
        device: str = "cuda",
    ):
        self.jam_fraction = float(jam_fraction)
        self.probe_eps = float(probe_eps)
        self.comm_floor = float(comm_floor)
        self.device = device

    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        """Per-team action slice. Enemy radar i → victim radar i's channel."""
        E = env.E
        dev = env.device
        et = 1 - team
        R = env.n_radars_per_team

        # task_alloc: jam-only with apple-to-apple jam_fraction.
        base = torch.zeros(E, R, 4, device=dev)
        jf = max(float(self.jam_fraction), 1e-9)
        base[..., 2] = jf
        base[..., 3] = 1.0 - jf
        task_alloc = base.clone()

        # Channel-split following: enemy radar i → victim radar i's channel
        victim_ch_idx = (
            (env.radar_freq_hz[:, et, :] - env.fc_hz)
            / env.channel_spacing_hz
        ).round().long().clamp(0, env.n_channels - 1)   # [E, R]

        if self.probe_eps > 0:
            probe_mask = torch.rand(E, device=dev) < self.probe_eps
            probe_ch = torch.randint(
                0, env.n_channels, (E, R), device=dev, dtype=torch.long,
            )
            channel_select = torch.where(
                probe_mask.unsqueeze(-1).expand_as(victim_ch_idx),
                probe_ch, victim_ch_idx,
            )
        else:
            channel_select = victim_ch_idx.clone()

        # Beam: enemy radar i → victim radar i (focus fire per pair)
        beam_target = torch.arange(R, device=dev).unsqueeze(0).expand(E, R).long()

        laser_target = torch.zeros(E, dtype=torch.long, device=dev)
        emission_on = torch.ones(E, R, device=dev)
        freq_hop_rate = torch.ones(E, R, device=dev)

        return {
            "task_alloc": task_alloc,
            "beam_target": beam_target,
            "laser_target": laser_target,
            "emission_on": emission_on,
            "freq_hop_rate": freq_hop_rate,
            "channel_select": channel_select,
        }
