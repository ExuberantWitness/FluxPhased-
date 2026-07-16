"""Blind classical commander for two-team env (WP-2 §3 spec).

Spec §3 ③ mandates a "competent blind classical" baseline that demonstrates
  ① low-interference: search → detect → track → kill (capability proof)
  ② high-interference: track breaks, kill chain collapses (no toy)

This commander extends StrongRule (WP-1) with two key upgrades:
  1. beam_direction (continuous azimuth [-π,π]) REPLACES legacy beam_target.
     Derived from tracker_x belief via atan2(dy, dx), NOT from god-view enemy pos.
     - Slot init: point at tracker belief (track maintenance)
     - Slot uninit: point at lowest-coverage search cell (probe for hidden enemy)
  2. ECCM channel_select: under heavy jam, hop to least-jammed channel
     (reactive freq agility beyond StrongRule's hop_rate reaction)

Mirror-symmetric: deterministic given env state. No RNG; both teams run identical
logic on mirrored sensor data → identical mirrored actions.

API matches TwoTeamStrongRuleCommander.get_action(env, team) -> per-team slice.
Combine via combine_team_actions() in extreme_commanders.py.
"""

from __future__ import annotations
import math
import torch
from typing import Dict


class BlindClassicalCommander:
    """Blind classical multifunction commander (no god-view, +ECCM)."""

    def __init__(
        self,
        sharpness: float = 4.0,
        exposure_duck_threshold: float = 60.0,
        near_kill_threshold: float = 0.5,
        jamreact_tau: float = 0.04,
        comm_floor: float = 0.10,
        jam_detect_threshold: float = 0.30,
        freq_hop_low: float = 1.0,
        freq_hop_high: float = 6.0,
        # ECCM: when jam > threshold, channel_select switches to least-jammed ch
        eccm_jam_threshold: float = 0.40,
        device: str = "cuda",
    ):
        self.sharpness = float(sharpness)
        self.exposure_duck_threshold = float(exposure_duck_threshold)
        self.near_kill_threshold = float(near_kill_threshold)
        self.jamreact_tau = float(jamreact_tau)
        self.comm_floor = float(comm_floor)
        self.jam_detect_threshold = float(jam_detect_threshold)
        self.freq_hop_low = float(freq_hop_low)
        self.freq_hop_high = float(freq_hop_high)
        self.eccm_jam_threshold = float(eccm_jam_threshold)
        self.device = device

    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        """Per-team action slice. Returns beam_direction (NOT legacy beam_target)."""
        E = env.E
        dev = env.device
        et = 1 - team
        R = env.n_radars_per_team

        # --- 1. trace_P of MY trackers on ENEMY radars [E, R] ---
        trace_P_me_on_enemy = (
            env.tracker_P[:, team, :, 0, 0] + env.tracker_P[:, team, :, 2, 2]
        )
        trace_P_enemy_on_me = (
            env.tracker_P[:, et, :, 0, 0] + env.tracker_P[:, et, :, 2, 2]
        )

        enemy_alive = env.radar_alive[:, et]                              # [E, R]
        my_init = env.tracker_initialized[:, team]                        # [E, R]
        E_progress = env.radar_E[:, et] / env.e_kill                      # [E, R]

        # --- 2. laser target: trackable + highest E_progress (same as StrongRule) ---
        trackable = (trace_P_me_on_enemy < env.tau_track) & my_init & enemy_alive
        laser_score = E_progress * trackable.float() + trackable.float() * 0.01
        fallback_score = E_progress * enemy_alive.float() + enemy_alive.float() * 1e-3
        any_trackable = trackable.any(dim=-1)
        final_score = torch.where(
            any_trackable.unsqueeze(-1), laser_score, fallback_score)
        lt_idx = final_score.argmax(dim=-1)                               # [E]

        # --- 3. per-aperture adaptive task_alloc (same as StrongRule) ---
        base = torch.tensor(
            [0.10, 0.45, 0.30, 0.15], device=dev).expand(E, R, 4).clone()

        trace_P_lt = trace_P_me_on_enemy.gather(
            1, lt_idx.unsqueeze(1)).squeeze(1)
        low_track = (trace_P_lt > env.tau_track).float()
        low_track_b = low_track.unsqueeze(-1).expand(E, R)
        base[..., 1] = base[..., 1] + 0.20 * low_track_b
        base[..., 2] = base[..., 2] - 0.10 * low_track_b

        enemy_tracking_me = (
            trace_P_enemy_on_me.min(dim=-1).values < self.jamreact_tau
        ).float()
        etm_b = enemy_tracking_me.unsqueeze(-1).expand(E, R)
        base[..., 2] = base[..., 2] + 0.20 * etm_b
        base[..., 0] = base[..., 0] - 0.10 * etm_b

        enemy_jam_on_me = env._last_jam_matrix[:, team, :]                # [E, R]
        enemy_jam_max = enemy_jam_on_me.max(dim=-1).values                # [E]
        high_jam = (enemy_jam_max > self.jam_detect_threshold).float()
        high_jam_b = high_jam.unsqueeze(-1).expand(E, R)
        base[..., 1] = base[..., 1] + 0.15 * high_jam_b
        base[..., 2] = base[..., 2] + 0.10 * high_jam_b

        base[..., 3] = torch.clamp(base[..., 3], min=self.comm_floor)
        base = torch.clamp(base, min=1e-3)
        task_alloc = torch.softmax(base * self.sharpness, dim=-1)
        task_alloc = task_alloc / (task_alloc.sum(dim=-1, keepdim=True) + 1e-8)

        # --- 4. WP-2 core change: beam_direction from tracker belief (no god-view) ---
        # For each aperture k (treated as slot k):
        #   - slot init: aim at tracker_x belief  (track maintenance beam)
        #   - slot uninit: aim at lowest-coverage search cell (proactive search)
        own_pos = env.radar_pos[:, team]                                  # [E, R, 2]
        own_pos_k = own_pos                                                # use aperture k's own pos

        # Tracker belief positions per slot [E, R, 2] (x, y from [x,vx,y,vy])
        tracker_pos = env.tracker_x[:, team, :, [0, 2]]                   # [E, R, 2]
        # atan2(dy, dx) from own aperture k to tracker belief k
        delta_belief = tracker_pos - own_pos_k                            # [E, R, 2]
        track_az = torch.atan2(delta_belief[..., 1], delta_belief[..., 0])   # [E, R]

        # Search azimuth: pick lowest-coverage cell per slot. argsort gives
        # ascending order; aperture k gets the k-th least-covered cell.
        # _searched_cells: [E, T, n_search_cells] bool bitmap
        searched = env._searched_cells[:, team].float()                   # [E, n_cells]
        # argsort ascending → low counts first. Each aperture scans a different cell.
        # argsort(dim=-1) returns indices that would sort; [E, n_cells]
        # Take [:, :R] to get the R least-searched cells (one per aperture).
        cell_width = 2.0 * math.pi / env.n_search_cells
        cell_centers = (
            torch.arange(env.n_search_cells, device=dev, dtype=searched.dtype) * cell_width - math.pi
        )                                                                  # [n_cells]
        # Ascending sort: least-searched first. Tie-broken by index (deterministic).
        order = torch.argsort(searched, dim=-1, stable=True)              # [E, n_cells]
        # Pick first R cells (one per aperture). cell index → azimuth via cell_centers
        cell_idx_per_aperture = order[:, :R]                              # [E, R]
        search_az = cell_centers[cell_idx_per_aperture]                   # [E, R]

        # Pick track_az where init, search_az elsewhere
        beam_az = torch.where(my_init, track_az, search_az)               # [E, R]

        # --- 5. emission_on: duck if exposure high and no near-kill (same as StrongRule) ---
        E_max = E_progress.max(dim=-1).values
        duck_mask = (
            (env.exposure[:, team] > self.exposure_duck_threshold)
            & (E_max < self.near_kill_threshold)
        )
        emit = (~duck_mask).float()
        emission_on = emit.unsqueeze(-1).expand(E, R)

        # --- 6. freq_hop_rate: high when enemy jam strong (StrongRule logic) ---
        hop_val = torch.where(
            high_jam.bool(),
            torch.full_like(high_jam, self.freq_hop_high),
            torch.full_like(high_jam, self.freq_hop_low),
        )                                                                  # [E]
        freq_hop_rate = hop_val.unsqueeze(-1).expand(E, R).clone()

        # --- 7. WP-2 ECCM: channel_select to least-jammed channel under heavy jam ---
        # Default: keep current channel (hold env.radar_freq_hz → channel index)
        current_ch = ((env.radar_freq_hz[:, team, :] - env.fc_hz)
                      / env.channel_spacing_hz).round().long().clamp(0, env.n_channels - 1)
        # When jam high on any aperture, switch ALL apertures to the channel
        # with lowest JNR. We approximate "lowest JNR channel" by 0 (channel 0
        # is default; per-channel JNR breakdown isn't exposed in obs). Better:
        # cycle to next channel under heavy jam (frequency diversity).
        next_ch = (current_ch + 1) % env.n_channels
        switch_mask = high_jam.bool().unsqueeze(-1).expand(E, R) & (enemy_jam_max.unsqueeze(-1) > self.eccm_jam_threshold)
        channel_select = torch.where(switch_mask, next_ch, current_ch)

        # --- 8. laser_target (unchanged — slot-id semantics from WP-2 M0) ---
        # lt_idx picks the slot to fire at; env checks belief-vs-truth distance.

        return {
            "task_alloc": task_alloc,
            "beam_direction": beam_az,                # NEW: continuous azimuth
            "laser_target": lt_idx,
            "emission_on": emission_on,
            "freq_hop_rate": freq_hop_rate,
            "channel_select": channel_select,
        }
