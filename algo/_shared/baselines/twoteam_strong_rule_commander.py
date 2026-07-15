"""Strong rule-based multifunction commander for two-team env (WP1 anti-strawman baseline).

Per TWOTEAM_MULTIFUNCTION_PLAN.md §WP1.1 + TWOTEAM_ENV_FIX_SPEC.md (2026-07-14 FIX 3):
  Anti-strawman strong rule commander with anti-jam frequency-agility reaction.

Required properties:
  - Strong enough that "G0 exploitability PASS" is meaningful (not beating a strawman)
  - Per-aperture adaptive 4-function allocation (NOT a fixed vector)
  - Tracks own kill progress, defensive jam when enemy tracks me, comm floor, exposure duck
  - FIX 3: anti-jam frequency hopping (boost hop when enemy jamming me hard)
  - FIX 3: relaxed duck threshold (was 30, now 60) to break mirror duck-mutex

Strategy:
  1. laser target = argmax over alive+trackable enemy radars of E_progress
  2. task_alloc per aperture adapts:
     - boost track if laser target's trace_P > tau_track
     - boost jam if enemy tracking me well (their trace_P on me < jamreact_tau)
     - FIX 3: boost track + jam if enemy jamming me (reactive anti-jam + counter-jam)
     - comm floor maintained
  3. beam = laser target (focus fire)
  4. emission_on = False iff exposure high AND no enemy near kill (FIX 3: threshold 60)
  5. FIX 1/3: freq_hop_rate = high when enemy jam strong, low otherwise

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
        exposure_duck_threshold: float = 60.0,   # FIX 3: was 30 → 60 (less eager duck)
        near_kill_threshold: float = 0.5,
        jamreact_tau: float = 0.04,
        comm_floor: float = 0.10,
        # FIX 1/3: freq_hop reaction thresholds
        jam_detect_threshold: float = 0.30,   # enemy jam level triggering hop reaction
        freq_hop_low: float = 1.0,            # default hop rate (no agility)
        freq_hop_high: float = 6.0,           # hop rate under heavy jam
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
        trackable = (trace_P_me_on_enemy < env.tau_track) & my_init & enemy_alive
        laser_score = E_progress * trackable.float() + trackable.float() * 0.01
        fallback_score = E_progress * enemy_alive.float() + enemy_alive.float() * 1e-3
        any_trackable = trackable.any(dim=-1)
        final_score = torch.where(
            any_trackable.unsqueeze(-1), laser_score, fallback_score)
        lt_idx = final_score.argmax(dim=-1)   # [E]

        # --- 3. per-aperture adaptive task_alloc ---
        base = torch.tensor(
            [0.10, 0.45, 0.30, 0.15], device=dev).expand(E, R, 4).clone()

        # Boost track if laser target's trace_P high
        trace_P_lt = trace_P_me_on_enemy.gather(
            1, lt_idx.unsqueeze(1)).squeeze(1)
        low_track = (trace_P_lt > env.tau_track).float()
        low_track_b = low_track.unsqueeze(-1).expand(E, R)
        base[..., 1] = base[..., 1] + 0.20 * low_track_b
        base[..., 2] = base[..., 2] - 0.10 * low_track_b

        # Boost jam if enemy tracking me
        enemy_tracking_me = (
            trace_P_enemy_on_me.min(dim=-1).values < self.jamreact_tau
        ).float()
        etm_b = enemy_tracking_me.unsqueeze(-1).expand(E, R)
        base[..., 2] = base[..., 2] + 0.20 * etm_b
        base[..., 0] = base[..., 0] - 0.10 * etm_b

        # FIX 3: Anti-jam reaction — if enemy jamming me hard, boost track + jam
        # (track to burn through their jam on me + jam to disrupt their track on me)
        enemy_jam_on_me = env._last_jam_matrix[:, team, :]   # [E, R]
        enemy_jam_max = enemy_jam_on_me.max(dim=-1).values   # [E]
        high_jam = (enemy_jam_max > self.jam_detect_threshold).float()
        high_jam_b = high_jam.unsqueeze(-1).expand(E, R)
        base[..., 1] = base[..., 1] + 0.15 * high_jam_b   # more track to burn through
        base[..., 2] = base[..., 2] + 0.10 * high_jam_b   # more counter-jam

        # Comm floor (clamp)
        base[..., 3] = torch.clamp(base[..., 3], min=self.comm_floor)

        # Clip negatives, then softmax with sharpness
        base = torch.clamp(base, min=1e-3)
        task_alloc = torch.softmax(base * self.sharpness, dim=-1)
        task_alloc = task_alloc / (task_alloc.sum(dim=-1, keepdim=True) + 1e-8)

        # --- 4. beam target: both apertures at laser target (focus fire) ---
        beam_target = lt_idx.unsqueeze(-1).expand(E, R).clone()

        # --- 5. emission_on: duck if exposure high and no near-kill ---
        E_max = E_progress.max(dim=-1).values
        duck_mask = (
            (env.exposure[:, team] > self.exposure_duck_threshold)
            & (E_max < self.near_kill_threshold)
        )
        emit = (~duck_mask).float()
        emission_on = emit.unsqueeze(-1).expand(E, R)

        # --- 6. FIX 1/3: freq_hop_rate per aperture ---
        # High jam → hop fast (anti-jam skill); low jam → hop=1 (no overhead)
        hop_val = torch.where(
            high_jam.bool(),
            torch.full_like(high_jam, self.freq_hop_high),
            torch.full_like(high_jam, self.freq_hop_low),
        )   # [E]
        freq_hop_rate = hop_val.unsqueeze(-1).expand(E, R).clone()

        # --- 7. WP-C R3: channel_select (hold current env freq — StrongRule
        # doesn't dynamically re-allocate channels; it's the FIXED-allocation
        # baseline. Channel index derived from env.radar_freq_hz so wrapper-set
        # orthogonal config (ch0/ch1) is preserved across steps.)
        ch_idx = ((env.radar_freq_hz[:, team, :] - env.fc_hz)
                  / env.channel_spacing_hz).round().long().clamp(0, env.n_channels - 1)

        return {
            "task_alloc": task_alloc,
            "beam_target": beam_target,
            "laser_target": lt_idx,
            "emission_on": emission_on,
            "freq_hop_rate": freq_hop_rate,
            "channel_select": ch_idx,
        }
