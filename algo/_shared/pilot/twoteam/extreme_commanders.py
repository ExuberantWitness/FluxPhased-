"""Extreme-strategy team commanders for WP0 four-function tradeoff check.

Per TWOTEAM_MULTIFUNCTION_PLAN.md WP0.6② + TWOTEAM_ENV_FIX_SPEC.md (2026-07-14):
  Construct extreme fixed-strategy teams; verify (2a) no dominant single strategy,
  (2b) decisive rate ≥ 0.5, (2c) kill density ≥ 0.5, (2d) no strategy with
  stalemate_rate > 0.50.

Strategies:
  pure_track       : task_alloc = [0, 1, 0, 0], freq_hop=1 — all subarrays to tracking, no agility
  pure_jam         : task_alloc = [0, 0, 1, 0], freq_hop=1 — all subarrays to jamming
  pure_comm        : task_alloc = [0, 0, 0, 1], freq_hop=1 — all subarrays to comm
  pure_detect      : task_alloc = [1, 0, 0, 0], freq_hop=1 — all subarrays to detect
  balanced         : task_alloc = [0.10, 0.45, 0.30, 0.15], freq_hop=1 — mix, no agility
  balanced_jam_heavy : task_alloc = [0.05, 0.30, 0.55, 0.10], freq_hop=1 — jam-leaning
  track_then_kill  : asymmetric per aperture
  track_agile      : pure_track + freq_hop=8 — anti-jam skill dimension (FIX 1 verification)

WP2 candidate-exploit strategies (target StrongRule's known thresholds):
  jam_spread       : per-target jam 0.28 < 0.30 threshold → rule's hop reaction never triggers,
                     jam_mul inflates rule's track_sigma on both radars, out-track + kill
  hard_jam_focus   : both apertures full-jam single rule radar (jam=1.0); even with rule's
                     hop=6, effective_jam = 0.167, jam_mul=2.0 → degrade rule's track 2x, focus-kill
  track_heavy_agile: 80% track concentration (rule uses 71%) + freq_hop=8 anti-jam → win track race
"""

from __future__ import annotations
import torch
from typing import Dict


def _build_alloc(E: int, team: int, allocations_per_aperture, device: str = "cuda") -> torch.Tensor:
    """Build task_alloc tensor [E, 2, 2, 4] for one team given fixed allocations."""
    ta = torch.zeros(E, 2, 2, 4, device=device)
    for k, alloc in enumerate(allocations_per_aperture):
        ta[:, team, k] = torch.tensor(alloc, device=device)
    return ta


class ExtremeCommander:
    """Fixed-strategy team commander.

    Acts the same across both apertures (for pure_track/jam/comm/detect)
    or per-aperture (for asymmetric strategies).

    FIX 1: freq_hop_rate is now part of the action dict. Default = 1.0 (no hopping).
    """

    def __init__(self, name: str, alloc_per_aperture, laser_strategy: str = "lowest_E",
                 beam_strategy: str = "same_as_laser", freq_hop: float = 1.0,
                 device: str = "cuda"):
        self.name = name
        self.alloc_per_aperture = alloc_per_aperture
        self.laser_strategy = laser_strategy
        self.beam_strategy = beam_strategy
        self.freq_hop = float(freq_hop)   # FIX 1: hop rate per aperture (constant)
        self.device = device

    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        """Generate PER-TEAM action slice (this team only).

        Returns:
            task_alloc: [E, 2_radars, 4]
            beam_target: [E, 2_radars] long, 0 or 1
            laser_target: [E] long, 0 or 1
            emission_on: [E, 2_radars] float
            freq_hop_rate: [E, 2_radars] float ∈ [1, freq_hop_max]   (FIX 1)
        """
        E = env.E
        dev = self.device

        ta = torch.zeros(E, 2, 4, device=dev)
        for k, alloc in enumerate(self.alloc_per_aperture):
            ta[:, k] = torch.tensor(alloc, device=dev)

        et = 1 - team
        if self.laser_strategy == "lowest_E":
            E_enemy = env.radar_E[:, et]
            lt_idx = E_enemy.argmax(dim=-1)
        elif self.laser_strategy == "radar_0":
            lt_idx = torch.zeros(E, dtype=torch.long, device=dev)
        elif self.laser_strategy == "alternating":
            trace_P = env.tracker_P[:, team, :, 0, 0] + env.tracker_P[:, team, :, 2, 2]
            lt_idx = trace_P.argmin(dim=-1)
        else:
            lt_idx = torch.zeros(E, dtype=torch.long, device=dev)

        bt = torch.zeros(E, 2, dtype=torch.long, device=dev)
        if self.beam_strategy == "same_as_laser":
            bt[:, 0] = lt_idx
            bt[:, 1] = lt_idx
        elif self.beam_strategy == "split":
            bt[:, 0] = 0
            bt[:, 1] = 1

        eo = torch.ones(E, 2, device=dev)

        # FIX 1: constant freq_hop_rate per aperture
        fh = torch.full((E, 2), self.freq_hop, device=dev)

        return {"task_alloc": ta, "beam_target": bt,
                "laser_target": lt_idx, "emission_on": eo,
                "freq_hop_rate": fh}


def combine_team_actions(env, action_t0: Dict, action_t1: Dict) -> Dict:
    """Combine two per-team action slices into a full env.step action dict.

    Stacks along team axis (dim=1): [E, 2_teams, ...].
    Backward compat: if freq_hop_rate absent, env defaults to 1.0.
    WP-C R3: channel_select stacked when present in both teams.
    WP-1 M3: beam_direction stacked when present in both teams (new continuous
    azimuth API, alternative to legacy beam_target).
    """
    out = {
        "task_alloc": torch.stack([action_t0["task_alloc"], action_t1["task_alloc"]], dim=1),
        "beam_target": torch.stack([action_t0["beam_target"], action_t1["beam_target"]], dim=1),
        "laser_target": torch.stack([action_t0["laser_target"], action_t1["laser_target"]], dim=1),
        "emission_on": torch.stack([action_t0["emission_on"], action_t1["emission_on"]], dim=1),
    }
    if "freq_hop_rate" in action_t0 and "freq_hop_rate" in action_t1:
        out["freq_hop_rate"] = torch.stack(
            [action_t0["freq_hop_rate"], action_t1["freq_hop_rate"]], dim=1)
    if "channel_select" in action_t0 and "channel_select" in action_t1:
        out["channel_select"] = torch.stack(
            [action_t0["channel_select"], action_t1["channel_select"]], dim=1)
    # WP-1 M3: stack beam_direction only when BOTH teams provide it (mixed legacy/new
    # is fine — env falls back to beam_target when beam_direction absent).
    if "beam_direction" in action_t0 and "beam_direction" in action_t1:
        out["beam_direction"] = torch.stack(
            [action_t0["beam_direction"], action_t1["beam_direction"]], dim=1)
    return out


# Strategy registry
STRATEGIES = {
    "pure_track": ExtremeCommander(
        "pure_track",
        [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    ),
    "pure_jam": ExtremeCommander(
        "pure_jam",
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    ),
    "pure_comm": ExtremeCommander(
        "pure_comm",
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
    ),
    "pure_detect": ExtremeCommander(
        "pure_detect",
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    ),
    "balanced": ExtremeCommander(
        "balanced",
        [[0.10, 0.45, 0.30, 0.15], [0.10, 0.45, 0.30, 0.15]],
    ),
    "balanced_jam_heavy": ExtremeCommander(
        "balanced_jam_heavy",
        [[0.05, 0.30, 0.55, 0.10], [0.05, 0.30, 0.55, 0.10]],
    ),
    "track_then_kill_asym": ExtremeCommander(
        "track_then_kill_asym",
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.5, 0.3, 0.2]],
    ),
    # FIX 1 verification strategy: tracks but hops fast — should beat pure_jam
    "track_agile": ExtremeCommander(
        "track_agile",
        [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        freq_hop=8.0,   # max hopping — pure_track with anti-jam skill
    ),
}


# ============================================================
# WP2 candidate-exploit commanders
# Target StrongRule's known thresholds (anti-strawman data: rule's
# weakest matchup is pure_jam WR=0.33 — jam mechanic is the lever).
# ============================================================


class JamSpreadCommander:
    """Sub-threshold jam spread across both rule radars.

    Mechanic: rule's anti-jam hop reaction triggers ONLY when
    `_last_jam_matrix[:, rule, :].max() > 0.30`. By keeping per-target
    jam just under 0.30, rule's hop never fires, so jam_mul = 1 + 6 * 0.28
    = 2.68 inflates rule's track_sigma on BOTH radARS → I out-track rule.

    Layout:
      - aperture 0: beam_target=0, jam=0.28, track=0.50, comm=0.22
      - aperture 1: beam_target=1, jam=0.28, track=0.50, comm=0.22

    Jam on rule radar 0 = task_alloc[aperture 0, jam] * emission_on[aperture 0] = 0.28
    Jam on rule radar 1 = task_alloc[aperture 1, jam] * emission_on[aperture 1] = 0.28
    Max jam = 0.28 < 0.30 → rule doesn't hop → effective_jam stays at 0.28.

    Laser target: pick the enemy radar we have better track on (lower trace_P).
    """

    def __init__(self, jam_fraction: float = 0.28, track_fraction: float = 0.50,
                 comm_fraction: float = 0.22, device: str = "cuda"):
        assert jam_fraction < 0.30, f"jam_fraction must be < 0.30 to dodge rule's hop reaction, got {jam_fraction}"
        self.jam_fraction = float(jam_fraction)
        self.track_fraction = float(track_fraction)
        self.comm_fraction = float(comm_fraction)
        self.device = device

    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        E = env.E
        dev = self.device
        et = 1 - team

        ta = torch.zeros(E, 2, 4, device=dev)
        ta[:, 0, 0] = 0.0                          # detect
        ta[:, 0, 1] = self.track_fraction          # track
        ta[:, 0, 2] = self.jam_fraction            # jam (sub-threshold)
        ta[:, 0, 3] = self.comm_fraction           # comm
        ta[:, 1] = ta[:, 0]                        # symmetric per aperture

        # beam_target: split jam across both enemy radars
        bt = torch.zeros(E, 2, dtype=torch.long, device=dev)
        bt[:, 0] = 0
        bt[:, 1] = 1

        # laser_target: enemy radar with lower trace_P (better tracked = closer to kill)
        trace_P_me_on_enemy = (
            env.tracker_P[:, team, :, 0, 0] + env.tracker_P[:, team, :, 2, 2]
        )
        enemy_alive = env.radar_alive[:, et]
        my_init = env.tracker_initialized[:, team]
        trackable = (trace_P_me_on_enemy < env.tau_track) & my_init & enemy_alive
        score = (1.0 / (trace_P_me_on_enemy + 1e-3)) * trackable.float() + trackable.float() * 1e-3
        lt_idx = score.argmax(dim=-1)

        eo = torch.ones(E, 2, device=dev)
        fh = torch.full((E, 2), 1.0, device=dev)   # no agility needed (rule can't jam back effectively at track=0.50)

        return {
            "task_alloc": ta, "beam_target": bt, "laser_target": lt_idx,
            "emission_on": eo, "freq_hop_rate": fh,
        }


class HardJamFocusCommander:
    """Overwhelming jam on one rule radar + focus fire.

    Mechanic: even when rule triggers hop=6 reaction, effective_jam = 1.0/6 = 0.167,
    jam_mul = 1 + 6 * 0.167 = 2.0 → rule's track_sigma on focused radar doubles.
    Combined with my full track concentration on that same radar, I out-track → kill.

    Layout:
      - both apertures: beam_target=0, jam=0.60, track=0.30, comm=0.10
      - jam on rule radar 0 = 0.60 + 0.60 = 1.20 (well over 0.30 → rule hops)
      - But post-hop: effective_jam = 1.20 / 6 = 0.20, jam_mul = 1 + 6*0.20 = 2.20
      - My track on rule radar 0 = 0.30 + 0.30 = 0.60 (concentrated)
      - Rule's track on me (rule uses 0.71 track, but split across both my radars
        because rule's beam follows laser argmax): I'm not focus-jammed.

    Laser target: rule radar 0 (the one we're jam-locking).
    """

    def __init__(self, jam_fraction: float = 0.60, track_fraction: float = 0.30,
                 comm_fraction: float = 0.10, device: str = "cuda"):
        self.jam_fraction = float(jam_fraction)
        self.track_fraction = float(track_fraction)
        self.comm_fraction = float(comm_fraction)
        self.device = device

    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        E = env.E
        dev = self.device
        et = 1 - team

        ta = torch.zeros(E, 2, 4, device=dev)
        ta[:, 0, 0] = 0.0
        ta[:, 0, 1] = self.track_fraction
        ta[:, 0, 2] = self.jam_fraction
        ta[:, 0, 3] = self.comm_fraction
        ta[:, 1] = ta[:, 0]

        # Both apertures beam at rule radar 0 (focus jam + focus fire)
        bt = torch.zeros(E, 2, dtype=torch.long, device=dev)
        # Pick the rule radar we have lowest trace_P on (in case radar 0 dies, switch)
        trace_P_me_on_enemy = (
            env.tracker_P[:, team, :, 0, 0] + env.tracker_P[:, team, :, 2, 2]
        )
        enemy_alive = env.radar_alive[:, et]
        my_init = env.tracker_initialized[:, team]
        trackable = (trace_P_me_on_enemy < env.tau_track) & my_init & enemy_alive
        score = (1.0 / (trace_P_me_on_enemy + 1e-3)) * trackable.float()
        focus_idx = score.argmax(dim=-1)   # [E]
        bt[:, 0] = focus_idx
        bt[:, 1] = focus_idx

        lt_idx = focus_idx.clone()

        eo = torch.ones(E, 2, device=dev)
        # Modest hop in case rule tries to counter-jam us
        fh = torch.full((E, 2), 3.0, device=dev)

        return {
            "task_alloc": ta, "beam_target": bt, "laser_target": lt_idx,
            "emission_on": eo, "freq_hop_rate": fh,
        }


class TrackHeavyAgileCommander:
    """Out-track rule by concentration + max agility.

    Mechanic: rule uses 71% track concentration. By going to 80% track +
    freq_hop=8 (max agility), I track rule's radars faster AND I'm robust to
    rule's counter-jam (rule's jam on me gets divided by my hop=8).

    Per-aperture:
      - track=0.80, jam=0.05, comm=0.10, detect=0.05
      - freq_hop=8 (max agility)
      - beam both apertures at the rule radar with lowest trace_P (focus fire)

    This is the "track_agile" strategy cranked to 80% (vs 100% pure track),
    leaving small jam/comm to enable counter-play if rule adapts.
    """

    def __init__(self, track_fraction: float = 0.80, jam_fraction: float = 0.05,
                 comm_fraction: float = 0.10, detect_fraction: float = 0.05,
                 freq_hop: float = 8.0, device: str = "cuda"):
        self.track_fraction = float(track_fraction)
        self.jam_fraction = float(jam_fraction)
        self.comm_fraction = float(comm_fraction)
        self.detect_fraction = float(detect_fraction)
        self.freq_hop = float(freq_hop)
        self.device = device

    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        E = env.E
        dev = self.device
        et = 1 - team

        ta = torch.zeros(E, 2, 4, device=dev)
        ta[:, 0, 0] = self.detect_fraction
        ta[:, 0, 1] = self.track_fraction
        ta[:, 0, 2] = self.jam_fraction
        ta[:, 0, 3] = self.comm_fraction
        ta[:, 1] = ta[:, 0]

        # Pick the rule radar with lowest trace_P (best tracked) → focus fire there
        trace_P_me_on_enemy = (
            env.tracker_P[:, team, :, 0, 0] + env.tracker_P[:, team, :, 2, 2]
        )
        enemy_alive = env.radar_alive[:, et]
        my_init = env.tracker_initialized[:, team]
        trackable = (trace_P_me_on_enemy < env.tau_track) & my_init & enemy_alive
        score = (1.0 / (trace_P_me_on_enemy + 1e-3)) * trackable.float()
        focus_idx = score.argmax(dim=-1)

        bt = torch.zeros(E, 2, dtype=torch.long, device=dev)
        bt[:, 0] = focus_idx
        bt[:, 1] = focus_idx
        lt_idx = focus_idx.clone()

        eo = torch.ones(E, 2, device=dev)
        fh = torch.full((E, 2), self.freq_hop, device=dev)

        return {
            "task_alloc": ta, "beam_target": bt, "laser_target": lt_idx,
            "emission_on": eo, "freq_hop_rate": fh,
        }


# Register WP2 candidate exploits
STRATEGIES["jam_spread"] = JamSpreadCommander()
STRATEGIES["hard_jam_focus"] = HardJamFocusCommander()
STRATEGIES["track_heavy_agile"] = TrackHeavyAgileCommander()
