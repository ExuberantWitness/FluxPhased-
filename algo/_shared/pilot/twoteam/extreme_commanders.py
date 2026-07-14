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
