"""Extreme-strategy team commanders for WP0 four-function tradeoff check.

Per TWOTEAM_MULTIFUNCTION_PLAN.md WP0.6②:
  Construct extreme fixed-strategy teams; verify no dominant single strategy
  (rock-paper-scissors non-transitivity or at least mutual wins).
  If a single strategy dominates all → game is trivial (root A) → don't enter WP1.

Strategies:
  pure_track   : task_alloc = [0, 1, 0, 0] — all subarrays to tracking
  pure_jam     : task_alloc = [0, 0, 1, 0] — all subarrays to jamming
  pure_comm    : task_alloc = [0, 0, 0, 1] — all subarrays to comm (control, useless alone)
  pure_detect  : task_alloc = [1, 0, 0, 0] — all subarrays to detect (no track/jam)
  balanced     : task_alloc = [0.10, 0.45, 0.30, 0.15] — mix
  balanced_jam_heavy : task_alloc = [0.05, 0.30, 0.55, 0.10] — jam-leaning mix
  track_then_kill : task_alloc rad0=[0,1,0,0], rad1=[0,0.5,0.3,0.2] — asymmetric per aperture
"""

from __future__ import annotations
import torch
from typing import Dict


def _build_alloc(E: int, team: int, allocations_per_aperture, device: str = "cuda") -> torch.Tensor:
    """Build task_alloc tensor [E, 2, 2, 4] for one team given fixed allocations.

    allocations_per_aperture: list of 2 lists of 4 floats (per aperture fractions).
    Other team gets zeros (caller fills).
    """
    ta = torch.zeros(E, 2, 2, 4, device=device)
    for k, alloc in enumerate(allocations_per_aperture):
        ta[:, team, k] = torch.tensor(alloc, device=device)
    return ta


class ExtremeCommander:
    """Fixed-strategy team commander.

    Acts the same across both apertures (for pure_track/jam/comm/detect)
    or per-aperture (for asymmetric strategies).
    """

    def __init__(self, name: str, alloc_per_aperture, laser_strategy: str = "lowest_E",
                 beam_strategy: str = "same_as_laser", device: str = "cuda"):
        self.name = name
        self.alloc_per_aperture = alloc_per_aperture   # list of 2 lists of 4
        self.laser_strategy = laser_strategy
        self.beam_strategy = beam_strategy
        self.device = device

    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        """Generate PER-TEAM action slice (this team only).

        Returns:
            task_alloc: [E, 2_radars, 4]
            beam_target: [E, 2_radars] long, 0 or 1
            laser_target: [E] long, 0 or 1
            emission_on: [E, 2_radars] float
        """
        E = env.E
        dev = self.device

        # task_alloc [E, 2_radars, 4]
        ta = torch.zeros(E, 2, 4, device=dev)
        for k, alloc in enumerate(self.alloc_per_aperture):
            ta[:, k] = torch.tensor(alloc, device=dev)

        # Laser target [E]
        et = 1 - team
        if self.laser_strategy == "lowest_E":
            E_enemy = env.radar_E[:, et]
            lt_idx = E_enemy.argmax(dim=-1)   # closest to kill
        elif self.laser_strategy == "radar_0":
            lt_idx = torch.zeros(E, dtype=torch.long, device=dev)
        elif self.laser_strategy == "alternating":
            trace_P = env.tracker_P[:, team, :, 0, 0] + env.tracker_P[:, team, :, 2, 2]
            lt_idx = trace_P.argmin(dim=-1)
        else:
            lt_idx = torch.zeros(E, dtype=torch.long, device=dev)

        # Beam target [E, 2_radars]
        bt = torch.zeros(E, 2, dtype=torch.long, device=dev)
        if self.beam_strategy == "same_as_laser":
            bt[:, 0] = lt_idx
            bt[:, 1] = lt_idx
        elif self.beam_strategy == "split":
            bt[:, 0] = 0
            bt[:, 1] = 1

        # Emission [E, 2_radars]
        eo = torch.ones(E, 2, device=dev)

        return {"task_alloc": ta, "beam_target": bt,
                "laser_target": lt_idx, "emission_on": eo}


def combine_team_actions(env, action_t0: Dict, action_t1: Dict) -> Dict:
    """Combine two per-team action slices into a full env.step action dict.

    Stacks along team axis (dim=1): [E, 2_teams, ...].
    """
    return {
        "task_alloc": torch.stack([action_t0["task_alloc"], action_t1["task_alloc"]], dim=1),
        "beam_target": torch.stack([action_t0["beam_target"], action_t1["beam_target"]], dim=1),
        "laser_target": torch.stack([action_t0["laser_target"], action_t1["laser_target"]], dim=1),
        "emission_on": torch.stack([action_t0["emission_on"], action_t1["emission_on"]], dim=1),
    }


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
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.5, 0.3, 0.2]],   # rad0 pure track, rad1 mixed
    ),
}
