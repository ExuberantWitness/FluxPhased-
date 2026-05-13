"""Bidirectional mapping between PettingZoo agent names and GPU tensor indices."""

from dataclasses import dataclass, field
from typing import Dict, List


TEAM_LABELS = ["red", "blue"]


@dataclass
class AgentIndexMap:
    """Mapping between agent names and internal tensor indices.

    Radar indices: 0..R-1 (match MFARVecEnv's axis-1 index).
    Commander indices: team ID (0=red, 1=blue).
    """

    radar_name_to_idx: Dict[str, int]
    commander_name_to_team: Dict[str, int]
    team_radar_indices: Dict[int, List[int]]
    possible_agents: List[str]
    radar_agents: List[str] = field(init=False)
    commander_agents: List[str] = field(init=False)

    def __post_init__(self):
        self.radar_agents = list(self.radar_name_to_idx.keys())
        self.commander_agents = list(self.commander_name_to_team.keys())

    # ---- factories ----

    @classmethod
    def from_config(cls, n_radars: int = 4, n_teams: int = 2) -> "AgentIndexMap":
        r_per_team = n_radars // n_teams
        radar_name_to_idx: Dict[str, int] = {}
        team_radar_indices: Dict[int, List[int]] = {}
        for t in range(n_teams):
            indices = list(range(t * r_per_team, (t + 1) * r_per_team))
            team_radar_indices[t] = indices
            for i, idx in enumerate(indices):
                radar_name_to_idx[f"{TEAM_LABELS[t]}_radar_{i}"] = idx

        commander_name_to_team = {
            f"{TEAM_LABELS[t]}_commander": t for t in range(n_teams)
        }

        possible_agents = list(radar_name_to_idx) + list(commander_name_to_team)
        return cls(
            radar_name_to_idx=radar_name_to_idx,
            commander_name_to_team=commander_name_to_team,
            team_radar_indices=team_radar_indices,
            possible_agents=possible_agents,
        )

    # ---- queries ----

    def is_radar(self, agent: str) -> bool:
        return agent in self.radar_name_to_idx

    def is_commander(self, agent: str) -> bool:
        return agent in self.commander_name_to_team

    def radar_idx(self, agent: str) -> int:
        return self.radar_name_to_idx[agent]

    def commander_team(self, agent: str) -> int:
        return self.commander_name_to_team[agent]

    def team_of_radar(self, agent: str) -> int:
        idx = self.radar_name_to_idx[agent]
        for t, indices in self.team_radar_indices.items():
            if idx in indices:
                return t
        return -1
