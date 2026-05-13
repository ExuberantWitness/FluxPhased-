"""Episode trajectory collection with random policy baseline.

EpisodeCollector: buffers step() results for post-hoc evaluation.
EpisodeData: container for collected trajectory data.
RandomPolicy: random-parameter neural network for metric validation.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch


@dataclass
class EpisodeData:
    """Container for one episode's collected trajectory data."""
    # Per-step tensors [n_steps, ...]
    spectrum: Optional[torch.Tensor] = None      # [T, E, R, N, P, n_bins]
    comm_data: Optional[torch.Tensor] = None      # [T, E, R, N, 2]
    task_ids: Optional[torch.Tensor] = None       # [T, E, R, N]
    radar_rewards: Optional[torch.Tensor] = None  # [T, E, R]
    commander_rewards: Optional[torch.Tensor] = None  # [T, E, n_teams]
    kills: Optional[torch.Tensor] = None          # [T, E, n_teams, n_enemy]
    dones: Optional[torch.Tensor] = None          # [T, E]
    winners: Optional[torch.Tensor] = None        # [T, E]
    missile_pos: Optional[torch.Tensor] = None    # [T, E, n_teams, 3]

    # Single-tensor state (only last step for memory)
    positions: Optional[dict] = None  # radar_pos, target_pos at each step

    # Timing
    timing: list = field(default_factory=list)

    # Episode metadata
    n_steps: int = 0
    episode_done: bool = False
    final_winner: int = -1


class RandomPolicy:
    """Random-parameter neural network for metric validation.

    Generates valid actions for MFARVecEnv. Uses a small linear network
    with random weights, producing outputs in the correct action space.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        commander_obs_dim: int,
        commander_action_dim: int,
        n_radars: int = 4,
        n_teams: int = 2,
        device: str = "cuda",
        seed: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.commander_action_dim = commander_action_dim
        self.n_radars = n_radars
        self.n_teams = n_teams

        if seed is not None:
            torch.manual_seed(seed)

        # Small random linear layers
        self.radar_net = torch.nn.Linear(
            min(state_dim, 256), action_dim, device=self.device,
        )
        # Initialize with small weights so outputs are in [0,1] range after sigmoid
        torch.nn.init.normal_(self.radar_net.weight, std=0.01)
        torch.nn.init.uniform_(self.radar_net.bias, -0.1, 0.1)

        self.commander_net = torch.nn.Linear(
            min(commander_obs_dim, 64), commander_action_dim, device=self.device,
        )
        torch.nn.init.normal_(self.commander_net.weight, std=0.01)
        torch.nn.init.uniform_(self.commander_net.bias, -0.05, 0.05)

    def get_radar_actions(self, state: torch.Tensor) -> torch.Tensor:
        """state: [E, R, state_dim] → actions: [E, R, action_dim] in [0,1]."""
        # Truncate or pad state to net input size
        inp = state[..., :self.radar_net.in_features]
        if state.shape[-1] < self.radar_net.in_features:
            pad = torch.zeros(
                *state.shape[:-1],
                self.radar_net.in_features - state.shape[-1],
                device=self.device,
            )
            inp = torch.cat([inp, pad], dim=-1)
        return torch.sigmoid(self.radar_net(inp))

    def get_commander_actions(self, commander_obs: torch.Tensor) -> torch.Tensor:
        """commander_obs: [E, n_teams, obs_dim] → actions: [E, n_teams, act_dim] in [-1,1]."""
        inp = commander_obs[..., :self.commander_net.in_features]
        if commander_obs.shape[-1] < self.commander_net.in_features:
            pad = torch.zeros(
                *commander_obs.shape[:-1],
                self.commander_net.in_features - commander_obs.shape[-1],
                device=self.device,
            )
            inp = torch.cat([inp, pad], dim=-1)
        return torch.tanh(self.commander_net(inp))


class EpisodeCollector:
    """Collects step() results for post-hoc evaluation.

    Pre-allocates CPU tensors for trajectory storage (transferred from GPU
    each step to avoid GPU memory buildup).
    """

    def __init__(self, env, max_steps: int = 200):
        self.env = env
        self.max_steps = max_steps
        self._step = 0
        self._reset_buffers()

    def _reset_buffers(self):
        E = self.env.num_envs
        R = self.env.n_radars
        N = self.env.n_elem
        P = self.env.n_pulses
        B = self.env.n_bins
        n_teams = self.env.n_teams

        T = self.max_steps
        self._buf_spectrum = torch.zeros(T, E, R, N, P, B)
        self._buf_comm = torch.zeros(T, E, R, N, 2)
        self._buf_task_ids = torch.zeros(T, E, R, N, dtype=torch.long)
        self._buf_radar_rew = torch.zeros(T, E, R)
        self._buf_cmd_rew = torch.zeros(T, E, n_teams)
        self._buf_kills = torch.zeros(T, E, n_teams, R // n_teams, dtype=torch.bool)
        self._buf_dones = torch.zeros(T, E, dtype=torch.bool)
        self._buf_winners = torch.zeros(T, E, dtype=torch.long)
        self._buf_missile_pos = torch.zeros(T, E, n_teams, 3)
        self._timing_list = []
        self._step = 0

    def collect_step(self, result: dict):
        """Store one step's results."""
        if self._step >= self.max_steps:
            return
        t = self._step
        self._buf_spectrum[t] = result["spectrum"].cpu()
        self._buf_comm[t] = result["comm_data"].cpu()
        self._buf_task_ids[t] = result["task_ids"].cpu()
        self._buf_radar_rew[t] = result["radar_rewards"].cpu()
        self._buf_cmd_rew[t] = result["commander_rewards"].cpu()
        self._buf_kills[t] = result["kills"].cpu()
        self._buf_dones[t] = result["dones"].cpu()
        self._buf_winners[t] = result["winners"].cpu()
        self._buf_missile_pos[t] = result["missile_pos"].cpu()
        self._timing_list.append(result.get("timing", {}))
        self._step += 1

    def finalize(self) -> EpisodeData:
        """Return collected trajectory data, trimmed to actual steps."""
        t = self._step
        if t == 0:
            return EpisodeData()

        done_mask = self._buf_dones[:t].any(dim=0)  # [E]
        winner_final = self._buf_winners[t - 1]  # [E]

        data = EpisodeData(
            spectrum=self._buf_spectrum[:t],
            comm_data=self._buf_comm[:t],
            task_ids=self._buf_task_ids[:t],
            radar_rewards=self._buf_radar_rew[:t],
            commander_rewards=self._buf_cmd_rew[:t],
            kills=self._buf_kills[:t],
            dones=self._buf_dones[:t],
            winners=self._buf_winners[:t],
            missile_pos=self._buf_missile_pos[:t],
            timing=self._timing_list,
            n_steps=t,
            episode_done=done_mask.all().item(),
            final_winner=winner_final[0].item() if done_mask[0] else -1,
        )

        self._reset_buffers()
        return data

    def run_episode(
        self,
        policy_fn: Optional[Callable] = None,
        commander_policy_fn: Optional[Callable] = None,
        max_steps: Optional[int] = None,
    ) -> EpisodeData:
        """Run one complete episode and collect data.

        Args:
            policy_fn: callable(result_dict) -> radar_actions [E, R, action_dim]
                If None, uses default (all detect, no vehicle motion).
            commander_policy_fn: callable(result_dict) -> commander_actions [E, n_teams, cmd_dim]
                If None, no commander actions.
            max_steps: override self.max_steps for this episode.
        """
        if max_steps is not None:
            self.max_steps = max_steps
            self._reset_buffers()

        self.env.reset()
        result = self.env.step()

        for _ in range(self.max_steps):
            self.collect_step(result)

            if result["dones"].all():
                break

            # Get actions
            actions = None
            cmd_actions = None

            if policy_fn is not None:
                actions = policy_fn(result)
            if commander_policy_fn is not None:
                cmd_actions = commander_policy_fn(result)

            result = self.env.step(
                actions=actions,
                commander_actions=cmd_actions,
            )

        return self.finalize()

    def run_episodes(
        self,
        policy_fn: Optional[Callable] = None,
        commander_policy_fn: Optional[Callable] = None,
        n_episodes: int = 10,
        max_steps: Optional[int] = None,
    ) -> list:
        """Run multiple episodes and collect data."""
        episodes = []
        for _ in range(n_episodes):
            ep = self.run_episode(
                policy_fn=policy_fn,
                commander_policy_fn=commander_policy_fn,
                max_steps=max_steps,
            )
            episodes.append(ep)
        return episodes
