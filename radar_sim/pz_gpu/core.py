"""PettingZoo ParallelEnv wrapping GPU-vectorized MFARVecEnv.

Single-env mode (E=1): standard PettingZoo interface.
One step = one CPI of radar simulation on GPU.

Agents:
  - 4 radar agents: red_radar_0, red_radar_1, blue_radar_0, blue_radar_1
  - 2 commander agents: red_commander, blue_commander
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from gymnasium import spaces

from ..gpu.vec_mfar_env import MFARVecEnv
from .agent_map import AgentIndexMap


class FluxPhasedPZEnv(ParallelEnv):
    """PettingZoo ParallelEnv for GPU-vectorized MFAR radar combat.

    Args:
        radar_latents_fn: Callable mapping radar state [R, state_dim] to
            latent vectors [R, num_input_length]. Called each step to produce
            commander observations. If None, commander obs latents are zero-filled.
        max_steps: Episode truncation limit.
        device: "cuda" or "cpu".
        **mfar_kwargs: Passed to MFARVecEnv. For testing, use small arrays
            (e.g. rows=2, cols=2, pulses_per_cpi=2).
    """

    metadata = {"name": "FluxPhased-v0", "render_modes": []}

    def __init__(
        self,
        radar_latents_fn: Optional[Callable[[NDArray], NDArray]] = None,
        max_steps: int = 10000,
        device: str = "cuda",
        **mfar_kwargs,
    ):
        super().__init__()
        mfar_kwargs["num_envs"] = 1
        mfar_kwargs.setdefault("device", device)

        self._device = device
        self._max_steps = max_steps
        self._radar_latents_fn = radar_latents_fn
        self._step_count = 0

        self._env = MFARVecEnv(**mfar_kwargs)
        self._agent_map = AgentIndexMap.from_config(
            n_radars=self._env.n_radars,
            n_teams=self._env.n_teams,
        )

        self.possible_agents = list(self._agent_map.possible_agents)
        self.agents = list(self.possible_agents)

        # Build spaces
        state_dim = self._env.state_dim
        cmd_obs_dim = self._env.battlefield.commander_obs_dim
        action_dim = self._env.action_dim
        cmd_act_dim = self._env.battlefield.commander_action_dim

        self._obs_spaces: Dict[str, spaces.Box] = {}
        self._act_spaces: Dict[str, spaces.Box] = {}

        for agent in self._agent_map.radar_agents:
            self._obs_spaces[agent] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32,
            )
            self._act_spaces[agent] = spaces.Box(
                low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32,
            )

        for agent in self._agent_map.commander_agents:
            self._obs_spaces[agent] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(cmd_obs_dim,), dtype=np.float32,
            )
            self._act_spaces[agent] = spaces.Box(
                low=-1.0, high=1.0, shape=(cmd_act_dim,), dtype=np.float32,
            )

        self._last_result = None

    # ------------------------------------------------------------------
    # PettingZoo required methods
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, NDArray], Dict[str, dict]]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self._step_count = 0
        self.agents = list(self.possible_agents)
        self._env.reset()

        # Run one default step to produce initial observations
        result = self._env.step(actions=None, commander_actions=None, radar_latents=None)
        self._last_result = result

        observations = {}
        infos = {}
        for agent in self.agents:
            observations[agent] = self._extract_obs(agent, result)
            infos[agent] = self._extract_info(agent, result)

        return observations, infos

    def step(
        self,
        actions: Dict[str, NDArray],
    ) -> Tuple[
        Dict[str, NDArray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        self._step_count += 1
        dev = torch.device(self._device)
        R = self._env.n_radars
        action_dim = self._env.action_dim
        cmd_dim = self._env.battlefield.commander_action_dim

        # Phase 1: Convert PZ action dict → batched GPU tensors
        radar_actions = torch.zeros(1, R, action_dim, device=dev)
        commander_actions = torch.zeros(1, self._env.n_teams, cmd_dim, device=dev)

        for agent_name, action_np in actions.items():
            action_np = np.asarray(action_np, dtype=np.float32)
            if self._agent_map.is_radar(agent_name):
                idx = self._agent_map.radar_idx(agent_name)
                radar_actions[0, idx] = torch.from_numpy(action_np).to(dev)
            elif self._agent_map.is_commander(agent_name):
                team = self._agent_map.commander_team(agent_name)
                commander_actions[0, team] = torch.from_numpy(action_np).to(dev)

        # Phase 2: Compute radar latents via callback
        radar_latents = None
        if self._radar_latents_fn is not None and self._last_result is not None:
            state_np = self._last_result["state"][0].cpu().numpy()  # [R, state_dim]
            latents_np = self._radar_latents_fn(state_np)
            radar_latents = (
                torch.from_numpy(np.asarray(latents_np, dtype=np.float32))
                .unsqueeze(0)
                .to(dev)
            )

        # Phase 3: GPU env step
        result = self._env.step(
            actions=radar_actions,
            commander_actions=commander_actions,
            radar_latents=radar_latents,
        )
        self._last_result = result

        # Phase 4: Extract per-agent results
        done = result["dones"][0].item()
        truncated = self._step_count >= self._max_steps
        alive_tensor = self._env.battlefield.alive[0]  # [R]

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in list(self.agents):
            rewards[agent] = self._extract_reward(agent, result)
            terminations[agent] = done
            truncations[agent] = truncated

            if self._check_agent_alive(agent, alive_tensor):
                observations[agent] = self._extract_obs(agent, result)
                infos[agent] = self._extract_info(agent, result)
            else:
                observations[agent] = np.zeros(
                    self._obs_spaces[agent].shape, dtype=np.float32,
                )
                infos[agent] = {"alive": False}

        # Phase 5: Remove dead agents
        dead = [
            agent for agent in self.agents
            if not self._check_agent_alive(agent, alive_tensor)
        ]
        for agent in dead:
            self.agents.remove(agent)

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def close(self):
        del self._env

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_agent_alive(self, agent: str, alive_tensor: torch.Tensor) -> bool:
        if self._agent_map.is_radar(agent):
            idx = self._agent_map.radar_idx(agent)
            return alive_tensor[idx].item()
        team = self._agent_map.commander_team(agent)
        indices = self._agent_map.team_radar_indices[team]
        return any(alive_tensor[i].item() for i in indices)

    def _extract_obs(self, agent: str, result: dict) -> NDArray:
        if self._agent_map.is_radar(agent):
            idx = self._agent_map.radar_idx(agent)
            return result["state"][0, idx, :].cpu().numpy().astype(np.float32)
        team = self._agent_map.commander_team(agent)
        return result["commander_obs"][0, team, :].cpu().numpy().astype(np.float32)

    def _extract_reward(self, agent: str, result: dict) -> float:
        if self._agent_map.is_radar(agent):
            idx = self._agent_map.radar_idx(agent)
            return result["radar_rewards"][0, idx].item()
        team = self._agent_map.commander_team(agent)
        return result["commander_rewards"][0, team].item()

    def _extract_info(self, agent: str, result: dict) -> dict:
        alive_tensor = self._env.battlefield.alive[0]
        info: dict = {"winner": result["winners"][0].item(), "step": self._step_count}

        if self._agent_map.is_radar(agent):
            idx = self._agent_map.radar_idx(agent)
            info["alive"] = alive_tensor[idx].item()
            info["position"] = self._env.radar_pos[0, idx, :3].cpu().numpy().tolist()
            info["task_ids"] = result["task_ids"][0, idx].cpu().numpy()
        else:
            team = self._agent_map.commander_team(agent)
            info["team"] = team
            info["missile_in_flight"] = (
                self._env.battlefield.missile.in_flight[0, team].item()
            )
            info["missile_pos"] = (
                self._env.battlefield.missile.missile_pos[0, team]
                .cpu()
                .numpy()
                .tolist()
            )

        return info
