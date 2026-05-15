"""PPO trainer for Commander and Radar actor-critic networks.

Handles:
- Rollout collection from MFARVecEnv
- GAE computation
- Clipped surrogate loss + value loss + entropy bonus
- Gradient clipping and optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from .buffer import RolloutBuffer
from .actor_critic import CommanderActorCritic, RadarActorCritic
from .reward_shaping import DenseRewardShaper


class PPOTrainer:
    """PPO training loop for one agent (commander or radar)."""

    def __init__(
        self,
        actor_critic: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: str = "cuda",
    ):
        self.ac = actor_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run PPO update on collected rollouts."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in buffer.get_minibatches(self.batch_size):
                obs = batch["obs"]
                old_actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                log_prob, entropy, value = self.ac.evaluate_actions(obs, old_actions)

                ratio = torch.exp(log_prob - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((value.squeeze() - returns) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }


class TeamPPOTrainer:
    """Manages PPO training for a full team (1 commander + shared radar policy).

    The radars on the same team share a single RadarActorCritic network,
    doubling effective sample efficiency.

    Usage:
        trainer = TeamPPOTrainer(commander, radar, ...)
        trainer.init_buffers(env.state_dim, env.action_dim)
        own = trainer.get_own_actions(env, team)
        result = env.step(actions, cmd_actions)
        trainer.store_transition(env, result, own["transition"], team)
        if buffer full: trainer.update()
    """

    def __init__(
        self,
        commander: CommanderActorCritic,
        radar: RadarActorCritic,
        commander_lr: float = 3e-4,
        radar_lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        commander_clip: float = 0.2,
        radar_clip: float = 0.1,
        commander_entropy: float = 0.01,
        radar_entropy: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: str = "cuda",
    ):
        self.commander_trainer = PPOTrainer(
            commander, lr=commander_lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=commander_clip, entropy_coef=commander_entropy,
            value_coef=value_coef, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
            buffer_size=buffer_size, device=device,
        )
        self.radar_trainer = PPOTrainer(
            radar, lr=radar_lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=radar_clip, entropy_coef=radar_entropy,
            value_coef=value_coef, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
            buffer_size=buffer_size, device=device,
        )
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Buffers initialized via init_buffers() after env is created
        self.commander_buffer = None
        self.radar_buffer = None
        self.reward_shaper = DenseRewardShaper(device=device)

    def init_buffers(self, env_state_dim: int, env_action_dim: int):
        """Initialize rollout buffers with correct dimensions from env."""
        self.commander_buffer = RolloutBuffer(
            self.buffer_size, obs_dim=68, act_dim=35,
            gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device,
        )
        self.radar_buffer = RolloutBuffer(
            self.buffer_size, obs_dim=env_state_dim, act_dim=env_action_dim,
            gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device,
        )

    def _get_observations(self, env):
        """Get current state and commander observations from env."""
        state = env._assemble_state(env._buf_spectrum, env._buf_comm_data)
        comm_input = torch.zeros(
            env.num_envs, env.n_radars, env.num_input_length, device=self.device,
        )
        commander_obs = env.battlefield.get_commander_observation(
            env.radar_pos, comm_input,
        )
        return state, commander_obs

    def get_own_actions(self, env, team: int, deterministic: bool = False):
        """Query own policies and return actions for env.step().

        Args:
            env: MFARVecEnv instance
            team: team index (0 or 1)
            deterministic: use mean actions (for evaluation)
        Returns:
            dict with radar_actions, commander_action, transition, r_start, r_end
        """
        r_per_team = env.n_radars // env.n_teams
        r_start = team * r_per_team
        r_end = r_start + r_per_team

        state, commander_obs = self._get_observations(env)

        with torch.no_grad():
            # Commander
            cmd_obs = commander_obs[:, team, :]  # [E, 68]
            cmd_action, cmd_logp, cmd_val = self.commander_trainer.ac.get_action(
                cmd_obs, deterministic=deterministic,
            )

            # Radars (shared policy, individual observations)
            radar_actions = []
            rep_logp = rep_val = rep_obs = rep_action = None
            for r in range(r_start, r_end):
                r_obs = state[:, r, :]  # [E, state_dim]
                r_act, r_logp, r_val = self.radar_trainer.ac.get_action(
                    r_obs, deterministic=deterministic,
                )
                radar_actions.append(r_act)
                if r == r_start:
                    rep_obs = r_obs
                    rep_action = r_act
                    rep_logp = r_logp          # [E]
                    rep_val = r_val.squeeze(-1) # [E]

        return {
            "radar_actions": radar_actions,    # list of [E, action_dim]
            "commander_action": cmd_action,    # [E, cmd_act_dim]
            "transition": {
                "cmd_obs": cmd_obs,
                "cmd_action": cmd_action,
                "cmd_logp": cmd_logp,
                "cmd_val": cmd_val.squeeze(-1),
                "radar_obs": rep_obs,
                "radar_action": rep_action,
                "radar_logp": rep_logp,
                "radar_val": rep_val,
            },
            "r_start": r_start,
            "r_end": r_end,
        }

    def store_transition(self, env, result: dict, transition: dict, team: int):
        """Compute shaped rewards and store transitions in buffers.

        Args:
            env: MFARVecEnv instance
            result: output from env.step()
            transition: dict from get_own_actions()["transition"]
            team: team index
        Returns:
            reward metrics dict
        """
        shaped = self.reward_shaper(result)
        r_per_team = env.n_radars // env.n_teams
        r_start = team * r_per_team

        total_radar_reward = shaped["total_shaped"] + result["radar_rewards"]  # [E, R]
        cmd_reward = result["commander_rewards"]  # [E, n_teams]

        for e in range(env.num_envs):
            done = float(result["dones"][e].item())

            # Commander transition
            self.commander_buffer.add(
                obs=transition["cmd_obs"][e].cpu(),
                action=transition["cmd_action"][e].cpu(),
                reward=cmd_reward[e, team].item(),
                done=done,
                value=transition["cmd_val"][e].item(),
                log_prob=transition["cmd_logp"][e].item(),
            )

            # Radar transition (representative radar for the team)
            radar_reward = total_radar_reward[e, r_start:r_start + r_per_team].sum().item()
            self.radar_buffer.add(
                obs=transition["radar_obs"][e].cpu(),
                action=transition["radar_action"][e].cpu(),
                reward=radar_reward,
                done=done,
                value=transition["radar_val"][e].item(),
                log_prob=transition["radar_logp"][e].item(),
            )

        return {
            "radar_reward": total_radar_reward,
            "commander_reward": cmd_reward,
            "shaped_rewards": shaped,
        }

    def update(self) -> dict:
        """Run PPO updates for both commander and radar when buffers are full."""
        cmd_metrics = {}
        radar_metrics = {}

        if self.commander_buffer and self.commander_buffer.size > self.batch_size:
            self.commander_buffer.compute_returns()
            cmd_metrics = self.commander_trainer.update(self.commander_buffer)
            self.commander_buffer.reset()

        if self.radar_buffer and self.radar_buffer.size > self.batch_size:
            self.radar_buffer.compute_returns()
            radar_metrics = self.radar_trainer.update(self.radar_buffer)
            self.radar_buffer.reset()

        return {"commander": cmd_metrics, "radar": radar_metrics}

    def save(self, path: str):
        """Save team policy checkpoint."""
        torch.save({
            "commander": self.commander_trainer.ac.state_dict(),
            "radar": self.radar_trainer.ac.state_dict(),
            "commander_optimizer": self.commander_trainer.optimizer.state_dict(),
            "radar_optimizer": self.radar_trainer.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load team policy checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.commander_trainer.ac.load_state_dict(ckpt["commander"])
        self.radar_trainer.ac.load_state_dict(ckpt["radar"])
        self.commander_trainer.optimizer.load_state_dict(ckpt["commander_optimizer"])
        self.radar_trainer.optimizer.load_state_dict(ckpt["radar_optimizer"])
