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
    """PPO training loop for one agent (commander or radar).

    Usage:
        trainer = PPOTrainer(actor_critic, config)
        for step in range(n_steps):
            trainer.collect_rollout(env, ...)
            if buffer.full:
                trainer.update()
    """

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

    def collect_step(
        self,
        obs: torch.Tensor,
        buffer: RolloutBuffer,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Collect one step: get action, log_prob, value.

        Args:
            obs: [obs_dim] or [B, obs_dim] tensor
            buffer: RolloutBuffer to store transition
            deterministic: if True, use mean action (no sampling)
        Returns:
            action tensor for env step
        """
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            action, value = self.ac(obs)
            # Approximate log_prob for flat action
            log_prob = torch.zeros(action.shape[0], device=action.device)

        return action.squeeze(0), value.squeeze(), log_prob.squeeze()

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run PPO update on collected rollouts.

        Args:
            buffer: Full RolloutBuffer with computed returns
        Returns:
            dict of loss metrics
        """
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

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass
                log_prob, entropy, value = self.ac.evaluate_actions(obs, old_actions)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_prob - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((value.squeeze() - returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
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
    """Manages PPO training for a full team (1 commander + 2 radars with shared params).

    The two radars on the same team share a single RadarActorCritic network,
    doubling the effective sample efficiency.
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

        # Separate buffers for commander and radar
        from .buffer import RolloutBuffer

        cmd_obs_dim = 68
        cmd_act_dim = 35
        radar_obs_dim = 0  # set later from env
        radar_act_dim = 13753

        self.commander_buffer = RolloutBuffer(
            buffer_size, cmd_obs_dim, cmd_act_dim,
            gamma=gamma, gae_lambda=gae_lambda, device=device,
        )
        self.radar_buffer = RolloutBuffer(
            buffer_size, radar_act_dim, radar_act_dim,
            gamma=gamma, gae_lambda=gae_lambda, device=device,
        )

        self.reward_shaper = DenseRewardShaper(device=device)

    def set_radar_obs_dim(self, obs_dim: int):
        """Set radar observation dimension (from env)."""
        self.radar_buffer = RolloutBuffer(
            self.radar_buffer.buffer_size,
            obs_dim, 13753,
            gamma=self.radar_buffer.gamma,
            gae_lambda=self.radar_buffer.gae_lambda,
            device=self.device,
        )

    def collect_env_step(
        self,
        env,  # MFARVecEnv
        step_output: dict,
    ) -> dict:
        """Process one env step output into buffer additions.

        Args:
            env: MFARVecEnv instance
            step_output: dict from env.step()
        Returns:
            metrics dict
        """
        # Compute shaped rewards
        shaped = self.reward_shaper(step_output)
        total_radar_reward = (
            shaped["total_shaped"] + step_output["radar_rewards"]
        )  # [E, R]
        commander_reward = step_output["commander_rewards"]  # [E, n_teams]

        return {
            "radar_reward": total_radar_reward,
            "commander_reward": commander_reward,
            "shaped_rewards": shaped,
        }

    def update(self) -> dict:
        """Run PPO updates for both commander and radar."""
        cmd_metrics = {}
        radar_metrics = {}

        if self.commander_buffer.size > self.batch_size:
            self.commander_buffer.compute_returns()
            cmd_metrics = self.commander_trainer.update(self.commander_buffer)
            self.commander_buffer.reset()

        if self.radar_buffer.size > self.batch_size:
            self.radar_buffer.compute_returns()
            radar_metrics = self.radar_trainer.update(self.radar_buffer)
            self.radar_buffer.reset()

        return {
            "commander": cmd_metrics,
            "radar": radar_metrics,
        }

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
