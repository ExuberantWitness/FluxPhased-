"""Rollout buffer for PPO with GAE advantage estimation.

Stores transitions (obs, action, reward, done, value, log_prob) and computes
Generalized Advantage Estimation (GAE) returns when full.
"""

import torch
import numpy as np


class RolloutBuffer:
    """On-policy rollout buffer for PPO training.

    Stores data in CPU tensors, transfers to device for training.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cuda",
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.reset()

    def reset(self):
        """Clear all buffers."""
        self.obs = torch.zeros(self.buffer_size, self.obs_dim, dtype=torch.float32)
        self.actions = torch.zeros(self.buffer_size, self.act_dim, dtype=torch.float32)
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.values = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.advantages = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, value, log_prob):
        """Add one transition.

        Caller is responsible for calling update() before the buffer
        fills (see e.g. PhasedTrainer / FluxLeague._train_against,
        which check `near_full` after every store_transition).
        """
        assert self.ptr < self.buffer_size, (
            f"RolloutBuffer overflow at ptr={self.ptr} "
            f"(buffer_size={self.buffer_size}); caller must update() "
            f"before adding past capacity."
        )
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    @property
    def near_full(self) -> bool:
        """True when only one slot remains; trigger update() now to avoid overflow."""
        return self.ptr >= self.buffer_size - 1

    def compute_returns(self, last_value: float = 0.0):
        """Compute GAE advantages and returns.

        Args:
            last_value: Value estimate for the state after the last transition.
        """
        gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_minibatches(self, batch_size: int):
        """Yield minibatches for PPO update.

        Shuffles data and yields chunks of batch_size.
        Returns dicts with all tensors on self.device.
        """
        indices = torch.randperm(self.ptr)
        n = self.ptr

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            yield {
                "obs": self.obs[idx].to(self.device),
                "actions": self.actions[idx].to(self.device),
                "old_log_probs": self.log_probs[idx].to(self.device),
                "advantages": self.advantages[idx].to(self.device),
                "returns": self.returns[idx].to(self.device),
            }

    @property
    def full(self):
        return self.ptr >= self.buffer_size

    @property
    def size(self):
        return self.ptr
