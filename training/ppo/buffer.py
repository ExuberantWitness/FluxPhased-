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
        privileged_dim: int = 0,
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.privileged_dim = privileged_dim

        self.reset()

    def reset(self):
        """Clear all buffers."""
        self.obs = torch.zeros(self.buffer_size, self.obs_dim, dtype=torch.float32)
        self.actions = torch.zeros(self.buffer_size, self.act_dim, dtype=torch.float32)
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.values = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.privileged_values = torch.zeros(self.buffer_size, dtype=torch.float32)
        if self.privileged_dim > 0:
            self.privileged_infos = torch.zeros(
                self.buffer_size, self.privileged_dim, dtype=torch.float32,
            )
        else:
            self.privileged_infos = None
        self.advantages = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(self.buffer_size, dtype=torch.float32)
        # Team-level credit assignment (CTDE architecture)
        self.team_rewards = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.team_returns = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.team_states = torch.zeros(self.buffer_size, 104, dtype=torch.float32)
        # BC pretrain log-probs for KL penalty during PPO fine-tuning
        self.pretrain_log_probs = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, value, log_prob,
            privileged_value=None, privileged_info=None,
            team_reward=None, team_state=None, pretrain_log_prob=None):
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
        self.privileged_values[self.ptr] = privileged_value if privileged_value is not None else value
        if self.privileged_infos is not None and privileged_info is not None:
            self.privileged_infos[self.ptr] = privileged_info
        if team_reward is not None:
            self.team_rewards[self.ptr] = team_reward
        if team_state is not None:
            self.team_states[self.ptr] = team_state
        if pretrain_log_prob is not None:
            self.pretrain_log_probs[self.ptr] = pretrain_log_prob
        self.ptr += 1

    @property
    def near_full(self) -> bool:
        """True when only one slot remains; trigger update() now to avoid overflow."""
        return self.ptr >= self.buffer_size - 1

    def fill_fraction(self) -> float:
        """Return fraction of buffer that is filled (0.0 → 1.0)."""
        return self.ptr / max(self.buffer_size, 1)

    def compute_returns(self, last_value: float = 0.0, last_privileged_value: float = None):
        """Compute GAE advantages and returns using privileged value estimates.

        Uses privileged_values (from asymmetric critic) for both delta and
        return computation, since the privileged critic produces more accurate
        value estimates than the deployment value head.

        Args:
            last_value: Value estimate for the state after the last transition
                        (from deployment value head, for compatibility).
            last_privileged_value: Privileged value estimate for the last state.
                                   If None, falls back to last_value.
        """
        if last_privileged_value is None:
            last_privileged_value = last_value

        gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_privileged_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.privileged_values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.privileged_values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.privileged_values[t]

    def compute_n_step_returns(self, n_steps: int = 200, last_value: float = 0.0):
        """Compute N-step truncated returns for long-horizon credit assignment.

        G_t = Σ_{k=0}^{N-1} γ^k r_{t+k}  +  γ^N V(s_{t+N})

        Uses the buffer's own rewards/dones/values.  Unlike GAE which has
        an effective horizon of ~17 steps (γ=0.99, λ=0.95), N=200 gives a
        ~200 step horizon — sufficient for missile flight (~600 steps at
        244 m/s over 10 km needs ~41 s; with 0.02 s CPI this is ~2000 steps;
        N=200 gives partial coverage with bootstrap fallback).

        The result is written to self.returns (overwriting any previous
        GAE returns).  Call this INSTEAD OF compute_returns() when using
        N-step credit assignment (e.g., for TeamCritic).
        """
        n = self.ptr
        gamma_n = self.gamma ** n_steps

        for t in range(n):
            # Sum rewards for steps t..min(t+N-1, n-1)
            end = min(t + n_steps, n)
            discounted_sum = 0.0
            g_pow = 1.0
            for k in range(t, end):
                discounted_sum += g_pow * self.rewards[k]
                g_pow *= self.gamma

            # Bootstrap from value after N steps (or 0 if episode ends)
            if end < n:
                bootstrap_val = self.values[end]
                bootstrap_scale = g_pow * (1.0 - self.dones[end])
            else:
                # Past end of buffer: use provided last_value
                bootstrap_val = last_value
                bootstrap_scale = gamma_n

            self.returns[t] = discounted_sum + bootstrap_scale * bootstrap_val

        # Advantages = returns - values
        self.advantages = self.returns - self.values

    def compute_team_n_step_returns(self, n_steps: int = 800, last_value: float = 0.0):
        """Compute N-step truncated returns for team rewards.

        TeamCritic uses a longer horizon (N=800) than per-agent critics
        (N=400) because kill events have ~600-step latency that requires
        longer credit propagation.  Uses team_rewards instead of agent
        rewards for the discounted sum.

        G_t = Σ_{k=0}^{N-1} γ^k team_reward_{t+k}  +  γ^N V_team(s_{t+N})
        """
        n = self.ptr
        gamma_n = self.gamma ** n_steps

        for t in range(n):
            end = min(t + n_steps, n)
            discounted_sum = 0.0
            g_pow = 1.0
            for k in range(t, end):
                discounted_sum += g_pow * self.team_rewards[k]
                g_pow *= self.gamma

            if end < n:
                bootstrap_val = self.values[end]
                bootstrap_scale = g_pow * (1.0 - self.dones[end])
            else:
                bootstrap_val = last_value
                bootstrap_scale = gamma_n

            self.team_returns[t] = discounted_sum + bootstrap_scale * bootstrap_val

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

            batch = {
                "obs": self.obs[idx].to(self.device),
                "actions": self.actions[idx].to(self.device),
                "old_log_probs": self.log_probs[idx].to(self.device),
                "advantages": self.advantages[idx].to(self.device),
                "returns": self.returns[idx].to(self.device),
                "privileged_info": None,
                "team_rewards": self.team_rewards[idx].to(self.device),
                "team_returns": self.team_returns[idx].to(self.device),
                "team_states": self.team_states[idx].to(self.device),
                "pretrain_log_probs": self.pretrain_log_probs[idx].to(self.device),
            }
            if self.privileged_infos is not None:
                batch["privileged_info"] = self.privileged_infos[idx].to(self.device)

            yield batch

    @property
    def full(self):
        return self.ptr >= self.buffer_size

    @property
    def size(self):
        return self.ptr
