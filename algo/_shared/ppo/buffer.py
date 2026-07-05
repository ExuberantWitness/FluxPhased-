"""Rollout buffer for PPO with GAE advantage estimation.

Stores transitions (obs, action, reward, done, value, log_prob) and computes
Generalized Advantage Estimation (GAE) returns when full.
"""

import torch
import numpy as np


class RunningMeanStd:
    """Numerically stable running mean/var (Welford's algorithm).

    Used by F8 (return-based scaling) to normalize reward magnitudes
    across batches. Reward scale imbalance (e.g. kill_bonus=100 vs
    shaping ~20000) collapses value learning without this.
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = x.shape[0] if x.ndim > 0 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, bm, bv, bc):
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        M2 = m_a + m_b + np.square(delta) * self.count * bc / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot


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
        reward_normalize: bool = False,  # F8: return-based scaling
        joint_action_dim: int = 0,  # COMA: centralized Q critic input
        store_coma_agent_idx: bool = False,  # COMA: per-transition agent_idx
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.privileged_dim = privileged_dim
        # F8: running reward stats for return-based scaling
        self.reward_normalize = reward_normalize
        self.reward_rms = RunningMeanStd() if reward_normalize else None
        # F1: team-path running reward stats (mirrors F8 for team_returns so
        # kill_bonus spikes don't keep team_value_loss in the millions).
        self.team_reward_rms = RunningMeanStd() if reward_normalize else None
        # Ablation switch: f1_disable=True skips team reward normalization +
        # done-mask (reverts to old cross-episode 800-step accumulation
        # behavior). For快速遍历 isolating F1's contribution.
        self.f1_disable = False
        # COMA: optional storage for all-agents joint action vector at each
        # timestep. Populated only when joint_action_dim > 0 (COMA on).
        # Layout documented in algo/_shared/ppo/coma_critic.py.
        self.joint_action_dim = joint_action_dim
        # COMA: per-transition agent index within its team (0 or 1 for radar,
        # 0 for commander). Needed because radar_buf stores transitions from
        # both team-0 radars mixed together; COMA must know which slot in
        # joint_actions was "self" to compute the counterfactual baseline.
        self.store_coma_agent_idx = store_coma_agent_idx

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
        # COMA: joint action storage (optional). When joint_action_dim > 0,
        # every add() must supply a joint_action vector (caller contract).
        if self.joint_action_dim > 0:
            self.joint_actions = torch.zeros(
                self.buffer_size, self.joint_action_dim, dtype=torch.float32,
            )
        else:
            self.joint_actions = None
        # COMA: per-transition agent_idx (0/1 for radar, 0 for commander)
        if self.store_coma_agent_idx:
            self.coma_agent_idx = torch.zeros(self.buffer_size, dtype=torch.long)
        else:
            self.coma_agent_idx = None
        self.ptr = 0

    def add(self, obs, action, reward, done, value, log_prob,
            privileged_value=None, privileged_info=None,
            team_reward=None, team_state=None, pretrain_log_prob=None,
            joint_action=None, coma_agent_idx=None):
        """Add one transition.

        Caller is responsible for calling update() before the buffer
        fills (see e.g. PhasedTrainer / FluxLeague._train_against,
        which check `near_full` after every store_transition).

        COMA: when self.joint_action_dim > 0, caller MUST pass joint_action
        (the all-agents action vector). When self.joint_action_dim == 0
        (default — ippo/mappo/pspfix), joint_action is ignored.
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
        if self.joint_actions is not None:
            assert joint_action is not None, (
                "COMA buffer requires joint_action on every add() "
                f"(ptr={self.ptr}, joint_action_dim={self.joint_action_dim})"
            )
            self.joint_actions[self.ptr] = joint_action
        if self.coma_agent_idx is not None:
            assert coma_agent_idx is not None, (
                "COMA buffer requires coma_agent_idx on every add() "
                f"(ptr={self.ptr})"
            )
            self.coma_agent_idx[self.ptr] = int(coma_agent_idx)
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

        # F8: return-based scaling — normalize rewards by running std before
        # GAE so value targets stay O(1). Without this, kill_bonus=100 vs
        # shaping=20000 produces value_loss ~2-3M and value head never converges.
        if self.reward_normalize and self.reward_rms is not None and self.ptr > 0:
            self.reward_rms.update(self.rewards[:self.ptr].numpy())
            std = float(np.sqrt(self.reward_rms.var + 1e-8))
            self.rewards[:self.ptr] = self.rewards[:self.ptr] / std

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

    def compute_team_returns(self):
        """F1: team value targets = reward-normalized, done-masked discounted return.

        Replaces the buggy compute_team_n_step_returns (3 bugs: cross-episode
        accumulation A, wrong agent-value bootstrap B, no reward normalization C).
        Mirrors the agent GAE's reward normalization + episode-boundary handling;
        produces O(1) value targets so team_value_loss actually descends.

        Uses λ=1 MC return: episode=500, kill at ~10 steps, γ=0.999 → in-ep
        coverage is sufficient; done mask guarantees no cross-episode leak.

        Ablation: f1_disable=True reverts to OLD cross-episode 800-step
        accumulation (Bug A active, no normalization) for baseline comparison.
        """
        n = self.ptr
        if n == 0:
            return

        if getattr(self, 'f1_disable', False):
            # OLD buggy path: 800-step cross-episode accumulation, no done-mask,
            # no reward normalization. Used for A/B baseline comparison.
            n_steps = 800
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
                    bootstrap_val = 0.0
                    bootstrap_scale = gamma_n
                self.team_returns[t] = discounted_sum + bootstrap_scale * bootstrap_val
            return

        # (C fix) normalize team rewards → O(1) value targets
        if self.reward_normalize and self.team_reward_rms is not None:
            self.team_reward_rms.update(self.team_rewards[:n].numpy())
            tstd = float(np.sqrt(self.team_reward_rms.var + 1e-8))
            tr = self.team_rewards[:n] / tstd
        else:
            tr = self.team_rewards[:n]
        # (A fix) done-masked discounted return; ret resets at episode boundary
        ret = 0.0
        for t in reversed(range(n)):
            nonterminal = 1.0 - self.dones[t]
            ret = tr[t] + self.gamma * nonterminal * ret
            self.team_returns[t] = ret
        # (B fix) no agent-value bootstrap; team_adv = team_returns - team_critic(team_states)
        #         is computed in ppo_trainer.update where team_critic is available.

        # S0: one-shot reward-magnitude diagnostic (10-min verification per
        # 修改建议 §5). Confirms kill_bonus / shaping actual magnitudes vs
        # ALPHA_COLLAPSE_REPORT.md claims.
        if not getattr(self, '_s0_printed', False):
            self._s0_printed = True
            raw = self.team_rewards[:n]
            post = self.team_returns[:n]
            print(f"[S0] team_rewards raw: max={float(raw.max()):.2f} "
                  f"mean={float(raw.mean()):.2f} std={float(raw.std()):.2f} "
                  f"|min|={float(raw.abs().min()):.4f}", flush=True)
            print(f"[S0] team_returns (post-norm): max={float(post.max()):.2f} "
                  f"mean={float(post.mean()):.2f} std={float(post.std()):.2f}",
                  flush=True)
            print(f"[S0] normalize={'ON' if (self.reward_normalize and self.team_reward_rms is not None) else 'OFF'}, "
                  f"n={n}, gamma={self.gamma}", flush=True)

    # Backward-compat alias: old callers passing (n_steps, last_value) keep working.
    def compute_team_n_step_returns(self, n_steps: int = 800, last_value: float = 0.0):
        """Deprecated alias — use compute_team_returns(). F1 fix."""
        self.compute_team_returns()

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
            if self.joint_actions is not None:
                batch["joint_actions"] = self.joint_actions[idx].to(self.device)
            if self.coma_agent_idx is not None:
                batch["coma_agent_idx"] = self.coma_agent_idx[idx].to(self.device)

            yield batch

    @property
    def full(self):
        return self.ptr >= self.buffer_size

    @property
    def size(self):
        return self.ptr
