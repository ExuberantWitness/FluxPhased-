"""TAES commander actor-critic for the joint action space.

Action (joint, discrete):
  task_alloc:     4 logits → softmax (subarray fractions for detect/track/jam/comm)
  beam_target:    N_max logits → categorical (which target to point main beam at)
  laser_target:   N_max logits → categorical (which target to fire laser at)
  emission_on:    1 logit → Bernoulli (emit vs passive)

Observation: 95-dim per spec (per-target block × 8 + global 7).

Network: shared MLP trunk (256, 256) → policy_heads + value_head.
CTDE: privileged critic adds jam_level + exposure_dual + per-target jam (extra ~10 dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


__all__ = ["TaesCommanderActorCritic"]


class TaesCommanderActorCritic(nn.Module):
    """Joint commander actor-critic for the TAES env.

    Single-agent (commander only); jammer is external (adversary.py).
    CTDE privileged critic adds extra info not in the agent obs.
    """

    def __init__(
        self,
        obs_dim: int = 95,
        n_targets_max: int = 8,
        privileged_dim: int = 10,
        hidden: int = 256,
        n_task_alloc: int = 4,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.N_max = int(n_targets_max)
        self.privileged_dim = int(privileged_dim)
        self.n_task_alloc = int(n_task_alloc)

        # Shared trunk (actor)
        self.actor_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Policy heads
        self.task_alloc_head = nn.Linear(hidden, n_task_alloc)
        self.beam_target_head = nn.Linear(hidden, n_targets_max)
        self.laser_target_head = nn.Linear(hidden, n_targets_max)
        self.emission_head = nn.Linear(hidden, 1)

        # Critic (privileged CTDE)
        critic_in = obs_dim + privileged_dim
        self.critic_trunk = nn.Sequential(
            nn.Linear(critic_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Local critic (decentralized, obs-only) — for noise-robust α_eff blending
        # Used by IPPO ablation as the sole critic; used by MAPPO α_eff as A_agent source
        self.local_critic_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, privileged: torch.Tensor = None,
                target_alive_mask: torch.Tensor = None):
        """Returns action dict, log_prob, value.

        Args:
            obs: [B, obs_dim]
            privileged: [B, privileged_dim] or None (use zeros)
            target_alive_mask: [B, N_max] bool — if set, dead slots get -inf logits
                in beam_target / laser_target heads (prevents sampling padded targets
                in mixed-N training).

        Returns:
            action: dict of [B] tensors
            log_prob: [B]
            value: [B]
        """
        B = obs.shape[0]
        h = self.actor_trunk(obs)

        # Policy distributions
        task_logits = self.task_alloc_head(h)              # [B, 4]
        beam_logits = self.beam_target_head(h)             # [B, N_max]
        laser_logits = self.laser_target_head(h)           # [B, N_max]
        emission_logit = self.emission_head(h).squeeze(-1)  # [B]

        # Mask dead target slots in mixed-N settings
        if target_alive_mask is not None:
            mask = ~target_alive_mask  # True where dead
            beam_logits = beam_logits.masked_fill(mask, -1e9)
            laser_logits = laser_logits.masked_fill(mask, -1e9)

        task_dist = torch.distributions.Categorical(logits=task_logits)
        beam_dist = torch.distributions.Categorical(logits=beam_logits)
        laser_dist = torch.distributions.Categorical(logits=laser_logits)
        emission_dist = torch.distributions.Bernoulli(logits=emission_logit)

        task_action = task_dist.sample()
        beam_action = beam_dist.sample()
        laser_action = laser_dist.sample()
        emission_action = emission_dist.sample().float()

        log_prob = (
            task_dist.log_prob(task_action)
            + beam_dist.log_prob(beam_action)
            + laser_dist.log_prob(laser_action)
            + emission_dist.log_prob(emission_action)
        )

        # Critic
        if privileged is None:
            privileged = torch.zeros(B, self.privileged_dim, device=obs.device)
        value = self.critic_trunk(torch.cat([obs, privileged], dim=-1)).squeeze(-1)
        value_local = self.local_critic_trunk(obs).squeeze(-1)

        action = {
            "task_alloc_idx": task_action,        # [B] long, 0..3
            "beam_target_idx": beam_action,        # [B] long
            "laser_target_idx": laser_action,      # [B] long
            "emission_on": emission_action,        # [B] float 0/1
        }
        return action, log_prob, value, value_local

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: Dict[str, torch.Tensor],
        privileged: torch.Tensor = None,
        target_alive_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """For PPO update: return (log_prob, value) for given (obs, action)."""
        B = obs.shape[0]
        h = self.actor_trunk(obs)

        task_logits = self.task_alloc_head(h)
        beam_logits = self.beam_target_head(h)
        laser_logits = self.laser_target_head(h)
        emission_logit = self.emission_head(h).squeeze(-1)

        # Mask dead target slots (mixed-N)
        if target_alive_mask is not None:
            mask = ~target_alive_mask
            beam_logits = beam_logits.masked_fill(mask, -1e9)
            laser_logits = laser_logits.masked_fill(mask, -1e9)

        task_dist = torch.distributions.Categorical(logits=task_logits)
        beam_dist = torch.distributions.Categorical(logits=beam_logits)
        laser_dist = torch.distributions.Categorical(logits=laser_logits)
        emission_dist = torch.distributions.Bernoulli(logits=emission_logit)

        log_prob = (
            task_dist.log_prob(action["task_alloc_idx"])
            + beam_dist.log_prob(action["beam_target_idx"])
            + laser_dist.log_prob(action["laser_target_idx"])
            + emission_dist.log_prob(action["emission_on"])
        )

        if privileged is None:
            privileged = torch.zeros(B, self.privileged_dim, device=obs.device)
        value = self.critic_trunk(torch.cat([obs, privileged], dim=-1)).squeeze(-1)
        value_local = self.local_critic_trunk(obs).squeeze(-1)

        entropy = (
            task_dist.entropy()
            + beam_dist.entropy()
            + laser_dist.entropy()
            + emission_dist.entropy()
        )
        return log_prob, value, entropy, value_local

    def get_action_for_env(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        target_alive_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert internal action representation to env's expected dict.

        Env expects: task_alloc[E, 4] (softmax probs), beam_target_idx[E] long,
                     laser_target_idx[E] long, emission_on[E] float.
        """
        h = self.actor_trunk(obs)
        task_logits = self.task_alloc_head(h)
        beam_logits = self.beam_target_head(h)
        laser_logits = self.laser_target_head(h)
        emission_logit = self.emission_head(h).squeeze(-1)

        # Mask dead target slots (mixed-N)
        if target_alive_mask is not None:
            mask = ~target_alive_mask
            beam_logits = beam_logits.masked_fill(mask, -1e9)
            laser_logits = laser_logits.masked_fill(mask, -1e9)

        if deterministic:
            task_alloc = F.softmax(task_logits, dim=-1)
            beam_idx = beam_logits.argmax(dim=-1)
            laser_idx = laser_logits.argmax(dim=-1)
            emission = (emission_logit > 0).float()
        else:
            task_dist = torch.distributions.Categorical(logits=task_logits)
            beam_dist = torch.distributions.Categorical(logits=beam_logits)
            laser_dist = torch.distributions.Categorical(logits=laser_logits)
            emission_dist = torch.distributions.Bernoulli(logits=emission_logit)
            task_alloc_idx = task_dist.sample()
            beam_idx = beam_dist.sample()
            laser_idx = laser_dist.sample()
            emission = emission_dist.sample().float()
            # Convert task_alloc_idx → one-hot for env
            task_alloc = F.one_hot(task_alloc_idx, self.n_task_alloc).float()

        return {
            "task_alloc": task_alloc,
            "beam_target_idx": beam_idx,
            "laser_target_idx": laser_idx,
            "emission_on": emission,
        }


def build_privileged(env, jam_level: torch.Tensor) -> torch.Tensor:
    """Build CTDE privileged vector from env state + jam_level.

    Shape: [E, privileged_dim]. Adds:
      - jam_level (1)
      - exposure normalized (1)
      - own_alive (1)
      - step_norm (1)
      - mean trace_P normalized (1)
      - mean E_i normalized (1)
      - n_alive_targets normalized (1)
      - zeros (3) for future use
    """
    E = env.E
    dev = env.device
    priv = torch.zeros(E, 10, device=dev)
    priv[:, 0] = jam_level
    priv[:, 1] = (env.exposure / 100.0).clamp(0.0, 10.0)
    priv[:, 2] = env.own_alive.float()
    priv[:, 3] = env.step_idx.float() / float(env.episode_steps)
    trace_P = env.tracker_P[..., 0, 0] + env.tracker_P[..., 2, 2]
    alive_f = env.target_alive_mask.float()
    mean_trace_P = (trace_P * alive_f).sum(dim=1) / alive_f.sum(dim=1).clamp(min=1.0)
    priv[:, 4] = mean_trace_P / max(env.tau_track_nominal, 1e-3)
    priv[:, 5] = (env.target_E * alive_f).sum(dim=1) / alive_f.sum(dim=1).clamp(min=1.0) / max(env.e_kill, 1e-6)
    priv[:, 6] = alive_f.sum(dim=1) / float(env.N_max)
    return priv
