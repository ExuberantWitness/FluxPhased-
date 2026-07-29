"""Masked PPO for G3-BSTA-lite (F4, Gate 3).

Per MODIFICATION_PLAN W5 + Gate 3:
  - Actor: 2 x 128 Tanh, masked categorical output.
  - Critic: 2 x 128 Tanh, scalar value head (same obs as actor; no
    privileged critic in the debug profile).
  - PPO hyperparameters (frozen, no HPO):
        lr = 3e-4, gamma = 0.99, GAE lambda = 0.95, clip = 0.2,
        grad clip = 0.5 (actor and critic separately).
  - Required trainer behavior:
        * save exact rollout mask; recompute log_prob with same mask
        * verify pre-update ratio == 1
        * separate actor/critic optimizers and grad clipping
        * log KL, clip_fraction, adv_std, explained_variance,
          grad norms, action frequencies, energy usage, return/drop
          correlation
  - Primary evaluation = sampled stochastic; argmax is secondary only.
  - Separate action RNG, multiple action replicates.

Gate 3 evaluation is on the FIXED debug suite (the same scenarios used
for training). This is a debugging/overfit gate, not an inferential
claim.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.gpu.g3_bsta_lite import (
    EnvConfig,
    G3BstaLiteVecEnv,
    N_ACTIONS,
    generate_paired_manifest,
)


class MaskedCategoricalActor(nn.Module):
    """2 x 128 Tanh MLP -> masked categorical over N_ACTIONS."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        return self.head(h)

    def distribution(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(obs)
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return torch.distributions.Categorical(logits=logits)


class ValueCritic(nn.Module):
    """2 x 128 Tanh MLP -> scalar V."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        return self.head(h).squeeze(-1)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    grad_clip: float = 0.5
    epochs_per_iteration: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_kl: float = 0.05  # early stop on KL excursion
    seed: int = 0
    device: str = "cpu"
    bc_warm_start_path: Optional[str] = None  # load actor from F3 .pt
    iterations: int = 100  # PPO outer iterations
    n_envs: int = 16
    horizon: int = 64  # env horizon H


@dataclass
class RolloutBuffer:
    obs: torch.Tensor           # [T, E, OBS_DIM]
    mask: torch.Tensor          # [T, E, N_ACTIONS]
    action: torch.Tensor        # [T, E]
    logp: torch.Tensor          # [T, E]
    reward: torch.Tensor        # [T, E]
    value: torch.Tensor         # [T, E]
    last_value: torch.Tensor    # [E]
    last_done: torch.Tensor     # [E] (always True at horizon end)


class PPOTrainer:
    def __init__(
        self,
        cfg: PPOConfig,
        env_cfg: EnvConfig,
        train_scenario_seeds: list[int],
    ):
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.train_seeds = list(train_scenario_seeds)
        torch.manual_seed(cfg.seed)

        from env.gpu.g3_bsta_lite import OBS_DIM
        self.obs_dim = OBS_DIM
        self.actor = MaskedCategoricalActor(self.obs_dim, N_ACTIONS).to(cfg.device)
        self.critic = ValueCritic(self.obs_dim).to(cfg.device)

        if cfg.bc_warm_start_path is not None:
            sd = torch.load(cfg.bc_warm_start_path, map_location=cfg.device)
            # F3 ImitationActor has identical architecture (fc1/fc2/head).
            self.actor.load_state_dict(sd)
            print(f"[ppo] BC warm-start loaded from {cfg.bc_warm_start_path}")

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # Build training envs (E = cfg.n_envs).
        self.env_cfg = EnvConfig(
            n_envs=cfg.n_envs, horizon=cfg.horizon, device=cfg.device,
            seed=cfg.seed,
        )
        self.env = G3BstaLiteVecEnv(self.env_cfg)

        self._action_gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed + 7)
        self.iteration = 0
        self.history: list[dict] = []

    def _assign_scenarios_and_reset(self):
        """Reset env on the next fixed train seed (cycles each iteration).

        The env is batched (all envs share one scenario), so for the
        overfit gate we run ONE fixed scenario at a time across all envs.
        `env.reset(seed=sd)` uses the env's canonical scenario generator
        (same path used by the F2 manifest), guaranteeing train/eval
        scenarios match.
        """
        sd = self.train_seeds[self.iteration % len(self.train_seeds)]
        self.env.reset(seed=sd)

    def collect_rollout(self) -> RolloutBuffer:
        T = self.env_cfg.horizon
        E = self.env_cfg.n_envs
        obs_buf = torch.zeros(T, E, self.obs_dim, device=self.cfg.device)
        mask_buf = torch.zeros(T, E, N_ACTIONS, device=self.cfg.device)
        act_buf = torch.zeros(T, E, dtype=torch.int64, device=self.cfg.device)
        logp_buf = torch.zeros(T, E, device=self.cfg.device)
        rew_buf = torch.zeros(T, E, device=self.cfg.device)
        val_buf = torch.zeros(T, E, device=self.cfg.device)

        for t in range(T):
            obs = self.env._build_observation()
            mask = self.env._compute_mask()
            with torch.no_grad():
                dist = self.actor.distribution(obs, mask)
                u = torch.rand(E, generator=self._action_gen, device=self.cfg.device)
                probs = dist.probs.clamp(min=1e-12)
                cdf = torch.cumsum(probs, dim=-1)
                action = (u.unsqueeze(-1) < cdf).float().argmax(dim=-1)
                logp = dist.log_prob(action)
                value = self.critic(obs)
            obs_buf[t] = obs
            mask_buf[t] = mask
            act_buf[t] = action
            logp_buf[t] = logp
            val_buf[t] = value
            step_out = self.env.step(action)
            rew_buf[t] = step_out[1]

        with torch.no_grad():
            last_obs = self.env._build_observation()
            last_value = self.critic(last_obs)
        last_done = torch.ones(E, device=self.cfg.device)  # horizon ends -> done

        return RolloutBuffer(
            obs=obs_buf, mask=mask_buf, action=act_buf, logp=logp_buf,
            reward=rew_buf, value=val_buf, last_value=last_value,
            last_done=last_done,
        )

    def compute_gae(self, rb: RolloutBuffer) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE-lambda advantages and returns. [T, E]."""
        T, E = rb.reward.shape
        adv = torch.zeros_like(rb.reward)
        last_gae = torch.zeros(E, device=self.cfg.device)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = rb.last_value
                next_done = rb.last_done
            else:
                next_value = rb.value[t + 1]
                next_done = torch.zeros(E, device=self.cfg.device)
            delta = rb.reward[t] + self.cfg.gamma * next_value * (1.0 - next_done) - rb.value[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - next_done) * last_gae
            adv[t] = last_gae
        returns = adv + rb.value
        return adv, returns

    def update(self, rb: RolloutBuffer) -> dict:
        adv, returns = self.compute_gae(rb)
        # Flatten.
        B = rb.obs.shape[0] * rb.obs.shape[1]
        obs_flat = rb.obs.reshape(B, -1)
        mask_flat = rb.mask.reshape(B, -1)
        act_flat = rb.action.reshape(B)
        logp_old_flat = rb.logp.reshape(B)
        adv_flat = adv.reshape(B)
        ret_flat = returns.reshape(B)

        # Normalize advantages.
        adv_std = float(adv_flat.std().item())
        if adv_std > 1e-8:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Sanity: verify pre-update ratio is 1.
        with torch.no_grad():
            dist_old = self.actor.distribution(obs_flat, mask_flat)
            logp_check = dist_old.log_prob(act_flat)
            ratio_check = (logp_check - logp_old_flat).exp()
            pre_ratio_offset = float((ratio_check - 1.0).abs().max().item())

        metrics_agg = {
            "adv_std": adv_std, "pre_ratio_offset": pre_ratio_offset,
            "kl_list": [], "clip_frac_list": [], "policy_loss_list": [],
            "value_loss_list": [], "entropy_list": [], "actor_grad_norm_list": [],
            "critic_grad_norm_list": [], "explained_variance": 0.0,
        }

        early_stop = False
        for epoch in range(self.cfg.epochs_per_iteration):
            perm = torch.randperm(B, device=self.cfg.device)
            for s in range(0, B, self.cfg.minibatch_size):
                bi = perm[s:s + self.cfg.minibatch_size]
                obs_b = obs_flat[bi]
                mask_b = mask_flat[bi]
                act_b = act_flat[bi]
                logp_old_b = logp_old_flat[bi]
                adv_b = adv_flat[bi]
                ret_b = ret_flat[bi]

                dist_new = self.actor.distribution(obs_b, mask_b)
                logp_new = dist_new.log_prob(act_b)
                ratio = (logp_new - logp_old_b).exp()
                with torch.no_grad():
                    kl = float((logp_old_b - logp_new).mean().item())
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = dist_new.entropy().mean()
                self.actor_opt.zero_grad()
                (policy_loss - self.cfg.entropy_coef * entropy).backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
                self.actor_opt.step()

                value_pred = self.critic(obs_b)
                value_loss = F.mse_loss(value_pred, ret_b)
                self.critic_opt.zero_grad()
                (self.cfg.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
                self.critic_opt.step()

                with torch.no_grad():
                    clip_frac = float(((ratio - 1.0).abs() > self.cfg.clip).float().mean().item())
                    actor_gn = float(sum(p.grad.norm().item() if p.grad is not None else 0.0
                                          for p in self.actor.parameters()))
                    critic_gn = float(sum(p.grad.norm().item() if p.grad is not None else 0.0
                                           for p in self.critic.parameters()))
                metrics_agg["kl_list"].append(kl)
                metrics_agg["clip_frac_list"].append(clip_frac)
                metrics_agg["policy_loss_list"].append(float(policy_loss.item()))
                metrics_agg["value_loss_list"].append(float(value_loss.item()))
                metrics_agg["entropy_list"].append(float(entropy.item()))
                metrics_agg["actor_grad_norm_list"].append(actor_gn)
                metrics_agg["critic_grad_norm_list"].append(critic_gn)

            # KL early stop (max KL per outer iter).
            if metrics_agg["kl_list"] and max(metrics_agg["kl_list"][-B // self.cfg.minibatch_size:]) > self.cfg.max_kl:
                early_stop = True
                break

        # Explained variance of critic.
        with torch.no_grad():
            v_pred = self.critic(obs_flat)
            var_y = ret_flat.var(unbiased=False)
            var_res = (ret_flat - v_pred).var(unbiased=False)
            ev = float((1.0 - var_res / (var_y + 1e-8)).item())
        metrics_agg["explained_variance"] = ev
        metrics_agg["early_stop"] = early_stop

        # Aggregate.
        def mean(xs): return sum(xs) / len(xs) if xs else 0.0
        return {
            "adv_std": metrics_agg["adv_std"],
            "pre_ratio_offset": metrics_agg["pre_ratio_offset"],
            "kl_mean": mean(metrics_agg["kl_list"]),
            "kl_max": max(metrics_agg["kl_list"]) if metrics_agg["kl_list"] else 0.0,
            "clip_frac_mean": mean(metrics_agg["clip_frac_list"]),
            "policy_loss": mean(metrics_agg["policy_loss_list"]),
            "value_loss": mean(metrics_agg["value_loss_list"]),
            "entropy": mean(metrics_agg["entropy_list"]),
            "actor_grad_norm": mean(metrics_agg["actor_grad_norm_list"]),
            "critic_grad_norm": mean(metrics_agg["critic_grad_norm_list"]),
            "explained_variance": metrics_agg["explained_variance"],
            "early_stop": early_stop,
        }

    def train_iteration(self) -> dict:
        self._assign_scenarios_and_reset()
        rb = self.collect_rollout()
        metrics = self.update(rb)
        # Also log rollout-level drop_ratio and action stats.
        drops = float(self.env.drop_ratio().mean().item())
        action_freq = torch.zeros(N_ACTIONS, device=self.cfg.device)
        for a in range(N_ACTIONS):
            action_freq[a] = (rb.action == a).float().mean().item()
        metrics["rollout_drop"] = drops
        metrics["action_freq"] = action_freq.tolist()
        metrics["iteration"] = self.iteration
        self.history.append(metrics)
        self.iteration += 1
        return metrics


def evaluate_ppo(
    actor: MaskedCategoricalActor,
    *,
    env_cfg: EnvConfig,
    scenario_seeds: list[int],
    n_action_reps: int = 4,
    sample: bool = True,
    device: str = "cpu",
    action_seed: int = 0,
) -> dict:
    """Run the actor on a list of scenarios; return per-scenario drop stats.

    `sample`: True -> stochastic sampling (primary). False -> argmax (secondary).

    Scenario arrivals are generated by `env.reset(seed=sd)` (same path as
    F2 manifest and PPO trainer), so seeds match across train/eval.
    """
    actor.eval()
    gen = torch.Generator(device=device).manual_seed(action_seed)
    per_seed_drops = []
    for sd in scenario_seeds:
        rep_drops = []
        for rep in range(n_action_reps):
            env = G3BstaLiteVecEnv(EnvConfig(
                n_envs=1, horizon=env_cfg.horizon, device=device, seed=sd,
            ))
            env.reset(seed=sd)
            for t in range(env_cfg.horizon):
                obs = env._build_observation()
                mask = env._compute_mask()
                with torch.no_grad():
                    if sample:
                        dist = actor.distribution(obs, mask)
                        u = torch.rand(1, generator=gen, device=device)
                        probs = dist.probs.clamp(min=1e-12)
                        cdf = torch.cumsum(probs, dim=-1)
                        action = (u.unsqueeze(-1) < cdf).float().argmax(dim=-1)
                    else:
                        logits = actor.forward(obs)
                        masked_logits = logits.masked_fill(~mask.bool(), float("-inf"))
                        action = masked_logits.argmax(dim=-1)
                env.step(action)
            rep_drops.append(float(env.drop_ratio()[0]))
        per_seed_drops.append(sum(rep_drops) / len(rep_drops))
    macro_mean = sum(per_seed_drops) / len(per_seed_drops) if per_seed_drops else float("nan")
    return {
        "per_seed_drops": per_seed_drops,
        "macro_mean_drop": macro_mean,
        "n_seeds": len(scenario_seeds),
        "n_action_reps": n_action_reps,
        "sample": sample,
    }
