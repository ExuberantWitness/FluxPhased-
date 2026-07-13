"""PPO trainer for the TAES commander.

Single-agent (commander only); jammer is external (adversary.py).
CTDE: privileged critic uses build_privileged() to see extra info.

Hyperparams (per spec):
  lr_actor = 3e-4, lr_critic = 1e-3
  clip = 0.2, GAE λ = 0.95, γ = 0.99
  entropy coef = 0.01, value coef = 0.5
  log_std_floor = -6 (do NOT lower to -4 — kills exploration on emission head)
  grad clip = 0.5
  rollout_horizon = 600 (one full episode), n_epochs = 4, mb_size = 64
"""

from __future__ import annotations

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

from .taes_actor_critic import TaesCommanderActorCritic, build_privileged


__all__ = ["TaesPPOTrainer"]


class _RolloutBuffer:
    """Stores transitions for one rollout epoch.

    Tensors are preallocated. Each step stores:
      obs[E, obs_dim], privileged[E, priv_dim],
      action dict (4 tensors of shape [E]),
      log_prob[E], value[E], reward[E], done[E]
    """

    def __init__(self, horizon: int, n_envs: int, obs_dim: int, priv_dim: int,
                 n_task_alloc: int, n_targets_max: int, device):
        self.horizon = int(horizon)
        self.E = int(n_envs)
        self.device = device
        self.obs = torch.zeros(horizon, n_envs, obs_dim, device=device)
        self.priv = torch.zeros(horizon, n_envs, priv_dim, device=device)
        self.alive_mask = torch.zeros(horizon, n_envs, n_targets_max,
                                       dtype=torch.bool, device=device)
        self.task_idx = torch.zeros(horizon, n_envs, dtype=torch.long, device=device)
        self.beam_idx = torch.zeros(horizon, n_envs, dtype=torch.long, device=device)
        self.laser_idx = torch.zeros(horizon, n_envs, dtype=torch.long, device=device)
        self.emission = torch.zeros(horizon, n_envs, device=device)
        self.log_prob = torch.zeros(horizon, n_envs, device=device)
        self.value = torch.zeros(horizon, n_envs, device=device)
        self.value_local = torch.zeros(horizon, n_envs, device=device)
        self.reward = torch.zeros(horizon, n_envs, device=device)
        self.done = torch.zeros(horizon, n_envs, device=device)
        self.advantage = torch.zeros(horizon, n_envs, device=device)
        self.advantage_local = torch.zeros(horizon, n_envs, device=device)
        self.ret = torch.zeros(horizon, n_envs, device=device)
        self.ret_local = torch.zeros(horizon, n_envs, device=device)
        self.t = 0

    def add(self, obs, priv, action, log_prob, value, value_local,
            reward, done, alive_mask):
        t = self.t
        self.obs[t] = obs
        self.priv[t] = priv
        self.alive_mask[t] = alive_mask
        self.task_idx[t] = action["task_alloc_idx"]
        self.beam_idx[t] = action["beam_target_idx"]
        self.laser_idx[t] = action["laser_target_idx"]
        self.emission[t] = action["emission_on"]
        self.log_prob[t] = log_prob
        self.value[t] = value
        self.value_local[t] = value_local
        self.reward[t] = reward
        self.done[t] = done
        self.t = (self.t + 1) % self.horizon

    def reset(self):
        self.t = 0


class TaesPPOTrainer:
    """Single-agent PPO for the TAES commander."""

    def __init__(
        self,
        env,
        ac: TaesCommanderActorCritic,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        minibatch_size: int = 64,
        horizon: Optional[int] = None,
        target_kl: float = 0.03,
        device: str = "cuda",
        log_std_floor: float = -6.0,
        critic_mode: str = "ctde",
        alpha_eff_alpha_max: float = 0.5,
        alpha_eff_beta: float = 2.0,
    ):
        self.env = env
        self.ac = ac.to(device)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip = float(clip)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.n_epochs = int(n_epochs)
        self.mb_size = int(minibatch_size)
        self.target_kl = float(target_kl)
        self.critic_mode = str(critic_mode)
        self.alpha_max = float(alpha_eff_alpha_max)
        self.beta = float(alpha_eff_beta)
        self.device = torch.device(device)
        self.log_std_floor = float(log_std_floor)

        horizon = horizon or env.episode_steps
        self.horizon = int(horizon)
        self.buf = _RolloutBuffer(
            horizon=horizon, n_envs=env.E,
            obs_dim=ac.obs_dim, priv_dim=ac.privileged_dim,
            n_task_alloc=ac.n_task_alloc, n_targets_max=ac.N_max,
            device=self.device,
        )

        # Separate actor/critic optimizers (different LRs)
        actor_params = list(ac.actor_trunk.parameters()) + [
            p for n, p in ac.named_parameters()
            if any(n.startswith(h) for h in
                   ["task_alloc_head", "beam_target_head",
                    "laser_target_head", "emission_head"])
        ]
        critic_params = list(ac.critic_trunk.parameters()) + \
                        list(ac.local_critic_trunk.parameters())
        self.opt_actor = torch.optim.Adam(actor_params, lr=lr_actor)
        self.opt_critic = torch.optim.Adam(critic_params, lr=lr_critic)

        # State: last obs from env
        self._last_obs = None

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_rollout(self, jammer=None, deterministic: bool = False):
        """Collect exactly `horizon` steps. Resets env when done.

        Returns dict of per-step metrics (means over rollout).
        """
        env = self.env
        ac = self.ac
        dev = self.device
        self.buf.reset()

        if self._last_obs is None:
            obs_dict = env.reset()
            self._last_obs = obs_dict["obs"]
            if jammer is not None:
                jammer.reset(env.E, 1, dev)

        ep_returns = torch.zeros(env.E, device=dev)
        ep_lens = torch.zeros(env.E, device=dev)
        n_resets = 0
        # cumulative episode metrics
        sum_kill = 0.0
        sum_surv = 0.0
        sum_exp = 0.0
        sum_homejam = 0.0
        sum_trackloss = 0.0
        n_done = 0

        for t in range(self.horizon):
            obs = self._last_obs
            priv = build_privileged(env, env._last_jam if hasattr(env, "_last_jam") and env._last_jam is not None else torch.zeros(env.E, device=dev))
            alive_mask = env.target_alive_mask
            action, log_prob, value, value_local = ac(obs, priv, target_alive_mask=alive_mask)
            # action: dict with internal indices (task_alloc_idx, beam, laser, emission)
            # env needs one-hot task_alloc + indices
            env_action = ac.get_action_for_env(obs, deterministic=False,
                                                target_alive_mask=alive_mask)
            # Override with sampled indices to match log_prob
            env_action["task_alloc"] = F.one_hot(
                action["task_alloc_idx"], ac.n_task_alloc).float()
            env_action["beam_target_idx"] = action["beam_target_idx"]
            env_action["laser_target_idx"] = action["laser_target_idx"]
            env_action["emission_on"] = action["emission_on"]

            obs_dict_new, reward, done, info = env.step(env_action, jammer=jammer)
            self.buf.add(obs, priv, action, log_prob, value, value_local,
                         reward, done.float(), alive_mask)

            ep_returns += reward
            ep_lens += 1
            self._last_obs = obs_dict_new["obs"]

            # Accumulate metrics
            sum_kill += float(info["n_kills_step"].sum())
            sum_exp += float(info["exposure"].sum())
            sum_homejam += float(info["homejam_death"].sum())
            sum_trackloss += float(info["track_loss_rate"].sum())

            # Reset envs that are done
            if done.any():
                # Snapshot per-episode metrics from done envs
                sum_surv += float(ep_returns[done].sum())  # proxy
                n_done += int(done.sum())
                # Reset done envs in-place (vectorized env doesn't auto-reset)
                # We do a manual reset of just done envs by swapping state.
                # Easier: full reset only when *all* done; otherwise mask out
                # done envs by leaving them in a terminal state until all done.
                if done.all():
                    obs_dict = env.reset()
                    self._last_obs = obs_dict["obs"]
                    if jammer is not None:
                        jammer.reset(env.E, 1, dev)
                    ep_returns = torch.zeros(env.E, device=dev)
                    ep_lens = torch.zeros(env.E, device=dev)
                    n_resets += 1

        # Compute GAE / returns using the final value bootstrap
        # If env is not done at end of horizon, bootstrap from critic
        with torch.no_grad():
            last_priv = build_privileged(
                env,
                env._last_jam if hasattr(env, "_last_jam") and env._last_jam is not None else torch.zeros(env.E, device=dev))
            last_value = ac.critic_trunk(
                torch.cat([self._last_obs, last_priv], dim=-1)).squeeze(-1)
            last_value_local = ac.local_critic_trunk(self._last_obs).squeeze(-1)

        self._compute_gae(last_value, last_value_local)

        return {
            "ep_rew_mean": float(ep_returns.mean()),
            "ep_len_mean": float(ep_lens.mean()),
            "n_kills_total": sum_kill,
            "exposure_mean": sum_exp / max(1, self.horizon * env.E),
            "homejam_total": sum_homejam,
            "trackloss_mean": sum_trackloss / max(1, self.horizon * env.E),
            "n_done": n_done,
            "n_resets": n_resets,
        }

    def _compute_gae(self, last_value: torch.Tensor, last_value_local: torch.Tensor):
        """GAE-λ for central critic (A_team) and local critic (A_agent),
        then blend via noise-robust α_eff:
            α_eff[t] = α_max · exp(-β · trace_P_norm[t])
            adv[t]   = (1 - α_eff) · A_agent[t] + α_eff · A_team[t]
        priv[:,4] is trace_P / tau_track_nominal (verified in patch ④).

        critic_mode="ctde": blend as above (MAPPO per spec §3.2).
        critic_mode="ippo": use only A_agent (no central critic), no blend.
        """
        gamma, lam = self.gamma, self.gae_lambda
        H = self.horizon
        E = self.buf.E
        adv_team = torch.zeros(E, device=self.device)
        adv_agent = torch.zeros(E, device=self.device)
        for t in reversed(range(H)):
            non_term = 1.0 - self.buf.done[t]
            next_value = last_value if t == H - 1 else self.buf.value[t + 1]
            next_value_local = (last_value_local if t == H - 1
                                else self.buf.value_local[t + 1])
            delta_team = (self.buf.reward[t]
                          + gamma * next_value * non_term
                          - self.buf.value[t])
            adv_team = delta_team + gamma * lam * non_term * adv_team
            self.buf.advantage[t] = adv_team
            self.buf.ret[t] = adv_team + self.buf.value[t]

            delta_agent = (self.buf.reward[t]
                           + gamma * next_value_local * non_term
                           - self.buf.value_local[t])
            adv_agent = delta_agent + gamma * lam * non_term * adv_agent
            self.buf.advantage_local[t] = adv_agent
            self.buf.ret_local[t] = adv_agent + self.buf.value_local[t]

        # Blend via α_eff (skip for IPPO mode)
        if self.critic_mode == "ippo":
            self.buf.advantage = self.buf.advantage_local.clone()
        else:
            trace_P_norm = self.buf.priv[..., 4].clamp(0.0, 50.0)  # [H, E]
            alpha_eff = self.alpha_max * torch.exp(-self.beta * trace_P_norm)
            self.buf.advantage = ((1.0 - alpha_eff) * self.buf.advantage_local
                                  + alpha_eff * self.buf.advantage)

        # Normalize advantages (helps PPO stability)
        adv_flat = self.buf.advantage.reshape(-1)
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std() + 1e-8
        self.buf.advantage = (self.buf.advantage - adv_mean) / adv_std

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """PPO update over the rollout buffer.

        Returns per-metric means.
        """
        H, E = self.horizon, self.env.E
        N = H * E
        mb = self.mb_size
        device = self.device

        # Flatten
        obs_flat = self.buf.obs.reshape(N, -1)
        priv_flat = self.buf.priv.reshape(N, -1)
        alive_mask_flat = self.buf.alive_mask.reshape(N, -1)
        task_flat = self.buf.task_idx.reshape(-1)
        beam_flat = self.buf.beam_idx.reshape(-1)
        laser_flat = self.buf.laser_idx.reshape(-1)
        emission_flat = self.buf.emission.reshape(-1)
        log_prob_old = self.buf.log_prob.reshape(-1)
        value_old = self.buf.value.reshape(-1)
        value_old_local = self.buf.value_local.reshape(-1)
        ret_flat = self.buf.ret.reshape(-1)
        ret_local_flat = self.buf.ret_local.reshape(-1)
        adv_flat = self.buf.advantage.reshape(-1)

        action_old = {
            "task_alloc_idx": task_flat,
            "beam_target_idx": beam_flat,
            "laser_target_idx": laser_flat,
            "emission_on": emission_flat,
        }

        metrics_acc = {"policy_loss": 0.0, "value_loss": 0.0,
                       "value_loss_local": 0.0,
                       "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        n_updates = 0

        for epoch in range(self.n_epochs):
            idx = torch.randperm(N, device=device)
            for i in range(0, N, mb):
                b = idx[i:i + mb]
                if b.numel() < 8:  # skip tiny tail
                    continue
                obs_b = obs_flat[b]
                priv_b = priv_flat[b]
                mask_b = alive_mask_flat[b]
                act_b = {k: v[b] for k, v in action_old.items()}
                lp_old_b = log_prob_old[b]
                v_old_b = value_old[b]
                v_old_local_b = value_old_local[b]
                ret_b = ret_flat[b]
                ret_local_b = ret_local_flat[b]
                adv_b = adv_flat[b]

                log_prob_b, value_b, entropy_b, value_local_b = self.ac.evaluate_actions(
                    obs_b, act_b, privileged=priv_b, target_alive_mask=mask_b)

                # PPO clipped surrogate
                ratio = torch.exp(log_prob_b - lp_old_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Central value loss (clipped) — only in ctde mode
                if self.critic_mode == "ctde":
                    value_clipped = v_old_b + torch.clamp(
                        value_b - v_old_b, -self.clip, self.clip)
                    v1 = (value_b - ret_b).pow(2)
                    v2 = (value_clipped - ret_b).pow(2)
                    value_loss = 0.5 * torch.max(v1, v2).mean()
                else:
                    value_loss = torch.tensor(0.0, device=device)

                # Local value loss (always trained — IPPO uses only this)
                value_local_clipped = v_old_local_b + torch.clamp(
                    value_local_b - v_old_local_b, -self.clip, self.clip)
                vl1 = (value_local_b - ret_local_b).pow(2)
                vl2 = (value_local_clipped - ret_local_b).pow(2)
                value_loss_local = 0.5 * torch.max(vl1, vl2).mean()

                entropy = entropy_b.mean()
                loss = (policy_loss
                        - self.entropy_coef * entropy
                        + self.value_coef * value_loss
                        + self.value_coef * value_loss_local)

                # Joint backward
                self.opt_actor.zero_grad(set_to_none=True)
                self.opt_critic.zero_grad(set_to_none=True)
                loss.backward()
                # Gradient clip on combined params
                params = (list(self.ac.actor_trunk.parameters()) +
                          list(self.ac.critic_trunk.parameters()) +
                          list(self.ac.local_critic_trunk.parameters()) +
                          [p for n, p in self.ac.named_parameters()
                           if any(n.startswith(h) for h in
                                  ["task_alloc_head", "beam_target_head",
                                   "laser_target_head", "emission_head"])])
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

                with torch.no_grad():
                    approx_kl = (lp_old_b - log_prob_b).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.clip).float().mean().item()

                metrics_acc["policy_loss"] += policy_loss.item()
                metrics_acc["value_loss"] += value_loss.item()
                metrics_acc["value_loss_local"] += value_loss_local.item()
                metrics_acc["entropy"] += entropy.item()
                metrics_acc["approx_kl"] += approx_kl
                metrics_acc["clip_frac"] += clip_frac
                n_updates += 1

            # Early stop on large KL
            with torch.no_grad():
                # Recompute KL over full buffer (cheap)
                lp_full, _, _, _ = self.ac.evaluate_actions(
                    obs_flat, action_old, privileged=priv_flat,
                    target_alive_mask=alive_mask_flat)
                approx_kl_full = (log_prob_old - lp_full).mean().item()
                if approx_kl_full > 1.5 * self.target_kl:
                    break

        for k in metrics_acc:
            metrics_acc[k] /= max(1, n_updates)
        metrics_acc["n_updates"] = n_updates
        return metrics_acc

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def train(
        self,
        n_iterations: int,
        jammer=None,
        eval_fn=None,
        eval_every: int = 10,
        save_dir: Optional[str] = None,
        save_every: int = 50,
        log_prefix: str = "taes_ppo",
    ) -> Dict[str, List[float]]:
        """Train for n_iterations × horizon steps.

        Args:
            jammer: optional jammer (already reset); if None, no jamming.
            eval_fn: callable(ac) → dict of eval metrics. Called every eval_every.
            save_dir: if set, save checkpoints here.
        """
        history = {"iter": [], "ep_rew": [], "kill": [], "trackloss": [],
                   "value_loss": [], "entropy": [], "approx_kl": []}

        os.makedirs(save_dir, exist_ok=True) if save_dir else None

        for it in range(n_iterations):
            t0 = time.time()
            rollout_metrics = self.collect_rollout(jammer=jammer)
            update_metrics = self.update()
            dt = time.time() - t0

            history["iter"].append(it)
            history["ep_rew"].append(rollout_metrics["ep_rew_mean"])
            history["kill"].append(rollout_metrics["n_kills_total"])
            history["trackloss"].append(rollout_metrics["trackloss_mean"])
            history["value_loss"].append(update_metrics["value_loss"])
            history["entropy"].append(update_metrics["entropy"])
            history["approx_kl"].append(update_metrics["approx_kl"])

            if it % 5 == 0:
                print(f"[{log_prefix}] it={it:4d} ep_rew={rollout_metrics['ep_rew_mean']:+.2f} "
                      f"kill={rollout_metrics['n_kills_total']:.1f} "
                      f"trackloss={rollout_metrics['trackloss_mean']:.3f} "
                      f"v_loss={update_metrics['value_loss']:.3f} "
                      f"H={update_metrics['entropy']:.3f} "
                      f"kl={update_metrics['approx_kl']:.4f} "
                      f"dt={dt:.1f}s",
                      flush=True)

            if eval_fn is not None and (it + 1) % eval_every == 0:
                em = eval_fn(self.ac)
                em_str = " ".join(f"{k}={v:.3f}" for k, v in em.items())
                print(f"[{log_prefix}] eval @ it={it+1}: {em_str}", flush=True)

            if save_dir and (it + 1) % save_every == 0:
                path = os.path.join(save_dir, f"{log_prefix}_it{it+1}.pt")
                torch.save(self.ac.state_dict(), path)

        return history
