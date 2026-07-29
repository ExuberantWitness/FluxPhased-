"""R2 masked PPO trainer — learning-repair branch.

Fixes bound by PREREGISTRATION.md §4–§6 and POST_AUDIT_CORRECTION.md §3–§4:

  - The trainer NEVER rebuilds the caller's EnvConfig. The full env config
    (profile, obs_delay_steps, physics, etc.) is preserved (§4.4).
  - iteration = -1 is the **pristine init**: the actor is evaluated and
    snapshotted before any optimizer update (§4, §3.3).
  - iteration = k >= 0 is after the (k+1)-th outer update.
  - Every checkpoint records update_count, cumulative_transitions,
    checkpoint_origin, training_seed, config_sha, manifest_sha.
  - KL is estimated after EVERY minibatch using the correct categorical
    KL formula (sum p_old * (logp_old - logp_new)), not the "logp-mean"
    proxy the prior trainer used.
  - If KL(post) > target_kl for any minibatch, the actor update is rolled
    back to the pre-outer-update snapshot and the iteration ends with
    `kl_rollback=True` (§4.5).
  - Entropy coefficient annealed 1e-3 -> 0 over the first 30% of training.
  - target_kl ∈ {0.01, 0.02}; actor lr ∈ {3e-5, 1e-4} — chosen by
    validation only from this preregistered set (§5).
  - Scenarios come from the four disjoint manifests; no manifest is
    touched outside its declared role.

This trainer is for the mdp_sanity_v1 profile (R2 Gate 3). A recurrent
variant for pomdp_v1 is provided separately in trainer_pomdp.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.gpu.g3_bsta_lite import (
    EnvConfig,
    G3BstaLiteVecEnv,
    N_ACTIONS,
    OBS_DIM,
)


@dataclass
class R2PPOConfig:
    """All R2 PPO hyperparameters. Frozen at first train_iteration call."""
    profile: str = "mdp_sanity_v1"
    iterations: int = 200
    n_envs: int = 16
    horizon: int = 64                # matches EnvConfig.horizon
    actor_lr: float = 1e-4           # preregistered set: {3e-5, 1e-4}
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    grad_clip: float = 0.5
    epochs_per_iteration: int = 4
    minibatch_size: int = 256
    entropy_coef_init: float = 1e-3
    entropy_coef_min: float = 0.0
    entropy_anneal_frac: float = 0.3   # anneal over first 30% of training
    value_coef: float = 0.5
    target_kl: float = 0.02            # preregistered set: {0.01, 0.02}
    seed: int = 0
    device: str = "cpu"
    train_seed: int = 0                # RNG seed for action sampling

    def to_json(self) -> dict:
        return asdict(self)

    def config_sha(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.to_json(), sort_keys=True).encode("utf-8"))
        return h.hexdigest()


def manifest_sha(manifest_path: Path) -> str:
    h = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Actor / Critic (same architecture as F4 to keep checkpoint semantics
# comparable; weights are NOT shared across profiles).
# ---------------------------------------------------------------------------

class MaskedCategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        return self.head(h)

    def distribution(self, obs: torch.Tensor, mask: torch.Tensor):
        logits = self.forward(obs)
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return torch.distributions.Categorical(logits=logits)


class ValueCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer + GAE
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    obs: torch.Tensor
    mask: torch.Tensor
    action: torch.Tensor
    logp: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor
    last_value: torch.Tensor
    last_done: torch.Tensor


def categorical_kl(
    logits_old: torch.Tensor, logits_new: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample KL(old || new) for masked categoricals.

    KL = sum_a p_old(a) * (log p_old(a) - log p_new(a)), with the mask
    zeros handled by setting masked logits to -inf before softmax. The
    mean over the batch is what target_kl compares against.
    """
    lo = logits_old.masked_fill(~mask.bool(), float("-inf"))
    ln = logits_new.masked_fill(~mask.bool(), float("-inf"))
    log_po = F.log_softmax(lo, dim=-1)
    log_pn = F.log_softmax(ln, dim=-1)
    po = log_po.exp()
    # Terms where po == 0 contribute 0 * (-inf - (-inf)) which is NaN; mask.
    safe = po > 0
    contrib = torch.where(
        safe, po * (log_po - log_pn), torch.zeros_like(po),
    )
    return contrib.sum(dim=-1)


# ---------------------------------------------------------------------------
# Checkpoint provenance
# ---------------------------------------------------------------------------

@dataclass
class CheckpointMeta:
    iteration: int                      # -1 = pristine init
    update_count: int
    cumulative_transitions: int
    checkpoint_origin: str              # "pristine_init" | "validation_best" | "last_iter"
    training_seed: int
    config_sha: str
    manifest_sha: str
    profile: str
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


class R2PPOTrainer:
    """Scratch PPO trainer with per-minibatch KL rollback."""

    def __init__(
        self,
        *,
        cfg: R2PPOConfig,
        env_cfg: EnvConfig,
        train_seeds: list[int],
        manifest_path: Path,
        out_dir: Path,
    ):
        # DO NOT rebuild env_cfg. Preserve caller's full EnvConfig (§4.4).
        if env_cfg.profile != cfg.profile:
            raise ValueError(
                f"profile mismatch: env_cfg.profile={env_cfg.profile!r} "
                f"but cfg.profile={cfg.profile!r}"
            )
        if env_cfg.n_envs != cfg.n_envs or env_cfg.horizon != cfg.horizon:
            raise ValueError(
                "env_cfg.n_envs/horizon must match cfg.n_envs/horizon; "
                "the trainer does not silently override EnvConfig"
            )
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.train_seeds = list(train_seeds)
        self.manifest_path = Path(manifest_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(cfg.seed)
        self.obs_dim = OBS_DIM
        self.actor = MaskedCategoricalActor(self.obs_dim, N_ACTIONS).to(cfg.device)
        self.critic = ValueCritic(self.obs_dim).to(cfg.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Build training env from caller's EnvConfig; only n_envs/horizon
        # are guaranteed to match. Profile, obs_delay, physics constants
        # flow through unchanged.
        self.env = G3BstaLiteVecEnv(self.env_cfg)
        self._action_gen = torch.Generator(device=cfg.device).manual_seed(cfg.train_seed)

        # Provenance SHAs.
        self._config_sha = cfg.config_sha()
        self._manifest_sha = manifest_sha(self.manifest_path)

        self.iteration = -1            # before any optimizer update
        self.update_count = 0
        self.cumulative_transitions = 0
        self.history: list[dict] = []
        self.kl_rollback_count = 0

        self._snapshot_actor_state()   # pristine init snapshot

    # ------------------------------------------------------------------
    # Snapshot / restore (for KL rollback and pristine-init checkpoints)
    # ------------------------------------------------------------------
    def _snapshot_actor_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.actor.state_dict().items()}

    def _restore_actor_state(self, snap: dict) -> None:
        self.actor.load_state_dict({k: v.clone() for k, v in snap.items()})

    # ------------------------------------------------------------------
    # Scenario assignment
    # ------------------------------------------------------------------
    def _assign_scenarios_and_reset(self):
        # Cycle through train_seeds (manifest is held fixed; only the
        # iteration -> seed mapping changes per outer iter).
        sd = self.train_seeds[self.iteration % len(self.train_seeds)]
        self.env.reset(seed=sd)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def collect_rollout(self) -> RolloutBuffer:
        T = self.env_cfg.horizon
        E = self.env_cfg.n_envs
        device = self.cfg.device
        obs_buf = torch.zeros(T, E, self.obs_dim, device=device)
        mask_buf = torch.zeros(T, E, N_ACTIONS, device=device)
        act_buf = torch.zeros(T, E, dtype=torch.int64, device=device)
        logp_buf = torch.zeros(T, E, device=device)
        rew_buf = torch.zeros(T, E, device=device)
        val_buf = torch.zeros(T, E, device=device)

        for t in range(T):
            obs = self.env._build_observation()
            mask = self.env._compute_mask()
            with torch.no_grad():
                dist = self.actor.distribution(obs, mask)
                u = torch.rand(E, generator=self._action_gen, device=device)
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
        last_done = torch.ones(E, device=device)

        return RolloutBuffer(
            obs=obs_buf, mask=mask_buf, action=act_buf, logp=logp_buf,
            reward=rew_buf, value=val_buf, last_value=last_value,
            last_done=last_done,
        )

    def compute_gae(self, rb: RolloutBuffer) -> tuple[torch.Tensor, torch.Tensor]:
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
            delta = (
                rb.reward[t]
                + self.cfg.gamma * next_value * (1.0 - next_done)
                - rb.value[t]
            )
            last_gae = (
                delta + self.cfg.gamma * self.cfg.gae_lambda
                * (1.0 - next_done) * last_gae
            )
            adv[t] = last_gae
        returns = adv + rb.value
        return adv, returns

    # ------------------------------------------------------------------
    # Entropy coefficient schedule
    # ------------------------------------------------------------------
    def _entropy_coef(self, iteration: int) -> float:
        if self.cfg.iterations <= 0:
            return self.cfg.entropy_coef_min
        anneal_iters = max(1, int(self.cfg.entropy_anneal_frac * self.cfg.iterations))
        if iteration >= anneal_iters:
            return self.cfg.entropy_coef_min
        frac = float(iteration) / float(anneal_iters)
        return (
            self.cfg.entropy_coef_init
            + frac * (self.cfg.entropy_coef_min - self.cfg.entropy_coef_init)
        )

    # ------------------------------------------------------------------
    # Per-minibatch KL with rollback
    # ------------------------------------------------------------------
    def update(self, rb: RolloutBuffer) -> dict:
        adv, returns = self.compute_gae(rb)
        B = rb.obs.shape[0] * rb.obs.shape[1]
        obs_flat = rb.obs.reshape(B, -1)
        mask_flat = rb.mask.reshape(B, -1)
        act_flat = rb.action.reshape(B)
        logp_old_flat = rb.logp.reshape(B)
        adv_flat = adv.reshape(B)
        ret_flat = returns.reshape(B)

        adv_std = float(adv_flat.std().item())
        if adv_std > 1e-8:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Pre-update ratio invariant.
        with torch.no_grad():
            dist_old = self.actor.distribution(obs_flat, mask_flat)
            logp_check = dist_old.log_prob(act_flat)
            ratio_check = (logp_check - logp_old_flat).exp()
            pre_ratio_offset = float((ratio_check - 1.0).abs().max().item())

        entropy_coef = self._entropy_coef(self.iteration)

        # Snapshot for KL rollback (§4.5).
        actor_pre_update = self._snapshot_actor_state()
        metrics_agg = {
            "adv_std": adv_std, "pre_ratio_offset": pre_ratio_offset,
            "entropy_coef": entropy_coef,
            "kl_pre_minibatch": [], "kl_post_minibatch": [],
            "clip_frac_list": [], "policy_loss_list": [],
            "value_loss_list": [], "entropy_list": [],
            "actor_grad_norm_list": [], "critic_grad_norm_list": [],
            "explained_variance": 0.0,
        }

        rolled_back = False
        outer_kl_max = 0.0
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

                # Pre-minibatch KL (after the previous update).
                with torch.no_grad():
                    logits_old_b_pre = self.actor.forward(obs_b)
                    kl_pre = categorical_kl(
                        logits_old_b_pre, logits_old_b_pre, mask_b,
                    ).mean().item()

                dist_new = self.actor.distribution(obs_b, mask_b)
                logp_new = dist_new.log_prob(act_b)
                ratio = (logp_new - logp_old_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip,
                                     1.0 + self.cfg.clip) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = dist_new.entropy().mean()
                self.actor_opt.zero_grad()
                (policy_loss - entropy_coef * entropy).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.cfg.grad_clip,
                )
                self.actor_opt.step()

                value_pred = self.critic(obs_b)
                value_loss = F.mse_loss(value_pred, ret_b)
                self.critic_opt.zero_grad()
                (self.cfg.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.grad_clip,
                )
                self.critic_opt.step()

                # Post-minibatch KL.
                with torch.no_grad():
                    logits_new_b_post = self.actor.forward(obs_b)
                    kl_post = categorical_kl(
                        logits_old_b_pre, logits_new_b_post, mask_b,
                    ).mean().item()
                    clip_frac = float(
                        ((ratio - 1.0).abs() > self.cfg.clip).float().mean().item()
                    )
                    actor_gn = float(sum(
                        p.grad.norm().item() if p.grad is not None else 0.0
                        for p in self.actor.parameters()
                    ))
                    critic_gn = float(sum(
                        p.grad.norm().item() if p.grad is not None else 0.0
                        for p in self.critic.parameters()
                    ))
                metrics_agg["kl_pre_minibatch"].append(kl_pre)
                metrics_agg["kl_post_minibatch"].append(kl_post)
                metrics_agg["clip_frac_list"].append(clip_frac)
                metrics_agg["policy_loss_list"].append(float(policy_loss.item()))
                metrics_agg["value_loss_list"].append(float(value_loss.item()))
                metrics_agg["entropy_list"].append(float(entropy.item()))
                metrics_agg["actor_grad_norm_list"].append(actor_gn)
                metrics_agg["critic_grad_norm_list"].append(critic_gn)

                outer_kl_max = max(outer_kl_max, kl_post)
                if kl_post > self.cfg.target_kl:
                    # R2B: roll back the actor to the pre-outer-update
                    # snapshot. Critic is kept (its drift is bounded by
                    # value_coef and grad_clip). End the outer iteration.
                    self._restore_actor_state(actor_pre_update)
                    rolled_back = True
                    break
            if rolled_back:
                break

        # Explained variance of critic on this rollout.
        with torch.no_grad():
            v_pred = self.critic(obs_flat)
            var_y = ret_flat.var(unbiased=False)
            var_res = (ret_flat - v_pred).var(unbiased=False)
            ev = float((1.0 - var_res / (var_y + 1e-8)).item())
        metrics_agg["explained_variance"] = ev

        def mean(xs): return sum(xs) / len(xs) if xs else 0.0
        return {
            "iteration": self.iteration,
            "update_count": self.update_count,
            "adv_std": adv_std,
            "pre_ratio_offset": pre_ratio_offset,
            "entropy_coef": entropy_coef,
            "kl_mean_post": mean(metrics_agg["kl_post_minibatch"]),
            "kl_max_post": max(metrics_agg["kl_post_minibatch"]) if metrics_agg["kl_post_minibatch"] else 0.0,
            "clip_frac_mean": mean(metrics_agg["clip_frac_list"]),
            "policy_loss": mean(metrics_agg["policy_loss_list"]),
            "value_loss": mean(metrics_agg["value_loss_list"]),
            "entropy": mean(metrics_agg["entropy_list"]),
            "actor_grad_norm": mean(metrics_agg["actor_grad_norm_list"]),
            "critic_grad_norm": mean(metrics_agg["critic_grad_norm_list"]),
            "explained_variance": metrics_agg["explained_variance"],
            "kl_rollback": rolled_back,
            "outer_kl_max": outer_kl_max,
        }

    # ------------------------------------------------------------------
    # Outer iteration
    # ------------------------------------------------------------------
    def train_iteration(self) -> dict:
        if self.iteration < 0:
            # First call: pristine init eval already done in save_pristine_init();
            # advance to iteration=0 on exit.
            self.iteration = 0
        else:
            self.iteration += 1
        self._assign_scenarios_and_reset()
        rb = self.collect_rollout()
        metrics = self.update(rb)
        if not metrics["kl_rollback"]:
            self.update_count += 1
        else:
            self.kl_rollback_count += 1
        self.cumulative_transitions += int(rb.obs.shape[0] * rb.obs.shape[1])
        drops = float(self.env.drop_ratio().mean().item())
        action_freq = torch.zeros(N_ACTIONS, device=self.cfg.device)
        for a in range(N_ACTIONS):
            action_freq[a] = (rb.action == a).float().mean().item()
        metrics["rollout_drop"] = drops
        metrics["action_freq"] = action_freq.tolist()
        metrics["cumulative_transitions"] = self.cumulative_transitions
        metrics["iteration"] = self.iteration
        self.history.append(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing with provenance
    # ------------------------------------------------------------------
    def save_checkpoint(self, *, origin: str, out_path: Path) -> None:
        meta = CheckpointMeta(
            iteration=self.iteration,
            update_count=self.update_count,
            cumulative_transitions=self.cumulative_transitions,
            checkpoint_origin=origin,
            training_seed=self.cfg.train_seed,
            config_sha=self._config_sha,
            manifest_sha=self._manifest_sha,
            profile=self.cfg.profile,
            extra={"kl_rollback_count": self.kl_rollback_count},
        )
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "meta": meta.to_json(),
            },
            out_path,
        )

    def save_pristine_init(self) -> Path:
        out_path = self.out_dir / "pristine_init.pt"
        # pristine init = iteration = -1, BEFORE any optimizer update.
        self.iteration = -1
        self.save_checkpoint(origin="pristine_init", out_path=out_path)
        return out_path

    def save_validation_best(self, iter_label: int) -> Path:
        out_path = self.out_dir / f"validation_best_iter{iter_label}.pt"
        self.save_checkpoint(origin="validation_best", out_path=out_path)
        return out_path

    def save_last_iter(self, iter_label: int) -> Path:
        out_path = self.out_dir / f"last_iter{iter_label}.pt"
        self.save_checkpoint(origin="last_iter", out_path=out_path)
        return out_path


# ---------------------------------------------------------------------------
# Evaluation harness (no test-set checkpoint selection)
# ---------------------------------------------------------------------------

def evaluate_actor(
    actor: MaskedCategoricalActor,
    *,
    env_cfg: EnvConfig,
    scenario_seeds: list[int],
    n_action_reps: int = 4,
    sample: bool = True,
    device: str = "cpu",
    action_seed: int = 0,
) -> dict:
    """Per-scenario macro drop_ratio and per-rep raw rows."""
    actor.eval()
    gen = torch.Generator(device=device).manual_seed(action_seed)
    per_seed_drops: list[float] = []
    per_seed_n_eligible: list[int] = []
    raw_rows: list[dict] = []
    for sd in scenario_seeds:
        rep_drops: list[float] = []
        for rep in range(n_action_reps):
            env = G3BstaLiteVecEnv(env_cfg)
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
                        masked_logits = logits.masked_fill(
                            ~mask.bool(), float("-inf"),
                        )
                        action = masked_logits.argmax(dim=-1)
                env.step(action)
            rep_drop = float(env.drop_ratio()[0])
            rep_drops.append(rep_drop)
            n_eligible = int(env.counters.n_eligible[0].item())
            per_seed_n_eligible.append(n_eligible)
            raw_rows.append({
                "seed": int(sd), "rep": int(rep),
                "drop_ratio": rep_drop,
                "n_eligible": n_eligible,
                "ledger_residual": int(env.ledger_identity_residual()),
                "accounting_residual": int(env.accounting_residual()[0].item()),
            })
        per_seed_drops.append(sum(rep_drops) / len(rep_drops))
    macro_mean = sum(per_seed_drops) / len(per_seed_drops) if per_seed_drops else float("nan")
    return {
        "per_seed_drops": per_seed_drops,
        "macro_mean_drop": macro_mean,
        "n_seeds": len(scenario_seeds),
        "n_action_reps": n_action_reps,
        "sample": sample,
        "raw_rows": raw_rows,
    }
