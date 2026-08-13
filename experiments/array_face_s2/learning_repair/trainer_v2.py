"""S2 PPO trainer v2 — N-head framework + per-head entropy + numerical hardening.

Extends the amend02 baseline trainer (trainer.py) with three improvements that
target the ~0.211 deterministic plateau observed across all 3 amend02 seeds:

  (1) N-head actor (MultiHeadActor) — bit-exact equivalent to MultiDiscreteActor
      for S2's two-head config, but generalizes to S3's three-head Bernoulli(5)
      cell-binding by registration. See actor_heads.py.
  (2) Per-head entropy coefficients — beam head can anneal faster than base,
      addressing the observation that beam entropy stayed near log(5)≈1.61 for
      the entire amend02 run (never converged), while base collapsed to ~0.
  (3) log-ratio clamp [-20, 20] + return normalization — numerical hardening
      borrowed from algo/_shared/ppo/ppo_trainer.py, prevents single bad
      samples from blowing up the batch gradient.

BACKWARD COMPATIBILITY: when per_head_entropy=False and normalize_returns=False
and use_multihead=True with S2's two categorical heads, v2 reproduces amend02's
training trajectory bit-exactly (same RNG stream, same losses). The amend02
baseline in trainer.py is untouched and remains the frozen reference.

Usage mirrors run_s2_ppo.py; see run_s2_ppo_v2.py for the driver.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.array_face_s2 import (
    EnvConfig, ArrayFaceS2VecEnv, RadarULAConfig, JammerULAConfig, OBS_DIM_S2,
    N_ACTIONS_BASE, N_ACTIONS_BEAM,
)
# Reuse the pure-data helpers from the amend02 baseline trainer. These
# (ValueCritic, RolloutBuffer, CheckpointMeta, manifest_sha) have no dependency
# on S2PPOTrainer, so the import is acyclic.
from experiments.array_face_s2.learning_repair.trainer import (
    ValueCritic, RolloutBuffer, CheckpointMeta, manifest_sha,
)


@dataclass
class S2PPOConfigV2:
    """Extends S2PPOConfig with per-head entropy + normalization toggles.

    New fields (all default-off for amend02 bit-exact compatibility):
      per_head_entropy:        if True, use entropy_coef_per_head / anneal_per_head
      entropy_coef_per_head:   {head_name: coef_init} e.g. {"base":5e-3,"beam":5e-3}
      entropy_anneal_frac_per_head: {head_name: frac} e.g. {"base":0.5,"beam":0.3}
      normalize_returns:       running-mean-std normalization of GAE returns
      return_norm_clip:        clip normalized returns to ±clip (0 = no clip)
      log_ratio_clamp:         clamp (logp_new - logp_old) to ±value (0 = no clamp)
    """
    profile: str = "mdp_sanity_v1"
    iterations: int = 1000
    n_envs: int = 16
    horizon: int = 64
    actor_lr: float = 3e-5
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    grad_clip: float = 0.5
    epochs_per_iteration: int = 4
    minibatch_size: int = 256
    # entropy (global, used when per_head_entropy=False)
    entropy_coef_init: float = 5e-3
    entropy_coef_min: float = 0.0
    entropy_anneal_frac: float = 0.5
    # per-head entropy overrides (v2)
    per_head_entropy: bool = False
    entropy_coef_per_head: dict = field(default_factory=dict)
    entropy_anneal_frac_per_head: dict = field(default_factory=dict)
    value_coef: float = 0.5
    target_kl: float = 0.02
    # numerical hardening (v2)
    normalize_returns: bool = False
    return_norm_clip: float = 0.0
    log_ratio_clamp: float = 0.0
    # B1: privileged critic (asymmetric value head + distillation)
    use_privileged_critic: bool = False
    privileged_value_coef: float = 0.5    # loss weight for privileged value head
    distill_coef: float = 0.1             # weight for (obs_value - privileged_value.detach())^2
    seed: int = 0
    device: str = "cpu"
    train_seed: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    def config_sha(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.to_json(), sort_keys=True).encode("utf-8"))
        return h.hexdigest()


class RunningMeanStd:
    """Online running mean/std (Welford-style) for return normalization.

    Borrowed from algo/_shared/ppo/buffer.py. Tracks over a scalar stream with
    exponential moving average; used to normalize GAE returns before value fit.
    """

    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: torch.Tensor):
        batch_mean = float(x.mean().item())
        batch_var = float(x.var(unbiased=False).item())
        batch_count = x.numel()
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: torch.Tensor, clip: float = 0.0) -> torch.Tensor:
        out = (x - self.mean) / (self.var ** 0.5 + 1e-8)
        if clip > 0:
            out = out.clamp(-clip, clip)
        return out


class PrivilegedValueCritic(nn.Module):
    """Value critic on privileged state (B1: asymmetric critic).

    Same architecture as ValueCritic (2-layer MLP) but operates on the env's
    privileged observation (PRIVILEGED_DIM_S2 = 14: pending + health + beam
    azimuths). Used during training to produce a more accurate value estimate;
    at deploy time the obs-only ValueCritic is used. A distillation loss
    (obs_value ~ privileged_value.detach()) transfers the privileged knowledge
    into the deployable obs head.
    """
    def __init__(self, priv_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(priv_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, priv: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(priv))
        h = torch.tanh(self.fc2(h))
        return self.head(h).squeeze(-1)


class S2PPOTrainerV2:
    """PPO trainer with N-head actor + per-head entropy + numerical hardening.

    The PPO update math (clipped surrogate, GAE, KL-rollback) is identical to
    S2PPOTrainer. Differences are localized to:
      - actor: MultiHeadActor (configurable heads) instead of MultiDiscreteActor
      - entropy loss: per-head coefficients when cfg.per_head_entropy=True
      - ratio: optional log-ratio clamp
      - returns: optional running-mean-std normalization
    """

    def __init__(
        self,
        *,
        cfg: S2PPOConfigV2,
        env_cfg: EnvConfig,
        physics: DebugPhysicsConfig,
        radar: RadarULAConfig,
        jammer: JammerULAConfig,
        train_seeds: list[int],
        manifest_path: Path,
        out_dir: Path,
        head_specs: Optional[list] = None,  # None -> default S2 two categorical heads
    ):
        from experiments.array_face_s2.learning_repair.actor_heads import HeadSpec, MultiHeadActor
        if env_cfg.profile != cfg.profile:
            raise ValueError(
                f"profile mismatch: env_cfg.profile={env_cfg.profile!r} but cfg.profile={cfg.profile!r}"
            )
        if env_cfg.n_envs != cfg.n_envs or env_cfg.horizon != cfg.horizon:
            raise ValueError("env_cfg.n_envs/horizon must match cfg.n_envs/horizon")
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.physics = physics
        self.radar = radar
        self.jammer = jammer
        self.train_seeds = list(train_seeds)
        self.manifest_path = Path(manifest_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if head_specs is None:
            head_specs = [
                HeadSpec("base", "categorical", N_ACTIONS_BASE),
                HeadSpec("beam", "categorical", N_ACTIONS_BEAM),
            ]
        self.head_specs = tuple(head_specs)
        self.head_names = tuple(s.name for s in self.head_specs)

        torch.manual_seed(cfg.seed)
        self.actor = MultiHeadActor(OBS_DIM_S2, self.head_specs).to(cfg.device)
        self.critic = ValueCritic(OBS_DIM_S2).to(cfg.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        # B1: privileged critic (optional, asymmetric value head)
        self.priv_critic = None
        self.priv_critic_opt = None
        if cfg.use_privileged_critic:
            from env.gpu.array_face_s2 import PRIVILEGED_DIM_S2
            self.priv_critic = PrivilegedValueCritic(PRIVILEGED_DIM_S2).to(cfg.device)
            self.priv_critic_opt = torch.optim.Adam(
                self.priv_critic.parameters(), lr=cfg.critic_lr)

        self.env = ArrayFaceS2VecEnv(
            self.env_cfg, physics=self.physics, radar=self.radar, jammer=self.jammer,
        )
        self._action_gen = torch.Generator(device=cfg.device).manual_seed(cfg.train_seed)

        self._config_sha = cfg.config_sha()
        self._manifest_sha = manifest_sha(self.manifest_path)

        self.iteration = -1
        self.update_count = 0
        self.cumulative_transitions = 0
        self.history: list[dict] = []
        self.kl_rollback_count = 0
        self._return_rms = RunningMeanStd() if cfg.normalize_returns else None
        self._snapshot_actor_state()

    # ---------- actor state snapshot/restore (KL-rollback) ----------
    def _snapshot_actor_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.actor.state_dict().items()}

    def _restore_actor_state(self, snap: dict) -> None:
        self.actor.load_state_dict({k: v.clone() for k, v in snap.items()})

    def _assign_scenarios_and_reset(self):
        sd = self.train_seeds[self.iteration % len(self.train_seeds)]
        self.env.reset(seed=sd)

    # ---------- sampling ----------
    def _sample_actions(self, obs, masks):
        """Delegate to actor_heads.sample_multihead (inverse-CDF, per-head RNG)."""
        from experiments.array_face_s2.learning_repair.actor_heads import sample_multihead
        return sample_multihead(self.actor, obs, masks, self._action_gen)

    # ---------- rollout ----------
    def collect_rollout(self):
        T = self.env_cfg.horizon
        E = self.env_cfg.n_envs
        device = self.cfg.device
        obs_buf = torch.zeros(T, E, OBS_DIM_S2, device=device)
        # masks/actions stored as dict-of-tensors keyed by head name
        mask_bufs = {n: torch.zeros(T, E, s.n_actions, device=device) for n, s in
                     zip(self.head_names, self.head_specs)}
        # action dtype: categorical -> int64, bernoulli -> float32
        act_bufs = {}
        for n, s in zip(self.head_names, self.head_specs):
            if s.kind == "categorical":
                act_bufs[n] = torch.zeros(T, E, dtype=torch.int64, device=device)
            else:
                act_bufs[n] = torch.zeros(T, E, s.n_actions, device=device)
        logp_buf = torch.zeros(T, E, device=device)
        rew_buf = torch.zeros(T, E, device=device)
        val_buf = torch.zeros(T, E, device=device)
        # B1: privileged buffers (only allocated when use_privileged_critic)
        use_priv = self.cfg.use_privileged_critic and self.priv_critic is not None
        if use_priv:
            from env.gpu.array_face_s2 import PRIVILEGED_DIM_S2
            priv_obs_buf = torch.zeros(T, E, PRIVILEGED_DIM_S2, device=device)
            priv_val_buf = torch.zeros(T, E, device=device)
        else:
            priv_obs_buf = None
            priv_val_buf = None

        for t in range(T):
            obs = self.env._build_observation()
            mask_base, mask_beam = self.env._compute_mask()
            masks = {"base": mask_base, "beam": mask_beam}
            with torch.no_grad():
                actions, logp = self._sample_actions(obs, masks)
                value = self.critic(obs)
                if use_priv:
                    priv_obs = self.env.privileged()
                    priv_value = self.priv_critic(priv_obs)
            obs_buf[t] = obs
            mask_bufs["base"][t] = mask_base
            mask_bufs["beam"][t] = mask_beam
            for n in self.head_names:
                act_bufs[n][t] = actions[n]
            logp_buf[t] = logp
            val_buf[t] = value
            if use_priv:
                priv_obs_buf[t] = priv_obs
                priv_val_buf[t] = priv_value
            # env.step currently takes positional (action_base, action_beam);
            # S3 will extend step() to accept a cell action. For S2 we unpack.
            step_out = self.env.step(actions["base"], actions["beam"])
            rew_buf[t] = step_out[1]

        with torch.no_grad():
            last_obs = self.env._build_observation()
            last_value = self.critic(last_obs)
            if use_priv:
                last_priv_obs = self.env.privileged()
                last_priv_value = self.priv_critic(last_priv_obs)
            else:
                last_priv_value = None
        last_done = torch.ones(E, device=device)

        rb = RolloutBuffer(
            obs=obs_buf,
            mask_base=mask_bufs["base"], mask_beam=mask_bufs["beam"],
            action_base=act_bufs["base"], action_beam=act_bufs["beam"],
            logp=logp_buf, reward=rew_buf, value=val_buf,
            last_value=last_value, last_done=last_done,
        )
        # stash privileged buffers on rb for compute_gae / update
        rb.priv_obs = priv_obs_buf
        rb.priv_value = priv_val_buf
        rb.last_priv_value = last_priv_value
        return rb

    # ---------- GAE ----------
    def compute_gae(self, rb):
        # B1: when privileged critic is active, GAE uses the (more accurate)
        # privileged value estimates; otherwise the obs-only value (amend02 path).
        use_priv = getattr(rb, "priv_value", None) is not None
        if use_priv:
            values = rb.priv_value
            last_value = rb.last_priv_value
        else:
            values = rb.value
            last_value = rb.last_value
        T, E = rb.reward.shape
        adv = torch.zeros_like(rb.reward)
        last_gae = torch.zeros(E, device=self.cfg.device)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_done = rb.last_done
            else:
                next_value = values[t + 1]
                next_done = torch.zeros(E, device=self.cfg.device)
            delta = rb.reward[t] + self.cfg.gamma * next_value * (1.0 - next_done) - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - next_done) * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    # ---------- entropy coefficient (per-head aware) ----------
    def _entropy_coef_for_head(self, head_name: str, iteration: int) -> float:
        """Return entropy coef for one head at this iteration.

        If cfg.per_head_entropy and head_name is in the per-head overrides,
        use that head's (coef_init, anneal_frac). Otherwise fall back to global.
        """
        cfg = self.cfg
        if cfg.per_head_entropy and head_name in cfg.entropy_coef_per_head:
            coef_init = cfg.entropy_coef_per_head[head_name]
            anneal_frac = cfg.entropy_anneal_frac_per_head.get(head_name, cfg.entropy_anneal_frac)
            coef_min = cfg.entropy_coef_min
        else:
            coef_init = cfg.entropy_coef_init
            anneal_frac = cfg.entropy_anneal_frac
            coef_min = cfg.entropy_coef_min
        if cfg.iterations <= 0:
            return coef_min
        anneal_iters = max(1, int(anneal_frac * cfg.iterations))
        if iteration >= anneal_iters:
            return coef_min
        frac = float(iteration) / float(anneal_iters)
        return coef_init + frac * (coef_min - coef_init)

    # ---------- PPO update ----------
    def update(self, rb) -> dict:
        from experiments.array_face_s2.learning_repair.actor_heads import joint_kl_multihead

        adv, returns = self.compute_gae(rb)
        # optional return normalization (running mean/std)
        if self._return_rms is not None:
            self._return_rms.update(returns)
            returns = self._return_rms.normalize(returns, self.cfg.return_norm_clip)

        B = rb.obs.shape[0] * rb.obs.shape[1]
        obs_flat = rb.obs.reshape(B, -1)
        logp_old_flat = rb.logp.reshape(B)
        adv_flat = adv.reshape(B)
        ret_flat = returns.reshape(B)
        # Fully generic per-head flatten: every head's mask/action is read by
        # name (rb.mask_<name> / rb.action_<name>). RolloutBuffer's base/beam
        # dataclass fields serve S2/S3 (their head names match); S4's cell head
        # (3-D) is stashed as rb.mask_cell/rb.action_cell and read here without
        # the 2-D reshape assumption the old hardcoded path made.
        masks_flat: dict[str, torch.Tensor] = {}
        actions_flat: dict[str, torch.Tensor] = {}
        for name, spec in zip(self.head_names, self.head_specs):
            mask_attr = "mask_" + name
            act_attr = "action_" + name
            if not (hasattr(rb, mask_attr) and hasattr(rb, act_attr)):
                raise AttributeError(
                    f"rollout buffer missing {mask_attr}/{act_attr} for head {name!r}"
                )
            masks_flat[name] = getattr(rb, mask_attr).reshape(B, -1)
            act_flat_tensor = getattr(rb, act_attr)
            if spec.kind == "categorical":
                # stored [T, E] int64 -> [B]
                actions_flat[name] = act_flat_tensor.reshape(B)
            else:
                # bernoulli stored [T, E, n_cells] -> [B, n_cells]
                actions_flat[name] = act_flat_tensor.reshape(B, -1)

        # B1: flatten privileged buffers if present
        use_priv_flat = None
        if getattr(rb, "priv_obs", None) is not None:
            use_priv_flat = rb.priv_obs.reshape(B, -1)

        adv_std = float(adv_flat.std().item())
        if adv_std > 1e-8:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        with torch.no_grad():
            logp_check = self.actor.joint_log_prob(obs_flat, masks_flat, actions_flat)
            if self.cfg.log_ratio_clamp > 0:
                log_ratio_check = (logp_check - logp_old_flat).clamp(
                    -self.cfg.log_ratio_clamp, self.cfg.log_ratio_clamp)
            else:
                log_ratio_check = logp_check - logp_old_flat
            ratio_check = log_ratio_check.exp()
            pre_ratio_offset = float((ratio_check - 1.0).abs().max().item())

        # per-head entropy coefficients at current iteration
        ent_coefs = {n: self._entropy_coef_for_head(n, self.iteration) for n in self.head_names}

        actor_pre_update = self._snapshot_actor_state()
        metrics_agg = {
            "adv_std": adv_std, "pre_ratio_offset": pre_ratio_offset,
            "entropy_coef": ent_coefs.get("base", self.cfg.entropy_coef_init),
            "kl_post_minibatch": [], "clip_frac_list": [],
            "policy_loss_list": [], "value_loss_list": [],
            "entropy_list": [], "actor_grad_norm_list": [],
            "explained_variance": 0.0,
            # per-head entropy lists (generic: base, beam, cell, ...)
            **{f"entropy_{n}_list": [] for n in self.head_names},
            "action_base_freq": [], "action_beam_freq": [],
            "log_ratio_clamped_count": 0,
            "priv_value_loss_list": [], "distill_loss_list": [],
        }

        rolled_back = False
        outer_kl_max = 0.0
        for epoch in range(self.cfg.epochs_per_iteration):
            perm = torch.randperm(B, device=self.cfg.device)
            for s in range(0, B, self.cfg.minibatch_size):
                bi = perm[s:s + self.cfg.minibatch_size]
                obs_b = obs_flat[bi]
                lpo_b = logp_old_flat[bi]
                adv_b = adv_flat[bi]
                ret_b = ret_flat[bi]
                # Build per-minibatch masks/actions dicts generically from the
                # flattened dicts (which include any extra heads beyond base/beam).
                masks_b = {name: tens[bi] for name, tens in masks_flat.items()}
                actions_b = {name: tens[bi] for name, tens in actions_flat.items()}
                # Keep mb_b / mm_b / ab_b / am_b available for any downstream
                # code that references them by the old names (e.g. privileged path).
                mb_b = masks_b.get("base")
                mm_b = masks_b.get("beam")
                ab_b = actions_b.get("base")
                am_b = actions_b.get("beam")

                with torch.no_grad():
                    logits_old_pre = self.actor.forward(obs_b)

                logp_new = self.actor.joint_log_prob(obs_b, masks_b, actions_b)
                if self.cfg.log_ratio_clamp > 0:
                    log_ratio = (logp_new - lpo_b).clamp(
                        -self.cfg.log_ratio_clamp, self.cfg.log_ratio_clamp)
                    metrics_agg["log_ratio_clamped_count"] += int(
                        ((logp_new - lpo_b).abs() > self.cfg.log_ratio_clamp).sum().item())
                else:
                    log_ratio = logp_new - lpo_b
                ratio = log_ratio.exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                ent_dict = self.actor.joint_entropy(obs_b, masks_b)
                # per-head weighted entropy loss
                ent_loss = sum(ent_coefs[n] * ent_dict[n].mean() for n in self.head_names)
                entropy_val = ent_dict["_sum"].mean()  # for logging

                self.actor_opt.zero_grad()
                (policy_loss - ent_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
                self.actor_opt.step()

                value_pred = self.critic(obs_b)
                value_loss = F.mse_loss(value_pred, ret_b)
                # B1: privileged critic + distillation. When enabled:
                #   - priv_critic is trained against returns (same target)
                #   - obs critic gets an extra distill term pulling it toward
                #     priv_critic's prediction (transfers privileged knowledge
                #     into the deployable obs-only head)
                if use_priv_flat is not None:
                    priv_pred = self.priv_critic(use_priv_flat[bi])
                    priv_value_loss = F.mse_loss(priv_pred, ret_b)
                    distill_loss = F.mse_loss(value_pred, priv_pred.detach())
                    total_value_loss = (self.cfg.value_coef * value_loss
                                        + self.cfg.privileged_value_coef * priv_value_loss
                                        + self.cfg.distill_coef * distill_loss)
                    self.priv_critic_opt.zero_grad()
                    self.critic_opt.zero_grad()
                    total_value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.priv_critic.parameters(), self.cfg.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
                    self.priv_critic_opt.step()
                    self.critic_opt.step()
                    metrics_agg["priv_value_loss_list"].append(float(priv_value_loss.item()))
                    metrics_agg["distill_loss_list"].append(float(distill_loss.item()))
                else:
                    self.critic_opt.zero_grad()
                    (self.cfg.value_coef * value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
                    self.critic_opt.step()

                with torch.no_grad():
                    kl_post = joint_kl_multihead(
                        self.actor, obs_b, masks_b, logits_old_pre,
                    ).mean().item()
                    clip_frac = float(((ratio - 1.0).abs() > self.cfg.clip).float().mean().item())
                    actor_gn = float(sum(
                        p.grad.norm().item() if p.grad is not None else 0.0
                        for p in self.actor.parameters()
                    ))
                metrics_agg["kl_post_minibatch"].append(kl_post)
                metrics_agg["clip_frac_list"].append(clip_frac)
                metrics_agg["policy_loss_list"].append(float(policy_loss.item()))
                metrics_agg["value_loss_list"].append(float(value_loss.item()))
                metrics_agg["entropy_list"].append(float(entropy_val.item()))
                # per-head entropy (generic: base, beam, cell, ...)
                for n in self.head_names:
                    if n in ent_dict:
                        metrics_agg[f"entropy_{n}_list"].append(float(ent_dict[n].mean().item()))
                metrics_agg["actor_grad_norm_list"].append(actor_gn)

                outer_kl_max = max(outer_kl_max, kl_post)
                if kl_post > self.cfg.target_kl:
                    self._restore_actor_state(actor_pre_update)
                    rolled_back = True
                    break
            if rolled_back:
                break

        with torch.no_grad():
            v_pred = self.critic(obs_flat)
            var_y = ret_flat.var(unbiased=False)
            var_res = (ret_flat - v_pred).var(unbiased=False)
            ev = float((1.0 - var_res / (var_y + 1e-8)).item())
        metrics_agg["explained_variance"] = ev

        with torch.no_grad():
            # Generic per-head action frequency: categorical -> per-action
            # counts; bernoulli -> per-cell on-rate. Keys: action_<name>_freq.
            for n, spec in zip(self.head_names, self.head_specs):
                act = getattr(rb, "action_" + n)
                if spec.kind == "categorical":
                    freq = [float((act == a).float().mean().item())
                            for a in range(spec.n_actions)]
                else:
                    freq = [float(act[:, :, c].float().mean().item())
                            for c in range(spec.n_actions)]
                metrics_agg[f"action_{n}_freq"] = freq

        def mean(xs): return sum(xs) / len(xs) if xs else 0.0
        return {
            "iteration": self.iteration,
            "update_count": self.update_count,
            "adv_std": adv_std,
            "pre_ratio_offset": pre_ratio_offset,
            "entropy_coef": ent_coefs,
            "kl_mean_post": mean(metrics_agg["kl_post_minibatch"]),
            "kl_max_post": max(metrics_agg["kl_post_minibatch"]) if metrics_agg["kl_post_minibatch"] else 0.0,
            "clip_frac_mean": mean(metrics_agg["clip_frac_list"]),
            "policy_loss": mean(metrics_agg["policy_loss_list"]),
            "value_loss": mean(metrics_agg["value_loss_list"]),
            "entropy": mean(metrics_agg["entropy_list"]),
            # per-head entropy (generic; keys: entropy_base, entropy_beam, entropy_cell, ...)
            **{f"entropy_{n}": mean(metrics_agg.get(f"entropy_{n}_list", [0.0]))
               for n in self.head_names},
            "actor_grad_norm": mean(metrics_agg["actor_grad_norm_list"]),
            "explained_variance": metrics_agg["explained_variance"],
            "kl_rollback": rolled_back,
            "outer_kl_max": outer_kl_max,
            # generic per-head action frequency (keys: action_<name>_freq)
            **{f"action_{n}_freq": metrics_agg[f"action_{n}_freq"] for n in self.head_names},
            "log_ratio_clamped_count": metrics_agg["log_ratio_clamped_count"],
            "priv_value_loss": mean(metrics_agg["priv_value_loss_list"]),
            "distill_loss": mean(metrics_agg["distill_loss_list"]),
        }

    def train_iteration(self) -> dict:
        if self.iteration < 0:
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
        metrics["rollout_drop"] = drops
        metrics["cumulative_transitions"] = self.cumulative_transitions
        metrics["iteration"] = self.iteration
        self.history.append(metrics)
        return metrics

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

    def save_periodic(self, iteration: int, optimizer_state: bool = True) -> Path:
        """Full-state checkpoint for resume-after-interrupt.

        Saves actor/critic weights + optimizer state + iteration/update_count/
        cumulative_transitions/kl_rollback_count + RNG generator state, so a
        resumed run reproduces the trajectory exactly. Written to
        checkpoint_iter{N}.pt; also updates checkpoint_latest.pt symlink-target
        (a copy, not a real symlink, for Windows portability).
        """
        out_path = self.out_dir / f"checkpoint_iter{iteration}.pt"
        payload = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "iteration": self.iteration,
            "update_count": self.update_count,
            "cumulative_transitions": self.cumulative_transitions,
            "kl_rollback_count": self.kl_rollback_count,
            "action_gen_state": self._action_gen.get_state(),
        }
        if optimizer_state:
            payload["actor_opt_state"] = self.actor_opt.state_dict()
            payload["critic_opt_state"] = self.critic_opt.state_dict()
        if self.priv_critic is not None:
            payload["priv_critic_state_dict"] = self.priv_critic.state_dict()
            if optimizer_state and self.priv_critic_opt is not None:
                payload["priv_critic_opt_state"] = self.priv_critic_opt.state_dict()
        torch.save(payload, out_path)
        # also write a stable "latest" pointer copy
        latest = self.out_dir / "checkpoint_latest.pt"
        torch.save(payload, latest)
        return out_path

    def load_checkpoint(self, path: Path) -> int:
        """Restore full training state from a save_periodic checkpoint.

        Returns the iteration index that was restored (the driver resumes from
        iteration+1). Verifies config_sha matches to refuse mismatched resumes.
        """
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        # config consistency check (config_sha covers all hyperparams)
        self.iteration = int(ckpt["iteration"])
        self.update_count = int(ckpt["update_count"])
        self.cumulative_transitions = int(ckpt["cumulative_transitions"])
        self.kl_rollback_count = int(ckpt["kl_rollback_count"])
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        if "actor_opt_state" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt_state"])
            self.critic_opt.load_state_dict(ckpt["critic_opt_state"])
        if "priv_critic_state_dict" in ckpt and self.priv_critic is not None:
            self.priv_critic.load_state_dict(ckpt["priv_critic_state_dict"])
            if "priv_critic_opt_state" in ckpt and self.priv_critic_opt is not None:
                self.priv_critic_opt.load_state_dict(ckpt["priv_critic_opt_state"])
        if "action_gen_state" in ckpt:
            gen_state = ckpt["action_gen_state"]
            # torch Generator state is always a 1-D uint8 ByteTensor on CPU,
            # regardless of the generator's device. Ensure it's CPU uint8.
            if not isinstance(gen_state, torch.Tensor):
                gen_state = torch.tensor(gen_state, dtype=torch.uint8)
            gen_state = gen_state.cpu().to(torch.uint8)
            self._action_gen.set_state(gen_state)
        return self.iteration

    def save_pristine_init(self) -> Path:
        out_path = self.out_dir / "pristine_init.pt"
        self.iteration = -1
        self.save_checkpoint(origin="pristine_init", out_path=out_path)
        return out_path

    def save_last_iter(self, iter_label: int) -> Path:
        out_path = self.out_dir / f"last_iter{iter_label}.pt"
        self.save_checkpoint(origin="last_iter", out_path=out_path)
        return out_path


def evaluate_actor_v2(
    actor, *,
    env_cfg: EnvConfig,
    physics: DebugPhysicsConfig,
    radar: RadarULAConfig,
    jammer: JammerULAConfig,
    scenario_seeds: list[int],
    n_action_reps: int = 4,
    sample: bool = True,
    device: str = "cpu",
    action_seed: int = 0,
) -> dict:
    """Per-scenario macro drop_ratio evaluation (S2 two-categorical-head path).

    Mirrors trainer.evaluate_actor but uses MultiHeadActor's dict interface.
    """
    from experiments.array_face_s2.learning_repair.actor_heads import sample_multihead
    actor.eval()
    gen = torch.Generator(device=device).manual_seed(action_seed)
    per_seed_drops: list[float] = []
    raw_rows: list[dict] = []
    for sd in scenario_seeds:
        rep_drops: list[float] = []
        for rep in range(n_action_reps):
            env = ArrayFaceS2VecEnv(env_cfg, physics=physics, radar=radar, jammer=jammer)
            env.reset(seed=sd)
            for t in range(env_cfg.horizon):
                obs = env._build_observation()
                mask_base, mask_beam = env._compute_mask()
                masks = {"base": mask_base, "beam": mask_beam}
                with torch.no_grad():
                    if sample:
                        actions, _ = sample_multihead(actor, obs, masks, gen)
                    else:
                        logits = actor.forward(obs)
                        action_base = logits["base"].masked_fill(
                            ~mask_base.bool(), float("-inf")).argmax(dim=-1)
                        action_beam = logits["beam"].masked_fill(
                            ~mask_beam.bool(), float("-inf")).argmax(dim=-1)
                        actions = {"base": action_base, "beam": action_beam}
                env.step(actions["base"], actions["beam"])
            rep_drop = float(env.drop_ratio()[0])
            rep_drops.append(rep_drop)
            raw_rows.append({
                "seed": int(sd), "rep": int(rep), "drop_ratio": rep_drop,
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
