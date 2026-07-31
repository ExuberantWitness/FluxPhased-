"""S2 PPO trainer — MultiDiscrete([3, 5]) PPO on ArrayFaceS2VecEnv.

Adapts S1 trainer to the MultiDiscrete action. The actor has two heads
(Categorical(3) base + Categorical(5) beam) sharing a trunk. Joint log_prob
and entropy are sums of the per-head values. PPO ratio uses the joint logp.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.array_face_s2 import (
    EnvConfig, ArrayFaceS2VecEnv, RadarULAConfig, JammerULAConfig, OBS_DIM_S2,
    N_ACTIONS_BASE, N_ACTIONS_BEAM,
)


@dataclass
class S2PPOConfig:
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
    entropy_coef_init: float = 5e-3
    entropy_coef_min: float = 0.0
    entropy_anneal_frac: float = 0.5
    value_coef: float = 0.5
    target_kl: float = 0.02
    seed: int = 0
    device: str = "cpu"
    train_seed: int = 0

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


class MultiDiscreteActor(nn.Module):
    """Two heads: Categorical(N_ACTIONS_BASE=3) base + Categorical(N_ACTIONS_BEAM=5) beam.
    Shared trunk. logp = logp_base + logp_beam. Entropy = entropy_base + entropy_beam.
    """
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head_base = nn.Linear(hidden, N_ACTIONS_BASE)
        self.head_beam = nn.Linear(hidden, N_ACTIONS_BEAM)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        return self.head_base(h), self.head_beam(h)

    def distribution(
        self, obs: torch.Tensor,
        mask_base: torch.Tensor, mask_beam: torch.Tensor,
    ) -> tuple[torch.distributions.Categorical, torch.distributions.Categorical]:
        logits_base, logits_beam = self.forward(obs)
        logits_base = logits_base.masked_fill(~mask_base.bool(), float("-inf"))
        logits_beam = logits_beam.masked_fill(~mask_beam.bool(), float("-inf"))
        return (
            torch.distributions.Categorical(logits=logits_base),
            torch.distributions.Categorical(logits=logits_beam),
        )

    def joint_log_prob(
        self, obs: torch.Tensor,
        mask_base: torch.Tensor, mask_beam: torch.Tensor,
        action_base: torch.Tensor, action_beam: torch.Tensor,
    ) -> torch.Tensor:
        d_base, d_beam = self.distribution(obs, mask_base, mask_beam)
        return d_base.log_prob(action_base) + d_beam.log_prob(action_beam)

    def joint_entropy(
        self, obs: torch.Tensor,
        mask_base: torch.Tensor, mask_beam: torch.Tensor,
    ) -> torch.Tensor:
        d_base, d_beam = self.distribution(obs, mask_base, mask_beam)
        return d_base.entropy() + d_beam.entropy()


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


@dataclass
class RolloutBuffer:
    obs: torch.Tensor
    mask_base: torch.Tensor
    mask_beam: torch.Tensor
    action_base: torch.Tensor
    action_beam: torch.Tensor
    logp: torch.Tensor          # joint
    reward: torch.Tensor
    value: torch.Tensor
    last_value: torch.Tensor
    last_done: torch.Tensor


def categorical_kl(
    logits_old: torch.Tensor, logits_new: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    lo = logits_old.masked_fill(~mask.bool(), float("-inf"))
    ln = logits_new.masked_fill(~mask.bool(), float("-inf"))
    log_po = F.log_softmax(lo, dim=-1)
    log_pn = F.log_softmax(ln, dim=-1)
    po = log_po.exp()
    safe = po > 0
    contrib = torch.where(safe, po * (log_po - log_pn), torch.zeros_like(po))
    return contrib.sum(dim=-1)


def joint_kl(
    actor: MultiDiscreteActor, obs: torch.Tensor,
    mask_base: torch.Tensor, mask_beam: torch.Tensor,
    logits_base_old: torch.Tensor, logits_beam_old: torch.Tensor,
) -> torch.Tensor:
    """KL(old || new) for the joint MultiDiscrete distribution. Independence → sum of per-head KL."""
    logits_base_new, logits_beam_new = actor.forward(obs)
    kl_base = categorical_kl(logits_base_old, logits_base_new, mask_base)
    kl_beam = categorical_kl(logits_beam_old, logits_beam_new, mask_beam)
    return kl_base + kl_beam


@dataclass
class CheckpointMeta:
    iteration: int
    update_count: int
    cumulative_transitions: int
    checkpoint_origin: str
    training_seed: int
    config_sha: str
    manifest_sha: str
    profile: str
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


class S2PPOTrainer:
    def __init__(
        self,
        *,
        cfg: S2PPOConfig,
        env_cfg: EnvConfig,
        physics: DebugPhysicsConfig,
        radar: RadarULAConfig,
        jammer: JammerULAConfig,
        train_seeds: list[int],
        manifest_path: Path,
        out_dir: Path,
    ):
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

        torch.manual_seed(cfg.seed)
        self.actor = MultiDiscreteActor(OBS_DIM_S2).to(cfg.device)
        self.critic = ValueCritic(OBS_DIM_S2).to(cfg.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

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
        self._snapshot_actor_state()

    def _snapshot_actor_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.actor.state_dict().items()}

    def _restore_actor_state(self, snap: dict) -> None:
        self.actor.load_state_dict({k: v.clone() for k, v in snap.items()})

    def _assign_scenarios_and_reset(self):
        sd = self.train_seeds[self.iteration % len(self.train_seeds)]
        self.env.reset(seed=sd)

    def _sample_actions(self, obs, mask_base, mask_beam):
        """Inverse-CDF sampling for joint MultiDiscrete using single uniform per head."""
        d_base, d_beam = self.actor.distribution(obs, mask_base, mask_beam)
        E = obs.shape[0]
        device = obs.device
        # One uniform per head per env (deterministic via _action_gen)
        u_base = torch.rand(E, generator=self._action_gen, device=device)
        u_beam = torch.rand(E, generator=self._action_gen, device=device)
        probs_base = d_base.probs.clamp(min=1e-12)
        probs_beam = d_beam.probs.clamp(min=1e-12)
        cdf_base = torch.cumsum(probs_base, dim=-1)
        cdf_beam = torch.cumsum(probs_beam, dim=-1)
        action_base = (u_base.unsqueeze(-1) < cdf_base).float().argmax(dim=-1)
        action_beam = (u_beam.unsqueeze(-1) < cdf_beam).float().argmax(dim=-1)
        action_base = action_base.long()
        action_beam = action_beam.long()
        logp = d_base.log_prob(action_base) + d_beam.log_prob(action_beam)
        return action_base, action_beam, logp

    def collect_rollout(self) -> RolloutBuffer:
        T = self.env_cfg.horizon
        E = self.env_cfg.n_envs
        device = self.cfg.device
        obs_buf = torch.zeros(T, E, OBS_DIM_S2, device=device)
        mask_base_buf = torch.zeros(T, E, N_ACTIONS_BASE, device=device)
        mask_beam_buf = torch.zeros(T, E, N_ACTIONS_BEAM, device=device)
        act_base_buf = torch.zeros(T, E, dtype=torch.int64, device=device)
        act_beam_buf = torch.zeros(T, E, dtype=torch.int64, device=device)
        logp_buf = torch.zeros(T, E, device=device)
        rew_buf = torch.zeros(T, E, device=device)
        val_buf = torch.zeros(T, E, device=device)

        for t in range(T):
            obs = self.env._build_observation()
            mask_base, mask_beam = self.env._compute_mask()
            with torch.no_grad():
                action_base, action_beam, logp = self._sample_actions(obs, mask_base, mask_beam)
                value = self.critic(obs)
            obs_buf[t] = obs
            mask_base_buf[t] = mask_base
            mask_beam_buf[t] = mask_beam
            act_base_buf[t] = action_base
            act_beam_buf[t] = action_beam
            logp_buf[t] = logp
            val_buf[t] = value
            step_out = self.env.step(action_base, action_beam)
            rew_buf[t] = step_out[1]

        with torch.no_grad():
            last_obs = self.env._build_observation()
            last_value = self.critic(last_obs)
        last_done = torch.ones(E, device=device)

        return RolloutBuffer(
            obs=obs_buf, mask_base=mask_base_buf, mask_beam=mask_beam_buf,
            action_base=act_base_buf, action_beam=act_beam_buf,
            logp=logp_buf, reward=rew_buf, value=val_buf,
            last_value=last_value, last_done=last_done,
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
            delta = rb.reward[t] + self.cfg.gamma * next_value * (1.0 - next_done) - rb.value[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - next_done) * last_gae
            adv[t] = last_gae
        returns = adv + rb.value
        return adv, returns

    def _entropy_coef(self, iteration: int) -> float:
        if self.cfg.iterations <= 0:
            return self.cfg.entropy_coef_min
        anneal_iters = max(1, int(self.cfg.entropy_anneal_frac * self.cfg.iterations))
        if iteration >= anneal_iters:
            return self.cfg.entropy_coef_min
        frac = float(iteration) / float(anneal_iters)
        return self.cfg.entropy_coef_init + frac * (self.cfg.entropy_coef_min - self.cfg.entropy_coef_init)

    def update(self, rb: RolloutBuffer) -> dict:
        adv, returns = self.compute_gae(rb)
        B = rb.obs.shape[0] * rb.obs.shape[1]
        obs_flat = rb.obs.reshape(B, -1)
        mask_base_flat = rb.mask_base.reshape(B, -1)
        mask_beam_flat = rb.mask_beam.reshape(B, -1)
        act_base_flat = rb.action_base.reshape(B)
        act_beam_flat = rb.action_beam.reshape(B)
        logp_old_flat = rb.logp.reshape(B)
        adv_flat = adv.reshape(B)
        ret_flat = returns.reshape(B)

        adv_std = float(adv_flat.std().item())
        if adv_std > 1e-8:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        with torch.no_grad():
            logp_check = self.actor.joint_log_prob(
                obs_flat, mask_base_flat, mask_beam_flat, act_base_flat, act_beam_flat,
            )
            ratio_check = (logp_check - logp_old_flat).exp()
            pre_ratio_offset = float((ratio_check - 1.0).abs().max().item())

        entropy_coef = self._entropy_coef(self.iteration)
        actor_pre_update = self._snapshot_actor_state()
        metrics_agg = {
            "adv_std": adv_std, "pre_ratio_offset": pre_ratio_offset,
            "entropy_coef": entropy_coef,
            "kl_post_minibatch": [], "clip_frac_list": [],
            "policy_loss_list": [], "value_loss_list": [],
            "entropy_list": [], "actor_grad_norm_list": [],
            "explained_variance": 0.0,
            "entropy_base_list": [], "entropy_beam_list": [],
            "action_base_freq": [], "action_beam_freq": [],
        }

        rolled_back = False
        outer_kl_max = 0.0
        for epoch in range(self.cfg.epochs_per_iteration):
            perm = torch.randperm(B, device=self.cfg.device)
            for s in range(0, B, self.cfg.minibatch_size):
                bi = perm[s:s + self.cfg.minibatch_size]
                obs_b = obs_flat[bi]
                mb_b = mask_base_flat[bi]
                mm_b = mask_beam_flat[bi]
                ab_b = act_base_flat[bi]
                am_b = act_beam_flat[bi]
                lpo_b = logp_old_flat[bi]
                adv_b = adv_flat[bi]
                ret_b = ret_flat[bi]

                with torch.no_grad():
                    logits_base_old_pre, logits_beam_old_pre = self.actor.forward(obs_b)

                d_base, d_beam = self.actor.distribution(obs_b, mb_b, mm_b)
                logp_new = d_base.log_prob(ab_b) + d_beam.log_prob(am_b)
                ratio = (logp_new - lpo_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = d_base.entropy().mean() + d_beam.entropy().mean()
                entropy_base = d_base.entropy().mean()
                entropy_beam = d_beam.entropy().mean()
                self.actor_opt.zero_grad()
                (policy_loss - entropy_coef * entropy).backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
                self.actor_opt.step()

                value_pred = self.critic(obs_b)
                value_loss = F.mse_loss(value_pred, ret_b)
                self.critic_opt.zero_grad()
                (self.cfg.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
                self.critic_opt.step()

                with torch.no_grad():
                    kl_post = joint_kl(
                        self.actor, obs_b, mb_b, mm_b,
                        logits_base_old_pre, logits_beam_old_pre,
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
                metrics_agg["entropy_list"].append(float(entropy.item()))
                metrics_agg["entropy_base_list"].append(float(entropy_base.item()))
                metrics_agg["entropy_beam_list"].append(float(entropy_beam.item()))
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

        # Action frequency diagnostics
        with torch.no_grad():
            base_freq = torch.zeros(N_ACTIONS_BASE, device=self.cfg.device)
            beam_freq = torch.zeros(N_ACTIONS_BEAM, device=self.cfg.device)
            for a in range(N_ACTIONS_BASE):
                base_freq[a] = (rb.action_base == a).float().mean().item()
            for a in range(N_ACTIONS_BEAM):
                beam_freq[a] = (rb.action_beam == a).float().mean().item()
            metrics_agg["action_base_freq"] = base_freq.tolist()
            metrics_agg["action_beam_freq"] = beam_freq.tolist()

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
            "entropy_base": mean(metrics_agg["entropy_base_list"]),
            "entropy_beam": mean(metrics_agg["entropy_beam_list"]),
            "actor_grad_norm": mean(metrics_agg["actor_grad_norm_list"]),
            "explained_variance": metrics_agg["explained_variance"],
            "kl_rollback": rolled_back,
            "outer_kl_max": outer_kl_max,
            "action_base_freq": metrics_agg["action_base_freq"],
            "action_beam_freq": metrics_agg["action_beam_freq"],
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

    def save_pristine_init(self) -> Path:
        out_path = self.out_dir / "pristine_init.pt"
        self.iteration = -1
        self.save_checkpoint(origin="pristine_init", out_path=out_path)
        return out_path

    def save_last_iter(self, iter_label: int) -> Path:
        out_path = self.out_dir / f"last_iter{iter_label}.pt"
        self.save_checkpoint(origin="last_iter", out_path=out_path)
        return out_path


def evaluate_actor(
    actor: MultiDiscreteActor,
    *,
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
    """Per-scenario macro drop_ratio evaluation. Each scenario is run n_action_reps times."""
    actor.eval()
    gen = torch.Generator(device=device).manual_seed(action_seed)
    per_seed_drops: list[float] = []
    per_seed_n_eligible: list[int] = []
    raw_rows: list[dict] = []
    for sd in scenario_seeds:
        rep_drops: list[float] = []
        for rep in range(n_action_reps):
            env = ArrayFaceS2VecEnv(env_cfg, physics=physics, radar=radar, jammer=jammer)
            env.reset(seed=sd)
            for t in range(env_cfg.horizon):
                obs = env._build_observation()
                mask_base, mask_beam = env._compute_mask()
                with torch.no_grad():
                    if sample:
                        d_base, d_beam = actor.distribution(obs, mask_base, mask_beam)
                        u_base = torch.rand(1, generator=gen, device=device)
                        u_beam = torch.rand(1, generator=gen, device=device)
                        pb = d_base.probs.clamp(min=1e-12)
                        pm = d_beam.probs.clamp(min=1e-12)
                        action_base = (u_base.unsqueeze(-1) < torch.cumsum(pb, dim=-1)).float().argmax(dim=-1).long()
                        action_beam = (u_beam.unsqueeze(-1) < torch.cumsum(pm, dim=-1)).float().argmax(dim=-1).long()
                    else:
                        logits_base, logits_beam = actor.forward(obs)
                        mb = logits_base.masked_fill(~mask_base.bool(), float("-inf"))
                        mm = logits_beam.masked_fill(~mask_beam.bool(), float("-inf"))
                        action_base = mb.argmax(dim=-1)
                        action_beam = mm.argmax(dim=-1)
                env.step(action_base, action_beam)
            rep_drop = float(env.drop_ratio()[0])
            rep_drops.append(rep_drop)
            n_eligible = int(env.counters.n_eligible[0].item())
            per_seed_n_eligible.append(n_eligible)
            raw_rows.append({
                "seed": int(sd), "rep": int(rep),
                "drop_ratio": rep_drop, "n_eligible": n_eligible,
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
