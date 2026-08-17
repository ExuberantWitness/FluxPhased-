"""S5 IPPO trainer — parameter-shared actor + central (privileged) critic.

Design (HANDOFF §11.3, IPPO-not-MAPPO):
  - ONE MultiHeadActor (cell Bernoulli(25) + beam Categorical(25)) shared by
    both jammers (parameter sharing); each jammer acts on its OWN observation.
  - Central critic = PrivilegedValueCritic on PRIVILEGED_DIM_S5 (84-dim:
    both jammers' beams/cells + pending/health + radar state) — CTDE.
  - K-flattening trick: the rollout batch is [T, E*K] with each (env, jammer)
    slot as an independent IPPO "trajectory" sharing the TEAM reward. The
    entire S2PPOTrainerV2.update() (GAE on central values, clipped surrogate,
    KL rollback, per-head entropy) is inherited UNCHANGED — S5 only overrides
    __init__ (env/actor/critic construction) and collect_rollout.
"""
from __future__ import annotations
from pathlib import Path

import torch

from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.array_face_s5 import (
    EnvConfig, ArrayFaceS5VecEnv, UPAConfig,
    OBS_DIM_S5, PRIVILEGED_DIM_S5, N_JAMMERS,
)
from experiments.array_face_s2.learning_repair.trainer_v2 import (
    S2PPOConfigV2, S2PPOTrainerV2,
)
from experiments.array_face_s2.learning_repair.actor_heads import sample_multihead


class S5IPPOTrainer(S2PPOTrainerV2):
    """S5 trainer: parameter-shared two-jammer IPPO with central critic."""

    def __init__(
        self,
        *,
        cfg: S2PPOConfigV2,
        env_cfg: EnvConfig,
        physics: DebugPhysicsConfig,
        radar: UPAConfig,
        jammer: UPAConfig,
        train_seeds: list[int],
        manifest_path: Path,
        out_dir: Path,
        head_specs: list,
    ):
        from experiments.array_face_s2.learning_repair.actor_heads import MultiHeadActor
        from experiments.array_face_s2.learning_repair.trainer import (
            ValueCritic, manifest_sha,
        )
        from experiments.array_face_s2.learning_repair.trainer_v2 import PrivilegedValueCritic
        import torch as _t

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

        self.head_specs = tuple(head_specs)
        self.head_names = tuple(s.name for s in self.head_specs)

        _t.manual_seed(cfg.seed)
        # ONE parameter-shared actor for both jammers (IPPO with sharing)
        self.actor = MultiHeadActor(OBS_DIM_S5, self.head_specs).to(cfg.device)
        self.critic = ValueCritic(OBS_DIM_S5).to(cfg.device)
        self.actor_opt = _t.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = _t.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # CENTRAL critic (CTDE): privileged state of BOTH jammers. Required
        # for S5 — the whole point of the cooperative stage.
        if not cfg.use_privileged_critic:
            raise ValueError("S5 requires use_privileged_critic=True (central critic)")
        self.priv_critic = PrivilegedValueCritic(PRIVILEGED_DIM_S5).to(cfg.device)
        self.priv_critic_opt = _t.optim.Adam(self.priv_critic.parameters(), lr=cfg.critic_lr)

        self.env = ArrayFaceS5VecEnv(
            self.env_cfg, physics=self.physics, radar=self.radar, jammer=self.jammer,
        )
        self._action_gen = _t.Generator(device=cfg.device).manual_seed(cfg.train_seed)

        self._config_sha = cfg.config_sha()
        self._manifest_sha = manifest_sha(self.manifest_path)

        self.iteration = -1
        self.update_count = 0
        self.cumulative_transitions = 0
        self.history: list[dict] = []
        self.kl_rollback_count = 0
        from experiments.array_face_s2.learning_repair.trainer_v2 import RunningMeanStd
        self._return_rms = RunningMeanStd() if cfg.normalize_returns else None
        self._snapshot_actor_state()

    def collect_rollout(self):
        """K-flattened rollout: [T, B=E*K] with k-major slots (b = k*E + e).

        Each slot is one jammer's independent trajectory; the team reward and
        the central critic's value are shared (duplicated) across the K slots
        of an env. The inherited update() sees a plain [T, B] batch.
        """
        T = self.env_cfg.horizon
        E = self.env_cfg.n_envs
        K = N_JAMMERS
        B = E * K
        device = self.cfg.device
        obs_buf = torch.zeros(T, B, OBS_DIM_S5, device=device)
        mask_bufs = {n: torch.zeros(T, B, s.n_actions, device=device)
                     for n, s in zip(self.head_names, self.head_specs)}
        act_bufs = {}
        for n, s in zip(self.head_names, self.head_specs):
            if s.kind == "categorical":
                act_bufs[n] = torch.zeros(T, B, dtype=torch.int64, device=device)
            else:
                act_bufs[n] = torch.zeros(T, B, s.n_actions, device=device)
        logp_buf = torch.zeros(T, B, device=device)
        rew_buf = torch.zeros(T, B, device=device)
        val_buf = torch.zeros(T, B, device=device)
        priv_obs_buf = torch.zeros(T, B, PRIVILEGED_DIM_S5, device=device)
        priv_val_buf = torch.zeros(T, B, device=device)

        for t in range(T):
            obs = self.env._build_observation()          # [E, K, OBS]
            mask_cell, mask_beam = self.env._compute_mask()  # [E, K, 25]
            priv_obs = self.env.privileged()             # [E, PRIV]
            with torch.no_grad():
                priv_value = self.priv_critic(priv_obs)  # [E]

            step_actions = {}
            for k in range(K):
                obs_k = obs[:, k]                        # [E, OBS]
                masks_k = {"cell": mask_cell[:, k], "beam": mask_beam[:, k]}
                with torch.no_grad():
                    actions_k, logp_k = self._sample_actions(obs_k, masks_k)
                    value_k = self.critic(obs_k)
                sl = slice(k * E, (k + 1) * E)           # k-major slot range
                obs_buf[t, sl] = obs_k
                mask_bufs["cell"][t, sl] = mask_cell[:, k]
                mask_bufs["beam"][t, sl] = mask_beam[:, k]
                for n in self.head_names:
                    act_bufs[n][t, sl] = actions_k[n]
                logp_buf[t, sl] = logp_k
                val_buf[t, sl] = value_k
                priv_obs_buf[t, sl] = priv_obs
                priv_val_buf[t, sl] = priv_value
                step_actions[k] = actions_k

            cell = torch.stack([step_actions[k]["cell"] for k in range(K)], dim=1)  # [E,K,25]
            beam = torch.stack([step_actions[k]["beam"] for k in range(K)], dim=1)  # [E,K]
            step_out = self.env.step(cell, beam)
            rew_buf[t] = step_out[1].repeat(K)  # team reward, shared

        with torch.no_grad():
            last_obs = self.env._build_observation()
            last_priv_obs = self.env.privileged()
            last_value_k0 = self.critic(last_obs[:, 0])
            last_value = torch.cat([last_value_k0, self.critic(last_obs[:, 1])])
            last_priv_value = self.priv_critic(last_priv_obs).repeat(K)
        last_done = torch.ones(B, device=device)

        from experiments.array_face_s2.learning_repair.trainer import RolloutBuffer
        rb = RolloutBuffer(
            obs=obs_buf,
            # base slots carry the cell head (S4 convention; generic update
            # reads rb.mask_cell / rb.action_cell stashed below)
            mask_base=mask_bufs["cell"], mask_beam=mask_bufs["beam"],
            action_base=act_bufs["cell"], action_beam=act_bufs["beam"],
            logp=logp_buf, reward=rew_buf, value=val_buf,
            last_value=last_value, last_done=last_done,
        )
        rb.mask_cell = mask_bufs["cell"]
        rb.action_cell = act_bufs["cell"]
        rb.priv_obs = priv_obs_buf
        rb.priv_value = priv_val_buf
        rb.last_priv_value = last_priv_value
        return rb


def evaluate_actor_s5(
    actor, *,
    env_cfg: EnvConfig,
    physics: DebugPhysicsConfig,
    radar: UPAConfig,
    jammer: UPAConfig,
    scenario_seeds: list[int],
    n_action_reps: int = 4,
    sample: bool = True,
    device: str = "cpu",
    action_seed: int = 0,
) -> dict:
    """Per-scenario macro drop_ratio evaluation for S5 (both jammers)."""
    actor.eval()
    gen = torch.Generator(device=device).manual_seed(action_seed)
    per_seed_drops: list[float] = []
    raw_rows: list[dict] = []
    for sd in scenario_seeds:
        rep_drops: list[float] = []
        for rep in range(n_action_reps):
            env = ArrayFaceS5VecEnv(env_cfg, physics=physics, radar=radar, jammer=jammer)
            env.reset(seed=sd)
            for t in range(env_cfg.horizon):
                obs = env._build_observation()              # [E, K, OBS]
                mask_cell, mask_beam = env._compute_mask()  # [E, K, 25]
                actions = {}
                for k in range(N_JAMMERS):
                    masks_k = {"cell": mask_cell[:, k], "beam": mask_beam[:, k]}
                    with torch.no_grad():
                        if sample:
                            a_k, _ = sample_multihead(actor, obs[:, k], masks_k, gen)
                        else:
                            logits = actor.forward(obs[:, k])
                            a_beam = logits["beam"].masked_fill(
                                ~mask_beam[:, k].bool(), float("-inf")).argmax(dim=-1)
                            a_cell = (logits["cell"].masked_fill(
                                ~mask_cell[:, k].bool(), float("-inf")) > 0).to(torch.float32)
                            a_k = {"cell": a_cell, "beam": a_beam}
                    actions[k] = a_k
                cell = torch.stack([actions[k]["cell"] for k in range(N_JAMMERS)], dim=1)
                beam = torch.stack([actions[k]["beam"] for k in range(N_JAMMERS)], dim=1)
                env.step(cell, beam)
            rep_drop = float(env.drop_ratio()[0])
            rep_drops.append(rep_drop)
            raw_rows.append({"seed": int(sd), "rep": int(rep), "drop_ratio": rep_drop})
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
