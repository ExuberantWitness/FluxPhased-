"""S3 PPO trainer — extends S2PPOTrainerV2 for three-head (base + beam + cell).

The only trainer-side changes from S2PPOTrainerV2:
  1. collect_rollout: passes cell mask to env.step (3-arg signature), collects
     mask_cell + action_cell into the rollout buffer.
  2. evaluate_actor: samples three heads, passes all three to env.step.

Everything else (PPO update, GAE, KL-rollback, checkpoint/resume, per-head
entropy) is inherited unchanged from S2PPOTrainerV2 because the update math
iterates over self.head_specs generically — adding a third HeadSpec("cell",
"bernoulli", 5) is enough; the actor/KL/entropy/sampling all handle it.

The driver (run_s3_ppo.py) registers the three heads via head_specs and
passes this trainer class.
"""
from __future__ import annotations
from pathlib import Path

import torch

from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.array_face_s3 import (
    EnvConfig, ArrayFaceS3VecEnv, RadarULAConfig, JammerULAConfig,
    OBS_DIM_S3, N_CELLS,
)
from experiments.array_face_s2.learning_repair.trainer_v2 import (
    S2PPOConfigV2, S2PPOTrainerV2, evaluate_actor_v2,
)
from experiments.array_face_s2.learning_repair.actor_heads import sample_multihead


class S3PPOTrainer(S2PPOTrainerV2):
    """S3 trainer: three-head (base + beam + cell) PPO on ArrayFaceS3VecEnv.

    Subclasses S2PPOTrainerV2 to override collect_rollout (3-arg env.step) and
    to build an S3 env. The PPO update is fully inherited.
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
        head_specs: list,
    ):
        # Force the env to be S3 (the parent __init__ builds the env from
        # env_cfg; we pass an S3 EnvConfig so it constructs ArrayFaceS3VecEnv).
        # We bypass the parent's env construction by building it ourselves,
        # then call a reduced init. Simplest: call parent __init__ but it
        # hardcodes ArrayFaceS2VecEnv. So we replicate the parent init with
        # the S3 env class.
        from experiments.array_face_s2.learning_repair.actor_heads import MultiHeadActor
        from experiments.array_face_s2.learning_repair.trainer import (
            ValueCritic, manifest_sha,
        )
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
        self.actor = MultiHeadActor(OBS_DIM_S3, self.head_specs).to(cfg.device)
        self.critic = ValueCritic(OBS_DIM_S3).to(cfg.device)
        self.actor_opt = _t.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = _t.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # B1 privileged critic (optional)
        self.priv_critic = None
        self.priv_critic_opt = None
        if cfg.use_privileged_critic:
            from env.gpu.array_face_s3 import PRIVILEGED_DIM_S3
            from experiments.array_face_s2.learning_repair.trainer_v2 import PrivilegedValueCritic
            self.priv_critic = PrivilegedValueCritic(PRIVILEGED_DIM_S3).to(cfg.device)
            self.priv_critic_opt = _t.optim.Adam(self.priv_critic.parameters(), lr=cfg.critic_lr)

        # S3 env (not S2)
        self.env = ArrayFaceS3VecEnv(
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
        """Override: collect cell mask + action, call env.step with 3 args."""
        T = self.env_cfg.horizon
        E = self.env_cfg.n_envs
        device = self.cfg.device
        obs_buf = torch.zeros(T, E, OBS_DIM_S3, device=device)
        mask_bufs = {n: torch.zeros(T, E, s.n_actions, device=device)
                     for n, s in zip(self.head_names, self.head_specs)}
        act_bufs = {}
        for n, s in zip(self.head_names, self.head_specs):
            if s.kind == "categorical":
                act_bufs[n] = torch.zeros(T, E, dtype=torch.int64, device=device)
            else:
                act_bufs[n] = torch.zeros(T, E, s.n_actions, device=device)
        logp_buf = torch.zeros(T, E, device=device)
        rew_buf = torch.zeros(T, E, device=device)
        val_buf = torch.zeros(T, E, device=device)

        use_priv = self.cfg.use_privileged_critic and self.priv_critic is not None
        if use_priv:
            from env.gpu.array_face_s3 import PRIVILEGED_DIM_S3
            priv_obs_buf = torch.zeros(T, E, PRIVILEGED_DIM_S3, device=device)
            priv_val_buf = torch.zeros(T, E, device=device)
        else:
            priv_obs_buf = None
            priv_val_buf = None

        for t in range(T):
            obs = self.env._build_observation()
            mask_base, mask_beam, mask_cell = self.env._compute_mask()
            masks = {"base": mask_base, "beam": mask_beam, "cell": mask_cell}
            with torch.no_grad():
                actions, logp = self._sample_actions(obs, masks)
                value = self.critic(obs)
                if use_priv:
                    priv_obs = self.env.privileged()
                    priv_value = self.priv_critic(priv_obs)
            obs_buf[t] = obs
            mask_bufs["base"][t] = mask_base
            mask_bufs["beam"][t] = mask_beam
            mask_bufs["cell"][t] = mask_cell
            for n in self.head_names:
                act_bufs[n][t] = actions[n]
            logp_buf[t] = logp
            val_buf[t] = value
            if use_priv:
                priv_obs_buf[t] = priv_obs
                priv_val_buf[t] = priv_value
            # S3: three-arg env.step
            step_out = self.env.step(actions["base"], actions["beam"], actions["cell"])
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

        from experiments.array_face_s2.learning_repair.trainer import RolloutBuffer
        rb = RolloutBuffer(
            obs=obs_buf,
            mask_base=mask_bufs["base"], mask_beam=mask_bufs["beam"],
            action_base=act_bufs["base"], action_beam=act_bufs["beam"],
            logp=logp_buf, reward=rew_buf, value=val_buf,
            last_value=last_value, last_done=last_done,
        )
        rb.priv_obs = priv_obs_buf
        rb.priv_value = priv_val_buf
        rb.last_priv_value = last_priv_value
        # S3 extra: stash cell actions for diagnostics (update loop doesn't need
        # them since cell head's logp is already in the joint logp_buf, but we
        # keep them for action_freq diagnostics).
        rb.action_cell = act_bufs["cell"]
        rb.mask_cell = mask_bufs["cell"]
        return rb


def evaluate_actor_s3(
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
    """Per-scenario macro drop_ratio evaluation for S3 (three heads)."""
    actor.eval()
    gen = torch.Generator(device=device).manual_seed(action_seed)
    per_seed_drops: list[float] = []
    raw_rows: list[dict] = []
    for sd in scenario_seeds:
        rep_drops: list[float] = []
        for rep in range(n_action_reps):
            env = ArrayFaceS3VecEnv(env_cfg, physics=physics, radar=radar, jammer=jammer)
            env.reset(seed=sd)
            for t in range(env_cfg.horizon):
                obs = env._build_observation()
                mask_base, mask_beam, mask_cell = env._compute_mask()
                masks = {"base": mask_base, "beam": mask_beam, "cell": mask_cell}
                with torch.no_grad():
                    if sample:
                        actions, _ = sample_multihead(actor, obs, masks, gen)
                    else:
                        logits = actor.forward(obs)
                        action_base = logits["base"].masked_fill(
                            ~mask_base.bool(), float("-inf")).argmax(dim=-1)
                        action_beam = logits["beam"].masked_fill(
                            ~mask_beam.bool(), float("-inf")).argmax(dim=-1)
                        # cell: threshold at 0 (bernoulli logit > 0 -> cell on)
                        action_cell = (logits["cell"] > 0).to(torch.float32)
                        actions = {"base": action_base, "beam": action_beam, "cell": action_cell}
                env.step(actions["base"], actions["beam"], actions["cell"])
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
