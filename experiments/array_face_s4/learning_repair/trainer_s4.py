"""S4 PPO trainer — extends S2PPOTrainerV2 for two-head (cell + beam) on the UPA.

Trainer-side changes from S2PPOTrainerV2:
  1. collect_rollout: 2-arg env.step(action_cell, action_beam) — no base head;
     masks/actions keyed by head name ("cell", "beam").
  2. evaluate_actor_s4: same 2-head sampling path.

The PPO update is fully inherited: trainer_v2.update() flattens every head
generically by name (rb.mask_<name>/rb.action_<name>), so registering
HeadSpec("cell","bernoulli",25) + HeadSpec("beam","categorical",25) is enough
— no change to the clipped surrogate / GAE / KL-rollback / per-head entropy.
"""
from __future__ import annotations
from pathlib import Path

import torch

from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.array_face_s4 import (
    EnvConfig, ArrayFaceS4VecEnv, UPAConfig,
    OBS_DIM_S4,
)
from experiments.array_face_s2.learning_repair.trainer_v2 import (
    S2PPOConfigV2, S2PPOTrainerV2,
)
from experiments.array_face_s2.learning_repair.actor_heads import sample_multihead


class S4PPOTrainer(S2PPOTrainerV2):
    """S4 trainer: two-head (cell + beam) PPO on ArrayFaceS4VecEnv.

    Subclasses S2PPOTrainerV2 to override collect_rollout (2-arg env.step) and
    to build an S4 env. The PPO update is fully inherited.
    """

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
        self.actor = MultiHeadActor(OBS_DIM_S4, self.head_specs).to(cfg.device)
        self.critic = ValueCritic(OBS_DIM_S4).to(cfg.device)
        self.actor_opt = _t.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = _t.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.priv_critic = None
        self.priv_critic_opt = None
        if cfg.use_privileged_critic:
            from env.gpu.array_face_s4 import PRIVILEGED_DIM_S4
            from experiments.array_face_s2.learning_repair.trainer_v2 import PrivilegedValueCritic
            self.priv_critic = PrivilegedValueCritic(PRIVILEGED_DIM_S4).to(cfg.device)
            self.priv_critic_opt = _t.optim.Adam(self.priv_critic.parameters(), lr=cfg.critic_lr)

        self.env = ArrayFaceS4VecEnv(
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
        """Override: 2-head (cell + beam) sampling, 2-arg env.step."""
        T = self.env_cfg.horizon
        E = self.env_cfg.n_envs
        device = self.cfg.device
        obs_buf = torch.zeros(T, E, OBS_DIM_S4, device=device)
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

        for t in range(T):
            obs = self.env._build_observation()
            mask_cell, mask_beam = self.env._compute_mask()
            masks = {"cell": mask_cell, "beam": mask_beam}
            with torch.no_grad():
                actions, logp = self._sample_actions(obs, masks)
                value = self.critic(obs)
            obs_buf[t] = obs
            mask_bufs["cell"][t] = mask_cell
            mask_bufs["beam"][t] = mask_beam
            for n in self.head_names:
                act_bufs[n][t] = actions[n]
            logp_buf[t] = logp
            val_buf[t] = value
            step_out = self.env.step(actions["cell"], actions["beam"])
            rew_buf[t] = step_out[1]

        with torch.no_grad():
            last_obs = self.env._build_observation()
            last_value = self.critic(last_obs)
        last_done = torch.ones(E, device=device)

        from experiments.array_face_s2.learning_repair.trainer import RolloutBuffer
        rb = RolloutBuffer(
            obs=obs_buf,
            # RolloutBuffer's dataclass requires base/beam fields; S4 has no
            # base head, so the cell head occupies the base slots (never read
            # generically) and the beam head its own slots. The generic update
            # reads rb.mask_cell/rb.action_cell (stashed below) and the beam
            # fields directly.
            mask_base=mask_bufs["cell"], mask_beam=mask_bufs["beam"],
            action_base=act_bufs["cell"], action_beam=act_bufs["beam"],
            logp=logp_buf, reward=rew_buf, value=val_buf,
            last_value=last_value, last_done=last_done,
        )
        rb.mask_cell = mask_bufs["cell"]
        rb.action_cell = act_bufs["cell"]
        return rb


def evaluate_actor_s4(
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
    """Per-scenario macro drop_ratio evaluation for S4 (cell + beam heads)."""
    actor.eval()
    gen = torch.Generator(device=device).manual_seed(action_seed)
    per_seed_drops: list[float] = []
    raw_rows: list[dict] = []
    for sd in scenario_seeds:
        rep_drops: list[float] = []
        for rep in range(n_action_reps):
            env = ArrayFaceS4VecEnv(env_cfg, physics=physics, radar=radar, jammer=jammer)
            env.reset(seed=sd)
            for t in range(env_cfg.horizon):
                obs = env._build_observation()
                mask_cell, mask_beam = env._compute_mask()
                masks = {"cell": mask_cell, "beam": mask_beam}
                with torch.no_grad():
                    if sample:
                        actions, _ = sample_multihead(actor, obs, masks, gen)
                    else:
                        logits = actor.forward(obs)
                        # beam: masked argmax over 25 directions
                        action_beam = logits["beam"].masked_fill(
                            ~mask_beam.bool(), float("-inf")).argmax(dim=-1)
                        # cell: threshold at 0 (bernoulli logit > 0 -> cell on);
                        # masked cells (energy exhausted) forced off
                        action_cell = (logits["cell"].masked_fill(
                            ~mask_cell.bool(), float("-inf")) > 0).to(torch.float32)
                        actions = {"cell": action_cell, "beam": action_beam}
                env.step(actions["cell"], actions["beam"])
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
