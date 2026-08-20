"""S6 self-play trainer — 1 jammer (S4-style) vs 2 radars (parameter-shared team).

Reuse strategy: the ENTIRE S2PPOTrainerV2.update() (GAE, clipped surrogate,
KL rollback, per-head entropy) is side-agnostic — it operates on
(self.actor, self.critic, self.actor_opt, self.critic_opt, rb). This trainer:

  1. builds BOTH sides' nets (jammer: cell+beam heads; radar team: beam+svc
     heads, parameter-shared across R=2 via S5's K-flattening),
  2. collects ONE shared rollout producing TWO buffers (rb_j [T,E] and
     rb_r [T,E*R] with the team reward duplicated),
  3. updates the jammer side, then SWAPS self.actor/critic/optimizers to the
     radar side and runs the same update on rb_r, then swaps back.

Rewards: jammer = newly_dropped + pending shaping; radar = newly_succeeded −
newly_dropped (event-level opposing signals; see env docstring).
"""
from __future__ import annotations
from pathlib import Path

import torch

from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.array_face_s6 import (
    EnvConfig, ArrayFaceS6VecEnv, UPAConfig,
    OBS_DIM_JAMMER, OBS_DIM_RADAR, N_RADARS,
)
from experiments.array_face_s2.learning_repair.trainer_v2 import (
    S2PPOConfigV2, S2PPOTrainerV2,
)
from experiments.array_face_s2.learning_repair.actor_heads import (
    MultiHeadActor, sample_multihead,
)
from experiments.array_face_s2.learning_repair.trainer import (
    ValueCritic, RolloutBuffer, manifest_sha,
)


class S6SelfPlayTrainer(S2PPOTrainerV2):
    """Adversarial trainer: jammer PPO vs radar-team PPO on one shared env."""

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
        jammer_specs: list,
        radar_specs: list,
    ):
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.physics = physics
        self.radar = radar
        self.jammer = jammer
        self.train_seeds = list(train_seeds)
        self.manifest_path = Path(manifest_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.jammer_specs = tuple(jammer_specs)
        self.radar_specs = tuple(radar_specs)

        torch.manual_seed(cfg.seed)
        # jammer side
        self.jam_actor = MultiHeadActor(OBS_DIM_JAMMER, self.jammer_specs).to(cfg.device)
        self.jam_critic = ValueCritic(OBS_DIM_JAMMER).to(cfg.device)
        self.jam_actor_opt = torch.optim.Adam(self.jam_actor.parameters(), lr=cfg.actor_lr)
        self.jam_critic_opt = torch.optim.Adam(self.jam_critic.parameters(), lr=cfg.critic_lr)
        # radar team (parameter-shared across R=2)
        self.rad_actor = MultiHeadActor(OBS_DIM_RADAR, self.radar_specs).to(cfg.device)
        self.rad_critic = ValueCritic(OBS_DIM_RADAR).to(cfg.device)
        self.rad_actor_opt = torch.optim.Adam(self.rad_actor.parameters(), lr=cfg.actor_lr)
        self.rad_critic_opt = torch.optim.Adam(self.rad_critic.parameters(), lr=cfg.critic_lr)

        # default side = jammer (base-class compat for checkpointing etc.)
        self.actor, self.critic = self.jam_actor, self.jam_critic
        self.actor_opt, self.critic_opt = self.jam_actor_opt, self.jam_critic_opt
        self.head_specs, self.head_names = self.jammer_specs, tuple(s.name for s in self.jammer_specs)

        self.priv_critic = None  # S6 v1: plain obs critics both sides
        self.priv_critic_opt = None

        self.env = ArrayFaceS6VecEnv(
            self.env_cfg, physics=self.physics, radar=self.radar, jammer=self.jammer,
        )
        self._action_gen = torch.Generator(device=cfg.device).manual_seed(cfg.train_seed)
        self._radar_gen = torch.Generator(device=cfg.device).manual_seed(cfg.train_seed + 7)

        self._config_sha = cfg.config_sha()
        self._manifest_sha = manifest_sha(self.manifest_path)

        self.iteration = -1
        self.update_count = 0
        self.cumulative_transitions = 0
        self.history: list[dict] = []
        self.kl_rollback_count = 0
        self._return_rms = None
        self._snapshot_actor_state()

    # ---------- side swapping ----------
    def _swap_to_radar(self):
        self.actor, self.critic = self.rad_actor, self.rad_critic
        self.actor_opt, self.critic_opt = self.rad_actor_opt, self.rad_critic_opt
        self.head_specs, self.head_names = self.radar_specs, tuple(s.name for s in self.radar_specs)
        self._snapshot_actor_state()

    def _swap_to_jammer(self):
        self.actor, self.critic = self.jam_actor, self.jam_critic
        self.actor_opt, self.critic_opt = self.jam_actor_opt, self.jam_critic_opt
        self.head_specs, self.head_names = self.jammer_specs, tuple(s.name for s in self.jammer_specs)
        self._snapshot_actor_state()

    # ---------- rollout ----------
    def collect_rollout(self):
        T = self.env_cfg.horizon
        E = self.env_cfg.n_envs
        R = N_RADARS
        B = E * R
        device = self.cfg.device

        obsj_buf = torch.zeros(T, E, OBS_DIM_JAMMER, device=device)
        mask_cell_buf = torch.zeros(T, E, 25, device=device)
        mask_beam_buf = torch.zeros(T, E, 25, device=device)
        act_cell_buf = torch.zeros(T, E, 25, device=device)
        act_beam_buf = torch.zeros(T, E, dtype=torch.int64, device=device)
        logp_j = torch.zeros(T, E, device=device)
        rew_j = torch.zeros(T, E, device=device)
        val_j = torch.zeros(T, E, device=device)

        obsr_buf = torch.zeros(T, B, OBS_DIM_RADAR, device=device)
        mask_rbeam_buf = torch.zeros(T, B, 25, device=device)
        mask_rsvc_buf = torch.zeros(T, B, 2, device=device)
        act_rbeam_buf = torch.zeros(T, B, dtype=torch.int64, device=device)
        act_rsvc_buf = torch.zeros(T, B, dtype=torch.int64, device=device)
        logp_r = torch.zeros(T, B, device=device)
        rew_r = torch.zeros(T, B, device=device)
        val_r = torch.zeros(T, B, device=device)

        for t in range(T):
            obs_j, obs_r = self.env._build_observation()
            mask_cell, mask_beam = self.env._compute_masks()
            with torch.no_grad():
                actions_j, lp_j = sample_multihead(
                    self.jam_actor, obs_j,
                    {"cell": mask_cell, "beam": mask_beam}, self._action_gen)
                vj = self.jam_critic(obs_j)
            obsj_buf[t] = obs_j
            mask_cell_buf[t] = mask_cell
            mask_beam_buf[t] = mask_beam
            act_cell_buf[t] = actions_j["cell"]
            act_beam_buf[t] = actions_j["beam"]
            logp_j[t] = lp_j
            val_j[t] = vj

            step_radars = {}
            for r in range(R):
                obs_rk = obs_r[:, r]
                masks_r = {"beam": torch.ones(E, 25, dtype=torch.bool, device=device),
                           "svc": torch.ones(E, 2, dtype=torch.bool, device=device)}
                with torch.no_grad():
                    actions_r, lp_r = sample_multihead(
                        self.rad_actor, obs_rk, masks_r, self._radar_gen)
                    vr = self.rad_critic(obs_rk)
                sl = slice(r * E, (r + 1) * E)
                obsr_buf[t, sl] = obs_rk
                mask_rbeam_buf[t, sl] = masks_r["beam"]
                mask_rsvc_buf[t, sl] = masks_r["svc"]
                act_rbeam_buf[t, sl] = actions_r["beam"]
                act_rsvc_buf[t, sl] = actions_r["svc"]
                logp_r[t, sl] = lp_r
                val_r[t, sl] = vr
                step_radars[r] = actions_r

            r_beam = torch.stack([step_radars[r]["beam"] for r in range(R)], dim=1)
            r_svc = torch.stack([step_radars[r]["svc"] for r in range(R)], dim=1)
            (oj2, or2), (rj, rr), done, info = self.env.step(
                actions_j["cell"], actions_j["beam"], r_beam, r_svc)
            rew_j[t] = rj
            rew_r[t] = rr.repeat(R)

        with torch.no_grad():
            obs_j, obs_r = self.env._build_observation()
            last_vj = self.jam_critic(obs_j)
            last_vr = torch.cat([self.rad_critic(obs_r[:, r]) for r in range(R)])

        ones_E = torch.ones(E, device=device)
        ones_B = torch.ones(B, device=device)
        rb_j = RolloutBuffer(
            obs=obsj_buf, mask_base=mask_cell_buf, mask_beam=mask_beam_buf,
            action_base=act_cell_buf, action_beam=act_beam_buf,
            logp=logp_j, reward=rew_j, value=val_j,
            last_value=last_vj, last_done=ones_E,
        )
        rb_j.mask_cell = mask_cell_buf
        rb_j.action_cell = act_cell_buf

        rb_r = RolloutBuffer(
            obs=obsr_buf, mask_base=mask_rbeam_buf, mask_beam=mask_rbeam_buf,
            action_base=act_rbeam_buf, action_beam=act_rbeam_buf,
            logp=logp_r, reward=rew_r, value=val_r,
            last_value=last_vr, last_done=ones_B,
        )
        rb_r.mask_svc = mask_rsvc_buf
        rb_r.action_svc = act_rsvc_buf

        self._rb_j = rb_j
        self._rb_r = rb_r
        return rb_j

    # ---------- iteration ----------
    def train_iteration(self) -> dict:
        if self.iteration < 0:
            self.iteration = 0
        else:
            self.iteration += 1
        self._assign_scenarios_and_reset()
        rb = self.collect_rollout()

        self._swap_to_jammer()
        metrics_j = self.update(self._rb_j)
        self._swap_to_radar()
        metrics_r = self.update(self._rb_r)
        self._swap_to_jammer()

        self.update_count += 1
        self.cumulative_transitions += int(rb.obs.shape[0] * rb.obs.shape[1])
        drops = float(self.env.drop_ratio().mean().item())
        succ = float(self.env.success_ratio().mean().item())
        metrics = {
            "iteration": self.iteration,
            "rollout_drop": drops,
            "rollout_success": succ,
            "jammer_entropy": metrics_j.get("entropy", 0.0),
            "jammer_entropy_cell": metrics_j.get("entropy_cell", 0.0),
            "jammer_entropy_beam": metrics_j.get("entropy_beam", 0.0),
            "radar_entropy": metrics_r.get("entropy", 0.0),
            "radar_entropy_beam": metrics_r.get("entropy_beam", 0.0),
            "radar_entropy_svc": metrics_r.get("entropy_svc", 0.0),
            "jammer_clip_frac": metrics_j.get("clip_frac", 0.0),
            "radar_clip_frac": metrics_r.get("clip_frac", 0.0),
            "cumulative_transitions": self.cumulative_transitions,
        }
        self.history.append(metrics)
        return metrics

    # ---------- checkpointing both sides ----------
    def save_selfplay(self, path: Path):
        """ATOMIC checkpoint: write to a temp file, then os.replace.

        Two crash-corrupted checkpoints taught us the lesson — torch.save
        directly to the live file leaves a truncated archive when the
        process dies mid-write, silently resetting training to scratch.
        """
        import os
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "iteration": self.iteration,
            "jam_actor": self.jam_actor.state_dict(),
            "jam_critic": self.jam_critic.state_dict(),
            "jam_actor_opt": self.jam_actor_opt.state_dict(),
            "jam_critic_opt": self.jam_critic_opt.state_dict(),
            "rad_actor": self.rad_actor.state_dict(),
            "rad_critic": self.rad_critic.state_dict(),
            "rad_actor_opt": self.rad_actor_opt.state_dict(),
            "rad_critic_opt": self.rad_critic_opt.state_dict(),
        }, tmp)
        os.replace(tmp, path)

    def load_selfplay(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.jam_actor.load_state_dict(ckpt["jam_actor"])
        self.jam_critic.load_state_dict(ckpt["jam_critic"])
        self.jam_actor_opt.load_state_dict(ckpt["jam_actor_opt"])
        self.jam_critic_opt.load_state_dict(ckpt["jam_critic_opt"])
        self.rad_actor.load_state_dict(ckpt["rad_actor"])
        self.rad_critic.load_state_dict(ckpt["rad_critic"])
        self.rad_actor_opt.load_state_dict(ckpt["rad_actor_opt"])
        self.rad_critic_opt.load_state_dict(ckpt["rad_critic_opt"])
        self._swap_to_jammer()
        # Restore the counter too: without this a resumed session re-reports
        # iterations 0..N into the metrics logs (duplicate segments) and
        # restarts the entropy-anneal schedule from full exploration.
        self.iteration = int(ckpt["iteration"])
        return self.iteration


# ---------- evaluation ----------

def evaluate_s6(
    jam_actor, rad_actor, *,
    env_cfg: EnvConfig,
    physics: DebugPhysicsConfig,
    radar: UPAConfig,
    jammer: UPAConfig,
    scenario_seeds: list[int],
    n_action_reps: int = 2,
    device: str = "cpu",
    action_seed: int = 4242,
) -> dict:
    """Three-view evaluation: head-to-head, jammer vs scripted sweep, radar vs idle jammer."""
    from env.gpu.array_face_s6 import ArrayFaceS6VecEnv

    def run(seed: int, mode: str) -> tuple[float, float]:
        env = ArrayFaceS6VecEnv(env_cfg, physics=physics, radar=radar, jammer=jammer)
        env.reset(seed=seed)
        gen = torch.Generator(device=device).manual_seed(action_seed + seed)
        E = env_cfg.n_envs
        for t in range(env_cfg.horizon):
            obs_j, obs_r = env._build_observation()
            mask_cell, mask_beam = env._compute_masks()
            if mode in ("h2h", "jam_only"):
                with torch.no_grad():
                    a_j, _ = sample_multihead(jam_actor, obs_j,
                                              {"cell": mask_cell, "beam": mask_beam}, gen)
                jcell, jbeam = a_j["cell"], a_j["beam"]
            else:  # idle jammer
                jcell = torch.zeros(E, 25, device=device)
                jbeam = torch.zeros(E, dtype=torch.int64, device=device)
            if mode in ("h2h", "rad_only"):
                rb_ = torch.zeros(E, N_RADARS, dtype=torch.int64, device=device)
                rs_ = torch.zeros(E, N_RADARS, dtype=torch.int64, device=device)
                for r in range(N_RADARS):
                    masks = {"beam": torch.ones(E, 25, dtype=torch.bool, device=device),
                             "svc": torch.ones(E, 2, dtype=torch.bool, device=device)}
                    with torch.no_grad():
                        a_r, _ = sample_multihead(rad_actor, obs_r[:, r], masks, gen)
                    rb_[:, r] = a_r["beam"]
                    rs_[:, r] = a_r["svc"]
            else:  # scripted sweep (both radars): S4-style deterministic cycling
                b = t % 25
                rb_ = torch.full((E, N_RADARS), b, dtype=torch.int64, device=device)
                rs_ = torch.full((E, N_RADARS), t % 2, dtype=torch.int64, device=device)
            env.step(jcell, jbeam, rb_, rs_)
        return float(env.drop_ratio()[0]), float(env.success_ratio()[0])

    views = {}
    for mode in ("h2h", "jam_only", "rad_only"):
        drops, succs = [], []
        for sd in scenario_seeds:
            for _ in range(n_action_reps):
                d, s = run(sd, mode)
                drops.append(d)
                succs.append(s)
        views[mode] = {"mean_drop": sum(drops) / len(drops), "mean_success": sum(succs) / len(succs)}
    return views
