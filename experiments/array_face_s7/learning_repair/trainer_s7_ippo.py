"""S7 IPPO baseline — independent PPO learners, no parameter sharing, no CTDE.

Scientific role: algorithm control for the MAPPO main results. The question is
whether the attacker-count containment collapse is a property of the game or
of MAPPO specifically. IPPO is the standard independent-learner control
(de Witt et al., 2020): every agent keeps its OWN actor and local critic and
is updated only on its own observation/reward stream. Differences from the
MAPPO trainer are exactly:

  1. per-agent actors (no parameter sharing within a team);
  2. per-agent local critics only — no privileged central critic, no
     distillation (cfg keeps use_privileged_critic=True only because the
     parent constructor demands it; the priv critics are never trained);
  3. per-agent optimizers; 4 PPO update() calls per iteration, each on a
     [T, E] buffer holding only that agent's slots.

Everything else (env, team reward duplication, GAE, clipping, per-head
entropy anneal keyed on self.iteration, KL rollback, action/mask layouts) is
reused verbatim from the base classes so the comparison isolates the learning
rule. The base update() skips the privileged path whenever the rollout buffer
carries no priv fields, which is how the CTDE machinery is disabled here.
"""
from __future__ import annotations

from pathlib import Path

import torch

from env.gpu.array_face_s7 import (
    OBS_DIM_JAMMER, OBS_DIM_RADAR, N_JAMMERS, N_RADARS,
)
from experiments.array_face_s2.learning_repair.actor_heads import (
    MultiHeadActor, sample_multihead,
)
from experiments.array_face_s2.learning_repair.trainer import RolloutBuffer
from experiments.array_face_s7.learning_repair.trainer_s7 import S7SelfPlayTrainer


class S7IPPOTrainer(S7SelfPlayTrainer):
    """Two-team IPPO: 4 independent learners in the same S7 game."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        K, R = N_JAMMERS, N_RADARS
        dev = self.cfg.device
        # per-agent independent learners (replaces the shared team nets)
        self.jam_actors, self.jam_critics = [], []
        self.jam_actor_opts, self.jam_critic_opts = [], []
        for _ in range(K):
            a = MultiHeadActor(OBS_DIM_JAMMER, self.jammer_specs).to(dev)
            from experiments.array_face_s2.learning_repair.trainer import ValueCritic
            c = ValueCritic(OBS_DIM_JAMMER).to(dev)
            self.jam_actors.append(a)
            self.jam_critics.append(c)
            self.jam_actor_opts.append(torch.optim.Adam(a.parameters(), lr=self.cfg.actor_lr))
            self.jam_critic_opts.append(torch.optim.Adam(c.parameters(), lr=self.cfg.critic_lr))
        self.rad_actors, self.rad_critics = [], []
        self.rad_actor_opts, self.rad_critic_opts = [], []
        for _ in range(R):
            a = MultiHeadActor(OBS_DIM_RADAR, self.radar_specs).to(dev)
            from experiments.array_face_s2.learning_repair.trainer import ValueCritic
            c = ValueCritic(OBS_DIM_RADAR).to(dev)
            self.rad_actors.append(a)
            self.rad_critics.append(c)
            self.rad_actor_opts.append(torch.optim.Adam(a.parameters(), lr=self.cfg.actor_lr))
            self.rad_critic_opts.append(torch.optim.Adam(c.parameters(), lr=self.cfg.critic_lr))
        # per-agent action RNG streams (seeded, reproducible)
        self._jam_gens = [torch.Generator(device=dev).manual_seed(self.cfg.train_seed + 101 * k)
                          for k in range(K)]
        self._rad_gens = [torch.Generator(device=dev).manual_seed(self.cfg.train_seed + 7 + 101 * r)
                          for r in range(R)]
        self._swap_agent("jam", 0)

    # ---------- per-agent side swapping ----------
    def _swap_agent(self, team: str, i: int):
        if team == "jam":
            self.actor = self.jam_actors[i]
            self.critic = self.jam_critics[i]
            self.actor_opt = self.jam_actor_opts[i]
            self.critic_opt = self.jam_critic_opts[i]
            self.head_specs = self.jammer_specs
        else:
            self.actor = self.rad_actors[i]
            self.critic = self.rad_critics[i]
            self.actor_opt = self.rad_actor_opts[i]
            self.critic_opt = self.rad_critic_opts[i]
            self.head_specs = self.radar_specs
        self.head_names = tuple(s.name for s in self.head_specs)
        self._snapshot_actor_state()

    # ---------- rollout ----------
    def collect_rollout(self, singleton_opponent: bool = False):
        assert not singleton_opponent, "IPPO control does not implement R5 mixing"
        T = self.env_cfg.horizon
        E = self.env_cfg.n_envs
        K, R = N_JAMMERS, N_RADARS
        device = self.cfg.device

        # per-agent [T, E] buffers (independent learners never see partner slots)
        def mk_jam():
            return dict(
                obs=torch.zeros(T, E, OBS_DIM_JAMMER, device=device),
                mask_cell=torch.zeros(T, E, 25, device=device),
                mask_beam=torch.zeros(T, E, 25, device=device),
                act_cell=torch.zeros(T, E, 25, device=device),
                act_beam=torch.zeros(T, E, dtype=torch.int64, device=device),
                logp=torch.zeros(T, E, device=device),
                rew=torch.zeros(T, E, device=device),
                val=torch.zeros(T, E, device=device),
            )

        def mk_rad():
            return dict(
                obs=torch.zeros(T, E, OBS_DIM_RADAR, device=device),
                mask_beam=torch.zeros(T, E, 25, device=device),
                mask_svc=torch.zeros(T, E, 2, device=device),
                act_beam=torch.zeros(T, E, dtype=torch.int64, device=device),
                act_svc=torch.zeros(T, E, dtype=torch.int64, device=device),
                logp=torch.zeros(T, E, device=device),
                rew=torch.zeros(T, E, device=device),
                val=torch.zeros(T, E, device=device),
            )

        jb = [mk_jam() for _ in range(K)]
        rb = [mk_rad() for _ in range(R)]

        obs_j, obs_r = self.env._build_observation()
        for t in range(T):
            mask_cell, mask_beam = self.env._compute_masks()
            step_jammers = {}
            for k in range(K):
                obs_k = obs_j[:, k]
                masks_k = {"cell": mask_cell[:, k], "beam": mask_beam[:, k]}
                with torch.no_grad():
                    actions_k, lp_k = sample_multihead(
                        self.jam_actors[k], obs_k, masks_k, self._jam_gens[k])
                    vk = self.jam_critics[k](obs_k)
                jb[k]["obs"][t] = obs_k
                jb[k]["mask_cell"][t] = masks_k["cell"]
                jb[k]["mask_beam"][t] = masks_k["beam"]
                jb[k]["act_cell"][t] = actions_k["cell"]
                jb[k]["act_beam"][t] = actions_k["beam"]
                jb[k]["logp"][t] = lp_k
                jb[k]["val"][t] = vk
                step_jammers[k] = actions_k

            step_radars = {}
            for r in range(R):
                obs_rk = obs_r[:, r]
                masks_r = {"beam": self.env._radar_mask_beam,
                           "svc": self.env._radar_mask_svc}
                with torch.no_grad():
                    actions_r, lp_rk = sample_multihead(
                        self.rad_actors[r], obs_rk, masks_r, self._rad_gens[r])
                    vr = self.rad_critics[r](obs_rk)
                rb[r]["obs"][t] = obs_rk
                rb[r]["mask_beam"][t] = masks_r["beam"]
                rb[r]["mask_svc"][t] = masks_r["svc"]
                rb[r]["act_beam"][t] = actions_r["beam"]
                rb[r]["act_svc"][t] = actions_r["svc"]
                rb[r]["logp"][t] = lp_rk
                rb[r]["val"][t] = vr
                step_radars[r] = actions_r

            j_cell = torch.stack([step_jammers[k]["cell"] for k in range(K)], dim=1)
            j_beam = torch.stack([step_jammers[k]["beam"] for k in range(K)], dim=1)
            r_beam = torch.stack([step_radars[r]["beam"] for r in range(R)], dim=1)
            r_svc = torch.stack([step_radars[r]["svc"] for r in range(R)], dim=1)
            (obs_j, obs_r), (rj, rr), done, info = self.env.step(
                j_cell, j_beam, r_beam, r_svc)
            for k in range(K):
                jb[k]["rew"][t] = rj  # team reward (IPPO shares the env reward)
            for r in range(R):
                rb[r]["rew"][t] = rr

        ones = torch.ones(E, device=device)
        self._agent_rbs = []
        for k in range(K):
            with torch.no_grad():
                last_v = self.jam_critics[k](obs_j[:, k])
            b = jb[k]
            rbk = RolloutBuffer(
                obs=b["obs"], mask_base=b["mask_cell"], mask_beam=b["mask_beam"],
                action_base=b["act_cell"], action_beam=b["act_beam"],
                logp=b["logp"], reward=b["rew"], value=b["val"],
                last_value=last_v, last_done=ones,
            )
            rbk.mask_cell = b["mask_cell"]
            rbk.action_cell = b["act_cell"]
            # no priv fields attached -> base update() runs local-critic PPO
            self._agent_rbs.append(("jam", k, rbk))
        for r in range(R):
            with torch.no_grad():
                last_v = self.rad_critics[r](obs_r[:, r])
            b = rb[r]
            rbr = RolloutBuffer(
                obs=b["obs"], mask_base=b["mask_beam"], mask_beam=b["mask_beam"],
                action_base=b["act_beam"], action_beam=b["act_beam"],
                logp=b["logp"], reward=b["rew"], value=b["val"],
                last_value=last_v, last_done=ones,
            )
            rbr.mask_svc = b["mask_svc"]
            rbr.action_svc = b["act_svc"]
            self._agent_rbs.append(("rad", r, rbr))
        return self._agent_rbs[0][2]

    # ---------- iteration: 4 independent PPO updates ----------
    def train_iteration(self) -> dict:
        if self.iteration < 0:
            self.iteration = 0
        else:
            self.iteration += 1
        self._assign_scenarios_and_reset()
        rb = self.collect_rollout()

        metrics_by_agent = []
        for team, i, agent_rb in self._agent_rbs:
            self._swap_agent(team, i)
            metrics_by_agent.append(self.update(agent_rb))
        self._swap_agent("jam", 0)

        self.update_count += 1
        self.cumulative_transitions += int(rb.obs.shape[0] * rb.obs.shape[1]) * 4
        drops = float(self.env.drop_ratio().mean().item())
        succ = float(self.env.success_ratio().mean().item())

        def agg(key, idxs):
            return sum(metrics_by_agent[i].get(key, 0.0) for i in idxs) / len(idxs)

        jam_idx = [0, 1]
        rad_idx = [2, 3]
        metrics = {
            "iteration": self.iteration,
            "rollout_drop": drops,
            "rollout_success": succ,
            "jammer_entropy": agg("entropy", jam_idx),
            "jammer_entropy_cell": agg("entropy_cell", jam_idx),
            "jammer_entropy_beam": agg("entropy_beam", jam_idx),
            "radar_entropy": agg("entropy", rad_idx),
            "radar_entropy_beam": agg("entropy_beam", rad_idx),
            "radar_entropy_svc": agg("entropy_svc", rad_idx),
            "jammer_clip_frac": agg("clip_frac", jam_idx),
            "radar_clip_frac": agg("clip_frac", rad_idx),
            "cumulative_transitions": self.cumulative_transitions,
        }
        self.history.append(metrics)
        return metrics

    # ---------- per-agent checkpointing ----------
    def save_selfplay(self, path: Path):
        import os
        tmp = path.with_suffix(".pt.tmp")
        state = {"iteration": self.iteration, "algo": "ippo"}
        for k in range(N_JAMMERS):
            state[f"jam_actor_{k}"] = self.jam_actors[k].state_dict()
            state[f"jam_critic_{k}"] = self.jam_critics[k].state_dict()
            state[f"jam_actor_opt_{k}"] = self.jam_actor_opts[k].state_dict()
            state[f"jam_critic_opt_{k}"] = self.jam_critic_opts[k].state_dict()
        for r in range(N_RADARS):
            state[f"rad_actor_{r}"] = self.rad_actors[r].state_dict()
            state[f"rad_critic_{r}"] = self.rad_critics[r].state_dict()
            state[f"rad_actor_opt_{r}"] = self.rad_actor_opts[r].state_dict()
            state[f"rad_critic_opt_{r}"] = self.rad_critic_opts[r].state_dict()
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load_selfplay(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.cfg.device)
        assert ckpt.get("algo") == "ippo", "not an IPPO checkpoint"
        for k in range(N_JAMMERS):
            self.jam_actors[k].load_state_dict(ckpt[f"jam_actor_{k}"])
            self.jam_critics[k].load_state_dict(ckpt[f"jam_critic_{k}"])
            self.jam_actor_opts[k].load_state_dict(ckpt[f"jam_actor_opt_{k}"])
            self.jam_critic_opts[k].load_state_dict(ckpt[f"jam_critic_opt_{k}"])
        for r in range(N_RADARS):
            self.rad_actors[r].load_state_dict(ckpt[f"rad_actor_{r}"])
            self.rad_critics[r].load_state_dict(ckpt[f"rad_critic_{r}"])
            self.rad_actor_opts[r].load_state_dict(ckpt[f"rad_actor_opt_{r}"])
            self.rad_critic_opts[r].load_state_dict(ckpt[f"rad_critic_opt_{r}"])
        self._swap_agent("jam", 0)
        self.iteration = int(ckpt["iteration"])
        return self.iteration


# ---------- per-agent evaluation ----------
def evaluate_s7_ippo(jam_actors, rad_actors, **kwargs):
    """evaluate_s7() for independent per-agent actors; identical protocol.

    jam_actors: list[K] MultiHeadActor, rad_actors: list[R] MultiHeadActor.
    Accepts the same keyword protocol as trainer_s7.evaluate_s7.
    """
    from env.gpu.array_face_s7 import ArrayFaceS7VecEnv, N_JAMMERS, N_RADARS
    from dataclasses import replace

    env_cfg = kwargs["env_cfg"]
    physics = kwargs["physics"]
    radar, jammer = kwargs["radar"], kwargs["jammer"]
    scenario_seeds = kwargs["scenario_seeds"]
    n_action_reps = kwargs.get("n_action_reps", 2)
    device = kwargs.get("device", "cpu")
    action_seed = kwargs.get("action_seed", 4242)

    eval_env_cfg = replace(env_cfg, n_envs=1)

    def run(seed: int, mode: str) -> tuple[float, float]:
        env = ArrayFaceS7VecEnv(eval_env_cfg, physics=physics, radar=radar, jammer=jammer)
        env.reset(seed=seed)
        gen = torch.Generator(device=device).manual_seed(action_seed + seed)
        E, K, R = 1, N_JAMMERS, N_RADARS
        for t in range(env_cfg.horizon):
            obs_j, obs_r = env._build_observation()
            mask_cell, mask_beam = env._compute_masks()
            if mode in ("h2h", "jam_only", "j1_only"):
                cells, beams = [], []
                for k in range(K):
                    with torch.no_grad():
                        a_j, _ = sample_multihead(
                            jam_actors[k], obs_j[:, k],
                            {"cell": mask_cell[:, k], "beam": mask_beam[:, k]}, gen)
                    cells.append(a_j["cell"])
                    beams.append(a_j["beam"])
                jcell = torch.stack(cells, dim=1)
                jbeam = torch.stack(beams, dim=1)
                if mode == "j1_only":
                    jcell[:, 1] = 0.0
                    jbeam[:, 1] = 0
            else:
                jcell = torch.zeros(E, K, 25, device=device)
                jbeam = torch.zeros(E, K, dtype=torch.int64, device=device)
            if mode in ("h2h", "rad_only", "j1_only"):
                rb_ = torch.zeros(E, R, dtype=torch.int64, device=device)
                rs_ = torch.zeros(E, R, dtype=torch.int64, device=device)
                for r in range(R):
                    masks = {"beam": env._radar_mask_beam,
                             "svc": env._radar_mask_svc}
                    with torch.no_grad():
                        a_r, _ = sample_multihead(rad_actors[r], obs_r[:, r], masks, gen)
                    rb_[:, r] = a_r["beam"]
                    rs_[:, r] = a_r["svc"]
            else:
                b = t % 25
                rb_ = torch.full((E, R), b, dtype=torch.int64, device=device)
                rs_ = torch.full((E, R), t % 2, dtype=torch.int64, device=device)
            env.step(jcell, jbeam, rb_, rs_)
        return float(env.drop_ratio()[0]), float(env.success_ratio()[0])

    views = {}
    for mode in ("h2h", "jam_only", "rad_only", "j1_only"):
        drops, succs = [], []
        for sd in scenario_seeds:
            for _ in range(n_action_reps):
                d, s = run(sd, mode)
                drops.append(d)
                succs.append(s)
        views[mode] = {"mean_drop": sum(drops) / len(drops),
                       "mean_success": sum(succs) / len(succs)}
    return views
