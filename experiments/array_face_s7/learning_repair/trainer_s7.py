"""S7 self-play trainer — 2-jammer team vs 2-radar team, full two-team MAPPO.

Reuse strategy: the S2PPOTrainerV2.update() is side-agnostic (GAE, clipped
surrogate, KL rollback, per-head entropy) and already supports the privileged
central critic path (compute_gae uses rb.priv_value when present; update()
trains priv_critic + distillation). This trainer:

  1. builds BOTH teams' nets. Each team = ONE parameter-shared MultiHeadActor
     + per-agent ValueCritic + ONE PrivilegedValueCritic (the CTDE central
     critic — the whole point of MAPPO),
  2. collects ONE shared rollout producing TWO K-flattened buffers
     (rb_j [T, E*K] jammer slots, rb_r [T, E*R] radar slots; team rewards
     duplicated; env-level priv obs/value duplicated into each team slot),
  3. updates the jammer team, then SWAPS self.actor/critic/priv_critic/
     optimizers/head_specs to the radar team and runs the same update on
     rb_r, then swaps back (S6's swap-update pattern, both sides K-flattened).

Rewards (env): jammer team = newly_dropped + pending shaping; radar team =
newly_succeeded − newly_dropped (event-level opposing signals).
"""
from __future__ import annotations
from pathlib import Path

import torch

from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.array_face_s7 import (
    EnvConfig, ArrayFaceS7VecEnv, UPAConfig,
    OBS_DIM_JAMMER, OBS_DIM_RADAR,
    PRIVILEGED_DIM_JAMMER, PRIVILEGED_DIM_RADAR,
    N_JAMMERS, N_RADARS,
    obs_dim_jammer, obs_dim_radar,
    priv_dim_jammer, priv_dim_radar,
)
from experiments.array_face_s2.learning_repair.trainer_v2 import (
    S2PPOConfigV2, S2PPOTrainerV2, PrivilegedValueCritic,
)
from experiments.array_face_s2.learning_repair.actor_heads import (
    MultiHeadActor, sample_multihead,
)
from experiments.array_face_s2.learning_repair.trainer import (
    ValueCritic, RolloutBuffer, manifest_sha,
)


class S7SelfPlayTrainer(S2PPOTrainerV2):
    """Two-team MAPPO trainer: jammer-team PPO vs radar-team PPO, shared env."""

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
        singleton_mix_frac: float = 0.0,
        radar_scripted: str | None = None,
    ):
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.physics = physics
        self.radar = radar
        self.jammer = jammer
        self.train_seeds = list(train_seeds)
        self.manifest_path = Path(manifest_path)
        self.out_dir = Path(out_dir)
        # Counter-adaptation control: 'greedy' pins BOTH radar heads to the
        # hottest pending (svc, az) cell each step (the scripted baseline that
        # exploits the self-play jammer team). Only the jammer team learns;
        # the radar update is skipped. Used to test whether jammers co-trained
        # against a stare learn to punish it.
        if radar_scripted not in (None, "greedy"):
            raise ValueError(f"radar_scripted must be None or 'greedy', got {radar_scripted!r}")
        self.radar_scripted = radar_scripted
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # R5 opponent-class mixing: fraction of TRAINING iterations on which
        # the radars face the singleton (jammer 1 forced idle) instead of the
        # pair. Deterministic cycling keeps the schedule reproducible. On
        # singleton iterations the jammer side's own update is SKIPPED (its
        # rollout no longer matches its policy), so the jammer team learns
        # purely from pair-vs-radar self-play while the radar team is
        # league-trained across opponent classes — the minimal league form.
        from fractions import Fraction
        frac = Fraction(singleton_mix_frac).limit_denominator(8)
        if frac < 0 or frac > 1:
            raise ValueError(f"singleton_mix_frac must be in [0,1], got {singleton_mix_frac}")
        if singleton_mix_frac > 0 and env_cfg.n_jammers != 2:
            raise ValueError("singleton mixing is defined only for the 2-jammer game")
        self.singleton_mix_frac = float(singleton_mix_frac)
        self._mix_num, self._mix_den = frac.numerator, frac.denominator

        self.jammer_specs = tuple(jammer_specs)
        self.radar_specs = tuple(radar_specs)
        # attacker-count scaling: per-n observation dims drive every net
        self._obs_dim_jam = obs_dim_jammer(env_cfg.n_jammers)
        self._obs_dim_rad = obs_dim_radar(env_cfg.n_jammers)
        self._priv_dim_jam = priv_dim_jammer(env_cfg.n_jammers)
        self._priv_dim_rad = priv_dim_radar(env_cfg.n_jammers)

        torch.manual_seed(cfg.seed)
        # jammer team (parameter-shared actor + per-agent critic + central critic)
        self.jam_actor = MultiHeadActor(self._obs_dim_jam, self.jammer_specs).to(cfg.device)
        self.jam_critic = ValueCritic(self._obs_dim_jam).to(cfg.device)
        self.jam_priv_critic = PrivilegedValueCritic(self._priv_dim_jam).to(cfg.device)
        self.jam_actor_opt = torch.optim.Adam(self.jam_actor.parameters(), lr=cfg.actor_lr)
        self.jam_critic_opt = torch.optim.Adam(self.jam_critic.parameters(), lr=cfg.critic_lr)
        self.jam_priv_critic_opt = torch.optim.Adam(
            self.jam_priv_critic.parameters(), lr=cfg.critic_lr)
        # radar team (same structure)
        self.rad_actor = MultiHeadActor(self._obs_dim_rad, self.radar_specs).to(cfg.device)
        self.rad_critic = ValueCritic(self._obs_dim_rad).to(cfg.device)
        self.rad_priv_critic = PrivilegedValueCritic(self._priv_dim_rad).to(cfg.device)
        self.rad_actor_opt = torch.optim.Adam(self.rad_actor.parameters(), lr=cfg.actor_lr)
        self.rad_critic_opt = torch.optim.Adam(self.rad_critic.parameters(), lr=cfg.critic_lr)
        self.rad_priv_critic_opt = torch.optim.Adam(
            self.rad_priv_critic.parameters(), lr=cfg.critic_lr)

        if not cfg.use_privileged_critic:
            raise ValueError("S7 requires use_privileged_critic=True (central critic)")

        # default side = jammer (base-class compat for checkpointing etc.)
        self.actor, self.critic = self.jam_actor, self.jam_critic
        self.actor_opt, self.critic_opt = self.jam_actor_opt, self.jam_critic_opt
        self.priv_critic, self.priv_critic_opt = self.jam_priv_critic, self.jam_priv_critic_opt
        self.head_specs, self.head_names = self.jammer_specs, tuple(s.name for s in self.jammer_specs)

        self.env = ArrayFaceS7VecEnv(
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
        self.priv_critic, self.priv_critic_opt = self.rad_priv_critic, self.rad_priv_critic_opt
        self.head_specs, self.head_names = self.radar_specs, tuple(s.name for s in self.radar_specs)
        self._snapshot_actor_state()

    def _swap_to_jammer(self):
        self.actor, self.critic = self.jam_actor, self.jam_critic
        self.actor_opt, self.critic_opt = self.jam_actor_opt, self.jam_critic_opt
        self.priv_critic, self.priv_critic_opt = self.jam_priv_critic, self.jam_priv_critic_opt
        self.head_specs, self.head_names = self.jammer_specs, tuple(s.name for s in self.jammer_specs)
        self._snapshot_actor_state()

    # ---------- rollout ----------
    def collect_rollout(self, singleton_opponent: bool = False):
        T = self.env_cfg.horizon
        E = self.env_cfg.n_envs
        K, R = self.env_cfg.n_jammers, N_RADARS
        Bj, Br = E * K, E * R
        device = self.cfg.device
        OJ, OR = self._obs_dim_jam, self._obs_dim_rad
        PJ, PR = self._priv_dim_jam, self._priv_dim_rad

        # jammer team buffer [T, Bj] (k-major slots)
        obsj_buf = torch.zeros(T, Bj, OJ, device=device)
        mask_cell_buf = torch.zeros(T, Bj, 25, device=device)
        mask_beam_buf = torch.zeros(T, Bj, 25, device=device)
        act_cell_buf = torch.zeros(T, Bj, 25, device=device)
        act_beam_buf = torch.zeros(T, Bj, dtype=torch.int64, device=device)
        logp_j = torch.zeros(T, Bj, device=device)
        rew_j = torch.zeros(T, Bj, device=device)
        val_j = torch.zeros(T, Bj, device=device)
        priv_j_buf = torch.zeros(T, Bj, PJ, device=device)
        priv_val_j = torch.zeros(T, Bj, device=device)

        # radar team buffer [T, Br] (r-major slots)
        obsr_buf = torch.zeros(T, Br, OR, device=device)
        mask_rbeam_buf = torch.zeros(T, Br, 25, device=device)
        mask_rsvc_buf = torch.zeros(T, Br, 2, device=device)
        act_rbeam_buf = torch.zeros(T, Br, dtype=torch.int64, device=device)
        act_rsvc_buf = torch.zeros(T, Br, dtype=torch.int64, device=device)
        logp_r = torch.zeros(T, Br, device=device)
        rew_r = torch.zeros(T, Br, device=device)
        val_r = torch.zeros(T, Br, device=device)
        priv_r_buf = torch.zeros(T, Br, PR, device=device)
        priv_val_r = torch.zeros(T, Br, device=device)

        # Carry the post-step observation forward. Env.step() already builds
        # this exact next state, so rebuilding it at the next loop head is
        # pure overhead.
        obs_j, obs_r = self.env._build_observation()
        for t in range(T):
            # The initial observation is carried forward from the previous
            # env.step() below; this avoids rebuilding the full public state
            # once per timestep.
            mask_cell, mask_beam = self.env._compute_masks()  # [E,K,25] x2
            # privileged() is exactly a reshape of these public observations;
            # reuse them instead of triggering a second tracker/one-hot pass.
            priv_j, priv_r = obs_j.reshape(E, -1), obs_r.reshape(E, -1)
            with torch.no_grad():
                pv_j = self.jam_priv_critic(priv_j)        # [E]
                pv_r = self.rad_priv_critic(priv_r)        # [E]

            step_jammers = {}
            for k in range(K):
                obs_k = obs_j[:, k]
                masks_k = {"cell": mask_cell[:, k], "beam": mask_beam[:, k]}
                with torch.no_grad():
                    actions_k, lp_k = sample_multihead(
                        self.jam_actor, obs_k, masks_k, self._action_gen)
                    vk = self.jam_critic(obs_k)
                sl = slice(k * E, (k + 1) * E)
                obsj_buf[t, sl] = obs_k
                mask_cell_buf[t, sl] = masks_k["cell"]
                mask_beam_buf[t, sl] = masks_k["beam"]
                act_cell_buf[t, sl] = actions_k["cell"]
                act_beam_buf[t, sl] = actions_k["beam"]
                logp_j[t, sl] = lp_k
                val_j[t, sl] = vk
                priv_j_buf[t, sl] = priv_j
                priv_val_j[t, sl] = pv_j
                step_jammers[k] = actions_k

            step_radars = {}
            for r in range(R):
                obs_rk = obs_r[:, r]
                if self.radar_scripted == "greedy":
                    # hottest pending (svc, az) from the radar observation;
                    # beam = azimuth + 10 (horizon-plane row), as in the
                    # evaluation-only greedy baseline
                    masks_r = {"beam": self.env._radar_mask_beam,
                               "svc": self.env._radar_mask_svc}
                    pm = obs_rk[0, 1:11].reshape(2, 5)
                    svc_idx = int(pm.sum(dim=1).argmax())
                    az_idx = int(pm[svc_idx].argmax())
                    actions_r = {"beam": torch.full((E,), az_idx + 10,
                                                    dtype=torch.int64, device=device),
                                 "svc": torch.full((E,), svc_idx,
                                                   dtype=torch.int64, device=device)}
                    lp_rk = torch.zeros(E, device=device)
                    vr = torch.zeros(E, device=device)
                else:
                    masks_r = {"beam": self.env._radar_mask_beam,
                               "svc": self.env._radar_mask_svc}
                    with torch.no_grad():
                        actions_r, lp_rk = sample_multihead(
                            self.rad_actor, obs_rk, masks_r, self._radar_gen)
                        vr = self.rad_critic(obs_rk)
                sl = slice(r * E, (r + 1) * E)
                obsr_buf[t, sl] = obs_rk
                mask_rbeam_buf[t, sl] = masks_r["beam"]
                mask_rsvc_buf[t, sl] = masks_r["svc"]
                act_rbeam_buf[t, sl] = actions_r["beam"]
                act_rsvc_buf[t, sl] = actions_r["svc"]
                logp_r[t, sl] = lp_rk
                val_r[t, sl] = vr
                priv_r_buf[t, sl] = priv_r
                priv_val_r[t, sl] = pv_r
                step_radars[r] = actions_r

            if singleton_opponent:
                # League member = the singleton: zero jammer 1's executed cells
                # in both the buffer and the env-bound action (idle is always
                # legal). The jammer update is skipped this iteration, so the
                # stale sampled logp for jammer 1 is never consumed.
                act_cell_buf[t, E:] = 0.0
                logp_j[t, E:] = 0.0
                step_jammers[1]["cell"] = torch.zeros_like(step_jammers[1]["cell"])

            j_cell = torch.stack([step_jammers[k]["cell"] for k in range(K)], dim=1)  # [E,K,25]
            j_beam = torch.stack([step_jammers[k]["beam"] for k in range(K)], dim=1)  # [E,K]
            r_beam = torch.stack([step_radars[r]["beam"] for r in range(R)], dim=1)   # [E,R]
            r_svc = torch.stack([step_radars[r]["svc"] for r in range(R)], dim=1)     # [E,R]
            (obs_j, obs_r), (rj, rr), done, info = self.env.step(
                j_cell, j_beam, r_beam, r_svc)
            rew_j[t] = rj.repeat(K)  # team reward, shared by both jammers
            rew_r[t] = rr.repeat(R)  # team reward, shared by both radars

        with torch.no_grad():
            # Use the final post-step observation already returned by step().
            priv_j, priv_r = obs_j.reshape(E, -1), obs_r.reshape(E, -1)
            last_vj = torch.cat([self.jam_critic(obs_j[:, k]) for k in range(K)])
            last_vr = torch.cat([self.rad_critic(obs_r[:, r]) for r in range(R)])
            last_pvj = self.jam_priv_critic(priv_j).repeat(K)
            last_pvr = self.rad_priv_critic(priv_r).repeat(R)

        ones_Bj = torch.ones(Bj, device=device)
        ones_Br = torch.ones(Br, device=device)
        rb_j = RolloutBuffer(
            obs=obsj_buf, mask_base=mask_cell_buf, mask_beam=mask_beam_buf,
            action_base=act_cell_buf, action_beam=act_beam_buf,
            logp=logp_j, reward=rew_j, value=val_j,
            last_value=last_vj, last_done=ones_Bj,
        )
        rb_j.mask_cell = mask_cell_buf
        rb_j.action_cell = act_cell_buf
        rb_j.priv_obs = priv_j_buf
        rb_j.priv_value = priv_val_j
        rb_j.last_priv_value = last_pvj

        rb_r = RolloutBuffer(
            obs=obsr_buf, mask_base=mask_rbeam_buf, mask_beam=mask_rbeam_buf,
            action_base=act_rbeam_buf, action_beam=act_rbeam_buf,
            logp=logp_r, reward=rew_r, value=val_r,
            last_value=last_vr, last_done=ones_Br,
        )
        rb_r.mask_svc = mask_rsvc_buf
        rb_r.action_svc = act_rsvc_buf
        rb_r.priv_obs = priv_r_buf
        rb_r.priv_value = priv_val_r
        rb_r.last_priv_value = last_pvr

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
        use_singleton = (self._mix_den > 0
                         and self._mix_num > 0
                         and (self.iteration % self._mix_den) < self._mix_num)
        rb = self.collect_rollout(singleton_opponent=use_singleton)

        self._swap_to_jammer()
        if use_singleton:
            metrics_j = {}  # jammer trains only on pair-vs-radar iterations
        else:
            metrics_j = self.update(self._rb_j)
        if self.radar_scripted:
            metrics_r = {}  # scripted radar side: nothing to update
        else:
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

    # ---------- checkpointing both teams ----------
    def save_selfplay(self, path: Path):
        """ATOMIC checkpoint: write to a temp file, then os.replace."""
        import os
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "iteration": self.iteration,
            "jam_actor": self.jam_actor.state_dict(),
            "jam_critic": self.jam_critic.state_dict(),
            "jam_priv_critic": self.jam_priv_critic.state_dict(),
            "jam_actor_opt": self.jam_actor_opt.state_dict(),
            "jam_critic_opt": self.jam_critic_opt.state_dict(),
            "jam_priv_critic_opt": self.jam_priv_critic_opt.state_dict(),
            "rad_actor": self.rad_actor.state_dict(),
            "rad_critic": self.rad_critic.state_dict(),
            "rad_priv_critic": self.rad_priv_critic.state_dict(),
            "rad_actor_opt": self.rad_actor_opt.state_dict(),
            "rad_critic_opt": self.rad_critic_opt.state_dict(),
            "rad_priv_critic_opt": self.rad_priv_critic_opt.state_dict(),
        }, tmp)
        os.replace(tmp, path)

    def load_selfplay(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.jam_actor.load_state_dict(ckpt["jam_actor"])
        self.jam_critic.load_state_dict(ckpt["jam_critic"])
        self.jam_priv_critic.load_state_dict(ckpt["jam_priv_critic"])
        self.jam_actor_opt.load_state_dict(ckpt["jam_actor_opt"])
        self.jam_critic_opt.load_state_dict(ckpt["jam_critic_opt"])
        self.jam_priv_critic_opt.load_state_dict(ckpt["jam_priv_critic_opt"])
        self.rad_actor.load_state_dict(ckpt["rad_actor"])
        self.rad_critic.load_state_dict(ckpt["rad_critic"])
        self.rad_priv_critic.load_state_dict(ckpt["rad_priv_critic"])
        self.rad_actor_opt.load_state_dict(ckpt["rad_actor_opt"])
        self.rad_critic_opt.load_state_dict(ckpt["rad_critic_opt"])
        self.rad_priv_critic_opt.load_state_dict(ckpt["rad_priv_critic_opt"])
        self._swap_to_jammer()
        # Restore the counter too: without this a resumed session re-reports
        # iterations 0..N into the metrics logs (duplicate segments) and
        # restarts the entropy-anneal schedule from full exploration.
        self.iteration = int(ckpt["iteration"])
        return self.iteration


# ---------- evaluation ----------

def evaluate_s7(
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
    """Four-view evaluation of the 2v2 game.

    h2h       — 2 learned jammers vs 2 learned radars (the game itself)
    jam_only  — 2 learned jammers vs scripted sweep radars (raw jammer power)
    rad_only  — 2 learned radars vs idle jammers (radar competence floor)
    j1_only   — jammer 0 learned + jammer 1 idle vs learned radars
                (the S7-env 1v2 control: quantifies the second jammer's
                marginal disruptive power at equal per-jammer leverage)
    """
    from env.gpu.array_face_s7 import ArrayFaceS7VecEnv
    from dataclasses import replace

    # The evaluator reports lane 0 only. Use one lane instead of simulating
    # fifteen discarded copies; scenario generation for lane 0 is unchanged.
    eval_env_cfg = replace(env_cfg, n_envs=1)

    def run(seed: int, mode: str) -> tuple[float, float]:
        env = ArrayFaceS7VecEnv(eval_env_cfg, physics=physics, radar=radar, jammer=jammer)
        env.reset(seed=seed)
        gen = torch.Generator(device=device).manual_seed(action_seed + seed)
        E, K, R = eval_env_cfg.n_envs, eval_env_cfg.n_jammers, N_RADARS
        for t in range(env_cfg.horizon):
            obs_j, obs_r = env._build_observation()
            mask_cell, mask_beam = env._compute_masks()
            if mode in ("h2h", "jam_only", "j1_only"):
                cells, beams = [], []
                for k in range(K):
                    with torch.no_grad():
                        a_j, _ = sample_multihead(
                            jam_actor, obs_j[:, k],
                            {"cell": mask_cell[:, k], "beam": mask_beam[:, k]}, gen)
                    cells.append(a_j["cell"])
                    beams.append(a_j["beam"])
                jcell = torch.stack(cells, dim=1)   # [E,K,25]
                jbeam = torch.stack(beams, dim=1)   # [E,K]
                if mode == "j1_only":
                    jcell[:, 1:] = 0.0              # all jammers but slot 0 idle
                    jbeam[:, 1:] = 0                # ("j1-of-n" for n > 2)
            else:  # both jammers idle
                jcell = torch.zeros(E, K, 25, device=device)
                jbeam = torch.zeros(E, K, dtype=torch.int64, device=device)
            if mode in ("h2h", "rad_only", "j1_only"):
                rb_ = torch.zeros(E, R, dtype=torch.int64, device=device)
                rs_ = torch.zeros(E, R, dtype=torch.int64, device=device)
                for r in range(R):
                    masks = {"beam": env._radar_mask_beam,
                             "svc": env._radar_mask_svc}
                    with torch.no_grad():
                        a_r, _ = sample_multihead(rad_actor, obs_r[:, r], masks, gen)
                    rb_[:, r] = a_r["beam"]
                    rs_[:, r] = a_r["svc"]
            else:  # scripted sweep (both radars): S4-style deterministic cycling
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
        views[mode] = {"mean_drop": sum(drops) / len(drops), "mean_success": sum(succs) / len(succs)}
    return views
