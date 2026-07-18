"""Fixed-opponent PPO trainer for two-team env (WP1 BR training).

Per plan snuggly-exploring-parrot.md Step 3.

Design:
  - Learning team (default team 0): TwoTeamCommanderActorCritic, trained via PPO
  - Frozen opponent (default team 1): any commander with get_action(env, team) API
                                       (StrongRule, ExtremeCommander, or another AC in eval mode)
  - Rollout: H steps × E envs, collect per-step transitions for learning team only
  - GAE: dual trunk (central + local) → α_eff blend → advantage
  - α_eff = α_max · exp(−β · priv[:, 4])  (priv[:,4] is normalized trace_P from WP0)
  - PPO update with clipped surrogate + clipped value loss (both critics)

Reusable template: algo/_shared/pilot/taes/run_wp2.py::JammerPPOTrainer + taes_ppo.py
"""

from __future__ import annotations
import os
import sys
import time
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Callable

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


class _RolloutBuffer:
    """Per-team rollout buffer for BR training (team axis absent — only learning team).

    WP-3 M1: removed `beam_target` (god-view, was killed in M0 actor-critic).
    Added `beam_direction[H,E,R]` (continuous azimuth ∈ [-π,π]) and
    `detect_list[H,E,K_max,5]` (detection encoder input from env.get_detect_list()).
    """

    def __init__(self, horizon: int, n_envs: int, obs_dim: int, priv_dim: int,
                 n_aperture: int, n_fn: int, k_max: int = 5,
                 device: str = "cuda"):
        self.H = horizon
        self.E = n_envs
        self.dev = device
        self.obs = torch.zeros(horizon, n_envs, obs_dim, device=device)
        self.priv = torch.zeros(horizon, n_envs, priv_dim, device=device)
        # Actions
        self.task_alloc = torch.zeros(horizon, n_envs, n_aperture, n_fn, device=device)
        # WP-3 M0/M1: beam_direction (continuous azimuth ∈ [-π,π], Beta head)
        self.beam_direction = torch.zeros(horizon, n_envs, n_aperture, device=device)
        self.laser_target = torch.zeros(horizon, n_envs, dtype=torch.long, device=device)
        self.emission_on = torch.zeros(horizon, n_envs, n_aperture, device=device)
        self.freq_hop_rate = torch.zeros(horizon, n_envs, n_aperture, device=device)   # FIX 1
        self.channel_select = torch.zeros(horizon, n_envs, n_aperture, dtype=torch.long, device=device)   # WP-C R3
        # WP-3 M0/M1: detection list for DeepSets encoder
        self.detect_list = torch.zeros(horizon, n_envs, k_max, 5, device=device)
        # PPO bookkeeping
        self.log_prob = torch.zeros(horizon, n_envs, device=device)
        self.value = torch.zeros(horizon, n_envs, device=device)
        self.value_local = torch.zeros(horizon, n_envs, device=device)
        self.reward = torch.zeros(horizon, n_envs, device=device)
        self.done = torch.zeros(horizon, n_envs, device=device)
        # Computed in _compute_gae
        self.advantage = torch.zeros(horizon, n_envs, device=device)
        self.ret = torch.zeros(horizon, n_envs, device=device)


class TwoTeamBRTrainer:
    """Fixed-opponent PPO trainer for two-team learning commander (WP1 BR)."""

    def __init__(
        self,
        br_ac,
        frozen_opponent,
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
        target_kl: float = 0.03,
        alpha_eff_alpha_max: float = 0.5,
        alpha_eff_beta: float = 2.0,
        reward_scale: float = 0.1,   # divide per-step reward (env cumulative-kill makes magnitude large)
        value_huber_delta: float = 1.0,   # use Huber for value loss (robust to large returns)
        lr_decay_iters: int = 0,   # >0 enables cosine annealing over n_iterations
        lr_decay_min_frac: float = 0.1,   # min LR = lr_init * frac
        entropy_coef_min: float = 0.001,   # WP-3 M1: cosine anneal floor
        entropy_decay_iters: int = 0,   # >0 enables cosine anneal over n_iterations
        shape_track_bonus: float = 0.0,   # WP-3 dense reward: per-step bonus per radar tracked (tau_track)
        shape_exposure_penalty: float = 0.0,   # WP-3 dense reward: per-step penalty × exposure
        # WP-3.1 Fix A: dense dwell progress + kill bonus (per WP3_1_KILL_FIX_SPEC §1.2)
        # Without these, optimal policy under non-zero-sum shaping is "never fire":
        # track_bonus rewards tracking, exposure_penalty punishes laser emission,
        # no term rewards the dwell→kill chain.
        shape_dwell_bonus: float = 0.0,   # per-step bonus × Σ_radar_E[:,enemy]/e_kill (dwell progress)
        shape_kill_bonus: float = 0.0,   # per new kill (delta info["team_kills"][:,learning_team])
        # WP-3.1 Fix D1: active-perception shaping (track acquisition chain).
        # All three target the "tracker_initialized gate" wall: without them, RL gets
        # zero reward gradient for acquiring tracks (dwell+kill+track shaping all
        # gate on tracker_initialized → all zero when tracker fails to init vs BC).
        shape_init_bonus: float = 0.0,        # per new slot tracker_init False→True
        shape_detect_in_beam_bonus: float = 0.0,  # per detection falling inside beam_direction HPBW
        shape_belief_bonus: float = 0.0,      # per-step potential: -trace_P (lower cov = higher reward)
        # WP-3.1 Fix D2: reverse curriculum (Florensa CoRL 2017). Warm-start
        # tracker for fraction of envs at episode start; anneal p → 0 over training.
        curriculum_p_start: float = 0.0,       # 0 = disabled; 1.0 = always warm-start at iter 0
        curriculum_anneal_iters: int = 0,      # iters to anneal p from curriculum_p_start → 0
        # WP-3.1 Fix C: hold entropy_coef at max until first kill observed (spec §3).
        entropy_gate_on_kill: bool = False,
        device: str = "cuda",
    ):
        self.br_ac = br_ac
        self.frozen_opponent = frozen_opponent
        self.lr_actor = float(lr_actor)
        self.lr_critic = float(lr_critic)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip = float(clip)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.n_epochs = int(n_epochs)
        self.mb = int(minibatch_size)
        self.target_kl = float(target_kl)
        self.alpha_max = float(alpha_eff_alpha_max)
        self.beta = float(alpha_eff_beta)
        self.reward_scale = float(reward_scale)
        self.value_huber_delta = float(value_huber_delta)
        self.lr_decay_iters = int(lr_decay_iters)
        self.lr_decay_min_frac = float(lr_decay_min_frac)
        self.entropy_coef_max = float(entropy_coef)
        self.entropy_coef_min = float(entropy_coef_min)
        self.entropy_decay_iters = int(entropy_decay_iters)
        self.shape_track_bonus = float(shape_track_bonus)
        self.shape_exposure_penalty = float(shape_exposure_penalty)
        self.shape_dwell_bonus = float(shape_dwell_bonus)
        self.shape_kill_bonus = float(shape_kill_bonus)
        # WP-3.1 Fix D1: active-perception shaping (Kreucher-Hero; track acquisition).
        self.shape_init_bonus = float(shape_init_bonus)
        self.shape_detect_in_beam_bonus = float(shape_detect_in_beam_bonus)
        self.shape_belief_bonus = float(shape_belief_bonus)
        # WP-3.1 Fix D2: reverse curriculum.
        self.curriculum_p_start = float(curriculum_p_start)
        self.curriculum_anneal_iters = int(curriculum_anneal_iters)
        # WP-3.1 Fix C: kill-appeared flag (latch). entropy_gate_on_kill=True 时
        # _entropy_coef 在 kill>0 出现前不退火。
        self._kill_appeared = False
        self.entropy_gate_on_kill = bool(entropy_gate_on_kill)
        self.device = torch.device(device)

        # Separate actor/critic LRs via param groups.
        # WP-3 M0/M1: actor heads now include beam_direction_head (replaces removed
        # beam_target_head) and detect_mlp (DeepSets encoder for env detection list).
        actor_params = list(br_ac.actor_trunk.parameters()) + \
                       list(br_ac.task_alloc_head.parameters()) + \
                       list(br_ac.beam_direction_head.parameters()) + \
                       list(br_ac.laser_target_head.parameters()) + \
                       list(br_ac.emission_on_head.parameters()) + \
                       list(br_ac.freq_hop_head.parameters()) + \
                       list(br_ac.channel_select_head.parameters()) + \
                       list(br_ac.detect_mlp.parameters())
        critic_params = list(br_ac.central_trunk.parameters()) + \
                        list(br_ac.local_trunk.parameters())
        self.opt = torch.optim.Adam([
            {"params": actor_params, "lr": lr_actor},
            {"params": critic_params, "lr": lr_critic},
        ])

    # ------------------------------------------------------------------
    # WP-3.1 Fix D1: active-perception helpers
    # ------------------------------------------------------------------
    def _curriculum_p(self, iter_idx: int) -> float:
        """WP-3.1 Fix D2: reverse curriculum probability.

        Linear anneal from curriculum_p_start (at iter 0) → 0 (at iter_anneal).
        Returns 0 after anneal complete (pure cold-start phase).
        """
        if self.curriculum_anneal_iters <= 0 or self.curriculum_p_start <= 0.0:
            return 0.0
        if iter_idx >= self.curriculum_anneal_iters:
            return 0.0
        frac = 1.0 - iter_idx / max(1, self.curriculum_anneal_iters)
        return self.curriculum_p_start * frac

    def _trace_P_mean(self, env, learning_team: int) -> torch.Tensor:
        """Mean trace_P across learning_team radars. Lower = tighter track."""
        trace_P_t = env.tracker_P[:, learning_team, :, 0, 0] + \
                    env.tracker_P[:, learning_team, :, 2, 2]   # [E, R]
        return trace_P_t.mean(dim=-1)   # [E]

    def _n_detects_in_beam(self, env, learning_team: int, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Count real (non-FA) detections whose bearing from a learning_team radar
        falls inside that radar's beam_direction HPBW.

        Args:
            env: TwoTeamVecEnv
            learning_team: 0 or 1
            action: action dict from br_ac; must contain `beam_direction` [E, R]

        Returns:
            [E] tensor of normalized count ∈ [0, K_max]. Per real detection inside
            any beam, contributes 1/K_max (so max = K_max detections all in beam = 1.0).
        """
        E = env.E
        R = env.n_radars_per_team
        K = env.k_max
        dev = env.device
        beam_dir = action["beam_direction"]   # [E, R] azimuth ∈ [-π, π]
        hpbw = float(env.beam_width_rad)      # half-power beam width (rad)
        radar_pos = env.radar_pos[:, learning_team]   # [E, R, 2]

        detect_list = env.get_detect_list()[:, learning_team]   # [E, K, 5]
        # z_x, z_y cartesian; is_fa at idx 3
        z = detect_list[..., 0:2]   # [E, K, 2]
        is_fa = detect_list[..., 3]   # [E, K]
        # Real detection mask
        real_mask = (is_fa < 0.5) & (detect_list[..., 4] > 0.5)   # [E, K]

        # For each (E, R, K): bearing from radar r to detection k
        # radar_pos[E, R, 1, 2] - z[E, 1, K, 2] → bearing diff
        rel = z.unsqueeze(1) - radar_pos.unsqueeze(2)   # [E, R, K, 2]
        bearing = torch.atan2(rel[..., 1], rel[..., 0])   # [E, R, K]
        # beam_dir [E, R] → broadcast to [E, R, 1]
        bd = beam_dir.unsqueeze(-1)   # [E, R, 1]
        # angular diff wrapped to [-π, π]
        ang_diff = (bearing - bd + math.pi) % (2 * math.pi) - math.pi
        in_beam = (ang_diff.abs() < hpbw / 2.0)   # [E, R, K]
        # Only count real detections; per (E, K), if any radar r has it in beam, count once
        real_in_beam_per_radar = in_beam & real_mask.unsqueeze(1)   # [E, R, K]
        # [E, K] → any radar → sum over K → [E]
        n_in_beam = real_in_beam_per_radar.any(dim=1).float().sum(dim=-1)   # [E]
        return n_in_beam / max(K, 1)   # normalize to [0, 1]

    def _entropy_coef(self, iter_idx: int, n_iters: int) -> float:
        """Cosine-anneal entropy_coef from entropy_coef_max → entropy_coef_min.

        WP-3.1 Fix C (spec §3): 若 `entropy_gate_on_kill=True`,在 kill>0 出现前
        不退火(保持 entropy_coef_max),让 RL 在没拿到 kill 信号前持续 explore;
        kill 出现后才按 cosine 衰减到 min。
        """
        if self.entropy_decay_iters <= 0:
            return self.entropy_coef_max
        # Fix C gate: hold max until first kill observed.
        if getattr(self, "entropy_gate_on_kill", False) and not self._kill_appeared:
            return self.entropy_coef_max
        frac = 0.5 * (1 + math.cos(math.pi * iter_idx / max(1, n_iters)))
        return self.entropy_coef_min + (self.entropy_coef_max - self.entropy_coef_min) * frac

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_rollout(self, env, horizon: int, learning_team: int = 0,
                        iter_idx: int = 0) -> _RolloutBuffer:
        """Run `horizon` env steps; collect learning_team transitions."""
        E = env.E
        dev = self.device
        buf = _RolloutBuffer(
            horizon, E, env.obs_dim, env.privileged_dim,
            env.n_radars_per_team, env.n_fn,
            k_max=getattr(env, "k_max", 5), device=dev)

        # Reset env
        obs_dict = env.reset()
        # WP-3.1 Fix D2: reverse curriculum — warm-start tracker for fraction of envs.
        p_cur = self._curriculum_p(iter_idx)
        if p_cur > 0.0 and hasattr(env, "warm_start_tracker"):
            env.warm_start_tracker(learning_team, p=p_cur)
        # WP-3.1 Fix A: baseline team_kills for delta-kill bonus
        et = 1 - learning_team   # enemy team; radar_E[:,et] = learning_team's dwell progress on enemy
        prev_kills = env.team_kills[:, learning_team].clone()
        # WP-3.1 Fix D1: prev tracker_init mask for delta-init bonus
        prev_init = env.tracker_initialized[:, learning_team].clone()   # [E, R] bool
        # WP-3.1 Fix D1: prev trace_P mean for potential-based belief shaping
        prev_trace_P_mean = self._trace_P_mean(env, learning_team)   # [E]
        for t in range(horizon):
            obs_lt = obs_dict["obs"][:, learning_team]            # [E, obs_dim]
            priv_lt = obs_dict["privileged"][:, learning_team]    # [E, priv_dim]
            # WP-3 M0/M1: detection list (post-step; zeros before first step)
            detect_lt = env.get_detect_list()[:, learning_team]   # [E, K_max, 5]
            # Learning team acts (sampled)
            br_action, br_logp, br_value, br_value_local = self.br_ac(
                obs_lt, detect_lt, priv_lt)
            # Frozen opponent acts
            opp_team = 1 - learning_team
            opp_action = self.frozen_opponent.get_action(env, opp_team)
            # Combine — order depends on which team is learning
            if learning_team == 0:
                action = combine_team_actions(env, br_action, opp_action)
            else:
                action = combine_team_actions(env, opp_action, br_action)

            obs_dict, reward, done, info = env.step(action)
            rew_lt = reward[:, learning_team] * self.reward_scale   # [E]

            # WP-3.1 Fix A: dense dwell progress (laser dwell → e_kill chain).
            # Absolute form (robust; radar_E saturates at e_kill, can't be farmed).
            # spec §1.2 (A): dwell_frac = Σ_r radar_E[:,et]/e_kill
            if self.shape_dwell_bonus > 0.0:
                dwell_frac = env.radar_E[:, et].sum(dim=-1) / env.e_kill   # [E]
                rew_lt = rew_lt + self.shape_dwell_bonus * dwell_frac
            # WP-3.1 Fix A: explicit kill bonus (large one-shot per new kill).
            # spec §1.2 (B): kill_bonus = shape_kill_bonus × Δteam_kills[:,learning_team]
            if self.shape_kill_bonus > 0.0:
                now_kills = info["team_kills"][:, learning_team]
                new_kills = (now_kills - prev_kills).clamp(min=0).float()
                rew_lt = rew_lt + self.shape_kill_bonus * new_kills
                prev_kills = now_kills.clone()
                # WP-3.1 Fix C: kill-appeared flag (latch on first new_kill).
                # spec §3: "kill>0 出现前不退火"。
                if not self._kill_appeared and float(new_kills.sum().item()) > 0.5:
                    self._kill_appeared = True

            # WP-3 dense reward shaping (non-zero-sum, learning team only).
            # Mitigates mirror-symmetric pool where zero-sum env reward → 0 gradient.
            if self.shape_track_bonus > 0.0:
                trace_P_t = env.tracker_P[:, learning_team, :, 0, 0] + \
                            env.tracker_P[:, learning_team, :, 2, 2]
                n_tracked = ((trace_P_t < env.tau_track) &
                             env.tracker_initialized[:, learning_team]).float().sum(dim=-1)
                rew_lt = rew_lt + self.shape_track_bonus * n_tracked
            if self.shape_exposure_penalty > 0.0:
                exp_lt = info["exposure"][:, learning_team]
                rew_lt = rew_lt - self.shape_exposure_penalty * exp_lt

            # WP-3.1 Fix D1: active-perception shaping (Kreucher-Hero track acquisition).
            # All three break the "tracker_initialized gate" wall — they give RL dense
            # reward for *acquiring* tracks, not just for downstream dwell/kill that
            # gates on tracker_init and thus zero-gradient when tracker fails vs BC.
            #
            # (D1.a) shape_init_bonus: per slot False→True transition (event-based).
            if self.shape_init_bonus > 0.0:
                now_init = env.tracker_initialized[:, learning_team]   # [E, R] bool
                new_inits = (now_init & ~prev_init).float().sum(dim=-1)   # [E]
                rew_lt = rew_lt + self.shape_init_bonus * new_inits
                prev_init = now_init.clone()
            # (D1.b) shape_belief_bonus: potential-based -Φ where Φ = mean trace_P.
            # Lower cov (tighter track) = higher reward; PBRS-style policy-invariant.
            if self.shape_belief_bonus > 0.0:
                now_trace_P_mean = self._trace_P_mean(env, learning_team)   # [E]
                # ΔΦ = Φ_post - Φ_prev = -(now - prev); reward = -ΔΦ = (now - prev)↓ → +
                delta = prev_trace_P_mean - now_trace_P_mean   # [E]; + when cov shrinks
                rew_lt = rew_lt + self.shape_belief_bonus * delta.clamp(min=0.0)
                prev_trace_P_mean = now_trace_P_mean.clone()
            # (D1.c) shape_detect_in_beam_bonus: per real detection falling inside
            # any learning_team beam_direction HPBW. Encourages beam to point at RF
            # sources even when tracker hasn't locked yet (chicken-egg breaker).
            if self.shape_detect_in_beam_bonus > 0.0:
                rew_lt = rew_lt + self.shape_detect_in_beam_bonus * \
                    self._n_detects_in_beam(env, learning_team, br_action)

            # Record
            buf.obs[t] = obs_lt
            buf.priv[t] = priv_lt
            buf.task_alloc[t] = br_action["task_alloc"]
            buf.beam_direction[t] = br_action["beam_direction"]
            buf.laser_target[t] = br_action["laser_target"]
            buf.emission_on[t] = br_action["emission_on"]
            buf.freq_hop_rate[t] = br_action["freq_hop_rate"]   # FIX 1
            buf.channel_select[t] = br_action["channel_select"]   # WP-C R3
            buf.detect_list[t] = detect_lt
            buf.log_prob[t] = br_logp
            buf.value[t] = br_value
            buf.value_local[t] = br_value_local
            buf.reward[t] = rew_lt
            buf.done[t] = done.float()

            if done.all():
                obs_dict = env.reset()
                # WP-3.1 Fix D2: re-apply curriculum warm-start after mid-rollout reset.
                if p_cur > 0.0 and hasattr(env, "warm_start_tracker"):
                    env.warm_start_tracker(learning_team, p=p_cur)
                # WP-3.1 Fix A: resync kill baseline after env reset
                prev_kills = env.team_kills[:, learning_team].clone()
                # WP-3.1 Fix D1: resync prev tracker_init mask + trace_P baseline
                prev_init = env.tracker_initialized[:, learning_team].clone()
                prev_trace_P_mean = self._trace_P_mean(env, learning_team)

        return buf

    # ------------------------------------------------------------------
    # GAE + α_eff blend
    # ------------------------------------------------------------------
    def _compute_gae(self, buf: _RolloutBuffer):
        """Compute central + local GAE, then α_eff blend."""
        H, E = buf.H, buf.E
        dev = self.device

        # Central GAE
        adv_team = torch.zeros(E, device=dev)
        for t in reversed(range(H)):
            non_term = 1.0 - buf.done[t]
            next_v = buf.value[t + 1] if t + 1 < H else torch.zeros(E, device=dev)
            delta = buf.reward[t] + self.gamma * next_v * non_term - buf.value[t]
            adv_team = delta + self.gamma * self.gae_lambda * non_term * adv_team
            buf.ret[t] = adv_team + buf.value[t]
        A_team = buf.ret - buf.value   # [H, E]

        # Local GAE (uses value_local)
        adv_agent = torch.zeros(E, device=dev)
        ret_local = torch.zeros_like(buf.ret)
        for t in reversed(range(H)):
            non_term = 1.0 - buf.done[t]
            next_v = buf.value_local[t + 1] if t + 1 < H else torch.zeros(E, device=dev)
            delta = buf.reward[t] + self.gamma * next_v * non_term - buf.value_local[t]
            adv_agent = delta + self.gamma * self.gae_lambda * non_term * adv_agent
            ret_local[t] = adv_agent + buf.value_local[t]
        A_agent = ret_local - buf.value_local   # [H, E]

        # α_eff blend (per WP0 priv[:,4] = normalized trace_P)
        # ASSERT: α_eff bug guard (memory: twoteam_multifunction_pivot)
        priv_4_max = float(buf.priv[..., 4].max().item())
        assert priv_4_max < 100.0, (
            f"priv[:, 4] max = {priv_4_max:.1f} — looks like raw trace_P, not normalized. "
            f"This is the α_eff bug."
        )
        alpha_eff = self.alpha_max * torch.exp(-self.beta * buf.priv[..., 4])   # [H, E]
        buf.advantage = (1 - alpha_eff) * A_agent + alpha_eff * A_team   # [H, E]

        # Global normalization
        adv_flat = buf.advantage.reshape(-1)
        buf.advantage = (buf.advantage - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self, buf: _RolloutBuffer, iter_idx: int = 0, n_iters: int = 0) -> Dict[str, float]:
        dev = self.device
        H, E = buf.H, buf.E
        N = H * E

        # Flatten
        obs_flat = buf.obs.reshape(N, -1)
        priv_flat = buf.priv.reshape(N, -1)
        task_flat = buf.task_alloc.reshape(N, self.br_ac.n_aperture, self.br_ac.n_fn)
        beam_flat = buf.beam_direction.reshape(N, self.br_ac.n_aperture)
        laser_flat = buf.laser_target.reshape(N,)
        emit_flat = buf.emission_on.reshape(N, self.br_ac.n_aperture)
        fh_flat = buf.freq_hop_rate.reshape(N, self.br_ac.n_aperture)   # FIX 1
        chan_flat = buf.channel_select.reshape(N, self.br_ac.n_aperture)   # WP-C R3
        detect_flat = buf.detect_list.reshape(N, -1, 5)   # [N, K_env, 5] (AC mean-pools over K)
        action_flat = {
            "task_alloc": task_flat,
            "beam_direction": beam_flat,
            "laser_target": laser_flat,
            "emission_on": emit_flat,
            "freq_hop_rate": fh_flat,
            "channel_select": chan_flat,
        }
        lp_old = buf.log_prob.reshape(N)
        v_old = buf.value.reshape(N)
        v_loc_old = buf.value_local.reshape(N)
        ret_flat = buf.ret.reshape(N)
        adv_flat = buf.advantage.reshape(N)

        # WP-3 M1: cosine-annealed entropy coef
        ent_coef = self._entropy_coef(iter_idx, max(n_iters, 1))

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                   "approx_kl": 0.0, "clip_frac": 0.0, "entropy_coef": ent_coef}
        n_updates = 0
        early_stop = False
        for epoch in range(self.n_epochs):
            idx = torch.randperm(N, device=dev)
            for i in range(0, N, self.mb):
                b = idx[i:i + self.mb]
                if b.numel() < 8:
                    continue
                obs_b = obs_flat[b]
                priv_b = priv_flat[b]
                detect_b = detect_flat[b]
                act_b = {k: v[b] for k, v in action_flat.items()}

                lp_new, v_new, vl_new, entropy = self.br_ac.evaluate_actions(
                    obs_b, detect_b, act_b, priv_b)
                ratio = torch.exp(lp_new - lp_old[b])
                adv_b = adv_flat[b]
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss (central + local) — Huber for robustness to large returns
                v_clipped = v_old[b] + (v_new - v_old[b]).clamp(-self.clip, self.clip)
                huber_delta = self.value_huber_delta
                # Huber: 0.5*x^2 if |x|<d else d*(|x|-0.5*d)
                def huber(x, d=huber_delta):
                    abs_x = x.abs()
                    quad = torch.where(abs_x < d, 0.5 * x.pow(2), torch.zeros_like(x))
                    lin = torch.where(abs_x >= d, d * (abs_x - 0.5 * d), torch.zeros_like(x))
                    return (quad + lin).mean()
                v_loss_central = huber(v_new - ret_flat[b])
                v_loss_clipped = huber(v_clipped - ret_flat[b])
                value_loss = torch.max(v_loss_central, v_loss_clipped)
                # Local critic
                value_loss_local = huber(vl_new - ret_flat[b])
                value_loss = value_loss + value_loss_local

                loss = policy_loss - ent_coef * entropy.mean() \
                       + self.value_coef * value_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                params = list(self.br_ac.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (lp_old[b] - lp_new).mean()
                    clip_frac = ((ratio - 1).abs() > self.clip).float().mean()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["approx_kl"] += approx_kl.item()
                metrics["clip_frac"] += clip_frac.item()
                n_updates += 1

            # Early-stop on KL blow-up
            if metrics["approx_kl"] / max(1, n_updates) > 1.5 * self.target_kl:
                early_stop = True
                break

        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            metrics[k] /= max(1, n_updates)
        metrics["early_stop"] = float(early_stop)
        return metrics

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def train(self, env, n_iterations: int, horizon: int = 300,
              learning_team: int = 0, save_path: Optional[str] = None,
              log_every: int = 10, log_history: Optional[list] = None):
        """Train BR commander against frozen_opponent. Returns history list."""
        if log_history is None:
            log_history = []
        t0 = time.time()
        # Cosine LR decay (helps Dirichlet concentration in high-dim action space).
        # Scale from initial LRs each iter — not from current — so the schedule is
        # deterministic regardless of any external LR modifications.
        init_lrs = [g["lr"] for g in self.opt.param_groups]
        decay_total = max(self.lr_decay_iters, n_iterations) if self.lr_decay_iters > 0 else n_iterations
        for it in range(n_iterations):
            if self.lr_decay_iters > 0:
                frac = 0.5 * (1 + math.cos(math.pi * it / decay_total))
                lr_scale = self.lr_decay_min_frac + (1 - self.lr_decay_min_frac) * frac
                for g, lr0 in zip(self.opt.param_groups, init_lrs):
                    g["lr"] = lr0 * lr_scale
            buf = self.collect_rollout(env, horizon, learning_team=learning_team)
            self._compute_gae(buf)
            metrics = self.update(buf, iter_idx=it, n_iters=n_iterations)

            # Health monitor
            r_mean = buf.reward.mean().item()
            kills_per_ep = float("nan")   # filled if info available — collect_rollout doesn't return info
            adv_std = buf.advantage.std().item()
            entropy = metrics["entropy"]
            kl = metrics["approx_kl"]
            elapsed = (time.time() - t0) / 60.0

            log_history.append({
                "iter": it, "reward_mean": r_mean, "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"], "entropy": entropy,
                "approx_kl": kl, "adv_std": adv_std, "clip_frac": metrics["clip_frac"],
                "early_stop": metrics["early_stop"], "elapsed_min": elapsed,
            })

            if it % log_every == 0 or it == n_iterations - 1:
                print(f"  [BR] it={it:3d}/{n_iterations} r={r_mean:+.3f} "
                      f"v_loss={metrics['value_loss']:.3f} "
                      f"pi_loss={metrics['policy_loss']:+.3f} "
                      f"ent={entropy:+.3f} kl={kl:.4f} adv_std={adv_std:.2f} "
                      f"clip={metrics['clip_frac']:.2f} "
                      f"es={int(metrics['early_stop'])} t={elapsed:.1f}min",
                      flush=True)

            # NaN / blowup guard
            if any(torch.isnan(p).any().item() for p in self.br_ac.parameters()):
                print(f"  ❌ NaN detected in AC params at iter {it}, aborting", flush=True)
                break
            if not (0.1 < adv_std < 100.0):
                print(f"  ⚠️ adv_std={adv_std:.2f} outside [0.1, 100] at iter {it}",
                      flush=True)

            # Periodic save
            if save_path and ((it + 1) % 50 == 0 or it == n_iterations - 1):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "ac_state": self.br_ac.state_dict(),
                    "iter": it + 1,
                    "metrics_recent": log_history[-5:],
                }, save_path)

        return log_history
