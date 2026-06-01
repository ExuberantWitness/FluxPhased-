"""PPO trainer for Commander and Radar actor-critic networks.

Handles:
- Rollout collection from MFARVecEnv
- GAE computation
- Clipped surrogate loss + value loss + entropy bonus
- Gradient clipping and optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

from .buffer import RolloutBuffer
from .actor_critic import CommanderActorCritic, RadarActorCritic, build_team_state
from .reward_shaping import DenseRewardShaper


class PPOTrainer:
    """PPO training loop for one agent (commander or radar)."""

    def __init__(
        self,
        actor_critic: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: str = "cuda",
    ):
        self.ac = actor_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

    def update(self, buffer: RolloutBuffer,
               team_critic=None, alpha: float = 0.0, beta_kl: float = 0.0,
               team_critic_optimizer=None) -> Dict[str, float]:
        """Run PPO update on collected rollouts.

        Args:
            buffer: RolloutBuffer with computed returns/advantages.
            team_critic: optional TeamCritic for hierarchical advantage blend.
            alpha: team advantage weight (0→1 over training).
            beta_kl: KL penalty weight for domain-shift mitigation.
            team_critic_optimizer: optional optimizer for TeamCritic parameters.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_privileged_value_loss = 0.0
        total_team_value_loss = 0.0
        total_kl_penalty = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in buffer.get_minibatches(self.batch_size):
                obs = batch["obs"]
                old_actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                privileged_info = batch.get("privileged_info")
                team_states = batch.get("team_states")
                team_returns = batch.get("team_returns")

                # ── TeamCritic α-blend advantage ──
                # Hierarchical advantage: A_final = (1-α)*A_agent + α*A_team
                # TeamCritic sees global state (missile positions, task allocation,
                # alive status) → can predict long-horizon kill rewards that
                # per-agent local critics miss.
                if team_critic is not None and alpha > 0 and team_states is not None:
                    team_value = team_critic(team_states)  # [B, 1]
                    team_adv = team_returns.unsqueeze(-1) - team_value.detach()
                    # Normalize team advantage within batch
                    team_adv = (team_adv - team_adv.mean()) / (team_adv.std() + 1e-8)
                    advantages = (1.0 - alpha) * advantages.unsqueeze(-1) + alpha * team_adv
                    advantages = advantages.squeeze(-1)
                    # TeamCritic value loss (trained alongside agent critics)
                    team_value_loss = ((team_value.squeeze(-1) - team_returns) ** 2).mean()
                    total_team_value_loss += team_value_loss.item()
                    # ── TeamCritic optimizer step ──
                    if team_critic_optimizer is not None:
                        team_critic_optimizer.zero_grad()
                        team_value_loss.backward()
                        nn.utils.clip_grad_norm_(
                            team_critic.parameters(), self.max_grad_norm,
                        )
                        team_critic_optimizer.step()

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                log_prob, entropy, value, privileged_value = self.ac.evaluate_actions(
                    obs, old_actions, privileged_info=privileged_info,
                )

                ratio = torch.exp(log_prob - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Deployment value head
                value_loss = ((value.squeeze() - returns) ** 2).mean()

                # Privileged critic
                privileged_value_loss = ((privileged_value.squeeze() - returns) ** 2).mean()

                # Student distillation
                distill_loss = ((value.squeeze() - privileged_value.squeeze().detach()) ** 2).mean()

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.value_coef * privileged_value_loss
                    + 0.1 * self.value_coef * distill_loss
                    + self.entropy_coef * entropy_loss
                )

                # ── KL penalty (domain-shift mitigation) ──
                # Only active when pretrain_log_probs are filled (BC Config A).
                # pretrain_log_probs defaults to zero; KL activates only with
                # meaningful BC-pretrained log-prob values.
                pretrain_lp = batch.get("pretrain_log_probs")
                if beta_kl > 0 and pretrain_lp is not None and pretrain_lp.abs().sum() > 0:
                    kl = (pretrain_lp - log_prob).mean()  # KL approx
                    loss = loss + beta_kl * kl
                    total_kl_penalty += kl.item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_privileged_value_loss += privileged_value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        metrics = {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "privileged_value_loss": total_privileged_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }
        if total_team_value_loss > 0:
            metrics["team_value_loss"] = total_team_value_loss / max(n_updates, 1)
        if total_kl_penalty > 0:
            metrics["kl_penalty"] = total_kl_penalty / max(n_updates, 1)
        return metrics


class TeamPPOTrainer:
    """Manages PPO training for a full team (1 commander + shared radar policy).

    The radars on the same team share a single RadarActorCritic network,
    doubling effective sample efficiency.

    Usage:
        trainer = TeamPPOTrainer(commander, radar, ...)
        trainer.init_buffers(env.state_dim, env.action_dim)
        own = trainer.get_own_actions(env, team)
        result = env.step(actions, cmd_actions)
        trainer.store_transition(env, result, own["transition"], team)
        if buffer full: trainer.update()
    """

    def __init__(
        self,
        commander: CommanderActorCritic,
        radar: RadarActorCritic,
        commander_lr: float = 3e-4,
        radar_lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        commander_clip: float = 0.2,
        radar_clip: float = 0.1,
        commander_entropy: float = 0.01,
        radar_entropy: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        buffer_size_commander: int = 0,
        buffer_size_radar: int = 0,
        device: str = "cuda",
        stealth_weight: float = 0.1,
        reward_shaping_config: dict = None,
    ):
        # Commander buffer is tiny (obs=76, act=35) so it can always be
        # large.  Radar buffer is huge for 25×25 (obs=163783, act=13753)
        # so allow a separate, smaller cap to avoid blowing CPU RAM.
        cmd_buf = buffer_size_commander if buffer_size_commander > 0 else buffer_size
        rad_buf = buffer_size_radar if buffer_size_radar > 0 else buffer_size

        self.commander_trainer = PPOTrainer(
            commander, lr=commander_lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=commander_clip, entropy_coef=commander_entropy,
            value_coef=value_coef, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
            buffer_size=cmd_buf, device=device,
        )
        self.radar_trainer = PPOTrainer(
            radar, lr=radar_lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=radar_clip, entropy_coef=radar_entropy,
            value_coef=value_coef, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
            buffer_size=rad_buf, device=device,
        )
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer_size_commander = cmd_buf
        self.buffer_size_radar = rad_buf

        # BC-pretrained actor snapshots for KL penalty during PPO fine-tuning
        self.bc_commander = None  # CommanderActorCritic (frozen, eval mode)
        self.bc_radar = None      # RadarActorCritic (frozen, eval mode)

        # Buffers initialized via init_buffers() after env is created
        self.commander_buffer = None
        self.radar_buffer = None
        rsc = reward_shaping_config or {}
        self.reward_shaper = DenseRewardShaper(
            device=device,
            detect_snr_weight=rsc.get("detect_snr_weight", 0.1),
            detect_coverage_weight=rsc.get("detect_coverage_weight", 0.05),
            jam_effectiveness_weight=rsc.get("jam_effectiveness_weight", 0.1),
            comm_reliability_weight=rsc.get("comm_reliability_weight", 0.05),
            recon_intel_weight=rsc.get("recon_intel_weight", 0.03),
            beam_accuracy_weight=rsc.get("beam_accuracy_weight", 0.02),
            stealth_weight=rsc.get("stealth_weight", stealth_weight),
            missile_guidance_weight=rsc.get("missile_guidance_weight", 0.02),
            snr_threshold_db=rsc.get("snr_threshold_db", 10.0),
        )
        # Team reward weights for CTDE hierarchical critic
        self.team_reward_weight = rsc.get("team_reward_weight", 0.1)
        self.team_kill_weight = rsc.get("team_kill_weight", 1.0)

    def init_buffers(self, env_state_dim: int, env_action_dim: int):
        """Initialize rollout buffers with correct dimensions from env."""
        # Privileged extra dim: task_fingerprint (n_teams*4) + intercept (n_teams*3) + target (2)
        privileged_dim = 2 * 4 + 2 * 3 + 2  # assuming n_teams=2
        self.commander_buffer = RolloutBuffer(
            self.buffer_size_commander, obs_dim=76, act_dim=35,
            gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device,
        )
        self.radar_buffer = RolloutBuffer(
            self.buffer_size_radar, obs_dim=env_state_dim, act_dim=env_action_dim,
            gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device,
            privileged_dim=privileged_dim,
        )

    def set_bc_pretrained(self):
        """Snapshot current actors as BC-pretrained reference for KL penalty.

        After BC pretraining, call this to freeze copies of the commander
        and radar actors.  During PPO, store_transition() will compute
        pretrain_log_probs under these frozen models, enabling the KL
        penalty that prevents catastrophic forgetting of BC-pretrained
        behavior.
        """
        import copy
        self.bc_commander = copy.deepcopy(self.commander_trainer.ac)
        self.bc_commander.eval()
        for p in self.bc_commander.parameters():
            p.requires_grad_(False)
        self.bc_radar = copy.deepcopy(self.radar_trainer.ac)
        self.bc_radar.eval()
        for p in self.bc_radar.parameters():
            p.requires_grad_(False)

    def _get_observations(self, env):
        """Get current state and commander observations from env."""
        state = env._assemble_state(env._buf_spectrum, env._buf_comm_data)
        comm_input = torch.zeros(
            env.num_envs, env.n_radars, env.num_input_length, device=self.device,
        )
        commander_obs = env.battlefield.get_commander_observation(
            env.radar_pos, comm_input,
        )
        return state, commander_obs

    def get_own_actions(self, env, team: int, deterministic: bool = False):
        """Query own policies and return actions for env.step().

        Args:
            env: MFARVecEnv instance
            team: team index (0 or 1)
            deterministic: use mean actions (for evaluation)
        Returns:
            dict with radar_actions, commander_action, transition, r_start, r_end
        """
        r_per_team = env.n_radars // env.n_teams
        r_start = team * r_per_team
        r_end = r_start + r_per_team

        state, commander_obs = self._get_observations(env)

        # Build privileged info for asymmetric critic (only during training)
        privileged_info = self._build_privileged_info(env, team)

        with torch.no_grad():
            # Commander
            cmd_obs = commander_obs[:, team, :]  # [E, 68]
            cmd_action, cmd_logp, cmd_val, _ = self.commander_trainer.ac.get_action(
                cmd_obs, deterministic=deterministic,
            )

            # Radars (shared policy, individual observations)
            radar_actions = []
            rep_logp = rep_val = rep_privileged_val = rep_obs = rep_action = None
            for r in range(r_start, r_end):
                r_obs = state[:, r, :]  # [E, state_dim]
                r_act, r_logp, r_val, r_priv_val = self.radar_trainer.ac.get_action(
                    r_obs, deterministic=deterministic,
                    privileged_info=privileged_info,
                )
                radar_actions.append(r_act)
                if r == r_start:
                    rep_obs = r_obs
                    rep_action = r_act
                    rep_logp = r_logp               # [E]
                    rep_val = r_val.squeeze(-1)      # [E]
                    rep_privileged_val = r_priv_val.squeeze(-1)  # [E]

        return {
            "radar_actions": radar_actions,    # list of [E, action_dim]
            "commander_action": cmd_action,    # [E, cmd_act_dim]
            "transition": {
                "cmd_obs": cmd_obs,
                "cmd_action": cmd_action,
                "cmd_logp": cmd_logp,
                "cmd_val": cmd_val.squeeze(-1),
                "radar_obs": rep_obs,
                "radar_action": rep_action,
                "radar_logp": rep_logp,
                "radar_val": rep_val,
                "radar_privileged_val": rep_privileged_val,
                "privileged_info": privileged_info,
            },
            "r_start": r_start,
            "r_end": r_end,
        }

    def _build_privileged_info(self, env, team: int) -> torch.Tensor:
        """Build privileged info tensor for the asymmetric critic.

        Includes information only available during centralized training:
        - Task fingerprint: [n_teams, 4] — per-team task allocation fractions
        - Cross-team intercept: [n_teams, 3] — per-team per-task intercept scores
        - Target direction: [2] — target az/el from team's first radar to target

        Uses values cached from the previous step() call; zeros on first step.

        Returns:
            [E, privileged_extra_dim] tensor.
        """
        E = env.num_envs
        n_teams = env.n_teams
        dev = torch.device(self.device)
        r_per_team = env.n_radars // n_teams
        r_start = team * r_per_team

        # Cached from previous step (or zeros on first step)
        task_fp = getattr(env, "_cached_task_fingerprint", None)
        if task_fp is None:
            task_fp = torch.zeros(E, n_teams * 4, device=dev)
        else:
            task_fp = task_fp.reshape(E, n_teams * 4).to(dev)

        intercept = getattr(env, "_cached_cross_team_intercept", None)
        if intercept is None:
            intercept_flat = torch.zeros(E, n_teams * 3, device=dev)
        else:
            # Concatenate team0 and team1 intercept detail
            i0 = intercept.get("team0_intercept_detail", torch.zeros(E, 3, device=dev))
            i1 = intercept.get("team1_intercept_detail", torch.zeros(E, 3, device=dev))
            intercept_flat = torch.cat([i0, i1], dim=-1).to(dev)

        # Target direction from team's radar to first target
        rel_tgt = env.target_pos[:, 0, :] - env.radar_pos[:, r_start, :]
        tgt_az = torch.atan2(rel_tgt[:, 1], rel_tgt[:, 0]) * (180.0 / np.pi)
        tgt_el = torch.atan2(
            rel_tgt[:, 2],
            torch.sqrt(rel_tgt[:, 0]**2 + rel_tgt[:, 1]**2).clamp(min=1.0),
        ) * (180.0 / np.pi)
        tgt_azel = torch.stack([tgt_az, tgt_el], dim=-1)  # [E, 2]

        return torch.cat([task_fp, intercept_flat, tgt_azel], dim=-1)

    def store_transition(self, env, result: dict, transition: dict, team: int):
        """Compute shaped rewards and store transitions in buffers.

        Args:
            env: MFARVecEnv instance
            result: output from env.step()
            transition: dict from get_own_actions()["transition"]
            team: team index
        Returns:
            reward metrics dict
        """
        shaped = self.reward_shaper(result)
        r_per_team = env.n_radars // env.n_teams
        r_start = team * r_per_team

        total_radar_reward = shaped["total_shaped"] + result["radar_rewards"]  # [E, R]
        cmd_reward = result["commander_rewards"]  # [E, n_teams]

        # ── Team-level reward for CTDE hierarchical critic ──
        # R_team = w1 * Σ(all_radar_rewards) + w2 * Σ(commander_rewards)
        # The commander_rewards already include kill_bonus (+10) and
        # death_penalty (-10), so this captures the full team outcome.
        team_rewards = (
            self.team_reward_weight * result["radar_rewards"].sum(dim=-1)      # [E]
            + self.team_kill_weight * result["commander_rewards"].sum(dim=-1)  # [E]
        )

        # ── Build team_state for TeamCritic ──
        bf = env.battlefield
        team_states = build_team_state(
            commander_obs=result["commander_obs"],           # [E, n_teams, 68]
            task_fingerprint=result.get("task_fingerprint"), # [E, n_teams, 4]
            avg_snr=None,  # computed inside reward_shaper, not cached
            alive=bf.alive,                                   # [E, n_radars]
            missile_pos=bf.missile.missile_pos,                # [E, n_teams, 3]
            missile_in_flight=bf.missile.in_flight,             # [E, n_teams]
            missile_target=bf.missile.target_pos,               # [E, n_teams, 3]
        )  # [E, 88]

        for e in range(env.num_envs):
            done = float(result["dones"][e].item())

            # ── BC pretrain log-prob for KL penalty ──
            cmd_pretrain_lp = None
            radar_pretrain_lp = None
            if self.bc_commander is not None:
                with torch.no_grad():
                    _, cmd_pretrain_lp, _, _ = self.bc_commander.get_action(
                        transition["cmd_obs"][e:e+1].to(self.device),
                        deterministic=True,
                    )
                    cmd_pretrain_lp = cmd_pretrain_lp.item()
            if self.bc_radar is not None:
                with torch.no_grad():
                    _, radar_pretrain_lp, _, _ = self.bc_radar.get_action(
                        transition["radar_obs"][e:e+1].to(self.device),
                        deterministic=True,
                    )
                    radar_pretrain_lp = radar_pretrain_lp.item()

            # Commander transition
            self.commander_buffer.add(
                obs=transition["cmd_obs"][e].cpu(),
                action=transition["cmd_action"][e].cpu(),
                reward=cmd_reward[e, team].item(),
                done=done,
                value=transition["cmd_val"][e].item(),
                log_prob=transition["cmd_logp"][e].item(),
                team_reward=team_rewards[e].item(),
                team_state=team_states[e].cpu(),
                pretrain_log_prob=cmd_pretrain_lp,
            )

            # Radar transition (representative radar for the team)
            radar_reward = total_radar_reward[e, r_start:r_start + r_per_team].sum().item()
            priv_val = transition.get("radar_privileged_val")
            priv_info = transition.get("privileged_info")
            self.radar_buffer.add(
                obs=transition["radar_obs"][e].cpu(),
                action=transition["radar_action"][e].cpu(),
                reward=radar_reward,
                done=done,
                value=transition["radar_val"][e].item(),
                log_prob=transition["radar_logp"][e].item(),
                privileged_value=priv_val[e].item() if priv_val is not None else None,
                privileged_info=priv_info[e].cpu() if priv_info is not None else None,
                team_reward=team_rewards[e].item(),
                team_state=team_states[e].cpu(),
                pretrain_log_prob=radar_pretrain_lp,
            )

        # Cache privileged info on env for next get_own_actions call
        env._cached_task_fingerprint = result.get("task_fingerprint")
        env._cached_cross_team_intercept = result.get("cross_team_intercept")

        return {
            "radar_reward": total_radar_reward,
            "commander_reward": cmd_reward,
            "shaped_rewards": shaped,
        }

    def update(self,
               team_critic: "TeamCritic | None" = None,
               alpha: float = 0.0,
               beta_kl: float = 0.0,
               n_step: int = 0,
               n_step_team: int = 800,
               team_critic_optimizer=None) -> dict:
        """Run PPO updates for both commander and radar when buffers are full.

        Args:
            team_critic: optional TeamCritic for hierarchical advantage.
            alpha: team advantage weight (0→1 over training).
            beta_kl: KL penalty weight for domain-shift mitigation.
            n_step: if > 0, use N-step returns (long-horizon credit assignment)
                    instead of GAE. N=400 for ~600-step missile flight.
            n_step_team: N-step horizon for team returns (default 800, longer
                         than agent N=400 for kill-event credit propagation).
            team_critic_optimizer: optional optimizer for TeamCritic training.
        """
        cmd_metrics = {}
        radar_metrics = {}

        if self.commander_buffer and self.commander_buffer.size > self.batch_size:
            if n_step > 0:
                last_v = (self.commander_buffer.values[self.commander_buffer.ptr - 1].item()
                          if self.commander_buffer.ptr > 0 else 0.0)
                self.commander_buffer.compute_n_step_returns(n_steps=n_step, last_value=last_v)
                # Team returns: longer horizon for kill credit
                self.commander_buffer.compute_team_n_step_returns(
                    n_steps=n_step_team, last_value=last_v,
                )
            else:
                self.commander_buffer.compute_returns()
            cmd_metrics = self.commander_trainer.update(
                self.commander_buffer, team_critic=team_critic,
                alpha=alpha, beta_kl=beta_kl,
                team_critic_optimizer=team_critic_optimizer,
            )
            self.commander_buffer.reset()

        if self.radar_buffer and self.radar_buffer.size > self.batch_size:
            last_pv = None
            if self.radar_buffer.ptr > 0:
                last_pv = self.radar_buffer.privileged_values[self.radar_buffer.ptr - 1].item()
            if n_step > 0:
                last_v = (self.radar_buffer.values[self.radar_buffer.ptr - 1].item()
                          if self.radar_buffer.ptr > 0 else 0.0)
                self.radar_buffer.compute_n_step_returns(n_steps=n_step, last_value=last_v)
                # Team returns: longer horizon for kill credit
                self.radar_buffer.compute_team_n_step_returns(
                    n_steps=n_step_team, last_value=last_v,
                )
            else:
                self.radar_buffer.compute_returns(last_privileged_value=last_pv)
            radar_metrics = self.radar_trainer.update(
                self.radar_buffer, team_critic=team_critic,
                alpha=alpha, beta_kl=beta_kl,
                team_critic_optimizer=team_critic_optimizer,
            )
            self.radar_buffer.reset()

        return {"commander": cmd_metrics, "radar": radar_metrics}

    def save(self, path: str):
        """Save team policy checkpoint."""
        torch.save({
            "commander": self.commander_trainer.ac.state_dict(),
            "radar": self.radar_trainer.ac.state_dict(),
            "commander_optimizer": self.commander_trainer.optimizer.state_dict(),
            "radar_optimizer": self.radar_trainer.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load team policy checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.commander_trainer.ac.load_state_dict(ckpt["commander"])
        self.radar_trainer.ac.load_state_dict(ckpt["radar"])
        self.commander_trainer.optimizer.load_state_dict(ckpt["commander_optimizer"])
        self.radar_trainer.optimizer.load_state_dict(ckpt["radar_optimizer"])
