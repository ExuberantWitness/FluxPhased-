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
        total_team_adv_std = 0.0   # P2 §1 mechanism instrumentation
        total_kl_penalty = 0.0
        total_entropy = 0.0
        total_nan_skips = 0
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
                    # P2 §1 mechanism instrumentation: track pre-norm team_adv
                    # std. F1 broken → ~0 (noise amplified to unit by L92); F1
                    # fixed → O(1) (true team advantage signal).
                    total_team_adv_std += float(team_adv.std().item())
                    # F2: deleted the per-batch team_adv normalization here.
                    # L107 below already normalizes the blended advantage once;
                    # double-normalizing team_adv forced raw noise to unit scale
                    # and let it dominate A_agent once α>0.5. With F1 producing
                    # O(1) team_returns, team_adv has reasonable magnitude and
                    # the single normalization at L107 is sufficient.
                    #
                    # Ablation: f2_disable=True restores the OLD double
                    # normalization (team_adv standardized to unit variance
                    # before blend) for A/B baseline comparison.
                    if getattr(self, 'f2_disable', False):
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

                # F4: clamp log-ratio to [-20, 20] before exp. Without this, a
                # single mismatched log_prob (e.g. residual vs absolute aim before
                # F1) could blow ratio to ±inf and poison the whole minibatch.
                log_ratio = (log_prob - old_log_probs).clamp(-20.0, 20.0)
                ratio = torch.exp(log_ratio)
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
                # F4: NaN/Inf skip-guard. If loss or any grad is non-finite, drop
                # this minibatch entirely (don't step, don't update Adam moments).
                # This is the defensive backstop for cases F1/F2 don't catch.
                if not torch.isfinite(loss) or any(
                    not torch.isfinite(p.grad).all()
                    for p in self.ac.parameters() if p.grad is not None
                ):
                    self.optimizer.zero_grad()
                    total_nan_skips += 1
                    continue
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
            "nan_skips": total_nan_skips,
        }
        if total_team_value_loss > 0:
            metrics["team_value_loss"] = total_team_value_loss / max(n_updates, 1)
            metrics["team_adv_std"] = total_team_adv_std / max(n_updates, 1)
            # P2 §1.2 mechanism dashboard — print per-update so we can compare
            # A baseline (broken F1+F2 → team_value_loss百万级, team_adv_std趋0)
            # vs B (F1+F2 ON → team_value_loss~10万级, team_adv_std O(1))
            print(f"      [mech] team_value_loss={metrics['team_value_loss']:.3e}  "
                  f"team_adv_std={metrics['team_adv_std']:.3e}", flush=True)
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
        task_type: str = "generic",
        laser_cfg: dict = None,
        sensing_cfg: dict = None,
        reward_normalize: bool = False,  # F8: return-based scaling
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

        # ── Laser task hooks ─────────────────────────────────────────────
        # When task_type=="laser", swap DenseRewardShaper for LaserRewardShaper
        # and attach a KalmanTracker for multi-radar fused sensing. The laser
        # task replaces both radar and commander rewards (kill/illum/beam/fire_lock).
        self.task_type = task_type
        self.laser_cfg = laser_cfg or {}
        self.sensing_cfg = sensing_cfg or {}
        self.reward_normalize = reward_normalize  # F8
        if task_type == "laser":
            from training.laser.reward import LaserRewardShaper
            from training.laser.sensing import KalmanTracker
            # Override the DenseRewardShaper with laser shaper
            self.reward_shaper = LaserRewardShaper(
                self.laser_cfg, env=None, device=device,
            )
            scfg = self.sensing_cfg
            self.kalman_tracker = KalmanTracker(
                track_q_m=scfg.get("track_q_m", 0.05),
                track_burnin=scfg.get("track_burnin", 30),
                acq_baseline_m=scfg.get("acq_baseline_m", 0.0),
            )
            self.sensing_mode = scfg.get("mode", "single")
            self.sensing_range_sigma_m = scfg.get("range_sigma_m", 0.0)
            self.sensing_crossrange_factor = scfg.get("crossrange_factor", 0.0)
            self.jam_gain = scfg.get("jam_gain", 0.0)
            self.exposure_gain = scfg.get("exposure_gain", 0.0)
            # WP3.2: constant fused-position bias (calibration error damage)
            self.sensing_bias_m = float(scfg.get("sensing_bias_m", 0.0))
            self.residual_aim = self.laser_cfg.get("residual_aim", False)
            self.residual_scale_m = self.laser_cfg.get("residual_scale_m", 100.0)
            # Phase 3 v5c/v5d: optional Gaussian noise added to the Kalman-fused
            # anchor each step. Forces the policy to learn corrections instead
            # of free-riding on the (noisy but biased-low) Kalman output. Noise
            # is in metres, scaled to normalized [-1,1] space at apply time.
            self.anchor_noise_std_m = float(self.laser_cfg.get("anchor_noise_std_m", 0.0))
            self.min_radar_baseline_m = self.laser_cfg.get("min_radar_baseline_m", 0.0)
            # F1/F2 ablation switches (修改建议 §6 P1 A/B/C/D matrix).
            # Defaults: False = FIX ACTIVE (production behavior).
            # Set True in config to revert each fix for isolation testing.
            self.f1_disable = bool(self.laser_cfg.get("f1_disable", False))
            self.f2_disable = bool(self.laser_cfg.get("f2_disable", False))
            # Propagate F2 to inner PPOTrainers (commander + radar)
            self.commander_trainer.f2_disable = self.f2_disable
            self.radar_trainer.f2_disable = self.f2_disable
            self._laser_env_attached = False
        else:
            self.kalman_tracker = None
            self.sensing_mode = "single"
            self.sensing_range_sigma_m = 0.0
            self.sensing_crossrange_factor = 0.0
            self.jam_gain = 0.0
            self.exposure_gain = 0.0
            self.residual_aim = False
            self.residual_scale_m = 100.0
            self.anchor_noise_std_m = 0.0
            self.min_radar_baseline_m = 0.0
            self._laser_env_attached = False

    def _attach_laser_env(self, env):
        """Laser hooks need an env reference for reward/sensing — attach on first use."""
        if not self._laser_env_attached and self.task_type == "laser":
            self.reward_shaper.env = env
            self._laser_env_attached = True

    def reset_episode(self):
        """No-op: LaserEpisodeRunner.reset() already resets KalmanTracker and
        LaserRewardShaper state per episode (episode.py:168-176).

        Kept as a hook for future per-episode reset needs (and to avoid breaking
        any caller that already invokes it). Originally added under the
        mistaken belief that the Kalman tracker wasn't being reset; that turned
        out to be wrong — see V5_DARTBOARD_REPORT.md §"Correction after deeper
        audit" for the actual root cause (aim_z drift, not Kalman bias).
        """
        return

    def _apply_laser_sensing(self, cmd_obs: torch.Tensor, env) -> torch.Tensor:
        """Replace exact enemy_xy in cmd_obs[..., 68:72] with Kalman-fused multi-radar estimate.

        Operates on the full [E, n_teams, 76] commander obs (sensing applies
        per-team: each team's own jam degrades the enemy's view of it).

        For "single" mode (default if sensing_noise.range_sigma_m=0): returns obs unchanged.
        For "fused"/"tracked": runs information-filter fusion (+ optional KF over time).
        """
        if self.sensing_range_sigma_m <= 0.0 and self.sensing_crossrange_factor <= 0.0:
            return cmd_obs
        half_x = float(env.map_size[0]) / 2.0
        half_y = float(env.map_size[1]) / 2.0
        from training.laser.sensing import fused_sensing, add_sensing_noise
        if self.sensing_mode in ("fused", "tracked"):
            jam = self.reward_shaper._jam_level  # [E, n_teams]
            cmd_obs = fused_sensing(
                cmd_obs, half_x, half_y,
                self.sensing_range_sigma_m, self.sensing_crossrange_factor,
                tracker=self.kalman_tracker if self.sensing_mode == "tracked" else None,
                jam_gain=self.jam_gain, exposure_gain=self.exposure_gain,
                jam_level=jam,
            )
            # WP3.2 damage: apply constant fused-xy bias (calibration error).
            # cmd_obs[68:72] holds 2 enemies' fused xy in normalized [-1,1]:
            #   68=enemy0.x, 69=enemy0.y, 70=enemy1.x, 71=enemy1.y
            if self.sensing_bias_m != 0.0:
                bx = self.sensing_bias_m / half_x
                by = self.sensing_bias_m / half_y
                cmd_obs[..., 68].add_(bx)
                cmd_obs[..., 69].add_(by)
                cmd_obs[..., 70].add_(bx)
                cmd_obs[..., 71].add_(by)
            return torch.nan_to_num(cmd_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return add_sensing_noise(
            cmd_obs, self.sensing_range_sigma_m, self.sensing_crossrange_factor,
            half_x, half_y,
        )

    def _apply_residual_aim(self, cmd_action: torch.Tensor, cmd_obs: torch.Tensor,
                             env) -> torch.Tensor:
        """Build the env-action by anchoring aim_xy at sensed enemy + learned residual.

        F1 fix (LASER_LEAGUE_NAN_FULL_ANALYSIS.md): the env needs absolute aim
        (anchor + residual), but the PPO buffer must store the policy's raw
        residual so log_prob matches what was sampled. Returning a fresh tensor
        (not mutating cmd_action in-place) keeps the two decoupled — mirroring
        train_laser.py:1136-1139 which builds `env_a` as a separate tensor.

        The returned env-action uses a soft ±(1−1e-4) clamp so downstream
        consumers (env.step, sensing) never see exactly ±1 (which would
        re-introduce the atanh singularity via S4 nan_to_num posinf=1.0).

        aim_z fix: env scales action[3] by 1000m (vec_drone.py:195), but laser
        targets are ground radars at z=0. Untrained policy samples action[3] ~
        N(0, 0.37²) → aim_z ~ 300m avg → completely overshoots kill_radius.
        Force env_action[3]=0 to eliminate this DoF (it's reserved for future
        air targets; not needed for the current ground-target task).
        """
        if not self.residual_aim:
            return cmd_action
        half_x = float(env.map_size[0]) / 2.0
        half_y = float(env.map_size[1]) / 2.0
        anchor_x = cmd_obs[..., 68]  # sensed enemy-0 x (normalized [-1,1])
        anchor_y = cmd_obs[..., 69]
        # Phase 3 v5c/v5d: add Gaussian noise to anchor (normalized units).
        if self.anchor_noise_std_m > 0.0:
            noise_sigma_x = self.anchor_noise_std_m / half_x
            noise_sigma_y = self.anchor_noise_std_m / half_y
            anchor_x = anchor_x + torch.randn_like(anchor_x) * noise_sigma_x
            anchor_y = anchor_y + torch.randn_like(anchor_y) * noise_sigma_y
        # residual in physical metres → normalized [-1, 1]
        dx_norm = cmd_action[..., 1] * self.residual_scale_m / half_x
        dy_norm = cmd_action[..., 2] * self.residual_scale_m / half_y
        aim_x = (anchor_x + dx_norm).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        aim_y = (anchor_y + dy_norm).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        env_action = cmd_action.clone()
        env_action[..., 1] = aim_x
        env_action[..., 2] = aim_y
        env_action[..., 3] = 0.0  # aim_z=0 (ground targets only)
        return env_action

    def init_buffers(self, env_state_dim: int, env_action_dim: int,
                     commander_act_dim: int = 5):
        """Initialize rollout buffers with correct dimensions from env.

        Args:
            env_state_dim: radar obs dim (env.state_dim) — used as a fallback;
                the actor_critic's actual expected input dim takes precedence
                because env.state_dim is known to be off-by-N in some configs
                (laser vs generic disagree on missile_dim).
            env_action_dim: radar action dim (env.action_dim)
            commander_act_dim: commander action dim — pass env.battlefield.commander_action_dim.
                Defaults to 5 (matches env truth: [fire, aim_x, aim_y, aim_z, reserved]).
                Previously hardcoded 35, which was a latent bug — see plan Phase 0.
        """
        # Privileged extra dim: task_fingerprint (n_teams*4) + intercept (n_teams*3) + target (2)
        privileged_dim = 2 * 4 + 2 * 3 + 2  # assuming n_teams=2
        # Authoritative obs dim: ask the actor_critic, which knows its own input layout.
        ac = self.radar_trainer.ac
        obs_dim = getattr(ac, "spectrum_flat_dim", 0) + getattr(ac, "comm_flat_dim", 0) \
            + getattr(ac, "recon_flat_dim", 0) + getattr(ac, "other_dim", 0)
        if obs_dim == 0:
            obs_dim = env_state_dim  # fallback for AC variants without the breakdown
        self.commander_buffer = RolloutBuffer(
            self.buffer_size_commander, obs_dim=76, act_dim=commander_act_dim,
            gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device,
            reward_normalize=self.reward_normalize,  # F8
        )
        self.radar_buffer = RolloutBuffer(
            self.buffer_size_radar, obs_dim=obs_dim, act_dim=env_action_dim,
            gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device,
            privileged_dim=privileged_dim,
            reward_normalize=self.reward_normalize,  # F8
        )
        # P2 §3: wire ablation switches from laser_cfg so A baseline config
        # can disable F1 (cross-ep 800-step) and F2 (team_adv double-norm)
        # without code changes. Default False = F1+F2 ON (current behavior).
        if self.task_type == "laser":
            f1_off = bool(self.laser_cfg.get("f1_disable", False))
            f2_off = bool(self.laser_cfg.get("f2_disable", False))
            self.commander_buffer.f1_disable = f1_off
            self.radar_buffer.f1_disable = f1_off
            self.commander_trainer.f2_disable = f2_off
            self.radar_trainer.f2_disable = f2_off
        # F1 ablation switch propagation (f1_disable attribute on buffer)
        if hasattr(self, 'f1_disable'):
            self.commander_buffer.f1_disable = self.f1_disable
            self.radar_buffer.f1_disable = self.f1_disable

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

    def _get_observations(self, env, spectrum: torch.Tensor = None,
                          events: dict = None):
        """Build per-radar state + commander obs from spectrum + events.

        Args:
            env: MFARVecEnv
            spectrum: [E, R, N, P, n_bins] FFT-magnitude CPI (from runner).
                None on the very first control step → zero state.
            events: dict with radar_pos etc. (from runner).
        Returns:
            state: [E, R, state_dim] radar policy input.
            commander_obs: [E, n_teams, 76] from env.battlefield.
        """
        dev = self.device
        E = env.num_envs
        R = env.n_radars
        N = env.n_elem
        P = env.n_pulses
        n_bins = env.n_bins

        if spectrum is not None:
            spec_flat = spectrum.reshape(E, R, -1)
        else:
            spec_flat = torch.zeros(E, R, N * P * n_bins, device=dev)
        comm_flat = torch.zeros(E, R, N * 2, device=dev)
        recon_flat = torch.zeros(E, R, N * 4, device=dev)
        vehicle = torch.zeros(E, R, 5, device=dev)
        laser_state = torch.zeros(E, R, 12, device=dev)
        cmd_instr = torch.zeros(E, R, 16, device=dev)
        if events is not None and "radar_pos" in events:
            vehicle[:, :, 0] = events["radar_pos"][:, :, 0]
            vehicle[:, :, 1] = events["radar_pos"][:, :, 1]
        state = torch.cat(
            [spec_flat, comm_flat, recon_flat, vehicle, laser_state, cmd_instr],
            dim=-1,
        )

        comm_input = torch.zeros(E, R, 32, device=dev)
        commander_obs = env.battlefield.get_commander_observation(
            env.radar_pos, comm_input,
        )
        return state, commander_obs

    def get_own_actions(self, env, team: int, deterministic: bool = False,
                        spectrum: torch.Tensor = None, events: dict = None):
        """Query own policies and return per-team actions.

        Args:
            env: MFARVecEnv instance
            team: team index (0 or 1)
            deterministic: use mean actions (for evaluation)
            spectrum: [E, R, N, P, n_bins] FFT-magnitude CPI (from runner).
                None on first control step → zero state.
            events: dict from runner (radar_pos, alive, ...).
        Returns:
            dict with radar_actions, commander_action, transition, r_start, r_end
        """
        r_per_team = env.n_radars // env.n_teams
        r_start = team * r_per_team
        r_end = r_start + r_per_team

        # Laser task: attach env to reward shaper + enforce radar baseline
        # BEFORE _get_observations so the first commander_obs sees spread radars.
        # (The reset-time enforce in LaserEpisodeRunner.reset() covers episode start;
        # this per-step call keeps radars spread as they move at 20 m/s through the ep.)
        if self.task_type == "laser":
            self._attach_laser_env(env)
            from training.laser.sensing import enforce_radar_baseline
            enforce_radar_baseline(env, self.min_radar_baseline_m)

        state, commander_obs = self._get_observations(env, spectrum, events)

        # Build privileged info for asymmetric critic (only during training)
        privileged_info = self._build_privileged_info(env, team)

        with torch.no_grad():
            # Laser task: replace exact enemy_xy with Kalman-fused estimate.
            # Operates on the full [E, n_teams, 76] before per-team slicing.
            if self.task_type == "laser":
                commander_obs = self._apply_laser_sensing(commander_obs, env)
            # Commander
            cmd_obs = commander_obs[:, team, :]  # [E, 76]
            cmd_action, cmd_logp, cmd_val, _ = self.commander_trainer.ac.get_action(
                cmd_obs, deterministic=deterministic,
            )
            # F1: build env-action separately so the PPO buffer can still store
            # the raw residual (which carries the correct log_prob). The previous
            # code mutated cmd_action in-place, which desynced buffer vs old_logp
            # AND could produce aim=±1.0 exactly → atanh singularity → NaN crash.
            # See LASER_LEAGUE_NAN_FULL_ANALYSIS.md §3.
            if self.task_type == "laser":
                cmd_action_env = self._apply_residual_aim(cmd_action, cmd_obs, env)
                self.reward_shaper.update_jam(cmd_action_env, team)
            else:
                cmd_action_env = cmd_action

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
            "radar_actions": radar_actions,        # list of [E, action_dim]
            "commander_action": cmd_action_env,    # [E, cmd_act_dim] — absolute aim, used by env.step
            "transition": {
                "cmd_obs": cmd_obs,
                "cmd_action": cmd_action,          # residual — what PPO must store to match cmd_logp
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

        if self.task_type == "laser":
            # Laser task: LaserRewardShaper returns full reward override (not delta).
            total_radar_reward = shaped["radar_rewards"]  # [E, R] full
            cmd_reward = shaped["commander_rewards"]      # [E, n_teams] full
        else:
            # Generic task: DenseRewardShaper returns add-on to radar reward.
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
        # commander_obs is passed via transition (env.step() doesn't return it).
        bf = env.battlefield
        # transition["cmd_obs"] is [E, 76] for own team only; rebuild the
        # full [E, n_teams, 76] view by querying env fresh — both teams' obs
        # share the same battlefield state.
        comm_input = torch.zeros(env.num_envs, env.n_radars, 32, device=self.device)
        full_cmd_obs = env.battlefield.get_commander_observation(
            env.radar_pos, comm_input,
        )
        # Missile is generic-task only; laser uses dwell-to-kill (no missile obj).
        missile = getattr(bf, "missile", None)
        E = env.num_envs
        T = env.n_teams
        if missile is not None:
            missile_pos = missile.missile_pos
            missile_in_flight = missile.in_flight
            missile_target = missile.target_pos
        else:
            missile_pos = torch.zeros(E, T, 3, device=self.device)
            missile_in_flight = torch.zeros(E, T, device=self.device)
            missile_target = torch.zeros(E, T, 3, device=self.device)
        team_states = build_team_state(
            commander_obs=full_cmd_obs,                       # [E, n_teams, 68 or 76]
            task_fingerprint=transition.get("task_fingerprint"),  # not used; placeholder
            avg_snr=None,
            alive=bf.alive,                                   # [E, n_radars]
            missile_pos=missile_pos,
            missile_in_flight=missile_in_flight,
            missile_target=missile_target,
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
                # F1: team returns — done-masked discounted return + reward
                # normalization (was: 800-step cross-episode accumulation).
                self.commander_buffer.compute_team_returns()
            else:
                # C1 fix: pass last_value (deployment) so the GAE bootstrap at
                # the buffer's final step isn't biased toward 0. The commander
                # buffer has no privileged_values populated (only deployment
                # value is stored), so last_privileged_value falls back to
                # last_value — same behavior as before but no longer zero-filled.
                last_v = (self.commander_buffer.values[self.commander_buffer.ptr - 1].item()
                          if self.commander_buffer.ptr > 0 else 0.0)
                self.commander_buffer.compute_returns(
                    last_value=last_v, last_privileged_value=last_v,
                )
                # F1: also compute team returns in GAE path (was missing —
                # without this, n_step=0 configs had team_returns=0 forever,
                # making team_adv = -team_value and F1 had no effect).
                self.commander_buffer.compute_team_returns()
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
                # F1: team returns — done-masked discounted return + reward
                # normalization (was: 800-step cross-episode accumulation).
                self.radar_buffer.compute_team_returns()
            else:
                self.radar_buffer.compute_returns(last_privileged_value=last_pv)
                # F1: also compute team returns in GAE path (see commander side).
                self.radar_buffer.compute_team_returns()
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
