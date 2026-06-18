"""FluxLeague: Full 3-role league training manager for FluxPhased.

Implements AlphaStar-style league with three agent roles per team:
  1. Main Agent — trains against full opponent population via PFSP
  2. Main Exploiter — trains against opponent's current Main Agent only
  3. League Exploiter — trains against full opponent population, resettable

Each role is backed by a TeamPPOTrainer (commander + shared radar).
The league manager orchestrates PSRO iterations:
  evaluate payoff matrix → compute meta-Nash → train best responses
"""

import os
import time
import torch
import numpy as np
import functools
import builtins
from typing import Dict, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

print = functools.partial(builtins.print, flush=True)

from .ppo.actor_critic import create_team_policy, TeamCritic, build_team_state
from .ppo.ppo_trainer import TeamPPOTrainer
from .self_play.opponent_pool import OpponentPool
from .self_play.payoff_matrix import PayoffMatrix
from .self_play.meta_solver import solve_nash, solve_rectified_nash, nash_conv
from .self_play.tc_dams_solver import (
    solve_tc_dams,
    task_fingerprint_entropy,
    effective_population_size,
)
from .self_play.elo_band_sampler import EloBandSampler


ROLE_MAIN = "main"
ROLE_MAIN_EXPLOITER = "main_exploiter"
ROLE_LEAGUE_EXPLOITER = "league_exploiter"


class FluxLeague:
    """Full 3-role league training manager.

    Usage:
        league = FluxLeague(env_config, league_config)
        league.initialize(env)
        for iteration in range(n_iterations):
            league.psro_iteration(env)
    """

    def __init__(
        self,
        n_elem: int = 625,
        n_pulses: int = 32,
        n_bins: int = 1024,
        num_output_length: int = 16,
        n_teams: int = 2,
        population_cap: int = 20,
        n_eval_games: int = 50,
        meta_solver: str = "nash",
        pfsp_temperature: float = 1.0,
        exploiter_reset_prob: float = 0.1,
        episodes_per_training: int = 1000,
        max_steps_per_episode: int = 1000,
        checkpoint_dir: str = "checkpoints/league",
        device: str = "cuda",
        sub_array_size: int = 0,
        # TC-DAMS + Elo-band knobs
        tcdams_lambda: float = 0.3,
        use_elo_band: bool = False,
        elo_band_init: float = 400.0,
        elo_band_final: float = 100.0,
        elo_anneal_iters: int = 15,
        # Buffer sizing: for large obs_dim (e.g., 25x25) a huge rollout buffer
        # can OOM CPU RAM; allow overriding.
        buffer_size_commander: int = 2048,
        buffer_size_radar: int = 64,
        # PPO hyperparams
        commander_lr: float = 3e-4,
        radar_lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_step_returns: int = 0,     # 0=use GAE, >0=N-step returns (e.g. 400)
        commander_clip: float = 0.2,
        radar_clip: float = 0.1,
        commander_entropy: float = 0.01,
        radar_entropy: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        stealth_weight: float = 0.1,
        # Full reward shaping config (all weights, not just stealth)
        reward_shaping_config: dict = None,
        # Policy mutation
        mutation_config: dict = None,
        # Laser task integration
        task_type: str = "generic",
        laser_cfg: dict = None,
        sensing_cfg: dict = None,
        pulses_per_control: int = 5,
    ):
        self.n_elem = n_elem
        self.n_pulses = n_pulses
        self.n_bins = n_bins
        self.num_output_length = num_output_length
        self.n_teams = n_teams
        self.population_cap = population_cap
        self.n_eval_games = n_eval_games
        self.meta_solver_name = meta_solver
        self.sub_array_size = sub_array_size
        self.pfsp_temperature = pfsp_temperature
        self.exploiter_reset_prob = exploiter_reset_prob
        self.episodes_per_training = episodes_per_training
        self.max_steps_per_episode = max_steps_per_episode
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        # PPO hyperparams
        self.ppo_config = dict(
            commander_lr=commander_lr,
            radar_lr=radar_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            commander_clip=commander_clip,
            radar_clip=radar_clip,
            commander_entropy=commander_entropy,
            radar_entropy=radar_entropy,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            n_epochs=n_epochs,
            batch_size=batch_size,
            buffer_size=buffer_size,
            buffer_size_commander=buffer_size_commander,
            buffer_size_radar=buffer_size_radar,
            device=device,
            stealth_weight=stealth_weight,
            reward_shaping_config=reward_shaping_config or {},
        )

        self.n_step_returns = n_step_returns

        # Components
        self.pool = OpponentPool(
            pool_dir=os.path.join(checkpoint_dir, "pool"),
            population_cap=population_cap,
            pfsp_temperature=pfsp_temperature,
        )
        self.payoff = None  # initialized after env
        self.trainers: Dict[str, TeamPPOTrainer] = {}
        self.meta_strategies: Dict[int, np.ndarray] = {}

        # TC-DAMS + Elo-band
        self.tcdams_lambda = tcdams_lambda
        self.use_elo_band = use_elo_band
        self.elo_sampler = EloBandSampler(
            self.pool,
            band_init=elo_band_init,
            band_final=elo_band_final,
            anneal_iters=elo_anneal_iters,
        ) if use_elo_band else None
        # Per-iteration diagnostics for the narrative report.
        self.diag_history: list = []

        self.iteration = 0

        # Policy mutation
        self.mutation_config = mutation_config or {}
        self.mutant_trainers: Dict[str, TeamPPOTrainer] = {}

        # Laser task integration
        self.task_type = task_type
        self.laser_cfg = laser_cfg or {}
        self.sensing_cfg = sensing_cfg or {}
        # Laser policy init flags — passed to create_team_policy so the aim head
        # gets zero-initialized (hybrid_fire) and value trunk gets decoupled
        # (decouple_value). Without these, an untrained commander outputs random
        # aim ~±1 → residual_aim pushes aim tens of km from target → no kills.
        self.hybrid_fire = self.laser_cfg.get("hybrid_fire", False)
        self.decouple_value = self.laser_cfg.get("decouple_value", False)
        self.pulses_per_control = int(pulses_per_control)

        # Hierarchical critic (CTDE architecture)
        self.team_critic_enabled = True  # set False for ablation (Config C)
        self.team_critic = TeamCritic(input_dim=104, hidden_dim=256).to(
            torch.device(device),
        )
        self.team_critic_optimizer = torch.optim.Adam(
            self.team_critic.parameters(), lr=radar_lr,
        )

        # Team reward weights (configurable via reward_shaping_config)
        self.team_reward_weight = 0.1   # w1: Σ(all_radar_rewards)
        self.team_kill_weight = 1.0     # w2: commander kill/death bonuses

        # Alpha/beta scheduling for league training
        self.alpha = 0.0          # team advantage weight (0→1)
        self.beta_kl = 0.1        # KL penalty weight (1→0)
        self.alpha_schedule = "linear"  # "linear" | "log" | "adaptive"

        os.makedirs(checkpoint_dir, exist_ok=True)

    def initialize(self, env):
        """Create initial 6 policies (3 roles × 2 teams) and register in pool.

        Args:
            env: MFARVecEnv instance (used to get obs/action dims)
        """
        # Laser task: start env at kill_radius_init (curriculum entry point)
        # rather than the final target. Annealing in _anneal_kill_radius
        # tightens toward kill_radius_final over iterations.
        if self.task_type == "laser":
            kr_init = float(self.laser_cfg.get("kill_radius_init", 50.0))
            if hasattr(env, "battlefield") and hasattr(env.battlefield, "laser"):
                env.battlefield.laser.kill_radius_m = kr_init
                print(f"[League] kill_radius initialized at {kr_init:.3f}m "
                      f"(will anneal toward "
                      f"{float(self.laser_cfg.get('kill_radius_final', 0.2)):.3f}m)",
                      flush=True)

        self.payoff = PayoffMatrix(
            self.pool, self.n_eval_games, self.device,
            max_steps_per_game=self.max_steps_per_episode,
            task_type=self.task_type,
            pulses_per_control=self.pulses_per_control,
        )

        for team in range(self.n_teams):
            for role in [ROLE_MAIN, ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER]:
                policy_dict = create_team_policy(
                    team=team,
                    n_elem=self.n_elem,
                    n_pulses=self.n_pulses,
                    n_bins=self.n_bins,
                    num_output_length=self.num_output_length,
                    device=self.device,
                    sub_array_size=self.sub_array_size,
                    hybrid_fire=self.hybrid_fire,
                    decouple_value=self.decouple_value,
                )
                trainer = TeamPPOTrainer(
                    commander=policy_dict["commander"],
                    radar=policy_dict["radar"],
                    **self.ppo_config,
                    task_type=self.task_type,
                    laser_cfg=self.laser_cfg,
                    sensing_cfg=self.sensing_cfg,
                )
                trainer.init_buffers(
                    env.state_dim, env.action_dim,
                    commander_act_dim=getattr(
                        env.battlefield, "commander_action_dim", 5,
                    ),
                )

                # Save initial checkpoint
                ckpt_name = f"{role}_team{team}_gen0.pt"
                ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
                trainer.save(ckpt_path)

                # Register in pool
                policy_id = self.pool.add_policy(
                    team=team,
                    role=role,
                    checkpoint_path=ckpt_path,
                    generation=0,
                )
                self.trainers[policy_id] = trainer

    def psro_iteration(self, env):
        """Run one PSRO iteration: evaluate → solve → train best responses.

        Args:
            env: MFARVecEnv instance
        Returns:
            metrics dict
        """
        metrics = {"iteration": self.iteration}
        t0 = time.time()

        # Step 1: Evaluate payoff matrix
        print(f"[League] Iteration {self.iteration}: Evaluating payoff matrix...")
        self.payoff.evaluate_all(env, {**self.trainers, **self.mutant_trainers})

        # Step 2: Compute meta-strategies
        if self.elo_sampler is not None:
            self.elo_sampler.update_from_payoff_matrix(self.payoff.matrix)
        iter_diag = {"iteration": self.iteration, "teams": {}}
        for team in range(self.n_teams):
            payoff_mat, own_ids, opp_ids = self.payoff.get_submatrix(
                team, exclude_roles=["mutant"])
            F = self.payoff.get_fingerprints(own_ids)  # [K_own, 4]
            if self.meta_solver_name == "nash":
                sigma = solve_nash(payoff_mat)
            elif self.meta_solver_name == "rectified_nash":
                sigma = solve_rectified_nash(payoff_mat)
            elif self.meta_solver_name == "tc_dams":
                sigma = solve_tc_dams(
                    payoff_mat, fingerprints=F,
                    lambda_diversity=self.tcdams_lambda,
                )
            else:
                K = len(own_ids)
                sigma = np.ones(K) / max(K, 1)
            self.meta_strategies[team] = sigma

            nc = nash_conv(payoff_mat, sigma)
            H_task = task_fingerprint_entropy(sigma, F)
            eff_K = effective_population_size(sigma)
            print(
                f"  Team {team} sigma={sigma.round(3)} "
                f"NashConv={nc:.4f} H_task={H_task:.3f} effK={eff_K:.2f}"
            )
            iter_diag["teams"][team] = dict(
                sigma=sigma.tolist(),
                nash_conv=float(nc),
                task_entropy=float(H_task),
                effective_K=float(eff_K),
                fingerprints=F.tolist(),
                own_ids=list(own_ids),
            )

            # WandB: meta-strategy metrics per team
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    f"meta/team{team}/nash_conv": float(nc),
                    f"meta/team{team}/task_entropy": float(H_task),
                    f"meta/team{team}/effective_K": float(eff_K),
                    "meta/iteration": self.iteration,
                })

        # WandB: payoff matrix summary stats
        if WANDB_AVAILABLE and wandb.run is not None:
            all_payoffs = [v for v in self.payoff.matrix.values()]
            if all_payoffs:
                n_active = sum(1 for r in self.pool.policies.values() if r.is_active)
                log_dict = {
                    "eval/mean_payoff": float(np.mean(all_payoffs)),
                    "eval/max_payoff": float(np.max(all_payoffs)),
                    "eval/min_payoff": float(np.min(all_payoffs)),
                    "eval/population_size": n_active,
                    "eval/iteration": self.iteration,
                }
                # Per-team max/min win rate
                for team in range(self.n_teams):
                    team_payoffs = [
                        v for (own_id, _), v in self.payoff.matrix.items()
                        if self.pool.policies[own_id].team == team
                    ]
                    if team_payoffs:
                        log_dict[f"eval/team{team}/max_win_rate"] = float(np.max(team_payoffs))
                        log_dict[f"eval/team{team}/min_win_rate"] = float(np.min(team_payoffs))
                wandb.log(log_dict)

        self.diag_history.append(iter_diag)

        # Step 3: Train each active policy against sampled opponents.
        # Snapshot keys so add_policy() below (which mutates self.trainers)
        # does not raise "dictionary changed size during iteration".
        active_ids = list(self.trainers.keys())
        for policy_id in active_ids:
            trainer = self.trainers[policy_id]
            record = self.pool.policies[policy_id]
            if not record.is_active:
                continue

            # Determine opponent based on role
            if record.role == ROLE_MAIN:
                # Main Agent: PFSP (Elo-banded if enabled) against full opponent population
                opponents = self._sample_opponents(policy_id, n_samples=1)
            elif record.role == ROLE_MAIN_EXPLOITER:
                # Main Exploiter: train against opponent's current Main Agent
                opp_main = self.pool.get_active_main(1 - record.team)
                opponents = [opp_main] if opp_main else []
            elif record.role == ROLE_LEAGUE_EXPLOITER:
                # League Exploiter: PFSP (Elo-banded if enabled) against full population
                opponents = self._sample_opponents(policy_id, n_samples=1)
            else:
                opponents = self.pool.sample_uniform(policy_id, n_samples=1)

            if not opponents:
                continue

            # Maybe reset exploiter to earlier checkpoint
            if record.role in [ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER]:
                if np.random.random() < self.exploiter_reset_prob:
                    self._maybe_reset(policy_id, trainer)

            # Train against opponent
            opp_id = opponents[0]
            print(f"  Training {record.role} (team {record.team}, {policy_id}) "
                  f"against {opp_id}...")
            train_metrics = self._train_against(env, trainer, opp_id, policy_id)
            metrics[f"{policy_id}_train"] = train_metrics

            # Save updated checkpoint
            ckpt_name = f"{record.role}_team{record.team}_gen{self.iteration + 1}.pt"
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
            trainer.save(ckpt_path)

            # Add new checkpoint to pool (keeps old versions)
            new_id = self.pool.add_policy(
                team=record.team,
                role=record.role,
                checkpoint_path=ckpt_path,
                generation=self.iteration + 1,
                parent_id=policy_id,
            )
            self.trainers[new_id] = trainer

        elapsed = time.time() - t0
        metrics["elapsed_s"] = elapsed
        print(f"[League] Iteration {self.iteration} complete in {elapsed:.1f}s")

        # Alpha/beta scheduling
        # alpha: 0 → 1 over training (gradually trust team critic)
        # beta_kl: 0.1 → 0 (gradually release KL constraint)
        # When team_critic_enabled=False (Config C ablation), alpha stays at 0.
        total_iters = max(30, 1)  # matches psro_iterations in config
        t = self.iteration / total_iters  # 0→1 normalised time
        if self.team_critic_enabled:
            if self.alpha_schedule == "linear":
                # Linear: α ramps from 0→1 in first 50% of training
                self.alpha = min(1.0, t / 0.5)
            elif self.alpha_schedule == "log":
                # Logarithmic: slow start, steep late — α=log(1+9t)
                # t=0→α=0, t=0.25→α=0.53, t=0.5→α=0.85, t=1→α=1
                import math
                self.alpha = min(1.0, math.log(1.0 + 9.0 * t) / math.log(10.0))
            elif self.alpha_schedule == "adaptive":
                # Adaptive: scales with max per-team mean win rate (wr*2 capped at 1)
                # As agents learn to win, team critic gains influence.
                # Falls back to linear if win rates unavailable.
                max_team_wr = 0.0
                for t_idx in range(self.n_teams):
                    payoffs = [v for (oid, _), v in self.payoff.matrix.items()
                               if self.pool.policies.get(oid)
                               and self.pool.policies[oid].team == t_idx]
                    if payoffs:
                        max_team_wr = max(max_team_wr, float(np.mean(payoffs)))
                self.alpha = min(1.0, max_team_wr * 2.0) if max_team_wr > 0 else min(1.0, t / 0.5)
            else:
                self.alpha = min(1.0, t / 0.5)  # fallback linear
        else:
            self.alpha = 0.0
        self.beta_kl = max(0.0, 0.1 * (1.0 - t))
        metrics["alpha"] = self.alpha
        metrics["beta_kl"] = self.beta_kl
        metrics["alpha_schedule"] = self.alpha_schedule
        print(f"[League] alpha={self.alpha:.3f} (schedule={self.alpha_schedule}) "
              f"beta_kl={self.beta_kl:.3f}", flush=True)

        # WandB: alpha/beta schedule
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "schedule/alpha": self.alpha,
                "schedule/beta_kl": self.beta_kl,
                "schedule/type": self.alpha_schedule,
                "schedule/iteration": self.iteration,
            })

        # Generate mutant policies from top performers
        mutant_ids = self._generate_mutants()
        if mutant_ids:
            metrics["mutants_generated"] = len(mutant_ids)

        self.iteration += 1
        self.pool.save_metadata()

        # Laser task: success-gated kill_radius curriculum annealing.
        # Tighten only when at least one policy pair demonstrates decisive
        # kills this iteration (payoff.last_kill_rate is the best pair's
        # fraction of games that ended in a real win, not a step-cap draw).
        if self.task_type == "laser":
            self._anneal_kill_radius(env)

        return metrics

    def _anneal_kill_radius(self, env):
        """Success-gated kill_radius curriculum (50m → 0.2m).

        Tightens `env.battlefield.laser.kill_radius_m` when the latest eval
        shows a decisive-kill rate ≥ kill_rate_threshold (default 0.5).
        Otherwise holds the current radius.

        This mirrors `train_laser.py`'s anneal logic, but driven by league-
        level eval signal (best-pair kill rate) rather than single-policy
        per-episode kill rate.
        """
        cfg = self.laser_cfg
        kr_final = float(cfg.get("kill_radius_final", 0.2))
        threshold = float(cfg.get("kill_rate_threshold", 0.5))
        decay = float(cfg.get("kill_radius_decay", 0.5))

        eval_kill_rate = float(getattr(self.payoff, "last_kill_rate", 0.0))
        cur_kr = float(env.battlefield.laser.kill_radius_m)
        if eval_kill_rate >= threshold and cur_kr > kr_final:
            new_kr = max(kr_final, cur_kr * decay)
            env.battlefield.laser.kill_radius_m = new_kr
            print(f"[League] kill_radius anneal: {cur_kr:.3f}m → {new_kr:.3f}m "
                  f"(kill_rate={eval_kill_rate:.2f} ≥ {threshold})", flush=True)
        else:
            print(f"[League] kill_radius hold at {cur_kr:.3f}m "
                  f"(kill_rate={eval_kill_rate:.2f} < {threshold})", flush=True)

    def _train_against(
        self,
        env,
        trainer: TeamPPOTrainer,
        opponent_id: str,
        own_policy_id: str,
    ) -> dict:
        """Train one team policy against an opponent for N episodes."""
        opp_trainer = self.trainers.get(opponent_id) or self.mutant_trainers.get(opponent_id)
        record = self.pool.policies[own_policy_id]
        team = record.team
        opp_team = 1 - team

        # Build a dummy opponent trainer if missing — random actions.
        if opp_trainer is None:
            opp_trainer = trainer  # borrow for env API; we won't store its transitions

        # LaserEpisodeRunner owns the pulse loop + CPI buffer.
        # Generic (non-laser) task falls back to the same runner with random
        # physical-layer actions — the existing trainer was never wired to
        # a real env API, so this is the only path that runs end-to-end.
        from training.laser.episode import LaserEpisodeRunner
        runner = LaserEpisodeRunner(
            env, pulses_per_control=self.pulses_per_control, device=self.device,
        )

        total_rewards = 0.0
        wins = 0
        episodes = 0

        for ep in range(self.episodes_per_training):
            runner.reset(red_trainer=trainer, blue_trainer=opp_trainer)
            episode_reward = 0.0
            last_step = 0

            for step in range(self.max_steps_per_episode):
                with torch.no_grad():
                    step_out = runner.step_control(trainer, opp_trainer, deterministic=False)
                result = step_out["result"]
                if result is None:
                    break  # env.step failed internally
                last_step = step

                # Credit the PREVIOUS control step's actions (timing: action →
                # env.step → reward → store). First control step has no prior.
                if not step_out["first_step"]:
                    # Pre-check: if radar buffer would overflow on this call (it
                    # adds E entries), trigger update() first to flush it.
                    E = env.num_envs
                    if trainer.radar_buffer and \
                            trainer.radar_buffer.ptr + E >= trainer.radar_buffer.buffer_size:
                        update_metrics = trainer.update(
                            team_critic=self.team_critic if self.team_critic_enabled else None,
                            alpha=self.alpha,
                            beta_kl=self.beta_kl,
                            n_step=self.n_step_returns,
                            team_critic_optimizer=self.team_critic_optimizer if self.team_critic_enabled else None,
                        )
                        if WANDB_AVAILABLE and wandb.run is not None:
                            self._log_ppo_metrics(update_metrics, record, episodes)
                        # Surface PPO update events to stdout so the log shows
                        # actual gradient updates firing (effectiveness signal).
                        cmd_m = update_metrics.get("commander", {}) or {}
                        rad_m = update_metrics.get("radar", {}) or {}
                        if cmd_m or rad_m:
                            print(f"      [PPO] {record.role[:4]}-t{record.team} "
                                  f"ep{episodes} step{step} "
                                  f"cmd(pl={cmd_m.get('policy_loss', 0):.4f} "
                                  f"vl={cmd_m.get('value_loss', 0):.4f} "
                                  f"ent={cmd_m.get('entropy', 0):.4f}) "
                                  f"rad(pl={rad_m.get('policy_loss', 0):.4f} "
                                  f"vl={rad_m.get('value_loss', 0):.4f} "
                                  f"ent={rad_m.get('entropy', 0):.4f})",
                                  flush=True)

                    own_transition = (
                        step_out["red_transition_prev"] if team == 0
                        else step_out["blue_transition_prev"]
                    )
                    reward_info = trainer.store_transition(
                        env, result, own_transition, team,
                    )
                    r_per_team = env.n_radars // env.n_teams
                    r_start = team * r_per_team
                    r_end = r_start + r_per_team
                    episode_reward += reward_info["radar_reward"][:, r_start:r_end].sum().item()

                done_mask = result["dones"]
                if done_mask.any():
                    winners = result["winners"]
                    for e in range(env.num_envs):
                        if done_mask[e] and winners[e] == team:
                            wins += 1
                    break

            total_rewards += episode_reward
            episodes += 1

            wr = wins / max(episodes, 1)
            avg_r = total_rewards / max(episodes, 1)

            # Per-episode completion: step count tells us if episode ended early.
            ended_naturally = last_step + 1 < self.max_steps_per_episode
            status = "kill" if ended_naturally else "timeout"
            print(f"    ep {episodes}/{self.episodes_per_training}  "
                  f"steps={last_step + 1} {status}  wr={wr:.2f}  "
                  f"avg_r={avg_r:.4f}", flush=True)

            # Terminal progress every 5 episodes (more detailed summary)
            if episodes % 5 == 0:
                print(f"    ── ep {episodes}/{self.episodes_per_training}  "
                      f"wr={wr:.2f}  avg_r={avg_r:.4f}",
                      flush=True)

            # WandB update every episode
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    f"train/{record.role}_team{record.team}/episode": episodes,
                    f"train/{record.role}_team{record.team}/win_rate": float(wr),
                    f"train/{record.role}_team{record.team}/avg_reward": float(avg_r),
                    "train/iteration": self.iteration,
                })

            if episodes % 10 == 0:
                update_metrics = trainer.update(
                    team_critic=self.team_critic,
                    alpha=self.alpha,
                    beta_kl=self.beta_kl,
                    n_step=self.n_step_returns,
                    team_critic_optimizer=self.team_critic_optimizer,
                )
                if WANDB_AVAILABLE and wandb.run is not None and update_metrics:
                    self._log_ppo_metrics(update_metrics, record, episodes)

        return {
            "episodes": episodes,
            "win_rate": wins / max(episodes, 1),
            "avg_reward": total_rewards / max(episodes, 1),
        }

    def _log_ppo_metrics(self, update_metrics: dict, record, episode: int):
        """Log PPO update metrics to WandB with policy context."""
        prefix = f"ppo/{record.role}_team{record.team}"
        cmd = update_metrics.get("commander", {})
        radar = update_metrics.get("radar", {})
        log_dict = {"ppo/episode": episode, "ppo/iteration": self.iteration}
        if cmd:
            for k, v in cmd.items():
                log_dict[f"{prefix}/commander/{k}"] = float(v)
        if radar:
            for k, v in radar.items():
                log_dict[f"{prefix}/radar/{k}"] = float(v)
        wandb.log(log_dict)

    def _maybe_reset(self, policy_id: str, trainer: TeamPPOTrainer):
        """Maybe reset exploiter to an earlier checkpoint."""
        record = self.pool.policies[policy_id]
        if record.parent_id and record.parent_id in self.pool.policies:
            parent = self.pool.policies[record.parent_id]
            if os.path.exists(parent.checkpoint_path):
                print(f"  Resetting {policy_id} to parent checkpoint")
                trainer.load(parent.checkpoint_path)

    def _generate_mutants(self) -> list:
        """Generate mutant policies by perturbing weights of top-performing policies.

        For each team, identifies the top-K policies by mean cross-team win rate
        (from the payoff matrix), then creates N perturbed copies of each.
        Mutants use role "mutant", are frozen, and serve only as opponents.

        Returns:
            List of newly created mutant policy_ids.
        """
        cfg = self.mutation_config
        if not cfg.get("enabled", False):
            return []

        epsilon = float(cfg.get("epsilon", 0.05))
        n_mutants = int(cfg.get("n_mutants_per_policy", 3))
        n_top = int(cfg.get("n_top_policies", 2))
        wr_threshold = float(cfg.get("win_rate_threshold", 0.55))

        mutant_ids = []

        for team in range(self.n_teams):
            payoff_mat, own_ids, _opp_ids = self.payoff.get_submatrix(team)

            if len(own_ids) == 0:
                continue

            mean_wrs = payoff_mat.mean(axis=1)

            eligible = []
            for i, pid in enumerate(own_ids):
                record = self.pool.policies[pid]
                if record.role == "mutant":
                    continue
                if mean_wrs[i] >= wr_threshold:
                    eligible.append((pid, float(mean_wrs[i])))

            eligible.sort(key=lambda x: x[1], reverse=True)
            top_policies = eligible[:n_top]

            for base_policy_id, wr in top_policies:
                base_record = self.pool.policies[base_policy_id]

                # Load base checkpoint
                base_ckpt = torch.load(
                    base_record.checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )

                for mutant_idx in range(n_mutants):
                    # Fresh network, then load + perturb
                    policy_dict = create_team_policy(
                        team=team,
                        n_elem=self.n_elem,
                        n_pulses=self.n_pulses,
                        n_bins=self.n_bins,
                        num_output_length=self.num_output_length,
                        device=self.device,
                        sub_array_size=self.sub_array_size,
                        hybrid_fire=self.hybrid_fire,
                        decouple_value=self.decouple_value,
                    )
                    policy_dict["commander"].load_state_dict(base_ckpt["commander"])
                    policy_dict["radar"].load_state_dict(base_ckpt["radar"])

                    with torch.no_grad():
                        for ac in [policy_dict["commander"], policy_dict["radar"]]:
                            for param in ac.parameters():
                                param.add_(torch.randn_like(param) * epsilon)

                    # Inference-only trainer (no buffer init)
                    mutant_trainer = TeamPPOTrainer(
                        commander=policy_dict["commander"],
                        radar=policy_dict["radar"],
                        **self.ppo_config,
                    )

                    ckpt_name = (
                        f"mutant_team{team}_"
                        f"from_{base_policy_id}_"
                        f"v{mutant_idx}_gen{self.iteration + 1}.pt"
                    )
                    ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
                    mutant_trainer.save(ckpt_path)

                    mutant_id = self.pool.add_policy(
                        team=team,
                        role="mutant",
                        checkpoint_path=ckpt_path,
                        generation=self.iteration + 1,
                        parent_id=base_policy_id,
                    )
                    self.mutant_trainers[mutant_id] = mutant_trainer
                    mutant_ids.append(mutant_id)

                    print(
                        f"  [mutant] Created {mutant_id} from {base_policy_id} "
                        f"(team={team}, wr={wr:.3f}, eps={epsilon}, "
                        f"{mutant_idx + 1}/{n_mutants})",
                        flush=True,
                    )

        if mutant_ids:
            print(f"[League] Generated {len(mutant_ids)} mutant policies "
                  f"in iteration {self.iteration}", flush=True)
        return mutant_ids

    def pretrain_critic(self, data: dict, n_epochs: int = 50, batch_size: int = 256):
        """Pre-train all critic value heads via supervised regression on MC returns.

        Uses collected rollout data from scripted policies. Only pre-trains the
        critic (both deployment value_head and privileged_value_head) — the actor
        weights are unchanged.

        Args:
            data: dict from RolloutDataCollector.collect() with keys:
                  obs, returns, privileged_infos, commander_obs, commander_returns
            n_epochs: number of supervised training epochs
            batch_size: minibatch size
        """
        import torch.nn.functional as F

        obs = data["obs"].to(self.device)
        returns = data["returns"].to(self.device)
        priv = data["privileged_infos"].to(self.device)

        print(f"\n[CriticPretrain] Pre-training critics on {obs.shape[0]} transitions "
              f"({n_epochs} epochs)...", flush=True)

        # Pre-train each trainer's critic heads
        for policy_id, trainer in self.trainers.items():
            record = self.pool.policies[policy_id]
            if not record.is_active:
                continue

            ac = trainer.radar_trainer.ac
            opt = trainer.radar_trainer.optimizer

            # Prepare minibatch sampler
            T = obs.shape[0]
            indices = torch.randperm(T)

            for epoch in range(n_epochs):
                epoch_loss = 0.0
                epoch_priv_loss = 0.0
                n_batches = 0

                for start in range(0, T, batch_size):
                    batch_idx = indices[start:start + batch_size]
                    batch_obs = obs[batch_idx]
                    batch_returns = returns[batch_idx].unsqueeze(-1)  # [B, 1]
                    batch_priv = priv[batch_idx]

                    # Forward pass: get both value estimates
                    _, _, _, value, privileged_value = ac._get_distributions(
                        batch_obs, batch_priv,
                    )

                    # Deployment value head loss
                    value_loss = F.mse_loss(value, batch_returns)

                    # Privileged value head loss
                    priv_value_loss = F.mse_loss(privileged_value, batch_returns)

                    total_loss = value_loss + priv_value_loss

                    opt.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                    opt.step()

                    epoch_loss += value_loss.item()
                    epoch_priv_loss += priv_value_loss.item()
                    n_batches += 1

                avg_vl = epoch_loss / max(n_batches, 1)
                avg_pvl = epoch_priv_loss / max(n_batches, 1)
                if epoch % 10 == 0 or epoch == n_epochs - 1:
                    print(f"  [{policy_id}] epoch {epoch}/{n_epochs}: "
                          f"value_loss={avg_vl:.4f} priv_loss={avg_pvl:.4f}",
                          flush=True)
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        f"pretrain/{policy_id}/value_loss": avg_vl,
                        f"pretrain/{policy_id}/priv_loss": avg_pvl,
                        f"pretrain/{policy_id}/epoch": epoch,
                    })

        print("[CriticPretrain] Done.", flush=True)

    def pretrain_commander_critic(self, data: dict, n_epochs: int = 50,
                                   batch_size: int = 256):
        """Pre-train commander value head via supervised regression on MC returns.

        Uses collected commander rollout data from scripted HPEDF policies.
        The commander value_head learns to predict discounted returns from
        commander observations — giving it calibrated value estimates before
        PPO training begins.

        This is separate from pretrain_critic() which only handles radar critics.
        """
        import torch.nn.functional as F

        cmd_obs = data["commander_obs"].to(self.device)           # [T, 76]
        cmd_returns = data["commander_returns"].to(self.device)    # [T]

        print(f"\n[CriticPretrain] Pre-training commander critics on "
              f"{cmd_obs.shape[0]} transitions ({n_epochs} epochs)...", flush=True)

        for policy_id, trainer in self.trainers.items():
            record = self.pool.policies[policy_id]
            if not record.is_active:
                continue

            ac = trainer.commander_trainer.ac
            opt = trainer.commander_trainer.optimizer

            T = cmd_obs.shape[0]
            for epoch in range(n_epochs):
                indices = torch.randperm(T, device=self.device)
                total_loss = 0.0
                n_batches = 0

                for start in range(0, T, batch_size):
                    end = min(start + batch_size, T)
                    idx = indices[start:end]

                    batch_obs = cmd_obs[idx]
                    batch_returns = cmd_returns[idx]

                    _, value = ac(batch_obs)
                    loss = F.mse_loss(value.squeeze(), batch_returns)

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                    opt.step()

                    total_loss += loss.item()
                    n_batches += 1

                if epoch % 10 == 0 or epoch == n_epochs - 1:
                    avg_loss = total_loss / max(n_batches, 1)
                    print(f"  [{policy_id}] epoch {epoch}/{n_epochs - 1}: "
                          f"cmd_value_loss={avg_loss:.4f}", flush=True)

        print("[CriticPretrain] Commander critics done.", flush=True)

    def pretrain_commander_bc(self, data: dict, n_epochs: int = 30,
                               batch_size: int = 128):
        """Pre-train commander actor via BC on HPEDF demonstration actions.

        Trains action_head (pre-tanh mean) using weighted MSE. Focus on
        dims [0:3] (launch_flag + target_x + target_y) with higher weight.
        Dims [3:35] (radar instructions) are trained to output 0.5 (neutral).

        The commander uses tanh-squashed Gaussian policy, so we train the
        pre-tanh mean to match atanh(clipped_target_action).
        """
        import torch.nn.functional as F

        cmd_obs = data["commander_obs"].to(self.device)           # [T, 76]
        cmd_actions = data["commander_actions"].to(self.device)    # [T, 35]

        # Only train on transitions with meaningful commander actions
        active_mask = cmd_actions.abs().sum(dim=-1) > 0.01
        n_active = active_mask.sum().item()
        if n_active == 0:
            print("[BCPretrain] WARNING: no active commander transitions, skipping",
                  flush=True)
            return

        active_obs = cmd_obs[active_mask]
        active_actions = cmd_actions[active_mask]

        print(f"\n[BCPretrain] Commander BC on {n_active}/{cmd_obs.shape[0]} "
              f"transitions ({n_epochs} epochs)...", flush=True)

        for policy_id, trainer in self.trainers.items():
            record = self.pool.policies[policy_id]
            if not record.is_active:
                continue

            ac = trainer.commander_trainer.ac
            opt = trainer.commander_trainer.optimizer

            # Weighted MSE: launch dim gets 5x, target gets 2x
            dim_weights = torch.ones(35, device=self.device)
            dim_weights[0] = 5.0   # launch_flag
            dim_weights[1] = 2.0   # target_x
            dim_weights[2] = 2.0   # target_y

            T = active_obs.shape[0]
            for epoch in range(n_epochs):
                indices = torch.randperm(T, device=self.device)
                total_loss = 0.0
                n_batches = 0

                for start in range(0, T, batch_size):
                    end = min(start + batch_size, T)
                    idx = indices[start:end]

                    batch_obs = active_obs[idx]
                    batch_actions = active_actions[idx]

                    features = ac.shared(batch_obs)
                    mean = ac.action_head(features)

                    # atanh target to match Gaussian policy architecture
                    clipped = batch_actions.clamp(-0.99, 0.99)
                    raw_target = 0.5 * torch.log(
                        (1.0 + clipped) / (1.0 - clipped + 1e-6)
                    )

                    loss = (dim_weights * (mean - raw_target) ** 2).mean()

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                    opt.step()

                    total_loss += loss.item()
                    n_batches += 1

                if epoch % 10 == 0 or epoch == n_epochs - 1:
                    avg_loss = total_loss / max(n_batches, 1)
                    print(f"  [{policy_id}] epoch {epoch}/{n_epochs - 1}: "
                          f"cmd_bc_loss={avg_loss:.4f}", flush=True)

        print("[BCPretrain] Commander BC done.", flush=True)

    def pretrain_actor_bc(self, data: dict, n_epochs: int = 30, batch_size: int = 128):
        """Pre-train actor via behavior cloning on scripted demonstration actions.

        The actor learns to imitate the scripted policy's task selection
        (cross-entropy) and continuous parameters (MSE). This gives the actor
        a reasonable starting point before PPO fine-tuning.

        Args:
            data: dict from RolloutDataCollector.collect() with keys:
                  obs, actions, privileged_infos
            n_epochs: number of supervised training epochs
            batch_size: minibatch size
        """
        import torch.nn.functional as F

        obs = data["obs"].to(self.device)
        target_actions = data["actions"].to(self.device)
        priv = data["privileged_infos"].to(self.device)

        print(f"\n[BCPretrain] Behavior cloning on {obs.shape[0]} transitions "
              f"({n_epochs} epochs)...", flush=True)

        for policy_id, trainer in self.trainers.items():
            record = self.pool.policies[policy_id]
            if not record.is_active:
                continue

            ac = trainer.radar_trainer.ac
            opt = trainer.radar_trainer.optimizer

            T = obs.shape[0]
            indices = torch.randperm(T)
            N = ac.n_elem
            K = ac.n_sub

            # Loss weights: task loss (cross-entropy) typically dominates
            # because it's log-loss over 4 classes × K sub-arrays.
            task_loss_weight = 1.0
            param_loss_weight = 0.1
            vehicle_loss_weight = 0.01

            for epoch in range(n_epochs):
                epoch_task_loss = 0.0
                epoch_param_loss = 0.0
                epoch_veh_loss = 0.0
                n_batches = 0

                for start in range(0, T, batch_size):
                    batch_idx = indices[start:start + batch_size]
                    batch_obs = obs[batch_idx]
                    batch_act = target_actions[batch_idx]
                    batch_priv = priv[batch_idx]

                    # Extract target sub-array actions from flat element actions
                    target_task_frac, target_params, target_vehicle = \
                        ac._extract_sub_from_elem(batch_act)

                    # Get model distributions (forward pass)
                    task_dist, param_dist, vehicle_dist, _, _ = \
                        ac._get_distributions(batch_obs, batch_priv)

                    # Task loss: cross-entropy (teacher-forcing)
                    target_task = target_task_frac.argmax(dim=-1)  # [B, K]
                    task_logp = task_dist.log_prob(target_task)     # [B, K]
                    task_loss = -task_logp.mean()

                    # Param loss: MSE between model mean and target params
                    param_loss = F.mse_loss(param_dist.mean, target_params)

                    # Vehicle loss: MSE
                    vehicle_loss = F.mse_loss(vehicle_dist.mean, target_vehicle)

                    total_loss = (
                        task_loss_weight * task_loss
                        + param_loss_weight * param_loss
                        + vehicle_loss_weight * vehicle_loss
                    )

                    opt.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                    opt.step()

                    epoch_task_loss += task_loss.item()
                    epoch_param_loss += param_loss.item()
                    epoch_veh_loss += vehicle_loss.item()
                    n_batches += 1

                avg_task = epoch_task_loss / max(n_batches, 1)
                avg_param = epoch_param_loss / max(n_batches, 1)
                avg_veh = epoch_veh_loss / max(n_batches, 1)
                if epoch % 10 == 0 or epoch == n_epochs - 1:
                    print(f"  [{policy_id}] epoch {epoch}/{n_epochs}: "
                          f"task={avg_task:.4f} "
                          f"param={avg_param:.4f} "
                          f"veh={avg_veh:.4f}",
                          flush=True)
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        f"bc_pretrain/{policy_id}/task_loss": avg_task,
                        f"bc_pretrain/{policy_id}/param_loss": avg_param,
                        f"bc_pretrain/{policy_id}/veh_loss": avg_veh,
                        f"bc_pretrain/{policy_id}/epoch": epoch,
                    })

        print("[BCPretrain] Done.", flush=True)

    def _sample_opponents(self, policy_id: str, n_samples: int = 1) -> list:
        """Pick opponents via Elo-band PFSP if enabled, else standard PFSP."""
        if self.elo_sampler is not None:
            return self.elo_sampler.sample(
                policy_id, iteration=self.iteration, n_samples=n_samples,
            )
        return self.pool.sample_pfsp(policy_id, n_samples=n_samples)

    def get_final_agent(self, team: int) -> str:
        """Get the best policy ID for deployment (meta-strategy weighted)."""
        if team in self.meta_strategies:
            own_ids = [
                pid for pid, rec in self.pool.policies.items()
                if rec.team == team and rec.is_active
            ]
            weights = self.meta_strategies[team]
            if len(own_ids) == len(weights):
                best_idx = np.argmax(weights)
                return own_ids[best_idx]
        return self.pool.get_active_main(team)

    def save(self):
        """Save full league state."""
        state = {
            "iteration": self.iteration,
            "meta_strategies": {k: v.tolist() for k, v in self.meta_strategies.items()},
            "ppo_config": self.ppo_config,
            "tcdams_lambda": self.tcdams_lambda,
            "use_elo_band": self.use_elo_band,
            "meta_solver_name": self.meta_solver_name,
        }
        torch.save(state, os.path.join(self.checkpoint_dir, "league_state.pt"))
        self.pool.save_metadata()
        # Persist Elo and per-iteration diagnostics as plain JSON for analysis.
        if self.elo_sampler is not None:
            self.elo_sampler.save(os.path.join(self.checkpoint_dir, "elo.json"))
        try:
            import json
            with open(os.path.join(self.checkpoint_dir, "diag_history.json"), "w") as f:
                json.dump(self.diag_history, f, indent=2)
        except Exception as exc:
            print(f"[League] WARN: failed to write diag_history: {exc}")

    def load(self):
        """Load full league state."""
        state_path = os.path.join(self.checkpoint_dir, "league_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.iteration = state["iteration"]
            self.meta_strategies = {
                int(k): np.array(v) for k, v in state["meta_strategies"].items()
            }
        self.pool.load_metadata()

        # Reconstruct mutant trainers from pool
        self.mutant_trainers = {}
        for pid, record in self.pool.policies.items():
            if record.role != "mutant":
                continue
            if not os.path.exists(record.checkpoint_path):
                print(f"[League] WARN: mutant checkpoint missing: "
                      f"{record.checkpoint_path}")
                continue
            policy_dict = create_team_policy(
                team=record.team,
                n_elem=self.n_elem,
                n_pulses=self.n_pulses,
                n_bins=self.n_bins,
                num_output_length=self.num_output_length,
                device=self.device,
                sub_array_size=self.sub_array_size,
                hybrid_fire=self.hybrid_fire,
                decouple_value=self.decouple_value,
            )
            mutant_trainer = TeamPPOTrainer(
                commander=policy_dict["commander"],
                radar=policy_dict["radar"],
                **self.ppo_config,
            )
            mutant_trainer.load(record.checkpoint_path)
            self.mutant_trainers[pid] = mutant_trainer
        if self.mutant_trainers:
            print(f"[League] Reconstructed {len(self.mutant_trainers)} "
                  f"mutant trainers from disk")

        if self.elo_sampler is not None:
            self.elo_sampler.load(os.path.join(self.checkpoint_dir, "elo.json"))
