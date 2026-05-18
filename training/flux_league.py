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
from typing import Dict, Optional

from .ppo.actor_critic import create_team_policy
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
        commander_clip: float = 0.2,
        radar_clip: float = 0.1,
        commander_entropy: float = 0.01,
        radar_entropy: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
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
            device=device,
        )

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
        os.makedirs(checkpoint_dir, exist_ok=True)

    def initialize(self, env):
        """Create initial 6 policies (3 roles × 2 teams) and register in pool.

        Args:
            env: MFARVecEnv instance (used to get obs/action dims)
        """
        self.payoff = PayoffMatrix(
            self.pool, self.n_eval_games, self.device,
            max_steps_per_game=self.max_steps_per_episode,
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
                )
                trainer = TeamPPOTrainer(
                    commander=policy_dict["commander"],
                    radar=policy_dict["radar"],
                    **self.ppo_config,
                )
                trainer.init_buffers(env.state_dim, env.action_dim)

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
        self.payoff.evaluate_all(env, self.trainers)

        # Step 2: Compute meta-strategies
        if self.elo_sampler is not None:
            self.elo_sampler.update_from_payoff_matrix(self.payoff.matrix)
        iter_diag = {"iteration": self.iteration, "teams": {}}
        for team in range(self.n_teams):
            payoff_mat, own_ids, opp_ids = self.payoff.get_submatrix(team)
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

        self.iteration += 1
        self.pool.save_metadata()

        return metrics

    def _train_against(
        self,
        env,
        trainer: TeamPPOTrainer,
        opponent_id: str,
        own_policy_id: str,
    ) -> dict:
        """Train one team policy against an opponent for N episodes."""
        opp_trainer = self.trainers.get(opponent_id)
        record = self.pool.policies[own_policy_id]
        team = record.team
        opp_team = 1 - team

        total_rewards = 0.0
        wins = 0
        episodes = 0

        for ep in range(self.episodes_per_training):
            env.reset()
            episode_reward = 0.0

            for step in range(self.max_steps_per_episode):
                with torch.no_grad():
                    # Own team: real policy inference
                    own = trainer.get_own_actions(env, team)

                    # Opponent team: real policy or random fallback
                    actions = torch.zeros(
                        env.num_envs, env.n_radars, env.action_dim, device=self.device,
                    )
                    for i, r in enumerate(range(own["r_start"], own["r_end"])):
                        actions[:, r, :] = own["radar_actions"][i]

                    if opp_trainer:
                        opp = opp_trainer.get_own_actions(env, opp_team)
                        for i, r in enumerate(range(opp["r_start"], opp["r_end"])):
                            actions[:, r, :] = opp["radar_actions"][i]
                    else:
                        opp_r_start = opp_team * (env.n_radars // env.n_teams)
                        opp_r_end = opp_r_start + (env.n_radars // env.n_teams)
                        actions[:, opp_r_start:opp_r_end, :] = torch.rand(
                            env.num_envs, opp_r_end - opp_r_start, env.action_dim,
                            device=self.device,
                        )

                    commander_actions = torch.zeros(
                        env.num_envs, env.n_teams,
                        env.battlefield.commander_action_dim, device=self.device,
                    )
                    commander_actions[:, team, :] = own["commander_action"]
                    if opp_trainer:
                        commander_actions[:, opp_team, :] = opp["commander_action"]
                    else:
                        commander_actions[:, opp_team, :] = (
                            torch.rand(
                                env.num_envs, env.battlefield.commander_action_dim,
                                device=self.device,
                            ) * 2 - 1
                        )

                result = env.step(actions=actions, commander_actions=commander_actions)

                # Store transitions for training team only
                reward_info = trainer.store_transition(env, result, own["transition"], team)
                episode_reward += reward_info["radar_reward"][
                    :, own["r_start"]:own["r_end"]
                ].sum().item()

                if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
                   (trainer.radar_buffer and trainer.radar_buffer.near_full):
                    trainer.update()

                if result["dones"].any():
                    if result["winners"][0] == team:
                        wins += 1
                    break

            total_rewards += episode_reward
            episodes += 1

            if episodes % 10 == 0:
                trainer.update()

        return {
            "episodes": episodes,
            "win_rate": wins / max(episodes, 1),
            "avg_reward": total_rewards / max(episodes, 1),
        }

    def _maybe_reset(self, policy_id: str, trainer: TeamPPOTrainer):
        """Maybe reset exploiter to an earlier checkpoint."""
        record = self.pool.policies[policy_id]
        if record.parent_id and record.parent_id in self.pool.policies:
            parent = self.pool.policies[record.parent_id]
            if os.path.exists(parent.checkpoint_path):
                print(f"  Resetting {policy_id} to parent checkpoint")
                trainer.load(parent.checkpoint_path)

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
        if self.elo_sampler is not None:
            self.elo_sampler.load(os.path.join(self.checkpoint_dir, "elo.json"))
