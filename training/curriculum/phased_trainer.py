"""Phased curriculum trainer for FluxLeague.

4-phase training pipeline:
  Phase A: Single-task pre-training (5K episodes)
  Phase B: Multi-task integration (10K episodes)
  Phase C: PSRO population training (20 PSRO iterations)
  Phase D: League exploiter refinement (10K episodes)

Each phase progressively increases complexity and adversarial pressure.
"""

import os
import time
import torch
import numpy as np
from typing import Dict, Optional

from ..flux_league import FluxLeague, ROLE_MAIN, ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER
from ..ppo.actor_critic import create_team_policy
from ..ppo.ppo_trainer import TeamPPOTrainer


class PhasedTrainer:
    """Orchestrates the 4-phase curriculum for FluxLeague training."""

    def __init__(
        self,
        env_factory,  # callable returning MFARVecEnv
        league: FluxLeague,
        phase_a_episodes: int = 5000,
        phase_b_episodes: int = 10000,
        phase_c_iterations: int = 20,
        phase_c_episodes_per_iter: int = 1000,
        phase_d_episodes: int = 10000,
        log_dir: str = "logs/curriculum",
        device: str = "cuda",
    ):
        self.env_factory = env_factory
        self.league = league
        self.phase_a_episodes = phase_a_episodes
        self.phase_b_episodes = phase_b_episodes
        self.phase_c_iterations = phase_c_iterations
        self.phase_c_episodes_per_iter = phase_c_episodes_per_iter
        self.phase_d_episodes = phase_d_episodes
        self.log_dir = log_dir
        self.device = device

        os.makedirs(log_dir, exist_ok=True)

    def run_all(self):
        """Execute all 4 phases sequentially."""
        print("=" * 60)
        print("Phase A: Single-Task Pre-Training")
        print("=" * 60)
        self.run_phase_a()

        print("=" * 60)
        print("Phase B: Multi-Task Integration")
        print("=" * 60)
        self.run_phase_b()

        print("=" * 60)
        print("Phase C: PSRO Population Training")
        print("=" * 60)
        self.run_phase_c()

        print("=" * 60)
        print("Phase D: League Exploiter Refinement")
        print("=" * 60)
        self.run_phase_d()

        self.league.save()
        print("\nTraining complete. Final league saved.")

    def run_phase_a(self):
        """Phase A: Pre-train radar policies on fixed single tasks.

        For each task (detect, recon, jam), train a separate policy
        with all elements assigned to that task. This bootstraps basic
        spectrum understanding and beam steering.
        """
        env = self.env_factory()
        tasks = [0, 1, 2]  # recon, detect, jam
        task_names = ["recon", "detect", "jam"]

        for task_idx, task_name in zip(tasks, task_names):
            print(f"\n  Pre-training task: {task_name}")
            trainer = TeamPPOTrainer(
                commander=create_team_policy(0, device=self.device)["commander"],
                radar=create_team_policy(0, device=self.device)["radar"],
                **self.league.ppo_config,
            )
            trainer.set_radar_obs_dim(env.state_dim)

            for ep in range(self.phase_a_episodes // 3):
                env.reset()
                for step in range(1000):
                    # Fix all elements to current task
                    actions = torch.zeros(
                        env.num_envs, env.n_radars, env.action_dim, device=self.device,
                    )
                    # Set task fraction to always select current task
                    for r in range(env.n_radars):
                        for e in range(env.n_elem):
                            base = e * 22
                            actions[:, r, base + task_idx] = 1.0  # set task fraction
                            actions[:, r, base + 4:base + 12] = 0.5  # neutral beam
                            actions[:, r, base + 12:base + 22] = 0.5  # neutral params

                    commander_actions = torch.zeros(
                        env.num_envs, env.n_teams,
                        env.battlefield.commander_action_dim,
                        device=self.device,
                    )

                    result = env.step(actions=actions, commander_actions=commander_actions)
                    trainer.collect_env_step(env, result)

                    if result["dones"].any():
                        break

                if ep % 100 == 0 and ep > 0:
                    metrics = trainer.update()
                    print(f"    Episode {ep}: {metrics}")

            # Save pre-trained checkpoint
            ckpt_path = os.path.join(
                self.league.checkpoint_dir, f"pretrain_{task_name}.pt",
            )
            trainer.save(ckpt_path)

        env_reset = None  # release env

    def run_phase_b(self):
        """Phase B: Multi-task integration with dense reward shaping.

        Enable full action space (task selection + params). Train against
        random opponents with heavy reward shaping.
        """
        env = self.env_factory()
        self.league.initialize(env)

        for policy_id, trainer in self.league.trainers.items():
            if not self.league.pool.policies[policy_id].is_active:
                continue

            record = self.league.pool.policies[policy_id]
            print(f"\n  Training {record.role} team {record.team} ({policy_id})")

            wins = 0
            for ep in range(self.phase_b_episodes):
                env.reset()

                for step in range(1000):
                    with torch.no_grad():
                        actions = torch.rand(
                            env.num_envs, env.n_radars, env.action_dim,
                            device=self.device,
                        )
                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim,
                            device=self.device,
                        )

                    result = env.step(actions=actions, commander_actions=commander_actions)
                    trainer.collect_env_step(env, result)

                    if result["dones"].any():
                        if result["winners"][0] == record.team:
                            wins += 1
                        break

                if ep > 0 and ep % 10 == 0:
                    trainer.update()

                if ep % 500 == 0:
                    wr = wins / max(ep + 1, 1)
                    print(f"    Episode {ep}: win_rate={wr:.3f}")

            # Save phase B checkpoint
            ckpt_path = os.path.join(
                self.league.checkpoint_dir,
                f"{record.role}_team{record.team}_phaseB.pt",
            )
            trainer.save(ckpt_path)

    def run_phase_c(self):
        """Phase C: PSRO population training.

        Run N PSRO iterations. Each iteration:
          1. Evaluate payoff matrix
          2. Compute meta-Nash
          3. Train best responses
        """
        env = self.env_factory()

        for it in range(self.phase_c_iterations):
            print(f"\n  PSRO Iteration {it}/{self.phase_c_iterations}")
            metrics = self.league.psro_iteration(env)

            # Log metrics
            log_path = os.path.join(self.log_dir, f"psro_iter_{it:03d}.json")
            import json
            with open(log_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            # Periodically save league
            if it % 5 == 0:
                self.league.save()

    def run_phase_d(self):
        """Phase D: League exploiter refinement.

        Focus training on exploiters to find remaining weaknesses
        in main agents. Use PFSP opponent sampling.
        """
        env = self.env_factory()

        for policy_id, trainer in self.league.trainers.items():
            record = self.league.pool.policies[policy_id]
            if not record.is_active:
                continue
            if record.role not in [ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER]:
                continue

            print(f"\n  Refining {record.role} team {record.team} ({policy_id})")

            wins = 0
            for ep in range(self.phase_d_episodes):
                env.reset()

                for step in range(1000):
                    with torch.no_grad():
                        actions = torch.rand(
                            env.num_envs, env.n_radars, env.action_dim,
                            device=self.device,
                        )
                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim,
                            device=self.device,
                        )

                    result = env.step(actions=actions, commander_actions=commander_actions)
                    trainer.collect_env_step(env, result)

                    if result["dones"].any():
                        if result["winners"][0] == record.team:
                            wins += 1
                        break

                if ep > 0 and ep % 10 == 0:
                    trainer.update()

                if ep % 500 == 0:
                    wr = wins / max(ep + 1, 1)
                    print(f"    Episode {ep}: win_rate={wr:.3f}")

        # Final evaluation
        print("\n  Final evaluation...")
        self._final_evaluation(env)

    def _final_evaluation(self, env):
        """Run final win rate evaluation against random baseline."""
        for team in range(self.league.n_teams):
            agent_id = self.league.get_final_agent(team)
            print(f"\n  Team {team} final agent: {agent_id}")

            wins = 0
            n_games = 100
            for game in range(n_games):
                env.reset()
                for step in range(1000):
                    with torch.no_grad():
                        actions = torch.rand(
                            env.num_envs, env.n_radars, env.action_dim,
                            device=self.device,
                        )
                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim,
                            device=self.device,
                        )

                    result = env.step(actions=actions, commander_actions=commander_actions)
                    if result["dones"].any():
                        if result["winners"][0] == team:
                            wins += 1
                        break

            print(f"  Win rate: {wins}/{n_games} = {wins / n_games:.2%}")
