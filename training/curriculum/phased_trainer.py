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
import functools
import builtins
from typing import Dict, Optional

from ..flux_league import FluxLeague, ROLE_MAIN, ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER
from ..ppo.actor_critic import create_team_policy
from ..ppo.ppo_trainer import TeamPPOTrainer

# Force every print() in this module to flush immediately so progress lines
# land in the log file even when stdout is piped through Tee on Windows.
print = functools.partial(builtins.print, flush=True)


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

        For each task (detect, recon, jam), query the radar policy for
        beam/params, then override the task assignment to the fixed task.
        Compute corrected log_probs for the modified action and store
        transitions so PPO actually updates the network.
        """
        env = self.env_factory()
        tasks = [0, 1, 2]  # recon, detect, jam
        task_names = ["recon", "detect", "jam"]

        for task_idx, task_name in zip(tasks, task_names):
            print(f"\n  Pre-training task: {task_name}")
            policy = create_team_policy(
                0, device=self.device,
                n_elem=self.league.n_elem,
                n_pulses=self.league.n_pulses,
                n_bins=self.league.n_bins,
                num_output_length=self.league.num_output_length,
                sub_array_size=self.league.sub_array_size,
            )
            trainer = TeamPPOTrainer(
                commander=policy["commander"],
                radar=policy["radar"],
                **self.league.ppo_config,
            )
            trainer.init_buffers(env.state_dim, env.action_dim)

            max_steps = getattr(self.league, "max_steps_per_episode", 1000)
            for ep in range(self.phase_a_episodes // 3):
                env.reset()
                for step in range(max_steps):
                    with torch.no_grad():
                        state, commander_obs = trainer._get_observations(env)

                        actions = torch.zeros(
                            env.num_envs, env.n_radars, env.action_dim,
                            device=self.device,
                        )
                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim,
                            device=self.device,
                        )

                        # Commander actions for all teams
                        for team in range(env.n_teams):
                            cmd_obs = commander_obs[:, team, :]
                            cmd_act, _, _ = trainer.commander_trainer.ac.get_action(cmd_obs)
                            commander_actions[:, team, :] = cmd_act

                        # Radar actions: query policy, override task
                        rep_obs = rep_action = None
                        for r in range(env.n_radars):
                            radar_obs = state[:, r, :]
                            radar_act, _, _ = trainer.radar_trainer.ac.get_action(radar_obs)

                            # Override task assignment to fixed task
                            for e in range(env.n_elem):
                                base = e * 22
                                radar_act[:, base:base + 4].zero_()
                                radar_act[:, base + task_idx] = 1.0

                            actions[:, r, :] = radar_act

                            if r == 0:
                                rep_obs = radar_obs
                                rep_action = radar_act.clone()

                    result = env.step(actions=actions, commander_actions=commander_actions)

                    # Compute corrected log_probs for modified actions
                    with torch.no_grad():
                        rep_logp, _, rep_val = trainer.radar_trainer.ac.evaluate_actions(
                            rep_obs, rep_action,
                        )
                        cmd_obs_t0 = commander_obs[:, 0, :]
                        cmd_act_t0 = commander_actions[:, 0, :]
                        cmd_logp, _, cmd_val = trainer.commander_trainer.ac.evaluate_actions(
                            cmd_obs_t0, cmd_act_t0,
                        )

                    transition = {
                        "cmd_obs": cmd_obs_t0,
                        "cmd_action": cmd_act_t0,
                        "cmd_logp": cmd_logp,
                        "cmd_val": cmd_val.squeeze(-1),
                        "radar_obs": rep_obs,
                        "radar_action": rep_action,
                        "radar_logp": rep_logp,
                        "radar_val": rep_val.squeeze(-1),
                    }
                    trainer.store_transition(env, result, transition, team=0)

                    # Flush buffers before they overflow. Cheap when not near full.
                    if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
                       (trainer.radar_buffer and trainer.radar_buffer.near_full):
                        trainer.update()

                    if result["dones"].any():
                        break

                if ep % 5 == 0:
                    metrics = trainer.update()
                    print(f"    Episode {ep}: {metrics}", flush=True)

            # Save pre-trained checkpoint
            ckpt_path = os.path.join(
                self.league.checkpoint_dir, f"pretrain_{task_name}.pt",
            )
            trainer.save(ckpt_path)

    def run_phase_b(self):
        """Phase B: Multi-task integration with dense reward shaping.

        Enable full action space (task selection + params). Train against
        random opponents with heavy reward shaping. Own team uses real
        policy inference, opponent uses random actions.
        """
        env = self.env_factory()
        self.league.initialize(env)

        for policy_id, trainer in self.league.trainers.items():
            if not self.league.pool.policies[policy_id].is_active:
                continue

            record = self.league.pool.policies[policy_id]
            team = record.team
            opp_team = 1 - team
            r_per_team = env.n_radars // env.n_teams
            opp_r_start = opp_team * r_per_team
            opp_r_end = opp_r_start + r_per_team

            print(f"\n  Training {record.role} team {team} ({policy_id})")

            wins = 0
            for ep in range(self.phase_b_episodes):
                env.reset()

                for step in range(getattr(self.league, "max_steps_per_episode", 1000)):
                    with torch.no_grad():
                        # Own team: real policy
                        own = trainer.get_own_actions(env, team)

                        # Opponent: random
                        actions = torch.zeros(
                            env.num_envs, env.n_radars, env.action_dim,
                            device=self.device,
                        )
                        for i, r in enumerate(range(own["r_start"], own["r_end"])):
                            actions[:, r, :] = own["radar_actions"][i]
                        actions[:, opp_r_start:opp_r_end, :] = torch.rand(
                            env.num_envs, opp_r_end - opp_r_start, env.action_dim,
                            device=self.device,
                        )

                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim,
                            device=self.device,
                        )
                        commander_actions[:, team, :] = own["commander_action"]
                        commander_actions[:, opp_team, :] = (
                            torch.rand(
                                env.num_envs, env.battlefield.commander_action_dim,
                                device=self.device,
                            ) * 2 - 1
                        )

                    result = env.step(actions=actions, commander_actions=commander_actions)
                    trainer.store_transition(env, result, own["transition"], team)

                    if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
                       (trainer.radar_buffer and trainer.radar_buffer.near_full):
                        trainer.update()

                    if result["dones"].any():
                        if result["winners"][0] == team:
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
                f"{record.role}_team{team}_phaseB.pt",
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

        Exploiters train against the opponent's main agent using real
        policy inference for both sides. Finds weaknesses in main agents.
        """
        env = self.env_factory()

        for policy_id, trainer in self.league.trainers.items():
            record = self.league.pool.policies[policy_id]
            if not record.is_active:
                continue
            if record.role not in [ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER]:
                continue

            team = record.team
            opp_team = 1 - team
            r_per_team = env.n_radars // env.n_teams
            opp_r_start = opp_team * r_per_team
            opp_r_end = opp_r_start + r_per_team

            # Get opponent's main agent trainer
            opp_main_id = self.league.pool.get_active_main(opp_team)
            opp_trainer = self.league.trainers.get(opp_main_id) if opp_main_id else None

            print(f"\n  Refining {record.role} team {team} ({policy_id})")

            wins = 0
            for ep in range(self.phase_d_episodes):
                env.reset()

                for step in range(getattr(self.league, "max_steps_per_episode", 1000)):
                    with torch.no_grad():
                        # Exploiter: real policy
                        own = trainer.get_own_actions(env, team)

                        actions = torch.zeros(
                            env.num_envs, env.n_radars, env.action_dim,
                            device=self.device,
                        )
                        for i, r in enumerate(range(own["r_start"], own["r_end"])):
                            actions[:, r, :] = own["radar_actions"][i]

                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim,
                            device=self.device,
                        )
                        commander_actions[:, team, :] = own["commander_action"]

                        # Opponent: real main agent or random fallback
                        if opp_trainer:
                            opp = opp_trainer.get_own_actions(env, opp_team)
                            for i, r in enumerate(range(opp["r_start"], opp["r_end"])):
                                actions[:, r, :] = opp["radar_actions"][i]
                            commander_actions[:, opp_team, :] = opp["commander_action"]
                        else:
                            actions[:, opp_r_start:opp_r_end, :] = torch.rand(
                                env.num_envs, opp_r_end - opp_r_start, env.action_dim,
                                device=self.device,
                            )
                            commander_actions[:, opp_team, :] = (
                                torch.rand(
                                    env.num_envs, env.battlefield.commander_action_dim,
                                    device=self.device,
                                ) * 2 - 1
                            )

                    result = env.step(actions=actions, commander_actions=commander_actions)
                    trainer.store_transition(env, result, own["transition"], team)

                    if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
                       (trainer.radar_buffer and trainer.radar_buffer.near_full):
                        trainer.update()

                    if result["dones"].any():
                        if result["winners"][0] == team:
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
        """Run final win rate evaluation using deterministic policies."""
        r_per_team = env.n_radars // env.n_teams

        for team in range(self.league.n_teams):
            agent_id = self.league.get_final_agent(team)
            trainer = self.league.trainers.get(agent_id)
            if not trainer:
                print(f"\n  Team {team} final agent: {agent_id} (no trainer)")
                continue

            opp_team = 1 - team
            opp_r_start = opp_team * r_per_team
            opp_r_end = opp_r_start + r_per_team

            print(f"\n  Team {team} final agent: {agent_id}")

            wins = 0
            n_games = 100
            for game in range(n_games):
                env.reset()
                for step in range(getattr(self.league, "max_steps_per_episode", 1000)):
                    with torch.no_grad():
                        own = trainer.get_own_actions(env, team, deterministic=True)

                        actions = torch.zeros(
                            env.num_envs, env.n_radars, env.action_dim,
                            device=self.device,
                        )
                        for i, r in enumerate(range(own["r_start"], own["r_end"])):
                            actions[:, r, :] = own["radar_actions"][i]
                        actions[:, opp_r_start:opp_r_end, :] = torch.rand(
                            env.num_envs, opp_r_end - opp_r_start, env.action_dim,
                            device=self.device,
                        )

                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim,
                            device=self.device,
                        )
                        commander_actions[:, team, :] = own["commander_action"]
                        commander_actions[:, opp_team, :] = (
                            torch.rand(
                                env.num_envs, env.battlefield.commander_action_dim,
                                device=self.device,
                            ) * 2 - 1
                        )

                    result = env.step(actions=actions, commander_actions=commander_actions)
                    if result["dones"].any():
                        if result["winners"][0] == team:
                            wins += 1
                        break

            print(f"  Win rate: {wins}/{n_games} = {wins / n_games:.2%}")
