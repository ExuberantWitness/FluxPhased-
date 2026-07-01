"""Pure PSRO curriculum trainer for FluxLeague.

Replaces the old 4-phase pipeline (A: single-task pretrain, B: multi-task
integration, C: PSRO, D: exploiter refinement) with a single PSRO loop.

The asymmetric critic + bidirectional dense reward (including stealth
penalty) provide enough signal to bootstrap from scratch — no policy-
constraint hacks needed.  Task diversity emerges naturally through
TC-DAMS meta-strategy solving.
"""

import os
import time
import torch
import functools
import builtins
from typing import Dict, Optional

from ..flux_league import FluxLeague, ROLE_MAIN, ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER

print = functools.partial(builtins.print, flush=True)


class PhasedTrainer:
    """Orchestrates pure-PSRO training with optional environment-side warmup."""

    def __init__(
        self,
        env_factory,  # callable returning MFARVecEnv
        league: FluxLeague,
        n_psro_iterations: int = 30,
        episodes_per_iter: int = 1000,
        warmup_episodes: int = 0,
        critic_pretrain_episodes: int = 0,
        critic_pretrain_epochs: int = 50,
        bc_pretrain_epochs: int = 0,
        bc_pretrain_batch_size: int = 128,
        log_dir: str = "logs/curriculum",
        device: str = "cuda",
    ):
        self.env_factory = env_factory
        self.league = league
        self.n_psro_iterations = n_psro_iterations
        self.episodes_per_iter = episodes_per_iter
        self.warmup_episodes = warmup_episodes
        self.critic_pretrain_episodes = critic_pretrain_episodes
        self.critic_pretrain_epochs = critic_pretrain_epochs
        self.bc_pretrain_epochs = bc_pretrain_epochs
        self.bc_pretrain_batch_size = bc_pretrain_batch_size
        self.log_dir = log_dir
        self.device = device
        self.scripted_policy_name = "hpedf"  # "hpedf" or "legacy"

        os.makedirs(log_dir, exist_ok=True)

    def run_all(self):
        """Execute pure-PSRO training with optional critic/actor pre-training."""
        import gc

        # 1. Initialize league (creates initial policies + trainers)
        env = self.env_factory()
        self.league.initialize(env)

        # 2. Optional critic pre-training with scripted demonstration data
        if self.critic_pretrain_episodes > 0:
            print("=" * 60)
            print(f"Critic Pre-training: {self.critic_pretrain_episodes} demo episodes")
            print("=" * 60)
            self._run_critic_pretrain(env)

        # 3. BC actor pre-training is now done within _run_critic_pretrain
        #    (reuses collected data to avoid OOM from duplicate collection)

        # 4. Optional environment-side warmup (weak opponents, no policy constraint)
        if self.warmup_episodes > 0:
            print("=" * 60)
            print(f"Environment Warmup: {self.warmup_episodes} episodes")
            print("=" * 60)
            self._run_warmup(env)

        # 5. Pure PSRO loop
        for it in range(self.n_psro_iterations):
            print(f"\n{'=' * 60}")
            print(f"PSRO Iteration {it}/{self.n_psro_iterations}")
            print(f"{'=' * 60}")

            # Use existing PSRO iteration implementation in FluxLeague
            self.league.episodes_per_training = self.episodes_per_iter
            metrics = self.league.psro_iteration(env)

            # Log metrics
            log_path = os.path.join(self.log_dir, f"psro_iter_{it:03d}.json")
            import json
            with open(log_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            # Periodically save league
            if it % 5 == 0 or it == self.n_psro_iterations - 1:
                self.league.save()
                print(f"  Checkpoint saved at iteration {it}")

        # 4. Final save
        self.league.save()
        print("\nTraining complete. Final league saved.")

    def _run_warmup(self, env):
        """Optional environment-side warmup with passive opponents.

        Unlike the old Phase A, there is NO policy constraint or log_prob
        correction.  The agent explores freely; opponents are passive
        (heuristic: recon-only, no transmission) so the agent can collect
        basic reward signal before facing real adversaries.

        This is purely optional — if the asymmetric critic + bidirectional
        dense reward provide enough signal, warmup_episodes can stay at 0.
        """
        n_teams = self.league.n_teams
        r_per_team = env.n_radars // n_teams
        max_steps = getattr(self.league, "max_steps_per_episode", 1000)

        for ep in range(self.warmup_episodes):
            env.reset()

            for policy_id, trainer in self.league.trainers.items():
                record = self.league.pool.policies[policy_id]
                if not record.is_active:
                    continue
                team = record.team

                for step in range(max_steps):
                    with torch.no_grad():
                        own = trainer.get_own_actions(env, team)

                        # Own team: real policy
                        actions = torch.zeros(
                            env.num_envs, env.n_radars, env.action_dim,
                            device=self.device,
                        )
                        for i, r in enumerate(range(own["r_start"], own["r_end"])):
                            actions[:, r, :] = own["radar_actions"][i]

                        # Opponent: passive heuristic (all recon, no transmit)
                        for opp_team in range(n_teams):
                            if opp_team == team:
                                continue
                            opp_r0 = opp_team * r_per_team
                            opp_r1 = opp_r0 + r_per_team
                            for r in range(opp_r0, opp_r1):
                                # All-recon action: task_frac[0]=1, others 0
                                a = torch.zeros(env.num_envs, env.action_dim,
                                                device=self.device)
                                for e in range(env.n_elem):
                                    a[:, e * 22] = 1.0  # recon task
                                actions[:, r, :] = a

                        commander_actions = torch.zeros(
                            env.num_envs, n_teams,
                            env.battlefield.commander_action_dim,
                            device=self.device,
                        )
                        commander_actions[:, team, :] = own["commander_action"]

                    result = env.step(actions=actions, commander_actions=commander_actions)
                    trainer.store_transition(env, result, own["transition"], team)

                    if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
                       (trainer.radar_buffer and trainer.radar_buffer.near_full):
                        trainer.update()

                    if result["dones"].any():
                        break

                # End-of-episode update
                trainer.update()

            if ep % 5 == 0:
                print(f"  Warmup episode {ep}/{self.warmup_episodes}")

    def _run_critic_pretrain(self, env):
        """Generate demonstration data and pre-train all critic value heads.

        Uses scripted (heuristic) policies to collect rollout trajectories,
        computes Monte Carlo returns, then trains both value_head and
        privileged_value_head via MSE regression.
        """
        from ..data_collector import RolloutDataCollector
        from ..scripted_policy import (
            scripted_radar_policy, scripted_commander_policy,
            hpedf_radar_policy, hpedf_commander_policy, _hpedf_scheduler,
        )

        policy_name = getattr(self, "scripted_policy_name", "hpedf")
        if policy_name == "hpedf":
            use_radar_policy = hpedf_radar_policy
            use_commander_policy = hpedf_commander_policy
        else:
            use_radar_policy = scripted_radar_policy
            use_commander_policy = scripted_commander_policy

        # Enable HPEDF augmentation noise for diverse demo data
        prev_augment = _hpedf_scheduler.augment_noise
        _hpedf_scheduler.augment_noise = 1.0

        collector = RolloutDataCollector(
            device=self.device,
            augment_noise=1.0,  # full augmentation for critic/BC pretraining
        )
        n_teams = self.league.n_teams

        for team in range(n_teams):
            print(f"\n  Collecting demo data for team {team}...", flush=True)

            # ── Chunked collection + incremental pretraining ──
            # 50 episodes × 2000 steps × 700KB/transition ≈ 280 GB on CPU.
            # Collect in chunks of 10, train, free, repeat to stay within RAM.
            chunk_size = 5   # 5 episodes ≈ 14 GB CPU — safe for 91 GB system
            total_needed = self.critic_pretrain_episodes
            total_episodes = 0
            coverage_pass = False

            for chunk_start in range(0, total_needed, chunk_size):
                n_ep = min(chunk_size, total_needed - chunk_start)
                print(f"  [chunk {chunk_start // chunk_size}] Collecting {n_ep} episodes...",
                      flush=True)
                data = collector.collect(
                    env=env,
                    scripted_policy=use_radar_policy,
                    scripted_commander=use_commander_policy,
                    team=team,
                    n_episodes=n_ep,
                    max_steps=getattr(self.league, "max_steps_per_episode", 1000),
                )
                total_episodes += n_ep

                cov = data["coverage"]
                print(f"  Launch rate: {cov['launch_pct']:.0f}% | Scenes: {cov['scene_counts']}",
                      flush=True)
                if cov["ok"]:
                    coverage_pass = True

                # ── Incremental pretraining on this chunk ──
                team_trainers = {
                    pid: t for pid, t in self.league.trainers.items()
                    if self.league.pool.policies[pid].team == team
                }
                all_trainers = self.league.trainers
                self.league.trainers = team_trainers
                self.league.pretrain_critic(
                    data,
                    n_epochs=self.critic_pretrain_epochs,
                    batch_size=256,
                )
                self.league.pretrain_commander_critic(
                    data,
                    n_epochs=getattr(self, "critic_pretrain_epochs", 50),
                    batch_size=256,
                )

                if self.bc_pretrain_epochs > 0:
                    self.league.pretrain_actor_bc(
                        data,
                        n_epochs=self.bc_pretrain_epochs,
                        batch_size=self.bc_pretrain_batch_size,
                    )
                    self.league.pretrain_commander_bc(
                        data,
                        n_epochs=self.bc_pretrain_epochs,
                        batch_size=self.bc_pretrain_batch_size,
                    )
                    for trainer in team_trainers.values():
                        trainer.set_bc_pretrained()

                self.league.trainers = all_trainers
                del data
                torch.cuda.empty_cache()

            if coverage_pass:
                print(f"  ✓ Coverage passed after {total_episodes} episodes", flush=True)
            else:
                print(f"  ⚠ Coverage not fully met after {total_episodes} episodes "
                      f"— proceeding anyway", flush=True)
            print(f"  BC models snapshotted for KL penalty", flush=True)

        # ── Restore HPEDF scheduler augmentation setting ──
        _hpedf_scheduler.augment_noise = prev_augment

        print("  Critic pre-training complete.", flush=True)

    def _run_bc_pretrain(self, env):
        """Pre-train actor networks via behavior cloning on scripted demonstrations.

        Uses the same scripted policies as critic pre-training. The actor
        learns to imitate task selection (cross-entropy) and continuous
        params (MSE). After BC, PPO fine-tunes with good initial behavior.
        """
        from ..data_collector import RolloutDataCollector
        from ..scripted_policy import (
            scripted_radar_policy, scripted_commander_policy,
            hpedf_radar_policy, hpedf_commander_policy, _hpedf_scheduler,
        )

        policy_name = getattr(self, "scripted_policy_name", "hpedf")
        if policy_name == "hpedf":
            use_radar_policy = hpedf_radar_policy
            use_commander_policy = hpedf_commander_policy
        else:
            use_radar_policy = scripted_radar_policy
            use_commander_policy = scripted_commander_policy

        # Enable HPEDF augmentation noise for diverse demo data
        prev_augment = _hpedf_scheduler.augment_noise
        _hpedf_scheduler.augment_noise = 1.0

        collector = RolloutDataCollector(
            device=self.device,
            augment_noise=1.0,  # full augmentation for critic/BC pretraining
        )
        n_teams = self.league.n_teams

        for team in range(n_teams):
            print(f"\n  Collecting demo data for team {team} (BC)...", flush=True)
            data = collector.collect(
                env=env,
                scripted_policy=use_radar_policy,
                scripted_commander=use_commander_policy,
                team=team,
                n_episodes=max(self.critic_pretrain_episodes, 50),
                max_steps=getattr(self.league, "max_steps_per_episode", 1000),
            )

            team_trainers = {
                pid: t for pid, t in self.league.trainers.items()
                if self.league.pool.policies[pid].team == team
            }
            all_trainers = self.league.trainers
            self.league.trainers = team_trainers
            self.league.pretrain_actor_bc(
                data,
                n_epochs=self.bc_pretrain_epochs,
                batch_size=self.bc_pretrain_batch_size,
            )
            self.league.pretrain_commander_bc(
                data,
                n_epochs=self.bc_pretrain_epochs,
                batch_size=self.bc_pretrain_batch_size,
            )
            # ── Snapshot BC-pretrained actors for KL penalty ──
            for trainer in team_trainers.values():
                trainer.set_bc_pretrained()
            self.league.trainers = all_trainers

            del data
            torch.cuda.empty_cache()

        # ── Restore HPEDF scheduler augmentation setting ──
        _hpedf_scheduler.augment_noise = prev_augment

        print("  BC actor pre-training complete.", flush=True)
