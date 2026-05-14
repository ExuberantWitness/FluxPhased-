"""FluxPhased training CLI entry point.

Usage:
    python -m training.train --config configs/league.yaml
    python -m training.train --config configs/league.yaml --phase c
    python -m training.train --resume checkpoints/league/league_state.pt
"""

import argparse
import os
import sys
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from radar_sim.gpu.vec_mfar_env import MFARVecEnv
from training.flux_league import FluxLeague
from training.curriculum.phased_trainer import PhasedTrainer


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_env(config: dict) -> MFARVecEnv:
    """Create MFARVecEnv from config."""
    env_cfg = config.get("env", {})
    return MFARVecEnv(
        num_envs=env_cfg.get("num_envs", 4),
        n_radars=env_cfg.get("n_radars", 4),
        rows=env_cfg.get("rows", 5),
        cols=env_cfg.get("cols", 5),
        pulses_per_cpi=env_cfg.get("pulses_per_cpi", 4),
        fft_size=env_cfg.get("fft_size", 64),
        device=env_cfg.get("device", "cuda"),
    )


def create_league(config: dict, env: MFARVecEnv) -> FluxLeague:
    """Create FluxLeague from config."""
    league_cfg = config.get("league", {})
    ppo_cfg = config.get("ppo", {})
    shared = ppo_cfg.get("shared", {})
    cmd = ppo_cfg.get("commander", {})
    radar = ppo_cfg.get("radar", {})

    # Determine n_bins from env
    n_bins = env.n_bins

    return FluxLeague(
        n_elem=env.n_elem,
        n_pulses=env.n_pulses,
        n_bins=n_bins,
        num_output_length=env.num_output_length,
        n_teams=env.n_teams,
        population_cap=league_cfg.get("population_cap", 20),
        n_eval_games=league_cfg.get("n_eval_games", 50),
        meta_solver=league_cfg.get("meta_solver", "nash"),
        pfsp_temperature=league_cfg.get("pfsp_temperature", 1.0),
        exploiter_reset_prob=league_cfg.get("exploiter_reset_prob", 0.1),
        episodes_per_training=league_cfg.get("episodes_per_training", 1000),
        max_steps_per_episode=league_cfg.get("max_steps_per_episode", 1000),
        checkpoint_dir=league_cfg.get("checkpoint_dir", "checkpoints/league"),
        device=env.device if hasattr(env, "device") else "cuda",
        commander_lr=cmd.get("lr", 3e-4),
        radar_lr=radar.get("lr", 1e-4),
        gamma=shared.get("gamma", 0.99),
        gae_lambda=shared.get("gae_lambda", 0.95),
        commander_clip=cmd.get("clip_range", 0.2),
        radar_clip=radar.get("clip_range", 0.1),
        commander_entropy=cmd.get("entropy_coef", 0.01),
        radar_entropy=radar.get("entropy_coef", 0.02),
        value_coef=shared.get("value_coef", 0.5),
        max_grad_norm=shared.get("max_grad_norm", 0.5),
        n_epochs=shared.get("n_epochs", 10),
        batch_size=shared.get("batch_size", 64),
        buffer_size=shared.get("buffer_size", 2048),
    )


def main():
    parser = argparse.ArgumentParser(description="FluxPhased League Training")
    parser.add_argument("--config", type=str, default="configs/league.yaml",
                        help="Path to league config YAML")
    parser.add_argument("--phase", type=str, default=None,
                        choices=["a", "b", "c", "d"],
                        help="Run specific phase only (default: all phases)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to league state checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    # Load config
    config = load_config(args.config)
    if args.device:
        config["env"]["device"] = args.device

    print(f"FluxPhased League Training")
    print(f"Config: {args.config}")
    print(f"Device: {config['env'].get('device', 'cuda')}")

    # Create environment
    env = create_env(config)
    print(f"Environment: E={env.num_envs}, R={env.n_radars}, "
          f"N={env.n_elem}, P={env.n_pulses}, bins={env.n_bins}")
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")

    # Create league
    league = create_league(config, env)

    if args.resume:
        print(f"Resuming from {args.resume}")
        league.load()
        # Re-create trainers from pool
        env_for_init = env
        league.initialize(env_for_init)

    # Create curriculum trainer
    curriculum_cfg = config.get("curriculum", {})
    trainer = PhasedTrainer(
        env_factory=lambda: create_env(config),
        league=league,
        phase_a_episodes=curriculum_cfg.get("phase_a_episodes", 5000),
        phase_b_episodes=curriculum_cfg.get("phase_b_episodes", 10000),
        phase_c_iterations=curriculum_cfg.get("phase_c_iterations", 20),
        phase_c_episodes_per_iter=curriculum_cfg.get("phase_c_episodes_per_iter", 1000),
        phase_d_episodes=curriculum_cfg.get("phase_d_episodes", 10000),
        device=config["env"].get("device", "cuda"),
    )

    # Run selected phase(s)
    if args.phase == "a":
        trainer.run_phase_a()
    elif args.phase == "b":
        trainer.run_phase_b()
    elif args.phase == "c":
        trainer.run_phase_c()
    elif args.phase == "d":
        trainer.run_phase_d()
    else:
        trainer.run_all()


if __name__ == "__main__":
    main()
