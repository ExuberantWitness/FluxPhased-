"""FluxPhased training CLI entry point.

Usage:
    python -m training.train --config configs/league.yaml
    python -m training.train --config configs/league.yaml --phase c
    python -m training.train --resume checkpoints/league/league_state.pt
"""

import argparse
import os
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import sys
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from radar_sim.gpu.vec_mfar_env import MFARVecEnv
from radar_sim.config import DEFAULT_ROWS, DEFAULT_COLS
from training.flux_league import FluxLeague
from training.curriculum.phased_trainer import PhasedTrainer


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_env(config: dict) -> MFARVecEnv:
    """Create MFARVecEnv from config."""
    env_cfg = config.get("env", {})
    kwargs = dict(
        num_envs=env_cfg.get("num_envs", 4),
        n_radars=env_cfg.get("n_radars", 4),
        rows=env_cfg.get("rows", DEFAULT_ROWS),
        cols=env_cfg.get("cols", DEFAULT_COLS),
        pulses_per_cpi=env_cfg.get("pulses_per_cpi", 4),
        fft_size=env_cfg.get("fft_size", 64),
        device=env_cfg.get("device", "cuda"),
        tx_power_w=env_cfg.get("tx_power_w", 1.0),
        n_teams=env_cfg.get("n_teams", 2),
        bandwidth=env_cfg.get("bandwidth", 200e6),
        prf=env_cfg.get("prf", 10e3),
        vehicle_speed_ms=env_cfg.get("vehicle_speed_ms", 244.4),
    )
    # Laser task extra args (only pass if config explicitly provides them)
    for k in ("kill_radius_m", "illumination_time_s", "drone_altitude_m", "map_size"):
        if k in env_cfg:
            kwargs[k] = env_cfg[k]
    return MFARVecEnv(**kwargs)


def compute_env_params(config: dict) -> dict:
    """Compute environment dimensions from config without GPU allocation.

    Avoids creating a full MFARVecEnv just to read n_elem / n_bins / etc.
    """
    env_cfg = config.get("env", {})
    rows = env_cfg.get("rows", DEFAULT_ROWS)
    cols = env_cfg.get("cols", DEFAULT_COLS)
    n_elem = rows * cols
    n_pulses = env_cfg.get("pulses_per_cpi", 4)
    n_radars = env_cfg.get("n_radars", 4)
    n_teams = env_cfg.get("n_teams", 2)
    num_output_length = env_cfg.get("num_output_length", 16)
    fft_size = env_cfg.get("fft_size", 64)
    device = env_cfg.get("device", "cuda")

    # n_bins: VecElementProcessor uses fft_size when > 0, else n_samples
    if fft_size > 0:
        n_bins = fft_size
    else:
        bandwidth = env_cfg.get("bandwidth", 200e6)
        prf = env_cfg.get("prf", 10e3)
        n_samples = max(1, int((1.0 / prf) * bandwidth))
        n_bins = n_samples

    return dict(
        n_elem=n_elem, n_pulses=n_pulses, n_bins=n_bins,
        n_radars=n_radars, n_teams=n_teams,
        num_output_length=num_output_length, device=device,
    )


def create_league(config: dict, env_params: dict) -> FluxLeague:
    """Create FluxLeague from config and pre-computed env params."""
    league_cfg = config.get("league", {})
    ppo_cfg = config.get("ppo", {})
    shared = ppo_cfg.get("shared", {})
    cmd = ppo_cfg.get("commander", {})
    radar = ppo_cfg.get("radar", {})

    return FluxLeague(
        n_elem=env_params["n_elem"],
        n_pulses=env_params["n_pulses"],
        n_bins=env_params["n_bins"],
        num_output_length=env_params["num_output_length"],
        n_teams=env_params["n_teams"],
        population_cap=league_cfg.get("population_cap", 20),
        n_eval_games=league_cfg.get("n_eval_games", 50),
        meta_solver=league_cfg.get("meta_solver", "nash"),
        pfsp_temperature=league_cfg.get("pfsp_temperature", 1.0),
        exploiter_reset_prob=league_cfg.get("exploiter_reset_prob", 0.1),
        episodes_per_training=league_cfg.get("episodes_per_training", 1000),
        max_steps_per_episode=league_cfg.get("max_steps_per_episode", 1000),
        checkpoint_dir=league_cfg.get("checkpoint_dir", "checkpoints/league"),
        device=env_params["device"],
        sub_array_size=config.get("sub_array_size", 0),
        commander_lr=cmd.get("lr", 3e-4),
        radar_lr=radar.get("lr", 1e-4),
        gamma=shared.get("gamma", 0.99),
        gae_lambda=shared.get("gae_lambda", 0.95),
        n_step_returns=shared.get("n_step_returns", 0),
        commander_clip=cmd.get("clip_range", 0.2),
        radar_clip=radar.get("clip_range", 0.1),
        commander_entropy=cmd.get("entropy_coef", 0.01),
        radar_entropy=radar.get("entropy_coef", 0.02),
        value_coef=shared.get("value_coef", 0.5),
        max_grad_norm=shared.get("max_grad_norm", 0.5),
        n_epochs=shared.get("n_epochs", 10),
        batch_size=shared.get("batch_size", 64),
        buffer_size=shared.get("buffer_size", 2048),
        buffer_size_commander=shared.get("buffer_size_commander", 2048),
        buffer_size_radar=shared.get("buffer_size_radar", 64),
        stealth_weight=config.get("reward_shaping", {}).get("stealth_weight", 0.1),
        reward_shaping_config=config.get("reward_shaping", {}),
        tcdams_lambda=league_cfg.get("tcdams_lambda", 0.3),
        use_elo_band=league_cfg.get("use_elo_band", False),
        elo_band_init=league_cfg.get("elo_band_init", 400.0),
        elo_band_final=league_cfg.get("elo_band_final", 100.0),
        elo_anneal_iters=league_cfg.get("elo_anneal_iters", 15),
        mutation_config=league_cfg.get("mutation", {}),
        task_type=config.get("task_type", "generic"),
        pulses_per_control=config.get("env", {}).get("pulses_per_control", 5),
        laser_cfg={
            "kill_radius_init": config.get("training", {}).get("kill_radius_init", 50.0),
            "kill_radius_final": config["env"].get("kill_radius_m", 0.2),
            "kill_rate_threshold": config.get("training", {}).get("kill_rate_threshold", 0.5),
            "kill_radius_decay": config.get("training", {}).get("kill_radius_decay", 0.5),
            "residual_scale_m": config.get("training", {}).get("residual_scale_m", 6.0),
            "reward_shaping": config.get("reward_shaping", {}),
            "hybrid_fire": config.get("training", {}).get("hybrid_fire", False),
            "decouple_value": config.get("training", {}).get("decouple_value", False),
            # Critical: without residual_aim=True, aim is not anchored to enemy
            # obs → hybrid_fire's zero-init aim-head is meaningless. Without
            # min_radar_baseline_m, enforce_radar_baseline is a no-op → near-collinear
            # radar geometry → info-matrix singular → fused estimate explodes →
            # clamped to map corner → kill_radius never met → progress=0 → 0.5.
            "residual_aim": config.get("training", {}).get("residual_aim", False),
            "min_radar_baseline_m": config.get("env", {}).get("min_radar_baseline_m", 0.0),
        },
        sensing_cfg=config.get("sensing_noise", {}),
    )
    # ── TeamCritic toggle (Config C disables for ablation) ──
    league.team_critic_enabled = league_cfg.get("team_critic_enabled", True)
    league.alpha_schedule = league_cfg.get("alpha_schedule", "linear")
    rsc = config.get("reward_shaping", {})
    league.team_reward_weight = rsc.get("team_reward_weight", 0.1)
    league.team_kill_weight = rsc.get("team_kill_weight", 1.0)
    return league


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
    parser.add_argument("--sub-array-size", type=int, default=0,
                        help="Sub-array block size (e.g., 5 for 5×5 blocks in a 25×25 array). "
                             "0 = per-element policy (default)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override a config field with KEY=VALUE (dot-separated key, "
             "e.g. league.meta_solver=tc_dams). Repeatable.",
    )
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    # Load config
    config = load_config(args.config)
    # Apply --override KEY=VALUE entries.
    for spec in args.override:
        if "=" not in spec:
            raise SystemExit(f"--override must be KEY=VALUE, got: {spec}")
        key, raw_val = spec.split("=", 1)
        # Parse value (yaml handles bool/int/float/string cleanly)
        val = yaml.safe_load(raw_val)
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
        print(f"[config] override {key}={val!r}")
    if args.device:
        config["env"]["device"] = args.device
    if args.sub_array_size > 0:
        config["sub_array_size"] = args.sub_array_size

    print(f"FluxPhased League Training")
    print(f"Config: {args.config}")
    print(f"Device: {config['env'].get('device', 'cuda')}")

    # Compute env dimensions from config WITHOUT creating a GPU env.
    # Creating two MFARVecEnv instances simultaneously (one here, one in
    # run_phase_a) would OOM on large 25×25 configs (~36 GB VRAM each).
    env_params = compute_env_params(config)
    print(f"Environment: E={config['env'].get('num_envs', 4)}, "
          f"R={env_params['n_radars']}, "
          f"N={env_params['n_elem']}, P={env_params['n_pulses']}, "
          f"bins={env_params['n_bins']}")
    n_elem = env_params['n_elem']
    state_dim = (n_elem * (env_params['n_pulses'] * env_params['n_bins'] + 2 + 4)
                 + 5 + 12 + env_params['num_output_length'])
    print(f"State dim: {state_dim}, Action dim: {n_elem * 22 + 3}")

    # --- WandB init ---
    if WANDB_AVAILABLE:
        run_name = f"league_{env_params['n_elem']}elem_{config['env'].get('num_envs', 4)}env_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.login(key="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", relogin=True, verify=True)
        wandb.init(
            project="fluxphased",
            name=run_name,
            config={
                "env": config.get("env", {}),
                "league": config.get("league", {}),
                "ppo": config.get("ppo", {}),
                "training": config.get("training", {}),
                "reward_shaping": config.get("reward_shaping", {}),
                "sub_array_size": config.get("sub_array_size", 0),
                "state_dim": state_dim,
                "action_dim": n_elem * 22 + 3,
            },
            save_code=False,
        )
        print(f"[wandb] Initialized run: {run_name}")
    else:
        print("[wandb] wandb not available; skipping logging")

    # Create league (no GPU env needed)
    league = create_league(config, env_params)

    if args.resume:
        print(f"Resuming from {args.resume}")
        league.load()
        # Re-create trainers from pool — need a real env for init_buffers().
        # Create, initialize, and immediately destroy to avoid holding VRAM.
        env = create_env(config)
        league.initialize(env)
        env.destroy()
        del env
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Create trainer (pure PSRO, no hardcoded phases)
    training_cfg = config.get("training", {})
    trainer = PhasedTrainer(
        env_factory=lambda: create_env(config),
        league=league,
        n_psro_iterations=training_cfg.get("psro_iterations", 30),
        episodes_per_iter=training_cfg.get(
            "episodes_per_iter", league.episodes_per_training,
        ),
        warmup_episodes=training_cfg.get("warmup_episodes", 0),
        critic_pretrain_episodes=training_cfg.get("critic_pretrain_episodes", 0),
        critic_pretrain_epochs=training_cfg.get("critic_pretrain_epochs", 50),
        bc_pretrain_epochs=training_cfg.get("bc_pretrain_epochs", 0),
        bc_pretrain_batch_size=training_cfg.get("bc_pretrain_batch_size", 128),
        device=config["env"].get("device", "cuda"),
    )

    trainer.run_all()


if __name__ == "__main__":
    main()
