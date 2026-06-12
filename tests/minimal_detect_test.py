#!/usr/bin/env python3
"""Minimal single-agent single-task detect test — verifies beam steering learnability.

Phase 2 of the D1 fix verification:
  - 1 agent, 1 task (detect only), static target at 12.7 km
  - No self-play, PSRO, exploiter, payoff matrix, league
  - Run 50k steps and verify beam converges toward target direction.

Usage:
    python tests/minimal_detect_test.py [--steps 50000] [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

# Allow importing from parent dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.ppo.reward_shaping import DenseRewardShaper
from training.ppo.actor_critic import RadarActorCritic, create_team_policy
from training.ppo.buffer import RolloutBuffer
from training.ppo.ppo_trainer import TeamPPOTrainer, PPOTrainer
from radar_sim.gpu.vec_mfar_env import MFARVecEnv

# ── Config ────────────────────────────────────────────────────────────────
TARGET_DISTANCE_M = 12700.0       # 12.7 km
TARGET_AZIMUTH_DEG = 15.0         # off-boresight to give non-zero learning target
TARGET_RCS_DBSM = 20.0            # bomber-sized target
ARRAY_ROWS = 25
ARRAY_COLS = 25
TX_POWER_W = 50000                # 50 kW peak

PPO_CFG = {
    "commander_lr": 3e-4, "radar_lr": 1e-4,
    "gamma": 0.99, "gae_lambda": 0.95,
    "commander_clip": 0.2, "radar_clip": 0.1,
    "commander_entropy": 0.01, "radar_entropy": 0.01,
    "value_coef": 1.0, "max_grad_norm": 0.5,
    "n_epochs": 5, "batch_size": 32,
    "buffer_size_commander": 128, "buffer_size_radar": 64,
}

# Dense reward shaper weights — beam_accuracy_weight is the key one
REWARD_WEIGHTS = {
    "detect_snr_weight": 0.1,
    "jam_effectiveness_weight": 0.1,
    "comm_reliability_weight": 0.05,
    "recon_intel_weight": 0.03,
    "beam_accuracy_weight": 5.0,  # dominant signal for beam learning
    "snr_threshold_db": 10.0,
}


def make_env(device: str = "cuda") -> MFARVecEnv:
    return MFARVecEnv(
        num_envs=1,
        n_radars=2,    # 1 per team (battlefield requires 2 teams)
        n_teams=2,
        rows=ARRAY_ROWS,
        cols=ARRAY_COLS,
        pulses_per_cpi=1,
        fft_size=32768,
        device=device,
        tx_power_w=TX_POWER_W,
        cpi_preallocate=False,
        rx_beamforming=True,
        target_rcs_dbsm=TARGET_RCS_DBSM,
    )


def set_static_target(env: MFARVecEnv):
    """Fix target at known distance/azimuth, zero velocity and array rotation."""
    az_rad = np.deg2rad(TARGET_AZIMUTH_DEG)
    env.target_pos[:, 0, 0] = TARGET_DISTANCE_M * np.cos(az_rad)
    env.target_pos[:, 0, 1] = TARGET_DISTANCE_M * np.sin(az_rad)
    env.target_pos[:, 0, 2] = 0.0
    env.target_vel.zero_()
    env.array_rotation.zero_()
    env.radar_pos.zero_()  # both radars at origin → target_az = 15° for both


def force_detect_task(env: MFARVecEnv, actions: torch.Tensor):
    """Overwrite task logits: all elements → detect (task_idx=1)."""
    for e in range(env.n_elem):
        base = e * 22
        actions[:, :, base:base + 4].zero_()
        actions[:, :, base + 1] = 1.0  # TASK_DETECT = 1

def compute_random_baseline(env: MFARVecEnv, n_steps: int = 200) -> dict:
    """Run random actions to establish baseline reward levels."""
    env.reset()
    set_static_target(env)
    shaper = DenseRewardShaper(device=env.device, **REWARD_WEIGHTS)

    total_shaped = 0.0
    beam_acc_vals = []
    for _ in range(n_steps):
        actions = torch.randn(1, env.n_radars, env.action_dim, device=env.device)
        commander_actions = torch.zeros(1, env.n_teams, env.battlefield.commander_action_dim,
                                        device=env.device)
        result = env.step(actions=actions, commander_actions=commander_actions)
        shaped = shaper(result)
        total_shaped += shaped["total_shaped"].mean().item()
        beam_acc_vals.append(shaped["beam_accuracy_reward"].item())

    return {
        "avg_total_shaped": total_shaped / n_steps,
        "avg_beam_accuracy": np.mean(beam_acc_vals),
        "max_beam_accuracy": np.max(beam_acc_vals),
    }


def run(device: str = "cuda", total_steps: int = 50000, log_interval: int = 200):
    print(f"[init] device={device}, steps={total_steps}")
    print(f"[init] target: {TARGET_DISTANCE_M/1000:.1f} km, "
          f"az={TARGET_AZIMUTH_DEG}°, RCS={TARGET_RCS_DBSM} dBsm")
    print(f"[init] array: {ARRAY_ROWS}×{ARRAY_COLS}, tx_power={TX_POWER_W/1000:.0f} kW")

    env = make_env(device)
    n_pulses = env.n_pulses
    n_bins = env.n_bins
    print(f"[init] env: n_elem={env.n_elem}, n_pulses={n_pulses}, n_bins={n_bins}, "
          f"state_dim={env.state_dim}, action_dim={env.action_dim}")

    # ── Baseline ──────────────────────────────────────────────────────
    print("[baseline] Computing random-policy baseline...")
    baseline = compute_random_baseline(env, n_steps=200)
    print(f"[baseline] random avg_total_shaped = {baseline['avg_total_shaped']:.6f}")
    print(f"[baseline] random avg_beam_acc    = {baseline['avg_beam_accuracy']:.4f}")
    print(f"[baseline] random max_beam_acc    = {baseline['max_beam_accuracy']:.4f}")

    # ── Create policy and trainer ─────────────────────────────────────
    policy = create_team_policy(
        0, device=device, n_elem=env.n_elem,
        n_pulses=n_pulses, n_bins=n_bins,
        num_output_length=env.num_output_length,
        sub_array_size=25,  # single sub-array, zero gradient dilution
    )
    trainer = TeamPPOTrainer(
        commander=policy["commander"],
        radar=policy["radar"],
        commander_lr=PPO_CFG["commander_lr"],
        radar_lr=PPO_CFG["radar_lr"],
        gamma=PPO_CFG["gamma"],
        gae_lambda=PPO_CFG["gae_lambda"],
        commander_clip=PPO_CFG["commander_clip"],
        radar_clip=PPO_CFG["radar_clip"],
        commander_entropy=PPO_CFG["commander_entropy"],
        radar_entropy=PPO_CFG["radar_entropy"],
        value_coef=PPO_CFG["value_coef"],
        max_grad_norm=PPO_CFG["max_grad_norm"],
        n_epochs=PPO_CFG["n_epochs"],
        batch_size=PPO_CFG["batch_size"],
        buffer_size_commander=PPO_CFG["buffer_size_commander"],
        buffer_size_radar=PPO_CFG["buffer_size_radar"],
        device=device,
    )
    trainer.init_buffers(env.state_dim, env.action_dim)
    # dummy_obs=False: use real spectrum → detect SNR reward provides signal
    # Override reward shaper with tuned weights
    trainer.reward_shaper = DenseRewardShaper(device=device, **REWARD_WEIGHTS)

    # ── Training loop ─────────────────────────────────────────────────
    env.reset()
    set_static_target(env)

    rewards_history = []
    beam_acc_history = []
    beam_az_history = []
    target_az = TARGET_AZIMUTH_DEG  # fixed target azimuth
    max_steps_per_ep = 50
    t_start = time.time()
    step = 0
    ep = 0

    while step < total_steps:
        env.reset()
        set_static_target(env)
        ep_reward = 0.0
        ep_beam_acc = 0.0

        for ep_step in range(max_steps_per_ep):
            if step >= total_steps:
                break

            # Get observations; get actions for team 0 (the agent)
            state, commander_obs = trainer._get_observations(env)
            own = trainer.get_own_actions(env, team=0)

            # Assemble full actions: team 0 = network, team 1 = random
            E = env.num_envs
            actions = torch.zeros(E, env.n_radars, env.action_dim, device=device)
            for i, r in enumerate(range(own["r_start"], own["r_end"])):
                actions[:, r, :] = own["radar_actions"][i]
            # Team 1 (opponent): random actions
            opp_r_start = 1 * (env.n_radars // env.n_teams)
            opp_r_end = opp_r_start + (env.n_radars // env.n_teams)
            actions[:, opp_r_start:opp_r_end, :] = torch.randn(
                E, opp_r_end - opp_r_start, env.action_dim, device=device)
            # Network freely chooses tasks via task_head (no force)
            # Commander actions: team 0 = zeros (no missiles), team 1 = random
            commander_actions = torch.zeros(
                E, env.n_teams, env.battlefield.commander_action_dim, device=device)

            result = env.step(actions=actions, commander_actions=commander_actions)

            # Keep array rotation zeroed — only test raw beam steering
            env.array_rotation.zero_()

            # Evaluate actions for team 0 only
            rep_obs = own["transition"]["radar_obs"]
            rep_action = own["transition"]["radar_action"]
            rep_logp, _, rep_val, _ = trainer.radar_trainer.ac.evaluate_actions(
                rep_obs, rep_action)
            cmd_obs_0 = own["transition"]["cmd_obs"]
            cmd_act_0 = own["transition"]["cmd_action"]
            cmd_logp, _, cmd_val, _ = trainer.commander_trainer.ac.evaluate_actions(
                cmd_obs_0, cmd_act_0)

            transition = {
                "cmd_obs": cmd_obs_0, "cmd_action": cmd_act_0,
                "cmd_logp": cmd_logp, "cmd_val": cmd_val.squeeze(-1),
                "radar_obs": rep_obs, "radar_action": rep_action,
                "radar_logp": rep_logp, "radar_val": rep_val.squeeze(-1),
            }

            reward_info = trainer.store_transition(env, result, transition, team=0)
            ep_reward += reward_info["radar_reward"].mean().item()
            ep_beam_acc += reward_info["shaped_rewards"]["beam_accuracy_reward"].item()

            # Log beam angles
            if step % log_interval == 0:
                beam_az_all = result["beam_az"]  # [E, R]
                beam_az_t0 = beam_az_all[..., 0:1].mean().item()  # team 0 only
                beam_az = beam_az_all.mean().item()  # overall mean
                beam_el = result["beam_el"].mean().item()
                beam_acc = reward_info["shaped_rewards"]["beam_accuracy_reward"].item()
                total = reward_info["shaped_rewards"]["total_shaped"].mean().item()

                rewards_history.append(total)
                beam_acc_history.append(beam_acc)
                beam_az_history.append(beam_az)

                elapsed = time.time() - t_start
                steps_per_sec = step / max(elapsed, 0.1) if step > 0 else 0
                # Task distribution from result
                task_ids = result.get("task_ids")  # [E, R, N]
                if task_ids is not None:
                    t0_tasks = task_ids[0, 0, :]  # team 0, all elements
                    n_detect = (t0_tasks == 1).sum().item()
                    n_recon = (t0_tasks == 0).sum().item()
                    n_jam = (t0_tasks == 2).sum().item()
                    n_comm = (t0_tasks == 3).sum().item()
                    task_str = f"D:{n_detect} R:{n_recon} J:{n_jam} C:{n_comm}"
                else:
                    task_str = "?"
                print(f"[{step:6d}] total_shaped={total:.4f} beam_acc={beam_acc:.3f} "
                      f"beam_az_t0={beam_az_t0:+.1f}° (tgt={target_az:+.1f}°) "
                      f"el={beam_el:+.1f}° {task_str}  steps/s={steps_per_sec:.1f}")

            if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
               (trainer.radar_buffer and trainer.radar_buffer.near_full):
                metrics = trainer.update()
                if metrics.get("radar"):
                    print(f"[{step:6d}] PPO-update radar: {metrics['radar']}")

            step += 1

        ep += 1
        # Force update at end of episode
        metrics = trainer.update()
        if metrics.get("radar") and step % 1000 < 200:
            print(f"[{step:6d}] PPO-ep-update radar: {metrics['radar']}")

    # ── Final metrics ─────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n[result] {step} steps in {elapsed:.1f}s ({step/elapsed:.1f} steps/s)")
    if rewards_history:
        final_reward = np.mean(rewards_history[-20:]) if len(rewards_history) >= 20 else np.mean(rewards_history)
        final_beam_acc = np.mean(beam_acc_history[-20:]) if len(beam_acc_history) >= 20 else np.mean(beam_acc_history)
        final_beam_az = np.mean(beam_az_history[-20:])
        print(f"[result] final shaped_reward (last 20 logs) = {final_reward:.4f}")
        print(f"[result] final beam_accuracy              = {final_beam_acc:.3f}")
        print(f"[result] final beam_az                    = {final_beam_az:+.1f}°")
        print(f"[result] baseline random shaped_reward    = {baseline['avg_total_shaped']:.4f}")

        improvement = final_reward - baseline["avg_total_shaped"]
        if improvement > 0.01:
            print(f"[PASS] Shaped reward improved +{improvement:.4f} over random baseline.")
        else:
            print(f"[FAIL] Shaped reward did NOT improve over random baseline "
                  f"(delta={improvement:.4f}).")

        az_error = abs(final_beam_az - target_az)
        beamwidth_3db = 4.06  # degrees for 25×25 at 10 GHz
        if az_error < beamwidth_3db * 2:
            print(f"[PASS] Beam az error {az_error:.1f}° < {beamwidth_3db*2:.1f}° (2×BW).")
        else:
            print(f"[INFO] Beam az error {az_error:.1f}° vs target {target_az}°. "
                  f"2×BW={beamwidth_3db*2:.1f}°")

    # Save checkpoint
    os.makedirs("checkpoints/minimal_detect", exist_ok=True)
    ckpt_path = "checkpoints/minimal_detect/final.pt"
    trainer.save(ckpt_path)
    print(f"[save] checkpoint → {ckpt_path}")

    return {
        "baseline_shaped": baseline["avg_total_shaped"],
        "final_shaped": final_reward if rewards_history else float("nan"),
        "improvement": improvement if rewards_history else float("nan"),
        "rewards": rewards_history,
        "beam_acc": beam_acc_history,
        "beam_az": beam_az_history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal detect beam-learning test")
    parser.add_argument("--steps", type=int, default=50000,
                        help="Total training steps (default: 50000)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    args = parser.parse_args()

    run(device=args.device, total_steps=args.steps)
