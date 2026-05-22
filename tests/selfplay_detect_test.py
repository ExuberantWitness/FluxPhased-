#!/usr/bin/env python3
"""Self-play detect test — both teams train simultaneously with independent policies.

Phase 3 of the capability ladder:
  - Team 0 and Team 1 each have their own PPO trainer
  - Both teams learn beam steering toward the same static target
  - Each team's beam_acc is team-filtered (no cross-contamination)
  - Task heads freely choose tasks (bias-init toward detect)
  - Verifies: two-agent simultaneous training + beam convergence

Usage:
    python tests/selfplay_detect_test.py [--steps 50000] [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_league import (
    DenseRewardShaper,
    TeamPPOTrainer,
    create_team_policy,
)
from radar_sim.gpu.vec_mfar_env import MFARVecEnv

# ── Config ────────────────────────────────────────────────────────────────
TARGET_DISTANCE_M = 12700.0
TARGET_AZIMUTH_DEG = 15.0
TARGET_RCS_DBSM = 20.0
ARRAY_ROWS = 25
ARRAY_COLS = 25
TX_POWER_W = 50000

PPO_CFG = {
    "commander_lr": 3e-4, "radar_lr": 1e-4,
    "gamma": 0.99, "gae_lambda": 0.95,
    "commander_clip": 0.2, "radar_clip": 0.1,
    "commander_entropy": 0.01, "radar_entropy": 0.01,
    "value_coef": 1.0, "max_grad_norm": 0.5,
    "n_epochs": 5, "batch_size": 32,
    "buffer_size_commander": 128, "buffer_size_radar": 64,
}

REWARD_WEIGHTS = {
    "detect_snr_weight": 0.1,
    "jam_effectiveness_weight": 0.1,
    "comm_reliability_weight": 0.05,
    "recon_intel_weight": 0.03,
    "beam_accuracy_weight": 5.0,
    "beam_sigma": 15.0,
    "snr_threshold_db": 10.0,
}


def make_env(device: str = "cuda") -> MFARVecEnv:
    return MFARVecEnv(
        num_envs=1, n_radars=2, n_teams=2,
        rows=ARRAY_ROWS, cols=ARRAY_COLS,
        pulses_per_cpi=1, fft_size=32768,
        device=device, tx_power_w=TX_POWER_W,
        cpi_preallocate=False, rx_beamforming=True,
        target_rcs_dbsm=TARGET_RCS_DBSM,
    )


def set_static_target(env: MFARVecEnv):
    az_rad = np.deg2rad(TARGET_AZIMUTH_DEG)
    env.target_pos[:, 0, 0] = TARGET_DISTANCE_M * np.cos(az_rad)
    env.target_pos[:, 0, 1] = TARGET_DISTANCE_M * np.sin(az_rad)
    env.target_pos[:, 0, 2] = 0.0
    env.target_vel.zero_()
    env.array_rotation.zero_()
    env.radar_pos.zero_()


def create_trainer(team: int, env: MFARVecEnv, device: str) -> TeamPPOTrainer:
    policy = create_team_policy(
        team, device=device, n_elem=env.n_elem,
        n_pulses=env.n_pulses, n_bins=env.n_bins,
        num_output_length=env.num_output_length,
        sub_array_size=25,
    )
    trainer = TeamPPOTrainer(
        commander=policy["commander"], radar=policy["radar"],
        commander_lr=PPO_CFG["commander_lr"], radar_lr=PPO_CFG["radar_lr"],
        gamma=PPO_CFG["gamma"], gae_lambda=PPO_CFG["gae_lambda"],
        commander_clip=PPO_CFG["commander_clip"], radar_clip=PPO_CFG["radar_clip"],
        commander_entropy=PPO_CFG["commander_entropy"], radar_entropy=PPO_CFG["radar_entropy"],
        value_coef=PPO_CFG["value_coef"], max_grad_norm=PPO_CFG["max_grad_norm"],
        n_epochs=PPO_CFG["n_epochs"], batch_size=PPO_CFG["batch_size"],
        buffer_size_commander=PPO_CFG["buffer_size_commander"],
        buffer_size_radar=PPO_CFG["buffer_size_radar"],
        device=device,
    )
    trainer.init_buffers(env.state_dim, env.action_dim)
    trainer.reward_shaper = DenseRewardShaper(device=device, **REWARD_WEIGHTS)
    return trainer


def run(device: str = "cuda", total_steps: int = 50000, log_interval: int = 200):
    print(f"[init] device={device}, steps={total_steps}")
    print(f"[init] target: {TARGET_DISTANCE_M/1000:.1f} km, "
          f"az={TARGET_AZIMUTH_DEG}°, RCS={TARGET_RCS_DBSM} dBsm")

    env = make_env(device)
    print(f"[init] env: n_elem={env.n_elem}, state_dim={env.state_dim}, action_dim={env.action_dim}")

    # ── Create trainers for BOTH teams ──────────────────────────────────
    print("[init] Creating trainers for team 0 and team 1...")
    trainer0 = create_trainer(0, env, device)
    trainer1 = create_trainer(1, env, device)
    trainers = [trainer0, trainer1]

    env.reset()
    set_static_target(env)

    target_az = TARGET_AZIMUTH_DEG
    max_steps_per_ep = 50
    t_start = time.time()
    step = 0

    # Track metrics per team
    history = {0: {"beam_acc": [], "beam_az": [], "total": []},
               1: {"beam_acc": [], "beam_az": [], "total": []}}

    while step < total_steps:
        env.reset()
        set_static_target(env)

        for ep_step in range(max_steps_per_ep):
            if step >= total_steps:
                break

            E = env.num_envs
            actions = torch.zeros(E, env.n_radars, env.action_dim, device=device)
            transitions = {}

            # Get actions from BOTH teams
            for team, trainer in enumerate(trainers):
                own = trainer.get_own_actions(env, team=team)
                for i, r in enumerate(range(own["r_start"], own["r_end"])):
                    actions[:, r, :] = own["radar_actions"][i]
                # Evaluate actions for log_prob
                rep_obs = own["transition"]["radar_obs"]
                rep_action = own["transition"]["radar_action"]
                rep_logp, _, rep_val = trainer.radar_trainer.ac.evaluate_actions(rep_obs, rep_action)
                cmd_obs = own["transition"]["cmd_obs"]
                cmd_act = own["transition"]["cmd_action"]
                cmd_logp, _, cmd_val = trainer.commander_trainer.ac.evaluate_actions(cmd_obs, cmd_act)
                transitions[team] = {
                    "cmd_obs": cmd_obs, "cmd_action": cmd_act,
                    "cmd_logp": cmd_logp, "cmd_val": cmd_val.squeeze(-1),
                    "radar_obs": rep_obs, "radar_action": rep_action,
                    "radar_logp": rep_logp, "radar_val": rep_val.squeeze(-1),
                }

            commander_actions = torch.zeros(
                E, env.n_teams, env.battlefield.commander_action_dim, device=device)

            result = env.step(actions=actions, commander_actions=commander_actions)
            env.array_rotation.zero_()

            # Store transitions and update BOTH teams
            for team, trainer in enumerate(trainers):
                reward_info = trainer.store_transition(env, result, transitions[team], team=team)

                if step % log_interval == 0:
                    beam_acc = reward_info["shaped_rewards"]["beam_accuracy"].item()
                    total_shaped = reward_info["shaped_rewards"]["total_shaped"].mean().item()
                    beam_az_t0 = result["beam_az"][0, team].mean().item()
                    history[team]["beam_acc"].append(beam_acc)
                    history[team]["beam_az"].append(beam_az_t0)
                    history[team]["total"].append(total_shaped)

                # Trigger update when buffer is full
                if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
                   (trainer.radar_buffer and trainer.radar_buffer.near_full):
                    trainer.update()

            # Logging
            if step % log_interval == 0:
                t0_az = history[0]["beam_az"][-1] if history[0]["beam_az"] else 0
                t1_az = history[1]["beam_az"][-1] if history[1]["beam_az"] else 0
                t0_acc = history[0]["beam_acc"][-1] if history[0]["beam_acc"] else 0
                t1_acc = history[1]["beam_acc"][-1] if history[1]["beam_acc"] else 0
                task_ids = result.get("task_ids")
                if task_ids is not None:
                    t0t = task_ids[0, 0, :]; t1t = task_ids[0, 1, :]
                    d0 = (t0t == 1).sum().item(); d1 = (t1t == 1).sum().item()
                    task_str = f"T0:D{d0} T1:D{d1}"
                else:
                    task_str = "?"
                elapsed = time.time() - t_start
                sps = step / max(elapsed, 0.1) if step > 0 else 0
                print(f"[{step:6d}] T0:acc={t0_acc:.3f} az={t0_az:+.1f}° | "
                      f"T1:acc={t1_acc:.3f} az={t1_az:+.1f}° | "
                      f"{task_str}  steps/s={sps:.1f}")

            step += 1

        # Force update at end of episode for both teams
        for trainer in trainers:
            trainer.update()

    # ── Final metrics ─────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n[result] {step} steps in {elapsed:.1f}s ({step/elapsed:.1f} steps/s)")
    for team in [0, 1]:
        h = history[team]
        if h["beam_acc"]:
            n = min(20, len(h["beam_acc"]))
            avg_acc = np.mean(h["beam_acc"][-n:])
            avg_az = np.mean(h["beam_az"][-n:])
            print(f"[result] Team {team}: final beam_acc={avg_acc:.3f}, beam_az={avg_az:+.1f}° "
                  f"(tgt={target_az:+.1f}°)")
            az_error = abs(avg_az - target_az)
            if az_error < 10.0 and avg_acc > 0.5:
                print(f"[PASS] Team {team} beam converged (error={az_error:.1f}°, acc={avg_acc:.3f})")
            else:
                print(f"[WARN] Team {team} beam not fully converged (error={az_error:.1f}°, acc={avg_acc:.3f})")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play detect beam-learning test")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run(device=args.device, total_steps=args.steps)
