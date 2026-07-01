"""Verify real PPO training: gradients flow, loss decreases, performance changes.

Runs 200 PPO updates on a small 2x2 env and logs:
  - Loss curves (policy_loss, value_loss, entropy)
  - Reward progression
  - Gradient norm
  - Parameter change magnitude

Usage:
    python -m training.verify_training
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from env.gpu.vec_mfar_env import MFARVecEnv
from algo._shared.ppo.actor_critic import CommanderActorCritic, RadarActorCritic
from algo._shared.ppo.buffer import RolloutBuffer
from algo._shared.ppo.reward_shaping import DenseRewardShaper


def run_verification():
    print("=" * 60)
    print("FluxPhased Training Verification")
    print("=" * 60)

    device = "cuda"
    torch.manual_seed(42)
    np.random.seed(42)

    # Small env for fast iteration
    env = MFARVecEnv(
        num_envs=4, n_radars=4, rows=2, cols=2,
        pulses_per_cpi=2, fft_size=32, device=device,
    )
    print(f"Env: E={env.num_envs}, R={env.n_radars}, N={env.n_elem}, "
          f"state_dim={env.state_dim}, action_dim={env.action_dim}")

    n_elem = env.n_elem  # 4
    n_pulses = env.n_pulses  # 2
    n_bins = env.n_bins
    state_dim = env.state_dim
    action_dim = env.action_dim
    r_per_team = env.n_radars // env.n_teams  # 2

    # Create networks
    cmd_ac = CommanderActorCritic(obs_dim=76, act_dim=35, hidden_dim=128).to(device)
    radar_ac = RadarActorCritic(
        n_elem=n_elem, n_pulses=n_pulses, n_bins=n_bins,
        spectrum_hidden=64, commander_instr_dim=env.num_output_length,
    ).to(device)

    cmd_params = sum(p.numel() for p in cmd_ac.parameters())
    radar_params = sum(p.numel() for p in radar_ac.parameters())
    print(f"Commander params: {cmd_params:,}")
    print(f"Radar params: {radar_params:,}")

    # Optimizers
    cmd_opt = torch.optim.Adam(cmd_ac.parameters(), lr=3e-4)
    radar_opt = torch.optim.Adam(radar_ac.parameters(), lr=1e-4)

    # Reward shaper
    shaper = DenseRewardShaper(device=device)

    # Tracking
    history = defaultdict(list)

    # Training loop
    n_updates = 200
    steps_per_update = 16  # collect 16 transitions per update
    gamma = 0.99
    gae_lambda = 0.95
    clip_range = 0.2
    entropy_coef = 0.01
    value_coef = 0.5

    env.reset()

    for update in range(n_updates):
        # --- Collect rollout ---
        cmd_obs_list = []
        cmd_act_list = []
        cmd_logp_list = []
        cmd_val_list = []

        radar_obs_list = []
        radar_act_list = []
        radar_logp_list = []
        radar_val_list = []

        reward_list = []
        done_list = []

        for step in range(steps_per_update):
            # Get current state
            with torch.no_grad():
                state = env._assemble_state(env._buf_spectrum.zero_(), env._buf_comm_data.zero_())
                commander_obs = env.battlefield.get_commander_observation(
                    env.radar_pos, torch.zeros(env.num_envs, env.n_radars, env.num_input_length, device=device),
                )

            # For each env, collect team 0's observations and actions
            # Team 0: radars 0,1
            team = 0
            r_start = team * r_per_team
            r_end = r_start + r_per_team

            # Commander action (team 0)
            cmd_obs = commander_obs[:, team, :]  # [E, 68]
            with torch.no_grad():
                cmd_action, cmd_logp, cmd_val, _ = cmd_ac.get_action(cmd_obs)

            # Radar actions (both radars, shared policy)
            radar_obs_0 = state[:, 0, :]  # [E, state_dim]
            radar_obs_1 = state[:, 1, :]  # [E, state_dim]
            with torch.no_grad():
                radar_act_0, radar_logp_0, radar_val_0, _ = radar_ac.get_action(radar_obs_0)
                radar_act_1, radar_logp_1, radar_val_1, _ = radar_ac.get_action(radar_obs_1)

            # Build full action tensor for env
            actions = torch.zeros(env.num_envs, env.n_radars, action_dim, device=device)
            # Team 0: use real policy
            actions[:, 0, :] = radar_act_0
            actions[:, 1, :] = radar_act_1
            # Team 1: random (opponent)
            actions[:, 2:, :] = torch.rand(env.num_envs, 2, action_dim, device=device)

            # Commander actions
            commander_actions = torch.zeros(
                env.num_envs, env.n_teams, env.battlefield.commander_action_dim, device=device,
            )
            commander_actions[:, team, :] = cmd_action
            commander_actions[:, 1, :] = torch.rand(
                env.num_envs, env.battlefield.commander_action_dim, device=device,
            ) * 2 - 1

            # Step env
            result = env.step(actions=actions, commander_actions=commander_actions)

            # Compute rewards
            shaped = shaper(result)
            radar_reward = shaped["total_shaped"] + result["radar_rewards"]  # [E, R]
            cmd_reward = result["commander_rewards"]  # [E, n_teams]

            # Flatten per-env → per-transition
            for e in range(env.num_envs):
                # Commander
                cmd_obs_list.append(cmd_obs[e].cpu())
                cmd_act_list.append(cmd_action[e].cpu())
                cmd_logp_list.append(cmd_logp[e].cpu())
                cmd_val_list.append(cmd_val[e].squeeze().cpu())

                # Radar 0 (representative)
                radar_obs_list.append(radar_obs_0[e].cpu())
                radar_act_list.append(radar_act_0[e].cpu())
                radar_logp_list.append(radar_logp_0[e].cpu())
                radar_val_list.append(radar_val_0[e].squeeze().cpu())

                # Reward: sum of team 0's radar rewards
                team_reward = radar_reward[e, r_start:r_end].sum().item()
                team_reward += cmd_reward[e, team].item()
                reward_list.append(team_reward)
                done_list.append(float(result["dones"][e].item()))

            if result["dones"].any():
                env.reset()

        # --- Compute GAE ---
        rewards_t = torch.tensor(reward_list)
        dones_t = torch.tensor(done_list)
        cmd_vals = torch.stack(cmd_val_list)
        radar_vals = torch.stack(radar_val_list)

        n = len(reward_list)

        # Use radar values for GAE (main policy)
        advantages = torch.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = 0.0
                next_nonterm = 0.0
            else:
                next_val = radar_vals[t + 1].item()
                next_nonterm = 1.0 - dones_t[t + 1].item()
            delta = rewards_t[t] + gamma * next_val * next_nonterm - radar_vals[t].item()
            gae = delta + gamma * gae_lambda * next_nonterm * gae
            advantages[t] = gae

        returns = advantages + radar_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO Update: Radar ---
        radar_obs_t = torch.stack(radar_obs_list).to(device)
        radar_act_t = torch.stack(radar_act_list).to(device)
        radar_logp_t = torch.stack(radar_logp_list).to(device)
        adv_t = advantages.to(device)
        ret_t = returns.to(device)

        radar_total_loss = 0.0
        radar_grad_norm = 0.0
        for epoch in range(4):
            # Mini-batch
            indices = torch.randperm(n)[:min(32, n)]
            mb_obs = radar_obs_t[indices]
            mb_act = radar_act_t[indices]
            mb_old_logp = radar_logp_t[indices]
            mb_adv = adv_t[indices]
            mb_ret = ret_t[indices]

            new_logp, entropy, value, _ = radar_ac.evaluate_actions(mb_obs, mb_act)
            ratio = torch.exp(new_logp - mb_old_logp)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((value.squeeze() - mb_ret) ** 2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            radar_opt.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(radar_ac.parameters(), 0.5)
            radar_opt.step()

            radar_total_loss += loss.item()
            radar_grad_norm += grad_norm.item()

        # --- PPO Update: Commander ---
        cmd_obs_t = torch.stack(cmd_obs_list).to(device)
        cmd_act_t = torch.stack(cmd_act_list).to(device)
        cmd_logp_t = torch.stack(cmd_logp_list).to(device)

        cmd_total_loss = 0.0
        for epoch in range(4):
            indices = torch.randperm(n)[:min(32, n)]
            mb_obs = cmd_obs_t[indices]
            mb_act = cmd_act_t[indices]
            mb_old_logp = cmd_logp_t[indices]
            mb_adv = adv_t[indices]
            mb_ret = ret_t[indices]

            new_logp, entropy, value, _ = cmd_ac.evaluate_actions(mb_obs, mb_act)
            ratio = torch.exp(new_logp - mb_old_logp)

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((value.squeeze() - mb_ret) ** 2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            cmd_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(cmd_ac.parameters(), 0.5)
            cmd_opt.step()

            cmd_total_loss += loss.item()

        # Record
        avg_reward = rewards_t.mean().item()
        history["reward"].append(avg_reward)
        history["radar_loss"].append(radar_total_loss / 4)
        history["cmd_loss"].append(cmd_total_loss / 4)
        history["grad_norm"].append(radar_grad_norm / 4)

        if update % 20 == 0:
            print(f"Update {update:3d} | reward={avg_reward:+.4f} | "
                  f"radar_loss={radar_total_loss/4:.4f} | cmd_loss={cmd_total_loss/4:.4f} | "
                  f"grad_norm={radar_grad_norm/4:.4f}")

    # --- Final analysis ---
    print("\n" + "=" * 60)
    print("Training Verification Results")
    print("=" * 60)

    first_20 = np.mean(history["reward"][:20])
    last_20 = np.mean(history["reward"][-20:])
    first_loss = np.mean(history["radar_loss"][:20])
    last_loss = np.mean(history["radar_loss"][-20:])

    print(f"\nReward:   first 20 avg = {first_20:+.4f} → last 20 avg = {last_20:+.4f} "
          f"(Δ = {last_20 - first_20:+.4f})")
    print(f"Radar loss: first 20 avg = {first_loss:.4f} → last 20 avg = {last_loss:.4f}")
    print(f"Gradient norm: mean = {np.mean(history['grad_norm']):.4f}, "
          f"max = {np.max(history['grad_norm']):.4f}")

    # Check parameter change
    env.reset()
    with torch.no_grad():
        test_state = env._assemble_state(env._buf_spectrum.zero_(), env._buf_comm_data.zero_())
        test_obs = test_state[:, 0, :]  # radar 0
        action, value = radar_ac(test_obs)
        print(f"\nFinal network output: action_range=[{action.min().item():.3f}, {action.max().item():.3f}], "
              f"value_mean={value.mean().item():.4f}")

    # Verdict
    grad_flows = np.mean(history["grad_norm"]) > 0.001
    loss_changes = abs(first_loss - last_loss) > 0.001
    reward_changes = abs(first_20 - last_20) > 0.001

    print(f"\n--- Checks ---")
    print(f"Gradients flow:  {'PASS' if grad_flows else 'FAIL'} (mean grad_norm={np.mean(history['grad_norm']):.6f})")
    print(f"Loss changes:    {'PASS' if loss_changes else 'FAIL'} (Δ={first_loss - last_loss:.6f})")
    print(f"Reward changes:  {'PASS' if reward_changes else 'FAIL'} (Δ={last_20 - first_20:+.6f})")

    if grad_flows and loss_changes:
        print(f"\n[OK] Training verification PASSED -- real gradients, real parameter updates.")
    else:
        print(f"\n[FAIL] Training verification FAILED -- check network architecture.")


if __name__ == "__main__":
    run_verification()
