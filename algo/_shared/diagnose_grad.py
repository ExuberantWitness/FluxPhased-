"""Diagnose reward → advantage → gradient flow for laser training.

Runs one episode, checks:
1. Reward magnitudes (dense beam reward)
2. Advantage statistics (after GAE)
3. Gradient norms on commander actor-critic
4. Policy/value/BC loss magnitudes
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from algo._shared.ppo.buffer import RolloutBuffer
from algo._shared.train_laser import LaserTrainer, build_actors, build_env
import yaml


def main():
    # Phase 1.S sanity check: allow CLI override so we can point at the league config.
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/laser_25x25_train.yaml"
    print(f"[diagnose_grad] using config: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("=== Building env + actors ===")
    env = build_env(cfg)
    E, R, N = env.num_envs, env.n_radars, env.n_elem
    n_pulses, n_bins = env.n_pulses, env.n_bins
    print(f"E={E}, R={R}, N={N}")

    radar_ac, commander_ac = build_actors(cfg, N, n_pulses, n_bins, "cuda")
    trainer = LaserTrainer(env, radar_ac, commander_ac, cfg)

    print("\n=== Running one episode ===")
    stats = trainer.train_episode()
    print(f"Episode stats: {stats}")

    print("\n=== Buffer state ===")
    print(f"Radar buf: ptr={trainer.radar_buf.ptr}/{trainer.radar_buf.buffer_size}")
    print(f"Commander buf: ptr={trainer.cmd_buf.ptr}/{trainer.cmd_buf.buffer_size}")

    # Inspect commander buffer rewards
    cmd_buf = trainer.cmd_buf
    if cmd_buf.ptr > 0:
        n = cmd_buf.ptr
        rewards = cmd_buf.rewards[:n]
        values = cmd_buf.values[:n]
        print(f"\nCommander rewards (n={n}):")
        print(f"  mean={rewards.mean().item():.4f} std={rewards.std().item():.4f}")
        print(f"  min={rewards.min().item():.4f} max={rewards.max().item():.4f}")
        print(f"  nonzero frac={(rewards != 0).float().mean().item():.3f}")
        print(f"  |reward| > 0.01 frac={(rewards.abs() > 0.01).float().mean().item():.3f}")

        print(f"\nCommander values (critic predictions):")
        print(f"  mean={values.mean().item():.4f} std={values.std().item():.4f}")
        print(f"  min={values.min().item():.4f} max={values.max().item():.4f}")

        # Compute GAE returns
        cmd_buf.compute_returns(torch.zeros(1))
        advs = cmd_buf.advantages[:n]
        returns = cmd_buf.returns[:n]
        print(f"\nAdvantages (GAE):")
        print(f"  mean={advs.mean().item():.4f} std={advs.std().item():.4f}")
        print(f"  min={advs.min().item():.4f} max={advs.max().item():.4f}")
        print(f"  |adv| > 1e-6 frac={(advs.abs() > 1e-6).float().mean().item():.3f}")
        print(f"  |adv| > 0.1 frac={(advs.abs() > 0.1).float().mean().item():.3f}")

        print(f"\nReturns:")
        print(f"  mean={returns.mean().item():.4f} std={returns.std().item():.4f}")

    print("\n=== Manual gradient check on commander ===")
    # Sample a minibatch from commander buffer
    if cmd_buf.ptr >= 32:
        # Get one minibatch
        for batch in cmd_buf.get_minibatches(32):
            obs = batch["obs"]
            actions = batch["actions"]
            old_log_probs = batch["old_log_probs"]
            advantages = batch["advantages"]
            returns = batch["returns"]

            # Normalize advantages
            advs_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            print(f"\nMinibatch shapes: obs={obs.shape}, actions={actions.shape}")
            print(f"Normalized advantages: mean={advs_norm.mean():.4f} std={advs_norm.std():.4f}")
            print(f"Actions stats: mean={actions.mean():.4f} std={actions.std():.4f}")

            # Forward pass
            log_prob, entropy, value, _ = commander_ac.evaluate_actions(obs, actions)
            print(f"\nForward pass:")
            print(f"  log_prob: mean={log_prob.mean():.4f} std={log_prob.std():.4f}")
            print(f"  entropy: mean={entropy.mean():.4f}")
            print(f"  value: mean={value.mean():.4f} std={value.std():.4f}")

            # PPO loss
            ratio = torch.exp(log_prob - old_log_probs)
            surr1 = ratio * advs_norm
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advs_norm
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((value.squeeze(-1) - returns) ** 2).mean()
            entropy_loss = -entropy.mean()

            # BC loss
            enemy_xy = obs[:, 68:70]
            features = commander_ac.shared(obs)
            mean_raw = commander_ac.action_head(features)
            action_mean = torch.tanh(mean_raw)
            bc_loss = ((action_mean[:, 1:3] - enemy_xy) ** 2).mean()

            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss + 1.0 * bc_loss
            print(f"\nLosses:")
            print(f"  policy_loss = {policy_loss.item():.6f}")
            print(f"  value_loss  = {value_loss.item():.6f}")
            print(f"  entropy_loss = {entropy_loss.item():.6f}")
            print(f"  bc_loss     = {bc_loss.item():.6f}")
            print(f"  total       = {total_loss.item():.6f}")

            # Check ratios
            print(f"\nPPO ratios:")
            print(f"  ratio: mean={ratio.mean():.4f} std={ratio.std():.4f}")
            print(f"  ratio in clip range [0.8, 1.2]: {((ratio >= 0.8) & (ratio <= 1.2)).float().mean():.3f}")
            print(f"  ratio > 10 (explosion): {(ratio > 10).float().mean():.3f}")

            # Backward and check gradients
            total_loss.backward(retain_graph=True)
            print(f"\nGradient norms (PPO + BC combined):")
            total_grad = 0.0
            for name, p in commander_ac.named_parameters():
                if p.grad is not None:
                    g_norm = p.grad.norm().item()
                    total_grad += g_norm ** 2
                    if g_norm > 0:
                        print(f"  {name}: grad_norm={g_norm:.6f}")
            print(f"  TOTAL grad norm: {np.sqrt(total_grad):.6f}")

            # Sanity: check action_head grads specifically
            print(f"\nAction head gradients (drives aim):")
            for name, p in commander_ac.action_head.named_parameters():
                if p.grad is not None:
                    print(f"  action_head.{name}: shape={list(p.shape)} grad_norm={p.grad.norm().item():.6f}")

            # === PPO-ONLY gradient check (without BC) ===
            # This is the REAL success metric. If PPO is working, action_head
            # should receive gradient from policy_loss alone. If PPO is broken
            # (policy_loss≈0), this grad will be ~0 and BC is doing all the work.
            commander_ac.zero_grad()
            ppo_only_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            ppo_only_loss.backward()
            ppo_only_g = commander_ac.action_head.weight.grad.norm().item()
            print(f"\n=== PPO-ONLY gradient (no BC) ===")
            print(f"  action_head.weight grad_norm: {ppo_only_g:.6f}")
            print(f"  PPO working: {'YES' if ppo_only_g > 1e-4 else 'NO (BC is doing all the work)'}")
            # Also check shared trunk
            shared_grad = 0.0
            for name, p in commander_ac.shared.named_parameters():
                if p.grad is not None:
                    shared_grad += p.grad.norm().item() ** 2
            print(f"  shared trunk grad_norm: {np.sqrt(shared_grad):.6f}")
            print(f"  value_head.weight grad_norm: {commander_ac.value_head.weight.grad.norm().item():.6f}")

            break  # only first minibatch
    else:
        print(f"Commander buffer too small: {cmd_buf.ptr} < 32")


if __name__ == "__main__":
    main()
