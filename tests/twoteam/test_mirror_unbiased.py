"""WP-1 M4 §2.6⑥: MIRROR_GEOMETRY reward symmetry.

Spec §2.6⑥ mandates: under MIRROR_GEOMETRY, identical mirrored actions must
produce symmetric outcomes — reward asymmetry |mean| < 0.05·σ over many episodes.

The test is the "did we break mirror" canary for the entire WP-1 detection chain:
detection Bernoulli rolls, false alarm generation, nearest-neighbor association,
Kalman innovation — all stochastic steps that can break symmetry if RNG isn't
team-shared.

Pass criterion: reward A - reward B over N episodes has |mean| / std < 0.05.
Zero-sum structure gives reward_B = -reward_A, so this is essentially a check
that mean(reward_A) is small relative to its spread.

Strategy:
  1. Run N=50 episodes MIRROR_GEOMETRY, fixed seed, distinct channels (to
     avoid the ch0 intra-team interference cliff).
  2. Action: uniform task_alloc, beam_target=0 (both apertures → enemy radar 0),
     emission_on=true, freq_hop=1.0 — fully symmetric across teams.
  3. Collect per-episode team_A reward - team_B reward (= 2·team_A for zero-sum).
  4. Assert |mean(deltas)| < 0.05 · std(deltas).
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import math
import torch
import numpy as np
from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, MIRROR_GEOMETRY


def configure_distinct_channels(env):
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    freqs = torch.zeros(E, T, R, device=env.device)
    fc = env.fc_hz
    for e in range(E):
        freqs[e, 0, 0] = fc
        freqs[e, 0, 1] = fc + env.channel_spacing_hz
        freqs[e, 1, 0] = fc + 2 * env.channel_spacing_hz
        freqs[e, 1, 1] = fc + 3 * env.channel_spacing_hz
    env.set_radar_freqs(freqs)


def uniform_action(env):
    E = env.E
    return {
        "task_alloc": torch.full((E, 2, 2, 4), 0.25, device=env.device),
        "beam_target": torch.zeros(E, 2, 2, dtype=torch.long, device=env.device),
        "laser_target": torch.zeros(E, 2, dtype=torch.long, device=env.device),
        "emission_on": torch.ones(E, 2, 2, dtype=torch.bool, device=env.device),
        "freq_hop_rate": torch.ones(E, 2, 2, device=env.device),
    }


def test_mirror_unbiased_short_episode():
    """Short episode (50 steps) — verify reward asymmetry within tolerance."""
    n_episodes = 50
    episode_steps = 50
    env = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=episode_steps,
                        geometry=MIRROR_GEOMETRY, seed=42)
    deltas = []
    for ep in range(n_episodes):
        env.reset()
        configure_distinct_channels(env)
        action = uniform_action(env)
        ep_reward_A = torch.zeros(env.E, device=env.device)
        ep_reward_B = torch.zeros(env.E, device=env.device)
        done_mask = torch.zeros(env.E, dtype=torch.bool, device=env.device)
        while not done_mask.all():
            obs, reward, done, info = env.step(action)
            ep_reward_A += reward[:, 0]
            ep_reward_B += reward[:, 1]
            done_mask = done_mask | done
        # Per-env team_A - team_B delta (zero-sum → 2·team_A)
        delta = (ep_reward_A - ep_reward_B).cpu().numpy()
        deltas.append(delta)
    deltas = np.concatenate(deltas)
    mean = float(deltas.mean())
    std = float(deltas.std())
    if std < 1e-6:
        ratio = abs(mean)
    else:
        ratio = abs(mean) / std
    assert ratio < 0.05, (
        f"mirror biased: |mean|/std = {ratio:.4f} ≥ 0.05  "
        f"(mean={mean:.4f}, std={std:.4f}, N={len(deltas)})"
    )
    print(f"✅ MIRROR unbiased (short): mean={mean:+.4f}, std={std:.4f}, "
          f"|mean|/std={ratio:.4f} < 0.05, N={len(deltas)}")


def test_mirror_unbiased_kills_symmetric():
    """Kills should also be mirror-symmetric in distribution."""
    n_episodes = 50
    episode_steps = 100
    env = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=episode_steps,
                        geometry=MIRROR_GEOMETRY, seed=42)
    kills_A = []
    kills_B = []
    for ep in range(n_episodes):
        env.reset()
        configure_distinct_channels(env)
        action = uniform_action(env)
        done_mask = torch.zeros(env.E, dtype=torch.bool, device=env.device)
        while not done_mask.all():
            _, _, done, info = env.step(action)
            done_mask = done_mask | done
        kills_A.append(info["team_kills"][:, 0].cpu().numpy())
        kills_B.append(info["team_kills"][:, 1].cpu().numpy())
    kills_A = np.concatenate(kills_A)
    kills_B = np.concatenate(kills_B)
    delta = kills_A - kills_B
    mean = float(delta.mean())
    std = float(delta.std())
    if std < 1e-6:
        ratio = abs(mean)
    else:
        ratio = abs(mean) / std
    assert ratio < 0.05, (
        f"kills biased: |mean|/std = {ratio:.4f}  "
        f"(mean={mean:+.4f}, std={std:.4f})"
    )
    print(f"✅ kills symmetric: mean_delta={mean:+.4f}, std={std:.4f}, "
          f"|mean|/std={ratio:.4f}")
    print(f"   kills_A distribution: {np.bincount(kills_A.astype(int), minlength=3)}")
    print(f"   kills_B distribution: {np.bincount(kills_B.astype(int), minlength=3)}")


def test_mirror_unbiased_with_shutdown():
    """Mirror unbiased holds even with intermittent shutdown (more stochasticity)."""
    n_episodes = 30
    episode_steps = 80
    env = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=episode_steps,
                        geometry=MIRROR_GEOMETRY, seed=42)
    deltas = []
    for ep in range(n_episodes):
        env.reset()
        configure_distinct_channels(env)
        action = uniform_action(env)
        ep_reward_A = torch.zeros(env.E, device=env.device)
        ep_reward_B = torch.zeros(env.E, device=env.device)
        done_mask = torch.zeros(env.E, dtype=torch.bool, device=env.device)
        step = 0
        while not done_mask.all():
            # Both teams mirror-symmetric shutdown pattern: shut down radars
            # every other step (same pattern on both sides preserves symmetry).
            env.enemy_emitting[:] = (step % 2 == 0)
            obs, reward, done, info = env.step(action)
            ep_reward_A += reward[:, 0]
            ep_reward_B += reward[:, 1]
            done_mask = done_mask | done
            step += 1
        deltas.append((ep_reward_A - ep_reward_B).cpu().numpy())
    deltas = np.concatenate(deltas)
    mean = float(deltas.mean())
    std = float(deltas.std())
    ratio = abs(mean) / std if std > 1e-6 else abs(mean)
    assert ratio < 0.05, (
        f"mirror biased under shutdown: |mean|/std = {ratio:.4f}  "
        f"(mean={mean:.4f}, std={std:.4f})"
    )
    print(f"✅ MIRROR unbiased (with shutdown): mean={mean:+.4f}, std={std:.4f}, "
          f"|mean|/std={ratio:.4f}")


if __name__ == "__main__":
    print("=== WP-1 M4 §2.6⑥: MIRROR_GEOMETRY reward symmetry ===")
    print()
    print("--- test_mirror_unbiased_short_episode ---")
    test_mirror_unbiased_short_episode()
    print()
    print("--- test_mirror_unbiased_kills_symmetric ---")
    test_mirror_unbiased_kills_symmetric()
    print()
    print("--- test_mirror_unbiased_with_shutdown ---")
    test_mirror_unbiased_with_shutdown()
    print()
    print("🎉 all M4 mirror-unbiased tests PASS")
