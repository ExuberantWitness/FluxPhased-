"""WP-C D5 — Analyze RL channel_select behavior under AdaptiveSpectrum enemy.

D3 showed RL crowns 0/3 with chan_Δ/s = 0.00 (RL never changes channels).
D5 asks WHY:

  1. Is RL channel head truly deterministic (1 channel), or does it have
     probability mass on multiple channels that argmax hides?
  2. Does RL channel_select correlate with enemy channel_select?
  3. What's the channel head entropy? (high = exploring; low = locked)
  4. Does RL EVER change channel during a full episode, even once?

Answers determine whether:
  A) RL is STATIC (locked at BC prior ch0/ch1) — no dynamic learning
  B) RL is REACTIVE (changes channel after enemy follows) — suboptimal
     because enemy follows perfectly
  C) RL is PROACTIVE (changes channel before enemy catches up) — true
     dynamic coordination skill, just under-trained
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import os
import torch
import numpy as np
from collections import Counter

from env.gpu.twoteam import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.baselines.adaptive_spectrum_jammer import AdaptiveSpectrumJammer
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


CKPT_DIR = "/home/ubuntu/CODE/FluxPhased-/experiments/twoteam/_wp_c_ckpts"


def configure_channels(env, mode="orthogonal"):
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    dev = env.device
    freqs = torch.zeros(E, T, R, device=dev)
    ch_spacing = env.channel_spacing_hz
    fc = env.fc_hz
    for e in range(E):
        if mode == "orthogonal":
            freqs[e, 0, 0] = fc
            freqs[e, 0, 1] = fc + ch_spacing
            freqs[e, 1, 0] = fc
            freqs[e, 1, 1] = fc + ch_spacing
        else:
            freqs[e, 0, :] = fc
            freqs[e, 1, :] = fc
    env.set_radar_freqs(freqs)


def make_env(n_envs=8, episode_steps=200):
    return TwoTeamVecEnv(
        n_envs=n_envs, device="cuda",
        episode_steps=episode_steps,
        geometry=MIRROR_GEOMETRY,
        team_offset_m=2500.0,
    )


def make_ac(env):
    return TwoTeamCommanderActorCritic(
        obs_dim=env.obs_dim, privileged_dim=env.privileged_dim,
        n_fn=env.n_fn, n_aperture=env.n_radars_per_team,
        n_enemy=env.n_radars_per_team, freq_hop_max=env.freq_hop_max,
        n_channels=env.n_channels,
    ).to(env.device)


def analyze_rl_channels(env, ac, jam_fraction=1e-4, n_episodes=10, max_steps=200):
    """Run RL episodes and log channel behavior."""
    enemy = AdaptiveSpectrumJammer(jam_fraction=jam_fraction)

    all_rl_ch = []           # per-step RL channel_select [R] values
    all_enemy_ch = []        # per-step enemy channel_select [R] values
    all_head_entropy = []    # per-step RL channel head entropy (avg over R)
    all_head_probs = []      # per-step RL channel head softmax probs (avg over R)
    total_changes = 0        # # of steps where RL channel changed from prev
    total_steps_logged = 0
    episode_change_counts = []

    for ep in range(n_episodes):
        env.reset()
        configure_channels(env, "orthogonal")
        last_rl_ch = None
        ep_changes = 0

        for step in range(max_steps):
            obs_dict = env.get_obs()
            obs_t0 = obs_dict["obs"][:, 0]
            priv_t0 = obs_dict["privileged"][:, 0]

            with torch.no_grad():
                # Run forward to get action sample (stochastic to see true head behavior)
                action_dict, log_prob, value, value_local = ac(obs_t0, priv_t0)
                # Access channel head probs directly via trunk + head
                h = ac.actor_trunk(obs_t0)
                chan_logits = ac.channel_select_head(h).reshape(
                    obs_t0.shape[0], ac.n_aperture, ac.n_channels,
                )
                chan_probs = chan_logits.softmax(dim=-1)   # [B, R, n_channels]
                entropy_per_radar = -(chan_probs * (chan_probs + 1e-12).log()).sum(dim=-1)   # [B, R]
                # env 0 only
                entropy_env0 = entropy_per_radar[0].mean().item()
                probs_env0 = chan_probs[0].mean(dim=0).cpu().numpy()   # avg over R

            a_rl = {
                "task_alloc": action_dict["task_alloc"],
                "beam_target": action_dict["beam_target"],
                "laser_target": action_dict["laser_target"],
                "emission_on": action_dict["emission_on"],
                "freq_hop_rate": action_dict["freq_hop_rate"],
                "channel_select": action_dict["channel_select"],
            }
            a_enemy = enemy.get_action(env, team=1)
            action = combine_team_actions(env, a_rl, a_enemy)

            # Log channel selects
            rl_ch = a_rl["channel_select"][0].cpu().numpy()   # [R] for env 0
            enemy_ch = a_enemy["channel_select"][0].cpu().numpy()
            all_rl_ch.append(rl_ch.copy())
            all_enemy_ch.append(enemy_ch.copy())

            # Log channel head entropy + probs
            all_head_entropy.append(entropy_env0)
            all_head_probs.append(probs_env0)

            # Count changes
            if last_rl_ch is not None:
                changed = not np.array_equal(rl_ch, last_rl_ch)
                if changed:
                    total_changes += 1
                    ep_changes += 1
            last_rl_ch = rl_ch.copy()
            total_steps_logged += 1

            obs_dict, reward, done, info = env.step(action)
            if done.all():
                break

        episode_change_counts.append(ep_changes)

    # Summarize
    all_rl_ch = np.array(all_rl_ch)   # [n_steps, R]
    all_enemy_ch = np.array(all_enemy_ch)

    print(f"\n{'='*70}\nD5 — RL channel analysis (jam_frac={jam_fraction:.0e}, "
          f"{n_episodes} eps × env 0)\n{'='*70}")

    # Per-radar channel histogram
    R = all_rl_ch.shape[1]
    print(f"\n  Per-radar RL channel histogram (env 0):")
    for r in range(R):
        counts = Counter(all_rl_ch[:, r].tolist())
        hist_str = ", ".join(f"ch{k}:{counts.get(k, 0)}" for k in sorted(counts))
        print(f"    radar {r}: {hist_str}")

    print(f"\n  Per-radar enemy channel histogram (env 0):")
    for r in range(R):
        counts = Counter(all_enemy_ch[:, r].tolist())
        hist_str = ", ".join(f"ch{k}:{counts.get(k, 0)}" for k in sorted(counts))
        print(f"    radar {r}: {hist_str}")

    # Channel head entropy (if collected)
    if all_head_entropy:
        print(f"\n  RL channel head entropy:")
        print(f"    mean = {np.mean(all_head_entropy):.4f}")
        print(f"    min  = {np.min(all_head_entropy):.4f}")
        print(f"    max  = {np.max(all_head_entropy):.4f}")
        print(f"    (max possible = ln({env.n_channels}) = {np.log(env.n_channels):.4f})")
        # Average head probs
        avg_probs = np.mean(all_head_probs, axis=0)
        top3 = np.argsort(avg_probs)[::-1][:3]
        prob_str = ", ".join(f"ch{k}:{avg_probs[k]:.3f}" for k in top3)
        print(f"    top-3 channels by avg prob: {prob_str}")

    # Change statistics
    print(f"\n  Channel change statistics:")
    print(f"    total changes across all eps: {total_changes}/{total_steps_logged} steps")
    print(f"    per-episode change counts: {episode_change_counts}")
    never_changed = sum(1 for c in episode_change_counts if c == 0)
    print(f"    episodes with ZERO changes: {never_changed}/{n_episodes}")

    # Anti-correlation: when RL is on ch_X, is enemy on ch_X?
    # (low overlap = RL escapes; high overlap = RL jammed)
    overlap = (all_rl_ch == all_enemy_ch).mean()
    print(f"\n  RL-enemy channel overlap rate: {overlap:.3f}")
    print(f"    (1.0 = RL always on same ch as enemy = always jammed)")
    print(f"    (0.0 = RL always on different ch = always escaping)")

    # Proactive vs reactive analysis
    # For each step where RL changed channel, did enemy follow within 1-3 steps?
    return {
        "rl_ch_hist": all_rl_ch,
        "enemy_ch_hist": all_enemy_ch,
        "head_entropy": np.array(all_head_entropy),
        "overlap_rate": overlap,
        "total_changes": total_changes,
        "episode_change_counts": episode_change_counts,
    }


def main():
    env = make_env(n_envs=8, episode_steps=200)
    ac = make_ac(env)

    bc_ckpt = os.path.join(CKPT_DIR, "bc_pretrain.pt")
    if os.path.exists(bc_ckpt):
        state = torch.load(bc_ckpt, map_location=env.device, weights_only=False)
        ac.load_state_dict(state["ac_state"])
        print(f"Loaded BC ckpt from {bc_ckpt}")

    d3_ckpt = os.path.join(CKPT_DIR, "d3_ppo_adaptive.pt")
    if os.path.exists(d3_ckpt):
        state = torch.load(d3_ckpt, map_location=env.device, weights_only=False)
        ac.load_state_dict(state["ac_state"])
        print(f"Loaded D3 PPO ckpt from {d3_ckpt}")

    # Test 1: BC prior only (no PPO) — channel head should be locked at orth
    print("\n>>> Test 1: BC prior only (no PPO) <<<")
    if os.path.exists(bc_ckpt):
        state = torch.load(bc_ckpt, map_location=env.device, weights_only=False)
        ac.load_state_dict(state["ac_state"])
    bc_result = analyze_rl_channels(env, ac, jam_fraction=1e-4, n_episodes=10)

    # Test 2: D3 PPO trained
    print("\n>>> Test 2: D3 PPO trained <<<")
    if os.path.exists(d3_ckpt):
        state = torch.load(d3_ckpt, map_location=env.device, weights_only=False)
        ac.load_state_dict(state["ac_state"])
    ppo_result = analyze_rl_channels(env, ac, jam_fraction=1e-4, n_episodes=10)

    # Summary comparison
    print(f"\n{'='*70}\nD5 SUMMARY — Why RL didn't learn dynamic channel\n{'='*70}")
    print(f"\n  BC prior only:")
    print(f"    channel head entropy: {bc_result['head_entropy'].mean():.4f}")
    print(f"    overlap with enemy: {bc_result['overlap_rate']:.3f}")
    print(f"    total changes: {bc_result['total_changes']}")
    print(f"\n  After PPO:")
    print(f"    channel head entropy: {ppo_result['head_entropy'].mean():.4f}")
    print(f"    overlap with enemy: {ppo_result['overlap_rate']:.3f}")
    print(f"    total changes: {ppo_result['total_changes']}")

    # Verdict
    bc_locked = bc_result["total_changes"] == 0
    ppo_locked = ppo_result["total_changes"] == 0
    print(f"\n  Verdict:")
    if bc_locked and ppo_locked:
        print(f"    → RL channel head is STATIC (locked at BC prior).")
        print(f"    → 60-iter PPO did NOT break the BC lock.")
        print(f"    → This is option (A): no dynamic learning at all.")
        print(f"    → Production 5e7-step training MIGHT break the lock,")
        print(f"      but per 'no tuning games' rule, we report IET FLOOR.")
    elif bc_locked and not ppo_locked:
        print(f"    → PPO broke BC lock; RL now explores channels.")
        if ppo_result["overlap_rate"] < 0.3:
            print(f"    → RL is escaping enemy (low overlap).")
            print(f"    → This is option (C): proactive, just under-trained.")
        else:
            print(f"    → RL changes channels but enemy still follows.")
            print(f"    → This is option (B): reactive, suboptimal.")
    else:
        print(f"    → Both BC and PPO explore channels. Unexpected.")


if __name__ == "__main__":
    main()
