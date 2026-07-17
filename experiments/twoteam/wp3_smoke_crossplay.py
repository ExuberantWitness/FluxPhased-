"""WP-3 smoke cross-play: RL iter100 ckpt vs BlindClassical.

Per WP-3 plan "训练后 smoke cross-play vs BlindClassical":
  - 低干扰 (orthogonal channels): RL kill ≥ BlindClassical kill (打平即可)
  - 高干扰 (same-channel): RL kill > BlindClassical kill (若否则诚实记录)

Usage:
  python experiments/twoteam/wp3_smoke_crossplay.py \
      --rl-ckpt checkpoints/blind/wp3_100iter_dynamics/iter_final.pt \
      --episodes 30 --out experiments/twoteam/wp3_smoke_crossplay_report.md
"""

from __future__ import annotations
import sys
import os
import time
import argparse
import torch
import numpy as np

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


def make_env(n_envs, episode_steps=200, seed=42):
    return TwoTeamVecEnv(
        n_envs=n_envs, device="cuda",
        episode_steps=episode_steps,
        geometry=RANDOM_GEOMETRY, seed=seed,
    )


def make_ac(env):
    return TwoTeamCommanderActorCritic(
        obs_dim=env.obs_dim,
        privileged_dim=env.privileged_dim,
        n_fn=env.n_fn,
        n_aperture=env.n_radars_per_team,
        n_enemy=env.n_radars_per_team,
        freq_hop_max=env.freq_hop_max,
        n_channels=env.n_channels,
    ).to(env.device)


def configure_channels(env, mode: str):
    """orthogonal: ch0 + ch1 per team (low intra-team interference).
    same_channel: both radars on ch0 (high intra + cross-team interference)."""
    E, T, R = env.E, env.n_teams, env.n_radars_per_team
    dev = env.device
    freqs = torch.zeros(E, T, R, device=dev)
    fc = env.fc_hz
    cs = env.channel_spacing_hz
    for e in range(E):
        if mode == "orthogonal":
            freqs[e, 0, 0] = fc
            freqs[e, 0, 1] = fc + cs
            freqs[e, 1, 0] = fc
            freqs[e, 1, 1] = fc + cs
        else:  # same_channel
            freqs[e, 0, :] = fc
            freqs[e, 1, :] = fc
    env.set_radar_freqs(freqs)


def run_direction(env, rl_ac, rl_team: int, episode_steps=200, channel_mode="orthogonal"):
    """Run n_envs episodes; RL controls team=rl_team, BlindClassical controls other.
    Returns per-env metrics (numpy arrays)."""
    env.reset()
    configure_channels(env, channel_mode)
    opp_team = 1 - rl_team
    opp = BlindClassicalCommander()

    rl_kills = torch.zeros(env.E, device=env.device)
    opp_kills = torch.zeros(env.E, device=env.device)
    rl_exposure = torch.zeros(env.E, device=env.device)
    rl_trace_P = torch.zeros(env.E, device=env.device)
    rl_survival_steps = torch.zeros(env.E, device=env.device)
    last_info = None

    for step in range(episode_steps):
        obs_dict = env.get_obs()
        last_info = obs_dict if step == 0 else last_info

        # RL action
        rl_obs = obs_dict["obs"][:, rl_team]
        rl_detect = env.get_detect_list()[:, rl_team]
        rl_priv = obs_dict["privileged"][:, rl_team]
        a_rl, _ = rl_ac.get_action_for_env(rl_obs, rl_detect, rl_priv, deterministic=True)
        # Opponent (BlindClassical)
        a_opp = opp.get_action(env, opp_team)

        if rl_team == 0:
            action = combine_team_actions(env, a_rl, a_opp)
        else:
            action = combine_team_actions(env, a_opp, a_rl)

        _, reward, done, info = env.step(action)

        # Track metrics
        rl_kills = info["team_kills"][:, rl_team].float()
        opp_kills = info["team_kills"][:, opp_team].float()
        rl_exposure += info["exposure"][:, rl_team]
        trace_P_team = info["mean_trace_P"][:, rl_team]
        rl_alive = info["team_alive"][:, rl_team].float()
        rl_trace_P += trace_P_team * rl_alive  # only count while alive
        rl_survival_steps += rl_alive

        if done.all():
            break

    n_steps = max(step + 1, 1)
    return {
        "rl_kills": rl_kills.cpu().numpy(),
        "opp_kills": opp_kills.cpu().numpy(),
        "rl_exposure_avg": (rl_exposure / n_steps).cpu().numpy(),
        "rl_trace_P_avg": (rl_trace_P / rl_survival_steps.clamp(min=1.0)).cpu().numpy(),
        "rl_survival": (rl_survival_steps / n_steps).cpu().numpy(),
    }


def welch_t(a, b):
    """Welch's t-test statistic + p-value (two-sided)."""
    from scipy import stats
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rl-ckpt", required=True)
    p.add_argument("--episodes", type=int, default=30,
                   help="Episodes per direction per condition (total = 2× this per condition)")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Parallel envs per batch (episodes run in batches)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="experiments/twoteam/wp3_smoke_crossplay_report.md")
    args = p.parse_args()

    print(f"Loading RL ckpt: {args.rl_ckpt}")
    ckpt = torch.load(args.rl_ckpt, map_location="cuda", weights_only=False)
    print(f"  iter={ckpt.get('iter', '?')}")

    # Conditions per WP-3 spec
    conditions = [
        ("orthogonal", "low_interference"),
        ("same_channel", "high_interference"),
    ]

    results = {}
    t0 = time.time()
    for channel_mode, label in conditions:
        print(f"\n{'='*70}\n  Condition: {label} (channel_mode={channel_mode})\n{'='*70}")
        env = make_env(args.batch_size, episode_steps=args.horizon, seed=args.seed)
        ac = make_ac(env)
        ac.load_state_dict(ckpt["ac_state"])
        ac.eval()

        all_rl_kills, all_opp_kills = [], []
        all_rl_exp, all_rl_trace_P, all_rl_surv = [], [], []

        n_done = 0
        while n_done < args.episodes:
            # RL as team 0
            m = run_direction(env, ac, rl_team=0, episode_steps=args.horizon,
                              channel_mode=channel_mode)
            all_rl_kills.extend(m["rl_kills"].tolist())
            all_opp_kills.extend(m["opp_kills"].tolist())
            all_rl_exp.extend(m["rl_exposure_avg"].tolist())
            all_rl_trace_P.extend(m["rl_trace_P_avg"].tolist())
            all_rl_surv.extend(m["rl_survival"].tolist())
            n_done += args.batch_size
            print(f"  [RL=team0] batch done, total eps={min(n_done, args.episodes)}/{args.episodes}", flush=True)

        n_done = 0
        while n_done < args.episodes:
            # RL as team 1 (mirror)
            m = run_direction(env, ac, rl_team=1, episode_steps=args.horizon,
                              channel_mode=channel_mode)
            all_rl_kills.extend(m["rl_kills"].tolist())
            all_opp_kills.extend(m["opp_kills"].tolist())
            all_rl_exp.extend(m["rl_exposure_avg"].tolist())
            all_rl_trace_P.extend(m["rl_trace_P_avg"].tolist())
            all_rl_surv.extend(m["rl_survival"].tolist())
            n_done += args.batch_size
            print(f"  [RL=team1] batch done, total eps={min(n_done, args.episodes)}/{args.episodes}", flush=True)

        rl_k = np.array(all_rl_kills)
        opp_k = np.array(all_opp_kills)
        t_stat, p_val = welch_t(rl_k, opp_k)

        results[label] = {
            "rl_kills": rl_k,
            "opp_kills": opp_k,
            "rl_exposure": np.array(all_rl_exp),
            "rl_trace_P": np.array(all_rl_trace_P),
            "rl_survival": np.array(all_rl_surv),
            "t_stat": t_stat,
            "p_val": p_val,
            "n": len(rl_k),
        }

        print(f"\n  RL kills:    {rl_k.mean():.3f} ± {rl_k.std():.3f}  (n={len(rl_k)})")
        print(f"  BC kills:    {opp_k.mean():.3f} ± {opp_k.std():.3f}")
        print(f"  Δ = {rl_k.mean() - opp_k.mean():+.3f}  (Welch t={t_stat:+.2f}, p={p_val:.3f})")
        print(f"  RL survival: {results[label]['rl_survival'].mean():.3f}")
        print(f"  RL expos:    {results[label]['rl_exposure'].mean():.3f}")
        print(f"  RL trace_P:  {results[label]['rl_trace_P'].mean():.3f}")

        del env

    elapsed = time.time() - t0
    print(f"\n{'='*70}\nSmoke cross-play done in {elapsed/60:.1f} min\n{'='*70}")

    # Verdict
    print("\nVerdict per WP-3 spec:")
    for label, r in results.items():
        delta = r["rl_kills"].mean() - r["opp_kills"].mean()
        if "low" in label:
            ok = "PASS (打平或超过)" if delta >= -0.1 else "FAIL"
            print(f"  {label}: Δ_kills={delta:+.3f}  → {ok}")
        else:
            ok = "PASS (RL 超越 BC)" if delta > 0.05 else "MARGINAL/FAIL (诚实记录)"
            print(f"  {label}: Δ_kills={delta:+.3f}  → {ok}")

    # Write report
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# WP-3 Smoke Cross-Play Report\n\n")
        f.write(f"**RL ckpt**: `{args.rl_ckpt}` (iter {ckpt.get('iter', '?')})\n")
        f.write(f"**Episodes per direction**: {args.episodes}\n")
        f.write(f"**Horizon**: {args.horizon}\n")
        f.write(f"**Elapsed**: {elapsed/60:.1f} min\n\n")
        f.write("## Results\n\n")
        f.write("| Condition | RL kills | BC kills | Δ | Welch t | p-value | RL survival | RL trace_P |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for label, r in results.items():
            f.write(f"| {label} | {r['rl_kills'].mean():.3f}±{r['rl_kills'].std():.3f} | "
                    f"{r['opp_kills'].mean():.3f}±{r['opp_kills'].std():.3f} | "
                    f"{r['rl_kills'].mean()-r['opp_kills'].mean():+.3f} | "
                    f"{r['t_stat']:+.2f} | {r['p_val']:.3f} | "
                    f"{r['rl_survival'].mean():.3f} | {r['rl_trace_P'].mean():.3f} |\n")
        f.write("\n## Verdict\n\n")
        for label, r in results.items():
            delta = r["rl_kills"].mean() - r["opp_kills"].mean()
            if "low" in label:
                ok = "PASS" if delta >= -0.1 else "FAIL"
            else:
                ok = "PASS" if delta > 0.05 else "MARGINAL/FAIL"
            f.write(f"- **{label}**: Δ_kills={delta:+.3f} → {ok}\n")
    print(f"\nReport → {args.out}")


if __name__ == "__main__":
    main()
