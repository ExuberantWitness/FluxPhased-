"""WP-C D3 — Killer contrast: RL vs competent baseline under same dynamic enemy.

D1 PASS proved AdaptiveSpectrumJammer (channel-split follower) is meaningfully
harder than TrueFixedJammer. D3 asks the make-or-break question:

  Under the SAME dynamic enemy (AdaptiveSpectrum), does RL beat the
  competent fixed-rule baseline (StrongRule + orth + reactive hop)?

If RL trace_P < baseline trace_P with CI separation → CROWN (RL learns
dynamic coordination that fixed rule cannot).

If baseline trace_P ≈ RL trace_P, or baseline already low → IET FLOOR
(fixed rule is enough; honest finding).

Structure:
  1. Load BC pretrain ckpt (actor starts from StrongRule+orth demos)
  2. PPO fine-tune vs AdaptiveSpectrum enemy (single-opponent focus)
  3. Killer contrast: {RL, StrongRule+orth} × AdaptiveSpectrum × jam_frac
  4. Bootstrap 1000-resample CI 95% for trace_P gap
  5. Crown-vs-IET judgment

This is a sanity-scope script (50-100 PPO iters). Production 5e7-step
training is D4, only triggered if D3 shows crown signal.
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import argparse
import os
import time
import math
import torch
import numpy as np

from env.gpu.twoteam import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.baselines.adaptive_spectrum_jammer import AdaptiveSpectrumJammer
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.bc_pretrain import TwoTeamBCPretrainer
from algo._shared.pilot.twoteam.br_trainer import TwoTeamBRTrainer
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


CKPT_DIR = "/home/ubuntu/CODE/FluxPhased-/experiments/twoteam/_wp_c_ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)


def configure_channels(env, mode: str = "orthogonal"):
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


class OrthConfigStrongRuleWrapper:
    """StrongRule + external orth channel config (the competent baseline)."""
    def __init__(self):
        self.rule = TwoTeamStrongRuleCommander()

    def get_action(self, env, team: int):
        a = self.rule.get_action(env, team)
        E, R = env.E, env.n_radars_per_team
        ch = torch.zeros(E, R, dtype=torch.long, device=env.device)
        ch[:, 0] = 0
        ch[:, 1] = 1
        a["channel_select"] = ch
        return a


class AdaptiveSpectrumOpponent:
    """AdaptiveSpectrum jammer wrapped as frozen opponent."""
    def __init__(self, jam_fraction=1e-4):
        self.jam_fraction = jam_fraction
        self._jammer = None

    def get_action(self, env, team):
        # Re-init when jam_fraction changes (callers may mutate it between episodes)
        if self._jammer is None or self._jammer.jam_fraction != self.jam_fraction:
            self._jammer = AdaptiveSpectrumJammer(jam_fraction=self.jam_fraction)
        return self._jammer.get_action(env, team)


# ---------- PPO fine-tune ----------------------------------------------------

def train_rl_vs_adaptive(env, ac, n_iterations=80, horizon=300,
                          jam_fraction=1e-4, save_path=None):
    """PPO fine-tune RL vs AdaptiveSpectrum enemy."""
    print(f"\n{'='*70}\nPPO fine-tune: {n_iterations} iters × H={horizon} "
          f"vs AdaptiveSpectrum (jam_frac={jam_fraction:.0e})\n{'='*70}")
    frozen = AdaptiveSpectrumOpponent(jam_fraction=jam_fraction)
    trainer = TwoTeamBRTrainer(
        ac, frozen,
        lr_actor=3e-4, lr_critic=1e-3,
        entropy_coef=0.02,
        n_epochs=4, minibatch_size=64,
        reward_scale=0.1, device="cuda",
    )
    history = trainer.train(
        env, n_iterations=n_iterations, horizon=horizon,
        learning_team=0, save_path=save_path, log_every=10,
    )
    return history


# ---------- episode runners --------------------------------------------------

def run_rl_episode(env, rl_ac, opponent, max_steps=200):
    """RL (team 0) vs opponent (team 1). Returns per-env trace_P, kills, chan Δ/s."""
    env.reset()
    configure_channels(env, "orthogonal")
    E = env.E

    ep_trace_P = torch.zeros(E, device=env.device)
    ep_chan_changes = torch.zeros(E, device=env.device)
    last_chan = None
    ep_steps = 0
    last_info = None

    for step in range(max_steps):
        obs_dict = env.get_obs()
        obs_t0 = obs_dict["obs"][:, 0]
        priv_t0 = obs_dict["privileged"][:, 0]
        a_rl, _ = rl_ac.get_action_for_env(obs_t0, priv_t0, deterministic=True)
        a_opp = opponent.get_action(env, team=1)
        action = combine_team_actions(env, a_rl, a_opp)
        obs_dict, reward, done, info = env.step(action)
        last_info = info

        trace_P = env.tracker_P[:, 0, :, 0, 0] + env.tracker_P[:, 0, :, 2, 2]
        alive_mask = env.radar_alive[:, 1].float()
        n_alive = alive_mask.sum(dim=-1).clamp(min=1.0)
        ep_trace_P += (trace_P * alive_mask).sum(dim=-1) / n_alive

        rl_chan = a_rl["channel_select"]
        if last_chan is not None:
            changes = (rl_chan != last_chan).any(dim=-1).float()
            ep_chan_changes += changes
        last_chan = rl_chan.clone()

        ep_steps += 1
        if done.all():
            break

    kills_rl = last_info["team_kills"][:, 0].float()
    sc = max(ep_steps, 1)
    return {
        "trace_P": (ep_trace_P / sc).cpu().numpy(),
        "kills": kills_rl.cpu().numpy(),
        "chan_changes": (ep_chan_changes / sc).cpu().numpy(),
    }


def run_rule_episode(env, rule, opponent, max_steps=200):
    """Rule baseline (team 0) vs opponent (team 1)."""
    env.reset()
    configure_channels(env, "orthogonal")
    E = env.E

    ep_trace_P = torch.zeros(E, device=env.device)
    last_info = None
    ep_steps = 0

    for step in range(max_steps):
        a_rule = rule.get_action(env, 0)
        a_opp = opponent.get_action(env, 1)
        action = combine_team_actions(env, a_rule, a_opp)
        _, _, done, info = env.step(action)
        last_info = info

        trace_P = env.tracker_P[:, 0, :, 0, 0] + env.tracker_P[:, 0, :, 2, 2]
        alive_mask = env.radar_alive[:, 1].float()
        n_alive = alive_mask.sum(dim=-1).clamp(min=1.0)
        ep_trace_P += (trace_P * alive_mask).sum(dim=-1) / n_alive

        ep_steps += 1
        if done.all():
            break

    kills = last_info["team_kills"][:, 0].float()
    return {
        "trace_P": (ep_trace_P / max(ep_steps, 1)).cpu().numpy(),
        "kills": kills.cpu().numpy(),
    }


# ---------- bootstrap CI -----------------------------------------------------

def bootstrap_ci(data, n_resample=1000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    means = []
    n = len(data)
    for _ in range(n_resample):
        idx = rng.integers(0, n, n)
        means.append(data[idx].mean())
    means = np.array(means)
    alpha = (1 - ci) / 2
    lo = np.quantile(means, alpha)
    hi = np.quantile(means, 1 - alpha)
    return data.mean(), lo, hi


def bootstrap_diff_ci(a, b, n_resample=1000, ci=0.95, seed=0):
    """CI for (mean(a) - mean(b)) via paired bootstrap."""
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    diffs = []
    for _ in range(n_resample):
        idx = rng.integers(0, n, n)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    alpha = (1 - ci) / 2
    return diffs.mean(), np.quantile(diffs, alpha), np.quantile(diffs, 1 - alpha)


# ---------- main -------------------------------------------------------------

def make_env(n_envs=32, episode_steps=200):
    return TwoTeamVecEnv(
        n_envs=n_envs, device="cuda",
        episode_steps=episode_steps,
        geometry=MIRROR_GEOMETRY,
        team_offset_m=2500.0,
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="all", choices=["train", "eval", "all"])
    p.add_argument("--ppo-iters", type=int, default=80)
    p.add_argument("--ppo-jam-frac", type=float, default=1e-4)
    p.add_argument("--eval-episodes", type=int, default=60)
    p.add_argument("--n-envs", type=int, default=32)
    args = p.parse_args()

    env = make_env(args.n_envs)
    ac = make_ac(env)

    bc_ckpt = os.path.join(CKPT_DIR, "bc_pretrain.pt")
    if os.path.exists(bc_ckpt):
        state = torch.load(bc_ckpt, map_location=env.device, weights_only=False)
        ac.load_state_dict(state["ac_state"])
        print(f"Loaded BC ckpt from {bc_ckpt}")
    else:
        print(f"⚠️ No BC ckpt; aborting (run BC pretrain first)")
        return

    d3_ckpt = os.path.join(CKPT_DIR, "d3_ppo_adaptive.pt")

    if args.phase in ("train", "all"):
        train_rl_vs_adaptive(
            env, ac,
            n_iterations=args.ppo_iters,
            jam_fraction=args.ppo_jam_frac,
            save_path=d3_ckpt,
        )

    if args.phase in ("eval", "all"):
        if os.path.exists(d3_ckpt):
            state = torch.load(d3_ckpt, map_location=env.device, weights_only=False)
            ac.load_state_dict(state["ac_state"])
            print(f"Loaded D3 PPO ckpt from {d3_ckpt}")

        print(f"\n{'='*70}\nD3 EVAL — killer contrast under AdaptiveSpectrum enemy\n{'='*70}")
        jam_fracs = [1e-5, 1e-4, 1e-3]
        rule = OrthConfigStrongRuleWrapper()

        rl_results = {}
        base_results = {}

        for jf in jam_fracs:
            opp = AdaptiveSpectrumOpponent(jam_fraction=jf)
            rl_metrics = {"trace_P": [], "kills": [], "chan_changes": []}
            base_metrics = {"trace_P": [], "kills": []}

            n_per_ep = env.E
            n_eps = max(args.eval_episodes // n_per_ep, 1)
            for _ in range(n_eps):
                m = run_rl_episode(env, ac, opp)
                rl_metrics["trace_P"].extend(m["trace_P"])
                rl_metrics["kills"].extend(m["kills"])
                rl_metrics["chan_changes"].extend(m["chan_changes"])
                m = run_rule_episode(env, rule, opp)
                base_metrics["trace_P"].extend(m["trace_P"])
                base_metrics["kills"].extend(m["kills"])

            for k in rl_metrics:
                rl_metrics[k] = np.array(rl_metrics[k])
            for k in base_metrics:
                base_metrics[k] = np.array(base_metrics[k])
            rl_results[jf] = rl_metrics
            base_results[jf] = base_metrics

            m, lo, hi = bootstrap_ci(rl_metrics["trace_P"])
            bm, blo, bhi = bootstrap_ci(base_metrics["trace_P"])
            print(f"  [jam_frac={jf:.0e}] "
                  f"RL  trace_P = {m:>8.3f} [{lo:>7.2f}, {hi:>7.2f}]  "
                  f"kills={rl_metrics['kills'].mean():.2f}  "
                  f"chan_Δ/s={rl_metrics['chan_changes'].mean():.2f}")
            print(f"  {' ':>15}"
                  f"base trace_P = {bm:>8.3f} [{blo:>7.2f}, {bhi:>7.2f}]  "
                  f"kills={base_metrics['kills'].mean():.2f}")

        # ----- crown judgment -----
        print()
        print("=" * 100)
        print("D3 CROWN JUDGMENT — RL vs competent baseline under AdaptiveSpectrum")
        print("=" * 100)
        print(f"{'jam_frac':>9} | {'RL_trace_P':>10} {'base_trace_P':>14} "
              f"{'gap_mean':>10} {'95% CI':>20} | {'verdict':>14}")
        print("-" * 100)
        n_crown = 0
        for jf in jam_fracs:
            rl_tp = rl_results[jf]["trace_P"]
            base_tp = base_results[jf]["trace_P"]
            n = min(len(rl_tp), len(base_tp))
            diff_mean, diff_lo, diff_hi = bootstrap_diff_ci(base_tp[:n], rl_tp[:n])
            # Crown: RL < baseline (positive gap) AND CI lower bound > 0
            # AND gap ≥ 30% of baseline mean
            base_mean = base_tp.mean()
            gap_pct = diff_mean / max(base_mean, 1e-9) * 100
            ci_separated = diff_lo > 0
            large_enough = gap_pct >= 30.0
            if ci_separated and large_enough:
                verdict = "RL CROWN"
                n_crown += 1
            elif diff_mean > 0 and gap_pct >= 10:
                verdict = "RL marginal"
            elif abs(gap_pct) < 10:
                verdict = "tie / IET"
            else:
                verdict = "RL loses"
            print(f"{jf:>9.0e} | {rl_tp.mean():>10.3f} {base_mean:>14.3f} "
                  f"{diff_mean:>+10.3f} [{diff_lo:>+8.2f}, {diff_hi:>+8.2f}] | {verdict:>14}")

        print(f"\n  RL crowns at {n_crown}/{len(jam_fracs)} jam_frac points")
        if n_crown >= 2:
            print("  → D3 CROWN: RL beats competent baseline under dynamic enemy")
            print("    Proceed to D4 (production 5e7-step training)")
        elif n_crown == 1:
            print("  → D3 MARGINAL: crown signal at one point; "
                  "needs more training or different regime")
        else:
            print("  → D3 IET FLOOR: competent baseline is enough under dynamic enemy")
            print("    Honest finding — no production training warranted")


if __name__ == "__main__":
    main()
