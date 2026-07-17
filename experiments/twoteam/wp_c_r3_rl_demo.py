"""WP-C R3 — Learning commander (BC pretrain → PPO fine-tune → cross-play eval).

Per WP-C plan R3:
  1. BC pretrain: collect demos from [StrongRule + external orth config],
     NLL-fit AC's 6-head actor (5 original + channel_select new in R3).
  2. PPO fine-tune: TwoTeamBRTrainer vs {StrongRule+orth baseline, ReactiveJammer};
     channel_select head learns dynamic coordination (escape reactive follower).
  3. Cross-play eval: RL vs {competent baseline, reactive enemy, fixed jammer}
     × f_emit ∈ [1e-6, 1e-3]. Metrics: trace_P (primary), kills (auxiliary at
     tau=4.0), channel_select entropy (RL动态协同 evidence).

This is a SANITY-CHECK scope script — BC pretrain uses small n_samples, PPO
runs ~100 iters. The full 5e7-step league training is out of scope here; this
script establishes the pipeline works and reports preliminary results.

Usage:
  python experiments/twoteam/wp_c_r3_rl_demo.py --phase bc
  python experiments/twoteam/wp_c_r3_rl_demo.py --phase ppo
  python experiments/twoteam/wp_c_r3_rl_demo.py --phase eval
  python experiments/twoteam/wp_c_r3_rl_demo.py --phase all
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import argparse
import math
import os
import time
import torch
import numpy as np

from env.gpu.twoteam import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.baselines.reactive_jammer_commander import ReactiveJammerCommander
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.bc_pretrain import TwoTeamBCPretrainer
from algo._shared.pilot.twoteam.br_trainer import TwoTeamBRTrainer
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


CKPT_DIR = "/home/ubuntu/CODE/FluxPhased-/experiments/twoteam/_wp_c_ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)


def configure_channels(env, mode: str = "orthogonal"):
    """Set mirror-symmetric orthogonal channels (competent baseline config)."""
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
    """StrongRule + external orth channel config wrapper.

    Acts as frozen opponent for PPO training. Wraps StrongRule.get_action
    so that channel_select reflects the orth config (ch0/ch1) regardless
    of env state (since env.radar_freq_hz may be mutated by ReactiveJammer
    during training episodes).
    """
    def __init__(self):
        self.rule = TwoTeamStrongRuleCommander()

    def get_action(self, env, team: int):
        a = self.rule.get_action(env, team)
        # Force orth config: radar 0 = ch0, radar 1 = ch1
        E, R = env.E, env.n_radars_per_team
        ch = torch.zeros(E, R, dtype=torch.long, device=env.device)
        ch[:, 0] = 0
        ch[:, 1] = 1
        a["channel_select"] = ch
        return a


# ---------- Phase 1: BC pretrain ---------------------------------------------

def phase_bc(env, ac, n_samples=20000, n_epochs=10):
    """Collect demos from competent baseline (StrongRule + orth) and NLL-fit AC."""
    print(f"\n{'='*70}\nPhase BC: collect {n_samples} demos from StrongRule+orth, NLL-fit\n{'='*70}")
    rule = TwoTeamStrongRuleCommander()
    # Wrapper: each episode, set orth config so BC learns orth as channel prior
    bc = TwoTeamBCPretrainer(ac, lr=1e-3, batch_size=256)

    # Custom collect that sets orth config per episode
    samples = _collect_orth_demos(env, rule, n_samples=n_samples)
    history = bc.train(samples, n_epochs=n_epochs, early_stop_patience=3)

    save_path = os.path.join(CKPT_DIR, "bc_pretrain.pt")
    bc.save(save_path, history)
    print(f"  BC saved → {save_path}")
    return history


def _collect_orth_demos(env, rule, n_samples=20000, episode_steps=200):
    """Collect demos with orth config reset each episode."""
    from algo._shared.pilot.twoteam.extreme_commanders import STRATEGIES
    opp_strategies = ["pure_track", "pure_jam", "balanced", "track_agile"]

    E = env.E
    obs_buf, priv_buf = [], []
    task_buf, beam_buf, laser_buf, emit_buf, fh_buf, chan_buf = [], [], [], [], [], []

    total, ep, t0 = 0, 0, time.time()
    while total < n_samples:
        opp = STRATEGIES[opp_strategies[ep % len(opp_strategies)]]
        env.reset()
        configure_channels(env, "orthogonal")
        for step in range(episode_steps):
            obs_dict = env.get_obs()
            a_rule = rule.get_action(env, 0)
            a_opp = opp.get_action(env, 1)
            obs_buf.append(obs_dict["obs"][:, 0].clone())
            priv_buf.append(obs_dict["privileged"][:, 0].clone())
            task_buf.append(a_rule["task_alloc"].clone())
            beam_buf.append(a_rule["beam_target"].clone())
            laser_buf.append(a_rule["laser_target"].clone())
            emit_buf.append(a_rule["emission_on"].clone())
            fh_buf.append(a_rule["freq_hop_rate"].clone())
            # channel_select: BC target = orth config (ch0 / ch1)
            ch_orth = torch.zeros(E, env.n_radars_per_team, dtype=torch.long, device=env.device)
            ch_orth[:, 0] = 0
            ch_orth[:, 1] = 1
            chan_buf.append(ch_orth.clone())
            total += E
            if total >= n_samples:
                break
            action = combine_team_actions(env, a_rule, a_opp)
            env.step(action)
        ep += 1
        if ep % 5 == 0:
            print(f"  [BC collect] ep={ep} total={total}/{n_samples} t={time.time()-t0:.1f}s",
                  flush=True)

    n = n_samples
    return {
        "obs": torch.cat([b for b in obs_buf], dim=0)[:n].to(torch.float32),
        "priv": torch.cat([b for b in priv_buf], dim=0)[:n].to(torch.float32),
        "task_alloc": torch.cat([b for b in task_buf], dim=0)[:n].to(torch.float32),
        "beam_target": torch.cat([b for b in beam_buf], dim=0)[:n].to(torch.long),
        "laser_target": torch.cat([b for b in laser_buf], dim=0)[:n].to(torch.long),
        "emission_on": torch.cat([b for b in emit_buf], dim=0)[:n].to(torch.float32),
        "freq_hop_rate": torch.cat([b for b in fh_buf], dim=0)[:n].to(torch.float32),
        "channel_select": torch.cat([b for b in chan_buf], dim=0)[:n].to(torch.long),
    }


# ---------- Phase 2: PPO fine-tune -------------------------------------------

class ReactiveOpponentWrapper:
    """ReactiveJammer opponent for PPO training."""
    def __init__(self, jam_fraction=1e-4):
        self.jammer = ReactiveJammerCommander(jam_fraction=jam_fraction)

    def get_action(self, env, team):
        return self.jammer.get_action(env, team)


def phase_ppo(env, ac, n_iterations=100, horizon=300, opponent="competent"):
    """PPO fine-tune vs chosen opponent."""
    print(f"\n{'='*70}\nPhase PPO: {n_iterations} iters × H={horizon} vs opponent={opponent}\n{'='*70}")
    if opponent == "competent":
        frozen = OrthConfigStrongRuleWrapper()
    elif opponent == "reactive":
        frozen = ReactiveOpponentWrapper(jam_fraction=1e-4)
    else:
        raise ValueError(f"unknown opponent: {opponent}")

    trainer = TwoTeamBRTrainer(
        ac, frozen,
        lr_actor=3e-4, lr_critic=1e-3,
        entropy_coef=0.02,
        n_epochs=4, minibatch_size=64,
        reward_scale=0.1,
        device="cuda",
    )
    save_path = os.path.join(CKPT_DIR, f"ppo_{opponent}.pt")
    history = trainer.train(
        env, n_iterations=n_iterations, horizon=horizon,
        learning_team=0, save_path=save_path, log_every=10,
    )
    print(f"  PPO saved → {save_path}")
    return history


# ---------- Phase 3: Cross-play evaluation -----------------------------------

def _run_eval_episode(env, rl_ac, opponent, f_emit_A, max_steps=200):
    """Run one episode: RL (team 0) vs opponent (team 1). Opponent may be
    'competent' (StrongRule+orth), 'reactive' (ReactiveJammer), or 'fixed' (WP-B)."""
    env.reset()
    configure_channels(env, "orthogonal")
    if opponent == "competent":
        opp = OrthConfigStrongRuleWrapper()
    elif opponent == "reactive":
        opp = ReactiveOpponentWrapper(jam_fraction=f_emit_A if f_emit_A > 0 else 1e-5)
    else:  # fixed (use the WP-B-style fixed jammer)
        opp = _FixedJammerOpponent(f_emit_A)

    ep_trace_P = torch.zeros(env.E, device=env.device)
    ep_chan_changes = torch.zeros(env.E, device=env.device)
    last_chan = None
    ep_steps = 0
    last_info = None

    for step in range(max_steps):
        obs_dict = env.get_obs()
        obs_t0 = obs_dict["obs"][:, 0]
        detect_t0 = env.get_detect_list()[:, 0]
        priv_t0 = obs_dict["privileged"][:, 0]
        a_rl, _ = rl_ac.get_action_for_env(obs_t0, detect_t0, priv_t0, deterministic=True)
        a_opp = opp.get_action(env, team=1)
        action = combine_team_actions(env, a_rl, a_opp)
        obs_dict, reward, done, info = env.step(action)
        last_info = info

        trace_P = env.tracker_P[:, 0, :, 0, 0] + env.tracker_P[:, 0, :, 2, 2]
        victim_alive = env.radar_alive[:, 1]   # opponent (team 1) radars alive
        alive_mask = victim_alive.float()
        n_alive = alive_mask.sum(dim=-1).clamp(min=1.0)
        ep_trace_P += (trace_P * alive_mask).sum(dim=-1) / n_alive

        # Count RL channel changes (dynamic coordination evidence)
        rl_chan = a_rl["channel_select"]   # [E, R]
        if last_chan is not None:
            changes = (rl_chan != last_chan).any(dim=-1).float()
            ep_chan_changes += changes
        last_chan = rl_chan.clone()

        ep_steps += 1
        if done.all():
            break

    kills_rl = last_info["team_kills"][:, 0].float()   # RL team kills
    sc = max(ep_steps, 1)
    return {
        "trace_P": (ep_trace_P / sc).cpu().numpy(),
        "kills": kills_rl.cpu().numpy(),
        "chan_changes": (ep_chan_changes / sc).cpu().numpy(),
    }


class _FixedJammerOpponent:
    """WP-B style fixed fraction jammer as eval opponent."""
    def __init__(self, f_emit_A):
        self.f_emit_A = f_emit_A

    def get_action(self, env, team):
        E, R = env.E, env.n_radars_per_team
        dev = env.device
        alloc = torch.zeros(E, R, 4, device=dev)
        if self.f_emit_A > 0:
            alloc[:, :, 2] = self.f_emit_A
            alloc[:, :, 3] = max(0.0, 1.0 - self.f_emit_A)
        else:
            alloc[:, :, 3] = 1.0
        ch = ((env.radar_freq_hz[:, team, :] - env.fc_hz)
              / env.channel_spacing_hz).round().long().clamp(0, env.n_channels - 1)
        return {
            "task_alloc": alloc,
            "beam_target": torch.zeros(E, R, dtype=torch.long, device=dev),
            "laser_target": torch.zeros(E, dtype=torch.long, device=dev),
            "emission_on": torch.ones(E, R, device=dev) if self.f_emit_A > 0
                           else torch.zeros(E, R, device=dev),
            "freq_hop_rate": torch.ones(E, R, device=dev),
            "channel_select": ch,
        }


def phase_eval(env, rl_ac, n_episodes=50):
    """Cross-play: RL vs {competent, reactive, fixed} × f_emit ∈ [1e-6, 1e-3]."""
    print(f"\n{'='*70}\nPhase EVAL: RL vs {{competent, reactive, fixed}} × f_emit\n{'='*70}")
    f_emits = [1e-6, 1e-5, 1e-4, 1e-3]
    opp_types = ["competent", "reactive", "fixed"]

    results = {}
    for opp in opp_types:
        for f in f_emits:
            metrics = {"trace_P": [], "kills": [], "chan_changes": []}
            for _ in range(n_episodes):
                m = _run_eval_episode(env, rl_ac, opp, f)
                metrics["trace_P"].extend(m["trace_P"])
                metrics["kills"].extend(m["kills"])
                metrics["chan_changes"].extend(m["chan_changes"])
            results[(opp, f)] = {k: np.array(v) for k, v in metrics.items()}
            r = results[(opp, f)]
            print(f"  [opp={opp:>9} f={f:.0e}] "
                  f"trace_P={r['trace_P'].mean():>7.2f}±{r['trace_P'].std():>5.2f}  "
                  f"kills={r['kills'].mean():.2f}  "
                  f"chan_Δ/s={r['chan_changes'].mean():.2f}")

    # Also eval the competent baseline against itself for reference
    print("\n  Baseline reference (competent vs each opponent):")
    rule = OrthConfigStrongRuleWrapper()
    baseline_results = {}
    for opp in opp_types:
        for f in f_emits:
            metrics = {"trace_P": [], "kills": []}
            for _ in range(n_episodes // 2):
                m = _run_eval_episode_rule_vs_opp(env, rule, opp, f)
                metrics["trace_P"].extend(m["trace_P"])
                metrics["kills"].extend(m["kills"])
            baseline_results[(opp, f)] = {k: np.array(v) for k, v in metrics.items()}
            r = baseline_results[(opp, f)]
            print(f"  [base vs {opp:>9} f={f:.0e}] "
                  f"trace_P={r['trace_P'].mean():>7.2f}  kills={r['kills'].mean():.2f}")

    # Crown judgement
    print()
    print("=" * 100)
    print("R3 CROWN JUDGEMENT — RL vs competent baseline (trace_P comparison)")
    print("=" * 100)
    print(f"{'f_emit':>9} | {'RL_trace_P':>12} {'base_trace_P':>14} {'gap%':>8} | {'verdict':>12}")
    print("-" * 70)
    n_win = 0
    for f in f_emits:
        rl_tp = results[("competent", f)]["trace_P"].mean()
        base_tp = baseline_results[("competent", f)]["trace_P"].mean()
        if base_tp > 1e-6:
            gap_pct = (base_tp - rl_tp) / base_tp * 100
        else:
            gap_pct = 0.0
        verdict = "RL wins" if rl_tp < base_tp * 0.7 else ("RL marginal" if rl_tp < base_tp else "RL loses")
        if verdict == "RL wins":
            n_win += 1
        print(f"{f:>9.0e} | {rl_tp:>12.3f} {base_tp:>14.3f} {gap_pct:>+7.1f}% | {verdict:>12}")

    print(f"\n  RL wins (≥30% trace_P reduction) at {n_win}/{len(f_emits)} f_emit points")
    if n_win >= 3:
        print("  → CROWN: dynamic learning coordination beats competent fixed rule")
    elif n_win >= 1:
        print("  → MARGINAL: needs more training or different regime")
    else:
        print("  → IET FLOOR: competent fixed rule is enough; honest finding")

    return results, baseline_results


def _run_eval_episode_rule_vs_opp(env, rule, opponent_type, f_emit_A, max_steps=200):
    """For baseline reference: rule vs opponent."""
    env.reset()
    configure_channels(env, "orthogonal")
    if opponent_type == "competent":
        opp = OrthConfigStrongRuleWrapper()
    elif opponent_type == "reactive":
        opp = ReactiveOpponentWrapper(jam_fraction=f_emit_A if f_emit_A > 0 else 1e-5)
    else:
        opp = _FixedJammerOpponent(f_emit_A)

    ep_trace_P = torch.zeros(env.E, device=env.device)
    last_info = None
    for step in range(max_steps):
        a_rule = rule.get_action(env, 0)
        a_opp = opp.get_action(env, 1)
        action = combine_team_actions(env, a_rule, a_opp)
        _, _, done, info = env.step(action)
        last_info = info
        trace_P = env.tracker_P[:, 0, :, 0, 0] + env.tracker_P[:, 0, :, 2, 2]
        victim_alive = env.radar_alive[:, 1]
        alive_mask = victim_alive.float()
        n_alive = alive_mask.sum(dim=-1).clamp(min=1.0)
        ep_trace_P += (trace_P * alive_mask).sum(dim=-1) / n_alive
        if done.all():
            break
    kills = last_info["team_kills"][:, 0].float()
    return {
        "trace_P": (ep_trace_P / max_steps).cpu().numpy(),
        "kills": kills.cpu().numpy(),
    }


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
    p.add_argument("--phase", default="all", choices=["bc", "ppo", "eval", "all"])
    p.add_argument("--bc-samples", type=int, default=20000)
    p.add_argument("--bc-epochs", type=int, default=10)
    p.add_argument("--ppo-iters", type=int, default=100)
    p.add_argument("--ppo-opponent", default="competent",
                   choices=["competent", "reactive"])
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--n-envs", type=int, default=32)
    args = p.parse_args()

    env = make_env(args.n_envs)
    ac = make_ac(env)

    # Load BC ckpt if exists and phase != bc
    bc_ckpt = os.path.join(CKPT_DIR, "bc_pretrain.pt")
    if args.phase in ("ppo", "eval", "all") and os.path.exists(bc_ckpt):
        state = torch.load(bc_ckpt, map_location=env.device)
        ac.load_state_dict(state["ac_state"])
        print(f"Loaded BC ckpt from {bc_ckpt}")
    elif args.phase in ("ppo", "eval", "all"):
        print(f"⚠️ No BC ckpt at {bc_ckpt} — PPO/eval will use random init actor")

    # Load PPO ckpt if exists and phase == eval
    ppo_ckpt = os.path.join(CKPT_DIR, f"ppo_{args.ppo_opponent}.pt")
    if args.phase == "eval" and os.path.exists(ppo_ckpt):
        state = torch.load(ppo_ckpt, map_location=env.device)
        ac.load_state_dict(state["ac_state"])
        print(f"Loaded PPO ckpt from {ppo_ckpt}")

    if args.phase in ("bc", "all"):
        phase_bc(env, ac, n_samples=args.bc_samples, n_epochs=args.bc_epochs)

    if args.phase in ("ppo", "all"):
        phase_ppo(env, ac, n_iterations=args.ppo_iters, opponent=args.ppo_opponent)

    if args.phase in ("eval", "all"):
        phase_eval(env, ac, n_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
