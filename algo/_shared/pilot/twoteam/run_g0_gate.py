"""G0 exploitability gate driver for two-team env (WP1).

Per plan snuggly-exploring-parrot.md Step 4 + TWOTEAM_MULTIFUNCTION_PLAN.md §WP1.2.

G0 = U(π_rule vs mirror π_rule) − U(π_rule vs BR(π_rule))
where U = mean(kills_t0 − kills_t1) over episodes.

PASS: exploit_gap ≥ 0.5 kills/episode AND bootstrap 95% CI excludes 0
FAIL: gap ≈ 0 (BR can't exploit rule) → root A present, retreat to IET.

Output:
  experiments/twoteam/g0_gate_report.md
  experiments/twoteam/g0_mirror_metrics.csv
  experiments/twoteam/g0_br_metrics.csv
  checkpoints/twoteam/br_vs_strong_rule_final.pt
"""

from __future__ import annotations
import os
import sys
import csv
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Callable

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, MIRROR_GEOMETRY, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.br_trainer import TwoTeamBRTrainer
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions, STRATEGIES
from algo._shared.pilot.twoteam.bc_pretrain import TwoTeamBCPretrainer


# ----------------------------------------------------------------------
# Episode runner (rule vs rule, rule vs AC)
# ----------------------------------------------------------------------

def _winner_per_env(kills_t0, kills_t1, alive_t0, alive_t1) -> torch.Tensor:
    """0=t0 wins, 1=t1 wins, -1=draw."""
    return torch.where(
        kills_t0 > kills_t1, torch.zeros_like(kills_t0),
        torch.where(kills_t1 > kills_t0, torch.ones_like(kills_t0),
                    torch.where(alive_t0 > alive_t1, torch.zeros_like(kills_t0),
                                torch.where(alive_t1 > alive_t0, torch.ones_like(kills_t0),
                                            torch.full_like(kills_t0, -1)))))


def run_episodes_two_commanders(
    env, cmd0_fn, cmd1_fn, n_episodes: int, episode_steps: int,
    seed_base: int = 42,
) -> Dict[str, np.ndarray]:
    """Run n_episodes × n_envs episodes. cmd{0,1}_fn(env) → per-team action slice.

    Returns per-episode-per-env arrays (flattened to n_episodes * n_envs):
      kills_t0, kills_t1, alive_t0, alive_t1, winner, exposure_t0, exposure_t1,
      mean_trace_P_t0, mean_trace_P_t1
    """
    E = env.E
    all_metrics = {
        "kills_t0": [], "kills_t1": [], "alive_t0": [], "alive_t1": [],
        "exposure_t0": [], "exposure_t1": [],
        "mean_trace_P_t0": [], "mean_trace_P_t1": [],
    }
    for ep in range(n_episodes):
        env.seed = seed_base + ep
        env._reset_count = ep
        env.reset()
        for step in range(episode_steps):
            a0 = cmd0_fn(env, 0)
            a1 = cmd1_fn(env, 1)
            action = combine_team_actions(env, a0, a1)
            obs, r, done, info = env.step(action)
            if done.all():
                break
        # Final info
        all_metrics["kills_t0"].append(info["team_kills"][:, 0].cpu().numpy())
        all_metrics["kills_t1"].append(info["team_kills"][:, 1].cpu().numpy())
        all_metrics["alive_t0"].append(info["team_alive"][:, 0].cpu().numpy().astype(int))
        all_metrics["alive_t1"].append(info["team_alive"][:, 1].cpu().numpy().astype(int))
        all_metrics["exposure_t0"].append(info["exposure"][:, 0].cpu().numpy())
        all_metrics["exposure_t1"].append(info["exposure"][:, 1].cpu().numpy())
        all_metrics["mean_trace_P_t0"].append(info["mean_trace_P"][:, 0].cpu().numpy())
        all_metrics["mean_trace_P_t1"].append(info["mean_trace_P"][:, 1].cpu().numpy())

    # Stack: [n_episodes, E] → flatten
    out = {}
    for k, v in all_metrics.items():
        arr = np.stack(v, axis=0).reshape(-1)
        out[k] = arr
    # Winner per env per episode (computed last)
    out["winner"] = _winner_per_env(
        torch.tensor(out["kills_t0"]), torch.tensor(out["kills_t1"]),
        torch.tensor(out["alive_t0"]), torch.tensor(out["alive_t1"]),
    ).numpy()
    return out


def make_ac_action_fn(ac: TwoTeamCommanderActorCritic, deterministic: bool = True):
    """Wrap AC into a (env, team) → action callable for run_episodes_two_commanders."""
    @torch.no_grad()
    def fn(env, team):
        obs_dict = env.get_obs()
        action, _ = ac.get_action_for_env(
            obs_dict["obs"][:, team], obs_dict["privileged"][:, team],
            deterministic=deterministic)
        return action
    return fn


# ----------------------------------------------------------------------
# Anti-strawman check for StrongRule
# ----------------------------------------------------------------------

def anti_strawman_check(strong_rule, episode_steps: int = 200,
                        n_episodes: int = 5) -> Dict:
    """Verify StrongRule is neither too weak (loses to extremes) nor too strong
    (beats balanced > 70% — would make BR training infeasible)."""
    print("\n=== Anti-strawman check: StrongRule vs ExtremeCommanders ===", flush=True)
    env = TwoTeamVecEnv(n_envs=8, device="cuda", episode_steps=episode_steps,
                         geometry=RANDOM_GEOMETRY, seed=42)
    results = {}
    for name in ["pure_track", "pure_jam", "pure_comm", "pure_detect",
                 "balanced", "balanced_jam_heavy"]:
        opp = STRATEGIES[name]
        # rule=team0, opp=team1
        def cmd0(env, team, _r=strong_rule): return _r.get_action(env, team)
        def cmd1(env, team, _o=opp): return _o.get_action(env, team)
        m = run_episodes_two_commanders(env, cmd0, cmd1, n_episodes, episode_steps)
        wins = (m["winner"] == 0).mean()
        draws = (m["winner"] == -1).mean()
        results[name] = {
            "win_rate": float(wins),
            "draw_rate": float(draws),
            "kills_rule": float(m["kills_t0"].mean()),
            "kills_opp": float(m["kills_t1"].mean()),
        }
        print(f"  rule vs {name:25s}: WR={wins:.2f} draws={draws:.2f} "
              f"kills {m['kills_t0'].mean():.2f} vs {m['kills_t1'].mean():.2f}",
              flush=True)

    # Anti-strawman verdict
    extreme_wins = [results[n]["win_rate"] for n in
                    ["pure_track", "pure_jam", "pure_comm", "pure_detect"]]
    n_extreme_wins = sum(1 for w in extreme_wins if w >= 0.80)
    balanced_wr = results["balanced"]["win_rate"]
    jam_heavy_wr = results["balanced_jam_heavy"]["win_rate"]

    too_weak = n_extreme_wins < 4
    too_strong = (balanced_wr > 0.80) or (jam_heavy_wr > 0.80)
    if too_weak:
        verdict = "TOO_WEAK"
    elif too_strong:
        verdict = "TOO_STRONG"
    else:
        verdict = "OK"

    print(f"  verdict: {verdict} (extreme wins ≥80%: {n_extreme_wins}/4, "
          f"balanced WR: {balanced_wr:.2f}, jam_heavy WR: {jam_heavy_wr:.2f})", flush=True)
    return {"per_strategy": results, "verdict": verdict,
            "n_extreme_wins": n_extreme_wins,
            "balanced_wr": balanced_wr,
            "jam_heavy_wr": jam_heavy_wr}


# ----------------------------------------------------------------------
# Bootstrap CI
# ----------------------------------------------------------------------

def bootstrap_ci(samples: np.ndarray, n_boot: int = 10000, alpha: float = 0.05):
    """Bootstrap mean CI. Returns (mean, ci_low, ci_high)."""
    rng = np.random.RandomState(42)
    n = len(samples)
    means = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[b] = samples[idx].mean()
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(samples.mean()), float(lo), float(hi)


# ----------------------------------------------------------------------
# Main G0 gate
# ----------------------------------------------------------------------

def main(br_iters: int = 200, horizon: int = 200, n_envs: int = 8,
         n_episodes: int = 30, br_lr_actor: float = 3e-4,
         br_lr_critic: float = 1e-3, br_entropy_coef: float = 0.01,
         br_lr_decay_iters: int = 0,
         bc_pretrain_samples: int = 0,
         bc_pretrain_epochs: int = 10,
         bc_pretrain_batch_size: int = 256,
         bc_pretrain_lr: float = 1e-3,
         out_dir: str = "/home/ubuntu/CODE/FluxPhased-/experiments/twoteam"):

    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = "/home/ubuntu/CODE/FluxPhased-/checkpoints/twoteam"
    os.makedirs(ckpt_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 70, flush=True)
    print("G0 EXPLOITABILITY GATE — Two-team WP1", flush=True)
    print("=" * 70, flush=True)

    # === Step A: Anti-strawman check on StrongRule ===
    rule = TwoTeamStrongRuleCommander()
    anti = anti_strawman_check(rule, episode_steps=horizon, n_episodes=5)

    # === Step B: Cell 1 — π_rule vs π_rule (mirror) ===
    print(f"\n=== Cell 1: π_rule vs π_rule (mirror, {n_episodes} eps × {n_envs} envs) ===",
          flush=True)
    env = TwoTeamVecEnv(n_envs=n_envs, device="cuda", episode_steps=horizon,
                         geometry=RANDOM_GEOMETRY, seed=42)
    def rule_fn(env, team, _r=rule): return _r.get_action(env, team)
    mirror_metrics = run_episodes_two_commanders(
        env, rule_fn, rule_fn, n_episodes, horizon, seed_base=42)
    mirror_margin_samples = mirror_metrics["kills_t0"] - mirror_metrics["kills_t1"]
    mirror_margin_mean, mirror_ci_lo, mirror_ci_hi = bootstrap_ci(mirror_margin_samples)
    print(f"  mirror margin: {mirror_margin_mean:+.3f} "
          f"(95% CI [{mirror_ci_lo:+.3f}, {mirror_ci_hi:+.3f}])", flush=True)
    print(f"  kills: rule_t0={mirror_metrics['kills_t0'].mean():.2f}, "
          f"rule_t1={mirror_metrics['kills_t1'].mean():.2f}", flush=True)
    print(f"  winner: t0={(mirror_metrics['winner']==0).mean():.2f}, "
          f"t1={(mirror_metrics['winner']==1).mean():.2f}, "
          f"draw={(mirror_metrics['winner']==-1).mean():.2f}", flush=True)

    # === Step C: BR training ===
    print(f"\n=== BR training: BR(π_rule) vs π_rule frozen ({br_iters} iters) ===",
          flush=True)
    print(f"  hyperparams: lr_actor={br_lr_actor} lr_critic={br_lr_critic} "
          f"entropy_coef={br_entropy_coef} lr_decay_iters={br_lr_decay_iters}", flush=True)

    br_ac = TwoTeamCommanderActorCritic().to("cuda")

    # === Step C0: BC pretrain (NEW) ===
    bc_history: List = []
    bc_save_path = os.path.join(ckpt_dir, "bc_pretrained.pt")
    if bc_pretrain_samples > 0:
        print(f"\n=== Step C0: BC pretrain (collect {bc_pretrain_samples} samples from StrongRule) ===",
              flush=True)
        print(f"  BC hyperparams: epochs={bc_pretrain_epochs} batch_size={bc_pretrain_batch_size} "
              f"lr={bc_pretrain_lr}", flush=True)
        bc_env = TwoTeamVecEnv(n_envs=n_envs, device="cuda", episode_steps=horizon,
                                geometry=RANDOM_GEOMETRY, seed=100)
        bc_trainer = TwoTeamBCPretrainer(
            br_ac, lr=bc_pretrain_lr, batch_size=bc_pretrain_batch_size)
        samples = bc_trainer.collect_samples(
            bc_env, rule, n_samples=bc_pretrain_samples, episode_steps=horizon)
        print(f"  collected {samples['obs'].shape[0]} samples", flush=True)

        bc_history = bc_trainer.train(samples, n_epochs=bc_pretrain_epochs)
        bc_trainer.save(bc_save_path, bc_history)

        # Sanity: BC'd AC deterministic policy snapshot
        print(f"\n  BC sanity check — deterministic action profile (1 episode, 5 steps):",
              flush=True)
        sanity_env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=horizon,
                                    geometry=RANDOM_GEOMETRY, seed=200)
        sanity_env.reset()
        obs_dict = sanity_env.get_obs()
        with torch.no_grad():
            bc_action, _ = br_ac.get_action_for_env(
                obs_dict["obs"][:, 0], obs_dict["privileged"][:, 0], deterministic=True)
        ta_profile = bc_action["task_alloc"][0].mean(dim=0)   # [n_fn] averaged over apertures
        hop_mean = bc_action["freq_hop_rate"][0].mean().item()
        print(f"    BC task_alloc profile (4 fns, avg over 2 apertures): "
              f"[{ta_profile[0]:.2f}, {ta_profile[1]:.2f}, {ta_profile[2]:.2f}, {ta_profile[3]:.2f}]",
              flush=True)
        print(f"    BC freq_hop mean: {hop_mean:.2f}", flush=True)
        print(f"    Rule task_alloc profile: [0.10, 0.71, 0.09, 0.10] (approx, varies by scenario)",
              flush=True)
    br_env = TwoTeamVecEnv(n_envs=n_envs, device="cuda", episode_steps=horizon,
                            geometry=RANDOM_GEOMETRY, seed=43)
    trainer = TwoTeamBRTrainer(
        br_ac, frozen_opponent=rule,
        lr_actor=br_lr_actor, lr_critic=br_lr_critic,
        entropy_coef=br_entropy_coef,
        lr_decay_iters=br_lr_decay_iters,
        device="cuda")
    history: List = []
    br_save_path = os.path.join(ckpt_dir, "br_vs_strong_rule_final.pt")
    trainer.train(br_env, n_iterations=br_iters, horizon=horizon,
                  learning_team=0, save_path=br_save_path,
                  log_every=10, log_history=history)

    # === Step D: Cell 2 — π_rule vs BR(π_rule) ===
    print(f"\n=== Cell 2: π_rule vs BR(π_rule) ({n_episodes} eps × {n_envs} envs) ===",
          flush=True)
    eval_env = TwoTeamVecEnv(n_envs=n_envs, device="cuda", episode_steps=horizon,
                             geometry=RANDOM_GEOMETRY, seed=44)
    br_fn = make_ac_action_fn(br_ac, deterministic=True)
    br_metrics = run_episodes_two_commanders(
        eval_env, rule_fn, br_fn, n_episodes, horizon, seed_base=44)
    # rule = team0, BR = team1; margin = kills_t0 - kills_t1 (rule's POV)
    br_margin_samples = br_metrics["kills_t0"] - br_metrics["kills_t1"]
    br_margin_mean, br_ci_lo, br_ci_hi = bootstrap_ci(br_margin_samples)
    print(f"  br margin (rule POV): {br_margin_mean:+.3f} "
          f"(95% CI [{br_ci_lo:+.3f}, {br_ci_hi:+.3f}])", flush=True)
    print(f"  kills: rule_t0={br_metrics['kills_t0'].mean():.2f}, "
          f"br_t1={br_metrics['kills_t1'].mean():.2f}", flush=True)
    print(f"  winner: rule={(br_metrics['winner']==0).mean():.2f}, "
          f"BR={(br_metrics['winner']==1).mean():.2f}, "
          f"draw={(br_metrics['winner']==-1).mean():.2f}", flush=True)

    # === Step E: Compute exploitability ===
    # exploit_gap_samples = mirror_margin - br_margin  (per env-episode paired not required; treat as 2 indep samples)
    # We compute gap = mean(mirror_margin) - mean(br_margin), and bootstrap by
    # resampling both arrays independently and subtracting.
    rng = np.random.RandomState(123)
    n_boot = 10000
    n_m = len(mirror_margin_samples)
    n_b = len(br_margin_samples)
    gap_samples = np.zeros(n_boot)
    for b in range(n_boot):
        m_idx = rng.randint(0, n_m, size=n_m)
        b_idx = rng.randint(0, n_b, size=n_b)
        gap_samples[b] = mirror_margin_samples[m_idx].mean() - br_margin_samples[b_idx].mean()
    gap_mean = float(gap_samples.mean())
    gap_ci_lo = float(np.percentile(gap_samples, 2.5))
    gap_ci_hi = float(np.percentile(gap_samples, 97.5))

    print(f"\n=== Exploitability ===", flush=True)
    print(f"  mirror_margin = {mirror_margin_mean:+.3f}", flush=True)
    print(f"  br_margin     = {br_margin_mean:+.3f}", flush=True)
    print(f"  exploit_gap   = {gap_mean:+.3f} (95% CI [{gap_ci_lo:+.3f}, {gap_ci_hi:+.3f}])",
          flush=True)

    # BR training health
    final_metrics = history[-1] if history else {}
    br_healthy = (
        0.1 < final_metrics.get("adv_std", 0) < 100.0
        and -10 < final_metrics.get("entropy", 0) < 10
        and not any(torch.isnan(p).any().item() for p in br_ac.parameters())
    )
    br_win_rate = float((br_metrics["winner"] == 1).mean())

    # G0 verdict
    gap_pass = (gap_mean >= 0.5) and (gap_ci_lo > 0)
    ci_pass = gap_ci_lo > 0
    direction_pass = br_win_rate >= 0.55   # BR wins majority

    g0_pass = gap_pass and br_healthy

    print(f"\n=== G0 VERDICT ===", flush=True)
    print(f"  exploit_gap >= 0.5 AND CI excludes 0: {gap_pass}", flush=True)
    print(f"  CI excludes 0:                        {ci_pass}", flush=True)
    print(f"  BR win rate >= 0.55:                  {direction_pass} (actual={br_win_rate:.2f})",
          flush=True)
    print(f"  BR training healthy:                  {br_healthy}", flush=True)
    print(f"  adv_std (last):                       {final_metrics.get('adv_std', 'NA')}", flush=True)
    print(f"  entropy (last):                       {final_metrics.get('entropy', 'NA')}", flush=True)
    if g0_pass:
        print(f"\n  ✅ G0 PASS — exploitability confirmed. Rule is exploitable.", flush=True)
        print(f"     → Recommend: proceed to WP2 self-play / league training.", flush=True)
    else:
        print(f"\n  ❌ G0 FAIL — exploitability not confirmed.", flush=True)
        print(f"     → Recommend: pause. Either BR undertrained, or rule is genuinely", flush=True)
        print(f"       not exploitable (root A present). Discuss IET retreat.", flush=True)

    elapsed = (time.time() - t_start) / 60.0

    # === Write CSVs ===
    mirror_csv = os.path.join(out_dir, "g0_mirror_metrics.csv")
    with open(mirror_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ep", "env", "kills_t0", "kills_t1", "winner", "exposure_t0",
                    "mean_trace_P_t0", "mean_trace_P_t1"])
        n_total = len(mirror_metrics["kills_t0"])
        for i in range(n_total):
            w.writerow([i // n_envs, i % n_envs,
                        mirror_metrics["kills_t0"][i], mirror_metrics["kills_t1"][i],
                        int(mirror_metrics["winner"][i]),
                        mirror_metrics["exposure_t0"][i],
                        mirror_metrics["mean_trace_P_t0"][i],
                        mirror_metrics["mean_trace_P_t1"][i]])

    br_csv = os.path.join(out_dir, "g0_br_metrics.csv")
    with open(br_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ep", "env", "kills_rule", "kills_br", "winner", "exposure_rule",
                    "exposure_br", "mean_trace_P_rule", "mean_trace_P_br"])
        n_total = len(br_metrics["kills_t0"])
        for i in range(n_total):
            w.writerow([i // n_envs, i % n_envs,
                        br_metrics["kills_t0"][i], br_metrics["kills_t1"][i],
                        int(br_metrics["winner"][i]),
                        br_metrics["exposure_t0"][i], br_metrics["exposure_t1"][i],
                        br_metrics["mean_trace_P_t0"][i], br_metrics["mean_trace_P_t1"][i]])

    br_log_csv = os.path.join(out_dir, "g0_br_training_log.csv")
    with open(br_log_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "reward_mean", "policy_loss", "value_loss", "entropy",
                    "approx_kl", "adv_std", "clip_frac", "early_stop", "elapsed_min"])
        for row in history:
            w.writerow([row["iter"], row["reward_mean"], row["policy_loss"],
                        row["value_loss"], row["entropy"], row["approx_kl"],
                        row["adv_std"], row["clip_frac"], int(row["early_stop"]),
                        row["elapsed_min"]])

    # === Write markdown report ===
    report_path = os.path.join(out_dir, "g0_gate_report.md")
    with open(report_path, "w") as f:
        f.write("# G0 Exploitability Gate Report — Two-team WP1\n\n")
        f.write(f"**Date**: 2026-07-13\n")
        f.write(f"**Spec**: TWOTEAM_MULTIFUNCTION_PLAN.md §WP1.2\n")
        f.write(f"**Elapsed**: {elapsed:.1f} min\n")
        f.write(f"**Overall**: {'✅ PASS — proceed to WP2 self-play' if g0_pass else '❌ FAIL — diagnose before WP2'}\n\n")

        f.write("## Anti-strawman check (StrongRule vs ExtremeCommanders)\n\n")
        f.write(f"Verdict: **{anti['verdict']}**\n\n")
        f.write("| opponent | win_rate | draw_rate | kills_rule | kills_opp |\n")
        f.write("|---|---|---|---|---|\n")
        for name, r in anti["per_strategy"].items():
            f.write(f"| {name} | {r['win_rate']:.2f} | {r['draw_rate']:.2f} | "
                    f"{r['kills_rule']:.2f} | {r['kills_opp']:.2f} |\n")

        f.write("\n## Cell 1: π_rule vs π_rule (mirror)\n\n")
        f.write(f"- n_episodes × n_envs = {n_episodes} × {n_envs} = {n_episodes * n_envs}\n")
        f.write(f"- kills_t0 mean: {mirror_metrics['kills_t0'].mean():.3f}\n")
        f.write(f"- kills_t1 mean: {mirror_metrics['kills_t1'].mean():.3f}\n")
        f.write(f"- margin (t0−t1): {mirror_margin_mean:+.3f} "
                f"(95% CI [{mirror_ci_lo:+.3f}, {mirror_ci_hi:+.3f}])\n")
        f.write(f"- winner dist: t0={float((mirror_metrics['winner']==0).mean()):.2f}, "
                f"t1={float((mirror_metrics['winner']==1).mean()):.2f}, "
                f"draw={float((mirror_metrics['winner']==-1).mean()):.2f}\n")

        f.write("\n## BC pretrain (AlphaStar SL → RL paradigm)\n\n")
        if bc_pretrain_samples > 0 and bc_history:
            f.write(f"- samples collected: {bc_pretrain_samples}\n")
            f.write(f"- epochs: {bc_pretrain_epochs}, batch_size: {bc_pretrain_batch_size}, lr: {bc_pretrain_lr}\n")
            f.write(f"- final train_loss: {bc_history[-1]['train_loss']:+.3f}\n")
            f.write(f"- final val_loss:   {bc_history[-1]['val_loss']:+.3f}\n")
            f.write(f"- checkpoint: `{bc_save_path}`\n\n")
            f.write("| epoch | train_loss | val_loss |\n")
            f.write("|---|---|---|\n")
            for row in bc_history:
                f.write(f"| {row['epoch']+1} | {row['train_loss']:+.3f} | {row['val_loss']:+.3f} |\n")
        else:
            f.write("(skipped — set --bc-pretrain-samples > 0 to enable)\n")

        f.write("\n## BR training\n\n")
        f.write(f"- iters: {br_iters}, horizon: {horizon}, n_envs: {n_envs}\n")
        f.write(f"- lr_actor={br_lr_actor}, lr_critic={br_lr_critic}, entropy_coef={br_entropy_coef}\n")
        f.write(f"- final reward_mean: {final_metrics.get('reward_mean', 'NA')}\n")
        f.write(f"- final adv_std: {final_metrics.get('adv_std', 'NA')}\n")
        f.write(f"- final entropy: {final_metrics.get('entropy', 'NA')}\n")
        f.write(f"- final approx_kl: {final_metrics.get('approx_kl', 'NA')}\n")
        f.write(f"- checkpoint: `{br_save_path}`\n")
        f.write(f"- training log: `g0_br_training_log.csv`\n")

        f.write("\n## Cell 2: π_rule vs BR(π_rule)\n\n")
        f.write(f"- n_episodes × n_envs = {n_episodes} × {n_envs} = {n_episodes * n_envs}\n")
        f.write(f"- kills_rule mean: {br_metrics['kills_t0'].mean():.3f}\n")
        f.write(f"- kills_BR mean: {br_metrics['kills_t1'].mean():.3f}\n")
        f.write(f"- margin (rule POV): {br_margin_mean:+.3f} "
                f"(95% CI [{br_ci_lo:+.3f}, {br_ci_hi:+.3f}])\n")
        f.write(f"- winner dist: rule={float((br_metrics['winner']==0).mean()):.2f}, "
                f"BR={float((br_metrics['winner']==1).mean()):.2f}, "
                f"draw={float((br_metrics['winner']==-1).mean()):.2f}\n")

        f.write("\n## Exploitability\n\n")
        f.write("exploit_gap = mean(mirror_margin) − mean(br_margin)\n\n")
        f.write(f"- **exploit_gap = {gap_mean:+.3f}** (95% CI [{gap_ci_lo:+.3f}, {gap_ci_hi:+.3f}])\n")
        f.write(f"- BR win rate vs rule: {br_win_rate:.2f}\n")
        f.write(f"- BR healthy: {br_healthy}\n\n")

        f.write("## Verdict\n\n")
        f.write("| check | threshold | actual | pass |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| exploit_gap ≥ 0.5 | 0.500 | {gap_mean:+.3f} | {'✅' if gap_mean >= 0.5 else '❌'} |\n")
        f.write(f"| CI excludes 0 | > 0 | {gap_ci_lo:+.3f} | {'✅' if ci_pass else '❌'} |\n")
        f.write(f"| BR win rate ≥ 0.55 | 0.55 | {br_win_rate:.2f} | {'✅' if direction_pass else '❌'} |\n")
        f.write(f"| BR healthy | adv_std∈[0.1,100], no NaN | {final_metrics.get('adv_std', 'NA')} | {'✅' if br_healthy else '❌'} |\n\n")

        if g0_pass:
            f.write("✅ **G0 PASS** — StrongRule is exploitable. Multi-function game is non-trivial.\n")
            f.write("**→ Recommend: proceed to WP2 self-play + league training.**\n")
        else:
            f.write("❌ **G0 FAIL** — StrongRule is NOT exploitable (or BR undertrained).\n")
            f.write("**Diagnose before any WP2 self-play burn:**\n")
            if not ci_pass:
                f.write("- CI includes 0 → exploit not statistically significant. ")
                f.write("Try more episodes or longer BR training.\n")
            if not direction_pass:
                f.write(f"- BR win rate {br_win_rate:.2f} < 0.55 → BR didn't learn to beat rule. ")
                f.write("Check BR training curves for collapse / undertraining.\n")
            if not br_healthy:
                f.write("- BR training unhealthy (NaN / adv_std out of range). ")
                f.write("Inspect training log.\n")
            if ci_pass and direction_pass and br_healthy and not gap_pass:
                f.write("- Gap direction OK but magnitude < 0.5. ")
                f.write("Likely genuine weak exploitability — discuss if strong enough for WP2 thesis.\n")
            f.write("\n**If diagnosis confirms rule genuinely not exploitable** → root A present.\n")
            f.write("**→ Recommend retreat to IET (C0+C1 IQ/CRLB baseline paper).**\n")

    print(f"\nReport: {report_path}", flush=True)
    print(f"CSVs: {mirror_csv}, {br_csv}, {br_log_csv}", flush=True)
    print(f"BR checkpoint: {br_save_path}", flush=True)
    return {
        "g0_pass": g0_pass,
        "exploit_gap": gap_mean,
        "ci_lo": gap_ci_lo,
        "ci_hi": gap_ci_hi,
        "br_win_rate": br_win_rate,
        "mirror_margin": mirror_margin_mean,
        "br_margin": br_margin_mean,
        "anti_strawman_verdict": anti["verdict"],
        "elapsed_min": elapsed,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--br-iters", type=int, default=200)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--br-lr-actor", type=float, default=3e-4)
    p.add_argument("--br-lr-critic", type=float, default=1e-3)
    p.add_argument("--br-entropy-coef", type=float, default=0.01)
    p.add_argument("--br-lr-decay-iters", type=int, default=0,
                   help=">0 enables cosine LR decay over this many iters")
    p.add_argument("--bc-pretrain-samples", type=int, default=0,
                   help=">0 enables BC pretrain; collects this many (obs, action) pairs from StrongRule")
    p.add_argument("--bc-pretrain-epochs", type=int, default=10)
    p.add_argument("--bc-pretrain-batch-size", type=int, default=256)
    p.add_argument("--bc-pretrain-lr", type=float, default=1e-3)
    p.add_argument("--out", type=str,
                   default="/home/ubuntu/CODE/FluxPhased-/experiments/twoteam/g0_gate_report.md")
    args = p.parse_args()
    out_dir = os.path.dirname(args.out)
    main(br_iters=args.br_iters, horizon=args.horizon, n_envs=args.n_envs,
         n_episodes=args.n_episodes, br_lr_actor=args.br_lr_actor,
         br_lr_critic=args.br_lr_critic, br_entropy_coef=args.br_entropy_coef,
         br_lr_decay_iters=args.br_lr_decay_iters,
         bc_pretrain_samples=args.bc_pretrain_samples,
         bc_pretrain_epochs=args.bc_pretrain_epochs,
         bc_pretrain_batch_size=args.bc_pretrain_batch_size,
         bc_pretrain_lr=args.bc_pretrain_lr,
         out_dir=out_dir)
