"""WP2 league main loop: BC pretrain → PFSP population self-play.

Per plan snuggly-exploring-parrot.md (WP2 BC→League) + WP2_BC_LEAGUE_PLAN.md.

Pipeline:
  [A] BC pretrain (reuse bc_pretrain.py) — bootstrap AC from StrongRule.
  [B] Initialize opponent pool: StrongRule + 7 Extreme + 3 candidate-exploit + BC snapshot.
  [C] League main loop (N_iters):
      - PFSP-sample one opponent from pool (weighted by f_hard(1-wr)^p)
      - Swap it into br_trainer.frozen_opponent (commander-like adapter)
      - Run 1 PPO iteration (collect_rollout + GAE + update)
      - Quick eval (n_eval_episodes) → win rate → pool.update_win_rate(EMA)
      - Every snapshot_every iters: save AC, add self-snapshot to pool
  [D] Final snapshot + pool metadata saved for cross-play eval.

Bug guards (memory: twoteam_multifunction_pivot):
  - priv[:, 4] normalized trace_P (α_eff bug) — assert every 100 iters
  - log_std_floor = -6 (gradient explosion prevention)
  - PFSP EMA with first-observation-replaces (not blended toward 0.5)
  - checkpoint_dir = checkpoints/twoteam/wp2_league/ (NEVER /tmp)
  - NaN/adv_std/entropy monitoring each iter

Output:
  checkpoints/twoteam/wp2_league/iter{NNN}.pt  (BC + every snapshot)
  checkpoints/twoteam/wp2_league/iter_final.pt
  experiments/twoteam/wp2_league.log
  experiments/twoteam/wp2_league_report.md
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
import math
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.br_trainer import TwoTeamBRTrainer
from algo._shared.pilot.twoteam.bc_pretrain import TwoTeamBCPretrainer
from algo._shared.pilot.twoteam.extreme_commanders import STRATEGIES, combine_team_actions
from algo._shared.pilot.twoteam.opponent_pool import (
    TwoTeamOpponentPool, PolicyRecord, build_opponent_action_fn,
)


# ----------------------------------------------------------------------
# AC commander adapter (wraps an AC as a commander-like frozen opponent)
# ----------------------------------------------------------------------

class ACCommander:
    """Wrap a TwoTeamCommanderActorCritic as a commander-like opponent.

    Required API: `get_action(env, team) -> action_dict` (matches ExtremeCommander
    and TwoTeamStrongRuleCommander, which is what br_trainer.frozen_opponent expects).

    WP-3 M0/M1: passes `detect_list` slice to AC's get_action_for_env.
    """

    def __init__(self, ac: TwoTeamCommanderActorCritic, deterministic: bool = True):
        self.ac = ac
        self.deterministic = bool(deterministic)

    @torch.no_grad()
    def get_action(self, env, team: int) -> Dict[str, torch.Tensor]:
        obs_dict = env.get_obs()
        detect_team = env.get_detect_list()[:, team]   # WP-3 M0/M1
        action, _ = self.ac.get_action_for_env(
            obs_dict["obs"][:, team], detect_team, obs_dict["privileged"][:, team],
            deterministic=self.deterministic,
        )
        return action


def load_ac_commander(checkpoint_path: str, device: str = "cuda",
                      deterministic: bool = True) -> ACCommander:
    """Load an AC from checkpoint and wrap as ACCommander."""
    ac = TwoTeamCommanderActorCritic().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("ac_state", ckpt)
    ac.load_state_dict(state)
    ac.eval()
    return ACCommander(ac, deterministic=deterministic)


# ----------------------------------------------------------------------
# Pool initialization
# ----------------------------------------------------------------------

def make_factory_commander(name: str):
    """Return a factory() that creates a fresh commander instance by name."""
    if name == "strong_rule":
        return lambda: TwoTeamStrongRuleCommander()
    if name == "blind_classical":
        return lambda: BlindClassicalCommander()
    if name in STRATEGIES:
        # STRATEGIES holds singletons; for factory semantics, return the same instance
        # (extreme commanders are stateless — safe to share)
        return (lambda _c=STRATEGIES[name]: lambda: _c)()
    raise KeyError(f"Unknown commander name: {name!r}")


def initialize_pool(
    bc_ckpt_path: str,
    bc_iter: int = 0,
    population_cap: int = 30,
    seed: int = 42,
) -> TwoTeamOpponentPool:
    """Build the initial opponent pool with rule + extreme + exploit + BC snapshot."""
    pool = TwoTeamOpponentPool(
        population_cap=population_cap,
        pfsp_hardness_p=1.0,
        ema_alpha=0.1,
        rng_seed=seed,
    )

    # 1. StrongRule (the eventual champion target)
    pool.add(PolicyRecord(
        name="strong_rule", kind="rule",
        factory=make_factory_commander("strong_rule"),
    ))

    # 1b. WP-3 M3: BlindClassical — spec §0.3④ "competent blind classical" baseline.
    # This is the primary BC teacher and a pool opponent (same blind API as the AC).
    pool.add(PolicyRecord(
        name="blind_classical", kind="rule",
        factory=make_factory_commander("blind_classical"),
    ))

    # 2. 7 extreme strategies
    extreme_names = [
        "pure_track", "pure_jam", "pure_comm", "pure_detect",
        "balanced", "balanced_jam_heavy", "track_agile",
    ]
    for nm in extreme_names:
        pool.add(PolicyRecord(
            name=f"extreme/{nm}", kind="extreme",
            factory=make_factory_commander(nm),
        ))

    # 3. 3 candidate exploits
    exploit_names = ["jam_spread", "hard_jam_focus", "track_heavy_agile"]
    for nm in exploit_names:
        pool.add(PolicyRecord(
            name=f"exploit/{nm}", kind="script",
            factory=make_factory_commander(nm),
        ))

    # 4. BC starting snapshot
    pool.add(PolicyRecord(
        name=f"self/iter{bc_iter:03d}_bc", kind="checkpoint",
        checkpoint_path=bc_ckpt_path,
        is_self_snapshot=True, created_at_iter=bc_iter,
    ))

    return pool


# ----------------------------------------------------------------------
# Quick eval (win rate of current AC vs one opponent, n_episodes)
# ----------------------------------------------------------------------

@torch.no_grad()
def quick_eval_winrate(
    env: TwoTeamVecEnv,
    learning_ac_commander: ACCommander,
    opp_action_fn: Callable,
    n_episodes: int,
    horizon: int,
    learning_team: int = 0,
    seed_base: int = 9000,
) -> float:
    """Quick win-rate eval: current AC vs one opponent. Draw counts as 0.5."""
    E = env.E
    wins, total = 0, 0
    other_team = 1 - learning_team
    for ep in range(n_episodes):
        env.seed = seed_base + ep
        env._reset_count = ep
        env.reset()
        for step in range(horizon):
            if learning_team == 0:
                a0 = learning_ac_commander.get_action(env, 0)
                a1 = opp_action_fn(env, 1)
                action = combine_team_actions(env, a0, a1)
            else:
                a0 = opp_action_fn(env, 0)
                a1 = learning_ac_commander.get_action(env, 1)
                action = combine_team_actions(env, a0, a1)
            obs, r, done, info = env.step(action)
            if done.all():
                break
        kills_lt = info["team_kills"][:, learning_team].cpu().numpy()
        kills_opp = info["team_kills"][:, other_team].cpu().numpy()
        alive_lt = info["team_alive"][:, learning_team].cpu().numpy().astype(int)
        alive_opp = info["team_alive"][:, other_team].cpu().numpy().astype(int)
        for i in range(E):
            total += 1
            if kills_lt[i] > kills_opp[i]:
                wins += 1
            elif kills_lt[i] == kills_opp[i]:
                if alive_lt[i] > alive_opp[i]:
                    wins += 1
                elif alive_lt[i] == alive_opp[i]:
                    wins += 0.5
            # else: loss → +0
    return wins / max(1, total)


# ----------------------------------------------------------------------
# α_eff bug guard
# ----------------------------------------------------------------------

def assert_priv_normalized(env: TwoTeamVecEnv, tag: str = "") -> None:
    """Assert priv[:, 4] is normalized trace_P (not raw ≈ 200)."""
    obs_dict = env.get_obs()
    priv = obs_dict["privileged"]
    p4 = priv[..., 4]
    p4_max = float(p4.max().item())
    p4_min = float(p4.min().item())
    assert p4_max < 100.0, (
        f"[{tag}] priv[:,4] max={p4_max:.1f} — looks like raw trace_P (α_eff bug). "
        f"Expected normalized values < ~10."
    )
    assert p4_min >= 0.0, f"[{tag}] priv[:,4] min={p4_min:.3f} < 0 (should be ≥ 0)"


# ----------------------------------------------------------------------
# Main league loop
# ----------------------------------------------------------------------

def run_league(args):
    t_start = time.time()
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[gpu] Using device: cuda")
    print(f"[gpu] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[gpu] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 70)
    print("WP2 LEAGUE — BC pretrain → PFSP population self-play")
    print("=" * 70)
    print(f"  n_iters         = {args.n_iters}")
    print(f"  snapshot_every  = {args.snapshot_every}")
    print(f"  pfsp_hardness_p = {args.pfsp_hardness}")
    print(f"  bc_samples      = {args.bc_samples}")
    print(f"  bc_epochs       = {args.bc_epochs}")
    print(f"  ckpt_dir        = {ckpt_dir}  (persistent disk)")
    print(f"  log_std_floor   = {args.log_std_floor}")

    # ------------------------------------------------------------------
    # Step A: env + BC pretrain
    # ------------------------------------------------------------------
    print("\n=== Step A: env + BC pretrain ===")
    env = TwoTeamVecEnv(
        n_envs=args.n_envs, device="cuda", episode_steps=args.horizon,
        geometry=RANDOM_GEOMETRY, seed=42,
    )
    # α_eff guard before any training
    env.reset()
    assert_priv_normalized(env, tag="init")

    br_ac = TwoTeamCommanderActorCritic().to("cuda")
    # NOTE: this AC uses Dirichlet/Categorical/Bernoulli/Beta heads with
    # built-in min_concentration floors (softplus + min). The `log_std_floor`
    # concept from Gaussian heads doesn't apply — `args.log_std_floor` is
    # accepted for CLI compatibility but unused here.

    # WP-3 M2: default teacher = BlindClassical (spec §0.3④ baseline; BC must
    # not leak god-view, so it must match the AC's blind API).
    if args.blind_teacher:
        rule = BlindClassicalCommander()
        print(f"  BC teacher: BlindClassicalCommander (blind, spec §0.3④ baseline)")
    else:
        rule = TwoTeamStrongRuleCommander()
        print(f"  BC teacher: TwoTeamStrongRuleCommander (legacy god-view — for ablation only)")

    if args.bc_samples > 0:
        print(f"\n  BC: collect {args.bc_samples} samples...")
        bc_trainer = TwoTeamBCPretrainer(br_ac, lr=args.bc_lr, batch_size=args.bc_batch_size)

        bc_env = TwoTeamVecEnv(
            n_envs=args.n_envs, device="cuda", episode_steps=args.horizon,
            geometry=RANDOM_GEOMETRY, seed=100,
        )
        samples = bc_trainer.collect_samples(
            bc_env, rule, n_samples=args.bc_samples, episode_steps=args.horizon,
        )
        print(f"  BC: collected {samples['obs'].shape[0]} samples")
        bc_history = bc_trainer.train(samples, n_epochs=args.bc_epochs, log_every=1)

        bc_ckpt = os.path.join(ckpt_dir, "iter000_bc.pt")
        bc_trainer.save(bc_ckpt, bc_history)
        print(f"  BC: saved → {bc_ckpt}")
    else:
        bc_ckpt = os.path.join(ckpt_dir, "iter000_noinit.pt")
        torch.save({"ac_state": br_ac.state_dict(), "iter": 0}, bc_ckpt)
        print(f"  BC skipped (--bc-samples=0); random-init saved → {bc_ckpt}")

    # ------------------------------------------------------------------
    # Step B: pool init
    # ------------------------------------------------------------------
    print("\n=== Step B: initialize opponent pool ===")
    pool = initialize_pool(
        bc_ckpt_path=bc_ckpt, bc_iter=0,
        population_cap=args.population_cap, seed=args.seed,
    )
    print(f"  pool size: {pool.num_records()} (rule×2 + 7 extreme + 3 exploit + 1 BC)")
    assert pool.num_records() == 13, f"expected 13 seed records, got {pool.num_records()}"

    # ------------------------------------------------------------------
    # Step C: trainer setup (we'll swap frozen_opponent each iter)
    # ------------------------------------------------------------------
    print("\n=== Step C: PPO trainer setup ===")
    # Initial frozen_opponent = rule (will be replaced each iter by PFSP sample)
    trainer = TwoTeamBRTrainer(
        br_ac, frozen_opponent=rule,
        lr_actor=args.ppo_lr_actor, lr_critic=args.ppo_lr_critic,
        entropy_coef=args.ppo_entropy_coef,
        gamma=0.99, gae_lambda=0.95, clip=0.2,
        n_epochs=4, minibatch_size=64,
        target_kl=0.03, max_grad_norm=0.5,
        alpha_eff_alpha_max=0.5, alpha_eff_beta=2.0,
        reward_scale=0.1, value_huber_delta=1.0,
        lr_decay_iters=0,
        # WP-3 M1: cosine entropy anneal — keeps exploration alive early, decays late
        entropy_coef_min=args.ppo_entropy_coef_min,
        entropy_decay_iters=args.n_iters,
        # WP-3 dense reward shaping — mitigates zero-sum mirror symmetry (weak gradient)
        shape_track_bonus=args.shape_track_bonus,
        shape_exposure_penalty=args.shape_exposure_penalty,
        device="cuda",
    )
    print(f"  lr_actor={args.ppo_lr_actor} lr_critic={args.ppo_lr_critic} "
          f"entropy_coef={args.ppo_entropy_coef} → {args.ppo_entropy_coef_min} (cosine)")

    # ------------------------------------------------------------------
    # Step D: league main loop
    # ------------------------------------------------------------------
    print("\n=== Step D: league main loop ===")
    history: List[Dict[str, Any]] = []
    eval_env = TwoTeamVecEnv(
        n_envs=min(args.n_envs, 4), device="cuda", episode_steps=args.horizon,
        geometry=RANDOM_GEOMETRY, seed=200,
    )
    learning_ac_commander = ACCommander(br_ac, deterministic=True)

    t_loop_start = time.time()
    last_opp_name = None

    for it in range(args.n_iters):
        # 1) PFSP sample
        opp_rec = pool.sample_pfsp(exclude=last_opp_name)
        if opp_rec is None:
            print(f"  [it={it}] pool empty, aborting")
            break
        last_opp_name = opp_rec.name

        # 2) Build action_fn for this opponent
        opp_action_fn, opp_cleanup = build_opponent_action_fn(opp_rec, device="cuda")

        # 3) Wrap as a commander-like object for br_trainer.frozen_opponent
        #    For non-checkpoint: factory() returns the commander directly
        #    For checkpoint: we wrap the loaded AC in ACCommander
        if opp_rec.kind == "checkpoint":
            frozen_cmd = load_ac_commander(opp_rec.checkpoint_path, device="cuda",
                                            deterministic=True)
        else:
            frozen_cmd = opp_rec.factory()
        trainer.frozen_opponent = frozen_cmd

        # 4) One PPO iteration: collect_rollout + GAE + update
        try:
            buf = trainer.collect_rollout(env, args.horizon, learning_team=0)
            trainer._compute_gae(buf)
            metrics = trainer.update(buf, iter_idx=it, n_iters=args.n_iters)
        except Exception as e:
            print(f"  [it={it}] PPO step FAILED: {e}", flush=True)
            opp_cleanup()
            break

        # 5) Health monitor
        r_mean = buf.reward.mean().item()
        adv_std = buf.advantage.std().item()
        entropy = metrics["entropy"]
        kl = metrics["approx_kl"]
        elapsed = (time.time() - t_loop_start) / 60.0

        # NaN guard
        ac_has_nan = any(torch.isnan(p).any().item() for p in br_ac.parameters())
        if ac_has_nan:
            print(f"  [it={it}] ❌ NaN in AC params, aborting", flush=True)
            opp_cleanup()
            break

        # 6) Quick eval (win rate vs THIS iter's opponent)
        wr = quick_eval_winrate(
            eval_env, learning_ac_commander, opp_action_fn,
            n_episodes=args.n_eval_episodes, horizon=args.horizon,
            learning_team=0, seed_base=9000 + it * 17,
        )
        pool.update_win_rate(opp_rec.name, wr >= 0.5)

        # Log
        history.append({
            "iter": it, "opp": opp_rec.name, "reward_mean": r_mean,
            "policy_loss": metrics["policy_loss"], "value_loss": metrics["value_loss"],
            "entropy": entropy, "approx_kl": kl,
            "adv_std": adv_std, "clip_frac": metrics["clip_frac"],
            "early_stop": metrics["early_stop"],
            "win_rate_vs_opp": wr,
            "pool_ema_var": pool.ema_variance(),
            "elapsed_min": elapsed,
        })

        if it % args.log_every == 0 or it == args.n_iters - 1:
            ema_var = pool.ema_variance()
            print(f"  [it={it:4d}/{args.n_iters}] opp={opp_rec.name:30s} "
                  f"r={r_mean:+.3f} v_loss={metrics['value_loss']:.3f} "
                  f"ent={entropy:+.3f} kl={kl:.4f} adv_std={adv_std:.2f} "
                  f"clip={metrics['clip_frac']:.2f} es={int(metrics['early_stop'])} "
                  f"wr_vs_opp={wr:.2f} ema_var={ema_var:.3f} "
                  f"t={elapsed:.1f}min",
                  flush=True)

        # WP-3 M3: health monitor every 100 iters (per PID 1296303 incident memory).
        # Catches: (1) entropy collapse → 0 (deterministic policy, PPO stuck),
        #          (2) policy_loss → 0 (PID 1296303 main symptom), (3) pool imbalance.
        if it > 0 and (it + 1) % 100 == 0:
            ent_val = float(entropy)
            pi_loss_val = abs(float(metrics["policy_loss"]))
            ema_var_val = float(pool.ema_variance())
            warnings_emitted = []
            if ent_val < 0.3:
                warnings_emitted.append(
                    f"entropy={ent_val:.3f} < 0.3 floor (policy near-deterministic)")
            if pi_loss_val < 1e-4:
                warnings_emitted.append(
                    f"|policy_loss|={pi_loss_val:.2e} → 0 (PID 1296303 signature, "
                    f"PPO may be stuck — check KL early-stop frequency)")
            if ema_var_val < 0.05:
                warnings_emitted.append(
                    f"pool ema_var={ema_var_val:.3f} < 0.05 (PFSP collapsed to one opponent)")
            if warnings_emitted:
                print(f"  [it={it}] HEALTH WARN:", " | ".join(warnings_emitted),
                      flush=True)

        # Periodic α_eff guard
        if it > 0 and it % 100 == 0:
            try:
                env.reset()
                assert_priv_normalized(env, tag=f"iter-{it}")
            except AssertionError as e:
                print(f"  [it={it}] ❌ {e}", flush=True)
                break

        # Periodic snapshot
        if (it + 1) % args.snapshot_every == 0 or it == args.n_iters - 1:
            snap_path = os.path.join(ckpt_dir, f"iter{it + 1:03d}.pt")
            torch.save({
                "ac_state": br_ac.state_dict(),
                "iter": it + 1,
                "metrics_recent": history[-5:],
            }, snap_path)
            pool.add(PolicyRecord(
                name=f"self/iter{it + 1:03d}", kind="checkpoint",
                checkpoint_path=snap_path,
                is_self_snapshot=True, created_at_iter=it + 1,
            ))
            if (it + 1) % (args.snapshot_every * 5) == 0 or it == args.n_iters - 1:
                print(f"  [it={it}] snapshot → {snap_path}  (pool size: {pool.num_records()})",
                      flush=True)

        # Cleanup opponent resources
        opp_cleanup()
        if opp_rec.kind == "checkpoint":
            del frozen_cmd
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step E: final snapshot + reports
    # ------------------------------------------------------------------
    print("\n=== Step E: finalize ===")
    final_path = os.path.join(ckpt_dir, "iter_final.pt")
    torch.save({
        "ac_state": br_ac.state_dict(),
        "iter": args.n_iters,
        "metrics_recent": history[-10:],
        "pool_summary": pool.summary(),
    }, final_path)
    print(f"  final snapshot → {final_path}")

    pool_path = os.path.join(ckpt_dir, "pool_metadata.json")
    pool.save_metadata(pool_path)
    print(f"  pool metadata  → {pool_path}")

    # Write markdown report
    write_report(args, history, pool, t_start)


def write_report(args, history: List[Dict], pool: TwoTeamOpponentPool, t_start: float):
    """Write league training report."""
    out_path = args.out
    lines = []
    lines.append("# WP2 League Training Report\n")
    lines.append(f"**Started**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start))}\n")

    lines.append("## Setup\n")
    lines.append(f"- n_iters: {args.n_iters}")
    lines.append(f"- snapshot_every: {args.snapshot_every}")
    lines.append(f"- pfsp_hardness: {args.pfsp_hardness}")
    lines.append(f"- bc_samples: {args.bc_samples}, bc_epochs: {args.bc_epochs}")
    lines.append(f"- ppo_lr_actor: {args.ppo_lr_actor}, entropy_coef: {args.ppo_entropy_coef}")
    lines.append(f"- horizon: {args.horizon}, n_envs: {args.n_envs}")
    lines.append(f"- ckpt_dir: `{args.ckpt_dir}`\n")

    if not history:
        lines.append("## ⚠️ No training history (loop aborted at iter 0)\n")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        return

    lines.append("## Training health\n")
    final = history[-1]
    initial = history[0]
    lines.append(f"- final reward: {final['reward_mean']:+.3f}  (initial: {initial['reward_mean']:+.3f})")
    lines.append(f"- final entropy: {final['entropy']:+.3f}")
    lines.append(f"- final adv_std: {final['adv_std']:.3f}  (healthy range [0.1, 100])")
    lines.append(f"- final pool EMA variance: {final['pool_ema_var']:.3f}  (low → PFSP may be stuck)")
    lines.append(f"- total elapsed: {final['elapsed_min']:.1f} min\n")

    lines.append("## Per-iter log (every {} iters)\n".format(args.log_every))
    lines.append("| iter | opp | reward | adv_std | entropy | kl | wr_vs_opp | ema_var | t(min) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for h in history[::args.log_every]:
        lines.append(f"| {h['iter']} | `{h['opp']}` | {h['reward_mean']:+.3f} | "
                     f"{h['adv_std']:.2f} | {h['entropy']:+.3f} | {h['approx_kl']:.4f} | "
                     f"{h['win_rate_vs_opp']:.2f} | {h['pool_ema_var']:.3f} | "
                     f"{h['elapsed_min']:.1f} |")

    lines.append("\n## Pool final state\n")
    lines.append("| name | kind | is_self | win_rate_vs_current | games |")
    lines.append("|---|---|---|---|---|")
    for r in pool.all_records():
        wr = r.win_rate_vs_current
        wr_str = f"{wr:.3f}" if wr is not None else "—"
        lines.append(f"| `{r.name}` | {r.kind} | {r.is_self_snapshot} | "
                     f"{wr_str} | {r.games_played_vs_current} |")

    lines.append("\n## Notes\n")
    lines.append("- Pool win_rate_vs_current = EMA of how often the *current* training AC "
                 "beats this opponent (draw counts as 0.5). Low values → hard opponents → "
                 "PFSP samples them more.")
    lines.append("- self/iterNNN snapshots are added to the pool as the league evolves, "
                 "enabling population-level diversity.")
    lines.append("- Pool EMA variance near 0 → PFSP degenerating toward uniform; investigate.")
    lines.append("- Run `run_wp2_crossplay.py` next for cross-play Elo + non-transitivity detection.\n")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  report → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-iters", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--pfsp-hardness", type=float, default=1.0)
    p.add_argument("--population-cap", type=int, default=30)
    # BC
    p.add_argument("--bc-samples", type=int, default=50000)
    p.add_argument("--bc-epochs", type=int, default=15)
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    # PPO
    p.add_argument("--ppo-lr-actor", type=float, default=1e-4)
    p.add_argument("--ppo-lr-critic", type=float, default=1e-3)
    p.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    p.add_argument("--ppo-entropy-coef-min", type=float, default=0.001,
                   help="WP-3 M1: cosine anneal floor for entropy_coef")
    p.add_argument("--shape-track-bonus", type=float, default=0.0,
                   help="WP-3 dense reward: per-step bonus per radar tracked (tau_track)")
    p.add_argument("--shape-exposure-penalty", type=float, default=0.0,
                   help="WP-3 dense reward: per-step penalty × exposure")
    p.add_argument("--log-std-floor", type=float, default=-6.0)
    # WP-3 M3: BC teacher toggle (default blind, per spec §0.3④)
    p.add_argument("--blind-teacher", action="store_true", default=True,
                   help="Use BlindClassicalCommander as BC teacher (default)")
    p.add_argument("--strong-rule-teacher", dest="blind_teacher", action="store_false",
                   help="Use legacy StrongRule (god-view) as BC teacher — ablation only")
    # Env
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    # I/O
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/twoteam/wp2_league")
    p.add_argument("--out", type=str, default="experiments/twoteam/wp2_league_report.md")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Hard guard: ckpt_dir must NEVER be /tmp
    if "/tmp" in args.ckpt_dir:
        raise ValueError(f"ckpt_dir must not be in /tmp (got {args.ckpt_dir!r})")

    run_league(args)


if __name__ == "__main__":
    main()
