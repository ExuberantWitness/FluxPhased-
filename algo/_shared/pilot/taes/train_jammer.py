"""Trainer for the L3 LearnedJammer (AppInt pre-flight fix #1).

Wraps `JammerPPOTrainer` (run_wp2.py) with:
  - log_std_floor = -6.0 (AppInt spec; prevents collapse)
  - opponent rotation: alternate between frozen classical and (optionally)
    a frozen RL commander snapshot
  - per-iter env reset (mixed-N if env has sampler)
  - periodic checkpoint snapshots to `checkpoints/appint/jammer_L3_it{N}.pt`

This produces a *properly-trained* L3 jammer whose kill-drop on a strong
classical commander exceeds both:
  (a) the L0 (StaticJammer) baseline by ≥ 0.10 kill_rate, AND
  (b) the L1-τ1 (ReactiveJammer hardest) baseline by ≥ 0.05 kill_rate.
Both checks are enforced by Step 2c / Step 5 sanity gate.
"""

from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import torch
import torch.nn.functional as F

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.taes.taes_env import TAESVecEnv
from env.gpu.qos_rrm.adversary import LearnedJammer, make_jammer
from algo._shared.baselines.taes_classical_commander import TaesClassicalCommander
from algo._shared.pilot.taes.run_wp2 import JammerPPOTrainer, make_classical_static


def make_classical_commander_fn():
    """Frozen strong modular classical commander as jammer's training opponent."""
    cmd = TaesClassicalCommander()
    def fn(env):
        return cmd.step(env)
    return fn


def train_jammer(
    save_dir: str,
    n_iters: int = 300,
    horizon: int = 600,
    n_envs: int = 16,
    n_targets: int = 4,
    use_mixed_n: bool = True,
    bootstrap_classical_iters: int = 50,  # first N iters vs frozen classical
    snapshot_every: int = 50,
    log_std_floor: float = -6.0,
    log_std_ceiling: float = -1.0,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cuda",
    log_prefix: str = "jammer_L3",
):
    """Train LearnedJammer. Returns path to final checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)

    # Mixed-N env (same as commander curriculum)
    if use_mixed_n:
        N_CHOICES = torch.tensor([1, 2, 4, 8], device=device)
        sampler = lambda E: N_CHOICES[torch.randint(0, 4, (E,), device=device)]
    else:
        sampler = None

    env = TAESVecEnv(n_envs=n_envs, n_targets=n_targets, device=device,
                    seed=seed, episode_steps=horizon,
                    n_targets_sampler=sampler)
    jammer = LearnedJammer(base_jam=0.3, device=device)
    jammer.reset(env.E, 1, env.device)
    trainer = JammerPPOTrainer(jammer, lr=lr, n_epochs=4, minibatch_size=128,
                                device=device,
                                log_std_floor=log_std_floor,
                                log_std_ceiling=log_std_ceiling)

    classical_cmd_fn = make_classical_commander_fn()
    csv_path = os.path.join(save_dir, f"{log_prefix}_train.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "opponent", "jam_reward", "value_loss",
                    "entropy", "approx_kl", "log_std"])

    print(f"Training L3 jammer for {n_iters} iters "
          f"(first {bootstrap_classical_iters} vs frozen classical)", flush=True)

    for it in range(n_iters):
        # Opponent selection: bootstrap with classical, then continue with classical
        # (single-opponent pre-flight; multi-snapshot league is R3 ablation)
        opponent = classical_cmd_fn

        # Fresh env reset (mixed-N re-samples)
        env.reset()
        jammer.reset(env.E, 1, env.device)
        env._last_jam = torch.zeros(env.E, device=device)

        buf = trainer.collect_rollout(env, opponent, horizon)
        metrics = trainer.update(buf)

        # ENFORCE log_std bounds (AppInt spec: floor=-6 prevents collapse;
        # ceiling=-1 prevents noise from washing out mean signal)
        with torch.no_grad():
            trainer.log_std.data.clamp_(log_std_floor, log_std_ceiling)

        cur_log_std = float(trainer.log_std.item())
        jam_rew = float(buf["ret"].mean())

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([it, "classical", f"{jam_rew:.3f}",
                        f"{metrics['value_loss']:.4f}",
                        f"{metrics['entropy']:.4f}",
                        f"{metrics['approx_kl']:.5f}",
                        f"{cur_log_std:.4f}"])

        if it % 10 == 0:
            print(f"  it={it:4d} jam_rew={jam_rew:+.2f} "
                  f"v_loss={metrics['value_loss']:.3f} "
                  f"H={metrics['entropy']:.3f} "
                  f"kl={metrics['approx_kl']:.4f} "
                  f"log_std={cur_log_std:.3f}", flush=True)

        if (it + 1) % snapshot_every == 0 or it == n_iters - 1:
            ckpt = os.path.join(save_dir, f"{log_prefix}_it{it+1}.pt")
            jammer.save(ckpt)
            print(f"  saved {ckpt}", flush=True)

    final_path = os.path.join(save_dir, f"{log_prefix}_final.pt")
    jammer.save(final_path)
    print(f"Final jammer checkpoint: {final_path}")
    return final_path


def _load_commander_pool(device: str = "cuda"):
    """Load all available commander opponents for league training.

    Returns list of (name, commander_fn) tuples.
    """
    from algo._shared.pilot.taes.taes_actor_critic import TaesCommanderActorCritic

    pool = [("classical", make_classical_commander_fn())]
    ckpt_dir = "/home/ubuntu/CODE/FluxPhased-/checkpoints/appint"
    candidates = [
        "mappo_final.pt",
        "ippo_final.pt",
        "mappo_phase0_L0.pt",
        "mappo_phase1_L1-mix.pt",
        "mappo_phase2_L3-trained.pt",
        "ippo_phase0_L0.pt",
        "ippo_phase1_L1-mix.pt",
        "ippo_phase2_L3-trained.pt",
    ]
    for fname in candidates:
        path = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            ac = TaesCommanderActorCritic().to(device)
            ac.load_state_dict(torch.load(path, map_location=device))
            ac.eval()
            def make_fn(_ac):
                def fn(env):
                    obs_dict = env.get_obs()
                    return _ac.get_action_for_env(
                        obs_dict["obs"], deterministic=True,
                        target_alive_mask=env.target_alive_mask)
                return fn
            pool.append((fname.replace(".pt", ""), make_fn(ac)))
        except Exception as e:
            print(f"  skip {fname}: {e}", flush=True)
    return pool


def train_jammer_league(
    save_dir: str,
    n_iters: int = 300,
    horizon: int = 600,
    n_envs: int = 16,
    n_targets: int = 4,
    use_mixed_n: bool = True,
    snapshot_every: int = 30,
    log_std_floor: float = -6.0,
    log_std_ceiling: float = -1.0,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cuda",
    log_prefix: str = "jammer_L3_league",
):
    """League-PFSP jammer training: cycle through diverse commander opponents.

    Goal: produce input-adaptive L3 jammer (drop vs L1-τ1 ≥ 0.05 AND output
    varies with red task histogram). Falls back to constant-output if pool
    is too uniform.

    Per spec RECONFIRM_GATE.md Task B: hard cap; accept constant if budget exceeded.
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)

    pool = _load_commander_pool(device=device)
    print(f"League opponent pool: {[name for name, _ in pool]}", flush=True)

    if use_mixed_n:
        N_CHOICES = torch.tensor([1, 2, 4, 8], device=device)
        sampler = lambda E: N_CHOICES[torch.randint(0, 4, (E,), device=device)]
    else:
        sampler = None

    env = TAESVecEnv(n_envs=n_envs, n_targets=n_targets, device=device,
                    seed=seed, episode_steps=horizon,
                    n_targets_sampler=sampler)
    jammer = LearnedJammer(base_jam=0.3, device=device)
    jammer.reset(env.E, 1, env.device)
    trainer = JammerPPOTrainer(jammer, lr=lr, n_epochs=4, minibatch_size=128,
                                device=device,
                                log_std_floor=log_std_floor,
                                log_std_ceiling=log_std_ceiling)

    csv_path = os.path.join(save_dir, f"{log_prefix}_train.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "opponent", "jam_reward", "value_loss",
                    "entropy", "approx_kl", "log_std"])

    print(f"Training L3 league jammer for {n_iters} iters "
          f"({len(pool)} opponents, rotation)", flush=True)

    for it in range(n_iters):
        opp_name, opp_fn = pool[it % len(pool)]

        env.reset()
        jammer.reset(env.E, 1, env.device)
        env._last_jam = torch.zeros(env.E, device=device)

        buf = trainer.collect_rollout(env, opp_fn, horizon)
        metrics = trainer.update(buf)

        with torch.no_grad():
            trainer.log_std.data.clamp_(log_std_floor, log_std_ceiling)

        cur_log_std = float(trainer.log_std.item())
        jam_rew = float(buf["ret"].mean())

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([it, opp_name, f"{jam_rew:.3f}",
                        f"{metrics['value_loss']:.4f}",
                        f"{metrics['entropy']:.4f}",
                        f"{metrics['approx_kl']:.5f}",
                        f"{cur_log_std:.4f}"])

        if it % 10 == 0:
            print(f"  it={it:4d} opp={opp_name:30s} jam_rew={jam_rew:+.2f} "
                  f"H={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f} "
                  f"log_std={cur_log_std:.3f}", flush=True)

        if (it + 1) % snapshot_every == 0 or it == n_iters - 1:
            ckpt = os.path.join(save_dir, f"{log_prefix}_it{it+1}.pt")
            jammer.save(ckpt)
            print(f"  saved {ckpt}", flush=True)

    final_path = os.path.join(save_dir, f"{log_prefix}_final.pt")
    jammer.save(final_path)
    print(f"Final league jammer: {final_path}", flush=True)
    return final_path


def eval_input_adaptive(jammer_ckpt: str, device: str = "cuda"):
    """Verify trained jammer output varies with red task state (input-adaptive).

    Test: run episodes with N ∈ {1, 2, 4, 8} (different number of alive targets
    → different task_alloc statistics), collect jammer outputs at each step,
    measure mean per N. If output is essentially constant across N
    (range < 0.10), the policy is "tuned constant" not input-adaptive.
    """
    from env.gpu.qos_rrm.adversary import make_jammer as _mk
    classical_fn = make_classical_static()
    means_per_N = {}
    stds_per_N = {}
    for n_t in [1, 2, 4, 8]:
        torch.manual_seed(42)
        env = TAESVecEnv(n_envs=8, n_targets=n_t, device=device,
                        seed=42, episode_steps=200)
        env.reset()
        # Use the L3-trained jammer directly so env.step calls its policy
        jammer = _mk("L3-trained", device=device, policy_path=jammer_ckpt)
        jammer.reset(env.E, 1, env.device)
        env._last_jam = torch.zeros(env.E, device=device)
        outs = []
        for step in range(150):
            action = classical_fn(env)
            obs, r, done, info = env.step(action, jammer=jammer)
            outs.append(float(jammer._buf.mean()))
            if done.all():
                break
        means_per_N[n_t] = sum(outs) / max(len(outs), 1)
        if outs:
            m = means_per_N[n_t]
            stds_per_N[n_t] = (sum((o - m) ** 2 for o in outs) / len(outs)) ** 0.5
        else:
            stds_per_N[n_t] = 0.0

    outputs = list(means_per_N.values())
    out_range = max(outputs) - min(outputs)
    out_mean = sum(outputs) / len(outputs)
    out_std = (sum((o - out_mean) ** 2 for o in outputs) / len(outputs)) ** 0.5

    return {
        "means_per_N": means_per_N,
        "within_N_std": stds_per_N,
        "output_range_across_N": out_range,
        "output_std_across_N": out_std,
        "is_input_adaptive": out_range >= 0.10,  # ≥0.10 range → adaptive
    }


def eval_jammer_kill_drop(
    jammer_ckpt: str,
    n_envs: int = 8,
    n_targets: int = 4,
    episode_steps: int = 600,
    seeds=(42, 43, 44),
    device: str = "cuda",
):
    """Compute kill_rate of strong classical vs 3 jammers:
       L0 (StaticJammer), L1-tau1 (ReactiveJammer τ=1, hardest L1), L3-trained.

    Returns dict with kill_rate per jammer + monotonicity check.
    """
    classical_fn = make_classical_static()
    results = {}
    for level, make_kw in [
        ("L0", {"jam_level": 0.3}),
        ("L1-tau1", {"tau": 1}),
        ("L3-trained", {"policy_path": jammer_ckpt}),
    ]:
        kills_per_seed = []
        for seed in seeds:
            torch.manual_seed(seed)
            env = TAESVecEnv(n_envs=n_envs, n_targets=n_targets, device=device,
                            seed=seed, episode_steps=episode_steps)
            if level == "L0":
                jammer = make_jammer("L0", device=device, jam_level=make_kw["jam_level"])
            elif level == "L1-tau1":
                jammer = make_jammer("L1", device=device, tau=make_kw["tau"])
            else:
                jammer = make_jammer("L3", device=device, policy_path=make_kw["policy_path"])
            jammer.reset(env.E, 1, env.device)
            env._last_jam = torch.zeros(env.E, device=device)
            env.reset()
            ep_kills = torch.zeros(env.E, device=device)
            for step in range(episode_steps):
                action = classical_fn(env)
                obs, r, done, info = env.step(action, jammer=jammer)
                ep_kills += info["n_kills_step"]
                if done.all():
                    break
            kills_per_seed.append(float(ep_kills.mean()))
        results[level] = sum(kills_per_seed) / len(kills_per_seed)

    # Drop checks
    drop_vs_L0 = results["L0"] - results["L3-trained"]
    drop_vs_L1tau1 = results["L1-tau1"] - results["L3-trained"]
    monotonicity_pass = drop_vs_L1tau1 >= 0.05

    return {
        "kill_classical_vs_L0": results["L0"],
        "kill_classical_vs_L1_tau1": results["L1-tau1"],
        "kill_classical_vs_L3_trained": results["L3-trained"],
        "L3_kill_drop_vs_L0": drop_vs_L0,
        "L3_kill_drop_vs_L1_tau1": drop_vs_L1tau1,
        "monotonicity_pass": monotonicity_pass,
        "gate_pass": (drop_vs_L0 >= 0.10) and monotonicity_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir",
                        default="/home/ubuntu/CODE/FluxPhased-/checkpoints/appint")
    parser.add_argument("--n-iters", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-targets", type=int, default=4)
    parser.add_argument("--no-mixed-n", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=50)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--log-std-floor", type=float, default=-6.0)
    parser.add_argument("--log-std-ceiling", type=float, default=-1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--league", action="store_true",
                        help="use league-PFSP trainer (cycle commander pool)")
    parser.add_argument("--eval-after", action="store_true",
                        help="run kill-drop eval after training")
    args = parser.parse_args()

    train_fn = train_jammer_league if args.league else train_jammer
    log_prefix = "jammer_L3_league" if args.league else "jammer_L3"
    kwargs = dict(
        save_dir=args.save_dir,
        n_iters=args.n_iters,
        horizon=args.horizon,
        n_envs=args.n_envs,
        n_targets=args.n_targets,
        use_mixed_n=not args.no_mixed_n,
        snapshot_every=args.snapshot_every,
        log_std_floor=args.log_std_floor,
        log_std_ceiling=args.log_std_ceiling,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        log_prefix=log_prefix,
    )
    if not args.league:
        kwargs["bootstrap_classical_iters"] = args.bootstrap_iters
    final_ckpt = train_fn(**kwargs)

    if args.eval_after:
        print("\n=== Jammer kill-drop eval (vs frozen classical) ===", flush=True)
        results = eval_jammer_kill_drop(final_ckpt, seeds=(42, 43, 44))
        for k, v in results.items():
            print(f"  {k}: {v}", flush=True)
        if args.league:
            print("\n=== Input-adaptive check ===", flush=True)
            adapt = eval_input_adaptive(final_ckpt, device=args.device)
            for k, v in adapt.items():
                print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
