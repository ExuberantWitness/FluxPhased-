"""WP-3 M4 production RL training orchestrator.

Wraps run_wp2_league.py with WP-3 production settings per spec §4:
  - ~5e7 env steps (1000 iter × 300 horizon × 256 envs = 7.65e7)
  - BlindClassical teacher (BC warmup) → PFSP league
  - Cosine entropy anneal 0.01 → 0.001
  - Checkpoints to checkpoints/blind/wp3_<ts>/ (per spec §4.3 NEVER /tmp)
  - Health monitor every 100 iters (entropy ≥ 0.3, policy_loss ≠ 0, pool EMA var)
  - priv[:,4] normalization guard every 100 iters
  - assert_no_godview on RL AC every 500 iters
  - Final report: training curves + health log + checkpoint path + smoke cross-play

Usage:
  python experiments/twoteam/wp3_train.py --iters 1000 --n-envs 256 --horizon 300

For quick smoke (verify pipeline works in 5 min):
  python experiments/twoteam/wp3_train.py --iters 5 --n-envs 8 --horizon 50 --bc-samples 1000
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import subprocess

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LEAGUE_ENTRY = os.path.join(ROOT, "algo/_shared/pilot/twoteam/run_wp2_league.py")


def main():
    p = argparse.ArgumentParser(description="WP-3 production RL training (BC + PFSP league)")
    # Production scale
    p.add_argument("--iters", type=int, default=1000,
                   help="N PPO iterations (1000 default → ~7.6e7 steps)")
    p.add_argument("--n-envs", type=int, default=256,
                   help="Parallel envs (256 default for RTX PRO 6000)")
    p.add_argument("--horizon", type=int, default=300,
                   help="Steps per rollout (300 default)")
    # League
    p.add_argument("--snapshot-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--population-cap", type=int, default=30)
    # BC warmup
    p.add_argument("--bc-samples", type=int, default=50000)
    p.add_argument("--bc-epochs", type=int, default=15)
    # PPO
    p.add_argument("--ppo-lr-actor", type=float, default=1e-4)
    p.add_argument("--ppo-lr-critic", type=float, default=1e-3)
    p.add_argument("--ppo-entropy-coef", type=float, default=0.01)
    p.add_argument("--ppo-entropy-coef-min", type=float, default=0.001)
    # WP-3 dense reward shaping
    p.add_argument("--shape-track-bonus", type=float, default=0.0,
                   help="Per-step bonus per radar tracked (dense signal; default off)")
    p.add_argument("--shape-exposure-penalty", type=float, default=0.0,
                   help="Per-step penalty × exposure (dense signal; default off)")
    # Eval
    p.add_argument("--n-eval-episodes", type=int, default=10)
    # I/O — MUST NOT be /tmp (spec §4.3)
    p.add_argument("--ckpt-dir", type=str, default="",
                   help="Default: checkpoints/blind/wp3_<timestamp>/")
    p.add_argument("--report", type=str, default="",
                   help="Default: experiments/twoteam/wp3_train_report.md")
    p.add_argument("--seed", type=int, default=42)
    # Teacher
    p.add_argument("--strong-rule-teacher", action="store_true",
                   help="Use legacy StrongRule BC teacher (ablation only; default = BlindClassical)")
    args = p.parse_args()

    # Generate timestamp-based paths if not provided
    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = args.ckpt_dir or f"checkpoints/blind/wp3_{ts}"
    report = args.report or "experiments/twoteam/wp3_train_report.md"
    log_path = os.path.join(ckpt_dir, "wp3_train_log.txt")

    # HARD GUARD: spec §4.3 — checkpoints MUST NOT be in /tmp
    if "/tmp" in ckpt_dir:
        raise ValueError(f"ckpt_dir must not be in /tmp (got {ckpt_dir!r})")
    if "/tmp" in report:
        raise ValueError(f"report must not be in /tmp (got {report!r})")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report) or ".", exist_ok=True)

    # Build league command
    cmd = [
        "/home/ubuntu/miniconda3/envs/fluxphased/bin/python", LEAGUE_ENTRY,
        "--n-iters", str(args.iters),
        "--n-envs", str(args.n_envs),
        "--horizon", str(args.horizon),
        "--snapshot-every", str(args.snapshot_every),
        "--log-every", str(args.log_every),
        "--population-cap", str(args.population_cap),
        "--bc-samples", str(args.bc_samples),
        "--bc-epochs", str(args.bc_epochs),
        "--ppo-lr-actor", str(args.ppo_lr_actor),
        "--ppo-lr-critic", str(args.ppo_lr_critic),
        "--ppo-entropy-coef", str(args.ppo_entropy_coef),
        "--ppo-entropy-coef-min", str(args.ppo_entropy_coef_min),
        "--shape-track-bonus", str(args.shape_track_bonus),
        "--shape-exposure-penalty", str(args.shape_exposure_penalty),
        "--n-eval-episodes", str(args.n_eval_episodes),
        "--ckpt-dir", ckpt_dir,
        "--out", report,
        "--seed", str(args.seed),
    ]
    if args.strong_rule_teacher:
        cmd.append("--strong-rule-teacher")
    else:
        cmd.append("--blind-teacher")

    # Total step count estimate
    total_steps = args.iters * args.horizon * args.n_envs
    print("=" * 70)
    print("WP-3 PRODUCTION RL TRAINING (spec §4)")
    print("=" * 70)
    print(f"  iters={args.iters}  horizon={args.horizon}  n_envs={args.n_envs}")
    print(f"  total env steps   = {total_steps:.2e}  (target ≥ 5e7 per spec §4)")
    print(f"  bc_samples        = {args.bc_samples}  (BlindClassical teacher)")
    print(f"  entropy anneal    = {args.ppo_entropy_coef} → {args.ppo_entropy_coef_min}")
    print(f"  ckpt_dir          = {ckpt_dir}  (persistent disk; spec §4.3)")
    print(f"  report            = {report}")
    print(f"  log               = {log_path}")
    print(f"  teacher           = {'StrongRule (ablation)' if args.strong_rule_teacher else 'BlindClassical (default)'}")
    print("=" * 70)
    print(f"  launching: {' '.join(cmd)}")
    print("=" * 70)
    sys.stdout.flush()

    # Stream output to log + stdout
    log_fp = open(log_path, "w", buffering=1)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return_code = None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fp.write(line)
        return_code = proc.wait()
    finally:
        log_fp.close()

    elapsed_min = (time.time() - t0) / 60.0
    print("=" * 70)
    print(f"WP-3 training {'COMPLETED' if return_code == 0 else 'FAILED'} "
          f"in {elapsed_min:.1f} min (return code {return_code})")
    print(f"  final ckpt  → {ckpt_dir}/iter_final.pt")
    print(f"  report      → {report}")
    print(f"  log         → {log_path}")
    if return_code == 0:
        print("\nNext: WP-4 cross-play vs BlindClassical:")
        print(f"  python experiments/twoteam/run_wp2_crossplay.py \\")
        print(f"      --rl-ckpt {ckpt_dir}/iter_final.pt \\")
        print(f"      --baseline BlindClassical --episodes 50")
    print("=" * 70)
    sys.exit(return_code or 0)


if __name__ == "__main__":
    main()
