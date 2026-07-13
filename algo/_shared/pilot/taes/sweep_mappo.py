"""9-config MAPPO lr × entropy_coef sweep (AppInt Fix #3d).

Clip is FIXED at 0.2 (per plan: "clip fixed 0.2 saves ⅔ compute").
3 lr_actor × 3 entropy_coef = 9 configs, each 50 iters on L0/n4.
Logs to experiments/wp12_results/mappo_sweep.csv.

Selection criterion: max(kill_mean - 0.1·kl) over last 10 iters.
"""

from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import torch

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.taes.taes_env import TAESVecEnv
from algo._shared.pilot.taes.taes_actor_critic import TaesCommanderActorCritic
from algo._shared.pilot.taes.taes_ppo import TaesPPOTrainer


LR_CHOICES = [1e-4, 3e-4, 1e-3]
ENT_CHOICES = [0.005, 0.01, 0.02]
N_ITERS = 50
HORIZON = 300
N_ENVS = 8
N_TARGETS = 4
CLIP = 0.2  # fixed per plan


def run_one_config(lr_actor: float, entropy_coef: float, seed: int,
                   device: str = "cuda"):
    """Train 50 iters at L0/n4, return metrics over last 10 iters."""
    torch.manual_seed(seed)
    env = TAESVecEnv(n_envs=N_ENVS, n_targets=N_TARGETS, device=device,
                     seed=seed, episode_steps=HORIZON)
    ac = TaesCommanderActorCritic()
    trainer = TaesPPOTrainer(
        env=env, ac=ac,
        lr_actor=lr_actor, lr_critic=1e-3,
        clip=CLIP, entropy_coef=entropy_coef,
        n_epochs=4, minibatch_size=64,
        horizon=HORIZON, device=device,
        critic_mode="ctde",
    )
    last10 = {"kill": [], "ep_rew": [], "kl": [], "value_loss": []}
    for it in range(N_ITERS):
        rollout = trainer.collect_rollout()
        upd = trainer.update()
        if it >= N_ITERS - 10:
            last10["kill"].append(rollout["n_kills_total"])
            last10["ep_rew"].append(rollout["ep_rew_mean"])
            last10["kl"].append(upd["approx_kl"])
            last10["value_loss"].append(upd["value_loss"])
    # Selection score: kill_mean - 0.1 * kl_mean
    kill_m = sum(last10["kill"]) / 10
    kl_m = sum(last10["kl"]) / 10
    return {
        "kill_mean": kill_m,
        "ep_rew_mean": sum(last10["ep_rew"]) / 10,
        "kl_mean": kl_m,
        "value_loss_mean": sum(last10["value_loss"]) / 10,
        "score": kill_m - 0.1 * kl_m,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",
                        default="/home/ubuntu/CODE/FluxPhased-/experiments/wp12_results/mappo_sweep.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_actor", "entropy_coef", "clip", "kill_mean",
                    "ep_rew_mean", "kl_mean", "value_loss_mean", "score"])

    print(f"Sweeping 9 configs ({len(LR_CHOICES)}×{len(ENT_CHOICES)}), "
          f"{N_ITERS} iters each", flush=True)
    results = []
    for lr in LR_CHOICES:
        for ent in ENT_CHOICES:
            t0 = time.time()
            m = run_one_config(lr, ent, seed=args.seed, device=args.device)
            dt = time.time() - t0
            print(f"  lr={lr:.0e} ent={ent:.3f}: kill={m['kill_mean']:.2f} "
                  f"kl={m['kl_mean']:.4f} score={m['score']:.2f}  ({dt:.0f}s)",
                  flush=True)
            with open(args.out, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([lr, ent, CLIP,
                            f"{m['kill_mean']:.3f}",
                            f"{m['ep_rew_mean']:.3f}",
                            f"{m['kl_mean']:.5f}",
                            f"{m['value_loss_mean']:.4f}",
                            f"{m['score']:.3f}"])
            results.append(((lr, ent), m))

    # Pick best
    best = max(results, key=lambda x: x[1]["score"])
    print()
    print(f"=== Best config: lr_actor={best[0][0]:.0e}, "
          f"entropy_coef={best[0][1]:.3f}, score={best[1]['score']:.2f} ===")
    print(f"  kill_mean={best[1]['kill_mean']:.2f}, kl_mean={best[1]['kl_mean']:.4f}")


if __name__ == "__main__":
    main()
