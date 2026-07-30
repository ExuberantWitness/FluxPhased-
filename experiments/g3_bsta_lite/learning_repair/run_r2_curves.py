"""R2 训练曲线生成 — 仅跑最佳候选 (lr=3e-5, target_kl=0.01) 并落盘每 iter metrics."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

from env.gpu.g3_bsta_lite import EnvConfig, PROFILE_MDP_SANITY, N_ACTIONS
from experiments.g3_bsta_lite.learning_repair.trainer import (
    R2PPOConfig, R2PPOTrainer, evaluate_actor,
)
from experiments.g3_bsta_lite.learning_repair.run_r2_gate3 import (
    MANIFEST_DIR, OUT_DIR, load_seeds, env_cfg_for, VAL_EVERY,
    VAL_REPS_INTERMEDIATE, N_ITERATIONS,
)


def main():
    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")

    out_dir = OUT_DIR / "curves_lr3e-05_kl0.01"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = R2PPOConfig(
        profile=PROFILE_MDP_SANITY,
        iterations=N_ITERATIONS,
        n_envs=16, horizon=64,
        actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.01,
        seed=20260729, train_seed=20260729,
        device=device,
    )
    env_cfg = env_cfg_for(PROFILE_MDP_SANITY, device)
    trainer = R2PPOTrainer(
        cfg=cfg, env_cfg=env_cfg,
        train_seeds=train_seeds,
        manifest_path=MANIFEST_DIR / "ppo_train.json",
        out_dir=out_dir,
    )
    trainer.save_pristine_init()

    rows: list[dict] = []
    val_rows: list[dict] = []
    t0 = time.time()
    for it in range(N_ITERATIONS):
        m = trainer.train_iteration()
        rows.append(m)
        if (it + 1) % VAL_EVERY == 0 or it == N_ITERATIONS - 1:
            ve = evaluate_actor(
                trainer.actor, env_cfg=env_cfg,
                scenario_seeds=validation_seeds,
                n_action_reps=VAL_REPS_INTERMEDIATE,
                sample=True, device=device, action_seed=4242,
            )
            val_rows.append({
                "iter": trainer.iteration,
                "val_macro_drop": ve["macro_mean_drop"],
                "elapsed_s": time.time() - t0,
            })
            print(f"  iter {trainer.iteration:3d}  rollout_drop={m['rollout_drop']:.4f}  "
                  f"val_drop={ve['macro_mean_drop']:.4f}  "
                  f"policy_loss={m['policy_loss']:.4f}  value_loss={m['value_loss']:.4f}  "
                  f"kl_post_max={m['kl_max_post']:.4f}  entropy={m['entropy']:.4f}  "
                  f"trans={trainer.cumulative_transitions}")
        if trainer.cumulative_transitions >= 500_000:
            break

    trainer.save_last_iter(trainer.iteration)

    # 落盘 JSONL
    with open(out_dir / "train_metrics.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "val_metrics.jsonl", "w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_dir}/train_metrics.jsonl ({len(rows)} rows)")
    print(f"wrote {out_dir}/val_metrics.jsonl ({len(val_rows)} rows)")
    print(f"elapsed: {time.time() - t0:.1f}s  transitions: {trainer.cumulative_transitions}")


if __name__ == "__main__":
    main()
