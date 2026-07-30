"""R2 探索性续训 — 从既有 checkpoint 续训 N iter.

IMPORTANT: 这是探索性扩展,**不用于 R2 Gate-3 重新判定**。
- 原始 R2 判定 = BLOCKED_LEARNING_CONTRIBUTION 保持不变
- 超出 preregistration §6 的 0.5M transition cap
- 目的: 观察 PPO 学习曲线是否饱和 / 仍在上升
- Adam optimizer moments 重置 (checkpoint 未存 opt state); 前 ~30 iter 有 warmup 偏差

输出: r2_gate3_output/curves_lr3e-05_kl0.01_extended/  (累积, 不覆盖原始 R2 段)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from env.gpu.g3_bsta_lite import EnvConfig, PROFILE_MDP_SANITY, N_ACTIONS
from experiments.g3_bsta_lite.learning_repair.trainer import (
    R2PPOConfig, R2PPOTrainer, MaskedCategoricalActor, evaluate_actor,
)
from experiments.g3_bsta_lite.learning_repair.run_r2_gate3 import (
    MANIFEST_DIR, OUT_DIR, load_seeds, env_cfg_for, VAL_EVERY,
    VAL_REPS_INTERMEDIATE,
)


# 第二轮续训: 从 iter 1999 继续训 1000 iter 到 iter 2999
EXT_DIR = OUT_DIR / "curves_lr3e-05_kl0.01_extended"
RESUME_CKPT = EXT_DIR / "last_iter1999.pt"
EXTRA_ITERS = 1000     # 从 iter 2000 训到 iter 2999


def main():
    device = "cuda"
    train_seeds = load_seeds("ppo_train")
    validation_seeds = load_seeds("checkpoint_validation")

    EXT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = R2PPOConfig(
        profile=PROFILE_MDP_SANITY,
        iterations=EXTRA_ITERS,
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
        out_dir=EXT_DIR,
    )

    # pristine_init 已存在(R2 原始段已存), 不重写

    # 从既有 checkpoint resume
    sd = torch.load(RESUME_CKPT, map_location=device)
    trainer.actor.load_state_dict(sd["actor_state_dict"])
    trainer.critic.load_state_dict(sd["critic_state_dict"])
    meta = sd["meta"]   # already a dict (CheckpointMeta.to_json returns asdict)
    trainer.iteration = int(meta["iteration"])
    trainer.cumulative_transitions = int(meta["cumulative_transitions"])
    trainer.update_count = int(meta["update_count"])
    trainer.kl_rollback_count = int(meta.get("extra", {}).get("kl_rollback_count", 0))
    print(f"resumed from {RESUME_CKPT.name}: iter={trainer.iteration}  "
          f"cum_trans={trainer.cumulative_transitions}  "
          f"update_count={trainer.update_count}")

    rows: list[dict] = []
    val_rows: list[dict] = []
    t0 = time.time()
    for it in range(EXTRA_ITERS):
        m = trainer.train_iteration()
        rows.append(m)
        if (it + 1) % VAL_EVERY == 0 or it == EXTRA_ITERS - 1:
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
            print(f"  iter {trainer.iteration:4d}  rollout_drop={m['rollout_drop']:.4f}  "
                  f"val_drop={ve['macro_mean_drop']:.4f}  "
                  f"policy_loss={m['policy_loss']:.4f}  value_loss={m['value_loss']:.4f}  "
                  f"kl_max_post={m['kl_max_post']:.5f}  entropy={m['entropy']:.4f}  "
                  f"trans={trainer.cumulative_transitions}  "
                  f"elapsed={time.time()-t0:.0f}s")

    trainer.save_last_iter(trainer.iteration)

    # 追加模式: 累积所有续训段, 不断覆盖原始 R2 段
    with open(EXT_DIR / "train_metrics.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(EXT_DIR / "val_metrics.jsonl", "a") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nappended to {EXT_DIR}/train_metrics.jsonl (+{len(rows)} rows)")
    print(f"appended to {EXT_DIR}/val_metrics.jsonl (+{len(val_rows)} rows)")
    print(f"total elapsed: {time.time() - t0:.1f}s  "
          f"final transitions: {trainer.cumulative_transitions}")


if __name__ == "__main__":
    main()
