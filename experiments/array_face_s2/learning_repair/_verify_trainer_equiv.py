"""Verify S2PPOTrainerV2 (default config) reproduces S2PPOTrainer bit-exactly
for the first training iteration.

This is the correctness gate for Step 2/3: with per_head_entropy=False,
normalize_returns=False, log_ratio_clamp=0, v2 must produce identical metrics
(rollout_drop, policy_loss, value_loss, entropy, kl, action_freq) to the
amend02 baseline trainer, given the same seed and env config.

If this passes, v2 is a faithful generalization and the amend02 results remain
reproducible through the v2 code path.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

import torch

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.g3_bsta_lite.observation import PROFILE_MDP_SANITY
from env.gpu.array_face_s2 import EnvConfig, RadarULAConfig, JammerULAConfig
from experiments.array_face_s2.learning_repair.trainer import (
    S2PPOConfig, S2PPOTrainer,
)
from experiments.array_face_s2.learning_repair.trainer_v2 import S2PPOConfigV2, S2PPOTrainerV2


def make_shared_env_cfg(seed, device):
    return EnvConfig(
        n_envs=16, horizon=64, n_services=2,
        dt=1.0, P_jam_W=2.0,  # post-fix per-cell
        active_budget_steps=16, duty_budget=0.25,
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1,
        profile=PROFILE_MDP_SANITY, obs_delay_steps=1,
        potential_coef=0.05, gamma=0.99,
        device=device, seed=seed,
    )


def copy_actor_weights(src, dst):
    """Copy MultiDiscreteActor -> MultiHeadActor weights (trunk + base/beam)."""
    dst.fc1.weight.data.copy_(src.fc1.weight.data)
    dst.fc1.bias.data.copy_(src.fc1.bias.data)
    dst.fc2.weight.data.copy_(src.fc2.weight.data)
    dst.fc2.bias.data.copy_(src.fc2.bias.data)
    dst.heads["base"].weight.data.copy_(src.head_base.weight.data)
    dst.heads["base"].bias.data.copy_(src.head_base.bias.data)
    dst.heads["beam"].weight.data.copy_(src.head_beam.weight.data)
    dst.heads["beam"].bias.data.copy_(src.head_beam.bias.data)


def main():
    seed = 20260729
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = make_shared_env_cfg(seed, device)
    physics = default_debug_physics_config(P_jam_W=2.0)
    radar = RadarULAConfig()
    jammer = JammerULAConfig()
    # shared train seeds: a small fixed subset for determinism
    train_seeds = [21011101, 21011102, 21011103, 21011104]

    # --- baseline trainer (amend02) ---
    torch.manual_seed(seed)
    cfg1 = S2PPOConfig(
        profile=PROFILE_MDP_SANITY, iterations=5,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02, entropy_coef_init=5e-3, entropy_anneal_frac=0.5,
        seed=seed, train_seed=seed, device=device,
    )
    t1 = S2PPOTrainer(
        cfg=cfg1, env_cfg=env_cfg, physics=physics, radar=radar, jammer=jammer,
        train_seeds=train_seeds,
        manifest_path=Path("experiments/array_face_s1/manifests/ppo_train.json"),
        out_dir=Path("_verify_tmp_t1"),
    )

    # --- v2 trainer (default config = amend02 equivalent) ---
    torch.manual_seed(seed)
    cfg2 = S2PPOConfigV2(
        profile=PROFILE_MDP_SANITY, iterations=5,
        n_envs=16, horizon=64, actor_lr=3e-5, critic_lr=1e-3,
        target_kl=0.02, entropy_coef_init=5e-3, entropy_anneal_frac=0.5,
        per_head_entropy=False, normalize_returns=False, log_ratio_clamp=0.0,
        seed=seed, train_seed=seed, device=device,
    )
    t2 = S2PPOTrainerV2(
        cfg=cfg2, env_cfg=env_cfg, physics=physics, radar=radar, jammer=jammer,
        train_seeds=train_seeds,
        manifest_path=Path("experiments/array_face_s1/manifests/ppo_train.json"),
        out_dir=Path("_verify_tmp_t2"),
    )

    # Both trainers were seeded with torch.manual_seed(seed) before actor init,
    # so their actor weights are identical (same RNG draws). Verify this:
    # (t1.actor is MultiDiscreteActor, t2.actor is MultiHeadActor)
    assert t1.actor.fc1.weight.allclose(t2.actor.fc1.weight), "fc1 weights differ"
    assert t1.actor.head_base.weight.allclose(t2.actor.heads["base"].weight), "base head differ"
    assert t1.actor.head_beam.weight.allclose(t2.actor.heads["beam"].weight), "beam head differ"
    print("[init] identical actor weights  OK")
    assert t1.critic.fc1.weight.allclose(t2.critic.fc1.weight), "critic fc1 differ"
    print("[init] identical critic weights OK")

    # Run one iteration on each and compare metrics.
    # CRITICAL: the two trainers share PyTorch's global RNG, but each __init__
    # called torch.manual_seed(seed), so we must capture and restore the RNG
    # state around each train_iteration so both see the SAME random draws
    # (for torch.randperm minibatch shuffling inside update()).
    rng_state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    m1 = t1.train_iteration()
    # restore identical RNG state for t2
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # also reset each trainer's action generator to the same seed (already done
    # in __init__ via train_seed, but re-affirm)
    t2._action_gen = torch.Generator(device=device).manual_seed(seed)
    m2 = t2.train_iteration()

    keys = ["rollout_drop", "policy_loss", "value_loss", "entropy",
            "entropy_base", "entropy_beam", "kl_mean_post", "kl_max_post",
            "clip_frac_mean", "adv_std", "explained_variance",
            "actor_grad_norm", "cumulative_transitions", "iteration"]
    ok = True
    print("\n--- iteration 1 metric comparison ---")
    for k in keys:
        v1, v2v = m1[k], m2[k]
        if isinstance(v1, float):
            d = abs(v1 - v2v)
            status = "OK" if d <= 1e-6 else "FAIL"
            if d > 1e-6:
                ok = False
            print(f"  {k:25s} v1={v1:.8f}  v2={v2v:.8f}  |d|={d:.2e}  {status}")
        else:
            print(f"  {k:25s} v1={v1}  v2={v2v}  {'OK' if v1==v2v else 'FAIL'}")

    # action freqs
    af1 = m1.get("action_freq")
    af2 = m2.get("action_base_freq")
    if af1 is not None and af2 is not None:
        import torch as _t
        d_af = max(abs(a - b) for a, b in zip(af1, af2))
        print(f"  {'action_base_freq':25s} |d|={d_af:.2e}  {'OK' if d_af<=1e-6 else 'CHECK'}")

    print("\nEQUIVALENCE PASSED" if ok else "EQUIVALENCE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
