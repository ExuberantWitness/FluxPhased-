"""WP-3 M1 verification tests: BR trainer buffer + PPO threading for beam_direction.

Tests:
  1. test_buffer_has_beam_direction_field: _RolloutBuffer has .beam_direction shape [H,E,R].
  2. test_buffer_has_detect_list_field: _RolloutBuffer has .detect_list shape [H,E,K,5].
  3. test_buffer_no_beam_target: _RolloutBuffer does NOT have .beam_target (god-view killed).
  4. test_ppo_update_changes_beam_direction_head_weights: collect 1 rollout → update →
     beam_direction_head.weight.grad is non-zero (gradient actually flows).
  5. test_ppo_update_changes_detect_mlp_weights: same for detect_mlp (encoder receives grad).
  6. test_entropy_coef_anneal: cosine anneal schedule from 0.01 → 0.001.
  7. test_no_nan_after_5_updates: 5 small iters no NaN in any state.
"""

from __future__ import annotations
import sys
import torch

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.br_trainer import TwoTeamBRTrainer, _RolloutBuffer
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander


def _make_trainer_and_env(n_envs=4, horizon=30):
    env = TwoTeamVecEnv(n_envs=n_envs, device="cuda", episode_steps=horizon,
                        geometry=RANDOM_GEOMETRY, seed=42)
    env.reset()
    ac = TwoTeamCommanderActorCritic().to("cuda")
    opp = BlindClassicalCommander()
    trainer = TwoTeamBRTrainer(
        ac, frozen_opponent=opp,
        n_epochs=2, minibatch_size=16,
        entropy_coef=0.01, entropy_coef_min=0.001, entropy_decay_iters=10,
        device="cuda")
    return env, ac, trainer


def test_buffer_has_beam_direction_field():
    """_RolloutBuffer has .beam_direction shape [H, E, n_aperture]."""
    env, ac, trainer = _make_trainer_and_env()
    buf = _RolloutBuffer(
        horizon=20, n_envs=env.E, obs_dim=env.obs_dim, priv_dim=env.privileged_dim,
        n_aperture=env.n_radars_per_team, n_fn=env.n_fn,
        k_max=env.k_max, device="cuda")
    assert hasattr(buf, "beam_direction"), "buffer missing beam_direction field"
    assert buf.beam_direction.shape == (20, env.E, env.n_radars_per_team)
    print(f"buffer.beam_direction shape = {tuple(buf.beam_direction.shape)}")


def test_buffer_has_detect_list_field():
    """_RolloutBuffer has .detect_list shape [H, E, K_max, 5]."""
    env, ac, trainer = _make_trainer_and_env()
    buf = _RolloutBuffer(
        horizon=20, n_envs=env.E, obs_dim=env.obs_dim, priv_dim=env.privileged_dim,
        n_aperture=env.n_radars_per_team, n_fn=env.n_fn,
        k_max=env.k_max, device="cuda")
    assert hasattr(buf, "detect_list"), "buffer missing detect_list field"
    assert buf.detect_list.shape == (20, env.E, env.k_max, 5)
    print(f"buffer.detect_list shape = {tuple(buf.detect_list.shape)}")


def test_buffer_no_beam_target():
    """_RolloutBuffer does NOT have .beam_target (god-view killed)."""
    env, ac, trainer = _make_trainer_and_env()
    buf = _RolloutBuffer(
        horizon=20, n_envs=env.E, obs_dim=env.obs_dim, priv_dim=env.privileged_dim,
        n_aperture=env.n_radars_per_team, n_fn=env.n_fn,
        k_max=env.k_max, device="cuda")
    assert not hasattr(buf, "beam_target"), (
        "buffer still has beam_target — god-view leak not fully removed")
    print("buffer has no beam_target field (god-view killed)")


def test_ppo_update_changes_beam_direction_head_weights():
    """Collect 1 rollout → PPO update → beam_direction_head.weight.grad is non-zero."""
    env, ac, trainer = _make_trainer_and_env()
    buf = trainer.collect_rollout(env, horizon=30, learning_team=0)
    trainer._compute_gae(buf)
    # Zero grads before update
    trainer.opt.zero_grad(set_to_none=True)
    trainer.update(buf, iter_idx=0, n_iters=10)
    grad = ac.beam_direction_head.weight.grad
    assert grad is not None, "beam_direction_head got no gradient — PPO threading broken"
    grad_norm = grad.abs().sum().item()
    assert grad_norm > 0, f"beam_direction_head grad all-zero (norm={grad_norm:.2e})"
    print(f"beam_direction_head.weight.grad norm = {grad_norm:.4e} (non-zero → PPO threaded)")


def test_ppo_update_changes_detect_mlp_weights():
    """PPO update produces non-zero grad in detect_mlp (encoder receives gradient)."""
    env, ac, trainer = _make_trainer_and_env()
    buf = trainer.collect_rollout(env, horizon=30, learning_team=0)
    trainer._compute_gae(buf)
    trainer.opt.zero_grad(set_to_none=True)
    trainer.update(buf, iter_idx=0, n_iters=10)
    total_grad = 0.0
    for p in ac.detect_mlp.parameters():
        assert p.grad is not None, "detect_mlp param got no gradient"
        total_grad += p.grad.abs().sum().item()
    assert total_grad > 0, f"detect_mlp grad all-zero (sum={total_grad:.2e})"
    print(f"detect_mlp grad sum = {total_grad:.4e} (non-zero → encoder threaded)")


def test_entropy_coef_anneal():
    """Cosine anneal: iter 0 → entropy_coef_max; iter n/2 → ~mid; iter n → entropy_coef_min."""
    env, ac, trainer = _make_trainer_and_env()
    n = 10
    c0 = trainer._entropy_coef(0, n)
    c_mid = trainer._entropy_coef(n // 2, n)
    c_end = trainer._entropy_coef(n, n)
    assert abs(c0 - 0.01) < 1e-6, f"iter 0 coef {c0} != max 0.01"
    assert abs(c_end - 0.001) < 1e-6, f"iter n coef {c_end} != min 0.001"
    assert c_end < c_mid < c0, f"anneal not monotone: {c0} → {c_mid} → {c_end}"
    print(f"entropy_coef anneal: iter0={c0:.4f} → iter{n//2}={c_mid:.4f} → iter{n}={c_end:.4f}")


def test_no_nan_after_5_updates():
    """5 rollout + update cycles, no NaN in any AC parameter."""
    env, ac, trainer = _make_trainer_and_env(n_envs=4, horizon=20)
    for it in range(5):
        buf = trainer.collect_rollout(env, horizon=20, learning_team=0)
        trainer._compute_gae(buf)
        trainer.update(buf, iter_idx=it, n_iters=5)
    for name, p in ac.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in {name}"
        assert not torch.isinf(p).any(), f"Inf in {name}"
    print("5 PPO updates completed, no NaN/Inf in AC params")


if __name__ == "__main__":
    test_buffer_has_beam_direction_field()
    test_buffer_has_detect_list_field()
    test_buffer_no_beam_target()
    test_ppo_update_changes_beam_direction_head_weights()
    test_ppo_update_changes_detect_mlp_weights()
    test_entropy_coef_anneal()
    test_no_nan_after_5_updates()
    print("\nAll WP-3 M1 br-trainer-beam-direction tests PASS")
