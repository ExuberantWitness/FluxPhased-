"""F1 §6 Gate 0 contract test 1 — runtime invariants.

Covers: no NaN/Inf, no invalid mask, energy never negative, same seed/state/
action reproduces, observation_state_version monotonic.
"""

from __future__ import annotations

import pytest
import torch

from env.gpu.g3_bsta_lite import (
    EnvConfig,
    G3BstaLiteVecEnv,
    N_ACTIONS,
    OBS_DIM,
)


def _run_episode(env: G3BstaLiteVecEnv, actions_per_step: list[torch.Tensor]):
    obs0 = env.reset(seed=12345)
    obs_seq = [obs0]
    for a in actions_per_step:
        obs, reward, done, info = env.step(a)
        obs_seq.append(obs)
    return obs_seq


def test_reset_returns_finite_obs_with_correct_shape():
    cfg = EnvConfig(n_envs=4, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    obs = env.reset(seed=1)
    assert obs.shape == (4, OBS_DIM)
    assert torch.isfinite(obs).all()


def test_no_nan_inf_across_random_episode():
    cfg = EnvConfig(n_envs=4, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=42)
    for t in range(cfg.horizon):
        mask = env._compute_mask()
        # uniform among legal in env-0 (all envs share actions in this test)
        legal = [a for a in range(N_ACTIONS) if bool(mask[0, a])]
        action_val = legal[t % len(legal)]
        actions = torch.full((4,), action_val, dtype=torch.int64)
        obs, reward, done, info = env.step(actions)
        assert torch.isfinite(obs).all(), f"obs NaN/Inf at step {t}"
        assert torch.isfinite(reward).all(), f"reward NaN/Inf at step {t}"
        assert torch.isfinite(info["p_detect"]).all(), f"p_det NaN/Inf at step {t}"
        assert torch.isfinite(info["jnr_db"]).all() or (
            info["jnr_db"] == float("-inf")
        ).any(), f"jnr_db bad at step {t}"


def test_energy_never_negative():
    cfg = EnvConfig(n_envs=2, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=7)
    energies = [env.energy.clone()]
    for t in range(cfg.horizon):
        mask = env._compute_mask()
        legal = [a for a in range(N_ACTIONS) if bool(mask[0, a])]
        action_val = legal[t % len(legal)]
        actions = torch.full((2,), action_val, dtype=torch.int64)
        env.step(actions)
        energies.append(env.energy.clone())
    stack = torch.stack(energies)
    assert (stack >= 0).all(), "energy went negative"


def test_same_seed_state_action_reproduces_bitforbit():
    cfg = EnvConfig(n_envs=2, device="cpu")
    env_a = G3BstaLiteVecEnv(cfg)
    env_b = G3BstaLiteVecEnv(cfg)
    obs_a = env_a.reset(seed=99)
    obs_b = env_b.reset(seed=99)
    assert torch.equal(obs_a, obs_b), "reset(seed) not reproducible"
    # identical action sequence
    torch.manual_seed(0)
    actions_seq = [
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([2, 0], dtype=torch.int64),
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int64),
    ]
    obs_seq_a = []
    obs_seq_b = []
    for a in actions_seq:
        oa, ra, _, _ = env_a.step(a)
        ob, rb, _, _ = env_b.step(a)
        obs_seq_a.append((oa, ra))
        obs_seq_b.append((ob, rb))
    for ((oa, ra), (ob, rb)) in zip(obs_seq_a, obs_seq_b):
        assert torch.equal(oa, ob), "obs diverged under same seed+action"
        assert torch.equal(ra, rb), "reward diverged under same seed+action"


def test_observation_state_version_is_monotonic():
    cfg = EnvConfig(n_envs=2, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=3)
    versions = []
    for _ in range(5):
        a = torch.zeros(2, dtype=torch.int64)
        _, _, _, info = env.step(a)
        versions.append(info["trace"].observation_state_version)
    assert versions == sorted(versions), "version not monotonic"
    assert len(set(versions)) == 5, "version not strictly increasing"


def test_mask_shape_always_3_and_idle_always_legal():
    cfg = EnvConfig(n_envs=3, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=11)
    for t in range(cfg.horizon):
        mask = env._compute_mask()
        assert mask.shape == (3, N_ACTIONS), f"mask shape wrong at step {t}"
        assert mask[:, 0].all(), f"idle not always legal at step {t}"
        a = torch.zeros(3, dtype=torch.int64)
        env.step(a)
