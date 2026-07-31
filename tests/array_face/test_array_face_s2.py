"""S2 env contract tests.

Tests MultiDiscrete action handling, mask, obs shape, episode lifecycle,
mission accounting, and ledger identity.
"""
from __future__ import annotations
import pytest
import torch

from env.gpu.array_face_s2 import (
    EnvConfig, ArrayFaceS2VecEnv, RadarULAConfig, JammerULAConfig,
    OBS_DIM_S2, N_ACTIONS_BASE, N_ACTIONS_BEAM,
)
from env.gpu.array_face_s2.action_contract import ContractViolation, validate_actions
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config


def _make_env(n_envs=4, horizon=16, profile="mdp_sanity_v1", P_jam_W=10.0):
    physics = default_debug_physics_config(P_jam_W=P_jam_W)
    cfg = EnvConfig(
        n_envs=n_envs, horizon=horizon, active_budget_steps=horizon // 4,
        profile=profile, device="cpu", P_jam_W=P_jam_W,
    )
    env = ArrayFaceS2VecEnv(cfg, physics=physics, radar=RadarULAConfig(), jammer=JammerULAConfig())
    return env, cfg


def test_reset_obs_shape():
    env, _ = _make_env(n_envs=4, horizon=16)
    obs = env.reset(seed=42)
    assert obs.shape == (4, OBS_DIM_S2), f"expected (4, 21), got {tuple(obs.shape)}"
    assert obs.dtype == torch.float32


def test_action_validation_dtype():
    """Non-int64 action raises ContractViolation."""
    env, _ = _make_env()
    env.reset(seed=1)
    bad_base = torch.tensor([0, 1, 2, 0], dtype=torch.float32)  # wrong dtype
    good_beam = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ContractViolation):
        env.step(bad_base, good_beam)


def test_action_validation_range():
    """Out-of-range beam (>4) raises."""
    env, _ = _make_env()
    env.reset(seed=1)
    base = torch.zeros(4, dtype=torch.int64)
    bad_beam = torch.tensor([0, 1, 2, 9], dtype=torch.int64)
    with pytest.raises(ContractViolation):
        env.step(base, bad_beam)


def test_action_validation_shape():
    """Wrong shape raises."""
    env, _ = _make_env(n_envs=4)
    env.reset(seed=1)
    bad_base = torch.zeros((4, 1), dtype=torch.int64)
    beam = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ContractViolation):
        env.step(bad_base, beam)


def test_step_after_done_raises():
    env, _ = _make_env(n_envs=2, horizon=4)
    env.reset(seed=1)
    base = torch.zeros(2, dtype=torch.int64)
    beam = torch.zeros(2, dtype=torch.int64)
    for _ in range(4):
        env.step(base, beam)
    assert env._done_flag
    with pytest.raises(RuntimeError):
        env.step(base, beam)


def test_energy_depletes_on_jam():
    """Jamming costs 1 token; idle doesn't."""
    env, _ = _make_env(n_envs=2, horizon=8, profile="mdp_sanity_v1")
    env.reset(seed=1)
    e0 = env.energy_tokens.clone()
    base = torch.tensor([1, 0], dtype=torch.int64)  # jam svc_0, idle
    beam = torch.tensor([2, 2], dtype=torch.int64)
    env.step(base, beam)
    assert env.energy_tokens[0].item() == e0[0].item() - 1
    assert env.energy_tokens[1].item() == e0[1].item()


def test_mask_blocks_when_no_energy():
    """When out of energy, base=1/2 must raise."""
    env, _ = _make_env(n_envs=1, horizon=8, profile="mdp_sanity_v1")
    env.cfg.active_budget_steps = 1  # give 1 token
    env.reset(seed=1)
    env.energy_tokens = torch.zeros(1, dtype=torch.int64)
    env.energy = torch.zeros(1, dtype=torch.float32)
    base = torch.tensor([1], dtype=torch.int64)
    beam = torch.tensor([2], dtype=torch.int64)
    with pytest.raises(ContractViolation):
        env.step(base, beam)


def test_beam_always_legal():
    """Beam action 0..4 always accepted regardless of energy state."""
    env, _ = _make_env(n_envs=1, horizon=8, profile="mdp_sanity_v1")
    env.reset(seed=1)
    env.energy_tokens = torch.zeros(1, dtype=torch.int64)
    env.energy = torch.zeros(1, dtype=torch.float32)
    base = torch.tensor([0], dtype=torch.int64)  # idle, legal
    for beam_idx in range(N_ACTIONS_BEAM):
        beam = torch.tensor([beam_idx], dtype=torch.int64)
        env.step(base, beam)  # should not raise


def test_ledger_identity_after_episode():
    """Episode end: ledger residual = 0 (every eligible mission is success or timeout)."""
    env, cfg = _make_env(n_envs=4, horizon=64, profile="mdp_sanity_v1")
    env.reset(seed=20260801)
    while not env._done_flag:
        # Energy-aware: jam only where energy remains; vary beam_az deterministically
        can_jam = env.energy_tokens >= 1
        # alternate svc 0/1/0/1 for envs that have energy; idle otherwise
        target = torch.tensor([1, 2, 1, 2], dtype=torch.int64)
        base = torch.where(can_jam, target, torch.zeros_like(target))
        beam = torch.full((4,), (env.step_idx % N_ACTIONS_BEAM), dtype=torch.int64)
        env.step(base, beam)
    resid = env.ledger_identity_residual()
    assert resid == 0, f"ledger residual {resid} != 0 after episode end"


def test_done_flag_and_horizon():
    """Done after exactly horizon steps."""
    env, cfg = _make_env(n_envs=2, horizon=8, profile="mdp_sanity_v1")
    env.reset(seed=1)
    base = torch.zeros(2, dtype=torch.int64)
    beam = torch.zeros(2, dtype=torch.int64)
    for i in range(7):
        env.step(base, beam)
        assert not env._done_flag
    env.step(base, beam)
    assert env._done_flag
    assert env.step_idx == cfg.horizon


def test_prev_action_recorded_in_obs():
    """After step, prev_base/prev_beam updated (used by obs builder for one-hot)."""
    env, _ = _make_env(n_envs=1, horizon=16, profile="mdp_sanity_v1")
    env.reset(seed=1)
    base = torch.tensor([2], dtype=torch.int64)
    beam = torch.tensor([3], dtype=torch.int64)
    env.step(base, beam)
    assert env.prev_base.item() == 2
    assert env.prev_beam.item() == 3


def test_pomdp_profile_supported():
    """POMDP profile also works (no crash)."""
    env, _ = _make_env(n_envs=2, horizon=16, profile="pomdp_v1")
    obs = env.reset(seed=1)
    assert obs.shape == (2, OBS_DIM_S2)


def test_sample_action_rng():
    """sample_action_rng returns tuple of valid range tensors."""
    env, _ = _make_env(n_envs=4, horizon=16)
    env.reset(seed=1)
    base, beam = env.sample_action_rng()
    assert base.shape == (4,)
    assert beam.shape == (4,)
    assert base.dtype == torch.int64
    assert beam.dtype == torch.int64
    assert (base >= 0).all() and (base < N_ACTIONS_BASE).all()
    assert (beam >= 0).all() and (beam < N_ACTIONS_BEAM).all()
