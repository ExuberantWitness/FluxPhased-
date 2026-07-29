"""F1 §6 Gate 0 contract test 2 — resource / mask / requested==executed."""

from __future__ import annotations

import pytest
import torch

from env.gpu.g3_bsta_lite import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    ACTION_JAM_SERVICE_1,
    EnvConfig,
    G3BstaLiteVecEnv,
    ContractViolation,
)


def test_requested_equals_executed_for_legal_actions():
    cfg = EnvConfig(n_envs=2, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=5)
    # Three legal action batches; check trace equality.
    for a_val in (ACTION_IDLE, ACTION_JAM_SERVICE_0, ACTION_JAM_SERVICE_1):
        actions = torch.full((2,), a_val, dtype=torch.int64)
        _, _, _, info = env.step(actions)
        tr = info["trace"]
        assert torch.equal(tr.requested_action, tr.executed_action)
        assert tr.legal.all()
        # Selected service must match action - 1 for jam, -1 for idle.
        if a_val == ACTION_IDLE:
            assert (tr.selected_service == -1).all()
        else:
            assert (tr.selected_service == a_val - 1).all()


def test_illegal_action_raises_contract_violation_without_state_advance():
    cfg = EnvConfig(n_envs=1, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=2)
    # Burn all energy with jam actions
    cost_per = cfg.P_jam_W * cfg.dt
    assert cfg.E0 > 0
    n_jams_possible = int(cfg.E0 // cost_per)
    for _ in range(n_jams_possible):
        env.step(torch.tensor([ACTION_JAM_SERVICE_0], dtype=torch.int64))
    # Now energy is below cost; mask should forbid jam.
    mask = env._compute_mask()
    assert not bool(mask[0, ACTION_JAM_SERVICE_0])
    assert not bool(mask[0, ACTION_JAM_SERVICE_1])
    assert bool(mask[0, ACTION_IDLE])
    energy_before = env.energy.clone()
    step_before = env.step_idx
    with pytest.raises(ContractViolation):
        env.step(torch.tensor([ACTION_JAM_SERVICE_0], dtype=torch.int64))
    # State must not advance on contract violation (DEBUG_CONTRACT.md §3).
    assert torch.equal(env.energy, energy_before)
    assert env.step_idx == step_before


def test_energy_dynamics_exact_cost_per_nonidle_action():
    cfg = EnvConfig(n_envs=1, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=8)
    e0 = env.energy.clone()
    env.step(torch.tensor([ACTION_IDLE], dtype=torch.int64))
    assert float(env.energy[0]) == float(e0[0])  # idle: no cost
    e1 = env.energy.clone()
    env.step(torch.tensor([ACTION_JAM_SERVICE_0], dtype=torch.int64))
    expected = float(e1[0]) - float(cfg.P_jam_W * cfg.dt)
    assert abs(float(env.energy[0]) - expected) < 1e-6


def test_always_on_infeasible_by_construction():
    cfg = EnvConfig(n_envs=1, device="cpu")
    cost_h = cfg.P_jam_W * cfg.dt * cfg.horizon
    assert cfg.E0 < cost_h, "always-on must be infeasible"


def test_mask_never_reveals_channel_activity_or_hidden_value():
    """Mask must depend only on resource state, not on radar service or arrivals.

    Construct two envs with identical energy but different scenario arrivals
    at the next step; their masks must be identical.
    """
    cfg = EnvConfig(n_envs=1, device="cpu")
    env_a = G3BstaLiteVecEnv(cfg)
    env_b = G3BstaLiteVecEnv(cfg)
    env_a.reset(seed=100)
    env_b.reset(seed=200)
    # Drive both to same energy by running same number of jam actions.
    for _ in range(3):
        env_a.step(torch.tensor([ACTION_JAM_SERVICE_0], dtype=torch.int64))
        env_b.step(torch.tensor([ACTION_JAM_SERVICE_0], dtype=torch.int64))
    assert torch.allclose(env_a.energy, env_b.energy)
    # Masks must agree regardless of different arrival tables.
    assert torch.equal(env_a._compute_mask(), env_b._compute_mask())
