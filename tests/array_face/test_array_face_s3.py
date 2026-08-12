"""S3 env contract tests.

Tests the three-head action handling (base + beam + cell), per-cell energy
budget, zero-cell clamp, mask semantics, obs shape, episode lifecycle, and
mission accounting. Mirrors S2's contract test suite plus S3-specific cases.
"""
from __future__ import annotations
import pytest
import torch

from env.gpu.array_face_s3 import (
    EnvConfig, ArrayFaceS3VecEnv, RadarULAConfig, JammerULAConfig,
    OBS_DIM_S3, PRIVILEGED_DIM_S3, N_ACTIONS_BASE, N_ACTIONS_BEAM, N_CELLS,
)
from env.gpu.g3_bsta_lite.action_contract import ContractViolation
from env.gpu.g3_bsta_lite.observation import PROFILE_MDP_SANITY
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config


def _make_env(n_envs=4, horizon=16, profile="mdp_sanity_v1", P_jam_W=2.0):
    physics = default_debug_physics_config(P_jam_W=P_jam_W)
    # S3 per-cell token semantics: jamming with k cells costs k tokens. Tests
    # need a generous token budget (many jam steps). active_budget_steps is the
    # total token budget; the env enforces active_budget_steps < horizon (the
    # "always-on infeasible" guard) and active_budget_steps <= duty_budget*horizon.
    # horizon-1 satisfies the first guard; duty_budget=1.0 lifts the second.
    cfg = EnvConfig(
        n_envs=n_envs, horizon=horizon,
        active_budget_steps=horizon - 1, duty_budget=1.0,
        profile=profile, device="cpu", P_jam_W=P_jam_W,
    )
    env = ArrayFaceS3VecEnv(cfg, physics=physics, radar=RadarULAConfig(), jammer=JammerULAConfig())
    return env, cfg


def _all_cells(n_envs, device="cpu"):
    """Helper: all-cells-on mask [n_envs, N_CELLS]."""
    return torch.ones((n_envs, N_CELLS), dtype=torch.float32, device=device)


def test_reset_obs_shape():
    """reset() returns obs [E, OBS_DIM_S3=21] float32 (unchanged from S2)."""
    env, _ = _make_env(n_envs=4, horizon=16)
    obs = env.reset(seed=42)
    assert obs.shape == (4, OBS_DIM_S3), f"expected (4, 21), got {tuple(obs.shape)}"
    assert obs.dtype == torch.float32


def test_step_requires_three_action_args():
    """S3 step takes (action_base, action_beam, action_cell); missing cell raises TypeError."""
    env, _ = _make_env()
    env.reset(seed=42)
    base = torch.zeros(4, dtype=torch.int64)
    beam = torch.zeros(4, dtype=torch.int64)
    # two-arg call (S2 style) must fail
    with pytest.raises(TypeError):
        env.step(base, beam)
    # three-arg call succeeds
    cell = _all_cells(4)
    obs, reward, done, info = env.step(base, beam, cell)
    assert obs.shape == (4, OBS_DIM_S3)


def test_action_validation_cell_binary():
    """Non-binary cell value raises ContractViolation."""
    env, _ = _make_env()
    env.reset(seed=1)
    base = torch.zeros(4, dtype=torch.int64)
    beam = torch.zeros(4, dtype=torch.int64)
    bad_cell = torch.full((4, N_CELLS), 0.5, dtype=torch.float32)  # not binary
    with pytest.raises(ContractViolation):
        env.step(base, beam, bad_cell)


def test_action_validation_cell_shape():
    """Wrong cell mask shape raises ContractViolation."""
    env, _ = _make_env()
    env.reset(seed=1)
    base = torch.zeros(4, dtype=torch.int64)
    beam = torch.zeros(4, dtype=torch.int64)
    bad_cell = torch.zeros((4, N_CELLS + 1), dtype=torch.float32)  # wrong cell count
    with pytest.raises(ContractViolation):
        env.step(base, beam, bad_cell)


def test_energy_depletes_by_cell_count():
    """Per-cell token semantics: jamming with k active cells consumes k tokens.

    Idle (base=0) consumes 0 regardless of cell mask.
    """
    env, cfg = _make_env(n_envs=2, horizon=16)
    env.reset(seed=42)
    tokens_before = env.energy_tokens.clone()
    # env 0: idle (0 tokens); env 1: jam with all 5 cells (5 tokens)
    base = torch.tensor([0, 1], dtype=torch.int64)  # idle, jam_svc_0
    beam = torch.tensor([0, 2], dtype=torch.int64)
    cell = torch.zeros((2, N_CELLS), dtype=torch.float32)
    cell[1] = 1.0  # env 1: all 5 cells
    env.step(base, beam, cell)
    tokens_after = env.energy_tokens
    assert tokens_after[0].item() == tokens_before[0].item(), "idle must consume 0 tokens"
    assert tokens_after[1].item() == tokens_before[1].item() - N_CELLS, \
        f"jam with {N_CELLS} cells must consume {N_CELLS} tokens, got delta {tokens_before[1]-tokens_after[1]}"


def test_energy_depletes_by_partial_cell_count():
    """Jamming with 3 cells consumes 3 tokens (not 5)."""
    env, cfg = _make_env(n_envs=1, horizon=16)
    env.reset(seed=42)
    tokens_before = env.energy_tokens[0].item()
    base = torch.tensor([1], dtype=torch.int64)  # jam_svc_0
    beam = torch.tensor([2], dtype=torch.int64)
    cell = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)  # 3 cells
    env.step(base, beam, cell)
    delta = tokens_before - env.energy_tokens[0].item()
    assert delta == 3, f"3 active cells must consume 3 tokens, got {delta}"


def test_cell_zero_clamp_on_jam():
    """base=jam + all-zero cell mask -> env forces >=1 cell on (executed).

    The requested_cell (trace) keeps the original all-zero; executed_cell has
    >=1 forced on. Tokens consumed = >=1.
    """
    env, _ = _make_env(n_envs=1, horizon=16)
    env.reset(seed=42)
    base = torch.tensor([1], dtype=torch.int64)  # jam_svc_0
    beam = torch.tensor([2], dtype=torch.int64)
    cell = torch.zeros((1, N_CELLS), dtype=torch.float32)  # all zero
    obs, reward, done, info = env.step(base, beam, cell)
    trace = info["trace"]
    # requested stays all-zero
    assert trace.requested_cell.sum().item() == 0, "requested_cell should be preserved all-zero"
    # executed has >=1 cell
    assert trace.executed_cell.sum().item() >= 1, "executed_cell must have >=1 after clamp"
    assert info["n_active_cells"][0].item() >= 1
    assert info["tokens_consumed"][0].item() >= 1


def test_mask_returns_three_heads():
    """_compute_mask returns (mask_base, mask_beam, mask_cell) tuple of 3."""
    env, _ = _make_env(n_envs=4)
    env.reset(seed=42)
    mb, mm, mc = env._compute_mask()
    assert mb.shape == (4, N_ACTIONS_BASE)
    assert mm.shape == (4, N_ACTIONS_BEAM)
    assert mc.shape == (4, N_CELLS)
    # cell mask always all-True
    assert mc.all(), "cell mask must be all-True (no per-cell constraint at sample time)"


def test_mask_base_blocks_jam_when_no_energy():
    """When energy_tokens == 0, mask_base forbids jam (only idle legal)."""
    env, _ = _make_env(n_envs=1, horizon=16, profile="mdp_sanity_v1")
    env.reset(seed=42)
    env.energy_tokens = torch.zeros(1, dtype=torch.int64)
    mb, mm, mc = env._compute_mask()
    assert mb[0, 0].item() == True, "idle always legal"
    assert mb[0, 1].item() == False, "jam_svc_0 blocked when no energy"
    assert mb[0, 2].item() == False, "jam_svc_1 blocked when no energy"


def test_ledger_identity_after_episode():
    """After a full episode, accounting_residual == 0 and ledger balances."""
    env, cfg = _make_env(n_envs=3, horizon=8)
    env.reset(seed=42)
    torch.manual_seed(0)
    for _ in range(cfg.horizon):
        base = torch.randint(0, 2, (3,), dtype=torch.int64)  # idle or jam_svc_0
        beam = torch.randint(0, N_ACTIONS_BEAM, (3,), dtype=torch.int64)
        cell = torch.randint(0, 2, (3, N_CELLS), dtype=torch.float32)
        # respect mask: if no energy, force idle
        mb, _, _ = env._compute_mask()
        can = mb[:, 1]  # can jam?
        base = torch.where(can, base, torch.zeros_like(base))
        env.step(base, beam, cell)
    resid = env.accounting_residual()
    assert (resid == 0).all(), f"accounting_residual must be 0, got {resid.tolist()}"
    assert env.ledger_identity_residual() == 0


def test_done_flag_and_horizon():
    """Stepping past horizon raises RuntimeError."""
    env, cfg = _make_env(n_envs=2, horizon=4)
    env.reset(seed=42)
    base = torch.zeros(2, dtype=torch.int64)
    beam = torch.zeros(2, dtype=torch.int64)
    cell = _all_cells(2)
    for _ in range(cfg.horizon):
        _, _, done, _ = env.step(base, beam, cell)
    assert done.all()
    with pytest.raises(RuntimeError):
        env.step(base, beam, cell)


def test_sample_action_rng_returns_three_heads():
    """sample_action_rng returns (base, beam, cell) triple with correct shapes/dtypes."""
    env, _ = _make_env(n_envs=4)
    env.reset(seed=42)
    base, beam, cell = env.sample_action_rng()
    assert base.shape == (4,) and base.dtype == torch.int64
    assert beam.shape == (4,) and beam.dtype == torch.int64
    assert cell.shape == (4, N_CELLS) and cell.dtype == torch.float32
    assert set(cell.unique().tolist()) <= {0.0, 1.0}, "cell must be binary"


def test_privileged_dim_includes_cell_mask():
    """privileged() returns [E, PRIVILEGED_DIM_S3] (includes executed cell mask)."""
    env, _ = _make_env(n_envs=4)
    env.reset(seed=42)
    priv = env.privileged()
    assert priv.shape == (4, PRIVILEGED_DIM_S3), \
        f"expected (4, {PRIVILEGED_DIM_S3}), got {tuple(priv.shape)}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
