"""S4 env contract tests — 2-arg step, per-cell energy, UPA geometry.

Mirrors test_array_face_s3.py for the S4 env (Bernoulli(25) + Cat(25), no
base head). Uses a small 2-env config on CPU.
"""
import pytest
import torch

from env.gpu.g3_bsta_lite.action_contract import ContractViolation
from env.gpu.g3_bsta_lite.observation import PROFILE_MDP_SANITY
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.array_face_s4 import (
    EnvConfig, ArrayFaceS4VecEnv, UPAConfig,
    N_CELLS_S4, N_ACTIONS_CELL, N_ACTIONS_BEAM, N_BEAM_DIRS_S4,
    OBS_DIM_S4, PRIVILEGED_DIM_S4,
)


def _make_env(horizon=64, active_budget_steps=None, duty_budget=1.0):
    if active_budget_steps is None:
        active_budget_steps = horizon - 1
    cfg = EnvConfig(
        n_envs=2, horizon=horizon, n_services=2,
        dt=1.0, P_jam_W=2.0,
        active_budget_steps=active_budget_steps, duty_budget=duty_budget,
        arrival_rate_per_service=0.15, baseline_snr_db=22.0,
        mission_tau_window=6, detects_required=1,
        profile=PROFILE_MDP_SANITY, obs_delay_steps=1,
        potential_coef=0.05, gamma=0.99,
        device="cpu", seed=0,
    )
    physics = default_debug_physics_config(P_jam_W=2.0)
    return ArrayFaceS4VecEnv(cfg, physics=physics, radar=UPAConfig(), jammer=UPAConfig())


def _idle_cell(E):
    return torch.zeros((E, N_CELLS_S4), dtype=torch.float32)


def _k_cells_on(E, k):
    cell = torch.zeros((E, N_CELLS_S4), dtype=torch.float32)
    cell[:, :k] = 1.0
    return cell


def _beam(E, idx=0):
    return torch.full((E,), idx, dtype=torch.int64)


# ---------------------------------------------------------------------------

def test_reset_obs_shape():
    env = _make_env()
    obs = env.reset(seed=12345)
    assert obs.shape == (2, OBS_DIM_S4)
    assert OBS_DIM_S4 == 31


def test_step_signature_2args():
    env = _make_env()
    env.reset(seed=12345)
    obs, reward, done, info = env.step(_k_cells_on(2, 3), _beam(2))
    assert obs.shape == (2, OBS_DIM_S4)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert info["n_active_cells"].shape == (2,)


def test_energy_depletes_by_cell_count():
    env = _make_env()
    env.reset(seed=12345)
    before = int(env.energy_tokens[0].item())
    env.step(_k_cells_on(2, 3), _beam(2))
    after = int(env.energy_tokens[0].item())
    assert before - after == 3


def test_all_zero_cells_is_idle():
    env = _make_env()
    env.reset(seed=12345)
    before = int(env.energy_tokens[0].item())
    obs, rew, done, info = env.step(_idle_cell(2), _beam(2))
    assert int(env.energy_tokens[0].item()) == before  # no tokens consumed
    assert not torch.isfinite(info["jnr_db"]).any()     # no jamming → -inf
    assert info["n_active_cells"].tolist() == [0, 0]
    assert info["tokens_consumed"].tolist() == [0, 0]


def test_mask_cell_gated_by_energy():
    env = _make_env()
    env.reset(seed=12345)
    env.energy_tokens[:] = 0
    mask_cell, mask_beam = env._compute_mask()
    assert not mask_cell.any()
    assert mask_beam.all()  # beam always legal
    assert mask_beam.shape == (2, N_ACTIONS_BEAM)
    assert mask_cell.shape == (2, N_ACTIONS_CELL)


def test_mask_cell_all_true_when_energy():
    env = _make_env()
    env.reset(seed=12345)
    mask_cell, _ = env._compute_mask()
    assert mask_cell.all()


def test_over_budget_topk_clamp():
    env = _make_env()
    env.reset(seed=12345)
    env.energy_tokens[:] = 3
    env.step(_k_cells_on(2, 25), _beam(2))  # request all 25, only 3 affordable
    assert int(env.energy_tokens[0].item()) == 0
    # prev_cell holds the executed (clamped) mask: exactly 3 cells on
    assert int(env.prev_cell[0].sum().item()) == 3


def test_illegal_cells_when_no_energy():
    env = _make_env()
    env.reset(seed=12345)
    env.energy_tokens[:] = 0
    with pytest.raises(ContractViolation):
        env.step(_k_cells_on(2, 1), _beam(2))


def test_ledger_identity_after_episode():
    env = _make_env(horizon=16, active_budget_steps=15)
    env.reset(seed=12345)
    for t in range(16):
        if int(env.energy_tokens[0].item()) >= 2:
            cell = _k_cells_on(2, 2)
        else:
            cell = _idle_cell(2)
        env.step(cell, _beam(2, idx=t % N_BEAM_DIRS_S4))
    assert env.ledger_identity_residual() == 0
    assert torch.isfinite(env.drop_ratio()).all()


def test_contract_violation_bad_beam():
    env = _make_env()
    env.reset(seed=12345)
    with pytest.raises(ContractViolation):
        env.step(_idle_cell(2), torch.full((2,), 25, dtype=torch.int64))
    with pytest.raises(ContractViolation):
        env.step(_idle_cell(2), torch.full((2,), -1, dtype=torch.int64))


def test_contract_violation_bad_cell():
    env = _make_env()
    env.reset(seed=12345)
    with pytest.raises(ContractViolation):
        env.step(torch.zeros(2, N_CELLS_S4 + 1), _beam(2))
    bad = torch.zeros(2, N_CELLS_S4)
    bad[0, 0] = 0.5
    with pytest.raises(ContractViolation):
        env.step(bad, _beam(2))


def test_privileged_shape():
    env = _make_env()
    env.reset(seed=12345)
    env.step(_k_cells_on(2, 4), _beam(2))
    priv = env.privileged()
    assert priv.shape == (2, PRIVILEGED_DIM_S4)
    assert PRIVILEGED_DIM_S4 == 49


def test_radar_sweep_2d_az_fastest():
    env = _make_env()
    env.reset(seed=12345)
    env.step(_idle_cell(2), _beam(2))
    info = env.step(_idle_cell(2), _beam(2))[3]
    # step 1: radar_beam_idx = 1 → az=1, el=0 (az fastest raster)
    assert info["radar_beam_az"].tolist() == [1, 1]
    assert info["radar_beam_el"].tolist() == [0, 0]


def test_obs_onehot_layout():
    env = _make_env()
    env.reset(seed=12345)
    obs = env._build_observation()
    # prev_active one-hot (idle at init) in lite slots [8..10]
    assert obs[0, 8].item() == 1.0  # prev step was idle (all-zero cells)
    assert obs[0, 9].item() == 0.0
    # radar beam one-hots: az slot = 11 + 0 (broadside el=0 → az=0)
    assert obs[0, 11].item() == 1.0
    assert obs[0, 16].item() == 1.0  # radar el one-hot (el=0)
    assert obs.shape == (2, 31)
