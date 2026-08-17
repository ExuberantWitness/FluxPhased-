"""S5 test gates — physics (M0) + env contract (M1).

M0 physics gates (HANDOFF gating discipline):
  - single-jammer equivalence: jammer 2 idle -> JNR_5 == JNR_4
  - twin jammers -> +10*log10(2) = +3.0103 dB vs single
  - both idle -> -inf
  - p_detect monotone decreasing in JNR
  - asymmetric beams combine correctly in linear scale

M1 env contract gates:
  - obs [E, K, 43] / privileged [E, 84]
  - per-jammer independent energy depletion
  - per-jammer idle semantics (all-zero cells)
  - per-jammer mask gating + illegal raise
  - per-jammer over-budget top-k clamp
  - obs other-block reflects the OTHER jammer's state
  - ledger identity after a full episode
"""
from __future__ import annotations
import math
import pytest
import torch

from env.gpu.array_face_s5 import (
    EnvConfig, ArrayFaceS5VecEnv, UPAConfig,
    OBS_DIM_S5, PRIVILEGED_DIM_S5, N_JAMMERS, N_ACTIONS_CELL, N_ACTIONS_BEAM,
)
from env.gpu.array_face_s5.physics import compute_jnr_db_s5, compute_p_detect_s5
from env.gpu.array_face_s4 import ArrayFaceS4VecEnv, EnvConfig as EnvConfigS4
from env.gpu.array_face_s4.physics import compute_jnr_db_s4
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.g3_bsta_lite.observation import PROFILE_MDP_SANITY
from env.gpu.g3_bsta_lite.action_contract import ContractViolation


PHYSICS = default_debug_physics_config(P_jam_W=2.0)


def make_cfg(n_envs=2, seed=42):
    return EnvConfig(n_envs=n_envs, horizon=64, active_budget_steps=63,
                     profile=PROFILE_MDP_SANITY, device="cpu", seed=seed)


def make_env(n_envs=2, seed=42):
    return ArrayFaceS5VecEnv(make_cfg(n_envs, seed), physics=PHYSICS,
                             radar=UPAConfig(), jammer=UPAConfig())


def jnr_args(E, active, beam_az_el_per_jammer, cell_counts, radar_idx=12):
    """Builds physics args. beam_az_el_per_jammer: [(az,el), (az,el)]."""
    svc = torch.zeros(E, dtype=torch.int64)
    r_az = torch.full((E,), radar_idx % 5, dtype=torch.int64)
    r_el = torch.full((E,), radar_idx // 5, dtype=torch.int64)
    j_az = torch.tensor([[p[0] for p in beam_az_el_per_jammer]] * E, dtype=torch.int64)
    j_el = torch.tensor([[p[1] for p in beam_az_el_per_jammer]] * E, dtype=torch.int64)
    cells = torch.zeros(E, N_JAMMERS, 25)
    for k, n in enumerate(cell_counts):
        cells[:, k, :n] = 1.0
    return dict(jammer_active=torch.tensor([active] * E), victim_service_id=svc,
                radar_beam_az_idx=r_az, radar_beam_el_idx=r_el,
                jammer_beam_az_idx=j_az, jammer_beam_el_idx=j_el, cell_mask=cells)


# ===================== M0: physics =====================

def test_s5_single_jammer_equivalence_with_s4():
    """Jammer 2 idle -> JNR_5 == JNR_4 exactly (up to float32 round-trip)."""
    E = 4
    args = jnr_args(E, active=[True, False], beam_az_el_per_jammer=[(2, 2), (2, 2)], cell_counts=[1, 0])
    j5 = compute_jnr_db_s5(PHYSICS, UPAConfig(), UPAConfig(), **args)
    # S4 reference: single jammer with jammer-0's params
    j4 = compute_jnr_db_s4(PHYSICS, UPAConfig(), UPAConfig(),
                           jammer_active=torch.ones(E, dtype=torch.bool),
                           victim_service_id=args["victim_service_id"],
                           radar_beam_az_idx=args["radar_beam_az_idx"],
                           radar_beam_el_idx=args["radar_beam_el_idx"],
                           jammer_beam_az_idx=args["jammer_beam_az_idx"][:, 0],
                           jammer_beam_el_idx=args["jammer_beam_el_idx"][:, 0],
                           cell_mask=args["cell_mask"][:, 0])
    assert torch.allclose(j5, j4, atol=1e-3), f"{j5} vs {j4}"
    assert 50.0 < j5[0].item() < 57.0  # 1-cell broadside ~53.5 dB


def test_s5_twin_jammers_plus_3db():
    """Two identical active jammers -> +10*log10(2) vs single."""
    E = 4
    single = compute_jnr_db_s5(PHYSICS, UPAConfig(), UPAConfig(),
                               **jnr_args(E, [True, False], [(2, 2), (2, 2)], [1, 0]))
    twin = compute_jnr_db_s5(PHYSICS, UPAConfig(), UPAConfig(),
                             **jnr_args(E, [True, True], [(2, 2), (2, 2)], [1, 1]))
    delta = (twin - single).mean().item()
    assert abs(delta - 10.0 * math.log10(2.0)) < 1e-3, f"delta={delta}"


def test_s5_both_idle_neg_inf():
    j = compute_jnr_db_s5(PHYSICS, UPAConfig(), UPAConfig(),
                          **jnr_args(2, [False, False], [(0, 0), (0, 0)], [0, 0]))
    assert torch.isinf(j).all() and (j < 0).all()


def test_s5_asymmetric_beams_linear_sum():
    """Jammer 1 broadside (1 cell) + jammer 2 ridge (1 cell) combines in mW."""
    E = 2
    j_broad = compute_jnr_db_s5(PHYSICS, UPAConfig(), UPAConfig(),
                                **jnr_args(E, [True, False], [(2, 2), (0, 0)], [1, 0]))
    j_ridge = compute_jnr_db_s5(PHYSICS, UPAConfig(), UPAConfig(),
                                **jnr_args(E, [False, True], [(0, 0), (2, 3)], [0, 1]))
    j_both = compute_jnr_db_s5(PHYSICS, UPAConfig(), UPAConfig(),
                               **jnr_args(E, [True, True], [(2, 2), (2, 3)], [1, 1]))
    expected = 10.0 * torch.log10(10.0 ** (j_broad / 10.0) + 10.0 ** (j_ridge / 10.0))
    assert torch.allclose(j_both, expected, atol=1e-3)
    # dominant jammer sets the scale; the weak one adds a small amount
    assert (j_both > j_broad).all() and (j_both - j_broad).max() < 1.0


def test_s5_p_detect_monotone():
    # JNR listed in DECREASING order -> p_detect must INCREASE
    jnr = torch.tensor([80.0, 40.0, 20.0, 0.0, -20.0])
    p = compute_p_detect_s5(PHYSICS, baseline_snr_db=22.0, jnr_db=jnr)
    assert (p.diff() > 0).all(), f"p_detect must decrease in JNR: {p}"


# ===================== M1: env contract =====================

def test_s5_reset_obs_shape():
    env = make_env()
    obs = env.reset(seed=42)
    assert obs.shape == (2, N_JAMMERS, OBS_DIM_S5), obs.shape
    assert obs.dtype == torch.float32
    assert torch.isfinite(obs).all()


def test_s5_privileged_shape():
    env = make_env()
    env.reset(seed=42)
    priv = env.privileged()
    assert priv.shape == (2, PRIVILEGED_DIM_S5), priv.shape


def test_s5_energy_depletes_per_jammer():
    env = make_env()
    env.reset(seed=42)
    cell = torch.zeros(2, N_JAMMERS, 25)
    cell[:, 0, 0] = 1.0  # only jammer 0 transmits, 1 cell
    cell[:, 1, :3] = 1.0  # jammer 1 transmits 3 cells
    beam = torch.zeros(2, N_JAMMERS, dtype=torch.int64)
    env.step(cell, beam)
    assert (env.energy_tokens[:, 0] == 62).all()
    assert (env.energy_tokens[:, 1] == 60).all()


def test_s5_all_zero_cells_is_idle_per_jammer():
    env = make_env()
    env.reset(seed=42)
    cell = torch.zeros(2, N_JAMMERS, 25)
    beam = torch.full((2, N_JAMMERS), 12, dtype=torch.int64)
    _, _, _, info = env.step(cell, beam)
    assert (info["is_jam"] == 0).all()
    assert (env.energy_tokens == 63).all()
    assert torch.isinf(info["jnr_db"]).all()


def test_s5_mask_gated_per_jammer():
    env = make_env()
    env.reset(seed=42)
    # drain jammer 0 completely
    env.energy_tokens[:, 0] = 0
    env.energy[:, 0] = 0.0
    mask_cell, mask_beam = env._compute_mask()
    assert (mask_cell[:, 0] == False).all()
    assert (mask_cell[:, 1] == True).all()
    assert mask_beam.all()


def test_s5_illegal_cells_when_no_energy():
    env = make_env()
    env.reset(seed=42)
    env.energy_tokens[:, 1] = 0
    env.energy[:, 1] = 0.0
    cell = torch.zeros(2, N_JAMMERS, 25)
    cell[:, 1, 0] = 1.0  # illegal for jammer 1
    beam = torch.zeros(2, N_JAMMERS, dtype=torch.int64)
    with pytest.raises(ContractViolation):
        env.step(cell, beam)


def test_s5_over_budget_topk_clamp_per_jammer():
    env = make_env()
    env.reset(seed=42)
    env.energy_tokens[:, 0] = 2  # jammer 0 can only afford 2 cells
    env.energy[:, 0] = 2 * 2.0
    cell = torch.zeros(2, N_JAMMERS, 25)
    cell[:, 0, 5:10] = 1.0  # 5 requested
    cell[:, 1, 0] = 1.0
    beam = torch.zeros(2, N_JAMMERS, dtype=torch.int64)
    _, _, _, info = env.step(cell, beam)
    assert (info["n_active_cells"][:, 0] == 2).all()
    assert (info["n_active_cells"][:, 1] == 1).all()
    assert (env.energy_tokens[:, 0] == 0).all()


def test_s5_obs_other_block_reflects_other_jammer():
    env = make_env()
    env.reset(seed=42)
    # jammer 1 used beam idx 17 (az=2, el=3), was active, has full energy
    env.prev_beam[:, 1] = 17
    env.prev_cell[:, 1, 0] = 1.0
    obs0 = env._build_observation_for(0)
    # other az one-hot at slots [31..35]: az idx = 17 % 5 = 2
    assert obs0[0, 31 + 2] == 1.0 and obs0[0, 31:36].sum() == 1.0
    # other el one-hot at slots [36..40]: el idx = 17 // 5 = 3
    assert obs0[0, 36 + 3] == 1.0
    # other rem_E = 1.0 (slot 41), other prev_active = 1 (slot 42)
    assert obs0[0, 41] == 1.0 and obs0[0, 42] == 1.0
    # own slots [21..30] encode jammer 0's own prev beam (0) and prev_active (0)
    assert obs0[0, 21] == 1.0 and obs0[0, 42] == 1.0


def test_s5_ledger_identity_after_episode():
    env = make_env(n_envs=2, seed=7)
    env.reset(seed=7)
    g = torch.Generator().manual_seed(0)
    for t in range(64):
        cell = (torch.rand(2, N_JAMMERS, 25, generator=g) < 0.04).float()
        beam = torch.randint(0, 25, (2, N_JAMMERS), generator=g)
        # respect energy: zero out cells for exhausted jammers
        can = (env.energy_tokens >= 1).float().unsqueeze(-1)
        cell = cell * can
        env.step(cell, beam)
    assert env.ledger_identity_residual() == 0
    assert env.accounting_residual().abs().max() == 0


def test_s5_team_reward_shape_and_finiteness():
    env = make_env()
    env.reset(seed=42)
    cell = torch.zeros(2, N_JAMMERS, 25)
    cell[:, :, 0] = 1.0
    beam = torch.full((2, N_JAMMERS), 12, dtype=torch.int64)
    obs, reward, done, info = env.step(cell, beam)
    assert reward.shape == (2,)
    assert torch.isfinite(reward).all()
    assert obs.shape == (2, N_JAMMERS, OBS_DIM_S5)


def test_s5_contract_violation_bad_shapes():
    env = make_env()
    env.reset(seed=42)
    with pytest.raises(ContractViolation):
        env.step(torch.zeros(2, 25), torch.zeros(2, N_JAMMERS, dtype=torch.int64))  # missing K axis
    with pytest.raises(ContractViolation):
        env.step(torch.zeros(2, N_JAMMERS, 25), torch.zeros(2, 2, dtype=torch.float32))  # wrong dtype
