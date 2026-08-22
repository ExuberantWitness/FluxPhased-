"""S7 M1 env contract gates — 2 jammers vs 2 radars."""
from __future__ import annotations
import pytest
import torch

from env.gpu.array_face_s7 import (
    EnvConfig, ArrayFaceS7VecEnv, UPAConfig,
    OBS_DIM_JAMMER, OBS_DIM_RADAR, N_JAMMERS, N_RADARS,
)
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.g3_bsta_lite.action_contract import ContractViolation

PHYSICS = default_debug_physics_config(P_jam_W=0.1)  # S6b-validated regime


def make_env(n_envs=2, seed=42):
    cfg = EnvConfig(n_envs=n_envs, horizon=64, active_budget_steps=63,
                    device="cpu", seed=seed)
    return ArrayFaceS7VecEnv(cfg, physics=PHYSICS, radar=UPAConfig(), jammer=UPAConfig())


def jam_on(E, K=2, n_cells=1):
    cell = torch.zeros(E, K, 25); cell[:, :, :n_cells] = 1.0
    return cell


def idle_radar(E):
    return (torch.zeros(E, N_RADARS, dtype=torch.int64),
            torch.zeros(E, N_RADARS, dtype=torch.int64))


def test_s7_team_budget_split():
    cfg = EnvConfig(n_envs=2, horizon=64, active_budget_steps=63)
    assert sum(cfg.E0_tokens_per) == 63
    assert cfg.E0_tokens_per == (32, 31), f"unexpected split {cfg.E0_tokens_per}"


def test_s7_reset_obs_shapes():
    env = make_env()
    obs_j, obs_r = env.reset(seed=42)
    assert obs_j.shape == (2, N_JAMMERS, OBS_DIM_JAMMER)
    assert obs_r.shape == (2, N_RADARS, OBS_DIM_RADAR)
    assert torch.isfinite(obs_j).all() and torch.isfinite(obs_r).all()
    priv_j, priv_r = env.privileged()
    assert priv_j.shape == (2, 2 * OBS_DIM_JAMMER)
    assert priv_r.shape == (2, 2 * OBS_DIM_RADAR)


def test_s7_step_signature_and_rewards():
    env = make_env()
    env.reset(seed=42)
    E = 2
    (oj, orr), (rj, rr), done, info = env.step(
        jam_on(E), torch.full((E, N_JAMMERS), 12, dtype=torch.int64),
        *idle_radar(E),
    )
    assert oj.shape == (E, N_JAMMERS, OBS_DIM_JAMMER)
    assert orr.shape == (E, N_RADARS, OBS_DIM_RADAR)
    assert rj.shape == (E,) and rr.shape == (E,)
    assert torch.isfinite(rj).all() and torch.isfinite(rr).all()
    assert info["jnr_db"].shape == (E, N_RADARS)
    assert info["jnr_per"].shape == (E, N_JAMMERS, N_RADARS)
    assert info["snr_eff_db"].shape == (E, N_RADARS)


def test_s7_per_jammer_energy_and_idle():
    env = make_env()
    env.reset(seed=42)
    E = 2
    beam = torch.full((E, N_JAMMERS), 12, dtype=torch.int64)
    rb, rs = idle_radar(E)
    # both jammers active with 1 cell each
    env.step(jam_on(E, n_cells=1), beam, rb, rs)
    assert (env.energy_tokens == torch.tensor([[31, 30], [31, 30]])).all(), \
        f"per-jammer budgets: {env.energy_tokens.tolist()}"
    # only jammer 0 active with 2 cells
    cell = torch.zeros(E, N_JAMMERS, 25); cell[:, 0, :2] = 1.0
    env.step(cell, beam, rb, rs)
    assert (env.energy_tokens == torch.tensor([[29, 30], [29, 30]])).all()
    # fully idle consumes nothing
    env.step(torch.zeros(E, N_JAMMERS, 25), beam, rb, rs)
    assert (env.energy_tokens == torch.tensor([[29, 30], [29, 30]])).all()


def test_s7_illegal_cells_raise():
    env = make_env()
    env.reset(seed=42)
    E = 2
    env.energy_tokens[:] = 0
    with pytest.raises(ContractViolation):
        env.step(jam_on(E), torch.zeros(E, N_JAMMERS, dtype=torch.int64), *idle_radar(E))


def test_s7_bad_shapes_raise():
    env = make_env()
    env.reset(seed=42)
    E = 2
    with pytest.raises(ContractViolation):
        env.step(torch.zeros(E, 25), torch.zeros(E, N_JAMMERS, dtype=torch.int64),
                 *idle_radar(E))  # missing K axis
    with pytest.raises(ContractViolation):
        env.step(jam_on(E), torch.zeros(E, N_JAMMERS, dtype=torch.int64),
                 torch.zeros(E, dtype=torch.int64),  # missing R axis
                 torch.zeros(E, N_RADARS, dtype=torch.int64))


def test_s7_radar_service_routing():
    env = make_env(n_envs=4, seed=7)
    env.reset(seed=7)
    E = 4
    (oj, orr), (rj, rr), _, info = env.step(
        torch.zeros(E, N_JAMMERS, 25), torch.zeros(E, N_JAMMERS, dtype=torch.int64),
        *idle_radar(E),
    )
    assert torch.isinf(info["jnr_db"]).all()  # both jammers idle -> -inf


def test_s7_ledger_identity_full_episode():
    env = make_env(n_envs=2, seed=11)
    env.reset(seed=11)
    g = torch.Generator().manual_seed(0)
    E = 2
    for t in range(64):
        cell = (torch.rand(E, N_JAMMERS, 25, generator=g) < 0.04).float()
        cell = cell * (env.energy_tokens >= 1).float().unsqueeze(-1)
        env.step(
            cell,
            torch.randint(0, 25, (E, N_JAMMERS), generator=g),
            torch.randint(0, 25, (E, N_RADARS), generator=g),
            torch.randint(0, 2, (E, N_RADARS), generator=g),
        )
    assert env.ledger_identity_residual() == 0


def test_s7_detected_flags_and_esm_slots_in_obs():
    env = make_env()
    env.reset(seed=42)
    E = 2
    (oj, orr), _, _, _ = env.step(
        torch.zeros(E, N_JAMMERS, 25), torch.zeros(E, N_JAMMERS, dtype=torch.int64),
        *idle_radar(E),
    )
    # jammer obs slots [43..44] carry per-radar detected-last-step flags
    assert oj[0, 0, 43].item() in (0.0, 1.0) and oj[0, 0, 44].item() in (0.0, 1.0)
    # jammer obs partner channel: [55..59]/[60..64] one-hots + [66] active flag
    assert oj[0, 0, 55:60].sum() == 1.0 and oj[0, 0, 60:65].sum() == 1.0
    assert (oj[:, :, 66] == 0).all()  # both idle
    # radar obs per-jammer active flags at slots [48] and [59] (both idle)
    assert (orr[:, :, 48] == 0).all() and (orr[:, :, 59] == 0).all()


def test_s7_target_gain_boresight_zero():
    from env.gpu.array_face_s7.physics import target_gain_db
    beam_az = torch.arange(5)
    beam_el = torch.full((5,), 2)  # el = 0 plane
    same = target_gain_db(UPAConfig(), beam_az_idx=beam_az, beam_el_idx=beam_el,
                          mission_az_idx=beam_az)
    assert torch.allclose(same, torch.zeros(5), atol=1e-4), f"stare gain must be 0 dB: {same}"
    off = target_gain_db(UPAConfig(), beam_az_idx=beam_az, beam_el_idx=beam_el,
                         mission_az_idx=(beam_az + 1) % 5)
    assert (off < -1.0).all(), "off-axis gain must be negative"


def test_s7_per_mission_credit_by_bearing():
    env = make_env(n_envs=1, seed=3)
    env.reset(seed=3)
    env.tracker.pending[0] = []  # isolate from scenario arrivals
    env.tracker.admit(env_idx=0, step=0, service_id=0, az_idx=0, deadline_step=99)
    env.tracker.admit(env_idx=0, step=0, service_id=0, az_idx=4, deadline_step=99)
    ok = env.tracker.detect(env_idx=0, service_id=0, az_idx=0)
    assert ok
    ds = {m[1]: m[4] for m in env.tracker.pending[0]}
    assert ds[0] == 1 and ds[4] == 0, f"credit must be per-(svc,az): {ds}"


def test_s7_az_map_in_obs():
    env = make_env(n_envs=1, seed=3)
    env.reset(seed=3)
    env.tracker.admit(env_idx=0, step=0, service_id=1, az_idx=3, deadline_step=20)
    obs_j, obs_r = env._build_observation()
    # jammer obs: map appended at [45..54] as [svc0 az 5 | svc1 az 5]
    assert obs_j[0, 0, 45:50].sum() == 0 and obs_j[0, 0, 45 + 5 + 3] == 1.0
    # radar obs: map at [1..10] as [svc0 az 5 | svc1 az 5]
    assert obs_r[0, 0, 1:6].sum() == 0 and obs_r[0, 0, 1 + 5 + 3] == 1.0
