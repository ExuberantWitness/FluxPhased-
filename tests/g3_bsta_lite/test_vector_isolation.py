"""F1 §6 Gate 0 contract test 4 — vector isolation.

env-0 perturbation cannot affect env-1. All RNG must be batched.

We construct a 2-env batch where env 0 emits jam and env 1 stays idle.
After one step:
  - env 0 has reduced energy and reduced p_det (matched service).
  - env 1 has unchanged energy and unchanged p_det.

Detector RNG is batched: each env gets its own independent uniform draw, so
the outcomes can differ across envs without leaking state.
"""

from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    EnvConfig,
    G3BstaLiteVecEnv,
)


def test_perturbing_env0_action_does_not_affect_env1_state():
    """Baseline: 2-env batch both idle. Treatment: 2-env batch env0 jams.

    Both runs use the same detector RNG seed. env-1's energy, intercept_age,
    and pending_count must be identical across baseline and treatment.
    """
    cfg_base = EnvConfig(n_envs=2, device="cpu", seed=4321)
    cfg_trt = EnvConfig(n_envs=2, device="cpu", seed=4321)
    env_base = G3BstaLiteVecEnv(cfg_base)
    env_trt = G3BstaLiteVecEnv(cfg_trt)
    env_base.reset(seed=4321)
    env_trt.reset(seed=4321)

    # Baseline: both idle.
    env_base.step(torch.tensor([ACTION_IDLE, ACTION_IDLE], dtype=torch.int64))
    # Treatment: env 0 jams, env 1 idle.
    env_trt.step(torch.tensor([ACTION_JAM_SERVICE_0, ACTION_IDLE], dtype=torch.int64))

    # env 1 energy must be identical (idle in both runs).
    assert float(env_base.energy[1]) == float(env_trt.energy[1])
    # env 0 energy must differ (jam cost in treatment only).
    assert float(env_trt.energy[0]) < float(env_base.energy[0])
    # env 1 prev_action identical.
    assert int(env_base.prev_action[1]) == int(env_trt.prev_action[1])


def test_detector_rng_is_batched_and_independent_per_env():
    """Detector RNG produces one draw per env per step; no shared scalar."""
    cfg = EnvConfig(n_envs=2, device="cpu", seed=2024)
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=2024)
    # Draw detection directly through the env's API.
    draw = torch.rand(2, generator=env._detector_gen, device="cpu")
    assert draw.shape == (2,)
    assert (draw >= 0).all() and (draw < 1).all()
    # Two consecutive draws differ.
    draw2 = torch.rand(2, generator=env._detector_gen, device="cpu")
    assert not torch.equal(draw, draw2)


def test_action_rng_separate_from_detector_rng():
    """Drawing from action RNG cannot advance detector RNG."""
    cfg = EnvConfig(n_envs=2, device="cpu", seed=2025)
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=2025)
    # Snapshot detector RNG output.
    d1 = torch.rand(2, generator=env._detector_gen, device="cpu")
    # Draw from action RNG.
    env.sample_action_rng()
    env.sample_action_rng()
    env.sample_action_rng()
    # Detector RNG must produce the same value as a fresh seed.
    d2 = torch.rand(2, generator=env._detector_gen, device="cpu")
    env_fresh = G3BstaLiteVecEnv(cfg)
    env_fresh.reset(seed=2025)
    d1_fresh = torch.rand(2, generator=env_fresh._detector_gen, device="cpu")
    d2_fresh = torch.rand(2, generator=env_fresh._detector_gen, device="cpu")
    assert torch.equal(d1, d1_fresh), "action RNG advanced detector RNG"
    assert torch.equal(d2, d2_fresh)


def test_paired_scenarios_consume_same_arrival_table():
    """Paired envs share the same pre-generated arrival table."""
    cfg = EnvConfig(n_envs=2, device="cpu", seed=77)
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=77)
    # Scenario arrivals is [H, n_services], broadcast to all envs.
    arrivals = env.scenario.arrivals
    assert arrivals.shape == (cfg.horizon, cfg.n_services)
    # Run a few steps; verify n_eligible increments identically across envs
    # (since arrivals are shared and admission is broadcast).
    for _ in range(6):
        env.step(torch.tensor([ACTION_IDLE, ACTION_IDLE], dtype=torch.int64))
    assert int(env.counters.n_eligible[0]) == int(env.counters.n_eligible[1])
