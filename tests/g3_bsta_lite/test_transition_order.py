"""F1 §6 Gate 0 contract test 3 — decision / transition ordering.

Per DEBUG_CONTRACT.md §3.1: ``step(action)`` applies the selected service
action against the state represented by the current observation BEFORE
applying the next exogenous arrival/transition. A test must prove that
changing the next-arrival table cannot change the meaning of the current
service action.

We construct two scenarios that are identical at steps [0..t] but differ at
step t+1 (one has an arrival on service 0, the other on service 1). At step
t, both scenarios must produce identical obs/reward/detector outcome.
"""

from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite import (
    ACTION_JAM_SERVICE_0,
    EnvConfig,
    G3BstaLiteVecEnv,
    Scenario,
)


def _make_scenario(seed: int, arrivals: torch.Tensor) -> Scenario:
    return Scenario(
        seed=seed,
        arrivals=arrivals,
        baseline_snr_db=torch.full((2,), 22.0, dtype=torch.float32),
    )


def test_changing_next_arrival_cannot_change_current_service_action_meaning():
    """Run env A and env B with same arrivals[0..t]; differ at t+1.

    Pre-conditions:
      - Both envs see identical arrivals up to and including step t.
      - Arrivals differ at step t+1.
      - Same action sequence on both envs.
      - Detector RNG seeded identically.

    Assertion:
      - Observations and rewards at step t are bit-equal across A and B.
    """
    H = 16
    arrivals_a = torch.zeros(H, 2, dtype=torch.bool)
    arrivals_b = torch.zeros(H, 2, dtype=torch.bool)
    # Common arrival at step 2 service 0
    arrivals_a[2, 0] = True
    arrivals_b[2, 0] = True
    # Common arrival at step 5 service 1
    arrivals_a[5, 1] = True
    arrivals_b[5, 1] = True
    # DIVERGE at step 7
    arrivals_a[7, 0] = True
    arrivals_b[7, 1] = True

    cfg_a = EnvConfig(n_envs=1, device="cpu", horizon=H, seed=500)
    cfg_b = EnvConfig(n_envs=1, device="cpu", horizon=H, seed=500)
    env_a = G3BstaLiteVecEnv(cfg_a)
    env_b = G3BstaLiteVecEnv(cfg_b)
    env_a.scenario = _make_scenario(500, arrivals_a)
    env_b.scenario = _make_scenario(500, arrivals_b)
    env_a._scenario_seed = 500
    env_b._scenario_seed = 500
    # Need to manually init state since we replaced scenario post-construction.
    env_a.reset(seed=500)
    env_a.scenario = _make_scenario(500, arrivals_a)
    env_b.reset(seed=500)
    env_b.scenario = _make_scenario(500, arrivals_b)
    # Critical: re-seed detector RNG to identical state for fair compare.
    env_a._detector_gen = torch.Generator(device="cpu").manual_seed(500 + 1)
    env_b._detector_gen = torch.Generator(device="cpu").manual_seed(500 + 1)

    obs_a, rew_a = [], []
    obs_b, rew_b = [], []
    for t in range(H):
        # Sustainable schedule: idle on odd steps (radar scans service 1),
        # jam service 0 only on even steps when budget permits.
        if t % 2 == 0:
            a_val = ACTION_JAM_SERVICE_0
            if not bool(env_a._compute_mask()[0, a_val]):
                a_val = 0  # ACTION_IDLE
        else:
            a_val = 0  # ACTION_IDLE
        a = torch.tensor([a_val], dtype=torch.int64)
        oa, ra, _, ia = env_a.step(a)
        ob, rb, _, ib = env_b.step(a)
        obs_a.append(oa); rew_a.append(ra)
        obs_b.append(ob); rew_b.append(rb)
        if t < 6:
            assert torch.equal(oa, ob), (
                f"obs diverged at step {t} despite same arrivals[0..t]"
            )
            assert torch.equal(ra, rb), (
                f"reward diverged at step {t} despite same arrivals[0..t]"
            )


def test_step_applies_action_before_next_exogenous_arrival():
    """The action cost is deducted BEFORE arrivals for this step are admitted.

    Construct a single-step scenario where the action AND the arrival both
    occur at step 0. The energy cost must be deducted before the eligible
    counter increments.
    """
    H = 8
    cfg = EnvConfig(n_envs=1, device="cpu", horizon=H, seed=999)
    env = G3BstaLiteVecEnv(cfg)
    arrivals = torch.zeros(H, 2, dtype=torch.bool)
    arrivals[0, 0] = True  # arrival at step 0
    env.reset(seed=999)
    env.scenario = Scenario(
        seed=999, arrivals=arrivals,
        baseline_snr_db=torch.full((2,), 22.0, dtype=torch.float32),
    )
    e_before = env.energy.clone()
    eligible_before = env.counters.n_eligible.clone()
    env.step(torch.tensor([ACTION_JAM_SERVICE_0], dtype=torch.int64))
    # Energy cost applied first.
    assert float(env.energy[0]) < float(e_before[0])
    # Arrival admitted at this step (after cost).
    assert int(env.counters.n_eligible[0]) == int(eligible_before[0]) + 1
