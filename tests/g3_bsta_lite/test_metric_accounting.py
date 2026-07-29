"""F1 §6 Gate 0 contract test 7 — metric accounting.

Per DEBUG_CONTRACT.md §8, the accounting identity must hold:

    eligible_mission_arrivals
      = mission_success
      + mission_timeout
      + mission_admission_reject
      + mission_horizon_failure

Event rows (arrival + disposition) must recompute aggregate counters exactly.
"""

from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    EnvConfig,
    G3BstaLiteVecEnv,
    MissionCounterBatch,
)


def test_accounting_identity_holds_at_horizon():
    cfg = EnvConfig(n_envs=4, device="cpu", seed=555, arrival_rate_per_service=0.25)
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=555)
    for t in range(cfg.horizon):
        # Random-ish action schedule, idle on odd steps to ensure detect.
        a = ACTION_IDLE if t % 2 == 1 else ACTION_JAM_SERVICE_0
        if not bool(env._compute_mask()[0, a]):
            a = ACTION_IDLE
        env.step(torch.full((4,), a, dtype=torch.int64))
    res = env.counters.accounting_residual()
    assert torch.equal(res, torch.zeros(4, dtype=torch.int64)), (
        f"residual nonzero: {res.tolist()}"
    )


def test_event_log_recomputes_aggregate_exactly():
    """Track each disposition's counter delta; sum of deltas must match."""
    cfg = EnvConfig(n_envs=1, device="cpu", seed=313, arrival_rate_per_service=0.3)
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=313)
    # Per-disposition running totals accumulated from event deltas.
    delta_totals = {"success": 0, "timeout": 0, "horizon_failure": 0,
                    "admission_reject": 0}

    orig_finalize_step = env.tracker.finalize_step
    orig_finalize_horizon = env.tracker.finalize_horizon

    def traced_finalize_step(*, env_idx, step, counters):
        before_succ = int(counters.n_success[env_idx])
        before_to = int(counters.n_timeout[env_idx])
        orig_finalize_step(env_idx=env_idx, step=step, counters=counters)
        delta_totals["success"] += int(counters.n_success[env_idx]) - before_succ
        delta_totals["timeout"] += int(counters.n_timeout[env_idx]) - before_to

    def traced_finalize_horizon(*, env_idx, counters):
        before = int(counters.n_horizon_failure[env_idx])
        orig_finalize_horizon(env_idx=env_idx, counters=counters)
        delta_totals["horizon_failure"] += int(counters.n_horizon_failure[env_idx]) - before

    env.tracker.finalize_step = traced_finalize_step
    env.tracker.finalize_horizon = traced_finalize_horizon

    for t in range(cfg.horizon):
        env.step(torch.tensor([ACTION_IDLE], dtype=torch.int64))

    assert delta_totals["success"] == int(env.counters.n_success[0])
    assert delta_totals["timeout"] == int(env.counters.n_timeout[0])
    assert delta_totals["horizon_failure"] == int(env.counters.n_horizon_failure[0])
    total = sum(delta_totals.values())
    assert total == int(env.counters.n_eligible[0]), (
        f"event delta total {total} != eligible {int(env.counters.n_eligible[0])}"
    )


def test_drop_ratio_zero_when_all_succeed():
    cfg = EnvConfig(n_envs=1, device="cpu", seed=21, arrival_rate_per_service=0.1)
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=21)
    for _ in range(cfg.horizon):
        env.step(torch.tensor([ACTION_IDLE], dtype=torch.int64))
    # With idle (no jam), p_det ≈ 1, so all missions should succeed.
    assert int(env.counters.n_success[0]) == int(env.counters.n_eligible[0])
    assert float(env.drop_ratio()[0]) == 0.0


def test_drop_ratio_higher_under_matched_jam_than_under_idle():
    """Energy budget prevents all-time jamming, but matched jamming must still
    produce a measurably higher drop ratio than the idle baseline.
    """
    def run(action_strategy):
        cfg = EnvConfig(n_envs=1, device="cpu", seed=22, mission_tau_window=4,
                        arrival_rate_per_service=0.4)
        env = G3BstaLiteVecEnv(cfg)
        env.reset(seed=22)
        env.scenario.arrivals[:] = True
        for t in range(cfg.horizon):
            a = action_strategy(t, env)
            env.step(torch.tensor([a], dtype=torch.int64))
        return float(env.drop_ratio()[0])

    drop_matched = run(lambda t, env:
                       (t % 2) + 1 if bool(env._compute_mask()[0, (t % 2) + 1])
                       else ACTION_IDLE)
    drop_idle = run(lambda t, env: ACTION_IDLE)
    assert drop_matched > drop_idle + 0.10, (
        f"matched jam drop {drop_matched:.3f} not > idle drop {drop_idle:.3f} + 0.10"
    )


def test_drop_ratio_nan_when_zero_eligible():
    """Empty manifest case is NA, not 0.0."""
    counters = MissionCounterBatch.zeros(1, device="cpu")
    counters.n_eligible[0] = 0
    counters.n_success[0] = 0
    counters.n_timeout[0] = 0
    ratio = counters.drop_ratio()
    assert math_is_nan(ratio[0])


def math_is_nan(x: torch.Tensor) -> bool:
    import math
    return bool(math.isnan(float(x)))
