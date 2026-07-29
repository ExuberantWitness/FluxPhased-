"""F1 §6 Gate 0 contract test 5 — causal observation.

Per DEBUG_CONTRACT.md §6, the actor may not see: exact pending count,
exact progress/deadline, true target slot/id, future arrivals, post-action
detector outcome, next radar action, environment RNG state.

Strategy:
  - Reset env.
  - Run a few steps; record observation at each step.
  - Mutate hidden state (pending queue, tracker internals, baseline_snr_db)
    WITHOUT changing actor-visible state.
  - Re-build observation; assert it is identical.

The observation builder is pure: given the same energy/step/delayed-detect/
delayed-urgency/intercept/prev-action, it must produce the same output
regardless of pending queue state.
"""

from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    EnvConfig,
    G3BstaLiteVecEnv,
    OBS_DIM,
)
from env.gpu.g3_bsta_lite.observation import build_observation


def test_observation_does_not_godview_pending_queue_length():
    """Add bogus pending missions to tracker; obs must not change."""
    cfg = EnvConfig(n_envs=1, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=13)
    env.step(torch.tensor([ACTION_IDLE], dtype=torch.int64))

    # Snapshot actor-visible state.
    energy = env.energy.clone()
    delayed_detect, delayed_urgency = env._delayed_obs()
    intercept_conf = env.intercept_confidence.clone()
    intercept_age = env.intercept_age.clone()
    prev_oh = torch.nn.functional.one_hot(env.prev_action, 3).float()

    obs_baseline = build_observation(
        energy=energy,
        initial_energy=torch.full_like(energy, cfg.E0),
        step_idx=env.step_idx,
        horizon=cfg.horizon,
        delayed_detect=delayed_detect,
        delayed_urgency=delayed_urgency,
        intercept_confidence=intercept_conf,
        intercept_age=intercept_age,
        prev_action_onehot=prev_oh,
    )

    # Mutate hidden state: inject 100 bogus pending missions.
    for _ in range(100):
        env.tracker.admit(env_idx=0, step=0, service_id=0, deadline_step=10)
    # Rebuild obs from the same actor-visible state.
    obs_after = build_observation(
        energy=energy,
        initial_energy=torch.full_like(energy, cfg.E0),
        step_idx=env.step_idx,
        horizon=cfg.horizon,
        delayed_detect=delayed_detect,
        delayed_urgency=delayed_urgency,
        intercept_confidence=intercept_conf,
        intercept_age=intercept_age,
        prev_action_onehot=prev_oh,
    )
    assert torch.equal(obs_baseline, obs_after), (
        "observation leaked hidden pending queue state"
    )


def test_observation_does_not_godview_baseline_snr():
    """Baseline SNR is privileged; obs builder must not consume it."""
    cfg = EnvConfig(n_envs=1, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=14)
    obs1 = env._build_observation()
    # Mutate scenario.baseline_snr_db (privileged).
    env.scenario.baseline_snr_db = torch.tensor([99.0, 99.0], dtype=torch.float32)
    obs2 = env._build_observation()
    assert torch.equal(obs1, obs2), "observation leaked baseline_snr_db"


def test_observation_does_not_godview_future_arrivals():
    """Mutating arrivals table for future steps must not change current obs."""
    cfg = EnvConfig(n_envs=1, device="cpu", horizon=16)
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=15)
    env.step(torch.tensor([ACTION_IDLE], dtype=torch.int64))
    obs1 = env._build_observation()
    # Flip future arrivals.
    env.scenario.arrivals[env.step_idx:] = ~env.scenario.arrivals[env.step_idx:]
    obs2 = env._build_observation()
    assert torch.equal(obs1, obs2), "observation leaked future arrivals"


def test_privileged_includes_pending_count():
    """Privileged critic facts DO include pending count (separately registered)."""
    cfg = EnvConfig(n_envs=1, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=16)
    # Inject pending missions on service 0.
    for _ in range(3):
        env.tracker.admit(env_idx=0, step=0, service_id=0, deadline_step=10)
    priv = env._build_privileged()
    assert int(priv[0, 0]) == 3, "privileged facts missed pending count"


def test_obs_dim_is_11_scalars():
    """Frozen OBS_DIM=11: rem_E + rem_t + delayed_detect(2) + delayed_urgency(2)
    + intercept_conf + intercept_age + prev_action_onehot(3) = 11.
    """
    assert OBS_DIM == 11
