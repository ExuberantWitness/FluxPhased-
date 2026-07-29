"""Frozen scripted jammer baselines for G3-BSTA-lite (F2).

Per DEBUG_CONTRACT.md §7, six frozen baselines must be implemented and
frozen before Gate 1. All share the same action feasibility and resource
path (the env's mask + cost).

  - always_off: never jam (idle every step)
  - random_feasible: uniform over legal actions
  - budgeted_barrage: jam service 0 whenever legal until energy depletes
  - budgeted_round_robin: alternate services under budget
  - periodic_blink: jam K steps, idle K steps
  - causal_reactive_or_edf: act on delayed urgency (earliest-deadline-first
    proxy via delayed pending count per service)

All baselines are deterministic given (scenario, seed, action RNG seed)
EXCEPT ``random_feasible`` which is stochastic and must be averaged over
frozen action replicates within scenario.
"""

from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    ACTION_JAM_SERVICE_1,
    G3BstaLiteVecEnv,
    N_ACTIONS,
)


class Baseline:
    """Common interface for frozen scripted baselines."""

    name: str = "baseline"
    stochastic: bool = False

    def reset(self, env: G3BstaLiteVecEnv, *, seed: int) -> None:
        self._env = env
        self._seed = int(seed)

    def act(self, obs: torch.Tensor, mask: torch.Tensor, *, step_idx: int) -> torch.Tensor:
        raise NotImplementedError


class AlwaysOff(Baseline):
    name = "always_off"
    stochastic = False

    def act(self, obs, mask, *, step_idx):
        E = mask.shape[0]
        return torch.zeros(E, dtype=torch.int64, device=mask.device)


class RandomFeasible(Baseline):
    name = "random_feasible"
    stochastic = True

    def __init__(self):
        self._gen = None

    def reset(self, env, *, seed):
        super().reset(env, seed=seed)
        self._gen = torch.Generator(device=env.device).manual_seed(int(seed) + 17)

    def act(self, obs, mask, *, step_idx):
        E = mask.shape[0]
        actions = torch.zeros(E, dtype=torch.int64, device=mask.device)
        for e in range(E):
            legal = torch.nonzero(mask[e]).flatten()
            idx = torch.randint(0, legal.numel(), (1,), generator=self._gen).item()
            actions[e] = int(legal[idx].item())
        return actions


class BudgetedBarrage(Baseline):
    """Spend the full energy budget on service 0, then idle."""
    name = "budgeted_barrage"
    stochastic = False

    def act(self, obs, mask, *, step_idx):
        E = mask.shape[0]
        actions = torch.zeros(E, dtype=torch.int64, device=mask.device)
        for e in range(E):
            if bool(mask[e, ACTION_JAM_SERVICE_0]):
                actions[e] = ACTION_JAM_SERVICE_0
            else:
                actions[e] = ACTION_IDLE
        return actions


class BudgetedRoundRobin(Baseline):
    """Alternate service 0/1 whenever jam is legal; else idle."""
    name = "budgeted_round_robin"
    stochastic = False

    def act(self, obs, mask, *, step_idx):
        E = mask.shape[0]
        actions = torch.zeros(E, dtype=torch.int64, device=mask.device)
        target = ACTION_JAM_SERVICE_0 if step_idx % 2 == 0 else ACTION_JAM_SERVICE_1
        alt = ACTION_JAM_SERVICE_1 if step_idx % 2 == 0 else ACTION_JAM_SERVICE_0
        for e in range(E):
            if bool(mask[e, target]):
                actions[e] = target
            elif bool(mask[e, alt]):
                actions[e] = alt
            else:
                actions[e] = ACTION_IDLE
        return actions


class PeriodicBlink(Baseline):
    """Jam matched-service for K steps, then idle K steps."""
    name = "periodic_blink"
    stochastic = False

    def __init__(self, period: int = 4):
        self.period = int(period)

    def act(self, obs, mask, *, step_idx):
        E = mask.shape[0]
        actions = torch.zeros(E, dtype=torch.int64, device=mask.device)
        phase = step_idx % self.period
        jamming = phase < (self.period // 2)
        if not jamming:
            return actions
        target = ACTION_JAM_SERVICE_0 if step_idx % 2 == 0 else ACTION_JAM_SERVICE_1
        for e in range(E):
            if bool(mask[e, target]):
                actions[e] = target
            else:
                actions[e] = ACTION_IDLE
        return actions


class CausalReactiveOrEDF(Baseline):
    """Same-observation executable causal witness (DEBUG_CONTRACT §7).

    Strategy: jam the radar's currently-scanned service (so physics works)
    when delayed urgency on that service is positive (pending missions
    exist that we can still deny scans to), and apply an end-of-horizon
    "use it or lose it" override.

    With F2 obs_delay_steps=0 (causal-information repair per
    MODIFICATION_PLAN route), the witness sees current pending activity
    and can react within the mission tau_window to deny fresh arrivals'
    scans. This unlocks the learnability headroom that obs_delay=2 closed.
    """
    name = "causal_reactive_or_edf"
    stochastic = False

    def __init__(self):
        self._env = None

    def reset(self, env, *, seed):
        super().reset(env, seed=seed)
        self._env = env

    def act(self, obs, mask, *, step_idx):
        # obs layout: rem_E(0), rem_t(1), delayed_detect(2:4), delayed_urgency(4:6),
        #            intercept_conf(6), intercept_age(7), prev_action_onehot(8:11)
        delayed_urg = obs[:, 4:6]  # [E, 2]
        radar_svc = step_idx % 2
        matched_action = ACTION_JAM_SERVICE_0 if radar_svc == 0 else ACTION_JAM_SERVICE_1
        E = obs.shape[0]
        actions = torch.zeros(E, dtype=torch.int64, device=obs.device)

        if self._env is None:
            return actions

        H = self._env.cfg.horizon
        cost = float(self._env.cfg.P_jam_W * self._env.cfg.dt)

        for e in range(E):
            urg_matched = float(delayed_urg[e, radar_svc])
            energy = float(self._env.energy[e])
            steps_left = H - step_idx
            a = ACTION_IDLE
            # Pending missions on matched service → jam to deny scans.
            if urg_matched > 0.0 and bool(mask[e, matched_action]):
                a = matched_action
            # End-of-horizon: use it or lose it.
            if energy > steps_left * cost and bool(mask[e, matched_action]):
                a = matched_action
            actions[e] = a
        return actions


FROZEN_BASELINES = [
    AlwaysOff,
    RandomFeasible,
    BudgetedBarrage,
    BudgetedRoundRobin,
    PeriodicBlink,
    CausalReactiveOrEDF,
]
