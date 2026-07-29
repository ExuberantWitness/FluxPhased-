"""Executable clairvoyant oracle for G3-BSTA-lite (F2 §7).

Per DEBUG_CONTRACT.md §7, the oracle is an *executable* policy on the
reduced H=64 debug problem, used to establish reachable headroom. It is
NOT a certified admissible upper bound (that would be a separate concept,
diagnostic-only).

The oracle has full foreknowledge of the arrivals table. It solves a
greedy scheduling problem: at each step, decide whether to jam service 0,
service 1, or idle, in order to maximize the expected number of mission
drops subject to the energy budget.

Algorithm (exact greedy with one-step lookahead):

  At each step t:
    - radar_svc = t % 2 (deterministic)
    - For each candidate action a in {idle, jam_0, jam_1}:
        - if a != idle and E < cost: skip (infeasible)
        - Estimate immediate expected drop gain:
            - drop_gain(a, t) = number of pending missions on radar_svc
              with detects_so_far < detects_required AND deadline - t <= 1
              (i.e., would succeed this step if detected, would drop if not)
            - if a jams radar_svc: jammer suppresses detection → all those
              missions drop this step (gain = count × 1)
            - else: detection proceeds, missions may succeed (gain = 0)
    - Pick a maximizing immediate drop_gain; tie-break by energy conservation
      (prefer idle when drop_gain = 0).

This is "clairvoyant" in that it knows the arrivals and the pending queue's
deadline structure exactly, but it is NOT optimal globally — it does not
plan multi-step lookahead. The reachable headroom is therefore a lower
bound on the truly-optimal policy.

For an H=64 horizon, an exact DP over (step, energy, pending_state) is
intractable (state space blows up). The greedy clairvoyant is the
practical executable oracle.
"""

from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    ACTION_JAM_SERVICE_1,
    G3BstaLiteVecEnv,
)


class ClairvoyantGreedyOracle:
    """Executable clairvoyant greedy policy with full arrivals foreknowledge.

    Decision rule per step:
      1. Determine radar_svc = step_idx % 2.
      2. Among pending missions on radar_svc with detects_so_far < required
         and deadline == step_idx + 1 (about to succeed-or-drop next):
           - if we jam radar_svc this step, the radar's detection this step
             is suppressed, so those missions drop instead of succeeding.
           - gain = number of such pending missions.
      3. If gain > 0 and energy allows: jam radar_svc.
         Else if energy allows and there's pending on radar_svc with
         deadline > step_idx + 1: jam to deny accumulation (heuristic).
         Else: idle (save energy).
    """

    name = "clairvoyant_greedy_oracle"
    stochastic = False

    def __init__(self):
        self._env: G3BstaLiteVecEnv | None = None

    def reset(self, env: G3BstaLiteVecEnv, *, seed: int) -> None:
        self._env = env
        self._seed = int(seed)

    def act(self, obs: torch.Tensor, mask: torch.Tensor, *, step_idx: int) -> torch.Tensor:
        env = self._env
        E = mask.shape[0]
        actions = torch.zeros(E, dtype=torch.int64, device=mask.device)
        radar_svc = step_idx % 2
        cost = float(env.cfg.P_jam_W * env.cfg.dt)
        jam_radar_svc = ACTION_JAM_SERVICE_0 if radar_svc == 0 else ACTION_JAM_SERVICE_1

        for e in range(E):
            # Score pending missions on radar_svc that would succeed if not jammed.
            pending_on_radar = [
                m for m in env.tracker.pending[e]
                if m[0] == radar_svc and m[3] < env.cfg.detects_required
            ]
            # About-to-succeed set: deadline == step_idx + 1 means this is the
            # last chance for the radar to detect before finalize_step fires.
            about_to_succeed = [
                m for m in pending_on_radar if m[2] == step_idx + 1
            ]
            # Immediate gain from jamming radar_svc now.
            gain_immediate = len(about_to_succeed)

            # Future contributions: pending missions on radar_svc with
            # deadline > step_idx + 1. Jamming now denies one detect credit;
            # if detects_required=1, this could be valuable but the mission
            # still has time to recover. Use a fractional weight.
            future_pending = [
                m for m in pending_on_radar if m[2] > step_idx + 1
            ]
            gain_future = 0.25 * float(len(future_pending))

            total_gain = gain_immediate + gain_future
            expected_drops_per_energy = total_gain / cost if cost > 0 else 0.0

            if total_gain > 0 and bool(mask[e, jam_radar_svc]):
                actions[e] = jam_radar_svc
            else:
                actions[e] = ACTION_IDLE
        return actions


class ClairvoyantOptimalOracle:
    """Same as greedy but with a one-step lookahead on remaining budget.

    Decides whether to spend energy THIS step or save for future step with
    higher expected drop. Uses a simple threshold: jam if immediate gain >= 1
    OR if remaining budget is "use it or lose it" at end of horizon.
    """

    name = "clairvoyant_optimal_oracle"
    stochastic = False

    def __init__(self):
        self._env = None
        self._greedy = ClairvoyantGreedyOracle()

    def reset(self, env, *, seed):
        self._env = env
        self._greedy.reset(env, seed=seed)
        self._seed = int(seed)

    def act(self, obs, mask, *, step_idx):
        env = self._env
        E = mask.shape[0]
        greedy_actions = self._greedy.act(obs, mask, step_idx=step_idx)
        actions = greedy_actions.clone()

        # End-of-horizon "use it or lose it": if we have N steps left and
        # N*cost <= remaining energy, we MUST spend to avoid wasting budget.
        # For each env with idle action, check if energy > steps_left * cost.
        for e in range(E):
            if int(actions[e]) != ACTION_IDLE:
                continue
            steps_left = env.cfg.horizon - step_idx
            energy = float(env.energy[e])
            if energy > steps_left * float(env.cfg.P_jam_W * env.cfg.dt):
                # Would waste energy; jam radar_svc if legal.
                radar_svc = step_idx % 2
                jam = ACTION_JAM_SERVICE_0 if radar_svc == 0 else ACTION_JAM_SERVICE_1
                if bool(mask[e, jam]):
                    actions[e] = jam
        return actions


def make_clairvoyant_oracle() -> ClairvoyantOptimalOracle:
    """Default executable oracle used for Gate 1 headroom assessment."""
    return ClairvoyantOptimalOracle()
