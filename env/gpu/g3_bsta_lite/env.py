"""G3-BSTA-lite causal budgeted environment (F1 main env).

Implements the frozen debug profile from ``docs/g3-bsta-lite/DEBUG_CONTRACT.md``.
This module is the only public entry point of the env package; tests and
baselines import from here.

Design constraints (frozen):

  - action = masked categorical over {0=idle, 1=jam_service_0, 2=jam_service_1}
  - energy is a hard resource: E[t] >= 0 always
  - always-on is infeasible: E0 < P_fixed * dt * horizon
  - 3 RNG streams: environment-event, detector, action (separate torch.Generator)
  - paired scenarios share the same pre-generated arrival table
  - observation is causal (no god-view); privileged facts are critic-only

The env is batched: all per-step state is a tensor of shape [E] or [E, ...]
and all RNG draws are batched so env-0 perturbation cannot affect env-1
(test_vector_isolation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from .action_contract import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    ACTION_JAM_SERVICE_1,
    N_ACTIONS,
    SERVICE_FOR_ACTION,
    ContractViolation,
    TransitionTrace,
)
from .metrics import (
    MissionCounterBatch,
    MissionTracker,
)
from .observation import (
    OBS_DIM,
    PRIVILEGED_DIM,
    build_observation,
    build_privileged,
)
from .physics import (
    DebugPhysicsConfig,
    compute_service_jnr_db,
    default_debug_physics_config,
)
from .radar_opponent import FrozenRuleRadar
from .scenario import Scenario, generate_scenario


@dataclass
class EnvConfig:
    n_envs: int = 16
    horizon: int = 64
    n_services: int = 2
    dt: float = 1.0
    P_jam_W: float = 50.0
    active_budget_steps: int = 16
    duty_budget: float = 0.25
    arrival_rate_per_service: float = 0.15
    baseline_snr_db: float = 22.0
    detect_threshold_db: float = 15.0
    detect_width_db: float = 3.0
    mission_tau_window: int = 6
    detects_required: int = 1
    # F2 repair (MODIFICATION_PLAN route "repair causal information/history"):
    # delay=2 made the same-observation witness unable to demonstrate
    # learnability headroom (it could not react to fresh arrivals within
    # the mission tau_window). delay=0 keeps the channel causal (only
    # past/present observables, no future-leak) while letting the witness
    # react to currently-pending activity. Verified by test_causal_observation.
    obs_delay_steps: int = 0
    obs_ema_alpha: float = 0.5
    potential_coef: float = 0.05
    gamma: float = 0.99
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        # Always-on infeasibility (DEBUG_CONTRACT.md §3).
        cost_per_action = self.P_jam_W * self.dt
        max_actions = self.horizon  # upper bound: jamming every step
        if cost_per_action * max_actions <= 0:
            raise ValueError("cost_per_action * horizon must be > 0")
        # active_budget_steps = duty_budget * horizon (frozen profile).
        # If user-supplied active_budget_steps exceeds this (e.g. short test
        # horizon), clamp to leave at least one idle step so always-on
        # remains infeasible.
        implied = max(1, int(self.duty_budget * self.horizon))
        if self.active_budget_steps > implied:
            self.active_budget_steps = implied
        self.E0 = cost_per_action * float(self.active_budget_steps)
        if self.E0 >= cost_per_action * self.horizon:
            raise ValueError(
                "always-on feasibility: E0 must be < P_fixed*dt*horizon; "
                f"E0={self.E0}, cost*h={cost_per_action * self.horizon}; "
                f"active_budget_steps={self.active_budget_steps}, horizon={self.horizon}"
            )

    def to_json(self) -> dict:
        return {
            "n_envs": self.n_envs,
            "horizon": self.horizon,
            "n_services": self.n_services,
            "dt": self.dt,
            "P_jam_W": self.P_jam_W,
            "active_budget_steps": self.active_budget_steps,
            "duty_budget": self.duty_budget,
            "E0": self.E0,
            "arrival_rate_per_service": self.arrival_rate_per_service,
            "baseline_snr_db": self.baseline_snr_db,
            "detect_threshold_db": self.detect_threshold_db,
            "detect_width_db": self.detect_width_db,
            "mission_tau_window": self.mission_tau_window,
            "detects_required": self.detects_required,
            "obs_delay_steps": self.obs_delay_steps,
            "obs_ema_alpha": self.obs_ema_alpha,
            "potential_coef": self.potential_coef,
            "gamma": self.gamma,
            "device": self.device,
            "seed": self.seed,
        }


class G3BstaLiteVecEnv:
    """Batched causal budgeted environment for the G3-BSTA-lite debug profile."""

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        *,
        physics: Optional[DebugPhysicsConfig] = None,
    ):
        self.cfg = config or EnvConfig()
        self.physics = physics or default_debug_physics_config(P_jam_W=self.cfg.P_jam_W)
        self.E = self.cfg.n_envs
        self.H = self.cfg.horizon
        self.n_services = self.cfg.n_services
        self.device = self.cfg.device

        # Radar opponent (deterministic, no RNG).
        self.radar = FrozenRuleRadar(n_envs=self.E, n_services=self.n_services,
                                     device=self.device)

        # Three RNG streams — must be initialized fresh on each reset(seed=...).
        self._event_gen = torch.Generator(device=self.device)
        self._detector_gen = torch.Generator(device=self.device)
        self._action_gen = torch.Generator(device=self.device)

        self._scenario_seed: Optional[int] = None
        self.scenario: Optional[Scenario] = None

        # State (lazy-init on reset)
        self.energy: Optional[torch.Tensor] = None
        self.step_idx: int = 0
        self.prev_action: Optional[torch.Tensor] = None
        self.tracker: Optional[MissionTracker] = None
        self.counters: Optional[MissionCounterBatch] = None
        self._obs_state_version: int = 0

        # EMA-smoothed delayed detection and urgency per [E, n_services].
        self._delayed_detect_ema: Optional[torch.Tensor] = None
        self._delayed_urgency_ema: Optional[torch.Tensor] = None
        # Detection outcome ring buffer (per service, per step) for delay.
        self._detect_history: list[torch.Tensor] = []
        # Pending-count ring buffer (per service, per step) for delay.
        self._urgency_history: list[torch.Tensor] = []
        # Intercept confidence + age (jammer-side estimator).
        self.intercept_confidence: Optional[torch.Tensor] = None
        self.intercept_age: Optional[torch.Tensor] = None

        # Reward components log (raw, shaping, total).
        self._reward_raw_log: list[torch.Tensor] = []
        self._reward_shaping_log: list[torch.Tensor] = []
        self._trace_log: list[TransitionTrace] = []

    # ------------------------------------------------------------------
    # RNG management
    # ------------------------------------------------------------------
    def _reset_rng(self, *, seed: int):
        self._event_gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self._detector_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        self._action_gen = torch.Generator(device=self.device).manual_seed(int(seed) + 2)

    # ------------------------------------------------------------------
    # Scenario
    # ------------------------------------------------------------------
    def _ensure_scenario(self, *, seed: int):
        if self._scenario_seed == seed and self.scenario is not None:
            return
        self.scenario = generate_scenario(
            seed=seed,
            horizon=self.H,
            n_services=self.n_services,
            arrival_rate_per_service=self.cfg.arrival_rate_per_service,
            baseline_snr_db=self.cfg.baseline_snr_db,
            device=self.device,
        ).to(self.device)
        self._scenario_seed = seed

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        reset_metrics: bool = True,
    ) -> torch.Tensor:
        """Reset all envs. Returns [E, OBS_DIM] observation."""
        if seed is None:
            seed = int(self.cfg.seed) + self._episode_counter() * 1009
        self._reset_rng(seed=seed)
        self._ensure_scenario(seed=seed)

        self.energy = torch.full(
            (self.E,), float(self.cfg.E0), device=self.device, dtype=torch.float32
        )
        self.step_idx = 0
        self.prev_action = torch.zeros(self.E, dtype=torch.int64, device=self.device)
        self._obs_state_version = 0

        self.tracker = MissionTracker(
            n_envs=self.E,
            n_services=self.n_services,
            detects_required=self.cfg.detects_required,
        )
        self.tracker.initialize()
        if reset_metrics or self.counters is None:
            self.counters = MissionCounterBatch.zeros(self.E, device=self.device)

        # Init EMAs to neutral values.
        self._delayed_detect_ema = torch.zeros(self.E, self.n_services,
                                               device=self.device, dtype=torch.float32)
        self._delayed_urgency_ema = torch.zeros(self.E, self.n_services,
                                                device=self.device, dtype=torch.float32)
        self._detect_history = []
        self._urgency_history = []
        self.intercept_confidence = torch.zeros(self.E, device=self.device,
                                                dtype=torch.float32)
        self.intercept_age = torch.full((self.E,), float(self.H),
                                        device=self.device, dtype=torch.float32)
        self._reward_raw_log = []
        self._reward_shaping_log = []
        self._trace_log = []

        return self._build_observation()

    def _episode_counter(self) -> int:
        return getattr(self, "_episode_idx", 0)

    # ------------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------------
    def _compute_mask(self) -> torch.Tensor:
        """[E, N_ACTIONS] legal-action mask.

        idle is always legal. Service k is legal iff the env has enough
        energy for one more P_fixed*dt deduction. The mask never reveals
        current channel activity or hidden value.
        """
        cost = float(self.cfg.P_jam_W * self.cfg.dt)
        enough = self.energy >= cost
        mask = torch.zeros(self.E, N_ACTIONS, dtype=torch.bool, device=self.device)
        mask[:, ACTION_IDLE] = True
        mask[:, ACTION_JAM_SERVICE_0] = enough
        mask[:, ACTION_JAM_SERVICE_1] = enough
        return mask

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _delayed_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return delayed detect / urgency tensors.

        Delay = obs_delay_steps. If history shorter than delay, return zeros
        (the policy cannot observe what has not yet happened).
        """
        d = self.cfg.obs_delay_steps
        if len(self._detect_history) <= d:
            detect = torch.zeros(self.E, self.n_services,
                                 device=self.device, dtype=torch.float32)
        else:
            detect = self._detect_history[-1 - d]
        if len(self._urgency_history) <= d:
            urgency = torch.zeros(self.E, self.n_services,
                                  device=self.device, dtype=torch.float32)
        else:
            urgency = self._urgency_history[-1 - d]
        return detect, urgency

    def _build_observation(self) -> torch.Tensor:
        detect, urgency = self._delayed_obs()
        prev_oh = torch.nn.functional.one_hot(self.prev_action, N_ACTIONS).float()
        return build_observation(
            energy=self.energy,
            initial_energy=torch.full_like(self.energy, self.cfg.E0),
            step_idx=self.step_idx,
            horizon=self.H,
            delayed_detect=detect,
            delayed_urgency=urgency,
            intercept_confidence=self.intercept_confidence,
            intercept_age=self.intercept_age,
            prev_action_onehot=prev_oh,
        )

    def _build_privileged(self) -> torch.Tensor:
        pending = torch.zeros(self.E, self.n_services, dtype=torch.int64,
                              device=self.device)
        for e in range(self.E):
            pending[e] = self.tracker.pending_count_per_service(e)
        # Track health proxy: fraction of pending with detects_so_far >= 1.
        health = torch.zeros(self.E, self.n_services, dtype=torch.float32,
                             device=self.device)
        for e in range(self.E):
            for svc in range(self.n_services):
                items = [m for m in self.tracker.pending[e] if m[0] == svc]
                if not items:
                    continue
                with_detect = sum(1 for m in items if m[3] >= 1)
                health[e, svc] = float(with_detect) / float(len(items))
        return build_privileged(pending_per_service=pending,
                                track_health_per_service=health)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _potential(self) -> torch.Tensor:
        """Phi(belief) = -potential_coef * pending_count_total."""
        pending = torch.zeros(self.E, dtype=torch.float32, device=self.device)
        for e in range(self.E):
            pending[e] = float(self.tracker.pending_count(e))
        return -float(self.cfg.potential_coef) * pending

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict
    ]:
        """Apply one decision step.

        Args:
          actions: [E] int64 in {0, 1, 2}

        Returns:
          obs: [E, OBS_DIM]
          reward: [E] float32 (team-shared; jammer team gets +1 per drop)
          done: [E] bool
          info: dict with mask, trace, raw_drop, shaping
        """
        if self.energy is None:
            raise RuntimeError("call reset() before step()")
        if actions.shape != (self.E,):
            raise ValueError(f"actions shape {actions.shape}; expected ({self.E},)")
        if actions.dtype != torch.int64:
            raise TypeError(f"actions dtype {actions.dtype}; expected int64")

        # ---- 1. Compute mask and verify legality ------------------------
        mask = self._compute_mask()
        legal_per_env = mask.gather(1, actions.unsqueeze(1)).squeeze(1)
        if not bool(legal_per_env.all()):
            bad = torch.nonzero(~legal_per_env).flatten().tolist()
            # Per contract: explicit violation, never silent substitution.
            # Do not advance the env state when this is raised.
            raise ContractViolation(
                f"illegal action for envs {bad}; mask was\n{mask}\nactions={actions}"
            )

        # ---- 2. Snapshot pre-step observation_state_version -------------
        obs_version_before = self._obs_state_version

        # ---- 3. Apply executed action to jammer state -------------------
        energy_before = self.energy.clone()
        is_jam = (actions != ACTION_IDLE)
        selected_service = torch.where(
            is_jam, actions - 1,
            torch.full_like(actions, -1),
        )
        cost = float(self.cfg.P_jam_W * self.cfg.dt)
        self.energy = self.energy - is_jam.float() * cost
        # Clamp to non-negative (defense; mask should already prevent negatives).
        self.energy = self.energy.clamp(min=0.0)

        # ---- 4. Apply exogenous arrivals for this step ------------------
        arrivals_t = self.scenario.arrivals[self.step_idx]   # [n_services] bool
        for svc in range(self.n_services):
            if bool(arrivals_t[svc]):
                for e in range(self.E):
                    deadline = self.step_idx + self.cfg.mission_tau_window
                    self.tracker.admit(
                        env_idx=e, step=self.step_idx, service_id=svc,
                        deadline_step=min(deadline, self.H - 1),
                    )
                    self.counters.n_eligible[e] += 1

        # ---- 5. Radar opponent acts (deterministic service pick) -------
        radar_svc = self.radar.service_at_step(self.step_idx)

        # ---- 6. Detection: per-env JNR + Bernoulli draw ----------------
        # Batched JNR per env at the radar's current service.
        jnr_db_per_env = torch.full(
            (self.E,), float("-inf"), device=self.device, dtype=torch.float32,
        )
        for e in range(self.E):
            if bool(is_jam[e]):
                js = int(selected_service[e].item())
                jnr_db = compute_service_jnr_db(
                    self.physics,
                    jammer_active=True,
                    jammer_service_id=js,
                    victim_service_id=radar_svc,
                )
                jnr_db_per_env[e] = float(jnr_db)

        # Batched detection probability.
        baseline_snr = float(self.scenario.baseline_snr_db[radar_svc].item())
        jnr_lin = torch.where(
            torch.isfinite(jnr_db_per_env),
            10.0 ** (jnr_db_per_env / 10.0),
            torch.zeros_like(jnr_db_per_env),
        )
        snr_lin = 10.0 ** (baseline_snr / 10.0)
        snr_eff_lin = snr_lin / (1.0 + jnr_lin)
        snr_eff_db = 10.0 * torch.log10(snr_eff_lin.clamp(min=1e-12))
        snr_eff_db = snr_eff_db + self.physics.coherent_gain_db
        p_det = torch.sigmoid(
            (snr_eff_db - self.cfg.detect_threshold_db) / self.cfg.detect_width_db
        )

        # Batched detector RNG draw (vector-isolation safe).
        det_draw = torch.rand(self.E, generator=self._detector_gen, device=self.device)
        detected = det_draw < p_det

        # Apply detection to pending missions on radar_svc.
        for e in range(self.E):
            self.tracker.record_detection(
                env_idx=e, step=self.step_idx,
                service_id=radar_svc, detected=bool(detected[e]),
            )

        # ---- 7. Update delayed-observation EMAs -------------------------
        # Push current step's detect outcome to history.
        detect_row = torch.zeros(self.E, self.n_services,
                                 device=self.device, dtype=torch.float32)
        detect_row[:, radar_svc] = detected.float()
        self._detect_history.append(detect_row)

        urgency_now = torch.zeros(self.E, self.n_services,
                                  device=self.device, dtype=torch.float32)
        for e in range(self.E):
            for svc in range(self.n_services):
                n_pending = sum(1 for m in self.tracker.pending[e] if m[0] == svc)
                # Urgency normalized to [0, 1] by tau_window.
                urgency_now[e, svc] = float(n_pending) / float(self.cfg.mission_tau_window)
        self._urgency_history.append(urgency_now)

        # ---- 8. Intercept confidence + age ------------------------------
        # Confidence goes up when a service has many pending and few detects.
        # Age increments each step; resets on a successful emission-driven
        # detect drop (i.e., when p_det was depressed by our jammer).
        new_conf = torch.zeros(self.E, device=self.device, dtype=torch.float32)
        for e in range(self.E):
            if bool(is_jam[e]):
                # If we jammed and the radar scanned a different service,
                # we learn nothing new (no intercept). If same service,
                # the detection outcome (1 - p_det) reflects our effect.
                js = int(selected_service[e].item())
                if js == radar_svc:
                    new_conf[e] = 1.0 - float(p_det[e])
                else:
                    new_conf[e] = float(self.intercept_confidence[e])
            else:
                new_conf[e] = 0.0  # idle gives no intercept info
        self.intercept_confidence = (
            self.cfg.obs_ema_alpha * new_conf
            + (1.0 - self.cfg.obs_ema_alpha) * self.intercept_confidence
        )
        # Age: steps since last emission.
        new_age = torch.where(
            is_jam,
            torch.zeros_like(self.intercept_age),
            self.intercept_age + 1.0,
        )
        self.intercept_age = new_age

        # ---- 9. Deadline expiry (finalize this step's drops) ------------
        newly_dropped = torch.zeros(self.E, dtype=torch.int64, device=self.device)
        newly_succeeded = torch.zeros(self.E, dtype=torch.int64, device=self.device)
        success_before = self.counters.n_success.clone()
        timeout_before = self.counters.n_timeout.clone()
        for e in range(self.E):
            self.tracker.finalize_step(env_idx=e, step=self.step_idx, counters=self.counters)
            newly_succeeded[e] = self.counters.n_success[e] - success_before[e]
            newly_dropped[e] = self.counters.n_timeout[e] - timeout_before[e]

        # ---- 10. Reward: raw drop + potential-based shaping ------------
        phi_before = self._potential()
        self.step_idx += 1
        self._obs_state_version += 1
        phi_after = self._potential()
        raw_reward = newly_dropped.float()
        shaping = self.cfg.gamma * phi_after - phi_before
        reward = raw_reward + shaping

        self._reward_raw_log.append(raw_reward)
        self._reward_shaping_log.append(shaping)

        # ---- 11. Build trace and new observation -----------------------
        prev_oh = torch.nn.functional.one_hot(self.prev_action, N_ACTIONS).float()
        trace = TransitionTrace(
            observation_state_version=obs_version_before,
            mask=mask,
            requested_action=actions.clone(),
            executed_action=actions.clone(),
            selected_service=selected_service,
            energy_before=energy_before,
            energy_after=self.energy.clone(),
            legal=legal_per_env,
        )
        self._trace_log.append(trace)
        self.prev_action = actions.clone()

        obs = self._build_observation()
        done = torch.full((self.E,), self.step_idx >= self.H,
                          dtype=torch.bool, device=self.device)
        if bool(done.all()):
            for e in range(self.E):
                self.tracker.finalize_horizon(env_idx=e, counters=self.counters)

        info = {
            "trace": trace,
            "raw_drop": newly_dropped,
            "newly_succeeded": newly_succeeded,
            "shaping": shaping,
            "potential_before": phi_before,
            "potential_after": phi_after,
            "p_detect": p_det,
            "jnr_db": jnr_db_per_env,
            "mask": mask,
            "radar_service": int(radar_svc),
            "step_idx": self.step_idx - 1,
        }
        self._episode_idx = self._episode_counter() + 1
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def drop_ratio(self) -> torch.Tensor:
        return self.counters.drop_ratio()

    def accounting_residual(self) -> torch.Tensor:
        return self.counters.accounting_residual()

    def privileged(self) -> torch.Tensor:
        return self._build_privileged()

    def sample_action_rng(self) -> torch.Tensor:
        """[E] uniform int in [0, N_ACTIONS) using the dedicated action RNG."""
        return torch.randint(
            0, N_ACTIONS, (self.E,), generator=self._action_gen, device=self.device
        )
