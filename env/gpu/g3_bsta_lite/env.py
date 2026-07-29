"""G3-BSTA-lite causal budgeted environment (F1 main env, R1 repair).

Implements the frozen debug profile from ``docs/g3-bsta-lite/DEBUG_CONTRACT.md``
plus the learning-repair R1 fixes bound by PREREGISTRATION.md and
POST_AUDIT_CORRECTION.md. This module is the only public entry point of
the env package; tests and baselines import from here.

Design constraints (frozen + R1 repair):

  - action = masked categorical over {0=idle, 1=jam_service_0, 2=jam_service_1}
  - energy is a hard resource: E[t] >= 0 always; integer tokens (R1D)
  - always-on is infeasible: E0 < P_fixed * dt * horizon
  - 3 RNG streams: environment-event, detector, action (separate torch.Generator)
  - paired scenarios share the same pre-generated arrival table
  - observation is profile-dependent (R1B):
      * ``mdp_sanity_v1`` — fully observed MDP (exact pending, radar service)
      * ``pomdp_v1``      — genuine POMDP (delay>=1, non-invertible urgency
                            proxy, hidden radar phase)
  - privileged facts (true pending count, true track health) are critic-only
  - potential-based shaping uses the telescoping form (R1C):
        shaping_t = gamma * Phi(s_{t+1}) - Phi(s_t)
    where Phi(s_t) is captured BEFORE any transition touches the tracker
    and Phi(s_{t+1}) is captured AFTER all transition effects.
  - per-mission event ledger records (env_idx, arrival_step, service_id,
    deadline_step) -> disposition (R1D); the accounting identity is
    eligible = success + timeout + admission_reject + horizon_failure.

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
    DISPO_ADMISSION_REJECT,
    DISPO_HORIZON_FAILURE,
    DISPO_SUCCESS,
    DISPO_TIMEOUT,
    MissionCounterBatch,
    MissionTracker,
)
from .observation import (
    OBS_DIM,
    PRIVILEGED_DIM,
    PROFILE_MDP_SANITY,
    PROFILE_POMDP,
    PROFILES,
    build_observation,
    build_privileged,
    pomdp_urgency_proxy,
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
    # R1B: profile determines observation semantics. Default = pomdp_v1.
    # mdp_sanity_v1 is the fully-observed sanity profile, NOT a POMDP claim.
    profile: str = PROFILE_POMDP
    # R1B: pomdp_v1 requires delay >= 1 (no invertible leak of pending count);
    # mdp_sanity_v1 may use delay=0 because the obs already contains the
    # exact pending count.
    obs_delay_steps: int = 1
    obs_ema_alpha: float = 0.5
    potential_coef: float = 0.05
    gamma: float = 0.99
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(
                f"unknown profile {self.profile!r}; expected one of {PROFILES}"
            )
        # R1B: pomdp_v1 requires delay >= 1 to plug the invertible-leak path.
        if self.profile == PROFILE_POMDP and int(self.obs_delay_steps) < 1:
            raise ValueError(
                "pomdp_v1 profile requires obs_delay_steps >= 1 to avoid "
                "the invertible pending-count leak (see POST_AUDIT_CORRECTION.md §4.1)"
            )
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
        # E0 expressed in integer tokens; one token = one jam step.
        # The mask is computed from energy_tokens >= 1, so float rounding
        # cannot flip the legal-action set (POST_AUDIT_CORRECTION.md §4.6).
        self.E0 = float(cost_per_action * float(self.active_budget_steps))
        self.E0_tokens = int(self.active_budget_steps)
        self.cost_per_action_tokens = 1
        if self.E0_tokens >= self.horizon:
            raise ValueError(
                "always-on feasibility: E0_tokens must be < horizon; "
                f"E0_tokens={self.E0_tokens}, horizon={self.horizon}"
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
            "E0_tokens": self.E0_tokens,
            "arrival_rate_per_service": self.arrival_rate_per_service,
            "baseline_snr_db": self.baseline_snr_db,
            "detect_threshold_db": self.detect_threshold_db,
            "detect_width_db": self.detect_width_db,
            "mission_tau_window": self.mission_tau_window,
            "detects_required": self.detects_required,
            "profile": self.profile,
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
        self.profile = self.cfg.profile

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
        # R1D: energy_tokens is the canonical resource; the float self.energy
        # is a derived display quantity (energy_tokens * cost_per_action).
        self.energy_tokens: Optional[torch.Tensor] = None
        self.energy: Optional[torch.Tensor] = None
        self.step_idx: int = 0
        self.prev_action: Optional[torch.Tensor] = None
        self.tracker: Optional[MissionTracker] = None
        self.counters: Optional[MissionCounterBatch] = None
        self._obs_state_version: int = 0
        self._done_flag: bool = False  # R1D: track episode end for step-after-done

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

        # R1D: per-mission event ledger with stable identity.
        # Key: (env_idx, arrival_step, service_id); the same key is reused
        # whenever the mission's disposition is finalized, so the accounting
        # identity can be re-derived from the ledger alone.
        self.event_ledger: dict[tuple[int, int, int], dict] = {}

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

        self.energy_tokens = torch.full(
            (self.E,), int(self.cfg.E0_tokens), device=self.device, dtype=torch.int64
        )
        self.energy = self.energy_tokens.float() * float(self.cfg.P_jam_W * self.cfg.dt)
        self.step_idx = 0
        self.prev_action = torch.zeros(self.E, dtype=torch.int64, device=self.device)
        self._obs_state_version = 0
        self._done_flag = False

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
        self.event_ledger = {}

        return self._build_observation()

    def _episode_counter(self) -> int:
        return getattr(self, "_episode_idx", 0)

    # ------------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------------
    def _compute_mask(self) -> torch.Tensor:
        """[E, N_ACTIONS] legal-action mask.

        idle is always legal. Service k is legal iff the env has at least
        one energy token remaining (R1D: integer tokens, no float rounding
        of the legal-action set). The mask never reveals current channel
        activity or hidden value.
        """
        enough = self.energy_tokens >= 1
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

    def _pending_per_service_batched(self) -> torch.Tensor:
        """[E, n_services] int64 exact pending count (privileged for pomdp_v1)."""
        out = torch.zeros(self.E, self.n_services, dtype=torch.int64,
                          device=self.device)
        for e in range(self.E):
            for (svc, _arr, _dl, _ds) in self.tracker.pending[e]:
                out[e, svc] += 1
        return out

    def _build_observation(self) -> torch.Tensor:
        prev_oh = torch.nn.functional.one_hot(self.prev_action, N_ACTIONS).float()
        if self.profile == PROFILE_MDP_SANITY:
            # Fully-observed MDP: expose exact pending count and current radar
            # service one-hot. This profile is the sanity check that PPO can
            # learn at all on this env; it is NOT a POMDP claim.
            pending = self._pending_per_service_batched().float()
            radar_svc = int(self.radar.service_at_step(self.step_idx))
            radar_oh = torch.zeros(self.E, self.n_services,
                                   device=self.device, dtype=torch.float32)
            radar_oh[:, radar_svc] = 1.0
            return build_observation(
                energy=self.energy,
                initial_energy=torch.full_like(self.energy, self.cfg.E0),
                step_idx=self.step_idx,
                horizon=self.H,
                intercept_confidence=self.intercept_confidence,
                intercept_age=self.intercept_age,
                prev_action_onehot=prev_oh,
                profile=PROFILE_MDP_SANITY,
                pending_per_service=pending,
                radar_service_onehot=radar_oh,
            )
        # pomdp_v1
        detect, urgency = self._delayed_obs()
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
            profile=PROFILE_POMDP,
        )

    def _build_privileged(self) -> torch.Tensor:
        pending = self._pending_per_service_batched()
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
    # Potential (R1C: must capture s_t before any transition effect)
    # ------------------------------------------------------------------
    def _potential(self) -> torch.Tensor:
        """Phi(belief) = -potential_coef * pending_count_total.

        This is computed by reading the CURRENT tracker state; the caller
        is responsible for capturing Phi(s_t) BEFORE the transition touches
        the tracker and Phi(s_{t+1}) AFTER all transition effects.
        """
        pending = torch.zeros(self.E, dtype=torch.float32, device=self.device)
        for e in range(self.E):
            pending[e] = float(self.tracker.pending_count(e))
        return -float(self.cfg.potential_coef) * pending

    # ------------------------------------------------------------------
    # Per-mission event ledger (R1D)
    # ------------------------------------------------------------------
    def _ledger_admit(self, *, env_idx: int, step: int, service_id: int,
                       deadline_step: int) -> None:
        key = (env_idx, step, service_id)
        # An arrival key is unique by (env, step, service) because the
        # scenario table is Bernoulli per (step, service); at most one
        # mission per (env, step, service) is admitted.
        self.event_ledger[key] = {
            "env_idx": env_idx,
            "arrival_step": step,
            "service_id": service_id,
            "deadline_step": deadline_step,
            "disposition": "pending",
        }

    def _ledger_finalize(self, *, env_idx: int, step: int, service_id: int,
                          disposition: str) -> None:
        key = (env_idx, step, service_id)
        ev = self.event_ledger.get(key)
        if ev is None:
            return
        if ev["disposition"] != "pending":
            # Idempotency: do not overwrite a finalized disposition.
            return
        ev["disposition"] = disposition

    def ledger_rows(self) -> list[dict]:
        """Flat list of all per-mission ledger rows for tests/eval."""
        return list(self.event_ledger.values())

    def ledger_identity_residual(self) -> int:
        """Number of rows whose disposition is not in the accounting identity.

        Must be zero at episode end. The accounting identity is
            eligible == success + timeout + admission_reject + horizon_failure.
        Any row stuck in "pending" or with an unknown disposition is a bug.
        """
        bad = 0
        for ev in self.event_ledger.values():
            if ev["disposition"] not in (
                DISPO_SUCCESS, DISPO_TIMEOUT, DISPO_ADMISSION_REJECT,
                DISPO_HORIZON_FAILURE,
            ):
                bad += 1
        return bad

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
        # R1D: step-after-done guard. Once done is True the caller must
        # reset() before step() can be called again.
        if self._done_flag:
            raise RuntimeError(
                "step() called after episode done; call reset() first"
            )
        if actions.shape != (self.E,):
            raise ValueError(f"actions shape {actions.shape}; expected ({self.E},)")
        if actions.dtype != torch.int64:
            raise TypeError(f"actions dtype {actions.dtype}; expected int64")
        # R1D: action OOB guard.
        if bool((actions < 0).any() or (actions >= N_ACTIONS).any()):
            bad_idx = torch.nonzero(
                (actions < 0) | (actions >= N_ACTIONS)
            ).flatten().tolist()
            raise ContractViolation(
                f"action out of bounds for envs {bad_idx}; "
                f"legal range [0, {N_ACTIONS}); actions={actions.tolist()}"
            )

        # ---- 0. R1C: capture Phi(s_t) BEFORE the transition touches the tracker
        phi_before = self._potential()

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

        # ---- 3. Apply executed action to jammer state (R1D: int tokens) -
        energy_before = self.energy.clone()
        tokens_before = self.energy_tokens.clone()
        is_jam = (actions != ACTION_IDLE)
        selected_service = torch.where(
            is_jam, actions - 1,
            torch.full_like(actions, -1),
        )
        self.energy_tokens = self.energy_tokens - is_jam.to(torch.int64)
        # Tokens never go negative (mask already enforced); clamp is defense.
        self.energy_tokens = self.energy_tokens.clamp(min=0)
        self.energy = self.energy_tokens.float() * float(
            self.cfg.P_jam_W * self.cfg.dt
        )

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
                    self._ledger_admit(
                        env_idx=e, step=self.step_idx, service_id=svc,
                        deadline_step=min(deadline, self.H - 1),
                    )

        # ---- 5. Radar opponent acts (deterministic service pick) -------
        radar_svc = self.radar.service_at_step(self.step_idx)

        # ---- 6. Detection: per-env JNR + Bernoulli draw ----------------
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

        det_draw = torch.rand(self.E, generator=self._detector_gen, device=self.device)
        detected = det_draw < p_det

        for e in range(self.E):
            self.tracker.record_detection(
                env_idx=e, step=self.step_idx,
                service_id=radar_svc, detected=bool(detected[e]),
            )

        # ---- 7. Update delayed-observation EMAs -------------------------
        detect_row = torch.zeros(self.E, self.n_services,
                                 device=self.device, dtype=torch.float32)
        detect_row[:, radar_svc] = detected.float()
        self._detect_history.append(detect_row)

        # R1B: pomdp_v1 urgency proxy is the NON-INVERTIBLE saturating
        # transform of n_pending. mdp_sanity_v1 also records it for the
        # ring buffer (not consumed by the actor in that profile).
        pending_now = self._pending_per_service_batched()
        urgency_now = pomdp_urgency_proxy(pending_now)
        self._urgency_history.append(urgency_now)

        # ---- 8. Intercept confidence + age ------------------------------
        new_conf = torch.zeros(self.E, device=self.device, dtype=torch.float32)
        for e in range(self.E):
            if bool(is_jam[e]):
                js = int(selected_service[e].item())
                if js == radar_svc:
                    new_conf[e] = 1.0 - float(p_det[e])
                else:
                    new_conf[e] = float(self.intercept_confidence[e])
            else:
                new_conf[e] = 0.0
        self.intercept_confidence = (
            self.cfg.obs_ema_alpha * new_conf
            + (1.0 - self.cfg.obs_ema_alpha) * self.intercept_confidence
        )
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
        # R1D: ledger finalize for this step's dispositions.
        for e in range(self.E):
            for (svc, arr, _dl, _ds) in self.tracker._last_finalize_step.get(e, []):
                self._ledger_finalize(
                    env_idx=e, step=arr, service_id=svc, disposition=DISPO_SUCCESS,
                )
            for (svc, arr, _dl, _ds) in self.tracker._last_finalize_timeout.get(e, []):
                self._ledger_finalize(
                    env_idx=e, step=arr, service_id=svc, disposition=DISPO_TIMEOUT,
                )

        # ---- 10. R1C: capture Phi(s_{t+1}) AFTER all transition effects,
        # then advance step_idx. Shaping uses the telescoping form.
        phi_after = self._potential()
        self.step_idx += 1
        self._obs_state_version += 1
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
        done_any = self.step_idx >= self.H
        done = torch.full((self.E,), done_any,
                          dtype=torch.bool, device=self.device)
        if done_any:
            self._done_flag = True
            for e in range(self.E):
                self.tracker.finalize_horizon(env_idx=e, counters=self.counters)
                # R1D: ledger finalize for horizon-still-pending rows.
                # After finalize_horizon, tracker.pending[e] is empty; we
                # captured the keys above via the horizon_failures list.
                for (svc, arr, _dl, _ds) in self.tracker._last_finalize_horizon.get(e, []):
                    self._ledger_finalize(
                        env_idx=e, step=arr, service_id=svc,
                        disposition=DISPO_HORIZON_FAILURE,
                    )

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
            "energy_tokens_before": tokens_before,
            "energy_tokens_after": self.energy_tokens.clone(),
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
