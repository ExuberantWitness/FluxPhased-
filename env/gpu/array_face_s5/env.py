"""S5 vec env: TWO jammers (K=2) on 2D UPAs vs one radar — cooperative stage.

Differences from S4 (`env/gpu/array_face_s4/env.py`), per HANDOFF §11.3:
  - state is per-jammer: energy_tokens [E, K], prev_cell [E, K, 25],
    prev_beam [E, K]; each jammer has its OWN 63-token budget
  - action is per-jammer: (cell [E, K, 25], beam [E, K])
  - physics: per-jammer JNR combined by LINEAR power summation
    (compute_jnr_db_s5); a single p_detect drives the shared mission tracker
  - observation: each jammer sees its S4 obs + the OTHER jammer's coarse
    state (beam one-hots, remaining energy, prev activity) — the explicit
    coordination channel
  - reward: TEAM reward (newly_dropped + pending potential shaping), shared
    by both jammers (plain IPPO credit assignment)
  - NO beam shaping (S4 REPORT.md §3: shaping falsified)

All other semantics (mission tracker, arrivals, RNG streams, profiles) are
identical to S4.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from env.gpu.g3_bsta_lite.action_contract import ContractViolation
from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.g3_bsta_lite.observation import (
    PROFILE_POMDP, PROFILE_MDP_SANITY, pomdp_urgency_proxy,
)
from env.gpu.g3_bsta_lite.scenario import Scenario, generate_paired_manifest
from env.gpu.g3_bsta_lite.metrics import (
    MissionTracker, MissionCounterBatch,
)
from env.gpu.array_face_s5.array_factor import (
    UPAConfig, N_AZ, N_EL, N_BEAM_DIRS_S5, N_CELLS_S5, N_JAMMERS,
)
from env.gpu.array_face_s5.physics import compute_jnr_db_s5, compute_p_detect_s5
from env.gpu.array_face_s5.observation import (
    build_observation_s5, build_privileged_s5,
    OBS_DIM_S5, PRIVILEGED_DIM_S5, PROFILE_ARRAY_FACE_S5,
)
from env.gpu.array_face_s5.action_contract import (
    N_ACTIONS_CELL, N_ACTIONS_BEAM, K_JAMMERS,
    S5TransitionTrace, validate_actions,
)


@dataclass
class EnvConfig:
    n_envs: int = 16
    horizon: int = 64
    n_services: int = 2
    dt: float = 1.0
    P_jam_W: float = 2.0  # per-cell, PER JAMMER (each jammer has its own PA)
    active_budget_steps: int = 63  # PER JAMMER
    duty_budget: float = 1.0
    arrival_rate_per_service: float = 0.15
    baseline_snr_db: float = 22.0
    mission_tau_window: int = 6
    detects_required: int = 1
    profile: str = PROFILE_POMDP
    obs_delay_steps: int = 1
    obs_ema_alpha: float = 0.5
    potential_coef: float = 0.05
    gamma: float = 0.99
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        assert self.profile in (PROFILE_POMDP, PROFILE_MDP_SANITY), \
            f"profile must be a lite profile ({PROFILE_POMDP} or {PROFILE_MDP_SANITY}), got {self.profile}"
        max_budget = max(1, int(self.duty_budget * self.horizon))
        if self.active_budget_steps > max_budget:
            raise ValueError(
                f"active_budget_steps={self.active_budget_steps} exceeds duty cap {max_budget}"
            )
        if self.active_budget_steps >= self.horizon:
            raise ValueError("always-on jamming is infeasible (active_budget_steps >= horizon)")
        self.E0_tokens = int(self.active_budget_steps)
        self.cost_per_action_tokens = 1
        self.E0 = float(self.E0_tokens) * float(self.P_jam_W) * float(self.dt)


class ArrayFaceS5VecEnv:
    """S5 vec env: K=2 cooperative jammers, per-jammer budgets, team reward."""

    def __init__(
        self, cfg: EnvConfig, *,
        physics: DebugPhysicsConfig,
        radar: UPAConfig,
        jammer: UPAConfig,
    ):
        self.cfg = cfg
        self.physics = physics
        self.radar = radar
        self.jammer = jammer
        self.device = torch.device(cfg.device)
        self.E = cfg.n_envs
        self.H = cfg.horizon
        self.K = N_JAMMERS
        self.n_services = cfg.n_services

        self._scenarios: list[Scenario] | None = None
        self._init_state()
        self.tracker = MissionTracker(
            n_envs=self.E, n_services=self.n_services, detects_required=cfg.detects_required,
        )
        self.counters = MissionCounterBatch.zeros(self.E, device=str(self.device))

        self._event_gen = torch.Generator(device=str(self.device))
        self._detector_gen = torch.Generator(device=str(self.device))
        self._action_gen = torch.Generator(device=str(self.device))

        self.event_ledger: dict = {}
        self.step_logs: list[dict] = []

    def _init_state(self):
        E, K = self.E, self.K
        self.energy_tokens = torch.full((E, K), self.cfg.E0_tokens, dtype=torch.int64, device=self.device)
        self.energy = self.energy_tokens.float() * self.cfg.P_jam_W * self.cfg.dt
        self.step_idx = 0
        self.prev_cell = torch.zeros((E, K, N_CELLS_S5), dtype=torch.float32, device=self.device)
        self.prev_beam = torch.zeros((E, K), dtype=torch.int64, device=self.device)
        self._obs_state_version = 0
        self._done_flag = False

        self._delayed_detect_ema = torch.zeros((self.E, self.n_services), device=self.device)
        self._delayed_urgency_ema = torch.zeros((self.E, self.n_services), device=self.device)
        self._detect_history: list[torch.Tensor] = []
        self._urgency_history: list[torch.Tensor] = []

        self.intercept_confidence = torch.zeros(self.E, device=self.device)
        self.intercept_age = torch.full((self.E,), self.H, dtype=torch.int64, device=self.device)

    # ---------- Scenario ----------
    def set_scenarios(self, scenarios: list[Scenario]):
        assert len(scenarios) == self.E
        self._scenarios = scenarios

    def _ensure_scenario(self):
        if self._scenarios is None:
            self._scenarios = generate_paired_manifest(
                base_seed=self.cfg.seed, n_scenarios=self.E,
                horizon=self.H, n_services=self.n_services,
                arrival_rate_per_service=self.cfg.arrival_rate_per_service,
                baseline_snr_db=self.cfg.baseline_snr_db,
                device=str(self.device),
            )

    # ---------- Reset ----------
    def reset(self, *, seed: int | None = None, reset_metrics: bool = True) -> torch.Tensor:
        if seed is not None:
            self.cfg.seed = int(seed)
        self._event_gen.manual_seed(self.cfg.seed)
        self._detector_gen.manual_seed(self.cfg.seed + 1)
        self._action_gen.manual_seed(self.cfg.seed + 2)
        self._ensure_scenario()

        self._init_state()
        self.tracker.initialize()
        if reset_metrics:
            self.counters = MissionCounterBatch.zeros(self.E, device=str(self.device))
        self.event_ledger = {}
        self.step_logs = []
        return self._build_observation()

    # ---------- Mask ----------
    def _compute_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mask_cell[E,K,25], mask_beam[E,K,25]).

        mask_cell[e,k] is all-False when jammer k's energy_tokens == 0,
        else all-True (per-jammer over-budget clamp handled post-sample).
        """
        can_jam = self.energy_tokens >= 1  # [E, K]
        mask_cell = (can_jam.unsqueeze(-1)
                     .expand(self.E, self.K, N_ACTIONS_CELL).clone())
        mask_beam = torch.ones((self.E, self.K, N_ACTIONS_BEAM), dtype=torch.bool, device=self.device)
        return mask_cell, mask_beam

    # ---------- Potential ----------
    def _potential(self) -> torch.Tensor:
        phi = torch.zeros(self.E, dtype=torch.float32, device=self.device)
        coef = float(self.cfg.potential_coef)
        for e in range(self.E):
            phi[e] = -coef * float(self.tracker.pending_count(e))
        return phi

    # ---------- Obs ----------
    def _delayed_obs(self):
        d = self.cfg.obs_delay_steps
        if len(self._detect_history) <= d:
            zeros = torch.zeros((self.E, self.n_services), device=self.device)
            return zeros, zeros.clone()
        return self._detect_history[-1 - d], self._urgency_history[-1 - d]

    def _pending_per_service_batched(self) -> torch.Tensor:
        out = torch.zeros((self.E, self.n_services), dtype=torch.int64, device=self.device)
        for e in range(self.E):
            out[e] = self.tracker.pending_count_per_service(e).to(self.device)
        return out

    def _radar_beam_idx_batched(self) -> torch.Tensor:
        return torch.full((self.E,), self.step_idx % N_BEAM_DIRS_S5,
                          dtype=torch.int64, device=self.device)

    def _prev_active(self) -> torch.Tensor:
        """[E, K] int64 ∈ {0,1}: per-jammer previous-step transmit flag."""
        return (self.prev_cell.sum(dim=-1) > 0).long()

    def _build_observation_for(self, k: int) -> torch.Tensor:
        """Jammer k's [E, OBS_DIM_S5] observation (own + other's coarse state)."""
        other = 1 - k
        radar_beam = self._radar_beam_idx_batched()
        radar_az, radar_el = radar_beam % N_AZ, radar_beam // N_AZ
        own_az = self.prev_beam[:, k] % N_AZ
        own_el = self.prev_beam[:, k] // N_AZ
        oth_az = self.prev_beam[:, other] % N_AZ
        oth_el = self.prev_beam[:, other] // N_AZ
        prev_active = self._prev_active()
        oth_rem_E = self.energy[:, other]
        oth_E0 = torch.full_like(oth_rem_E, self.cfg.E0)

        if self.cfg.profile == PROFILE_MDP_SANITY:
            pending = self._pending_per_service_batched()
            radar_svc = self.step_idx % self.n_services
            radar_svc_t = torch.full((self.E,), radar_svc, dtype=torch.int64, device=self.device)
            radar_svc_oh = torch.nn.functional.one_hot(radar_svc_t, num_classes=self.n_services).to(torch.float32)
            return build_observation_s5(
                radar_beam_az_idx=radar_az, radar_beam_el_idx=radar_el,
                jammer_beam_az_idx=own_az, jammer_beam_el_idx=own_el,
                other_beam_az_idx=oth_az, other_beam_el_idx=oth_el,
                prev_active=prev_active[:, k], other_prev_active=prev_active[:, other],
                other_energy=oth_rem_E, other_initial_energy=oth_E0,
                energy=self.energy[:, k], initial_energy=torch.full_like(self.energy[:, k], self.cfg.E0),
                step_idx=self.step_idx, horizon=self.H,
                intercept_confidence=self.intercept_confidence, intercept_age=self.intercept_age.float(),
                profile=PROFILE_MDP_SANITY,
                pending_per_service=pending.float(), radar_service_onehot=radar_svc_oh,
            )
        delayed_detect, delayed_urgency = self._delayed_obs()
        return build_observation_s5(
            radar_beam_az_idx=radar_az, radar_beam_el_idx=radar_el,
            jammer_beam_az_idx=own_az, jammer_beam_el_idx=own_el,
            other_beam_az_idx=oth_az, other_beam_el_idx=oth_el,
            prev_active=prev_active[:, k], other_prev_active=prev_active[:, other],
            other_energy=oth_rem_E, other_initial_energy=oth_E0,
            energy=self.energy[:, k], initial_energy=torch.full_like(self.energy[:, k], self.cfg.E0),
            step_idx=self.step_idx, horizon=self.H,
            delayed_detect=delayed_detect, delayed_urgency=delayed_urgency,
            intercept_confidence=self.intercept_confidence, intercept_age=self.intercept_age.float(),
            profile=PROFILE_POMDP,
        )

    def _build_observation(self) -> torch.Tensor:
        """Stacked [E, K, OBS_DIM_S5] (slot 0 = jammer 0's view)."""
        return torch.stack([self._build_observation_for(k) for k in range(self.K)], dim=1)

    def _build_privileged(self) -> torch.Tensor:
        """[E, PRIVILEGED_DIM_S5] central-critic input (both jammers' facts)."""
        pending = self._pending_per_service_batched()
        health = torch.zeros((self.E, self.n_services), dtype=torch.float32, device=self.device)
        for e in range(self.E):
            per_svc_total = [0] * self.n_services
            per_svc_done = [0] * self.n_services
            for (svc, _arr, _dl, ds) in self.tracker.pending[e]:
                per_svc_total[svc] += 1
                if ds >= self.cfg.detects_required:
                    per_svc_done[svc] += 1
            for svc in range(self.n_services):
                health[e, svc] = per_svc_done[svc] / max(1, per_svc_total[svc])
        radar_beam = self._radar_beam_idx_batched()
        radar_az, radar_el = radar_beam % N_AZ, radar_beam // N_AZ
        return build_privileged_s5(
            radar_beam_az_idx=radar_az, radar_beam_el_idx=radar_el,
            jammer_beam_az_idx=self.prev_beam[:, 0] % N_AZ,
            jammer_beam_el_idx=self.prev_beam[:, 0] // N_AZ,
            other_beam_az_idx=self.prev_beam[:, 1] % N_AZ,
            other_beam_el_idx=self.prev_beam[:, 1] // N_AZ,
            pending_per_service=pending, track_health_per_service=health,
            executed_cell_mask=self.prev_cell[:, 0],
            other_executed_cell_mask=self.prev_cell[:, 1],
        )

    # ---------- Step ----------
    def step(
        self,
        action_cell: torch.Tensor,   # [E, K, 25]
        action_beam: torch.Tensor,   # [E, K]
    ):
        if self._done_flag:
            raise RuntimeError("step() called after episode done; call reset() first")
        E, K = self.E, self.K
        validate_actions(action_cell, action_beam, E=E, device=self.device)

        phi_before = self._potential()

        mask_cell, mask_beam = self._compute_mask()
        # contract: no cells may be on where mask_cell is False (energy==0)
        illegal = (action_cell > 0.5) & ~mask_cell
        if illegal.any():
            bad = torch.where(illegal.any(dim=(-1, -2)))[0].tolist()
            raise ContractViolation(f"illegal cell actions (insufficient energy) at envs {bad}")

        obs_version_before = self._obs_state_version

        energy_before = self.energy.clone()
        tokens_before = self.energy_tokens.clone()
        executed_cell = action_cell.clone()
        is_jam = executed_cell.sum(dim=-1) > 0  # [E, K]

        # per-jammer over-budget clamp: keep the top-k highest-valued cells
        # that fit that jammer's remaining tokens (same semantics as S3/S4).
        n_active = executed_cell.sum(dim=-1).long()  # [E, K]
        over_budget = is_jam & (n_active > self.energy_tokens)
        if over_budget.any():
            envs, ks = torch.where(over_budget)
            for e, k in zip(envs.tolist(), ks.tolist()):
                budget = int(self.energy_tokens[e, k].item())
                if budget <= 0:
                    executed_cell[e, k] = 0.0
                else:
                    vals = executed_cell[e, k]
                    topk_idx = vals.topk(min(budget, int((vals > 0).sum().item()))).indices
                    mask_e = torch.zeros_like(vals)
                    mask_e[topk_idx] = 1.0
                    executed_cell[e, k] = vals * mask_e
        is_jam = executed_cell.sum(dim=-1) > 0  # recompute post-clamp
        n_active = executed_cell.sum(dim=-1).long()
        tokens_consumed = torch.where(is_jam, n_active, torch.zeros_like(n_active))
        self.energy_tokens = (self.energy_tokens - tokens_consumed).clamp(min=0)
        self.energy = self.energy_tokens.float() * self.cfg.P_jam_W * self.cfg.dt

        arrivals_step = self._scenarios_step_arrivals()
        for e in range(self.E):
            for svc in range(self.n_services):
                if bool(arrivals_step[e, svc]):
                    deadline = min(self.step_idx + self.cfg.mission_tau_window, self.H - 1)
                    self.tracker.admit(env_idx=e, step=self.step_idx, service_id=svc, deadline_step=deadline)
                    self.counters.n_eligible[e] += 1
                    self.event_ledger[(e, self.step_idx, svc)] = {
                        "disposition": None, "admitted_at": self.step_idx,
                        "deadline": deadline, "detects": 0,
                    }

        radar_svc = self.step_idx % self.n_services
        radar_beam = self.step_idx % N_BEAM_DIRS_S5
        radar_svc_t = torch.full((self.E,), radar_svc, dtype=torch.int64, device=self.device)
        radar_az_t = torch.full((self.E,), radar_beam % N_AZ, dtype=torch.int64, device=self.device)
        radar_el_t = torch.full((self.E,), radar_beam // N_AZ, dtype=torch.int64, device=self.device)
        jammer_az_t = action_beam % N_AZ  # [E, K]
        jammer_el_t = action_beam // N_AZ

        jnr = compute_jnr_db_s5(
            self.physics, self.radar, self.jammer,
            jammer_active=is_jam,
            victim_service_id=radar_svc_t,
            radar_beam_az_idx=radar_az_t,
            radar_beam_el_idx=radar_el_t,
            jammer_beam_az_idx=jammer_az_t,
            jammer_beam_el_idx=jammer_el_t,
            cell_mask=executed_cell,
        )
        p_detect = compute_p_detect_s5(
            self.physics, baseline_snr_db=self.cfg.baseline_snr_db, jnr_db=jnr,
        )
        det_draw = torch.rand(self.E, generator=self._detector_gen, device=self.device, dtype=torch.float32)
        detected = det_draw < p_detect

        for e in range(self.E):
            self.tracker.record_detection(
                env_idx=e, step=self.step_idx, service_id=radar_svc, detected=bool(detected[e]),
            )

        detect_row = torch.zeros((self.E, self.n_services), dtype=torch.float32, device=self.device)
        detect_row[:, radar_svc] = detected.float()
        self._detect_history.append(detect_row)
        pending_now = self._pending_per_service_batched()
        urgency_now = pomdp_urgency_proxy(pending_now.float(), K=3.0)
        self._urgency_history.append(urgency_now)

        # both jammers always match the radar's current service; intercept
        # confidence reflects the COMBINED jamming (1 - p_detect when any
        # jammer transmits).
        any_jam = is_jam.any(dim=-1)
        self.intercept_confidence = torch.where(
            any_jam,
            1.0 - p_detect,
            torch.zeros_like(self.intercept_confidence),
        )
        self.intercept_age = torch.where(
            any_jam, torch.zeros_like(self.intercept_age), self.intercept_age + 1,
        )

        newly_dropped = torch.zeros(self.E, dtype=torch.int64, device=self.device)
        newly_succeeded = torch.zeros(self.E, dtype=torch.int64, device=self.device)
        for e in range(self.E):
            n_suc_before = int(self.counters.n_success[e].item())
            n_to_before = int(self.counters.n_timeout[e].item())
            self.tracker.finalize_step(env_idx=e, step=self.step_idx, counters=self.counters)
            newly_succeeded[e] = int(self.counters.n_success[e].item()) - n_suc_before
            newly_dropped[e] = int(self.counters.n_timeout[e].item()) - n_to_before

        phi_after = self._potential()
        self.step_idx += 1
        self._obs_state_version += 1
        raw_reward = newly_dropped.float()
        shaping = self.cfg.gamma * phi_after - phi_before
        reward = raw_reward + shaping  # TEAM reward, shared by both jammers

        trace = S5TransitionTrace(
            observation_state_version=obs_version_before,
            mask_cell=mask_cell, mask_beam=mask_beam,
            requested_cell=action_cell.clone(), requested_beam=action_beam.clone(),
            executed_cell=executed_cell.clone(), executed_beam=action_beam.clone(),
            is_jam=is_jam,
            n_active_cells=n_active,
            energy_before=energy_before, energy_after=self.energy.clone(),
            tokens_consumed=tokens_consumed,
            legal=torch.ones(self.E, dtype=torch.bool, device=self.device),
        )
        self.prev_cell = executed_cell.clone()
        self.prev_beam = action_beam.clone()
        obs = self._build_observation()
        done = self.step_idx >= self.H
        if done:
            for e in range(self.E):
                self.tracker.finalize_horizon(env_idx=e, counters=self.counters)
            self._done_flag = True

        info = {
            "trace": trace,
            "raw_drop": newly_dropped,
            "newly_succeeded": newly_succeeded,
            "shaping": shaping,
            "potential_before": phi_before,
            "potential_after": phi_after,
            "p_detect": p_detect,
            "jnr_db": jnr,
            "jnr_db_per_jammer_linear_sum": jnr,
            "mask_cell": mask_cell,
            "mask_beam": mask_beam,
            "radar_service": radar_svc_t,
            "radar_beam_az": radar_az_t,
            "radar_beam_el": radar_el_t,
            "jammer_beam_az": jammer_az_t,
            "jammer_beam_el": jammer_el_t,
            "executed_cell_mask": executed_cell.clone(),
            "n_active_cells": n_active,
            "tokens_consumed": tokens_consumed,
            "is_jam": is_jam,
            "step_idx": self.step_idx - 1,
            "energy_tokens_before": tokens_before,
            "energy_tokens_after": self.energy_tokens.clone(),
        }
        return obs, reward, torch.full((self.E,), done, dtype=torch.bool, device=self.device), info

    def _scenarios_step_arrivals(self) -> torch.Tensor:
        out = torch.zeros((self.E, self.n_services), dtype=torch.bool, device=self.device)
        if self._scenarios is None:
            return out
        for e, sc in enumerate(self._scenarios):
            if self.step_idx < sc.horizon:
                out[e] = sc.arrivals[self.step_idx].to(self.device)
        return out

    # ---------- Metrics ----------
    def drop_ratio(self) -> torch.Tensor:
        return self.counters.drop_ratio()

    def accounting_residual(self) -> torch.Tensor:
        return self.counters.accounting_residual()

    def privileged(self) -> torch.Tensor:
        return self._build_privileged()

    def sample_action_rng(self) -> tuple[torch.Tensor, torch.Tensor]:
        cell = torch.randint(0, 2, (self.E, self.K, N_ACTIONS_CELL), generator=self._action_gen,
                             device=self.device, dtype=torch.float32)
        beam = torch.randint(0, N_ACTIONS_BEAM, (self.E, self.K), generator=self._action_gen,
                             device=self.device, dtype=torch.int64)
        return cell, beam

    def ledger_identity_residual(self) -> int:
        total = (int(self.counters.n_eligible.sum().item())
                 - (int(self.counters.n_success.sum().item())
                    + int(self.counters.n_timeout.sum().item())
                    + int(self.counters.n_admission_reject.sum().item())
                    + int(self.counters.n_horizon_failure.sum().item())))
        return total
