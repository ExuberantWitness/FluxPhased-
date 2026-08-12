"""S3 vec env: S2 + Bernoulli(5) cell binding (per-cell on/off) + per-cell energy.

Differences from S2 (`env/gpu/array_face_s2/env.py`):
  - cell binding: actor chooses a per-cell on/off mask each step (3rd head)
  - energy cost: each jamming step consumes Σ(active cells) tokens, not 1
    (idle consumes 0 regardless of cell mask)
  - zero-cell clamp: base=jam + all-zero cell mask -> env forces the highest-
    logit cell on before physics (prevents "free idle" degenerate optimum)
  - physics: compute_jnr_db_s3 with dynamic N_active = Σ(active cells)
  - obs: unchanged (21 dims, HANDOFF §11.1)

All other semantics (mission tracker, potential shaping, RNG streams, POMDP
profile, arrival process) are identical to S2.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from env.gpu.g3_bsta_lite.action_contract import (
    ACTION_IDLE, ContractViolation,
)
from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.g3_bsta_lite.observation import (
    PROFILE_POMDP, PROFILE_MDP_SANITY, pomdp_urgency_proxy,
)
from env.gpu.g3_bsta_lite.scenario import Scenario, generate_paired_manifest
from env.gpu.g3_bsta_lite.metrics import (
    MissionTracker, MissionCounterBatch,
)
from env.gpu.array_face_s3.array_factor import (
    RadarULAConfig, JammerULAConfig, N_BEAM_DIRS_S1, N_BEAM_DIRS_S2, N_CELLS,
)
from env.gpu.array_face_s3.physics import compute_jnr_db_s3, compute_p_detect_s3
from env.gpu.array_face_s3.observation import (
    build_observation_s3, build_privileged_s3,
    OBS_DIM_S3, PRIVILEGED_DIM_S3, PROFILE_ARRAY_FACE_S3,
)
from env.gpu.array_face_s3.action_contract import (
    N_ACTIONS_BASE, N_ACTIONS_BEAM,
    BernoulliTransitionTrace, validate_actions,
)


@dataclass
class EnvConfig:
    n_envs: int = 16
    horizon: int = 64
    n_services: int = 2
    dt: float = 1.0
    P_jam_W: float = 2.0  # S3 per-cell (correct default; S2-fixed semantics)
    active_budget_steps: int = 16
    duty_budget: float = 0.25
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
        if self.profile == PROFILE_POMDP:
            self.obs_delay_steps = max(1, int(self.obs_delay_steps))
        max_budget = max(1, int(self.duty_budget * self.horizon))
        if self.active_budget_steps > max_budget:
            raise ValueError(
                f"active_budget_steps={self.active_budget_steps} exceeds duty cap {max_budget}"
            )
        if self.active_budget_steps >= self.horizon:
            raise ValueError("always-on jamming is infeasible (active_budget_steps >= horizon)")
        self.E0_tokens = int(self.active_budget_steps)
        self.cost_per_action_tokens = 1
        # E0 is the energy equivalent of the token budget. Note: actual energy
        # consumed depends on cells active (per-cell token semantics), so E0 is
        # an upper bound assuming all cells on every jam step.
        self.E0 = float(self.E0_tokens) * float(self.P_jam_W) * float(self.dt)


class ArrayFaceS3VecEnv:
    """S3 vec env: S2 + jammer cell binding + per-cell energy budget."""

    def __init__(
        self, cfg: EnvConfig, *,
        physics: DebugPhysicsConfig,
        radar: RadarULAConfig,
        jammer: JammerULAConfig,
    ):
        self.cfg = cfg
        self.physics = physics
        self.radar = radar
        self.jammer = jammer
        self.device = torch.device(cfg.device)
        self.E = cfg.n_envs
        self.H = cfg.horizon
        self.n_services = cfg.n_services

        self._scenarios: list[Scenario] | None = None

        self.energy_tokens = torch.full((self.E,), cfg.E0_tokens, dtype=torch.int64, device=self.device)
        self.energy = self.energy_tokens.float() * cfg.P_jam_W * cfg.dt
        self.step_idx = 0
        self.prev_base = torch.zeros(self.E, dtype=torch.int64, device=self.device)
        self.prev_beam = torch.zeros(self.E, dtype=torch.int64, device=self.device)
        # S3: previous executed cell mask (all-on at init, mirroring S2 "all cells active")
        self.prev_cell = torch.ones((self.E, N_CELLS), dtype=torch.float32, device=self.device)
        self.tracker = MissionTracker(
            n_envs=self.E, n_services=self.n_services, detects_required=cfg.detects_required,
        )
        self.counters = MissionCounterBatch.zeros(self.E, device=str(self.device))
        self._obs_state_version = 0
        self._done_flag = False

        self._delayed_detect_ema = torch.zeros((self.E, self.n_services), device=self.device)
        self._delayed_urgency_ema = torch.zeros((self.E, self.n_services), device=self.device)
        self._detect_history: list[torch.Tensor] = []
        self._urgency_history: list[torch.Tensor] = []

        self.intercept_confidence = torch.zeros(self.E, device=self.device)
        self.intercept_age = torch.full((self.E,), self.H, dtype=torch.int64, device=self.device)

        self._event_gen = torch.Generator(device=str(self.device))
        self._detector_gen = torch.Generator(device=str(self.device))
        self._action_gen = torch.Generator(device=str(self.device))

        self.event_ledger: dict = {}
        self.step_logs: list[dict] = []

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

        self.energy_tokens = torch.full((self.E,), self.cfg.E0_tokens, dtype=torch.int64, device=self.device)
        self.energy = self.energy_tokens.float() * self.cfg.P_jam_W * self.cfg.dt
        self.step_idx = 0
        self.prev_base = torch.zeros(self.E, dtype=torch.int64, device=self.device)
        self.prev_beam = torch.zeros(self.E, dtype=torch.int64, device=self.device)
        self.prev_cell = torch.ones((self.E, N_CELLS), dtype=torch.float32, device=self.device)
        self.tracker.initialize()
        if reset_metrics:
            self.counters = MissionCounterBatch.zeros(self.E, device=str(self.device))
        self._obs_state_version = 0
        self._done_flag = False

        self._delayed_detect_ema = torch.zeros((self.E, self.n_services), device=self.device)
        self._delayed_urgency_ema = torch.zeros((self.E, self.n_services), device=self.device)
        self._detect_history = []
        self._urgency_history = []
        self.intercept_confidence = torch.zeros(self.E, device=self.device)
        self.intercept_age = torch.full((self.E,), self.H, dtype=torch.int64, device=self.device)

        self.event_ledger = {}
        self.step_logs = []

        return self._build_observation()

    # ---------- Mask ----------
    def _compute_mask(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (mask_base[E,3], mask_beam[E,5], mask_cell[E,N_CELLS]).

        mask_base: idle always legal; jam requires >=1 energy token (need to
          power at least 1 cell). Note: with per-cell token semantics, jamming
          with k cells costs k tokens, but we only gate on >=1 here (the actual
          cost is deducted post-sample; if the agent picks more cells than it
          can afford, env clamps to remaining tokens).
        mask_beam: always all-True (no beam constraint).
        mask_cell: always all-True (no per-cell constraint at sample time;
          the zero-cell clamp is applied post-sample in step()).
        """
        mask_base = torch.zeros((self.E, N_ACTIONS_BASE), dtype=torch.bool, device=self.device)
        mask_base[:, ACTION_IDLE] = True
        can_jam = self.energy_tokens >= 1
        mask_base[:, 1] = can_jam
        mask_base[:, 2] = can_jam
        mask_beam = torch.ones((self.E, N_ACTIONS_BEAM), dtype=torch.bool, device=self.device)
        mask_cell = torch.ones((self.E, N_CELLS), dtype=torch.bool, device=self.device)
        return mask_base, mask_beam, mask_cell

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

    def _radar_beam_az_idx_batched(self) -> torch.Tensor:
        return torch.full((self.E,), self.step_idx % N_BEAM_DIRS_S1,
                          dtype=torch.int64, device=self.device)

    def _build_observation(self) -> torch.Tensor:
        prev_oh = torch.nn.functional.one_hot(self.prev_base.long(), num_classes=N_ACTIONS_BASE).to(torch.float32)
        radar_az = self._radar_beam_az_idx_batched()
        jammer_az = self.prev_beam
        if self.cfg.profile == PROFILE_MDP_SANITY:
            pending = self._pending_per_service_batched()
            radar_svc = self.step_idx % self.n_services
            radar_svc_t = torch.full((self.E,), radar_svc, dtype=torch.int64, device=self.device)
            radar_svc_oh = torch.nn.functional.one_hot(radar_svc_t, num_classes=self.n_services).to(torch.float32)
            obs = build_observation_s3(
                radar_beam_az_idx=radar_az, jammer_beam_az_idx=jammer_az,
                energy=self.energy, initial_energy=torch.full_like(self.energy, self.cfg.E0),
                step_idx=self.step_idx, horizon=self.H,
                intercept_confidence=self.intercept_confidence, intercept_age=self.intercept_age.float(),
                prev_action_onehot=prev_oh, profile=PROFILE_MDP_SANITY,
                pending_per_service=pending.float(), radar_service_onehot=radar_svc_oh,
            )
        else:
            delayed_detect, delayed_urgency = self._delayed_obs()
            obs = build_observation_s3(
                radar_beam_az_idx=radar_az, jammer_beam_az_idx=jammer_az,
                energy=self.energy, initial_energy=torch.full_like(self.energy, self.cfg.E0),
                step_idx=self.step_idx, horizon=self.H,
                delayed_detect=delayed_detect, delayed_urgency=delayed_urgency,
                intercept_confidence=self.intercept_confidence, intercept_age=self.intercept_age.float(),
                prev_action_onehot=prev_oh, profile=PROFILE_POMDP,
            )
        return obs

    def _build_privileged(self) -> torch.Tensor:
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
        radar_az = self._radar_beam_az_idx_batched()
        return build_privileged_s3(
            radar_beam_az_idx=radar_az, jammer_beam_az_idx=self.prev_beam,
            pending_per_service=pending, track_health_per_service=health,
            executed_cell_mask=self.prev_cell,
        )

    # ---------- Step ----------
    def step(
        self,
        action_base: torch.Tensor,
        action_beam: torch.Tensor,
        action_cell: torch.Tensor,
    ):
        if self._done_flag:
            raise RuntimeError("step() called after episode done; call reset() first")
        E = self.E
        validate_actions(action_base, action_beam, action_cell, E=E, device=self.device)

        phi_before = self._potential()

        mask_base, mask_beam, mask_cell = self._compute_mask()
        legal_base = mask_base.gather(1, action_base.unsqueeze(1)).squeeze(1)
        if not legal_base.all():
            bad = torch.where(~legal_base)[0].tolist()
            raise ContractViolation(f"illegal base actions (insufficient energy) at envs {bad}")

        obs_version_before = self._obs_state_version

        energy_before = self.energy.clone()
        tokens_before = self.energy_tokens.clone()
        is_jam = action_base != ACTION_IDLE
        selected_service = torch.where(is_jam, action_base - 1, torch.full_like(action_base, -1))

        # --- S3: zero-cell clamp + per-cell energy budget ---
        # Work on a copy so requested_cell is preserved for the trace.
        executed_cell = action_cell.clone()
        # Clamp: if base=jam but all cells off, force the highest-logit cell on.
        # We use the raw cell values (treated as logits/probabilities) to pick.
        zero_jam = is_jam & (executed_cell.sum(dim=-1) == 0)
        if zero_jam.any():
            # For zero-jam envs, set the first cell on (deterministic; avoids
            # needing actor logits here; env-side policy is "cell 0 fallback").
            executed_cell[zero_jam, 0] = 1.0
        # Clamp to available tokens: if Σ cells > energy_tokens, keep the top-k
        # cells that fit (by value), zero the rest. This respects the budget.
        n_active = executed_cell.sum(dim=-1).long()  # [E]
        over_budget = is_jam & (n_active > self.energy_tokens)
        if over_budget.any():
            idx = torch.where(over_budget)[0]
            for e in idx.tolist():
                budget = int(self.energy_tokens[e].item())
                if budget <= 0:
                    executed_cell[e] = 0.0
                    # budget exhausted: fall back to idle
                    # (shouldn't happen because mask_base blocks jam when tokens==0)
                else:
                    vals = executed_cell[e]
                    # keep the `budget` highest-valued cells
                    topk_idx = vals.topk(min(budget, int((vals > 0).sum().item()))).indices
                    mask_e = torch.zeros_like(vals)
                    mask_e[topk_idx] = 1.0
                    executed_cell[e] = vals * mask_e
        n_active = executed_cell.sum(dim=-1).long()  # recompute post-clamp
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
        radar_az = self.step_idx % N_BEAM_DIRS_S1
        radar_svc_t = torch.full((self.E,), radar_svc, dtype=torch.int64, device=self.device)
        radar_az_t = torch.full((self.E,), radar_az, dtype=torch.int64, device=self.device)

        jammer_service_id = torch.where(is_jam, action_base - 1, torch.zeros_like(action_base))
        # S3 physics: dynamic N_active via cell_mask. Idle envs get -inf (jammer_active=False).
        jnr = compute_jnr_db_s3(
            self.physics, self.radar, self.jammer,
            jammer_active=is_jam, jammer_service_id=jammer_service_id,
            victim_service_id=radar_svc_t,
            radar_beam_az_idx=radar_az_t,
            jammer_beam_az_idx=action_beam,
            cell_mask=executed_cell,
        )
        p_detect = compute_p_detect_s3(
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

        matched = is_jam & (selected_service == radar_svc)
        self.intercept_confidence = torch.where(
            matched,
            1.0 - p_detect,
            torch.where(is_jam, self.intercept_confidence, torch.zeros_like(self.intercept_confidence)),
        )
        self.intercept_age = torch.where(
            is_jam, torch.zeros_like(self.intercept_age), self.intercept_age + 1,
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
        reward = raw_reward + shaping

        trace = BernoulliTransitionTrace(
            observation_state_version=obs_version_before,
            mask_base=mask_base, mask_beam=mask_beam, mask_cell=mask_cell,
            requested_base=action_base.clone(), requested_beam=action_beam.clone(),
            requested_cell=action_cell.clone(),
            executed_base=action_base.clone(), executed_beam=action_beam.clone(),
            executed_cell=executed_cell.clone(),
            selected_service=selected_service,
            n_active_cells=n_active,
            energy_before=energy_before, energy_after=self.energy.clone(),
            tokens_consumed=tokens_consumed,
            legal=legal_base,
        )
        self.prev_base = action_base.clone()
        self.prev_beam = action_beam.clone()
        self.prev_cell = executed_cell.clone()
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
            "mask_base": mask_base,
            "mask_beam": mask_beam,
            "mask_cell": mask_cell,
            "radar_service": radar_svc_t,
            "radar_beam_az": radar_az_t,
            "jammer_beam_az": action_beam.clone(),
            "executed_cell_mask": executed_cell.clone(),
            "n_active_cells": n_active,
            "tokens_consumed": tokens_consumed,
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

    def sample_action_rng(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = torch.randint(0, N_ACTIONS_BASE, (self.E,), generator=self._action_gen,
                             device=self.device, dtype=torch.int64)
        beam = torch.randint(0, N_ACTIONS_BEAM, (self.E,), generator=self._action_gen,
                             device=self.device, dtype=torch.int64)
        cell = torch.randint(0, 2, (self.E, N_CELLS), generator=self._action_gen,
                             device=self.device, dtype=torch.float32)
        return base, beam, cell

    def ledger_identity_residual(self) -> int:
        total = (int(self.counters.n_eligible.sum().item())
                 - (int(self.counters.n_success.sum().item())
                    + int(self.counters.n_timeout.sum().item())
                    + int(self.counters.n_admission_reject.sum().item())
                    + int(self.counters.n_horizon_failure.sum().item())))
        return total
