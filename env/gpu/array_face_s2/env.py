"""S2 vec env: S1 + jammer 1D ULA + Tx AF + MultiDiscrete([3,5]) action.

Differences from S1 (`env/gpu/array_face_s1/env.py`):
  - jammer is now a 5-cell ULA; actor chooses jammer_beam_az per step
  - JNR uses S2 physics (`compute_jnr_db_s2`) with both AF_tx and AF_rx
  - obs: S1 16 dims + 5-dim jammer_beam_az one-hot = 21 dims
  - action: tuple (base[E], beam[E]) instead of single [E] tensor
  - mask: tuple (mask_base[E,3], mask_beam[E,5])

All other semantics (energy tokens, mission tracker, potential shaping,
RNG streams, POMDP profile) are identical to S1 / lite.
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
from env.gpu.array_face_s2.array_factor import (
    RadarULAConfig, JammerULAConfig, N_BEAM_DIRS_S1, N_BEAM_DIRS_S2,
)
from env.gpu.array_face_s2.physics import compute_jnr_db_s2, compute_p_detect_s2
from env.gpu.array_face_s2.observation import (
    build_observation_s2, build_privileged_s2,
    OBS_DIM_S2, PRIVILEGED_DIM_S2, PROFILE_ARRAY_FACE_S2,
)
from env.gpu.array_face_s2.action_contract import (
    N_ACTIONS_BASE, N_ACTIONS_BEAM,
    MultiDiscreteTransitionTrace, validate_actions,
)


@dataclass
class EnvConfig:
    n_envs: int = 16
    horizon: int = 64
    n_services: int = 2
    dt: float = 1.0
    P_jam_W: float = 10.0  # S2 default: 5 cells × 2.0 W/cell
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
        self.E0 = float(self.E0_tokens) * float(self.P_jam_W) * float(self.dt)


class ArrayFaceS2VecEnv:
    """S2 vec env: S1 + jammer 1D ULA + beam steering."""

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
    def _compute_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        mask_base = torch.zeros((self.E, N_ACTIONS_BASE), dtype=torch.bool, device=self.device)
        mask_base[:, ACTION_IDLE] = True
        can_jam = self.energy_tokens >= 1
        mask_base[:, 1] = can_jam
        mask_base[:, 2] = can_jam
        mask_beam = torch.ones((self.E, N_ACTIONS_BEAM), dtype=torch.bool, device=self.device)
        return mask_base, mask_beam

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
        jammer_az = self.prev_beam  # jammer's last chosen beam_az
        if self.cfg.profile == PROFILE_MDP_SANITY:
            pending = self._pending_per_service_batched()
            radar_svc = self.step_idx % self.n_services
            radar_svc_t = torch.full((self.E,), radar_svc, dtype=torch.int64, device=self.device)
            radar_svc_oh = torch.nn.functional.one_hot(radar_svc_t, num_classes=self.n_services).to(torch.float32)
            obs = build_observation_s2(
                radar_beam_az_idx=radar_az, jammer_beam_az_idx=jammer_az,
                energy=self.energy, initial_energy=torch.full_like(self.energy, self.cfg.E0),
                step_idx=self.step_idx, horizon=self.H,
                intercept_confidence=self.intercept_confidence, intercept_age=self.intercept_age.float(),
                prev_action_onehot=prev_oh, profile=PROFILE_MDP_SANITY,
                pending_per_service=pending.float(), radar_service_onehot=radar_svc_oh,
            )
        else:
            delayed_detect, delayed_urgency = self._delayed_obs()
            obs = build_observation_s2(
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
        return build_privileged_s2(
            radar_beam_az_idx=radar_az, jammer_beam_az_idx=self.prev_beam,
            pending_per_service=pending, track_health_per_service=health,
        )

    # ---------- Step ----------
    def step(
        self,
        action_base: torch.Tensor,
        action_beam: torch.Tensor,
    ):
        if self._done_flag:
            raise RuntimeError("step() called after episode done; call reset() first")
        E = self.E
        validate_actions(action_base, action_beam, E=E, device=self.device)

        phi_before = self._potential()

        mask_base, mask_beam = self._compute_mask()
        legal_base = mask_base.gather(1, action_base.unsqueeze(1)).squeeze(1)
        # mask_beam is all-True; no illegal beam
        if not legal_base.all():
            bad = torch.where(~legal_base)[0].tolist()
            raise ContractViolation(f"illegal base actions (insufficient energy) at envs {bad}")

        obs_version_before = self._obs_state_version

        energy_before = self.energy.clone()
        tokens_before = self.energy_tokens.clone()
        is_jam = action_base != ACTION_IDLE
        selected_service = torch.where(is_jam, action_base - 1, torch.full_like(action_base, -1))
        self.energy_tokens = (self.energy_tokens - is_jam.long()).clamp(min=0)
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
        jnr = compute_jnr_db_s2(
            self.physics, self.radar, self.jammer,
            jammer_active=is_jam, jammer_service_id=jammer_service_id,
            victim_service_id=radar_svc_t,
            radar_beam_az_idx=radar_az_t,
            jammer_beam_az_idx=action_beam,
        )
        p_detect = compute_p_detect_s2(
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

        trace = MultiDiscreteTransitionTrace(
            observation_state_version=obs_version_before,
            mask_base=mask_base, mask_beam=mask_beam,
            requested_base=action_base.clone(), requested_beam=action_beam.clone(),
            executed_base=action_base.clone(), executed_beam=action_beam.clone(),
            selected_service=selected_service,
            energy_before=energy_before, energy_after=self.energy.clone(),
            legal=legal_base,
        )
        self.prev_base = action_base.clone()
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
            "mask_base": mask_base,
            "mask_beam": mask_beam,
            "radar_service": radar_svc_t,
            "radar_beam_az": radar_az_t,
            "jammer_beam_az": action_beam.clone(),
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
        base = torch.randint(0, N_ACTIONS_BASE, (self.E,), generator=self._action_gen,
                             device=self.device, dtype=torch.int64)
        beam = torch.randint(0, N_ACTIONS_BEAM, (self.E,), generator=self._action_gen,
                             device=self.device, dtype=torch.int64)
        return base, beam

    def ledger_identity_residual(self) -> int:
        total = (int(self.counters.n_eligible.sum().item())
                 - (int(self.counters.n_success.sum().item())
                    + int(self.counters.n_timeout.sum().item())
                    + int(self.counters.n_admission_reject.sum().item())
                    + int(self.counters.n_horizon_failure.sum().item())))
        return total
