"""S6 vec env — 1 learning jammer vs 2 learning radars (first opponent learning).

Differences from S5 (HANDOFF §11.4):
  - defense upgraded: TWO radars at az = ±20° (geometry.py), each with a
    LEARNABLE beam (Cat(25)) + service choice (Cat(2)) — replacing S4/S5's
    deterministic step%25 sweep
  - one S4-style jammer (Bernoulli(25) cells + Cat(25) beam, 63 tokens)
  - per-radar JNR via generalized AFs (compute_jnr_db_s6); each radar draws
    its own detection; both feed the shared mission tracker
  - opposing rewards (not strictly zero-sum, event-level opposing signals):
      jammer reward = newly_dropped  + pending potential shaping (S4-style)
      radar  reward = newly_succeeded − newly_dropped
  - both sides see each other's LAST actions (ESM/DOA channels in obs)
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

from env.gpu.g3_bsta_lite.action_contract import ContractViolation
from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig
from env.gpu.g3_bsta_lite.scenario import Scenario, generate_paired_manifest
from env.gpu.g3_bsta_lite.metrics import MissionCounterBatch
from env.gpu.array_face_s6.array_factor import (
    UPAConfig, N_BEAM_DIRS_S6, N_CELLS_S6, N_RADARS, N_AZ,
)
from env.gpu.array_face_s6.geometry import radar_directions
from env.gpu.array_face_s6.tracker import S6MissionTracker
from env.gpu.array_face_s6.physics import (
    compute_jnr_db_s6, compute_p_detect_s6, compute_snr_eff_db_s6, target_gain_db,
)
from env.gpu.array_face_s6.observation import (
    build_observation_jammer, build_observation_radar,
    OBS_DIM_JAMMER, OBS_DIM_RADAR, PROFILE_ARRAY_FACE_S6,
)
from env.gpu.array_face_s6.action_contract import validate_actions


@dataclass
class EnvConfig:
    n_envs: int = 16
    horizon: int = 64
    n_services: int = 2
    dt: float = 1.0
    P_jam_W: float = 0.1   # S6b rebalance (contestability sweep; see geometry.py)
    active_budget_steps: int = 63
    duty_budget: float = 1.0
    arrival_rate_per_service: float = 0.15
    baseline_snr_db: float = 12.0  # S6b rebalance: thin cushion binds scan-vs-stare
    mission_tau_window: int = 6
    detects_required: int = 1
    potential_coef: float = 0.05
    gamma: float = 0.99
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        max_budget = max(1, int(self.duty_budget * self.horizon))
        if self.active_budget_steps > max_budget:
            raise ValueError(f"active_budget_steps={self.active_budget_steps} exceeds duty cap {max_budget}")
        if self.active_budget_steps >= self.horizon:
            raise ValueError("always-on jamming is infeasible")
        self.E0_tokens = int(self.active_budget_steps)
        self.E0 = float(self.E0_tokens) * float(self.P_jam_W) * float(self.dt)


class ArrayFaceS6VecEnv:
    """S6 vec env: adversarial 1-jammer-vs-2-radars with learnable defenders."""

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
        self.R = N_RADARS
        self.n_services = cfg.n_services
        self._radar_az, self._radar_el = radar_directions(str(self.device))

        self._scenarios: list[Scenario] | None = None
        self._az_table: torch.Tensor | None = None  # [H, n_services] mission bearings
        self._init_state()
        self.tracker = S6MissionTracker(
            n_envs=self.E, n_services=self.n_services, detects_required=cfg.detects_required,
        )
        self.counters = MissionCounterBatch.zeros(self.E, device=str(self.device))

        self._event_gen = torch.Generator(device=str(self.device))
        self._detector_gen = torch.Generator(device=str(self.device))
        self._action_gen = torch.Generator(device=str(self.device))
        self.event_ledger: dict = {}

    def _init_state(self):
        E, R = self.E, self.R
        self.energy_tokens = torch.full((E,), self.cfg.E0_tokens, dtype=torch.int64, device=self.device)
        self.energy = self.energy_tokens.float() * self.cfg.P_jam_W * self.cfg.dt
        self.step_idx = 0
        self.prev_cell = torch.zeros((E, N_CELLS_S6), dtype=torch.float32, device=self.device)
        self.prev_beam = torch.zeros(E, dtype=torch.int64, device=self.device)          # jammer
        self.prev_radar_beam = torch.zeros((E, R), dtype=torch.int64, device=self.device)
        self.prev_radar_svc = torch.zeros((E, R), dtype=torch.int64, device=self.device)
        self.radar_detected_last = torch.zeros((E, R), dtype=torch.float32, device=self.device)
        self.intercept_confidence = torch.zeros(E, device=self.device)
        self.intercept_age = torch.full((self.E,), self.H, dtype=torch.int64, device=self.device)
        self._done_flag = False

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
    def reset(self, *, seed: int | None = None, reset_metrics: bool = True):
        if seed is not None:
            self.cfg.seed = int(seed)
        self._event_gen.manual_seed(self.cfg.seed)
        self._detector_gen.manual_seed(self.cfg.seed + 1)
        self._action_gen.manual_seed(self.cfg.seed + 2)
        self._ensure_scenario()
        # mission-bearing table: deterministic per cfg.seed (paired scenarios
        # with the same seed share arrivals AND bearings)
        az_gen = torch.Generator(device=str(self.device)).manual_seed(self.cfg.seed + 999)
        self._az_table = torch.randint(0, N_AZ, (self.H, self.n_services),
                                       generator=az_gen, device=self.device, dtype=torch.int64)
        self._init_state()
        self.tracker.initialize()
        if reset_metrics:
            self.counters = MissionCounterBatch.zeros(self.E, device=str(self.device))
        self.event_ledger = {}
        return self._build_observation()

    # ---------- Masks ----------
    def _compute_masks(self):
        """(mask_cell[E,25], mask_beam[E,25]) for the jammer; radar heads unmasked."""
        can_jam = (self.energy_tokens >= 1)
        mask_cell = can_jam.unsqueeze(1).expand(self.E, N_CELLS_S6).clone()
        mask_beam = torch.ones((self.E, N_BEAM_DIRS_S6), dtype=torch.bool, device=self.device)
        return mask_cell, mask_beam

    # ---------- Obs ----------
    def _pending_per_service_batched(self) -> torch.Tensor:
        out = torch.zeros((self.E, self.n_services), dtype=torch.int64, device=self.device)
        for e in range(self.E):
            out[e] = self.tracker.pending_count_per_service(e).to(self.device)
        return out

    def _build_observation(self):
        """Returns (obs_j [E,55], obs_r [E,R,49])."""
        pending = self._pending_per_service_batched()
        az_map = torch.stack(
            [self.tracker.pending_az_map(e, device=self.device) for e in range(self.E)],
            dim=0)  # [E, n_services, N_AZ]
        j_az, j_el = self.prev_beam % 5, self.prev_beam // 5
        obs_j = build_observation_jammer(
            energy=self.energy, initial_energy=torch.full_like(self.energy, self.cfg.E0),
            step_idx=self.step_idx, horizon=self.H,
            pending_per_service=pending,
            pending_az_map=az_map,
            intercept_confidence=self.intercept_confidence, intercept_age=self.intercept_age,
            prev_active=(self.prev_cell.sum(-1) > 0).long(),
            radar_beam_az=self.prev_radar_beam % 5, radar_beam_el=self.prev_radar_beam // 5,
            radar_svc=self.prev_radar_svc,
            jammer_beam_az=j_az, jammer_beam_el=j_el,
            radar_detected_last=self.radar_detected_last,
        )
        jammer_active = (self.prev_cell.sum(-1) > 0).float()
        obs_r_list = []
        for r in range(self.R):
            other = 1 - r
            obs_r_list.append(build_observation_radar(
                step_idx=self.step_idx, horizon=self.H,
                pending_az_map=az_map,
                own_intercept_confidence=self.intercept_confidence,  # shared link-level stat
                own_detected_last=self.radar_detected_last[:, r],
                other_detected_last=self.radar_detected_last[:, other],
                own_beam_az=self.prev_radar_beam[:, r] % 5,
                own_beam_el=self.prev_radar_beam[:, r] // 5,
                own_svc=self.prev_radar_svc[:, r],
                other_beam_az=self.prev_radar_beam[:, other] % 5,
                other_beam_el=self.prev_radar_beam[:, other] // 5,
                other_svc=self.prev_radar_svc[:, other],
                jammer_beam_az=j_az, jammer_beam_el=j_el,
                jammer_active=jammer_active,
            ))
        return obs_j, torch.stack(obs_r_list, dim=1)

    # ---------- Step ----------
    def step(
        self,
        jammer_cell: torch.Tensor,  # [E, 25]
        jammer_beam: torch.Tensor,  # [E]
        radar_beam: torch.Tensor,   # [E, R]
        radar_svc: torch.Tensor,    # [E, R]
    ):
        if self._done_flag:
            raise RuntimeError("step() called after episode done; call reset() first")
        E, R = self.E, self.R
        validate_actions(jammer_cell, jammer_beam, radar_beam, radar_svc, E=E, device=self.device)

        phi_before = self._potential()

        mask_cell, mask_beam = self._compute_masks()
        illegal = (jammer_cell > 0.5) & ~mask_cell
        if illegal.any():
            bad = torch.where(illegal.any(dim=-1))[0].tolist()
            raise ContractViolation(f"illegal cell actions (insufficient energy) at envs {bad}")

        tokens_before = self.energy_tokens.clone()
        executed_cell = jammer_cell.clone()
        is_jam = executed_cell.sum(dim=-1) > 0
        n_active = executed_cell.sum(dim=-1).long()
        over_budget = is_jam & (n_active > self.energy_tokens)
        if over_budget.any():
            for e in torch.where(over_budget)[0].tolist():
                budget = int(self.energy_tokens[e].item())
                if budget <= 0:
                    executed_cell[e] = 0.0
                else:
                    vals = executed_cell[e]
                    topk_idx = vals.topk(min(budget, int((vals > 0).sum().item()))).indices
                    m = torch.zeros_like(vals); m[topk_idx] = 1.0
                    executed_cell[e] = vals * m
        is_jam = executed_cell.sum(dim=-1) > 0
        n_active = executed_cell.sum(dim=-1).long()
        tokens_consumed = torch.where(is_jam, n_active, torch.zeros_like(n_active))
        self.energy_tokens = (self.energy_tokens - tokens_consumed).clamp(min=0)
        self.energy = self.energy_tokens.float() * self.cfg.P_jam_W * self.cfg.dt

        arrivals_step = self._scenarios_step_arrivals()
        for e in range(E):
            for svc in range(self.n_services):
                if bool(arrivals_step[e, svc]):
                    deadline = min(self.step_idx + self.cfg.mission_tau_window, self.H - 1)
                    az_m = int(self._az_table[self.step_idx, svc].item())
                    self.tracker.admit(env_idx=e, step=self.step_idx, service_id=svc,
                                       az_idx=az_m, deadline_step=deadline)
                    self.counters.n_eligible[e] += 1

        jnr = compute_jnr_db_s6(
            self.physics, self.radar, self.jammer,
            jammer_active=is_jam,
            radar_beam_az_idx=radar_beam % 5, radar_beam_el_idx=radar_beam // 5,
            jammer_beam_az_idx=jammer_beam % 5, jammer_beam_el_idx=jammer_beam // 5,
            cell_mask=executed_cell,
            radar_az_rad=self._radar_az, radar_el_rad=self._radar_el,
            victim_service_id=radar_svc,
        )  # [E, R]
        snr_eff = compute_snr_eff_db_s6(
            self.physics, baseline_snr_db=self.cfg.baseline_snr_db, jnr_db=jnr)  # [E, R]
        # per-radar target gains toward ALL az bearings (missions live on the
        # az grid at el=0): tg[e, r, a] couples the radar's pointing to BOTH
        # detection gain (here) and jamming exposure (inside JNR above)
        tg = torch.stack([
            target_gain_db(self.radar, beam_az_idx=radar_beam % 5,
                           beam_el_idx=radar_beam // 5,
                           mission_az_idx=torch.full_like(radar_beam, a))
            for a in range(N_AZ)], dim=-1)  # [E, R, N_AZ]
        thr = float(self.physics.detect_threshold_db)
        width = float(self.physics.detect_width_db)

        detected = torch.zeros(E, R, device=self.device)  # per-radar contribution flags
        for e in range(E):
            for m in list(self.tracker.pending[e]):
                svc_m, az_m = int(m[0]), int(m[1])
                for r in range(R):
                    if int(radar_svc[e, r].item()) != svc_m:
                        continue
                    p = float(torch.sigmoid((snr_eff[e, r] + tg[e, r, az_m] - thr) / width).item())
                    draw = float(torch.rand((), generator=self._detector_gen,
                                            device=self.device).item())
                    if draw < p:
                        detected[e, r] = 1.0
                        self.tracker.detect(env_idx=e, service_id=svc_m, az_idx=az_m)
                        break  # one credit per mission per step

        self.intercept_confidence = torch.where(
            is_jam,
            (1.0 - torch.sigmoid((snr_eff - thr) / width)).mean(dim=-1),
            torch.zeros_like(self.intercept_confidence))
        self.intercept_age = torch.where(
            is_jam, torch.zeros_like(self.intercept_age), self.intercept_age + 1)

        newly_dropped = torch.zeros(E, dtype=torch.int64, device=self.device)
        newly_succeeded = torch.zeros(E, dtype=torch.int64, device=self.device)
        for e in range(E):
            n_suc = int(self.counters.n_success[e].item())
            n_to = int(self.counters.n_timeout[e].item())
            self.tracker.finalize_step(env_idx=e, step=self.step_idx, counters=self.counters)
            newly_succeeded[e] = int(self.counters.n_success[e].item()) - n_suc
            newly_dropped[e] = int(self.counters.n_timeout[e].item()) - n_to

        phi_after = self._potential()
        self.step_idx += 1
        raw_drop = newly_dropped.float()
        reward_j = raw_drop + (self.cfg.gamma * phi_after - phi_before)
        reward_r = (newly_succeeded.float() - raw_drop)  # opposing-interest signal

        self.prev_cell = executed_cell.clone()
        self.prev_beam = jammer_beam.clone()
        self.prev_radar_beam = radar_beam.clone()
        self.prev_radar_svc = radar_svc.clone()
        self.radar_detected_last = detected.clone()
        obs_j, obs_r = self._build_observation()
        done = self.step_idx >= self.H
        if done:
            for e in range(E):
                self.tracker.finalize_horizon(env_idx=e, counters=self.counters)
            self._done_flag = True

        info = {
            "raw_drop": newly_dropped, "newly_succeeded": newly_succeeded,
            "reward_jammer": reward_j, "reward_radar": reward_r,
            "snr_eff_db": snr_eff, "jnr_db": jnr, "detected": detected,
            "mask_cell": mask_cell, "mask_beam": mask_beam,
            "is_jam": is_jam, "n_active_cells": n_active,
            "tokens_consumed": tokens_consumed,
            "energy_tokens_after": self.energy_tokens.clone(),
            "step_idx": self.step_idx - 1,
        }
        return (obs_j, obs_r), (reward_j, reward_r), \
            torch.full((E,), done, dtype=torch.bool, device=self.device), info

    def _potential(self) -> torch.Tensor:
        phi = torch.zeros(self.E, dtype=torch.float32, device=self.device)
        coef = float(self.cfg.potential_coef)
        for e in range(self.E):
            phi[e] = -coef * float(self.tracker.pending_count(e))
        return phi

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

    def success_ratio(self) -> torch.Tensor:
        denom = self.counters.n_eligible.clamp(min=1)
        return self.counters.n_success.float() / denom.float()

    def ledger_identity_residual(self) -> int:
        return (int(self.counters.n_eligible.sum().item())
                - (int(self.counters.n_success.sum().item())
                   + int(self.counters.n_timeout.sum().item())
                   + int(self.counters.n_admission_reject.sum().item())
                   + int(self.counters.n_horizon_failure.sum().item())))

    def sample_action_rng(self):
        cell = torch.randint(0, 2, (self.E, N_CELLS_S6), generator=self._action_gen,
                             device=self.device, dtype=torch.float32)
        jbeam = torch.randint(0, N_BEAM_DIRS_S6, (self.E,), generator=self._action_gen,
                              device=self.device, dtype=torch.int64)
        rbeam = torch.randint(0, N_BEAM_DIRS_S6, (self.E, self.R), generator=self._action_gen,
                              device=self.device, dtype=torch.int64)
        rsvc = torch.randint(0, 2, (self.E, self.R), generator=self._action_gen,
                             device=self.device, dtype=torch.int64)
        return cell, jbeam, rbeam, rsvc
