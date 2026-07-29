"""Mission-drop accounting for G3-BSTA-lite debug env (F1 §8).

Per DEBUG_CONTRACT.md §8, the accounting identity is

    eligible_mission_arrivals
      = mission_success
      + mission_timeout
      + mission_admission_reject
      + mission_horizon_failure

and the primary metric is

    mission_drop_ratio =
      (mission_timeout + mission_admission_reject + preregistered_horizon_failures)
      / eligible_mission_arrivals

The Five mutually-exclusive counters are tracked per env and per service.
``event_log`` records each arrival's disposition so that the accounting
identity can be re-computed exactly from event rows (test_metric_accounting).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


DISPO_SUCCESS = "success"
DISPO_TIMEOUT = "timeout"
DISPO_ADMISSION_REJECT = "admission_reject"
DISPO_HORIZON_FAILURE = "horizon_failure"


@dataclass
class MissionArrivalEvent:
    step: int
    service_id: int
    deadline_step: int
    disposition: str = "pending"


@dataclass
class MissionCounterBatch:
    """Per-env tensor counters."""

    n_eligible: torch.Tensor         # [E] int64
    n_success: torch.Tensor          # [E] int64
    n_timeout: torch.Tensor          # [E] int64
    n_admission_reject: torch.Tensor # [E] int64
    n_horizon_failure: torch.Tensor  # [E] int64

    @classmethod
    def zeros(cls, n_envs: int, device: str = "cpu") -> "MissionCounterBatch":
        return cls(
            n_eligible=torch.zeros(n_envs, dtype=torch.int64, device=device),
            n_success=torch.zeros(n_envs, dtype=torch.int64, device=device),
            n_timeout=torch.zeros(n_envs, dtype=torch.int64, device=device),
            n_admission_reject=torch.zeros(n_envs, dtype=torch.int64, device=device),
            n_horizon_failure=torch.zeros(n_envs, dtype=torch.int64, device=device),
        )

    def drop_ratio(self) -> torch.Tensor:
        """[E] float32 mission_drop_ratio. NaN where n_eligible==0."""
        num = self.n_timeout + self.n_admission_reject + self.n_horizon_failure
        denom = self.n_eligible.float()
        out = torch.where(
            denom > 0,
            num.float() / denom.clamp(min=1.0),
            torch.full_like(denom, float("nan")),
        )
        return out

    def accounting_residual(self) -> torch.Tensor:
        """[E] int64 residual of the accounting identity (must be zero)."""
        rhs = self.n_success + self.n_timeout + self.n_admission_reject + self.n_horizon_failure
        return self.n_eligible - rhs


@dataclass
class MissionTracker:
    """Per-env, per-service pending mission queues.

    A "mission" arrives at a service with a deadline. The radar opponent
    must complete enough successful detections on that service before the
    deadline; otherwise the mission is dropped.

    For the debug profile, the admission rule is "always admit" (the
    jammer cannot prevent arrivals). The completion rule is "at least
    ``detects_required`` successful detections of the service between
    arrival and deadline".

    Admission rejection is included in the accounting identity for forward
    compatibility with the controlled expansion (W6) where the jammer may
    shape the eligibility rule; in F1..F5 it is always zero.
    """

    n_envs: int
    n_services: int
    detects_required: int = 1
    # pending[e] is a list of (arrival_step, deadline_step, detects_so_far)
    pending: list[list[tuple[int, int, int, int]]] = field(default_factory=list)

    def initialize(self):
        self.pending = [[] for _ in range(self.n_envs)]

    def admit(
        self,
        *,
        env_idx: int,
        step: int,
        service_id: int,
        deadline_step: int,
    ):
        self.pending[env_idx].append((service_id, step, deadline_step, 0))

    def record_detection(self, *, env_idx: int, step: int, service_id: int, detected: bool):
        """Increment detects_so_far for matching pending missions.

        Detection applies to the radar's currently-scanned service. Only
        missions for that service receive credit; others are unchanged.
        """
        if not detected:
            return
        new_pending = []
        for (svc, arr, dl, ds) in self.pending[env_idx]:
            if svc == service_id:
                ds2 = ds + 1
            else:
                ds2 = ds
            new_pending.append((svc, arr, dl, ds2))
        self.pending[env_idx] = new_pending

    def finalize_step(
        self,
        *,
        env_idx: int,
        step: int,
        counters: MissionCounterBatch,
    ):
        """Apply deadline expiry at end of step ``step``.

        Missions whose deadline_step <= step and have detects_so_far >=
        detects_required become success; otherwise become timeout.

        Missions still pending at horizon are NOT finalized here; they are
        flushed by ``finalize_horizon`` to ``horizon_failure``.

        R1D: per-env lists ``_last_finalize_step`` / ``_last_finalize_timeout``
        record which (svc, arrival_step, dl, ds) tuples were finalized on
        this call, so the per-mission event ledger can stamp the right
        disposition on the right identity. The lists are reset at the top
        of each call.
        """
        if not hasattr(self, "_last_finalize_step"):
            self._last_finalize_step = {env_idx: [] for env_idx in range(self.n_envs)}
            self._last_finalize_timeout = {env_idx: [] for env_idx in range(self.n_envs)}
            self._last_finalize_horizon = {env_idx: [] for env_idx in range(self.n_envs)}
        self._last_finalize_step[env_idx] = []
        self._last_finalize_timeout[env_idx] = []
        keep: list[tuple[int, int, int, int]] = []
        for (svc, arr, dl, ds) in self.pending[env_idx]:
            if dl <= step:
                if ds >= self.detects_required:
                    counters.n_success[env_idx] += 1
                    self._last_finalize_step[env_idx].append((svc, arr, dl, ds))
                else:
                    counters.n_timeout[env_idx] += 1
                    self._last_finalize_timeout[env_idx].append((svc, arr, dl, ds))
            else:
                keep.append((svc, arr, dl, ds))
        self.pending[env_idx] = keep

    def finalize_horizon(
        self,
        *,
        env_idx: int,
        counters: MissionCounterBatch,
    ):
        if not hasattr(self, "_last_finalize_horizon"):
            self._last_finalize_step = {env_idx: [] for env_idx in range(self.n_envs)}
            self._last_finalize_timeout = {env_idx: [] for env_idx in range(self.n_envs)}
            self._last_finalize_horizon = {env_idx: [] for env_idx in range(self.n_envs)}
        self._last_finalize_horizon[env_idx] = []
        for (svc, arr, dl, ds) in self.pending[env_idx]:
            counters.n_horizon_failure[env_idx] += 1
            self._last_finalize_horizon[env_idx].append((svc, arr, dl, ds))
        self.pending[env_idx] = []

    def pending_count(self, env_idx: int) -> int:
        return len(self.pending[env_idx])

    def pending_count_per_service(self, env_idx: int) -> torch.Tensor:
        out = torch.zeros(self.n_services, dtype=torch.int64)
        for (svc, _, _, _) in self.pending[env_idx]:
            out[svc] += 1
        return out
