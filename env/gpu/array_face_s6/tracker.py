"""S6 mission tracker — 5-tuple missions with BEARINGS + per-mission credit.

Extends the lite MissionTracker with the scan-vs-stare physics hinge:
  - every mission carries an az grid index (its bearing, el = 0)
  - credit is PER (service, az): only the mission actually looked at gets a
    detection (the parent credits all missions of a service)

Accounting semantics (identity, counters, horizon flush) are identical to
env/gpu/g3_bsta_lite/metrics.py — MissionCounterBatch is reused unchanged.
"""
from __future__ import annotations

from env.gpu.g3_bsta_lite.metrics import MissionCounterBatch
from env.gpu.array_face_s6.array_factor import N_AZ


class S6MissionTracker:
    """Per-env pending queues of (svc, az_idx, arr, dl, detects_so_far)."""

    def __init__(self, *, n_envs: int, n_services: int, detects_required: int = 1):
        self.n_envs = n_envs
        self.n_services = n_services
        self.detects_required = detects_required
        self.initialize()

    def initialize(self):
        # pending[e]: list of [svc, az_idx, arr, dl, ds] (lists: mutated in place)
        self.pending: list[list[list[int]]] = [[] for _ in range(self.n_envs)]

    def admit(self, *, env_idx: int, step: int, service_id: int, az_idx: int, deadline_step: int):
        self.pending[env_idx].append([service_id, az_idx, step, deadline_step, 0])

    def detect(self, *, env_idx: int, service_id: int, az_idx: int):
        """Credit ONE mission: exact (svc, az) match (oldest first)."""
        for m in self.pending[env_idx]:
            if m[0] == service_id and m[1] == az_idx and m[4] < self.detects_required:
                m[4] += 1
                return True
        return False

    def finalize_step(self, *, env_idx: int, step: int, counters: MissionCounterBatch):
        keep: list[list[int]] = []
        for m in self.pending[env_idx]:
            svc, az, arr, dl, ds = m
            if dl <= step:
                if ds >= self.detects_required:
                    counters.n_success[env_idx] += 1
                else:
                    counters.n_timeout[env_idx] += 1
            else:
                keep.append(m)
        self.pending[env_idx] = keep

    def finalize_horizon(self, *, env_idx: int, counters: MissionCounterBatch):
        for _ in self.pending[env_idx]:
            counters.n_horizon_failure[env_idx] += 1
        self.pending[env_idx] = []

    def pending_count(self, env_idx: int) -> int:
        return len(self.pending[env_idx])

    def pending_count_per_service(self, env_idx: int):
        out = [0] * self.n_services
        for m in self.pending[env_idx]:
            out[m[0]] += 1
        import torch
        return torch.tensor(out, dtype=torch.int64)

    def pending_az_map(self, env_idx: int, device=None) -> "torch.Tensor":
        """[n_services, N_AZ] pending counts — the scheduling map."""
        import torch
        out = torch.zeros(self.n_services, N_AZ, dtype=torch.float32,
                          device=device)
        for m in self.pending[env_idx]:
            out[m[0], m[1]] += 1.0
        return out
