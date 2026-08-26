"""Fixed-capacity S7 mission tracker backend.

This backend preserves S6/S7 list-tracker semantics while moving mission
state and aggregate queries to fixed tensors. The public ``pending`` property
materializes the logical FIFO list for the existing detector/tests, so the
first D1 integration does not change detector traversal or RNG consumption.

Capacity is a detector-phase bound: at most one arrival per service per step,
with a mission still live on its deadline step, hence
n_services * min(horizon, tau_window + 1).
"""
from __future__ import annotations

import torch

from env.gpu.g3_bsta_lite.metrics import MissionCounterBatch
from env.gpu.array_face_s6.array_factor import N_AZ


class FixedS7MissionTracker:
    """Tensor-backed FIFO queue with the S6MissionTracker public API."""

    def __init__(self, *, n_envs: int, n_services: int,
                 detects_required: int = 1, capacity: int = 14,
                 device: str = "cpu"):
        self.n_envs = int(n_envs)
        self.n_services = int(n_services)
        self.detects_required = int(detects_required)
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("fixed mission tracker capacity must be positive")
        self.device = torch.device(device)
        self.initialize()

    def initialize(self):
        E, M = self.n_envs, self.capacity
        dev = self.device
        self.active = torch.zeros((E, M), dtype=torch.bool, device=dev)
        self.service = torch.zeros((E, M), dtype=torch.int64, device=dev)
        self.azimuth = torch.zeros((E, M), dtype=torch.int64, device=dev)
        self.arrival = torch.zeros((E, M), dtype=torch.int64, device=dev)
        self.deadline = torch.zeros((E, M), dtype=torch.int64, device=dev)
        self.detect_count = torch.zeros((E, M), dtype=torch.int64, device=dev)
        # Strictly increasing insertion sequence gives a stable logical FIFO
        # even when physical slots are reused after deadline finalization.
        self.sequence = torch.zeros((E, M), dtype=torch.int64, device=dev)
        self._next_sequence = torch.zeros(E, dtype=torch.int64, device=dev)

    def _ordered_slots(self, env_idx: int) -> list[int]:
        slots = torch.where(self.active[env_idx])[0].tolist()
        return sorted(slots, key=lambda s: int(self.sequence[env_idx, s].item()))

    def admit(self, *, env_idx: int, step: int, service_id: int,
              az_idx: int, deadline_step: int):
        free = torch.where(~self.active[env_idx])[0]
        if free.numel() == 0:
            raise RuntimeError(
                f"fixed mission queue overflow at env {env_idx}; "
                f"capacity={self.capacity}")
        slot = int(free[0].item())
        e = int(env_idx)
        self.active[e, slot] = True
        self.service[e, slot] = int(service_id)
        self.azimuth[e, slot] = int(az_idx)
        self.arrival[e, slot] = int(step)
        self.deadline[e, slot] = int(deadline_step)
        self.detect_count[e, slot] = 0
        self.sequence[e, slot] = self._next_sequence[e]
        self._next_sequence[e] += 1

    def detect(self, *, env_idx: int, service_id: int, az_idx: int):
        """Credit one oldest incomplete exact (service, azimuth) mission."""
        e = int(env_idx)
        for slot in self._ordered_slots(e):
            if (int(self.service[e, slot].item()) == int(service_id)
                    and int(self.azimuth[e, slot].item()) == int(az_idx)
                    and int(self.detect_count[e, slot].item()) < self.detects_required):
                self.detect_count[e, slot] += 1
                return True
        return False

    def finalize_step(self, *, env_idx: int, step: int,
                      counters: MissionCounterBatch):
        e = int(env_idx)
        for slot in self._ordered_slots(e):
            if int(self.deadline[e, slot].item()) <= int(step):
                if int(self.detect_count[e, slot].item()) >= self.detects_required:
                    counters.n_success[e] += 1
                else:
                    counters.n_timeout[e] += 1
                self.active[e, slot] = False

    def finalize_horizon(self, *, env_idx: int, counters: MissionCounterBatch):
        e = int(env_idx)
        n = int(self.active[e].sum().item())
        counters.n_horizon_failure[e] += n
        self.active[e] = False

    def pending_count(self, env_idx: int) -> int:
        return int(self.active[int(env_idx)].sum().item())

    def pending_count_per_service(self, env_idx: int):
        e = int(env_idx)
        out = torch.zeros(self.n_services, dtype=torch.int64, device=self.device)
        slots = torch.where(self.active[e])[0]
        if slots.numel():
            out.scatter_add_(0, self.service[e, slots], torch.ones_like(slots))
        return out

    def pending_count_per_service_batched(self):
        out = torch.zeros((self.n_envs, self.n_services), dtype=torch.int64,
                          device=self.device)
        vals = self.service.clamp(min=0, max=self.n_services - 1)
        out.scatter_add_(1, vals, self.active.to(torch.int64))
        return out

    def pending_az_map(self, env_idx: int, device=None) -> torch.Tensor:
        e = int(env_idx)
        dev = self.device if device is None else torch.device(device)
        out = torch.zeros(self.n_services, N_AZ, dtype=torch.float32, device=dev)
        slots = torch.where(self.active[e])[0]
        if slots.numel():
            flat = self.service[e, slots] * N_AZ + self.azimuth[e, slots]
            out.view(-1).scatter_add_(0, flat.to(dev),
                                      torch.ones(slots.numel(), device=dev))
        return out

    def pending_az_map_batched(self, device=None) -> torch.Tensor:
        dev = self.device if device is None else torch.device(device)
        out = torch.zeros((self.n_envs, self.n_services, N_AZ),
                          dtype=torch.float32, device=dev)
        env_idx, slots = torch.where(self.active)
        if env_idx.numel():
            flat = (self.service[env_idx, slots] * N_AZ
                    + self.azimuth[env_idx, slots]).to(dev)
            # Build a single global index [env, service, az].
            global_idx = env_idx.to(dev) * (self.n_services * N_AZ) + flat
            out.view(-1).scatter_add_(0, global_idx,
                                      torch.ones(env_idx.numel(), device=dev))
        return out

    @property
    def pending(self) -> list[list[list[int]]]:
        """Compatibility materialization in the original FIFO tuple format."""
        out: list[list[list[int]]] = []
        for e in range(self.n_envs):
            rows = []
            for slot in self._ordered_slots(e):
                rows.append([
                    int(self.service[e, slot].item()), int(self.azimuth[e, slot].item()),
                    int(self.arrival[e, slot].item()), int(self.deadline[e, slot].item()),
                    int(self.detect_count[e, slot].item()),
                ])
            out.append(rows)
        return out

    @pending.setter
    def pending(self, value: list[list[list[int]]]):
        """Load a compatible list representation (used by direct test fixtures)."""
        self.initialize()
        for e, rows in enumerate(value):
            for row in rows:
                svc, az, arr, dl, ds = row
                self.admit(env_idx=e, step=arr, service_id=svc,
                           az_idx=az, deadline_step=dl)
                slot = self._ordered_slots(e)[-1]
                self.detect_count[e, slot] = int(ds)


S7MissionTracker = FixedS7MissionTracker

__all__ = ["FixedS7MissionTracker", "S7MissionTracker"]
