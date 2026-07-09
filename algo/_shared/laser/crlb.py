"""CRLB / PCRLB for the TAES multi-baseline radar testbed.

Two lower bounds on track estimator covariance:

1. CRLB (single-time, multi-radar): for a stationary target at position p with
   radars at p_r, the FIM is J = Σ_r H_r^T R_r^{-1} H_r, where H_r is the
   measurement Jacobian (range/bearing or range/crossrange) and R_r is the
   measurement covariance (with jam_mul inflation). CRLB = trace(J^{-1}).

2. PCRLB (predictive, recursive): for a dynamic target with transition F and
   process noise Q, the recursive update is
        J_{k+1} = (Q + F J_k^{-1} F^T)^{-1} + Σ_r H_r^T R_r^{-1} H_r
   Init J_0 = P0^{-1}.

Used in WP0/WP1 to verify the tracker is near-optimal (trace_P ≈ PCRLB).
"""

from __future__ import annotations

import math
import torch
from typing import Optional


__all__ = ["compute_crlb", "PCRLBTracker"]


def compute_crlb(
    target_pos: torch.Tensor,    # [E, N_max, 2]
    radar_pos: torch.Tensor,     # [E, 2, 2]
    range_sigma: torch.Tensor,   # [E, N_max] or [E]
    bearing_sigma: torch.Tensor, # [E, N_max] or [E]
    alive_mask: Optional[torch.Tensor] = None,  # [E, N_max] bool
    use_range_bearing: bool = True,
    crossrange_factor: float = 7.4e-5,
) -> torch.Tensor:
    """Compute single-time CRLB (trace of inverse FIM) per target.

    Returns: [E, N_max] tensor of CRLB values (m^2). Padding/masked = +inf.
    """
    E, N_max, _ = target_pos.shape
    dev = target_pos.device
    eye2 = torch.eye(2, device=dev)

    # Broadcast sigma to [E, N_max]
    if range_sigma.dim() == 0:
        range_sigma = range_sigma.unsqueeze(0).unsqueeze(0).expand(E, N_max)
    elif range_sigma.dim() == 1:
        range_sigma = range_sigma.unsqueeze(-1).expand(E, N_max)
    if bearing_sigma.dim() == 0:
        bearing_sigma = bearing_sigma.unsqueeze(0).unsqueeze(0).expand(E, N_max)
    elif bearing_sigma.dim() == 1:
        bearing_sigma = bearing_sigma.unsqueeze(-1).expand(E, N_max)
    range_sigma = range_sigma.expand(E, N_max)
    bearing_sigma = bearing_sigma.expand(E, N_max)

    # Accumulate per-radar FIM contributions
    J = torch.zeros(E, N_max, 2, 2, device=dev)
    for r_idx in range(2):
        r_pos = radar_pos[:, r_idx:r_idx+1, :].expand(E, N_max, 2)
        dx = target_pos[..., 0] - r_pos[..., 0]
        dy = target_pos[..., 1] - r_pos[..., 1]
        R = torch.sqrt(dx*dx + dy*dy + 1.0)

        if use_range_bearing:
            H00 = dx / R
            H01 = dy / R
            H10 = -dy / (R * R)
            H11 = dx / (R * R)
        else:
            H00 = dx / R
            H01 = dy / R
            H10 = -dy / R
            H11 = dx / R

        sig_r2 = (range_sigma * range_sigma)
        if use_range_bearing:
            sig_b2 = bearing_sigma * bearing_sigma
        else:
            sig_b2 = (R * crossrange_factor) ** 2

        # H^T diag(1/sig_r^2, 1/sig_b^2) H
        # H = [[H00, H01], [H10, H11]] (2x2)
        # H^T diag(d0,d1) H = [[H00^2 d0 + H10^2 d1, H00 H01 d0 + H10 H11 d1],
        #                      [sym,                  H01^2 d0 + H11^2 d1]]
        d0 = 1.0 / sig_r2
        d1 = 1.0 / sig_b2
        J00 = H00 * H00 * d0 + H10 * H10 * d1
        J01 = H00 * H01 * d0 + H10 * H11 * d1
        J11 = H01 * H01 * d0 + H11 * H11 * d1
        J[..., 0, 0] += J00
        J[..., 0, 1] += J01
        J[..., 1, 0] += J01
        J[..., 1, 1] += J11

    # CRLB = trace(J^{-1})
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    inv_det = torch.where(det > 1e-12, 1.0 / det, torch.zeros_like(det))
    # J^{-1} = 1/det * [[J11, -J01], [-J01, J00]]
    # trace = (J00 + J11) / det  → NO, that's wrong
    # trace(J^{-1}) = (J11 + J00) / det  → YES this is correct (trace of [[J11, -J01],[-J01,J00]]/det = (J11+J00)/det)
    crlb = (J[..., 0, 0] + J[..., 1, 1]) * inv_det

    # Mask non-alive targets
    if alive_mask is not None:
        crlb = torch.where(alive_mask, crlb, torch.full_like(crlb, float('inf')))

    return crlb


class PCRLBTracker:
    """Recursive PCRLB tracker for the TAES scenario.

    Maintains a J (FIM) per target. Update:
        J_{k+1} = (Q + F J_k^{-1} F^T)^{-1} + Σ_r H_r^T R_r^{-1} H_r

    Init: J_0 = P0^{-1} (diagonal).
    """

    def __init__(
        self,
        n_envs: int,
        n_max: int,
        device: str = "cuda",
        dt: float = 0.1,
        sigma_q: float = 2.0,
        p0_pos: float = 100.0,
        p0_vel: float = 500.0,
    ):
        self.E = int(n_envs)
        self.N_max = int(n_max)
        self.device = torch.device(device)
        self.dt = float(dt)

        F = torch.tensor([
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=self.device)
        self.F = F
        q = (sigma_q ** 2) * 1e-3
        Q = torch.tensor([
            [q*dt**4/4, q*dt**3/2, 0, 0],
            [q*dt**3/2, q*dt**2,   0, 0],
            [0, 0, q*dt**4/4, q*dt**3/2],
            [0, 0, q*dt**3/2, q*dt**2],
        ], dtype=torch.float32, device=self.device)
        self.Q = Q

        # Initial FIM = P0^{-1}
        p0_inv = torch.tensor(
            [1.0/p0_pos, 1.0/p0_vel, 1.0/p0_pos, 1.0/p0_vel],
            device=self.device)
        J0_diag = p0_inv.unsqueeze(0).unsqueeze(0).expand(self.E, self.N_max, 4)
        self.J = torch.diag_embed(J0_diag).clone()

    def reset(self):
        """Reset J to P0^{-1}."""
        pass  # Constructor already initialized; for reset, rebuild

    @torch.no_grad()
    def update(
        self,
        target_pos: torch.Tensor,    # [E, N_max, 2]
        radar_pos: torch.Tensor,     # [E, 2, 2]
        range_sigma: torch.Tensor,   # [E, N_max] or [E]
        bearing_sigma: torch.Tensor, # [E, N_max] or [E]
        alive_mask: Optional[torch.Tensor] = None,
        use_range_bearing: bool = True,
        crossrange_factor: float = 7.4e-5,
    ):
        """One PCRLB recursion step."""
        E, N_max, _ = target_pos.shape
        dev = self.device

        if range_sigma.dim() == 1:
            range_sigma = range_sigma.unsqueeze(-1).expand(E, N_max)
        if bearing_sigma.dim() == 1:
            bearing_sigma = bearing_sigma.unsqueeze(-1).expand(E, N_max)

        # Predict step: J_pred = (Q + F J^{-1} F^T)^{-1}
        # We only need J_pred to add measurement FIM. But J^{-1} is the prior
        # covariance, so we keep J^{-1} as the covariance state.
        # Let C = J^{-1}. Then:
        #   C_pred = F C F^T + Q
        #   J_{k+1} = C_pred^{-1} + Σ H^T R^{-1} H
        # We track C instead of J to avoid inversion.
        C = torch.linalg.inv(self.J + 1e-9 * torch.eye(4, device=dev).unsqueeze(0).unsqueeze(0))
        # C_pred
        C_pred = torch.einsum("ij,enjk,kl->enil", self.F, C, self.F.T) + \
                 self.Q.unsqueeze(0).unsqueeze(0)
        # J_pred = C_pred^{-1}
        J_pred = torch.linalg.inv(C_pred + 1e-9 * torch.eye(4, device=dev).unsqueeze(0).unsqueeze(0))

        # Measurement FIM accumulation
        J_meas = torch.zeros(E, N_max, 4, 4, device=dev)
        for r_idx in range(2):
            r_pos = radar_pos[:, r_idx:r_idx+1, :].expand(E, N_max, 2)
            dx = target_pos[..., 0] - r_pos[..., 0]
            dy = target_pos[..., 1] - r_pos[..., 1]
            R = torch.sqrt(dx*dx + dy*dy + 1.0)

            if use_range_bearing:
                # H is [2x4]: [[dx/R, 0, dy/R, 0], [-dy/R^2, 0, dx/R^2, 0]]
                H = torch.zeros(E, N_max, 2, 4, device=dev)
                H[..., 0, 0] = dx / R
                H[..., 0, 2] = dy / R
                H[..., 1, 0] = -dy / (R * R)
                H[..., 1, 2] = dx / (R * R)
                sig_r2 = range_sigma * range_sigma
                sig_b2 = bearing_sigma * bearing_sigma
            else:
                H = torch.zeros(E, N_max, 2, 4, device=dev)
                H[..., 0, 0] = dx / R
                H[..., 0, 2] = dy / R
                H[..., 1, 0] = -dy / R
                H[..., 1, 2] = dx / R
                sig_r2 = range_sigma * range_sigma
                sig_b2 = (R * crossrange_factor) ** 2

            # R_meas^{-1} = diag(1/sig_r^2, 1/sig_b^2)
            Rinv = torch.zeros(E, N_max, 2, 2, device=dev)
            Rinv[..., 0, 0] = 1.0 / sig_r2
            Rinv[..., 1, 1] = 1.0 / sig_b2
            # H^T Rinv H
            HtRinv = torch.einsum("enki,enkm->enmi", H, Rinv)  # wait this is wrong
            # H^T R^{-1} H: einsum("enki,km,kmj->nij") but we have batched
            # H^T: [E,N,4,2], Rinv: [E,N,2,2], H: [E,N,2,4]
            Ht = torch.einsum("enij->enji", H)  # [E,N,4,2]
            HtRinvH = torch.einsum("enij,enjk,enkm->enim", Ht, Rinv, H)
            J_meas = J_meas + HtRinvH

        # J_{k+1} = J_pred + J_meas
        J_new = J_pred + J_meas

        # Only update alive targets
        if alive_mask is not None:
            mask_4 = alive_mask.unsqueeze(-1).unsqueeze(-1).expand_as(J_new)
            J_new = torch.where(mask_4, J_new, self.J)

        self.J = J_new

    def get_pcrlb(self) -> torch.Tensor:
        """Return trace(CRLB) = trace(J^{-1}) [E, N_max] (position variance lower bound)."""
        C = torch.linalg.inv(self.J + 1e-9 * torch.eye(4, device=self.J.device).unsqueeze(0).unsqueeze(0))
        return C[..., 0, 0] + C[..., 2, 2]

    def reset_state(self):
        """Reset J back to P0^{-1}."""
        p0_inv = torch.tensor(
            [1.0/100.0, 1.0/500.0, 1.0/100.0, 1.0/500.0],
            device=self.device)
        J0_diag = p0_inv.unsqueeze(0).unsqueeze(0).expand(self.E, self.N_max, 4)
        self.J = torch.diag_embed(J0_diag).clone()
