"""Detection chain for two-team env (WP-1 M1).

Replaces the god-view `z = true_pos + noise` (twoteam_env.py:582) with a
probabilistic search-detect chain:

  1. SNR at each candidate (own_aperture, enemy_radar) via two-way Friis
     + monostatic radar equation, degraded by IQ JNR (from iq_interference.py).
  2. P_detect = sigmoid((SNR_dB - threshold) / width). Bernoulli draw.
  3. False alarms: per azimuth cell × Bernoulli(P_fa), random (range, az).
  4. Measurement noise σ_range = c / (2·B·√SNR_lin)  (CRLB).
  5. Output [E, T, K_max, 2] cartesian z + mask + is_false_alarm.

Mirror-symmetric: all Bernoulli rolls team-shared via
`rand(E, 1, R, R).expand(E, T, R, R)` (same pattern as
`twoteam_env.py:503` homejam_roll).

API:
    Detections = detect(...)                     # produce this step's detections
    z_assoc, mask_assoc = Detections.find_assoc(tracker_x_pred)
                                                 # nearest-neighbor by Euclidean
                                                 # distance (M4 replaces with PDAF).

This is M1 minimum-viable — no JPDA, no IMM, no gating. Those land in M4.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass


SPEED_OF_LIGHT = 299_792_458.0
BOLTZMANN_K = 1.38e-23
T0_KELVIN = 290.0


def _wrap(az: torch.Tensor) -> torch.Tensor:
    """Wrap azimuth to [-π, +π]."""
    return torch.atan2(torch.sin(az), torch.cos(az))


def _sinc2_db(rel_az: torch.Tensor, theta_3db: torch.Tensor) -> torch.Tensor:
    """sinc² beam pattern in dB, -30 dB sidelobe floor (matches iq_interference.py).

    Returns gain in dB relative to boresight (≤ 0).
    """
    x = 1.391 * rel_az / theta_3db.clamp(min=1e-4)
    sinc = torch.where(x.abs() < 1e-6, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))
    sinc2_db = 20.0 * torch.log10((sinc ** 2).clamp(min=1e-3))   # -30 dB floor
    return sinc2_db


def _sinc2_lin(rel_az: torch.Tensor, theta_3db: torch.Tensor) -> torch.Tensor:
    """sinc² beam pattern in linear scale, -30 dB sidelobe floor (≤ 1.0)."""
    return 10.0 ** (_sinc2_db(rel_az, theta_3db) / 10.0)


@dataclass
class Detections:
    """One step of detections for all envs and teams.

    z[E, T, K_max, 2]: cartesian (x, y) measurement positions (padded with 0).
    mask[E, T, K_max]: True at slots holding a real or false-alarm detection.
    is_false_alarm[E, T, K_max]: True if slot is a false alarm.
    snr_db[E, T, K_max]: SNR in dB for real detections (0 for FA / pad).
    """

    z: torch.Tensor
    mask: torch.Tensor
    is_false_alarm: torch.Tensor
    snr_db: torch.Tensor

    def find_assoc(
        self,
        tracker_x_pred: torch.Tensor,   # [E, T, R, 4]
    ):
        """Nearest-neighbor Euclidean association (M1 minimum viable).

        For each (env, team, own_radar r), pick the closest detection in (x, y)
        to the tracker prediction. M4 will replace this with Mahalanobis-gated
        PDAF association.

        Returns:
            z_assoc: [E, T, R, 2] — chosen measurement per track (0 if no assoc).
            mask_assoc: [E, T, R] — True if a real (non-FA) detection was associated.
            picked_fa: [E, T, R] — True if the associated detection was a false alarm.
        """
        E, T, R = tracker_x_pred.shape[:3]
        K = self.z.shape[2]
        dev = self.z.device

        pred_pos = tracker_x_pred[..., [0, 2]]   # [E, T, R, 2] — (x, y)

        z_assoc = torch.zeros(E, T, R, 2, device=dev, dtype=self.z.dtype)
        mask_assoc = torch.zeros(E, T, R, dtype=torch.bool, device=dev)
        picked_fa = torch.zeros(E, T, R, dtype=torch.bool, device=dev)

        for t in range(T):
            z_t = self.z[:, t]               # [E, K, 2]
            mask_t = self.mask[:, t]         # [E, K]
            fa_t = self.is_false_alarm[:, t]  # [E, K]
            for r in range(R):
                pred_r = pred_pos[:, t, r]   # [E, 2]
                # Distance [E, K]
                d = (z_t - pred_r.unsqueeze(1)).norm(dim=-1)
                d_masked = torch.where(mask_t, d, torch.full_like(d, 1e9))
                min_d, min_idx = d_masked.min(dim=-1)   # [E], [E]
                valid = min_d < 1e8

                # Gather z at min_idx
                idx_expanded = min_idx.view(-1, 1, 1).expand(-1, 1, 2)
                z_nearest = torch.gather(z_t, 1, idx_expanded).squeeze(1)   # [E, 2]
                fa_nearest = torch.gather(fa_t, 1, min_idx.unsqueeze(-1)).squeeze(-1)   # [E]

                z_assoc[:, t, r] = torch.where(valid.unsqueeze(-1), z_nearest, z_assoc[:, t, r])
                mask_assoc[:, t, r] = valid
                picked_fa[:, t, r] = valid & fa_nearest

        return z_assoc, mask_assoc, picked_fa


def detect(
    radar_pos: torch.Tensor,         # [E, T, R, 2]
    beam_az: torch.Tensor,           # [E, T, R]
    alloc: torch.Tensor,             # [E, T, R, 4]
    emission_on: torch.Tensor,       # [E, T, R] bool
    enemy_emitting: torch.Tensor,    # [E, T, R] bool (per-team enemy)
    radar_alive: torch.Tensor,       # [E, T, R] bool
    jnr_matrix: torch.Tensor,        # [E, N=4, N=4]
    *,
    range_max_m: float,
    fc_hz: float,
    channel_bw_hz: float,
    noise_figure_db: float,
    P_per_subarray_W: float,
    n_subarrays: int,
    aperture_D_m: float,
    aperture_eta: float,
    sigma_rcs_m2: float,
    detect_threshold_db: float,
    detect_width_db: float,
    p_fa: float,
    k_max: int,
    n_search_cells: int,
    beam_width_rad: float,
    coherent_processing_gain_db: float = 20.0,
    device,
) -> Detections:
    """Produce one step of detections. See module docstring for physics.

    coherent_processing_gain_db: pulse compression + coherent integration gain
    applied to the single-pulse radar-equation SNR. Realistic X-band pulse-
    Doppler radars achieve 13-23 dB from LFM/Barker compression + multi-pulse
    CPI; default 20 dB gives nominal SNR ~20 dB at 5 km (P_detect ~0.84 at
    threshold 15 dB, width 3 dB).
    """
    E, T, R = radar_pos.shape[0], radar_pos.shape[1], radar_pos.shape[2]
    dev = device

    # Derived constants
    lambda_m = SPEED_OF_LIGHT / fc_hz
    G_max_lin = 4.0 * math.pi * (aperture_D_m ** 2) * aperture_eta / (lambda_m ** 2)
    N_W = BOLTZMANN_K * T0_KELVIN * channel_bw_hz * (10.0 ** (noise_figure_db / 10.0))

    # Output buffers
    z_out = torch.zeros(E, T, k_max, 2, device=dev, dtype=radar_pos.dtype)
    mask_out = torch.zeros(E, T, k_max, dtype=torch.bool, device=dev)
    is_fa_out = torch.zeros(E, T, k_max, dtype=torch.bool, device=dev)
    snr_out = torch.zeros(E, T, k_max, device=dev, dtype=radar_pos.dtype)

    # ---- Mirror-symmetric Bernoulli rolls (team-shared) ----
    # Per (env, own_aperture, enemy_radar): real detection roll.
    roll_real = torch.rand(E, 1, R, R, device=dev).expand(E, T, R, R)
    # Per (env, azimuth cell): false alarm roll.
    roll_fa = torch.rand(E, 1, n_search_cells, device=dev).expand(E, T, n_search_cells)
    # Per (env, own_aperture, enemy_radar): measurement noise — team-shared so
    # tracker_x stays mirror-symmetric (without this, IMM-PDAF's weighted update
    # amplifies per-team noise into reward asymmetry that flunks §2.6⑥).
    meas_noise_shared = torch.randn(E, 1, R, R, 2, device=dev, dtype=radar_pos.dtype).expand(
        E, T, R, R, 2
    )

    for t in range(T):
        et = 1 - t

        # ---- Real detection candidates ----
        # Geometry [E, R_own=k, R_enemy=r, 2]
        own_pos = radar_pos[:, t]                              # [E, R, 2]
        enemy_pos = radar_pos[:, et]                          # [E, R, 2]
        delta = enemy_pos.unsqueeze(1) - own_pos.unsqueeze(2)   # [E, R_own, R_enemy, 2]
        d_er = delta.norm(dim=-1).clamp(min=100.0, max=range_max_m)   # [E, R, R]
        az_er = torch.atan2(delta[..., 1], delta[..., 0])             # [E, R, R]

        # Beam coverage per aperture
        beam_az_t = beam_az[:, t]                                # [E, R]
        rel_az = _wrap(az_er - beam_az_t.unsqueeze(-1))          # [E, R, R]
        in_beam = rel_az.abs() <= beam_width_rad                 # [E, R, R]

        # Active aperture (meaningful detect/track allocation + emitting)
        f_dt = alloc[:, t, :, 0] + alloc[:, t, :, 1]             # [E, R]
        emit_t = emission_on[:, t]                               # [E, R]
        active_k = (f_dt > 0.01) & emit_t                        # [E, R]
        active = active_k.unsqueeze(-1).expand(E, R, R)          # [E, R, R]

        # Enemy validity
        enemy_valid = (
            radar_alive[:, et].unsqueeze(1).expand(E, R, R)
            & enemy_emitting[:, et].unsqueeze(1).expand(E, R, R)
        )   # [E, R, R]

        # Tx power per aperture (Watts)
        f_emit = alloc[:, t, :, :3].sum(dim=-1).clamp(min=1.0 / n_subarrays)   # [E, R]
        P_tx_W = P_per_subarray_W * n_subarrays * f_emit                       # [E, R]
        P_tx_W_er = P_tx_W.unsqueeze(-1).expand(E, R, R)                       # [E, R, R]

        # Effective aperture + beamwidth (per iq_interference.py L123)
        D_eff = aperture_D_m * torch.sqrt(f_emit).unsqueeze(-1).expand(E, R, R)
        theta_3db = (0.886 * lambda_m / D_eff).clamp(min=1e-4)

        # Beam gains (sinc², -30 dB floor)
        G_tx_rel = _sinc2_lin(rel_az, theta_3db)               # [E, R, R]
        G_rx_rel = _sinc2_lin(-rel_az, theta_3db)              # return-path symmetric
        G_max_t = torch.tensor(G_max_lin, device=dev, dtype=radar_pos.dtype)
        G_tx = G_max_t * G_tx_rel
        G_rx = G_max_t * G_rx_rel

        # SNR (monostatic two-way radar equation, single-pulse baseline)
        # SNR_1p = P_tx · G_tx · G_rx · λ² · σ / ((4π)³ · R⁴ · k·T·B·F)
        denom = ((4.0 * math.pi) ** 3) * (d_er ** 4) * N_W
        snr_1p = (P_tx_W_er * G_tx * G_rx * (lambda_m ** 2) * sigma_rcs_m2) / denom.clamp(min=1e-30)
        # Pulse compression + coherent integration gain (dB → linear).
        proc_gain_lin = torch.tensor(
            10.0 ** (coherent_processing_gain_db / 10.0),
            device=dev, dtype=radar_pos.dtype,
        )
        snr_lin = snr_1p * proc_gain_lin

        # JNR inflation at victim (team t's aperture k = flat idx t*R + k)
        victim_indices = torch.arange(t * R, t * R + R, device=dev)
        # jnr_matrix[e, i, j] = JNR at victim j from interferer i
        # Sum over all interferers i for each victim j (skip self via diag=0)
        jnr_at_victim = jnr_matrix[:, :, victim_indices].sum(dim=1)   # [E, R]
        jnr_at_victim_er = jnr_at_victim.unsqueeze(-1).expand(E, R, R)

        snr_eff = snr_lin / (1.0 + jnr_at_victim_er.clamp(max=1e8))
        snr_db_er = 10.0 * torch.log10(snr_eff.clamp(min=1e-12))

        # P_detect (sigmoid)
        p_detect = torch.sigmoid((snr_db_er - detect_threshold_db) / detect_width_db)

        # Detection roll
        detect_roll = roll_real[:, t]                            # [E, R, R]
        detected = active & enemy_valid & in_beam & (detect_roll < p_detect)

        # σ_range (CRLB): σ = c / (2·B·√SNR_lin), clamp to [0.5, 500] m
        sigma_range = (
            SPEED_OF_LIGHT / (2.0 * channel_bw_hz * snr_eff.clamp(min=1e-3).sqrt())
        ).clamp(0.5, 500.0)   # [E, R, R]

        # Measurement z = enemy_pos + N(0, σ_range) — noise team-shared (meas_noise_shared)
        noise = meas_noise_shared[:, t] * sigma_range.unsqueeze(-1)
        enemy_pos_er = enemy_pos.unsqueeze(1).expand(E, R, R, 2)
        z_meas = enemy_pos_er + noise                            # [E, R_own, R_enemy, 2]

        # ---- False alarms ----
        fa_roll = roll_fa[:, t]                                  # [E, n_search_cells]
        fa_triggered = fa_roll < p_fa                            # [E, n_cells]

        cell_width = 2.0 * math.pi / n_search_cells
        cell_centers = (
            torch.arange(n_search_cells, device=dev, dtype=radar_pos.dtype) * cell_width - math.pi
        )
        fa_range = torch.rand(E, n_search_cells, device=dev, dtype=radar_pos.dtype) * range_max_m
        fa_az = cell_centers.unsqueeze(0).expand(E, n_search_cells)
        # FA position is absolute cartesian (own team centroid + polar offset)
        own_centroid = own_pos.mean(dim=1, keepdim=True)         # [E, 1, 2]
        fa_pos_abs = torch.stack(
            [fa_range * torch.cos(fa_az), fa_range * torch.sin(fa_az)],
            dim=-1,
        ) + own_centroid.expand(E, n_search_cells, 2)            # [E, n_cells, 2]

        # ---- Pack real + FA into [E, K_max] (real first, then FA) ----
        # Iterate per env (E is small ~32); vectorize later if needed.
        for e in range(E):
            count = 0
            # Real detections: row-major over (k, r)
            for k in range(R):
                for r in range(R):
                    if bool(detected[e, k, r]) and count < k_max:
                        z_out[e, t, count, 0] = z_meas[e, k, r, 0]
                        z_out[e, t, count, 1] = z_meas[e, k, r, 1]
                        mask_out[e, t, count] = True
                        is_fa_out[e, t, count] = False
                        snr_out[e, t, count] = snr_db_er[e, k, r]
                        count += 1
            # False alarms
            for c in range(n_search_cells):
                if bool(fa_triggered[e, c]) and count < k_max:
                    z_out[e, t, count, 0] = fa_pos_abs[e, c, 0]
                    z_out[e, t, count, 1] = fa_pos_abs[e, c, 1]
                    mask_out[e, t, count] = True
                    is_fa_out[e, t, count] = True
                    snr_out[e, t, count] = 0.0
                    count += 1

    return Detections(z=z_out, mask=mask_out, is_false_alarm=is_fa_out, snr_db=snr_out)
