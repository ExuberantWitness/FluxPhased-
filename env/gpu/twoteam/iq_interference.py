"""IQ-native co-channel interference physics for the two-team env.

Replaces the historical scalar `jam_mul` abstraction with Friis + beam-pattern +
frequency-overlap physics, computed on-device as a batched 4-radar JNR matrix.

Per-pair JNR (interferer i → victim j, i≠j):

    λ = c / fc
    d_ij = ‖pos_i − pos_j‖  (clamp ≥ 100 m)
    P_tx_i_W = P_per_subarray · n_subarrays · (alloc_i[detect]+alloc_i[track]+alloc_i[jam])
    D_eff_i  = D_full · sqrt(f_emit_i)              # subarray→aperture coupling
    θ_3db_i  = 0.886 · λ / D_eff_i
    G_tx_i(θ)= G_max · sinc²(1.391 · θ / θ_3db_i)   # -30 dB sidelobe floor
    L_path   = 20·log10(4π·d/λ)
    overlap  = rect_overlap(f_i, bw_i, f_j, bw_j) / bw_j
                / hop_rate_i                         # i spreads tx over hop subchannels
    J_ij_dBm = P_tx_i_dBm + G_tx_i_dB + G_rx_j_dB − L_path − L_pol + 10·log10(overlap)
    JNR_ij   = 10^((J_ij_dBm − N_j_dBm)/10)

Self-interference JNR[i,i] = 0 (perfect SIC).

Per-victim σ (victim j tracking target r):

    JNR_total_j = Σ_{i≠j} JNR_ij          (3 sources: 1 teammate + 2 enemies)
    σ_base_j    = range_sigma / sqrt(f_track_eff_j + 1e-3)
    σ_meas_j    = σ_base_j · sqrt(1 + JNR_total_j) · fusion_factor_j

All ops are NaN-guarded (clamp/where on every log/div/sqrt).
"""

from __future__ import annotations

import math
import torch
from typing import Dict


SPEED_OF_LIGHT = 299_792_458.0
N_RADARS = 4  # 2 teams × 2 radars per team, flat index = team * 2 + radar_slot


class IqInterference:
    """Batched torch IQ-native co-channel JNR physics.

    Stateless beyond constructor constants — safe to call from vectorized env.
    """

    def __init__(
        self,
        fc_hz: float = 10e9,
        channel_bw_hz: float = 10e6,
        noise_figure_db: float = 5.0,
        P_per_subarray_W: float = 5.0,
        aperture_D_m: float = 0.4,
        aperture_eta: float = 0.6,
        n_subarrays: int = 25,
        polarization_loss_db: float = 3.0,
        # WP-B Step 0: clamp raised 1e4 → 1e8 (80 dB).
        # At 1e4 (40 dB), full-power boresight coupling at 5km geometry saturated
        # immediately (JNR=82.7dB), flattening the [10,50]dB useful regime into a
        # single σ=5m point — "saturated calm sea". 1e8 keeps saturation reachable
        # only at extreme boresight-main-beam scenarios; tracker_P.clamp(-1e3,1e3)
        # in env still caps numerical blowup downstream.
        jnr_total_clamp: float = 1e8,
        distance_floor_m: float = 100.0,
    ):
        self.fc_hz = float(fc_hz)
        self.channel_bw_hz = float(channel_bw_hz)
        self.noise_figure_db = float(noise_figure_db)
        self.P_per_subarray_W = float(P_per_subarray_W)
        self.aperture_D_m = float(aperture_D_m)
        self.aperture_eta = float(aperture_eta)
        self.n_subarrays = int(n_subarrays)
        self.polarization_loss_db = float(polarization_loss_db)
        self.jnr_total_clamp = float(jnr_total_clamp)
        self.distance_floor_m = float(distance_floor_m)

        # Derived constants
        self.lambda_m = SPEED_OF_LIGHT / self.fc_hz
        # Square-aperture boresight gain: G = 4π·A_eff/λ² = 4π·D²·η/λ²
        self.G_max_db = 10.0 * math.log10(
            4.0 * math.pi * (self.aperture_D_m ** 2) * self.aperture_eta
            / (self.lambda_m ** 2)
        )
        # Thermal noise floor at victim (dBm)
        self.N_dbm = (
            -174.0
            + 10.0 * math.log10(self.channel_bw_hz)
            + self.noise_figure_db
        )

    # ---- public API --------------------------------------------------------

    def compute_jnr_matrix(
        self,
        pos: torch.Tensor,           # [E, T=2, R=2, 2]
        beam_az: torch.Tensor,       # [E, T, R]  continuous azimuth (rad)
        alloc: torch.Tensor,         # [E, T, R, 4]  fractions (detect/track/jam/comm)
        freq_hz: torch.Tensor,       # [E, T, R]  absolute tx center freq
        emission_on: torch.Tensor,   # [E, T, R]  bool
        hop_rate: torch.Tensor,      # [E, T, R]  ≥1.0
        radar_alive: torch.Tensor,   # [E, T, R]  bool
    ) -> torch.Tensor:
        """Return JNR linear matrix [E, N=4, N=4] where [e,i,j] = JNR at victim j from interferer i."""
        E = pos.shape[0]
        dev = pos.device
        N = N_RADARS

        # Flatten (T,R) → N. flat_idx = team * 2 + radar_slot.
        pos_flat = pos.reshape(E, N, 2)
        beam_az_flat = beam_az.reshape(E, N)
        alloc_flat = alloc.reshape(E, N, 4)
        freq_flat = freq_hz.reshape(E, N)
        emit_flat = emission_on.reshape(E, N).float()
        hop_flat = hop_rate.reshape(E, N)
        alive_flat = radar_alive.reshape(E, N)

        # f_emit = detect + track + jam (comm excluded — different waveform)
        f_emit = alloc_flat[..., :3].sum(dim=-1).clamp(min=0.0)  # [E,N]
        f_emit_safe = f_emit.clamp(min=1.0 / self.n_subarrays)

        # Effective aperture + beamwidth per radar
        D_eff = self.aperture_D_m * torch.sqrt(f_emit_safe)  # [E,N]
        theta_3db = (0.886 * self.lambda_m / D_eff).clamp(min=1e-4)  # [E,N]

        # Pairwise geometry: [E, N(i), N(j), 2] delta from i to j (= pos_j − pos_i)
        delta = pos_flat.unsqueeze(1) - pos_flat.unsqueeze(2)  # i,j
        d_ij = delta.norm(dim=-1).clamp(min=self.distance_floor_m)  # [E,N,N]

        # Azimuths i→j and j→i
        az_i_to_j = torch.atan2(delta[..., 1], delta[..., 0])  # [E,N,N]
        az_j_to_i = az_i_to_j + math.pi

        beam_az_i = beam_az_flat.unsqueeze(2).expand(E, N, N)  # [E,N,N] (i index)
        beam_az_j = beam_az_flat.unsqueeze(1).expand(E, N, N)  # [E,N,N] (j index)

        rel_az_i = _wrap(az_i_to_j - beam_az_i)
        rel_az_j = _wrap(az_j_to_i - beam_az_j)

        theta_3db_i = theta_3db.unsqueeze(2).expand(E, N, N)
        theta_3db_j = theta_3db.unsqueeze(1).expand(E, N, N)

        # Beam gain (sinc², -30 dB sidelobe floor, relative to G_max so ≤ 0 dB)
        G_tx_rel_db = _sinc2_db(rel_az_i, theta_3db_i)
        G_rx_rel_db = _sinc2_db(rel_az_j, theta_3db_j)

        G_max_db_t = torch.tensor(self.G_max_db, device=dev, dtype=pos.dtype)
        G_tx_total_db = G_max_db_t + G_tx_rel_db  # [E,N,N]
        G_rx_total_db = G_max_db_t + G_rx_rel_db

        # Tx power at interferer i
        P_tx_W = (self.P_per_subarray_W * self.n_subarrays * f_emit).clamp(min=1e-15)  # [E,N]
        P_tx_dBm = 10.0 * torch.log10(P_tx_W * 1000.0)  # [E,N]
        P_tx_dBm_i = P_tx_dBm.unsqueeze(2).expand(E, N, N)  # broadcast over j

        # Frequency overlap (victim-bandwidth fraction), then hop dilution on i
        overlap = _rect_overlap_frac(
            freq_flat.unsqueeze(2).expand(E, N, N),  # f_i
            freq_flat.unsqueeze(1).expand(E, N, N),  # f_j
            self.channel_bw_hz,
        )  # [E,N,N], fraction of j's BW overlapped by i
        hop_i = hop_flat.unsqueeze(2).expand(E, N, N).clamp(min=1.0)
        overlap_eff = overlap / hop_i  # i spreads power across hop subchannels

        # Path loss (one-way Friis)
        L_path_db = 20.0 * torch.log10(4.0 * math.pi * d_ij / self.lambda_m)

        # Jamming power at victim j (dBm)
        overlap_term_db = 10.0 * torch.log10(overlap_eff + 1e-15)
        J_dBm = (
            P_tx_dBm_i
            + G_tx_total_db
            + G_rx_total_db
            - L_path_db
            - self.polarization_loss_db
            + overlap_term_db
        )

        # Gate: interferer must be alive + emitting
        emit_i = emit_flat.unsqueeze(2).expand(E, N, N)
        alive_i = alive_flat.unsqueeze(2).expand(E, N, N)
        active_i = (emit_i > 0.5) & alive_i
        J_dBm = torch.where(active_i, J_dBm, torch.full_like(J_dBm, -1e9))

        # JNR linear
        N_dbm_t = torch.tensor(self.N_dbm, device=dev, dtype=pos.dtype)
        JNR_dB = J_dBm - N_dbm_t
        JNR_lin = torch.pow(10.0, JNR_dB / 10.0)
        JNR_lin = torch.where(active_i, JNR_lin, torch.zeros_like(JNR_lin))

        # Mask diagonal (self-interference = 0, perfect SIC)
        diag_mask = (~torch.eye(N, dtype=torch.bool, device=dev)).unsqueeze(0).expand(E, N, N)
        JNR_lin = torch.where(diag_mask, JNR_lin, torch.zeros_like(JNR_lin))

        return JNR_lin  # [E,N,N]

    def compute_meas_sigma(
        self,
        jnr_matrix: torch.Tensor,    # [E,N,N]
        f_track_eff: torch.Tensor,   # [E,T,R]
        range_sigma: float,
        fusion_factor: torch.Tensor, # [E,T,R]
    ) -> torch.Tensor:
        """Per-victim measurement σ [E,T,R] from JNR matrix + base σ + fusion factor."""
        # Sum over interferers i (dim=1) for each victim j → [E,N]
        JNR_total = jnr_matrix.sum(dim=1).clamp(min=0.0, max=self.jnr_total_clamp)
        # Reshape [E,N] → [E,T,R]
        E = jnr_matrix.shape[0]
        JNR_total_TR = JNR_total.view(E, 2, 2)

        sigma_base = range_sigma / (f_track_eff + 1e-3).sqrt()
        sigma_meas = sigma_base * (1.0 + JNR_total_TR).sqrt() * fusion_factor
        return sigma_meas


# ---- helpers ---------------------------------------------------------------

def _wrap(angles: torch.Tensor) -> torch.Tensor:
    """Wrap radians to (-π, π] symmetrically (mirror-unbiased safe)."""
    return torch.remainder(angles + math.pi, 2.0 * math.pi) - math.pi


def _sinc2_db(rel_az_rad: torch.Tensor, theta_3db_rad: torch.Tensor) -> torch.Tensor:
    """sinc² beam pattern (uniform rectangular aperture) in dB, relative to boresight.

    G(θ)/G_max = [sin(1.391·θ/θ_3db) / (1.391·θ/θ_3db)]²  with -30 dB sidelobe floor.
    Returns ≤ 0 dB (relative gain).
    """
    x = 1.391 * rel_az_rad / theta_3db_rad
    # sinc(x) = sin(x)/x with x=0 limit = 1
    sx = torch.where(
        x.abs() < 1e-6,
        torch.ones_like(x),
        torch.sin(x) / (x + 1e-12),
    )
    gain = sx * sx
    # -30 dB sidelobe floor (gain ≥ 1e-3 in linear)
    gain = gain.clamp(min=1e-3)
    return 10.0 * torch.log10(gain)


def _rect_overlap_frac(f_i: torch.Tensor, f_j: torch.Tensor, bw: float) -> torch.Tensor:
    """Fraction of victim j's bandwidth overlapped by interferer i (symmetric bandwidth)."""
    half = bw / 2.0
    lo_i = f_i - half
    hi_i = f_i + half
    lo_j = f_j - half
    hi_j = f_j + half
    overlap_lo = torch.maximum(lo_i, lo_j)
    overlap_hi = torch.minimum(hi_i, hi_j)
    overlap_bw = (overlap_hi - overlap_lo).clamp(min=0.0)
    return overlap_bw / bw  # ∈ [0, 1]
