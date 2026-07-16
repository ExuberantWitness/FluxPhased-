"""Batched IMM-PDAF tracker for two-team env (WP-2 M1).

Per spec §3 ③: per-team batched IMM (CV+CT 2-model) + PDAF data association
(5σ Mahalanobis gate, probabilistic β_i weights).

Replaces the WP-1 single-model EKF + σ-gate NN (`_kalman_update_step_external`
in twoteam_env.py). The IMM-PDAF provides:
  - Better track quality under maneuvering targets (CV+CT mix)
  - Better clutter rejection via probabilistic association
  - Spec-compliant "competent blind classical" tracker

Mirror-symmetric RNG pattern: any random draw uses `rand(E, 1, R).expand(E, T, R)`
(same pattern as `twoteam_env.py:503` homejam_roll). IMM μ init / Bernoulli
detection share the pattern so both teams see identical stochasticity.

API:
    tracker = BatchedIMMPDAF(env)
    tracker.update(detections, sigma_meas)   # one step; writes back to env state

Reads from env:
    env.E, env.n_teams, env.n_radars_per_team, env.dt, env.sigma_q,
    env.device, env.tau_track

Writes to env (each step):
    env.tracker_x[E, T, R, 4]
    env.tracker_P[E, T, R, 4, 4]
    env.tracker_initialized[E, T, R]

Reference: Bar-Shalom et al., "Estimation with Applications to Tracking and
Navigation" (2001), Ch. 5-6 for IMM-PDAF equations.
"""

from __future__ import annotations

import math
import torch


class BatchedIMMPDAF:
    """Per-team batched IMM (CV+CT) + PDAF tracker.

    State (per env, team, own-radar/track-slot):
      tracker_x[E, T, R, 4]            — fused mean state [x, vx, y, vy]
      tracker_P[E, T, R, 4, 4]         — fused covariance
      tracker_initialized[E, T, R]     — bool; True after first measurement
      mu[E, T, R, 2]                   — IMM model probs [CV, CT], sums to 1
      x_models[E, T, R, 2, 4]          — per-model mean
      P_models[E, T, R, 2, 4, 4]       — per-model covariance
    """

    def __init__(
        self,
        env,
        omega_ct: float = 0.3,         # CT model turn rate (rad/s)
        gate_sigma: float = 5.0,       # PDAF Mahalanobis gate (5σ in 2D)
        markov_cv_to_cv: float = 0.97, # IMM Markov: P(CV→CV)
        markov_ct_to_ct: float = 0.97, # IMM Markov: P(CT→CT)
        init_P: float = 1.0,           # initial diagonal covariance
        pfa_floor: float = 1e-6,       # PDAF clutter density floor
    ):
        self.env = env
        self.E = env.E
        self.T = env.n_teams
        self.R = env.n_radars_per_team
        self.dt = float(env.dt)
        self.sigma_q = float(env.sigma_q)
        self.device = env.device
        self.omega_ct = float(omega_ct)
        self.gate_sigma = float(gate_sigma)
        self.init_P = float(init_P)
        self.pfa_floor = float(pfa_floor)
        self.markov = torch.tensor(
            [
                [markov_cv_to_cv, 1.0 - markov_cv_to_cv],
                [1.0 - markov_ct_to_ct, markov_ct_to_ct],
            ],
            device=self.device,
        )
        # WP-2 M1: per-slot real-association flag (consumed by env.step for fsld update).
        # True iff at least one model's gate contained a real (non-FA) detection this step.
        self.last_real_assoc = torch.zeros(self.E, self.T, self.R, dtype=torch.bool, device=self.device)
        # Eagerly init internal state so first update() doesn't need _sync_from_env
        # to detect a fresh reset. Also lets callers warm-start tracker_x/P before
        # stepping (test_laser_api_slot_semantics does this).
        self.x_models = torch.zeros(self.E, self.T, self.R, 2, 4, device=self.device)
        self.P_models = (
            torch.eye(4, device=self.device).expand(self.E, self.T, self.R, 2, 4, 4).clone() * self.init_P
        )
        self.mu = torch.full((self.E, self.T, self.R, 2), 0.5, device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, detections, sigma_meas):
        """One IMM-PDAF step. Reads env state, writes back updated tracker state.

        Args:
            detections: Detections dataclass (z[E,T,K,2], mask[E,T,K], snr_db[E,T,K])
            sigma_meas: [E, T, R] measurement σ floor from IQ physics
        """
        E, T, R, dev = self.E, self.T, self.R, self.device
        dt = self.dt

        # ---- Lazy init if env state shape changed (e.g., after env.reset) ----
        self._sync_from_env()

        # ---- IMM Step 1: mixing (compute mixed priors bar_x, bar_P per model) ----
        # mu_mixing_probs[E, T, R, i, j] = markov[i,j] * mu[i] / sum_i markov[i,j] * mu[i]
        mu = self.mu                                                          # [E, T, R, 2]
        # Mixed model prob c_j = sum_i M[i,j] * mu[i]   [E, T, R, 2]
        c_j = torch.einsum("ij,...i->...j", self.markov, mu)                 # [E, T, R, 2]
        c_j = c_j.clamp(min=1e-8)
        # Mixing weights mu_{i|j} = M[i,j] * mu[i] / c_j  [E, T, R, 2(i), 2(j)]
        mu_mix = (
            self.markov.unsqueeze(0).unsqueeze(0).unsqueeze(0)               # [1,1,1,2,2]
            * mu.unsqueeze(-1)                                                # [E,T,R,2,1]
            / c_j.unsqueeze(-2)                                               # [E,T,R,1,2]
        )                                                                     # [E,T,R,2(i),2(j)]

        # bar_x[..., j, :] = sum_i mu_mix[..., i, j] * x_models[..., i, :]
        bar_x = torch.einsum("...ij,...ik->...jk", mu_mix, self.x_models)    # [E,T,R,2,4]
        # bar_P[..., j, :, :] = sum_i mu_mix[..., i, j] * (P_i + spread_i_j)
        # spread_i_j = (x_i - bar_x_j) outer (x_i - bar_x_j)
        diff = self.x_models.unsqueeze(-2) - bar_x.unsqueeze(-3)              # [E,T,R,2(i),2(j),4]
        spread = torch.einsum("...ijk,...ijl->...ijkl", diff, diff)          # outer per (i,j)
        P_weighted = self.P_models.unsqueeze(-3).expand(E, T, R, 2, 2, 4, 4) + spread
        bar_P = torch.einsum("...ij,...ijmn->...jmn", mu_mix, P_weighted)     # [E,T,R,2(j),4,4]

        # Symmetrize
        bar_P = 0.5 * (bar_P + bar_P.transpose(-1, -2))

        # ---- IMM Step 2: per-model EKF predict + PDAF update ----
        x_pred = torch.zeros_like(bar_x)
        P_pred = torch.zeros_like(bar_P)
        for j in range(2):   # 0=CV, 1=CT
            F_j = self._transition_matrix(j, dt)                              # [4, 4]
            Q_j = self._process_noise(dt)                                     # [4, 4]
            # Predict
            x_pred_j = bar_x[..., j, :] @ F_j.T                               # [E, T, R, 4]
            P_pred_j = F_j @ bar_P[..., j, :, :] @ F_j.T + Q_j                # [E, T, R, 4, 4]
            x_pred[..., j, :] = x_pred_j
            P_pred[..., j, :, :] = P_pred_j

        # PDAF update per (team, slot) over all K detections
        x_updated = x_pred.clone()
        P_updated = P_pred.clone()
        likelihoods = torch.zeros(E, T, R, 2, device=dev)                     # for IMM μ update

        H = torch.zeros(2, 4, device=dev)
        H[0, 0] = 1.0
        H[1, 2] = 1.0

        for t in range(T):
            z_t = detections.z[:, t]                                          # [E, K, 2]
            mask_t = detections.mask[:, t]                                    # [E, K]
            is_fa_t = detections.is_false_alarm[:, t]                         # [E, K]
            for r in range(R):
                sigma_r = sigma_meas[:, t, r]                                 # [E]
                R_meas = (sigma_r ** 2).unsqueeze(-1).unsqueeze(-1) * torch.eye(
                    2, device=dev
                ).expand(E, 2, 2)                                             # [E, 2, 2]
                real_assoc_tr = torch.zeros(E, dtype=torch.bool, device=dev)
                for j in range(2):   # per model
                    x_pred_jr = x_pred[:, t, r, j]                            # [E, 4]
                    P_pred_jr = P_pred[:, t, r, j]                            # [E, 4, 4]
                    z_hat = x_pred_jr[:, [0, 2]]                              # [E, 2]
                    # Innovation per detection [E, K, 2]
                    nu = z_t - z_hat.unsqueeze(1)                             # [E, K, 2]
                    # S = H P H^T + R  [E, 2, 2]  (H picks position rows/cols 0 and 2)
                    # NOTE: must use H @ P @ H.T (NOT P[..., :2, :2]) since state is
                    # [x, vx, y, vy] and H = [[1,0,0,0],[0,0,1,0]] selects x and y.
                    S = H @ P_pred_jr @ H.T + R_meas                          # [E, 2, 2]
                    # Symmetrize + jitter for stability
                    S = 0.5 * (S + S.transpose(-1, -2)) + 1e-6 * torch.eye(
                        2, device=dev
                    ).expand(E, 2, 2)
                    S_inv = torch.linalg.inv(S)                               # [E, 2, 2]
                    # Mahalanobis distance d² [E, K]
                    d2 = torch.einsum("...ki,...ij,...kj->...k", nu, S_inv, nu)  # [E, K]
                    d2 = d2.clamp(min=0.0)
                    # Gate: 5σ in 2D = chi2inv(0.99, 2) ≈ 9.21, but use gate_sigma² for tunable
                    gate_pass = mask_t & (d2 < self.gate_sigma ** 2)          # [E, K]
                    # WP-2 M1: aggregate real-association across models for env.step fsld update.
                    real_inside_gate_j = (mask_t & ~is_fa_t & gate_pass).any(dim=-1)   # [E]
                    real_assoc_tr = real_assoc_tr | real_inside_gate_j

                    # PDAF β_i weights (clutter density from pfa_floor)
                    # β_i ∝ N(z_i | z_hat, S) = (2π |S|)^-1 exp(-d²/2)
                    # β_0 = p_fa * cell_volume (probability all gated are clutter)
                    S_det = torch.linalg.det(S).clamp(min=1e-12)              # [E]
                    norm_const = 1.0 / (2.0 * math.pi * S_det.sqrt())        # [E]
                    pdf_i = norm_const.unsqueeze(-1) * torch.exp(-0.5 * d2)   # [E, K]
                    pdf_i = torch.where(gate_pass, pdf_i, torch.zeros_like(pdf_i))
                    # Clutter density λ (per m² per cell): approximate from pfa / cell area
                    lambda_clutter = self.pfa_floor / max(
                        self.env.channel_bw_hz * 1e-6, 1.0
                    )   # rough proxy; ratio keeps dimensionless
                    lambda_clutter = max(lambda_clutter, 1e-12)
                    # β_i = pdf_i / lambda_clutter (within gate), β_0 = 1 - sum β_i
                    beta_i = (pdf_i / lambda_clutter).clamp(max=1e6)          # [E, K]
                    beta_sum = beta_i.sum(dim=-1)                             # [E]
                    beta_0 = (1.0 / (1.0 + beta_sum)).clamp(1e-6, 1.0)        # [E]
                    beta_i_norm = beta_i * (1.0 - beta_0).unsqueeze(-1) / beta_sum.clamp(
                        min=1e-12
                    ).unsqueeze(-1)                                           # [E, K]
                    # Zero out non-gated
                    beta_i_norm = torch.where(gate_pass, beta_i_norm, torch.zeros_like(beta_i_norm))

                    # Combined innovation [E, 2]
                    nu_combined = (beta_i_norm.unsqueeze(-1) * nu).sum(dim=-2)   # [E, 2]
                    # Kalman gain K = P H^T S^-1   [E, 4, 2]
                    K = P_pred_jr @ H.T @ S_inv                              # [E, 4, 2]
                    # State update
                    x_new = x_pred_jr + (nu_combined.unsqueeze(-2) @ K.transpose(-1, -2)).squeeze(-2)
                    # Cov update (PDAF spread)
                    # P_new = beta_0 * P_pred + (1 - beta_0) * (I - KH) P_pred
                    #        + spread term sum_i beta_i K nu_i nu_i^T K^T (additional spread)
                    I4 = torch.eye(4, device=dev).expand(E, 4, 4)
                    KH = K @ H                                                 # [E, 4, 4]
                    P_post = (I4 - KH) @ P_pred_jr                            # [E, 4, 4]
                    # Spread: sum_i β_i K ν_i ν_i^T K^T - K ν_comb ν_comb^T K^T
                    # For numerical stability, use simple PDAF covariance:
                    P_new = beta_0.unsqueeze(-1).unsqueeze(-1) * P_pred_jr + (
                        1.0 - beta_0
                    ).unsqueeze(-1).unsqueeze(-1) * P_post
                    # Add spread term: K (sum β_i ν_i ν_i^T - ν_comb ν_comb^T) K^T
                    nu_outer_sum = torch.einsum(
                        "...ki,...kj->...ij", beta_i_norm.unsqueeze(-1) * nu, nu
                    )                                                          # [E, 2, 2]
                    nu_comb_outer = nu_combined.unsqueeze(-1) * nu_combined.unsqueeze(-2)   # [E, 2, 2]
                    spread_post = K @ (nu_outer_sum - nu_comb_outer) @ K.transpose(-1, -2)
                    P_new = P_new + spread_post
                    P_new = 0.5 * (P_new + P_new.transpose(-1, -2))
                    P_new = P_new.clamp(-1e3, 1e3)

                    x_updated[:, t, r, j] = x_new
                    P_updated[:, t, r, j] = P_new

                    # Model likelihood (for IMM μ update):
                    # Λ_j = sum over gated detections of pdf_i + β_0 contribution
                    likelihood_j = beta_sum.clamp(min=1e-30)                   # [E]
                    likelihoods[:, t, r, j] = likelihood_j

                # WP-2 M1: write per-slot real-association flag (consumed by env.step fsld).
                self.last_real_assoc[:, t, r] = real_assoc_tr

                # Handle first-time initialization (slot uninitialized):
                # If any real detection exists and slot is uninitialized, init at
                # the highest-SNR real detection. Both models start at same point.
                uninit = ~self.env.tracker_initialized[:, t, r]                # [E]
                if uninit.any():
                    # Find a real (non-FA) detection for this (t, r) — use NN of any
                    # For batched simplicity: pick the highest-SNR non-FA detection.
                    # Note: detections.snr_db[E, T, K] — non-FA slots have SNR > 0.
                    snr_t = detections.snr_db[:, t]                            # [E, K]
                    is_real = mask_t & (~detections.is_false_alarm[:, t]) & (
                        snr_t > 0
                    )                                                          # [E, K]
                    # Default: -1 so argmax picks a real det if any
                    snr_masked = torch.where(is_real, snr_t, torch.full_like(snr_t, -1.0))
                    best_snr, best_idx = snr_masked.max(dim=-1)                # [E], [E]
                    has_real = best_snr >= 0
                    init_mask = uninit & has_real                             # [E]
                    if init_mask.any():
                        # Gather z at best_idx for init
                        z_init = torch.gather(
                            z_t, 1, best_idx.view(-1, 1, 1).expand(-1, 1, 2)
                        ).squeeze(1)                                          # [E, 2]
                        x_init = torch.zeros(E, 4, device=dev)
                        x_init[:, 0] = z_init[:, 0]
                        x_init[:, 2] = z_init[:, 1]
                        P_init = (
                            torch.eye(4, device=dev).expand(E, 4, 4).clone() * self.init_P
                        )
                        # Set both models identically
                        for j in range(2):
                            x_updated[:, t, r, j] = torch.where(
                                init_mask.unsqueeze(-1), x_init, x_updated[:, t, r, j]
                            )
                            P_updated[:, t, r, j] = torch.where(
                                init_mask[:, None, None],
                                P_init,
                                P_updated[:, t, r, j],
                            )
                        # Mark initialized
                        self.env.tracker_initialized[:, t, r] = self.env.tracker_initialized[:, t, r] | init_mask
                        # WP-2 M1: init-via-real-detection is also a real-association
                        # event (consumed by env.step fsld update; without this,
                        # proactive-detect bonus missed the first-detect step).
                        self.last_real_assoc[:, t, r] = self.last_real_assoc[:, t, r] | init_mask

        # ---- IMM Step 3: model probability update + fusion ----
        # μ_j_new ∝ c_j * Λ_j  (c_j from mixing step)
        mu_new = c_j * likelihoods                                            # [E, T, R, 2]
        mu_new = mu_new / mu_new.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        # Avoid collapse to one model (keep min prob 1e-3)
        mu_new = mu_new.clamp(min=1e-3, max=1.0 - 1e-3)
        mu_new = mu_new / mu_new.sum(dim=-1, keepdim=True)
        self.mu = mu_new

        # Fused mean: x = sum_j μ_j * x_j
        x_fused = (mu_new.unsqueeze(-1) * x_updated).sum(dim=-2)              # [E, T, R, 4]
        # Fused cov: P = sum_j μ_j * (P_j + (x_j - x)(x_j - x)^T)
        diff_fused = x_updated - x_fused.unsqueeze(-2)                        # [E, T, R, 2, 4]
        spread_fused = diff_fused.unsqueeze(-1) * diff_fused.unsqueeze(-2)    # [E,T,R,2,4,4]
        P_with_spread = P_updated + spread_fused                              # [E,T,R,2,4,4]
        P_fused = (mu_new.unsqueeze(-1).unsqueeze(-1) * P_with_spread).sum(dim=-3)
        P_fused = 0.5 * (P_fused + P_fused.transpose(-1, -2))
        P_fused = P_fused.clamp(-1e3, 1e3)

        # ---- Step 4: write back per-model state + fused state ----
        self.x_models = x_updated
        self.P_models = P_updated

        self.env.tracker_x = x_fused
        self.env.tracker_P = P_fused
        # tracker_initialized was updated in-place above

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sync_from_env(self):
        """Sync internal state to match env's tensor shapes (handles reset).

        Cases:
          1. Fresh reset (tracker_initialized all False): init internal state to defaults.
          2. Warm-started (caller wrote tracker_x/P/initialized before stepping, e.g.,
             test_laser_api_slot_semantics.py): seed x_models from env.tracker_x so
             IMM continues from the caller's belief.
          3. Mid-episode (some slots init, others not): seed init'd slots from env,
             leave uninit'd slots at default.
        """
        E, T, R, dev = self.E, self.T, self.R, self.device
        # Ensure env tensors exist at correct shape first
        if self.env.tracker_x.shape != (E, T, R, 4):
            self.env.tracker_x = torch.zeros(E, T, R, 4, device=dev)
        if self.env.tracker_P.shape != (E, T, R, 4, 4):
            self.env.tracker_P = (
                torch.eye(4, device=dev).expand(E, T, R, 4, 4).clone() * self.init_P
            )
        if self.env.tracker_initialized.shape != (E, T, R):
            self.env.tracker_initialized = torch.zeros(E, T, R, dtype=torch.bool, device=dev)

        # Fresh reset (nothing init'd): drop internal state to defaults
        if self.env.tracker_initialized.sum() == 0:
            self.x_models = torch.zeros(E, T, R, 2, 4, device=dev)
            self.P_models = (
                torch.eye(4, device=dev).expand(E, T, R, 2, 4, 4).clone() * self.init_P
            )
            self.mu = torch.full((E, T, R, 2), 0.5, device=dev)
            self.last_real_assoc = torch.zeros(E, T, R, dtype=torch.bool, device=dev)
            return

        # Warm-start / mid-episode: seed x_models / P_models from env where init'd.
        # If internal state isn't allocated yet (first call), allocate then seed.
        if not hasattr(self, "x_models") or self.x_models.shape != (E, T, R, 2, 4):
            self.x_models = torch.zeros(E, T, R, 2, 4, device=dev)
            self.P_models = (
                torch.eye(4, device=dev).expand(E, T, R, 2, 4, 4).clone() * self.init_P
            )
            self.mu = torch.full((E, T, R, 2), 0.5, device=dev)
        # Seed per-slot from env.tracker_x where init'd; preserve where uninit'd
        init_mask = self.env.tracker_initialized                              # [E, T, R]
        # Broadcast to [E, T, R, 2(models), 4]
        seed_mask = init_mask.unsqueeze(-1).unsqueeze(-1).expand_as(self.x_models)
        env_x_expanded = self.env.tracker_x.unsqueeze(-2).expand_as(self.x_models)
        self.x_models = torch.where(seed_mask, env_x_expanded, self.x_models)
        # Same for P [E, T, R, 2, 4, 4]
        seed_mask_P = init_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(self.P_models)
        env_P_expanded = self.env.tracker_P.unsqueeze(-3).expand_as(self.P_models)
        self.P_models = torch.where(seed_mask_P, env_P_expanded, self.P_models)

    def _transition_matrix(self, model: int, dt: float) -> torch.Tensor:
        """Return F [4, 4] for given model (0=CV, 1=CT)."""
        F = torch.eye(4, device=self.device)
        if model == 0:   # CV
            F[0, 1] = dt
            F[2, 3] = dt
        else:            # CT (coordinate turn, turn rate ω_ct)
            omega = self.omega_ct
            omega_dt = omega * dt
            sin_ot = math.sin(omega_dt)
            cos_ot = math.cos(omega_dt)
            one_minus_cos = 1.0 - cos_ot
            # CT transition (Bar-Shalom Eq. 5.46):
            # x' = x + (sin(ωdt)/ω) vx - ((1-cos(ωdt))/ω) vy
            # vx' = cos(ωdt) vx - sin(ωdt) vy
            # y' = y + ((1-cos(ωdt))/ω) vx + (sin(ωdt)/ω) vy
            # vy' = sin(ωdt) vx + cos(ωdt) vy
            F = torch.zeros(4, 4, device=self.device)
            F[0, 0] = 1.0
            F[0, 1] = sin_ot / omega
            F[0, 2] = 0.0
            F[0, 3] = -one_minus_cos / omega
            F[1, 0] = 0.0
            F[1, 1] = cos_ot
            F[1, 2] = 0.0
            F[1, 3] = -sin_ot
            F[2, 0] = 0.0
            F[2, 1] = one_minus_cos / omega
            F[2, 2] = 1.0
            F[2, 3] = sin_ot / omega
            F[3, 0] = 0.0
            F[3, 1] = sin_ot
            F[3, 2] = 0.0
            F[3, 3] = cos_ot
        return F

    def _process_noise(self, dt: float) -> torch.Tensor:
        """Q [4, 4] — shared across models (matches env._kalman_update_step_external)."""
        q = self.sigma_q ** 2
        Q = torch.eye(4, device=self.device) * q
        Q[0, 0] = q * dt ** 2 / 4
        Q[1, 1] = q * dt ** 2
        Q[2, 2] = q * dt ** 2 / 4
        Q[3, 3] = q * dt ** 2
        return Q
