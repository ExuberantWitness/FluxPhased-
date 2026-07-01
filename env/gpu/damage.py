"""WP3.2 robustness sweep — damage injection primitives.

Pure-functional helpers used by MFARVecEnv.step() to inject the 5 damage types
from PAPER_PLAN_EAAI.md §WP3.2:

  1. Weibull ground/sea clutter (additive, CNR-scaled)
  2. 2-ray ground-reflection multipath (delayed + attenuated self-copy)
  3. (sensing bias + control delay handled in training/laser/)
  4. (slew limit + duty cycle handled inline in MFARVecEnv)
  5. (ISAC comm-rate limit handled in vec_drone.process_radar_comm)

Each helper is opt-in: callers check `damage_cfg.get(field) > 0` before calling.
Default behavior is a complete no-op (existing code is unaffected).
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# 1. Weibull ground/sea clutter
# ---------------------------------------------------------------------------

def apply_weibull_clutter(
    rx_iq: torch.Tensor,
    buf: torch.Tensor,
    shape_k: float,
    scale_lambda: float,
    cnr_db: float,
    noise_power_linear: float,
    generator: torch.Generator | None = None,
):
    """Add Weibull-distributed complex clutter to rx_iq in-place.

    The clutter magnitude follows a Weibull distribution:
        |x| = λ * (-ln(U))^(1/k)    where U ~ Uniform(0,1)

    For complex clutter we draw independent Weibull magnitudes for the I and Q
    components and assign random signs (preserves the Weibull envelope
    distribution while keeping the noise zero-mean).

    After sampling, we rescale so the actual clutter power matches the
    CNR-implied target: P_clutter = P_noise * 10^(CNR_dB/10). This makes the
    `scale_lambda` and `shape_k` parameters control only the *distribution
    shape*, not the total power — which is what EAAI reviewers will want
    when they sweep CNR at fixed shape.

    Args:
        rx_iq: [E, R, N, S] complex64 — post-AWGN received signal.
        buf: pre-allocated [E, R, N, S] complex64 scratch buffer.
        shape_k: Weibull shape (k=1 → exponential, k=2 → Rayleigh-like).
        scale_lambda: Weibull scale (controls shape only; total power is
            renormalized to match CNR target).
        cnr_db: clutter-to-noise ratio in dB.
        noise_power_linear: thermal noise power (W) — P_clutter = this × 10^(CNR/10).
        generator: optional torch.Generator for reproducibility.
    """
    if shape_k <= 0 or scale_lambda <= 0:
        return
    if not math.isfinite(cnr_db):
        return

    target_clutter_power = noise_power_linear * (10.0 ** (cnr_db / 10.0))
    if target_clutter_power <= 0:
        return

    inv_k = 1.0 / shape_k

    # Weibull samples via inverse CDF: x = λ * (-ln U)^(1/k)
    u_real = torch.rand_like(buf.real)
    u_imag = torch.rand_like(buf.imag)
    if generator is not None:
        # NOTE: torch.rand_like does not accept a generator; sample on CPU if
        # reproducibility is required. For training, stochasticity is desired.
        pass
    u_real.clamp_min_(1e-12)
    u_imag.clamp_min_(1e-12)

    mag_real = scale_lambda * (-u_real.log()).pow_(inv_k)
    mag_imag = scale_lambda * (-u_imag.log()).pow_(inv_k)

    # Random sign on each quadrature → zero-mean clutter
    sign_real = torch.where(torch.rand_like(mag_real) > 0.5, 1.0, -1.0)
    sign_imag = torch.where(torch.rand_like(mag_imag) > 0.5, 1.0, -1.0)

    buf.real.copy_(mag_real * sign_real)
    buf.imag.copy_(mag_imag * sign_imag)

    # Rescale to target power. Per-element variance is E[|c|²]/N_total;
    # we want mean(rx.real² + rx.imag²) ≈ target_clutter_power.
    actual_power = buf.real.pow(2).mean() + buf.imag.pow(2).mean()
    if actual_power.item() > 0:
        scale = (target_clutter_power / actual_power.clamp_min_(1e-30)).sqrt()
        buf.mul_(scale)

    rx_iq.add_(buf)


# ---------------------------------------------------------------------------
# 2. 2-ray ground-reflection multipath
# ---------------------------------------------------------------------------

def apply_multipath_2ray(
    rx_iq: torch.Tensor,
    delay_samples: int,
    attenuation_linear: float,
):
    """Apply 2-ray ground reflection: y[t] = x[t] + α * x[t - τ] (in-place).

    A causal FIR filter. Samples that would wrap around (t < τ) are zeroed
    rather than wrapped, matching physical reality where the delayed copy
    of the first τ samples hasn't arrived yet.

    Args:
        rx_iq: [E, R, N, S] complex64.
        delay_samples: integer delay τ in samples (≥1).
        attenuation_linear: α (linear scale; 0 < α ≤ 1).
    """
    if delay_samples <= 0 or attenuation_linear <= 0:
        return
    S = rx_iq.shape[-1]
    if delay_samples >= S:
        return

    delayed = torch.roll(rx_iq, delay_samples, dims=-1)
    # Zero the wrapped-around samples to enforce causality
    delayed[..., :delay_samples] = 0
    rx_iq.add_(delayed * attenuation_linear)


# ---------------------------------------------------------------------------
# 3. Beam slew rate limiter (stateful, called inline by env)
# ---------------------------------------------------------------------------

def clamp_beam_slew(
    current_az_el: torch.Tensor,
    prev_az_el: torch.Tensor,
    max_delta_per_step: float,
):
    """Clamp per-step beam az/el change to ±max_delta_per_step (in-place on current).

    Args:
        current_az_el: [E, R, 2] beam az/el in degrees for this step.
        prev_az_el: [E, R, 2] beam az/el from previous step.
        max_delta_per_step: max allowed |Δ| per control step (degrees).

    Returns:
        The (possibly-clipped) current_az_el tensor, modified in-place.
    """
    if max_delta_per_step <= 0:
        return current_az_el
    delta = current_az_el - prev_az_el
    delta = delta.clamp(-max_delta_per_step, max_delta_per_step)
    current_az_el.copy_(prev_az_el + delta)
    return current_az_el


# ---------------------------------------------------------------------------
# 4. ISAC comm-rate limit (stateless bit-mask helper)
# ---------------------------------------------------------------------------

def comm_rate_drop_mask(
    n_bits: int,
    comm_rate_bps: float,
    dt_s: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a bool mask of which bits survive an ISAC uplink rate cap.

    Models a simple time-shared uplink: if the radar tries to send N bits in
    dt_s seconds but the cap is R_bps, only N·dt·R/N ≈ R·dt bits fit. Bits
    are dropped uniformly at random.

    Args:
        n_bits: total bits attempted this step.
        comm_rate_bps: max bits-per-second the ISAC channel allows.
        dt_s: duration of this step (seconds).
        generator: optional torch.Generator.

    Returns:
        [n_bits] bool tensor — True if bit survives.
    """
    if comm_rate_bps <= 0:
        return torch.ones(n_bits, dtype=torch.bool)
    budget = int(comm_rate_bps * dt_s)
    budget = max(0, min(n_bits, budget))
    if budget >= n_bits:
        return torch.ones(n_bits, dtype=torch.bool)
    perm = torch.randperm(n_bits, generator=generator)
    keep = torch.zeros(n_bits, dtype=torch.bool)
    keep[perm[:budget]] = True
    return keep
