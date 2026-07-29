"""Legacy 4-node IQ regression test (F0 §7 guard rail).

This test locks the bit-exact output of ``env.gpu.twoteam.iq_interference
.IqInterference.compute_jnr_matrix`` for a fixed deterministic input on a
fixed torch version. Purpose: detect any unintended change to the shared
IQ kernel while the G3-BSTA-lite line is being built.

The G3-BSTA-lite clean line uses an isolated IQ adapter and does NOT modify
the shared IQ kernel. This test exists as defense-in-depth: if a future
commit (in this line or a sibling) does touch the shared kernel, this test
will fail loudly before the change reaches CI.

Bit-exactness caveats:
- The golden hash is computed against the torch version recorded in the
  ``fluxphased`` conda env (torch 2.12.x). Major torch upgrades may
  legitimately require re-baselining; that is the point.
- The hash covers dtype + shape + numerical bytes.
- Summary statistics (mean, std, frobenius) are also asserted so a numerical
  drift is flagged twice.

If this test fails:
- Do NOT silence it by re-baselining without independent verification.
- Investigate the actual change to ``iq_interference.py``.
- Re-baseline only with a recorded reason and a positive review of the
  numerical impact on the rest of the test suite.

Baseline established at F0 base commit ``80769974cb41fd86e2f80bc2a8992955fb228058``
on host node15, torch 2.12.0+cu132, conda env fluxphased.
"""

from __future__ import annotations

import hashlib
import math

import pytest
import torch

from env.gpu.twoteam.iq_interference import IqInterference


LEGACY_BASELINE_HASH_CPU = "c122602b9cfbf740ea21f79c476b9afac3ffabf287ccdada69bbb74ba205b9c7"
"""SHA-256 over dtype + shape + packed bytes of the legacy CPU JNR output.

Re-baseline only after independent verification of any change to
``iq_interference.py``. A new baseline must include a one-line reason in the
commit message and confirm the 109+ legacy tests still pass.
"""


def _deterministic_inputs(*, E: int = 2, device: str = "cpu") -> dict:
    """Build a deterministic 4-node (T=2, R=2) input batch.

    Layout (matches iq_interference.py docstring):
      pos[E, T, R, 2]      - cartesian position (m)
      beam_az[E, T, R]     - beam azimuth (rad)
      alloc[E, T, R, 4]    - fractions (detect, track, jam, comm), sums to <=1
      freq_hz[E, T, R]     - tx center frequency (Hz)
      emission_on[E, T, R] - bool
      hop_rate[E, T, R]    - >=1.0
      radar_alive[E, T, R] - bool

    Env 0: 3km geometry, all radars on the same channel (full overlap).
    Env 1: 5km geometry, two radars staggered by ±12 MHz (> 1 channel BW),
           so overlap drops to zero for those pairs.
    """
    pos = torch.tensor(
        [
            [[[-3000.0, -2000.0], [3000.0, -2000.0]],
             [[-3000.0,  2000.0], [3000.0,  2000.0]]],
            [[[-5000.0, -1500.0], [5000.0, -1500.0]],
             [[-5000.0,  1500.0], [5000.0,  1500.0]]],
        ],
        device=device,
        dtype=torch.float64,
    ).expand(E, 2, 2, 2).contiguous()

    beam_az = torch.zeros(E, 2, 2, device=device, dtype=torch.float64)
    beam_az[:, 0, :] = math.pi / 2
    beam_az[:, 1, :] = -math.pi / 2

    alloc = torch.zeros(E, 2, 2, 4, device=device, dtype=torch.float64)
    alloc[:, :, :, 0] = 0.6
    alloc[:, :, :, 1] = 0.3

    fc = 10.0e9
    freq_hz = torch.full((E, 2, 2), fc, device=device, dtype=torch.float64)
    freq_hz[1, 0, 0] = fc + 12.0e6
    freq_hz[1, 1, 1] = fc - 12.0e6

    emission_on = torch.ones(E, 2, 2, device=device, dtype=torch.bool)
    hop_rate = torch.ones(E, 2, 2, device=device, dtype=torch.float64)
    radar_alive = torch.ones(E, 2, 2, device=device, dtype=torch.bool)

    return dict(
        pos=pos,
        beam_az=beam_az,
        alloc=alloc,
        freq_hz=freq_hz,
        emission_on=emission_on,
        hop_rate=hop_rate,
        radar_alive=radar_alive,
    )


def _tensor_hash(t: torch.Tensor) -> str:
    """Stable hash over dtype, shape, and packed numerical bytes."""
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("ascii"))
    h.update(repr(tuple(t.shape)).encode("ascii"))
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_legacy_compute_jnr_matrix_shape_and_finiteness_cpu():
    iq = IqInterference(fc_hz=10e9, channel_bw_hz=10e6)
    inputs = _deterministic_inputs(E=2, device="cpu")
    jnr = iq.compute_jnr_matrix(**inputs)
    assert jnr.shape == (2, 4, 4), f"unexpected shape {jnr.shape}"
    assert jnr.dtype == torch.float64
    assert torch.isfinite(jnr).all()
    for e in range(2):
        for i in range(4):
            assert jnr[e, i, i].item() == 0.0, (
                f"diag [{e},{i},{i}] should be 0 (perfect SIC), got {jnr[e,i,i].item()}"
            )


def test_legacy_compute_jnr_matrix_bitexact_cpu():
    """Lock the bit-exact CPU output of compute_jnr_matrix."""
    iq = IqInterference(fc_hz=10e9, channel_bw_hz=10e6)
    inputs = _deterministic_inputs(E=2, device="cpu")
    jnr = iq.compute_jnr_matrix(**inputs)
    h = _tensor_hash(jnr)
    assert h == LEGACY_BASELINE_HASH_CPU, (
        "Bit-exact regression: legacy IQ JNR matrix hash changed.\n"
        f"  expected: {LEGACY_BASELINE_HASH_CPU}\n"
        f"  actual:   {h}\n"
        "If the change to iq_interference.py was intentional and verified, "
        "re-baseline LEGACY_BASELINE_HASH_CPU with a one-line reason in the "
        "commit message and confirm 109+ legacy tests still pass."
    )


def test_legacy_compute_jnr_matrix_summary_stats_cpu():
    """Defensive second check: summary statistics of the legacy JNR matrix.

    Catches the unlikely case where a hash collision masks numerical drift.
    """
    iq = IqInterference(fc_hz=10e9, channel_bw_hz=10e6)
    inputs = _deterministic_inputs(E=2, device="cpu")
    jnr = iq.compute_jnr_matrix(**inputs)

    # Env 0 (full overlap, 3km geometry) — high JNR. The 287M figure
    # corresponds to main-beam boresight coupling at ~4.2 km.
    # Env 1 (staggered freq) — JNR effectively zero everywhere except
    # the pair that is still on the common channel (5km geometry).
    mean = jnr.mean().item()
    std = jnr.std().item()
    fro = jnr.norm().item()
    assert abs(mean - 35_879_301.72) < 1.0, f"mean drifted: {mean}"
    assert abs(std - 96_446_555.65) < 1.0, f"std drifted: {std}"
    assert abs(fro - 574_068_339.46) < 1.0, f"frobenius drifted: {fro}"
    assert jnr.max().item() < 287_100_000.0, "max exceeded baseline"
    assert jnr.max().item() > 286_900_000.0, "max below baseline"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available; per F0 §6 NOT_RUN/ENV_UNAVAILABLE does not block.",
)
def test_legacy_compute_jnr_matrix_shape_cuda():
    """CUDA shape smoke (per F0 §6: NOT_RUN/ENV_UNAVAILABLE acceptable).

    No bit-exact claim across CPU/CUDA; just shape + finiteness + zero diag.
    """
    iq = IqInterference(fc_hz=10e9, channel_bw_hz=10e6)
    inputs = _deterministic_inputs(E=2, device="cuda")
    jnr = iq.compute_jnr_matrix(**inputs)
    assert jnr.shape == (2, 4, 4)
    assert torch.isfinite(jnr).all()
    for e in range(2):
        for i in range(4):
            assert jnr[e, i, i].item() == 0.0


def test_legacy_compute_meas_sigma_shape():
    """Lock the compute_meas_sigma reshape contract ([E,2,2] victim output)."""
    iq = IqInterference(fc_hz=10e9, channel_bw_hz=10e6)
    inputs = _deterministic_inputs(E=2, device="cpu")
    jnr = iq.compute_jnr_matrix(**inputs)
    f_track_eff = torch.full((2, 2, 2), 0.3, dtype=torch.float64)
    fusion_factor = torch.ones((2, 2, 2), dtype=torch.float64)
    sigma = iq.compute_meas_sigma(
        jnr_matrix=jnr,
        f_track_eff=f_track_eff,
        range_sigma=30.0,
        fusion_factor=fusion_factor,
    )
    assert sigma.shape == (2, 2, 2)
    assert torch.isfinite(sigma).all()
    assert (sigma > 0).all()
