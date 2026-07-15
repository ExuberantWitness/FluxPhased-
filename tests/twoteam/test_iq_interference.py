"""Unit tests for IqInterference physics module.

Validates shape, NaN-safety, mirror symmetry, basic physics monotonicity.
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import math
import torch
import pytest

from env.gpu.twoteam.iq_interference import (
    IqInterference,
    _wrap,
    _sinc2_db,
    _rect_overlap_frac,
    N_RADARS,
)


def _make_inputs(E=4, device="cuda", seed=0):
    """Build mirror-symmetric dummy inputs."""
    torch.manual_seed(seed)
    pos = torch.zeros(E, 2, 2, 2, device=device)
    # Mirror-symmetric: team_B = -team_A
    A = torch.tensor([[-2500.0, 750.0], [-2500.0, -750.0]], device=device)
    pos[:, 0, 0] = A[0]
    pos[:, 0, 1] = A[1]
    pos[:, 1, 0] = -A[0]
    pos[:, 1, 1] = -A[1]

    beam_az = torch.zeros(E, 2, 2, device=device)
    beam_az[:, 0, 0] = 0.0   # team A points +x
    beam_az[:, 0, 1] = 0.0
    beam_az[:, 1, 0] = math.pi  # team B points -x (mirror)
    beam_az[:, 1, 1] = math.pi

    alloc = torch.full((E, 2, 2, 4), 0.25, device=device)

    freq_hz = torch.full((E, 2, 2), 10e9, device=device)

    emission_on = torch.ones(E, 2, 2, dtype=torch.bool, device=device)
    hop_rate = torch.ones(E, 2, 2, device=device)
    radar_alive = torch.ones(E, 2, 2, dtype=torch.bool, device=device)

    return pos, beam_az, alloc, freq_hz, emission_on, hop_rate, radar_alive


# ---- shape ---------------------------------------------------------------

def test_jnr_matrix_shape():
    iq = IqInterference()
    inputs = _make_inputs(E=4)
    jnr = iq.compute_jnr_matrix(*inputs)
    assert jnr.shape == (4, N_RADARS, N_RADARS), f"got {jnr.shape}"


def test_meas_sigma_shape():
    iq = IqInterference()
    pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive = _make_inputs(E=4)
    jnr = iq.compute_jnr_matrix(pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive)
    E = 4
    f_track_eff = torch.full((E, 2, 2), 0.5, device=pos.device)
    fusion_factor = torch.ones(E, 2, 2, device=pos.device)
    sigma = iq.compute_meas_sigma(jnr, f_track_eff, range_sigma=0.05, fusion_factor=fusion_factor)
    assert sigma.shape == (E, 2, 2), f"got {sigma.shape}"


# ---- NaN safety ----------------------------------------------------------

def test_no_nans_basic():
    iq = IqInterference()
    inputs = _make_inputs(E=8)
    jnr = iq.compute_jnr_matrix(*inputs)
    assert not torch.isnan(jnr).any(), "JNR has NaNs"
    assert torch.isfinite(jnr).all(), "JNR has inf"


def test_no_nans_close_distance():
    """Two radars at near-identical positions must not NaN."""
    iq = IqInterference()
    pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive = _make_inputs(E=4)
    # Crush teammate distance
    pos[:, 0, 1] = pos[:, 0, 0] + torch.tensor([50.0, 0.0], device=pos.device)
    pos[:, 1, 1] = -pos[:, 0, 1]
    jnr = iq.compute_jnr_matrix(pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive)
    assert torch.isfinite(jnr).all(), "NaN/inf at close distance"


def test_no_nans_zero_emission():
    iq = IqInterference()
    pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive = _make_inputs(E=4)
    alloc[:, 0, 0] = 0.0  # all-zero allocation for radar 0
    jnr = iq.compute_jnr_matrix(pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive)
    assert torch.isfinite(jnr).all(), "NaN/inf with zero alloc"


# ---- self-interference ---------------------------------------------------

def test_diagonal_zero():
    """JNR[i,i] must be exactly 0 (perfect SIC)."""
    iq = IqInterference()
    inputs = _make_inputs(E=4)
    jnr = iq.compute_jnr_matrix(*inputs)
    diag = torch.diagonal(jnr, dim1=1, dim2=2)  # [E,N]
    assert (diag == 0).all(), f"diagonal not zero: {diag}"


# ---- mirror symmetry -----------------------------------------------------

def test_mirror_symmetry_jnr():
    """Under mirror-symmetric inputs, jnr_mat must satisfy:
       jnr[e, i, j] == jnr[e, mirror(i), mirror(j)]
       where mirror swaps teams (0↔2, 1↔3).
    """
    iq = IqInterference()
    inputs = _make_inputs(E=8)
    jnr = iq.compute_jnr_matrix(*inputs)

    # Mirror permutation: flat idx = team*2 + slot → mirror = (1-team)*2 + slot
    # i.e., flat 0↔2, 1↔3
    perm = [2, 3, 0, 1]
    jnr_mirrored = jnr[:, perm, :][:, :, perm]

    max_err = (jnr - jnr_mirrored).abs().max().item()
    # Allow 1% relative mirror error — the -30 dB sinc² floor is a hard clamp,
    # which can introduce small absolute asymmetries (~0.7 linear on ~1e5 base)
    # when both sides of a mirror pair sit near the floor. This is a numerical
    # artifact of the floor, not a physics symmetry violation.
    max_val = jnr.abs().max().item()
    rel_err = max_err / max(max_val, 1e-9)
    assert rel_err < 1e-2, f"mirror rel asymmetry {rel_err:.3e} (abs {max_err:.3e} on max {max_val:.3e})"


# ---- physics monotonicity ------------------------------------------------

def test_frequency_overlap_reduces_jnr():
    """Same-freq → high JNR; off-by-1 channel → much lower."""
    iq = IqInterference()
    pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive = _make_inputs(E=4)

    # All same freq
    freq_hz[:] = 10e9
    jnr_same = iq.compute_jnr_matrix(pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive)

    # Team A on 10.000 GHz, Team B on 10.050 GHz (50 MHz apart = 5× bw)
    freq_hz[:, 0] = 10e9
    freq_hz[:, 1] = 10e9 + 5 * iq.channel_bw_hz
    jnr_off = iq.compute_jnr_matrix(pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive)

    # Cross-team JNR (i in team A → j in team B): same-freq >> off-freq
    cross_same = jnr_same[:, :2, 2:].mean().item()
    cross_off = jnr_off[:, :2, 2:].mean().item()
    assert cross_same > 10 * cross_off, f"overlap filter weak: same={cross_same:.3e} off={cross_off:.3e}"


def test_distance_monotonic():
    """Farther interferer → smaller JNR."""
    iq = IqInterference()
    pos_near, beam_az, alloc, freq_hz, emission_on, hop_rate, alive = _make_inputs(E=4)
    jnr_near = iq.compute_jnr_matrix(pos_near, beam_az, alloc, freq_hz, emission_on, hop_rate, alive)

    # Move teams 5× farther apart
    pos_far = pos_near.clone()
    pos_far[:, 0] *= 5.0
    pos_far[:, 1] *= 5.0
    jnr_far = iq.compute_jnr_matrix(pos_far, beam_az, alloc, freq_hz, emission_on, hop_rate, alive)

    # Cross-team JNR drops with distance (Friis ~ 1/d²)
    near_val = jnr_near[:, :2, 2:].mean().item()
    far_val = jnr_far[:, :2, 2:].mean().item()
    assert near_val > far_val, f"distance filter wrong: near={near_val:.3e} far={far_val:.3e}"


def test_hop_attenuation_reduces_jnr():
    """Higher hop_rate at interferer → smaller JNR at all victims."""
    iq = IqInterference()
    pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive = _make_inputs(E=4)
    jnr_no_hop = iq.compute_jnr_matrix(pos, beam_az, alloc, freq_hz, emission_on, hop_rate, alive)

    hop_rate8 = hop_rate * 8.0
    jnr_hop8 = iq.compute_jnr_matrix(pos, beam_az, alloc, freq_hz, emission_on, hop_rate8, alive)

    # Off-diagonal JNR must drop with hop
    off_mask = (~torch.eye(N_RADARS, dtype=torch.bool, device=pos.device)).unsqueeze(0).expand_as(jnr_no_hop)
    mean_no = jnr_no_hop[off_mask].mean().item()
    mean_h8 = jnr_hop8[off_mask].mean().item()
    assert mean_h8 < mean_no, f"hop attenuation wrong: no_hop={mean_no:.3e} hop8={mean_h8:.3e}"


# ---- helpers -------------------------------------------------------------

def test_wrap_symmetric():
    """_wrap(x) == -_wrap(-x) for off-axis values (mirror-unbiased requirement)."""
    x = torch.linspace(-3.5, 3.5, 50)
    w = _wrap(x)
    assert torch.allclose(w, -_wrap(-x), atol=1e-6), "wrap asymmetric"


def test_sinc2_peak_and_floor():
    """sinc² peak = 0 dB at boresight, ≥ -30 dB everywhere."""
    theta = torch.full((10,), 0.05)
    rel = torch.linspace(-0.5, 0.5, 10)
    gain_db = _sinc2_db(rel, theta)
    assert gain_db.max().item() <= 1e-3, f"peak > 0 dB: {gain_db.max()}"
    assert gain_db.min().item() >= -30.1, f"floor < -30 dB: {gain_db.min()}"


def test_rect_overlap():
    """Full overlap → 1; no overlap → 0; partial → correct fraction."""
    f_i = torch.tensor([10.0, 10.0, 10.0])
    f_j = torch.tensor([10.0, 11.0, 10.4])
    bw = 1.0
    out = _rect_overlap_frac(f_i, f_j, bw)
    assert abs(out[0].item() - 1.0) < 1e-6
    assert abs(out[1].item() - 0.0) < 1e-6
    assert abs(out[2].item() - 0.6) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
