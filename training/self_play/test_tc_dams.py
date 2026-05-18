"""Unit tests for TC-DAMS meta-solver.

Verifies:
  1. lambda=0 produces identical mixture to standard Nash (key invariant).
  2. Increasing lambda monotonically raises fingerprint entropy on a
     rock-paper-scissors task game.
  3. Output is always a valid probability distribution (>=0, sums to 1).
  4. Degenerate inputs (K=1, K=0, mismatched fingerprints) don't crash.
"""

import numpy as np

from training.self_play.meta_solver import solve_nash
from training.self_play.tc_dams_solver import (
    solve_tc_dams,
    task_fingerprint_entropy,
    effective_population_size,
)


def _rps_payoff_and_fingerprints():
    """Build a 4-policy radar EW analogue of rock-paper-scissors-spock.

    4 pure strategies, each fully committing to one task axis.
    Payoff is the classic non-transitive cycle:
        detect beats comm
        comm   beats jam
        jam    beats detect
        recon  beats jam, weak everywhere else
    """
    # rows/cols: 0=detect, 1=jam, 2=recon, 3=comm
    U = np.array([
        [0.50, 0.20, 0.55, 0.80],   # detect: weak vs jam, strong vs comm
        [0.80, 0.50, 0.30, 0.20],   # jam: strong vs detect, weak vs recon/comm
        [0.45, 0.70, 0.50, 0.50],   # recon: counters jam
        [0.20, 0.80, 0.50, 0.50],   # comm: strong vs jam, weak vs detect
    ])
    # Each policy is a one-hot on its dominant task.
    F = np.eye(4)
    return U, F


def test_lambda_zero_matches_nash():
    U, F = _rps_payoff_and_fingerprints()
    sigma_nash = solve_nash(U)
    sigma_tcd = solve_tc_dams(U, F, lambda_diversity=0.0)
    assert sigma_tcd.shape == sigma_nash.shape
    assert np.allclose(sigma_tcd, sigma_nash, atol=1e-8), (
        f"lambda=0 must match Nash exactly, got nash={sigma_nash}, "
        f"tcd={sigma_tcd}"
    )
    print(f"[PASS] lambda=0 matches Nash. sigma = {sigma_nash.round(4)}")


def test_lambda_monotonically_increases_entropy():
    U, F = _rps_payoff_and_fingerprints()
    lambdas = [0.0, 0.1, 0.3, 1.0, 3.0]
    entropies = []
    for lam in lambdas:
        sigma = solve_tc_dams(U, F, lambda_diversity=lam)
        H = task_fingerprint_entropy(sigma, F)
        entropies.append(H)
        print(f"  lambda={lam:>4.1f}  sigma={sigma.round(3)}  H(sigma^T F)={H:.4f}")
    # Must be non-decreasing (allow small tolerance for FW convergence noise).
    for i in range(1, len(entropies)):
        assert entropies[i] + 1e-3 >= entropies[i - 1], (
            f"H must be non-decreasing in lambda, but H[{i}]={entropies[i]:.4f}"
            f" < H[{i-1}]={entropies[i-1]:.4f}"
        )
    # At sufficiently large lambda we should approach uniform task coverage
    # (max entropy = log(4) ~= 1.386) since fingerprints are one-hot.
    assert entropies[-1] > 1.0, (
        f"At lambda=3, expected near-max entropy log(4)={np.log(4):.3f}, "
        f"got {entropies[-1]:.4f}"
    )
    print(f"[PASS] H grows monotonically with lambda: {[round(e, 3) for e in entropies]}")


def test_output_is_valid_simplex():
    U, F = _rps_payoff_and_fingerprints()
    for lam in [0.0, 0.5, 2.0]:
        sigma = solve_tc_dams(U, F, lambda_diversity=lam)
        assert (sigma >= -1e-9).all(), f"negative weight at lambda={lam}: {sigma}"
        assert abs(sigma.sum() - 1.0) < 1e-6, (
            f"sigma doesn't sum to 1 at lambda={lam}: {sigma.sum()}"
        )
    print(f"[PASS] Output is a valid simplex at all lambdas.")


def test_degenerate_inputs():
    # Empty matrix
    assert solve_tc_dams(np.zeros((0, 0)), None, 0.5).size == 0
    # Single policy
    s = solve_tc_dams(np.array([[0.5]]), np.array([[0.25, 0.25, 0.25, 0.25]]), 0.5)
    assert np.allclose(s, [1.0])
    # Fingerprint K mismatch -> fall back to Nash without crashing
    U, F = _rps_payoff_and_fingerprints()
    bad_F = F[:2]  # only 2 rows
    sigma = solve_tc_dams(U, bad_F, lambda_diversity=0.5)
    assert np.allclose(sigma, solve_nash(U))
    # No fingerprints provided -> Nash fallback
    sigma2 = solve_tc_dams(U, None, lambda_diversity=0.5)
    assert np.allclose(sigma2, solve_nash(U))
    print(f"[PASS] Degenerate inputs handled gracefully.")


def test_effective_population_size():
    # Uniform on K=4 should give eff_size = 4
    eff = effective_population_size(np.array([0.25] * 4))
    assert abs(eff - 4.0) < 1e-6, f"expected 4.0, got {eff}"
    # One-hot should give 1
    eff = effective_population_size(np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(eff - 1.0) < 1e-6, f"expected 1.0, got {eff}"
    print(f"[PASS] effective_population_size correct.")


if __name__ == "__main__":
    print("=" * 60)
    print("TC-DAMS Solver Unit Tests")
    print("=" * 60)
    test_lambda_zero_matches_nash()
    print()
    test_lambda_monotonically_increases_entropy()
    print()
    test_output_is_valid_simplex()
    print()
    test_degenerate_inputs()
    print()
    test_effective_population_size()
    print()
    print("=" * 60)
    print("All TC-DAMS tests PASSED")
    print("=" * 60)
