"""Meta-strategy solvers for PSRO.

Given an empirical payoff matrix, computes the meta-strategy
(mixture weights over the policy population).

Solvers:
  1. Nash Equilibrium (via LP) — recommended default
  2. Uniform — baseline
  3. Rectified Nash — zero out below-threshold weights
"""

import numpy as np
from scipy.optimize import linprog


def solve_nash(payoff_matrix: np.ndarray) -> np.ndarray:
    """Solve 2-player zero-sum meta-game for Nash equilibrium.

    Uses linear programming: find σ (row player's mixed strategy)
    that maximizes v where σ^T U e_j ≥ v for all j.

    Args:
        payoff_matrix: [K, K_opp] array, payoff_matrix[i,j] = win rate of
                       policy i against opponent policy j
    Returns:
        [K] array of mixture weights (probability distribution)
    """
    K, K_opp = payoff_matrix.shape

    if K == 0 or K_opp == 0:
        return np.array([])

    if K == 1:
        return np.array([1.0])

    # LP: minimize -v
    # Variables: [σ_1, ..., σ_K, v]
    # Constraints: for each opponent j: Σ_i σ_i * U[i,j] ≥ v
    #   → -Σ_i σ_i * U[i,j] + v ≤ 0
    # Also: σ_i ≥ 0, Σ σ_i = 1

    n_vars = K + 1  # σ_1..σ_K + v

    # Objective: minimize -v
    c = np.zeros(n_vars)
    c[-1] = -1.0  # minimize -v = maximize v

    # Inequality constraints: -U^T σ + v ≤ 0
    A_ub = np.zeros((K_opp, n_vars))
    for j in range(K_opp):
        A_ub[j, :K] = -payoff_matrix[:, j]
        A_ub[j, K] = 1.0
    b_ub = np.zeros(K_opp)

    # Equality constraint: Σ σ_i = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])

    # Bounds: σ_i ≥ 0, v unbounded
    bounds = [(0, None)] * K + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if result.success:
        weights = result.x[:K]
        # Clip negatives from numerical precision
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum() + 1e-10
        return weights
    else:
        # Fallback to uniform
        return np.ones(K) / K


def solve_uniform(K: int) -> np.ndarray:
    """Uniform meta-strategy: equal weight on all policies."""
    if K == 0:
        return np.array([])
    return np.ones(K) / K


def solve_rectified_nash(
    payoff_matrix: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """Rectified Nash: zero out below-threshold weights from Nash solution.

    Args:
        payoff_matrix: [K, K_opp] array
        threshold: weights below this are zeroed out
    Returns:
        [K] array of mixture weights
    """
    weights = solve_nash(payoff_matrix)
    weights[weights < threshold] = 0.0
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        weights = np.ones(len(weights)) / len(weights)
    return weights


def nash_conv(payoff_matrix: np.ndarray, meta_strategy: np.ndarray) -> float:
    """Compute NashConv (exploitability) of a meta-strategy.

    Lower is better; 0 means exact Nash equilibrium.

    Args:
        payoff_matrix: [K, K_opp] array
        meta_strategy: [K] mixture weights
    Returns:
        float, the maximum regret
    """
    if len(meta_strategy) == 0:
        return 0.0

    # Expected payoff for each pure strategy against opponent's meta-strategy
    # (assume opponent plays best response to our meta-strategy)
    expected = payoff_matrix @ np.ones(payoff_matrix.shape[1]) / payoff_matrix.shape[1]
    br_value = expected.max()
    current_value = (meta_strategy * expected).sum()
    return br_value - current_value
