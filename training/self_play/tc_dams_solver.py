"""TC-DAMS: Task-Coverage Diversity-Aware Meta-Solver.

Drop-in extension of `meta_solver.solve_nash` that augments the standard
2-player zero-sum LP with an entropy term over the *task-allocation
fingerprint* induced by the meta-mixture.

For each policy pi_i, the fingerprint f(pi_i) is the long-run average
fraction of array elements assigned to each of the 4 tasks
{recon, detect, jam, comm}. It lives on the 3-simplex Delta^3 (4 entries
summing to 1) and is logged for free by `vec_mfar_env.step()`.

Given:
    payoff_matrix U in R^{K x K_opp}
    fingerprints  F in R^{K x 4}, rows on Delta^3
    lambda >= 0   diversity coefficient

TC-DAMS solves:

    sigma* = arg max_sigma [ min_tau sigma^T U tau  +  lambda * H( sigma^T F ) ]

where H is Shannon entropy on Delta^3. The objective is concave in sigma
(LP value is concave; entropy of a linear function of sigma is concave),
so it is solved by Frank-Wolfe (Conditional Gradient): repeatedly compute
the gradient and take a step toward the LP-optimal vertex of the
augmented objective.

When lambda == 0 the solver is numerically identical to `solve_nash`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .meta_solver import solve_nash


_EPS = 1e-12


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats, safe for zeros."""
    p = np.asarray(p, dtype=np.float64)
    mask = p > _EPS
    return float(-(p[mask] * np.log(p[mask])).sum())


def _entropy_grad_wrt_sigma(sigma: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Gradient of H(sigma^T F) with respect to sigma.

    Let F_bar = sigma^T F in Delta^3. Then
        d H(F_bar) / d sigma_i = - sum_t F[i, t] * (1 + log(F_bar[t]))

    The "+1" constants drop out under any normalization to a simplex,
    but we keep them for a faithful gradient computation; the LP step
    in Frank-Wolfe sees only relative magnitudes.
    """
    F_bar = sigma @ F  # [4]
    safe = np.clip(F_bar, _EPS, 1.0)
    log_term = -(1.0 + np.log(safe))  # [4]
    grad = F @ log_term  # [K]
    return grad


def _lp_nash_with_linear_bonus(
    payoff_matrix: np.ndarray,
    bonus: np.ndarray | None = None,
) -> np.ndarray:
    """Solve max_sigma [ min_tau sigma^T U tau  +  bonus^T sigma ].

    Bonus is a linear term in sigma. Reduces to standard Nash LP when
    bonus is None or all-zero.

    Variables: [sigma_1, ..., sigma_K, v]
    Maximize:  v + bonus^T sigma     (i.e. minimize -v - bonus^T sigma)
    Subject to:
        for each j: sum_i sigma_i * U[i, j] >= v   (sigma is robust to opp)
        sum_i sigma_i = 1
        sigma_i >= 0
    """
    K, K_opp = payoff_matrix.shape
    if K == 0 or K_opp == 0:
        return np.array([])
    if K == 1:
        return np.array([1.0])

    n_vars = K + 1
    c = np.zeros(n_vars)
    c[-1] = -1.0  # minimize -v  ==>  maximize v
    if bonus is not None:
        c[:K] = -bonus  # minimize -bonus^T sigma  ==>  maximize bonus^T sigma

    # Inequality: -U^T sigma + v <= 0   <=>   sum_i sigma_i U[i,j] >= v
    A_ub = np.zeros((K_opp, n_vars))
    for j in range(K_opp):
        A_ub[j, :K] = -payoff_matrix[:, j]
        A_ub[j, K] = 1.0
    b_ub = np.zeros(K_opp)

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * K + [(None, None)]

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )

    if not result.success:
        return np.ones(K) / K

    weights = np.maximum(result.x[:K], 0.0)
    s = weights.sum()
    if s <= 0:
        return np.ones(K) / K
    return weights / s


def solve_tc_dams(
    payoff_matrix: np.ndarray,
    fingerprints: np.ndarray | None = None,
    lambda_diversity: float = 0.3,
    n_iters: int = 25,
    tol: float = 1e-6,
) -> np.ndarray:
    """Task-Coverage Diversity-Aware Meta-Solver.

    Args:
        payoff_matrix: [K, K_opp] win-rate matrix (row = own policy).
        fingerprints:  [K, 4] task-allocation fingerprints on Delta^3.
                       If None or lambda_diversity == 0, falls back to Nash.
        lambda_diversity: weight of the H(sigma^T F) entropy bonus.
        n_iters: max Frank-Wolfe iterations.
        tol: convergence tolerance on sigma L2 change.
    Returns:
        [K] mixture weights, summing to 1.
    """
    K, K_opp = payoff_matrix.shape

    # Degenerate cases / disabled diversity: behave exactly like Nash.
    if K == 0 or K_opp == 0:
        return np.array([])
    if K == 1:
        return np.array([1.0])
    if fingerprints is None or lambda_diversity <= 0.0:
        return solve_nash(payoff_matrix)

    fingerprints = np.asarray(fingerprints, dtype=np.float64)
    if fingerprints.shape[0] != K:
        # Mismatch: ignore the bonus rather than crash.
        return solve_nash(payoff_matrix)
    # Re-normalize defensively in case the env-side rolling mean drifted.
    row_sums = fingerprints.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > _EPS, row_sums, 1.0)
    F = fingerprints / row_sums

    # Frank-Wolfe (Conditional Gradient) on  f(sigma) = nash_value(sigma) + lambda * H(sigma^T F).
    # At each step, linearize entropy at current sigma, then solve the
    # Nash-LP with that linear bonus on sigma. Step with a decaying schedule.
    sigma = solve_nash(payoff_matrix)
    if sigma.size == 0:
        return sigma

    for k in range(n_iters):
        grad_H = _entropy_grad_wrt_sigma(sigma, F)  # [K]
        bonus = lambda_diversity * grad_H
        # Center the bonus to keep numerical scale stable; the simplex
        # constraint absorbs any constant shift exactly.
        bonus = bonus - bonus.mean()

        vertex = _lp_nash_with_linear_bonus(payoff_matrix, bonus=bonus)
        if vertex.size == 0:
            break

        step = 2.0 / (k + 2.0)  # classic FW schedule
        new_sigma = (1.0 - step) * sigma + step * vertex

        if np.linalg.norm(new_sigma - sigma) < tol:
            sigma = new_sigma
            break
        sigma = new_sigma

    sigma = np.maximum(sigma, 0.0)
    s = sigma.sum()
    return sigma / s if s > 0 else np.ones(K) / K


def task_fingerprint_entropy(
    sigma: np.ndarray,
    fingerprints: np.ndarray,
) -> float:
    """Entropy H(sigma^T F) in nats. Convenience for logging.

    Range: [0, log 4] ≈ [0, 1.386]. Higher = more task-axis diversity.
    """
    if sigma.size == 0 or fingerprints.size == 0:
        return 0.0
    F = np.asarray(fingerprints, dtype=np.float64)
    if F.shape[0] != sigma.size:
        return 0.0
    F_bar = sigma @ F
    F_bar = np.clip(F_bar, 0.0, None)
    s = F_bar.sum()
    if s <= 0:
        return 0.0
    return _entropy(F_bar / s)


def effective_population_size(sigma: np.ndarray) -> float:
    """exp(H(sigma)): how many policies the mixture is effectively using."""
    if sigma.size == 0:
        return 0.0
    return float(np.exp(_entropy(sigma)))
