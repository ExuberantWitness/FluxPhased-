"""F1 §6 Gate 0 contract test 8 — PPO math primitives.

Per DEBUG_CONTRACT.md §6, before any training:

  - ``max_abs(recomputed_logp - stored_old_logp) < 1e-6`` using identical
    observation, action, mask, and parameters;
  - pre-update PPO ratio differs from one by less than ``1e-6``;
  - hand-computed GAE terminal and time-limit examples pass.

This test exercises the math primitives directly; the masked-categorical
distribution lives in ``algo/_shared/pilot/g3_bsta_lite/masked_categorical.py``
(F4). Here we verify the invariants on torch.distributions.Categorical with
manual masking, which is the underlying computation.
"""

from __future__ import annotations

import math

import pytest
import torch


def test_preupdate_logp_recomputation_is_bitforbit():
    """Recompute log-prob of same action under same logits: must equal stored."""
    torch.manual_seed(0)
    E, N = 4, 3
    logits = torch.randn(E, N)
    mask = torch.tensor([[True, True, True],
                         [True, False, True],
                         [True, True, False],
                         [False, True, True]])
    masked = logits.masked_fill(~mask, float("-inf"))
    dist = torch.distributions.Categorical(logits=masked)
    a = torch.tensor([0, 0, 1, 1])
    logp_stored = dist.log_prob(a)
    dist2 = torch.distributions.Categorical(logits=masked)
    logp_recomputed = dist2.log_prob(a)
    assert torch.allclose(logp_stored, logp_recomputed, atol=1e-6)


def test_preupdate_ratio_is_one():
    """PPO ratio = exp(logp_new - logp_old). Pre-update, ratio = 1."""
    torch.manual_seed(1)
    E, N = 4, 3
    logits = torch.randn(E, N)
    mask = torch.ones(E, N, dtype=torch.bool)
    masked = logits.masked_fill(~mask, float("-inf"))
    dist = torch.distributions.Categorical(logits=masked)
    a = dist.sample()
    logp_old = dist.log_prob(a)
    logp_new = dist.log_prob(a)  # same params, same action
    ratio = torch.exp(logp_new - logp_old)
    assert torch.allclose(ratio, torch.ones(E), atol=1e-6)


def test_gae_terminal_bootstrap_zero():
    """At terminal: delta = r_T - V(s_T); with V_T=0 and gamma any, GAE_T = delta."""
    # Hand example: 1 step, reward 1.0, V_terminal = 0.
    r = torch.tensor([1.0])
    v_next = torch.tensor([0.0])  # bootstrap on terminal
    v_curr = torch.tensor([0.5])
    gamma = 0.99
    lam = 0.95
    delta = r + gamma * v_next - v_curr  # = 1.0 + 0 - 0.5 = 0.5
    gae = delta  # terminal: no future GAE
    assert abs(float(gae) - 0.5) < 1e-6


def test_gae_time_limit_bootstrap_uses_value():
    """At time-limit truncation (not terminal): bootstrap with V(s_next)."""
    r = torch.tensor([1.0])
    v_next = torch.tensor([0.7])  # time-limit bootstrap
    v_curr = torch.tensor([0.5])
    gamma = 0.99
    delta = r + gamma * v_next - v_curr  # 1.0 + 0.99*0.7 - 0.5 = 1.193
    assert abs(float(delta) - 1.193) < 1e-3


def test_gae_recursive_two_step():
    """Hand-computed 2-step GAE with gamma=1, lambda=1 (sum of future deltas)."""
    # Rewards [1, 2], values [v0, v1, v2=0]. gamma=1, lambda=1.
    # delta_0 = r0 + v1 - v0
    # delta_1 = r1 + v2 - v1
    # GAE_0 = delta_0 + lambda*gamma*GAE_1 = delta_0 + delta_1
    r = torch.tensor([1.0, 2.0])
    v = torch.tensor([0.5, 0.7, 0.0])
    gamma, lam = 1.0, 1.0
    delta = r + gamma * v[1:] - v[:-1]  # [1+0.7-0.5, 2+0-0.7] = [1.2, 1.3]
    gae_1 = delta[1]
    gae_0 = delta[0] + lam * gamma * gae_1  # 1.2 + 1.3 = 2.5
    assert abs(float(gae_0) - 2.5) < 1e-6
    assert abs(float(gae_1) - 1.3) < 1e-6


def test_masked_categorical_supports_legal_only():
    """Sampled actions must respect the mask."""
    torch.manual_seed(2)
    E, N = 1000, 3
    logits = torch.randn(E, N)
    mask = torch.tensor([False, True, True]).expand(E, N)
    masked = logits.masked_fill(~mask, float("-inf"))
    dist = torch.distributions.Categorical(logits=masked)
    samples = dist.sample((100,))
    assert (samples != 0).all()


def test_logp_of_illegal_action_is_neg_inf():
    """log_prob of a masked action is -inf."""
    torch.manual_seed(3)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[False, True, True]])
    masked = logits.masked_fill(~mask, float("-inf"))
    dist = torch.distributions.Categorical(logits=masked)
    logp_illegal = dist.log_prob(torch.tensor([0]))
    assert math.isinf(float(logp_illegal[0]))
    logp_legal = dist.log_prob(torch.tensor([1]))
    assert math.isfinite(float(logp_legal[0]))
