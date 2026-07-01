"""Smoke test for Sub-Array Decomposed Policy (SADP).

Tests:
  1. Policy creation and parameter count
  2. Forward pass dimensions
  3. evaluate_actions log-prob consistency
  4. Sub-array action broadcasting correctness
  5. Integration with FluxLeague + MFARVecEnv

Usage:
    python validation/validate_sadp.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.testing


def test_policy_creation():
    """Test SubArrayRadarActorCritic creation and parameter count."""
    from algo._shared.ppo.actor_critic import SubArrayRadarActorCritic, create_team_policy

    # 25x25 array, P=4, bins=64, sub=5
    model = SubArrayRadarActorCritic(
        n_elem=625, n_pulses=4, n_bins=64,
        sub_array_size=5, commander_instr_dim=16,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    assert params < 2_000_000, f"Too many params: {params}"

    # Via create_team_policy
    policy = create_team_policy(
        team=0, n_elem=625, n_pulses=4, n_bins=64,
        num_output_length=16, sub_array_size=5,
    )
    assert isinstance(policy["radar"], SubArrayRadarActorCritic)
    print("  PASS: policy creation")


def test_forward_pass():
    """Test forward pass produces correct output shapes."""
    from algo._shared.ppo.actor_critic import SubArrayRadarActorCritic

    model = SubArrayRadarActorCritic(
        n_elem=625, n_pulses=4, n_bins=64,
        sub_array_size=5, commander_instr_dim=16,
    )
    state_dim = 625 * (4 * 64 + 2 + 4) + 5 + 12 + 16  # 163,783

    B = 4
    state = torch.randn(B, state_dim)
    action, log_prob, value = model.get_action(state)

    assert action.shape == (B, 13_753), f"Action shape: {action.shape}"
    assert log_prob.shape == (B,), f"Log prob shape: {log_prob.shape}"
    assert value.shape == (B, 1), f"Value shape: {value.shape}"
    assert action.isfinite().all(), "NaN/Inf in action"

    # Deterministic mode
    action_det, _, value_det = model.get_action(state, deterministic=True)
    assert action_det.shape == (B, 13_753)
    assert torch.allclose(action_det, model.forward(state)[0], atol=1e-5)
    print("  PASS: forward pass shapes")


def test_evaluate_actions():
    """Test evaluate_actions produces valid log-probs and entropy."""
    from algo._shared.ppo.actor_critic import SubArrayRadarActorCritic

    model = SubArrayRadarActorCritic(
        n_elem=625, n_pulses=4, n_bins=64,
        sub_array_size=5, commander_instr_dim=16,
    )
    state_dim = 625 * (4 * 64 + 2 + 4) + 5 + 12 + 16

    B = 4
    state = torch.randn(B, state_dim)
    action, old_logp, _ = model.get_action(state)

    logp, entropy, value = model.evaluate_actions(state, action)
    assert logp.shape == (B,), f"Eval log_prob shape: {logp.shape}"
    assert entropy.shape == (B,), f"Entropy shape: {entropy.shape}"
    assert value.shape == (B, 1)
    assert logp.isfinite().all(), "NaN in log_prob"
    assert entropy.isfinite().all(), "NaN in entropy"

    # Log-prob should be close to old_logp (same network, same action)
    diff = (logp - old_logp).abs().max().item()
    assert diff < 0.01, f"Log-prob mismatch: {diff}"
    print("  PASS: evaluate_actions consistency")


def test_sub_array_broadcast():
    """Test that all elements in a sub-array have identical actions."""
    from algo._shared.ppo.actor_critic import SubArrayRadarActorCritic

    model = SubArrayRadarActorCritic(
        n_elem=625, n_pulses=4, n_bins=64,
        sub_array_size=5, commander_instr_dim=16,
    )
    state_dim = 625 * (4 * 64 + 2 + 4) + 5 + 12 + 16

    state = torch.randn(2, state_dim)
    action, _, _ = model.get_action(state, deterministic=True)

    # Parse into per-element actions
    elem_act = action[:, :625 * 22].reshape(2, 625, 22)

    # Each sub-array has 25 elements — all should be identical
    for k in range(25):
        sub_start = k * 25
        sub_end = (k + 1) * 25
        sub_actions = elem_act[:, sub_start:sub_end, :]  # [2, 25, 22]
        # All 25 elements should be identical
        ref = sub_actions[:, 0:1, :]  # [2, 1, 22]
        max_diff = (sub_actions - ref).abs().max().item()
        assert max_diff < 1e-6, f"Sub-array {k} not uniform: max_diff={max_diff}"
    print("  PASS: sub-array broadcast uniformity")


def test_env_integration():
    """Test SADP policy works with MFARVecEnv step."""
    from env.gpu.vec_mfar_env import MFARVecEnv
    from algo._shared.ppo.actor_critic import create_team_policy

    env = MFARVecEnv(
        num_envs=1, n_radars=4, rows=25, cols=25,
        pulses_per_cpi=4, fft_size=64, device="cuda",
    )
    print(f"  Env state_dim={env.state_dim}, action_dim={env.action_dim}")

    policy = create_team_policy(
        team=0, n_elem=env.n_elem, n_pulses=env.n_pulses,
        n_bins=env.n_bins, num_output_length=env.num_output_length,
        sub_array_size=5, device="cuda",
    )

    env.reset()
    state = env._assemble_state(env._buf_spectrum, env._buf_comm_data)

    # Get action from sub-array policy
    with torch.no_grad():
        action, logp, val = policy["radar"].get_action(state[:, 0, :])

    assert action.shape == (1, env.action_dim), f"Action: {action.shape}"
    assert action.isfinite().all()

    # Step env with the action
    actions = torch.zeros(1, 4, env.action_dim, device="cuda")
    actions[:, 0, :] = action
    result = env.step(actions=actions)

    assert result["state"].shape[0] == 1
    assert result["dones"].isfinite().all()
    print("  PASS: env integration")


if __name__ == "__main__":
    print("SADP Smoke Test")
    print("=" * 50)
    test_policy_creation()
    test_forward_pass()
    test_evaluate_actions()
    test_sub_array_broadcast()
    test_env_integration()
    print("=" * 50)
    print("ALL TESTS PASSED")
