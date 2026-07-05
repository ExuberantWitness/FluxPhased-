"""Unit tests for COMA buffer extensions (joint_actions + coma_agent_idx)."""

import pytest
import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from algo._shared.ppo.buffer import RolloutBuffer


def test_default_buffer_no_coma_fields():
    """Default buffer (no joint_action_dim) must not produce COMA fields."""
    buf = RolloutBuffer(
        buffer_size=8, obs_dim=4, act_dim=2,
        device="cpu",
    )
    assert buf.joint_actions is None
    assert buf.coma_agent_idx is None
    buf.add(obs=torch.zeros(4), action=torch.zeros(2), reward=1.0,
            done=0.0, value=0.5, log_prob=0.0)
    batch = next(buf.get_minibatches(batch_size=1))
    assert "joint_actions" not in batch
    assert "coma_agent_idx" not in batch


def test_coma_buffer_allocates_joint_actions():
    """joint_action_dim > 0 must allocate joint_actions and require it on add()."""
    buf = RolloutBuffer(
        buffer_size=8, obs_dim=4, act_dim=2,
        device="cpu", joint_action_dim=10,
    )
    assert buf.joint_actions is not None
    assert buf.joint_actions.shape == (8, 10)

    # Missing joint_action must assert
    with pytest.raises(AssertionError):
        buf.add(obs=torch.zeros(4), action=torch.zeros(2), reward=1.0,
                done=0.0, value=0.5, log_prob=0.0)


def test_coma_buffer_allocates_agent_idx():
    """store_coma_agent_idx=True must allocate field and require it on add()."""
    buf = RolloutBuffer(
        buffer_size=8, obs_dim=4, act_dim=2,
        device="cpu", joint_action_dim=10, store_coma_agent_idx=True,
    )
    assert buf.coma_agent_idx is not None
    assert buf.coma_agent_idx.shape == (8,)
    assert buf.coma_agent_idx.dtype == torch.long

    # Missing coma_agent_idx must assert
    with pytest.raises(AssertionError):
        buf.add(obs=torch.zeros(4), action=torch.zeros(2), reward=1.0,
                done=0.0, value=0.5, log_prob=0.0,
                joint_action=torch.zeros(10))


def test_coma_buffer_minibatch_has_both_fields():
    """A COMA-mode buffer's minibatch must include both COMA fields."""
    buf = RolloutBuffer(
        buffer_size=4, obs_dim=4, act_dim=2,
        device="cpu", joint_action_dim=10, store_coma_agent_idx=True,
    )
    for i in range(4):
        buf.add(
            obs=torch.randn(4), action=torch.randn(2), reward=float(i),
            done=0.0, value=0.5, log_prob=0.0,
            joint_action=torch.full((10,), float(i)),
            coma_agent_idx=i % 2,
        )
    batch = next(buf.get_minibatches(batch_size=4))
    assert "joint_actions" in batch
    assert "coma_agent_idx" in batch
    assert batch["joint_actions"].shape == (4, 10)
    assert batch["coma_agent_idx"].shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
