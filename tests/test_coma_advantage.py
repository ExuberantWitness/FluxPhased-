"""Unit tests for COMA counterfactual advantage (sample-based COMA-S)."""

import pytest
import torch
import torch.nn as nn

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from algo._shared.ppo.coma_advantage import (
    coma_counterfactual_advantage,
    JOINT_ACTION_DIM, CMD_OFFSETS, RADAR_OFFSETS,
    CMD_BLOCK_DIM, RADAR_BLOCK_DIM,
)


class _ConstantCritic(nn.Module):
    """Mock critic returning a constant Q regardless of inputs."""
    def __init__(self, q_value: float = 1.0):
        super().__init__()
        self.q = q_value

    def forward(self, team_state, joint_action):
        B = joint_action.shape[0]
        return torch.full((B, 1), self.q)


class _LinearCritic(nn.Module):
    """Mock critic: Q = sum of agent's block (slot offset by `offset`).

    Used to verify the correct slot is being marginalized: Q depends only
    on the agent's own block in joint_action, so replacing it changes Q
    predictably.
    """
    def __init__(self, offset: int, block_dim: int):
        super().__init__()
        self.offset = offset
        self.block_dim = block_dim

    def forward(self, team_state, joint_action):
        block = joint_action[:, self.offset:self.offset + self.block_dim]
        return block.sum(dim=-1, keepdim=True)


class _MockCommanderActor(nn.Module):
    """Mock commander actor: get_action returns deterministic action."""
    def __init__(self, action_value: float = 0.5):
        super().__init__()
        self.action_value = action_value

    def get_action(self, obs):
        action = torch.full((obs.shape[0], CMD_BLOCK_DIM), self.action_value,
                            device=obs.device)
        return action, None, None, None


class _MockRadarActor(nn.Module):
    """Mock radar actor: get_action returns 303-dim compact block directly."""
    def __init__(self, action_value: float = 0.3):
        super().__init__()
        self.action_value = action_value

    def get_action(self, obs):
        action = torch.full((obs.shape[0], RADAR_BLOCK_DIM), self.action_value,
                            device=obs.device)
        return action, None, None, None


def test_constant_critic_zero_advantage():
    """If Q is constant, advantage = Q - mean(Q) = 0."""
    critic = _ConstantCritic(q_value=1.0)
    actor = _MockCommanderActor(action_value=0.5)
    B = 4
    team_state = torch.randn(B, 104)
    joint_action = torch.randn(B, JOINT_ACTION_DIM)
    obs = torch.randn(B, 76)

    torch.manual_seed(42)
    adv, baseline = coma_counterfactual_advantage(
        critic=critic, team_state=team_state, joint_action=joint_action,
        agent_kind="commander", team_idx=0, agent_idx=0,
        actor=actor, obs=obs, n_samples=4,
    )
    assert adv.shape == (B,)
    assert torch.allclose(adv, torch.zeros(B), atol=1e-6), (
        f"Expected zero advantage, got {adv}"
    )


def test_bit_exact_reproducibility():
    """Same seed → identical advantage (bit-exact)."""
    critic = _LinearCritic(offset=CMD_OFFSETS[0], block_dim=CMD_BLOCK_DIM)
    actor = _MockCommanderActor(action_value=0.7)
    B = 4
    team_state = torch.randn(B, 104)
    joint_action = torch.randn(B, JOINT_ACTION_DIM)
    obs = torch.randn(B, 76)

    torch.manual_seed(42)
    adv1, _ = coma_counterfactual_advantage(
        critic=critic, team_state=team_state, joint_action=joint_action,
        agent_kind="commander", team_idx=0, agent_idx=0,
        actor=actor, obs=obs, n_samples=4,
    )
    torch.manual_seed(42)
    adv2, _ = coma_counterfactual_advantage(
        critic=critic, team_state=team_state, joint_action=joint_action,
        agent_kind="commander", team_idx=0, agent_idx=0,
        actor=actor, obs=obs, n_samples=4,
    )
    assert torch.equal(adv1, adv2), "Same seed should give bit-exact advantage"


def test_radar_slot_isolation():
    """Marginalizing radar-0 vs radar-1 must affect different slots.

    With _LinearCritic on slot offset N, only marginalizing that slot
    changes Q. Marginalizing a different slot leaves Q unchanged →
    baseline = Q_actual (no change) → advantage = 0.
    """
    # Critic sensitive to team-0 radar-0's slot only.
    critic = _LinearCritic(offset=RADAR_OFFSETS[0], block_dim=RADAR_BLOCK_DIM)
    actor = _MockRadarActor(action_value=0.9)
    B = 2
    team_state = torch.randn(B, 104)
    # Make radar-0 slot distinct from sample
    joint_action = torch.zeros(B, JOINT_ACTION_DIM)
    joint_action[:, RADAR_OFFSETS[0]:RADAR_OFFSETS[0] + RADAR_BLOCK_DIM] = 0.1
    obs = torch.randn(B, 25 * 25 * 8)  # arbitrary obs shape

    torch.manual_seed(42)
    # Marginalize radar-0 → critic sees changes → nonzero advantage
    adv_r0, _ = coma_counterfactual_advantage(
        critic=critic, team_state=team_state, joint_action=joint_action,
        agent_kind="radar", team_idx=0, agent_idx=0,
        actor=actor, obs=obs, n_samples=4,
    )
    # Marginalize radar-1 → critic does NOT see changes (slot untouched) → 0 advantage
    adv_r1, _ = coma_counterfactual_advantage(
        critic=critic, team_state=team_state, joint_action=joint_action,
        agent_kind="radar", team_idx=0, agent_idx=1,
        actor=actor, obs=obs, n_samples=4,
    )
    assert adv_r0.abs().max() > 0.01, (
        f"radar-0 marginalize should change Q; got adv={adv_r0}"
    )
    assert adv_r1.abs().max() < 1e-4, (
        f"radar-1 marginalize should NOT change Q (critic ignores slot 1); "
        f"got adv={adv_r1}"
    )


def test_advantage_clip_sigma():
    """advantage_clip_sigma > 0 must clip to ±N·std."""
    # Critic returns Q = action[0] for the commander slot. With random samples,
    # advantage variance is nontrivial; clip should compress it.
    critic = _LinearCritic(offset=CMD_OFFSETS[0], block_dim=CMD_BLOCK_DIM)
    actor = _MockCommanderActor(action_value=0.7)
    B = 8
    team_state = torch.randn(B, 104)
    joint_action = torch.randn(B, JOINT_ACTION_DIM)
    obs = torch.randn(B, 76)

    torch.manual_seed(42)
    adv_unclipped, _ = coma_counterfactual_advantage(
        critic=critic, team_state=team_state, joint_action=joint_action,
        agent_kind="commander", team_idx=0, agent_idx=0,
        actor=actor, obs=obs, n_samples=4, advantage_clip_sigma=0.0,
    )
    torch.manual_seed(42)
    adv_clipped, _ = coma_counterfactual_advantage(
        critic=critic, team_state=team_state, joint_action=joint_action,
        agent_kind="commander", team_idx=0, agent_idx=0,
        actor=actor, obs=obs, n_samples=4, advantage_clip_sigma=2.0,
    )
    # Clipped max abs should be ≤ 2·std + tiny epsilon
    std = adv_unclipped.std() + 1e-8
    assert adv_clipped.abs().max() <= 2.0 * std + 1e-5, (
        f"clip should bound |adv| ≤ 2·std={2.0 * std:.4f}; "
        f"got max |adv_clipped|={adv_clipped.abs().max():.4f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
