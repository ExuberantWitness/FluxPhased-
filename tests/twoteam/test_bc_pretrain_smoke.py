"""Smoke tests for BC pretrainer (WP1 BC → PPO paradigm).

Two tests:
  1. test_bc_pretrainer_smoke: BC 3 epochs without crash, train_loss decreases, no NaN
  2. test_bc_changes_deterministic_policy: after BC, AC's deterministic action differs from prior
"""

from __future__ import annotations
import sys
import os
import torch

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.bc_pretrain import TwoTeamBCPretrainer


def test_bc_pretrainer_smoke():
    """BC 3 epochs without crash, train_loss decreases, no NaN in params."""
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=30,
                        geometry=RANDOM_GEOMETRY, seed=42)
    ac = TwoTeamCommanderActorCritic().to("cuda")
    rule = TwoTeamStrongRuleCommander()
    bc = TwoTeamBCPretrainer(ac, lr=1e-3, batch_size=64)

    samples = bc.collect_samples(env, rule, n_samples=200, episode_steps=30)
    assert samples["obs"].shape[0] >= 200, f"got {samples['obs'].shape[0]}"
    assert samples["task_alloc"].shape[1:] == (2, 4)
    assert samples["freq_hop_rate"].shape[1:] == (2,)

    history = bc.train(samples, n_epochs=3, log_every=1)
    assert len(history) == 3
    assert history[-1]["train_loss"] < history[0]["train_loss"], (
        f"train_loss didn't decrease: {history[0]['train_loss']:.3f} → {history[-1]['train_loss']:.3f}"
    )
    for p in ac.parameters():
        assert not torch.isnan(p).any(), "NaN in AC params after BC"
    print(f"✅ BC smoke OK; train_loss {history[0]['train_loss']:.3f} → {history[-1]['train_loss']:.3f}")


def test_bc_changes_deterministic_policy():
    """After BC, AC's deterministic task_alloc should differ from random init."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=10,
                        geometry=RANDOM_GEOMETRY, seed=42)
    env.reset()
    ac = TwoTeamCommanderActorCritic().to("cuda")

    obs_dict = env.get_obs()
    with torch.no_grad():
        action_before, _ = ac.get_action_for_env(
            obs_dict["obs"][:, 0], obs_dict["privileged"][:, 0], deterministic=True)

    rule = TwoTeamStrongRuleCommander()
    bc = TwoTeamBCPretrainer(ac, lr=1e-3, batch_size=64)
    samples = bc.collect_samples(env, rule, n_samples=500, episode_steps=30)
    bc.train(samples, n_epochs=5, log_every=1)

    with torch.no_grad():
        action_after, _ = ac.get_action_for_env(
            obs_dict["obs"][:, 0], obs_dict["privileged"][:, 0], deterministic=True)

    diff = (action_before["task_alloc"] - action_after["task_alloc"]).abs().max().item()
    assert diff > 0.05, f"BC didn't change task_alloc (max diff {diff:.4f})"
    print(f"✅ BC shifted task_alloc by max {diff:.3f}")


if __name__ == "__main__":
    test_bc_pretrainer_smoke()
    test_bc_changes_deterministic_policy()
    print("\n✅ All BC smoke tests passed")
