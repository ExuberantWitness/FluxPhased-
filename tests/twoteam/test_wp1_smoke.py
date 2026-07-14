"""Smoke test for WP1: StrongRuleCommander + AC + BR trainer."""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import torch
from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, MIRROR_GEOMETRY, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.br_trainer import TwoTeamBRTrainer
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


def test_strong_rule_runs():
    """StrongRuleCommander 30 steps NaN-free."""
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=30, geometry=RANDOM_GEOMETRY)
    rule = TwoTeamStrongRuleCommander()
    env.reset()
    for step in range(30):
        a0 = rule.get_action(env, 0)
        a1 = rule.get_action(env, 1)
        action = combine_team_actions(env, a0, a1)
        obs, r, d, info = env.step(action)
    assert not torch.isnan(obs["obs"]).any(), "NaN obs"
    assert not torch.isnan(r).any(), "NaN reward"
    # Strong rule should achieve SOME track on at least one enemy (trace_P < tau on at least 1)
    trace_P = env.tracker_P[..., 0, 0] + env.tracker_P[..., 2, 2]
    n_tracked = ((trace_P < env.tau_track) & env.tracker_initialized).sum(dim=-1)
    print(f"✅ StrongRule 30 steps NaN-free; mean tracked targets per team: "
          f"{n_tracked.float().mean(dim=-1).mean().item():.2f}")


def test_ac_action_shapes():
    """AC outputs correct action shapes per team."""
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=10)
    ac = TwoTeamCommanderActorCritic().to("cuda")
    obs = env.reset()
    a_t0, lp0, v0, vl0 = ac(obs["obs"][:, 0], obs["privileged"][:, 0])
    assert a_t0["task_alloc"].shape == (4, 2, 4), f"task_alloc shape: {a_t0['task_alloc'].shape}"
    assert a_t0["beam_target"].shape == (4, 2), f"beam_target shape: {a_t0['beam_target'].shape}"
    assert a_t0["laser_target"].shape == (4,), f"laser_target shape: {a_t0['laser_target'].shape}"
    assert a_t0["emission_on"].shape == (4, 2), f"emission_on shape: {a_t0['emission_on'].shape}"
    assert a_t0["freq_hop_rate"].shape == (4, 2), f"freq_hop_rate shape: {a_t0['freq_hop_rate'].shape}"
    # task_alloc sums to 1
    sums = a_t0["task_alloc"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), f"task_alloc sum: {sums}"
    # FIX 1: freq_hop_rate in [1, freq_hop_max]
    fh_min, fh_max = a_t0["freq_hop_rate"].min().item(), a_t0["freq_hop_rate"].max().item()
    assert 1.0 <= fh_min and fh_max <= ac.freq_hop_max + 1e-4, \
        f"freq_hop_rate out of [1, {ac.freq_hop_max}]: min={fh_min}, max={fh_max}"
    # Action fits env.step
    a_t1, _, _, _ = ac(obs["obs"][:, 1], obs["privileged"][:, 1])
    action = combine_team_actions(env, a_t0, a_t1)
    obs2, r, d, info = env.step(action)
    assert not torch.isnan(obs2["obs"]).any(), "NaN after AC step"
    print(f"✅ AC action shapes OK; task_alloc sums to 1; freq_hop ∈ [{fh_min:.2f}, {fh_max:.2f}]; env step NaN-free")


def test_evaluate_actions_consistent():
    """evaluate_actions should match forward log_prob for the same sample."""
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=5)
    ac = TwoTeamCommanderActorCritic().to("cuda")
    obs = env.reset()
    a, lp_fwd, _, _ = ac(obs["obs"][:, 0], obs["privileged"][:, 0])
    lp_eval, _, _, _ = ac.evaluate_actions(obs["obs"][:, 0], a, obs["privileged"][:, 0])
    diff = (lp_fwd - lp_eval).abs().max().item()
    assert diff < 1e-4, f"log_prob diff: {diff}"
    print(f"✅ evaluate_actions log_prob matches forward (diff {diff:.2e})")


def test_br_trainer_smoke():
    """BR trainer 5 iters without crash, no NaN."""
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=50, geometry=RANDOM_GEOMETRY)
    ac = TwoTeamCommanderActorCritic().to("cuda")
    rule = TwoTeamStrongRuleCommander()
    trainer = TwoTeamBRTrainer(ac, frozen_opponent=rule,
                                n_epochs=2, minibatch_size=32)
    history = trainer.train(env, n_iterations=5, horizon=50, learning_team=0,
                             log_every=1)
    assert len(history) == 5
    # No NaN in params
    for p in ac.parameters():
        assert not torch.isnan(p).any(), "NaN in AC params after BR training"
    # Adv std reasonable
    last_adv_std = history[-1]["adv_std"]
    assert 0.05 < last_adv_std < 100, f"adv_std out of range: {last_adv_std}"
    print(f"✅ BR trainer 5 iters OK; final adv_std={last_adv_std:.3f}, "
          f"entropy={history[-1]['entropy']:.3f}, kl={history[-1]['approx_kl']:.4f}")


if __name__ == "__main__":
    print("=== Test 1: StrongRule runs 30 steps ===")
    test_strong_rule_runs()
    print()
    print("=== Test 2: AC action shapes ===")
    test_ac_action_shapes()
    print()
    print("=== Test 3: evaluate_actions consistency ===")
    test_evaluate_actions_consistent()
    print()
    print("=== Test 4: BR trainer 5-iter smoke ===")
    test_br_trainer_smoke()
    print()
    print("🎉 all WP1 smoke tests PASS")
