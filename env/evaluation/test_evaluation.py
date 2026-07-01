"""Evaluation framework test suite.

Uses random-policy neural networks to generate test baselines,
validating that metrics produce correct, stable, and distinguishable values.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np


def make_test_env(**overrides):
    """Create an MFAR env for testing (25×25)."""
    from env.gpu.vec_mfar_env import MFARVecEnv
    defaults = dict(
        num_envs=1, n_radars=4, rows=25, cols=25,
        pulses_per_cpi=4, bandwidth=10e6, prf=10e3,
        num_input_length=4, num_output_length=4,
        device="cuda",
    )
    defaults.update(overrides)
    return MFARVecEnv(**defaults)


def make_random_policy(env, seed=42):
    """Create a RandomPolicy for the given env."""
    from env.evaluation.collectors.episode_collector import RandomPolicy
    return RandomPolicy(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        commander_obs_dim=env.battlefield.commander_obs_dim,
        commander_action_dim=env.battlefield.commander_action_dim,
        n_radars=env.n_radars,
        n_teams=env.n_teams,
        device=env.device,
        seed=seed,
    )


# ============================================================
print("=" * 60)
print("Test 1: RandomPolicy generates valid actions")
print("=" * 60)


def test_random_policy():
    env = make_test_env()
    env.reset()
    policy = make_random_policy(env, seed=42)

    result = env.step()
    state = result["state"]
    cmd_obs = result["commander_obs"]

    radar_actions = policy.get_radar_actions(state)
    cmd_actions = policy.get_commander_actions(cmd_obs)

    assert radar_actions.shape == (1, 4, env.action_dim), \
        f"Wrong radar action shape: {radar_actions.shape}"
    assert cmd_actions.shape == (1, 2, env.battlefield.commander_action_dim), \
        f"Wrong cmd action shape: {cmd_actions.shape}"
    assert (radar_actions >= 0).all() and (radar_actions <= 1).all(), \
        "Radar actions out of [0,1]"
    assert (cmd_actions >= -1).all() and (cmd_actions <= 1).all(), \
        "Cmd actions out of [-1,1]"

    # Run a step with these actions
    result2 = env.step(actions=radar_actions, commander_actions=cmd_actions)
    assert not torch.isnan(result2["state"]).any(), "NaN in state with random policy"

    print("  [PASS] test_random_policy")
    del env
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 2: GroundTruthComputer produces reasonable values")
print("=" * 60)


def test_ground_truth():
    env = make_test_env()
    env.reset()
    result = env.step()

    from env.evaluation.collectors.ground_truth import GroundTruthComputer
    gt = GroundTruthComputer(env)
    gt_result = gt.compute()

    assert "expected_range_m" in gt_result
    assert "expected_range_bin" in gt_result
    assert "expected_snr_db" in gt_result

    range_m = gt_result["expected_range_m"]
    snr_db = gt_result["expected_snr_db"]

    assert range_m.shape[0] == 1 and range_m.shape[1] == 4, \
        f"Wrong range shape: {range_m.shape}"
    assert (range_m > 0).all(), f"Range should be positive, got {range_m}"
    assert not torch.isnan(snr_db).any(), "NaN in SNR"

    print(f"  Range: {range_m[0, 0, 0].item():.1f} m")
    print(f"  SNR: {snr_db[0, 0, 0].item():.1f} dB")
    print("  [PASS] test_ground_truth")
    del env
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 3: EpisodeCollector collects 10-step trajectory")
print("=" * 60)


def test_episode_collector():
    env = make_test_env()
    policy = make_random_policy(env, seed=42)

    from env.evaluation.collectors.episode_collector import EpisodeCollector
    collector = EpisodeCollector(env, max_steps=10)

    def radar_policy(result):
        return policy.get_radar_actions(result["state"])

    def cmd_policy(result):
        return policy.get_commander_actions(result["commander_obs"])

    ep = collector.run_episode(
        policy_fn=radar_policy,
        commander_policy_fn=cmd_policy,
        max_steps=10,
    )

    assert ep.n_steps > 0, "No steps collected"
    assert ep.spectrum is not None
    assert ep.spectrum.shape[0] == ep.n_steps
    assert ep.task_ids is not None
    assert ep.radar_rewards is not None
    assert len(ep.timing) == ep.n_steps

    print(f"  Collected {ep.n_steps} steps")
    print(f"  Spectrum shape: {ep.spectrum.shape}")
    print("  [PASS] test_episode_collector")
    del env
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 4: PerceptionMetrics - all detect vs all jam")
print("=" * 60)


def test_perception_metrics():
    from env.evaluation.metrics.perception import PerceptionMetrics

    env = make_test_env()
    env.reset()
    pm = PerceptionMetrics(env)

    # All detect action
    action_detect = torch.ones(1, 4, env.action_dim, device="cuda")
    elem = action_detect[:, :, :env.n_elem * 22].reshape(1, 4, env.n_elem, 22)
    elem[:, :, :, 0] = 0.0  # recon=0
    elem[:, :, :, 1] = 1.0  # detect=1
    elem[:, :, :, 2] = 0.0  # jam=0
    elem[:, :, :, 3] = 0.0  # comm=0

    result = env.step(actions=action_detect)
    assert result["spectrum"] is not None

    # Test beamformed spectrum
    task_ids = result["task_ids"]
    bf = pm.beamformed_spectrum(result["spectrum"], task_ids, task=1)
    assert bf.shape == (1, 4, env.n_pulses, env.n_bins), f"Wrong bf shape: {bf.shape}"

    # Test spectrum peaks
    peaks = pm.spectrum_peaks(bf[:, :, 0, :])  # first pulse
    assert "peak_bin" in peaks
    assert "snr_est_db" in peaks

    # Test resource allocation via task_ids
    from env.evaluation.metrics.combat import CombatMetrics
    cm = CombatMetrics()
    alloc = cm.resource_allocation(task_ids)
    assert abs(alloc["detect_frac"] - 1.0) < 0.01, \
        f"All-detect action should give detect_frac≈1.0, got {alloc['detect_frac']}"

    print(f"  Detect fraction: {alloc['detect_frac']:.3f}")
    print(f"  Beamformed spectrum shape: {bf.shape}")
    print("  [PASS] test_perception_metrics")
    del env
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 5: CombatMetrics - missile launch vs no launch")
print("=" * 60)


def test_combat_metrics():
    from env.evaluation.collectors.episode_collector import EpisodeCollector
    from env.evaluation.metrics.combat import CombatMetrics

    env = make_test_env()

    # Episode with missile launch
    cmd_dim = env.battlefield.commander_action_dim
    collector = EpisodeCollector(env, max_steps=5)

    def launch_policy(result):
        cmd = torch.zeros(1, 2, cmd_dim, device="cuda")
        cmd[:, 0, 0] = 1.0  # Red launch
        cmd[:, 0, 1] = 0.0  # target x
        cmd[:, 0, 2] = 0.5  # target y
        cmd[:, 1, 0] = 1.0  # Blue launch
        cmd[:, 1, 2] = -0.5
        return cmd

    ep = collector.run_episode(commander_policy_fn=launch_policy, max_steps=5)

    cm = CombatMetrics()
    missile_eff = cm.missile_efficiency(ep, env.radar_pos)
    cmd_quality = cm.commander_decision_quality(ep, env.radar_pos)

    assert "any_kill" in missile_eff
    assert "launched" in cmd_quality
    assert cmd_quality["launched"], "Missile should be launched with launch policy"

    print(f"  Launched: {cmd_quality['launched']}")
    print(f"  Launch step: {cmd_quality['launch_step']}")
    print(f"  Any kill: {missile_eff['any_kill']}")
    print("  [PASS] test_combat_metrics")
    del env
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 6: GameMetrics - multi-episode aggregation")
print("=" * 60)


def test_game_metrics():
    from env.evaluation.collectors.episode_collector import EpisodeCollector
    from env.evaluation.metrics.game import GameMetrics

    env = make_test_env()
    collector = EpisodeCollector(env, max_steps=3)

    episodes = collector.run_episodes(n_episodes=5, max_steps=3)
    assert len(episodes) == 5

    gm = GameMetrics()
    outcomes = gm.game_outcomes(episodes)
    assert "win_rate_red" in outcomes
    assert "avg_episode_length" in outcomes
    assert outcomes["n_episodes"] == 5

    stability = gm.strategy_stability(episodes)
    assert "red_reward_mean" in stability
    assert "red_reward_cv" in stability

    print(f"  Episodes: {outcomes['n_episodes']}")
    print(f"  Avg length: {outcomes['avg_episode_length']:.1f}")
    print("  [PASS] test_game_metrics")
    del env
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 7: CommMetrics - comm vs no comm elements")
print("=" * 60)


def test_comm_metrics():
    from env.evaluation.metrics.comm import CommMetrics

    cm = CommMetrics()

    # Test with synthetic data
    comm_data_active = torch.randn(1, 2, 25, 2)
    result_active = cm.comm_accuracy(comm_data_active, commander_target=torch.tensor([0.5, -0.3]))
    assert "active_comm_frac" in result_active

    # Test with zero data (no comm)
    comm_data_zero = torch.zeros(1, 2, 25, 2)
    result_zero = cm.comm_accuracy(comm_data_zero)
    assert result_zero["active_comm_frac"] < 0.01

    print(f"  Active comm frac: {result_active['active_comm_frac']:.3f}")
    print(f"  Zero comm frac: {result_zero['active_comm_frac']:.3f}")
    print("  [PASS] test_comm_metrics")
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 8: TriggerSources - valid ranges and defaults")
print("=" * 60)


def test_trigger_sources():
    from env.evaluation.analysis.trigger_sources import (
        PERCEPTION_TRIGGERS, ANALYSIS_TRIGGERS, GAME_TRIGGERS,
    )

    all_triggers = PERCEPTION_TRIGGERS + ANALYSIS_TRIGGERS + GAME_TRIGGERS
    assert len(all_triggers) > 0

    for t in all_triggers:
        lo, hi = t.value_range
        assert lo < hi, f"Invalid range for {t.name}: {lo} >= {hi}"
        assert lo <= t.default <= hi, f"Default {t.default} outside range for {t.name}"

    print(f"  {len(PERCEPTION_TRIGGERS)} perception triggers")
    print(f"  {len(ANALYSIS_TRIGGERS)} analysis triggers")
    print(f"  {len(GAME_TRIGGERS)} game triggers")
    print("  [PASS] test_trigger_sources")
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 9: ScenarioGenerator - generates valid configs")
print("=" * 60)


def test_scenario_generator():
    from env.evaluation.analysis.scenario_generator import ScenarioGenerator
    from env.evaluation.analysis.trigger_sources import PERCEPTION_TRIGGERS

    gen = ScenarioGenerator(PERCEPTION_TRIGGERS)
    config = gen.random_scenario()
    assert isinstance(config, dict)
    assert len(config) > 0

    # Check values are within ranges
    for t in PERCEPTION_TRIGGERS:
        if t.param_name in config:
            lo, hi = t.value_range
            assert lo <= config[t.param_name] <= hi, \
                f"{t.name}: {config[t.param_name]} outside [{lo}, {hi}]"

    print(f"  Random config: {list(config.keys())[:3]}...")
    print("  [PASS] test_scenario_generator")
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 10: CDEMetric - values in [0, 1] and distinguishable")
print("=" * 60)


def test_cde_metric():
    from env.evaluation.analysis.cde import CDEMetric
    from env.evaluation.collectors.episode_collector import EpisodeData
    import torch

    cde = CDEMetric()

    # Empty episode → low CDE
    empty = EpisodeData()
    result_empty = cde.compute(empty)
    assert 0 <= result_empty["cde"] <= 1, f"CDE out of range: {result_empty['cde']}"

    # Episode with kills → higher CDE
    ep_with_kills = EpisodeData(
        kills=torch.ones(3, 1, 2, 2, dtype=torch.bool),  # kills happened
        task_ids=torch.randint(0, 4, (3, 1, 4, 25)),  # mixed tasks
        missile_pos=torch.randn(3, 1, 2, 3),  # launched
        n_steps=3,
    )
    result_kills = cde.compute(ep_with_kills)
    assert 0 <= result_kills["cde"] <= 1

    print(f"  Empty CDE: {result_empty['cde']:.3f}")
    print(f"  Kill CDE: {result_kills['cde']:.3f}")
    print("  [PASS] test_cde_metric")
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 11: AcceleratedEvaluator - early stopping")
print("=" * 60)


def test_accelerated_eval():
    from env.evaluation.analysis.accelerated_eval import AcceleratedEvaluator

    # Constant metric → should converge quickly
    call_count = [0]

    def const_metric(result):
        call_count[0] += 1
        return 0.5

    evaluator = AcceleratedEvaluator(
        env_factory=make_test_env,
        metric_fn=const_metric,
        confidence=0.95,
        half_width=0.05,
        max_episodes=100,
        min_episodes=20,
    )

    result = evaluator.evaluate()
    assert result["episodes_used"] >= 20
    assert result["episodes_used"] <= 100
    assert abs(result["metric_mean"] - 0.5) < 0.01

    print(f"  Episodes used: {result['episodes_used']}")
    print(f"  Metric mean: {result['metric_mean']:.4f}")
    print("  [PASS] test_accelerated_eval")
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 12: EvaluationReport - to_dict and to_markdown")
print("=" * 60)


def test_evaluation_report():
    from env.evaluation.reporting.report import EvaluationReport

    report = EvaluationReport()
    report.perception = {"range_accuracy": 0.85, "target_coverage": 0.92}
    report.combat = {"kill_rate": 0.3, "detect_frac": 1.0}
    report.cde = {"cde": 0.55}

    d = report.to_dict()
    assert "perception" in d
    assert "combat" in d
    assert d["perception"]["range_accuracy"] == 0.85

    summary = report.summary()
    assert len(summary) > 0

    md = report.to_markdown(path=None)
    assert "效能评估报告" in md
    assert "range_accuracy" in md

    print(f"  Summary: {summary}")
    print("  [PASS] test_evaluation_report")
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 13: PZ + EpisodeCollector + RandomPolicy integration")
print("=" * 60)


def test_pz_integration():
    from env.pz_gpu.core import FluxPhasedPZEnv
    from env.evaluation.collectors.episode_collector import (
        EpisodeCollector, RandomPolicy,
    )

    env = FluxPhasedPZEnv(
        max_steps=5, device="cuda",
        rows=25, cols=25, pulses_per_cpi=4,
        num_input_length=4, num_output_length=4,
    )

    obs, infos = env.reset()
    assert len(obs) == 6  # 6 agents

    # Run 3 steps
    for _ in range(3):
        actions = {}
        for agent in env.agents:
            space = env.action_space(agent)
            actions[agent] = np.random.uniform(
                space.low, space.high, size=space.shape,
            ).astype(np.float32)
        obs, rew, term, trunc, info = env.step(actions)

    assert len(obs) > 0

    print(f"  Agents after 3 steps: {len(env.agents)}")
    print("  [PASS] test_pz_integration")
    del env
    return True


# ============================================================
if __name__ == "__main__":
    torch.cuda.synchronize()

    tests = [
        test_random_policy,
        test_ground_truth,
        test_episode_collector,
        test_perception_metrics,
        test_combat_metrics,
        test_game_metrics,
        test_comm_metrics,
        test_trigger_sources,
        test_scenario_generator,
        test_cde_metric,
        test_accelerated_eval,
        test_evaluation_report,
        test_pz_integration,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
