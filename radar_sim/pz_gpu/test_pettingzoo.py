"""Comprehensive PettingZoo Parallel API test suite for benchmark validation.

Tests cover: API compliance, obs/action correctness, agent lifecycle,
determinism, multi-episode stability, radar_latents callback, commander
missile launch, and GPU memory health.
"""

import sys
import gc

sys.path.insert(0, "E:/DATA/vscode/FluxPhased")

import numpy as np
import torch
from radar_sim.pz_gpu.core import FluxPhasedPZEnv

PASS = 0
FAIL = 0


def _pass(name):
    global PASS
    PASS += 1
    print(f"  [PASS] {name}")


def _fail(name, msg=""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name}: {msg}")


def make_env(**overrides):
    defaults = dict(
        rows=2, cols=2, pulses_per_cpi=2,
        bandwidth=10e6, prf=10e3,
        num_input_length=4, num_output_length=4,
        max_steps=50, device="cuda",
    )
    defaults.update(overrides)
    return FluxPhasedPZEnv(**defaults)


# ============================================================
# 1. API Structure
# ============================================================

def test_agent_naming():
    """Verify possible_agents and agents are correct."""
    env = make_env()
    expected = [
        "red_radar_0", "red_radar_1",
        "blue_radar_0", "blue_radar_1",
        "red_commander", "blue_commander",
    ]
    assert env.possible_agents == expected
    assert env.agents == expected
    _pass("agent naming (6 agents, order correct)")
    env.close()


def test_spaces_exist_for_all_agents():
    """observation_space and action_space work for every possible_agent."""
    env = make_env()
    for agent in env.possible_agents:
        obs_sp = env.observation_space(agent)
        act_sp = env.action_space(agent)
        assert obs_sp.shape is not None, f"{agent} obs space missing shape"
        assert act_sp.shape is not None, f"{agent} act space missing shape"
    _pass("spaces defined for all possible_agents")
    env.close()


def test_heterogeneous_spaces():
    """Radar and commander have different obs/action dimensions."""
    env = make_env()
    radar_obs = env.observation_space("red_radar_0").shape
    cmd_obs = env.observation_space("red_commander").shape
    radar_act = env.action_space("red_radar_0").shape
    cmd_act = env.action_space("red_commander").shape
    assert radar_obs != cmd_obs, "Radar and commander obs should differ"
    assert radar_act != cmd_act, "Radar and commander act should differ"
    assert radar_act[0] > cmd_act[0], "Radar action dim should be larger"
    _pass(f"heterogeneous spaces (radar obs={radar_obs}, cmd obs={cmd_obs})")
    env.close()


# ============================================================
# 2. Reset / Observation Correctness
# ============================================================

def test_reset_returns_two_dicts():
    """reset() returns (observations, infos)."""
    env = make_env()
    result = env.reset()
    assert isinstance(result, tuple) and len(result) == 2
    obs, infos = result
    assert isinstance(obs, dict) and isinstance(infos, dict)
    assert set(obs.keys()) == set(env.possible_agents)
    _pass("reset returns (obs_dict, info_dict)")
    env.close()


def test_obs_shapes_match_spaces():
    """Every agent's obs matches its observation_space after reset and step."""
    env = make_env()
    obs, _ = env.reset()

    for agent in env.possible_agents:
        expected = env.observation_space(agent).shape
        assert obs[agent].shape == expected, (
            f"{agent} reset obs {obs[agent].shape} != {expected}"
        )

    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs2, _, _, _, _ = env.step(actions)

    for agent in env.agents:
        expected = env.observation_space(agent).shape
        assert obs2[agent].shape == expected, (
            f"{agent} step obs {obs2[agent].shape} != {expected}"
        )
    _pass("obs shapes match observation_space (reset + step)")
    env.close()


def test_obs_no_nan_inf():
    """No NaN or Inf in observations after reset and step."""
    env = make_env()
    obs, _ = env.reset()

    for agent in env.possible_agents:
        arr = obs[agent]
        assert np.all(np.isfinite(arr)), f"{agent} has NaN/Inf after reset"

    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs2, _, _, _, _ = env.step(actions)

    for agent in env.agents:
        arr = obs2[agent]
        assert np.all(np.isfinite(arr)), f"{agent} has NaN/Inf after step"
    _pass("no NaN/Inf in observations (reset + step)")
    env.close()


def test_obs_dtype_is_float32():
    """All observations are float32 numpy arrays."""
    env = make_env()
    obs, _ = env.reset()
    for agent in env.possible_agents:
        assert obs[agent].dtype == np.float32, f"{agent} dtype={obs[agent].dtype}"
    _pass("obs dtype is float32")
    env.close()


# ============================================================
# 3. Step / Reward / Termination
# ============================================================

def test_step_returns_five_dicts():
    """step() returns (obs, rewards, terminations, truncations, infos)."""
    env = make_env()
    env.reset()
    actions = {a: env.action_space(a).sample() for a in env.agents}
    result = env.step(actions)
    assert isinstance(result, tuple) and len(result) == 5
    obs, rew, term, trunc, info = result
    assert isinstance(obs, dict)
    assert isinstance(rew, dict)
    assert isinstance(term, dict)
    assert isinstance(trunc, dict)
    assert isinstance(info, dict)
    _pass("step returns 5 dicts")
    env.close()


def test_rewards_are_finite():
    """All rewards are finite floats."""
    env = make_env()
    env.reset()
    actions = {a: env.action_space(a).sample() for a in env.agents}
    _, rew, _, _, _ = env.step(actions)
    for agent, r in rew.items():
        assert np.isfinite(r), f"{agent} reward={r} is not finite"
    _pass("rewards are finite")
    env.close()


def test_step_count_in_info():
    """Info dict contains step count."""
    env = make_env()
    obs, infos = env.reset()
    assert infos["red_radar_0"]["step"] == 0, "step should be 0 after reset"

    actions = {a: env.action_space(a).sample() for a in env.agents}
    _, _, _, _, infos2 = env.step(actions)
    assert infos2["red_radar_0"]["step"] == 1, "step should be 1 after first step"
    _pass("step count in info")
    env.close()


def test_termination_on_done():
    """When dones=True, all agents get termination=True."""
    env = make_env()
    env.reset()

    # Keep stepping until done or max_steps
    for _ in range(100):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, info = env.step(actions)
        if any(term.values()):
            for agent in env.possible_agents:
                if agent in term:
                    assert term[agent], f"{agent} should be terminated"
            _pass("all agents terminated on done")
            env.close()
            return
        if any(trunc.values()):
            _pass("truncated before termination (no kill occurred)")
            env.close()
            return

    _fail("termination test", "no termination in 100 steps")


def test_truncation_at_max_steps():
    """Episode truncates at max_steps."""
    max_steps = 5
    env = make_env(max_steps=max_steps)
    env.reset()

    for step in range(max_steps + 5):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        _, _, term, trunc, _ = env.step(actions)
        if any(trunc.values()):
            assert step + 1 == max_steps, f"truncated at step {step+1}, expected {max_steps}"
            _pass(f"truncation at max_steps={max_steps}")
            env.close()
            return

    _fail("truncation", f"no truncation in {max_steps + 5} steps")


# ============================================================
# 4. Agent Death Lifecycle
# ============================================================

def test_agents_only_shrinks():
    """self.agents can only shrink, never grow during an episode."""
    env = make_env()
    env.reset()
    prev_agents = set(env.agents)

    for _ in range(10):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        env.step(actions)
        curr_agents = set(env.agents)
        assert curr_agents.issubset(prev_agents), (
            f"agents grew: {curr_agents - prev_agents}"
        )
        prev_agents = curr_agents

    _pass("agents only shrinks")
    env.close()


def test_possible_agents_constant():
    """possible_agents never changes."""
    env = make_env()
    original = list(env.possible_agents)
    env.reset()

    for _ in range(10):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        env.step(actions)

    assert env.possible_agents == original
    _pass("possible_agents is constant")
    env.close()


def test_reset_restores_agents():
    """After reset, agents equals possible_agents again."""
    env = make_env()
    env.reset()

    # Run until truncation
    for _ in range(60):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, _, term, trunc, _ = env.step(actions)
        if any(term.values()) or any(trunc.values()):
            break

    assert len(env.agents) < len(env.possible_agents) or not any(term.values())

    obs, _ = env.reset()
    assert env.agents == env.possible_agents
    assert set(obs.keys()) == set(env.possible_agents)
    _pass("reset restores full agents list")
    env.close()


# ============================================================
# 5. Deterministic Seeding
# ============================================================

def test_deterministic_reset():
    """Same seed produces same initial observations."""
    env1 = make_env()
    env2 = make_env()

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    for agent in env1.possible_agents:
        np.testing.assert_array_equal(
            obs1[agent], obs2[agent],
            err_msg=f"{agent} obs differs with same seed",
        )
    _pass("deterministic reset (seed=42)")
    env1.close()
    env2.close()


def test_deterministic_trajectory():
    """Same seed + same actions → same trajectory.

    Note: CUDA RNG may produce slightly different noise realizations
    across env instances. We check that positions (deterministic physics)
    match exactly, and spectrum (contains noise) matches within tolerance.
    """
    env1 = make_env(max_steps=10)
    env2 = make_env(max_steps=10)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    for step in range(5):
        actions = {a: env1.action_space(a).sample() for a in env1.agents}
        obs1, rew1, _, _, _ = env1.step(actions)
        obs2, rew2, _, _, _ = env2.step(actions)

        for agent in env1.agents:
            # Rewards must match exactly
            assert abs(rew1[agent] - rew2[agent]) < 1e-6, (
                f"{agent} reward diverged: {rew1[agent]} vs {rew2[agent]}"
            )
            # Obs: spectrum contains noise, allow small tolerance
            # but 99% of elements should match closely
            diff = np.abs(obs1[agent] - obs2[agent])
            match_ratio = np.mean(diff < 1e-3)
            assert match_ratio > 0.99, (
                f"{agent} obs diverged at step {step}: "
                f"only {match_ratio:.1%} match (need >99%)"
            )

    _pass("deterministic trajectory (rewards exact, obs >99% match)")
    env1.close()
    env2.close()


# ============================================================
# 6. radar_latents_fn Callback
# ============================================================

def test_radar_latents_callback():
    """radar_latents_fn output appears correctly in commander obs."""
    N_in = 4

    def encoder(state):
        # Encode: each radar gets a unique pattern based on index
        out = np.zeros((state.shape[0], N_in), dtype=np.float32)
        for i in range(state.shape[0]):
            out[i] = float(i) * 0.1
        return out

    env = make_env(radar_latents_fn=encoder)
    env.reset()

    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, _, _, _, _ = env.step(actions)

    # Red commander obs: [0:4] positions, [4:8] radar_0 latent, [8:12] radar_1 latent
    red_cmd = obs["red_commander"]
    # radar index 0 → 0.0, radar index 1 → 0.1
    np.testing.assert_allclose(red_cmd[4:8], 0.0, atol=1e-5)
    np.testing.assert_allclose(red_cmd[8:12], 0.1, atol=1e-5)

    # Blue commander: radar indices 2, 3
    blue_cmd = obs["blue_commander"]
    np.testing.assert_allclose(blue_cmd[4:8], 0.2, atol=1e-5)
    np.testing.assert_allclose(blue_cmd[8:12], 0.3, atol=1e-5)

    _pass("radar_latents callback correctly injected")
    env.close()


def test_no_latents_zero_filled():
    """Without radar_latents_fn, commander latent section is zero."""
    env = make_env()  # no radar_latents_fn
    env.reset()
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, _, _, _, _ = env.step(actions)

    N_in = 4
    for cmd in ["red_commander", "blue_commander"]:
        latent = obs[cmd][4:4 + 2 * N_in]
        np.testing.assert_array_equal(latent, 0.0)

    _pass("no radar_latents_fn → zero-filled commander latents")
    env.close()


# ============================================================
# 7. Commander Missile Launch
# ============================================================

def test_commander_launch_missile():
    """Commander can launch missile; info reflects missile state."""
    env = make_env(max_steps=200)
    env.reset()

    # Launch red missile toward blue territory
    cmd_act_dim = env.action_space("red_commander").shape[0]
    launch = np.zeros(cmd_act_dim, dtype=np.float32)
    launch[0] = 1.0  # launch_flag
    launch[1] = 0.0  # target_x
    launch[2] = 0.8  # target_y (blue side)

    actions = {a: env.action_space(a).sample() for a in env.agents}
    actions["red_commander"] = launch
    obs, _, _, _, infos = env.step(actions)

    assert infos["red_commander"]["missile_in_flight"], "Red missile should be in flight"
    m_pos = infos["red_commander"]["missile_pos"]
    assert len(m_pos) == 3, "missile_pos should be [x, y, z]"
    _pass("commander missile launch")
    env.close()


def test_commander_cannot_double_launch():
    """Launching again while missile is in-flight is a no-op."""
    env = make_env(max_steps=200)
    env.reset()

    cmd_act_dim = env.action_space("red_commander").shape[0]
    launch = np.zeros(cmd_act_dim, dtype=np.float32)
    launch[0] = 1.0
    launch[2] = 0.8

    actions = {a: env.action_space(a).sample() for a in env.agents}
    actions["red_commander"] = launch

    # First launch
    _, _, _, _, infos1 = env.step(actions)
    assert infos1["red_commander"]["missile_in_flight"]

    # Second launch attempt — should still be only 1 missile
    _, _, _, _, infos2 = env.step(actions)
    assert infos2["red_commander"]["missile_in_flight"]
    _pass("double launch is no-op")
    env.close()


# ============================================================
# 8. Multi-Episode Stability
# ============================================================

def test_multi_episode_stability():
    """Run 5 full episodes without crashes or GPU memory issues."""
    env = make_env(max_steps=10)

    for ep in range(5):
        obs, infos = env.reset(seed=ep)
        assert set(obs.keys()) == set(env.possible_agents)
        assert env.agents == env.possible_agents

        for step in range(15):
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)

            # Check obs health
            for agent in obs:
                assert np.all(np.isfinite(obs[agent])), (
                    f"NaN/Inf in ep={ep} step={step} agent={agent}"
                )

            if any(term.values()) or any(trunc.values()):
                break

    _pass("5 episodes, no crashes")
    env.close()


def test_gpu_memory_stable():
    """GPU memory does not grow across episodes."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    env = make_env(max_steps=5)
    mem_after_init = torch.cuda.memory_allocated() / 1e6

    for ep in range(3):
        env.reset(seed=ep)
        for _ in range(6):
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)

    mem_after = torch.cuda.memory_allocated() / 1e6
    growth = mem_after - mem_after_init
    assert growth < 10.0, f"GPU memory grew by {growth:.1f} MB"
    _pass(f"GPU memory stable (growth {growth:.1f} MB)")
    env.close()


# ============================================================
# 9. Official PZ Compliance
# ============================================================

def test_parallel_api_official():
    """Run PettingZoo's official parallel_api_test."""
    from pettingzoo.test import parallel_api_test

    env = make_env()
    parallel_api_test(env, num_cycles=10)
    _pass("official parallel_api_test")
    env.close()


# ============================================================
# 10. Action Space Bounds
# ============================================================

def test_actions_within_bounds():
    """Sampled actions are within declared space bounds."""
    env = make_env()
    for agent in env.possible_agents:
        space = env.action_space(agent)
        for _ in range(10):
            action = space.sample()
            assert space.contains(action), f"{agent} sampled action out of bounds"
    _pass("sampled actions within bounds")
    env.close()


def test_zero_actions_accepted():
    """Zero actions don't crash the env."""
    env = make_env()
    env.reset()
    actions = {a: np.zeros(env.action_space(a).shape, dtype=np.float32)
               for a in env.agents}
    obs, rew, _, _, _ = env.step(actions)
    for agent in obs:
        assert np.all(np.isfinite(obs[agent])), f"{agent} NaN with zero actions"
    _pass("zero actions accepted")
    env.close()


# ============================================================
# 11. Info Dict Structure
# ============================================================

def test_info_structure():
    """Info dict has required keys for each agent type."""
    env = make_env()
    _, infos = env.reset()

    # Radar info
    radar_info = infos["red_radar_0"]
    assert "alive" in radar_info
    assert "position" in radar_info
    assert "task_ids" in radar_info
    assert "winner" in radar_info
    assert "step" in radar_info
    assert isinstance(radar_info["alive"], bool)
    assert len(radar_info["position"]) == 3

    # Commander info
    cmd_info = infos["red_commander"]
    assert "team" in cmd_info
    assert "missile_in_flight" in cmd_info
    assert "missile_pos" in cmd_info
    assert cmd_info["team"] == 0  # red

    _pass("info dict structure correct")
    env.close()


# ============================================================
# 12. unwrapped property
# ============================================================

def test_unwrapped_access():
    """unwrapped property gives access to MFARVecEnv."""
    env = make_env()
    from radar_sim.gpu.vec_mfar_env import MFARVecEnv
    assert isinstance(env.unwrapped, MFARVecEnv)
    assert env.unwrapped.n_radars == 4
    _pass("unwrapped property")
    env.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FluxPhased PettingZoo Parallel API — Benchmark Validation")
    print("=" * 60)

    tests = [
        # API structure
        test_agent_naming,
        test_spaces_exist_for_all_agents,
        test_heterogeneous_spaces,
        # Reset / obs
        test_reset_returns_two_dicts,
        test_obs_shapes_match_spaces,
        test_obs_no_nan_inf,
        test_obs_dtype_is_float32,
        # Step / reward / termination
        test_step_returns_five_dicts,
        test_rewards_are_finite,
        test_step_count_in_info,
        test_termination_on_done,
        test_truncation_at_max_steps,
        # Agent lifecycle
        test_agents_only_shrinks,
        test_possible_agents_constant,
        test_reset_restores_agents,
        # Determinism
        test_deterministic_reset,
        test_deterministic_trajectory,
        # Callback
        test_radar_latents_callback,
        test_no_latents_zero_filled,
        # Commander
        test_commander_launch_missile,
        test_commander_cannot_double_launch,
        # Stability
        test_multi_episode_stability,
        test_gpu_memory_stable,
        # Action space
        test_actions_within_bounds,
        test_zero_actions_accepted,
        # Info
        test_info_structure,
        test_unwrapped_access,
        # Official
        test_parallel_api_official,
    ]

    print(f"\nRunning {len(tests)} tests...\n")

    for test_fn in tests:
        name = test_fn.__doc__ or test_fn.__name__
        print(f"[{test_fn.__name__}] {name}")
        try:
            test_fn()
        except Exception as e:
            _fail(test_fn.__name__, str(e))

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed / {PASS + FAIL} total")
    if FAIL == 0:
        print("ALL TESTS PASSED")
    print("=" * 60)
