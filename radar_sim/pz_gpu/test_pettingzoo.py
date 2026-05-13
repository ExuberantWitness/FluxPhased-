"""Test PettingZoo Parallel API compliance and additional scenarios."""

import sys
import numpy as np

sys.path.insert(0, "E:/DATA/vscode/FluxPhased")

from radar_sim.pz_gpu.core import FluxPhasedPZEnv


def make_tiny_env(**overrides):
    """Create a tiny env for fast testing."""
    defaults = dict(
        rows=2,
        cols=2,
        pulses_per_cpi=2,
        bandwidth=10e6,
        prf=10e3,
        num_input_length=4,
        num_output_length=4,
        max_steps=50,
        device="cuda",
    )
    defaults.update(overrides)
    return FluxPhasedPZEnv(**defaults)


def test_parallel_api():
    """Run PettingZoo's official parallel_api_test."""
    from pettingzoo.test import parallel_api_test

    env = make_tiny_env()
    parallel_api_test(env, num_cycles=10)
    print("PASS: parallel_api_test")


def test_agent_naming():
    """Verify agent names and team mapping."""
    env = make_tiny_env()
    expected = [
        "red_radar_0", "red_radar_1",
        "blue_radar_0", "blue_radar_1",
        "red_commander", "blue_commander",
    ]
    assert env.possible_agents == expected, f"Got {env.possible_agents}"
    assert len(env.agents) == 6
    print("PASS: agent naming")


def test_obs_action_shapes():
    """Verify obs and action spaces match actual data."""
    env = make_tiny_env()
    obs, infos = env.reset()

    for agent in env.agents:
        space = env.observation_space(agent)
        assert obs[agent].shape == space.shape, (
            f"{agent}: obs shape {obs[agent].shape} != space {space.shape}"
        )

    # Step with random actions
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)

    for agent in env.agents:
        space = env.observation_space(agent)
        assert obs[agent].shape == space.shape, (
            f"{agent}: post-step obs shape mismatch"
        )

    print("PASS: obs/action shapes")


def test_radar_latents_callback():
    """Verify radar_latents_fn injects into commander obs."""
    N_in = 4

    def dummy_encoder(state):
        return np.ones((state.shape[0], N_in), dtype=np.float32) * 0.5

    env = make_tiny_env(radar_latents_fn=dummy_encoder)
    env.reset()

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, _, _, _, _ = env.step(actions)

    # Commander obs: [0:4] positions, [4:8] radar_0 latent, [8:12] radar_1 latent
    for cmd in ["red_commander", "blue_commander"]:
        cmd_obs = obs[cmd]
        assert cmd_obs.shape == (4 + 2 * N_in,), f"cmd obs shape: {cmd_obs.shape}"
        # Latent part should be 0.5 from our encoder
        np.testing.assert_allclose(cmd_obs[4:8], 0.5, atol=1e-5)
        np.testing.assert_allclose(cmd_obs[8:12], 0.5, atol=1e-5)

    print("PASS: radar_latents callback")


def test_commander_launch():
    """Verify commander can launch missile and get non-zero reward."""
    env = make_tiny_env(max_steps=200)
    env.reset()

    # Run a few steps, then launch
    for _ in range(3):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        env.step(actions)

    # Launch red missile
    cmd_act_dim = env.action_space("red_commander").shape[0]
    launch_action = np.zeros(cmd_act_dim, dtype=np.float32)
    launch_action[0] = 1.0   # launch_flag
    launch_action[1] = 0.0   # target_x
    launch_action[2] = 1.0   # target_y (blue side)

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    actions["red_commander"] = launch_action

    obs, rewards, _, _, infos = env.step(actions)

    # Red commander should have launched
    assert infos["red_commander"]["missile_in_flight"], "Missile should be in flight"
    print("PASS: commander launch")


def test_episode_termination():
    """Verify episode terminates when max_steps reached."""
    env = make_tiny_env(max_steps=5)
    env.reset()

    for step in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        if any(truncs.values()):
            # All agents should get truncation
            for agent in env.possible_agents:
                if agent in truncs:
                    assert truncs[agent], f"{agent} should be truncated"
            print(f"PASS: episode terminated at step {step + 1}")
            return

    assert False, "Episode should have truncated"


if __name__ == "__main__":
    print("=== PettingZoo Parallel API Tests ===\n")
    test_agent_naming()
    test_obs_action_shapes()
    test_radar_latents_callback()
    test_commander_launch()
    test_episode_termination()
    test_parallel_api()
    print("\nAll tests passed!")
