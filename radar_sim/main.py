#!/usr/bin/env python3
"""Multi-Agent Phased Array EW Simulation Demo.

Runs the PettingZoo ParallelEnv with random agent policies to verify:
- 4 × 25×25 phased array physics (beam steering, array factor, SNR)
- 6-agent PettingZoo interface (2 radar + 1 commander per team)
- Battlefield dynamics (vehicle movement, array rotation, artillery)
- Mutual interference computation
- Detection and communication models

Usage:
    python -m radar_sim.main [--steps N] [--seed S] [--render]
"""

import argparse
import time
import numpy as np

from .config import EnvConfig, default_config
from .env.pettingzoo_env import PhasedArrayEWEnv


def random_radar_policy(rng: np.random.Generator) -> np.ndarray:
    """Random radar agent action (14-dim)."""
    action = np.zeros(14, dtype=np.float32)
    action[0] = rng.uniform(0, 1)       # freq channel
    action[1] = rng.uniform(0.2, 0.8)   # pri fraction
    action[2] = rng.uniform(0.1, 0.5)   # pulse_width fraction
    action[3] = rng.uniform(0, 1)       # beam_az norm
    action[4] = rng.uniform(0.3, 0.7)   # beam_el norm
    action[5] = rng.uniform(-0.3, 0.3)  # array rotation rate
    action[6] = rng.uniform(0.3, 1.0)   # power fraction
    action[7] = rng.integers(0, 8)      # waveform type
    action[8] = rng.integers(0, 16)     # code scheme
    action[9] = rng.integers(0, 4)      # function mode
    action[10] = rng.uniform(0, 1)      # speed
    action[11] = rng.uniform(-0.5, 0.5) # heading change
    action[12] = rng.uniform(0, 1)      # beam_az_2
    action[13] = rng.integers(0, 2)     # function mode 2
    return action


def random_commander_policy(rng: np.random.Generator) -> np.ndarray:
    """Random commander action (3-dim)."""
    action = np.zeros(3, dtype=np.float32)
    action[0] = rng.uniform(0, 1)       # fire flag
    action[1] = rng.uniform(0, 1)       # target_x norm
    action[2] = rng.uniform(0, 1)       # target_y norm
    return action


def run_demo(config: EnvConfig = None, num_steps: int = 500, seed: int = 42,
             render: bool = False):
    """Run the simulation demo."""
    if config is None:
        config = default_config

    print("=" * 70)
    print("  Multi-Agent Phased Array EW Simulation Demo")
    print("  4 × 25×25 Phased Arrays | 6 Agents | Red vs Blue")
    print("=" * 70)
    print(f"  Steps: {num_steps}  |  Seed: {seed}")
    print(f"  CPI: {config.cpi.cpi_duration*1000:.0f} ms  |  PRF: {config.cpi.prf/1000:.1f} kHz")
    print(f"  Array: {config.array.rows}×{config.array.cols} ({config.array.num_elements} elements)")
    print(f"  Band: {config.rf.fc/1e9:.1f} GHz  |  BW: {config.rf.bandwidth/1e6:.0f} MHz")
    print(f"  Map: {config.battlefield.map_size[0]/1000:.0f}×{config.battlefield.map_size[1]/1000:.0f} km")
    print(f"  Artillery: {config.artillery.flight_time}s delay | {config.artillery.kill_radius_min}-{config.artillery.kill_radius_max}m kill radius")
    print("=" * 70)

    # Create environment
    env = PhasedArrayEWEnv(config)
    rng = np.random.default_rng(seed)

    # Reset
    obs, infos = env.reset(seed=seed)
    print(f"\n  Agents: {env.agents}")
    for aid in env.agents:
        atype = env._get_agent_type(aid)
        ospace = env.observation_space(aid)
        aspace = env.action_space(aid)
        print(f"    {aid}: obs={ospace.shape} act={aspace.shape} type={atype}")

    # Tracking
    total_rewards = {aid: 0.0 for aid in env.possible_agents}
    detection_counts = {aid: 0 for aid in env.possible_agents if "commander" not in aid}
    artillery_shots = {"red": 0, "blue": 0}
    kills = {"red": 0, "blue": 0}

    t_start = time.perf_counter()

    # Main loop
    for step in range(num_steps):
        actions = {}
        for aid in env.agents:
            if env._get_agent_type(aid) == "radar":
                actions[aid] = random_radar_policy(rng)
            else:
                actions[aid] = random_commander_policy(rng)

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Accumulate stats
        for aid in env.agents:
            total_rewards[aid] += rewards.get(aid, 0.0)
            if "commander" not in aid:
                dets = infos.get(aid, {}).get("detections", [])
                if dets:
                    detection_counts[aid] += len(dets)

        # Check for artillery events
        for team in ["red", "blue"]:
            arty_info = infos.get(f"{team}_commander", {})
            if arty_info.get("shells_in_flight", 0) > 0:
                pass  # tracked in step

        # Check game over
        if any(terminations.values()):
            winner = env.battlefield.check_game_over()
            print(f"\n  *** Game Over at step {step+1}: {winner} wins! ***")
            break

        # Print progress
        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            steps_per_sec = (step + 1) / elapsed
            print(f"  Step {step+1:5d}/{num_steps} | {steps_per_sec:.1f} steps/s | "
                  f"Red: {total_rewards['red_radar_0']:.2f}/{total_rewards['red_radar_1']:.2f} "
                  f"Blue: {total_rewards['blue_radar_0']:.2f}/{total_rewards['blue_radar_1']:.2f}")

        # Optional render
        if render and step % 10 == 0:
            state = env.render()
            if state:
                _print_minimap(state)

    elapsed = time.perf_counter() - t_start

    # Final summary
    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Total steps: {num_steps}")
    print(f"  Wall time: {elapsed:.1f}s ({num_steps/elapsed:.1f} steps/s)")
    print(f"\n  Final Rewards:")
    for aid in env.possible_agents:
        print(f"    {aid}: {total_rewards[aid]:.3f}")
    print(f"\n  Detections per radar:")
    for aid, count in detection_counts.items():
        print(f"    {aid}: {count} total ({count/max(num_steps,1)*100:.1f} per 100 steps)")
    print(f"\n  Battlefield State:")
    state = env.render()
    if state:
        for aid, v in state["vehicles"].items():
            status = "ALIVE" if v["alive"] else "DEAD"
            print(f"    {aid}: ({v['x']:.0f}, {v['y']:.0f}) "
                  f"hdg={v['heading']:.0f}° arr={v['array_bearing']:.0f}° {status}")

    env.close()
    return total_rewards, state


def _print_minimap(state: dict):
    """Simple ASCII minimap render."""
    if not state:
        return
    vehicles = state["vehicles"]
    # 40-char wide ASCII minimap
    print("  ", "-" * 42)
    for aid, v in vehicles.items():
        if v["alive"]:
            team = "R" if "red" in aid else "B"
            marker = ">" if "radar" in aid else "C"
            print(f"  {team}{marker} {aid}: ({v['x']:.0f}, {v['y']:.0f})")
    print("  ", "-" * 42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phased Array EW Simulation Demo")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Enable ASCII rendering")
    args = parser.parse_args()

    run_demo(num_steps=args.steps, seed=args.seed, render=args.render)
