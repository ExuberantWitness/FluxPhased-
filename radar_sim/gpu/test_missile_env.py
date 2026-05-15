"""Test suite for cruise missile combat in MFAR environment.

Uses small array (5x5=25 elements, 8 pulses) to fit on 6GB GPU.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np


def test_missile_physics():
    """Test 1: launch, straight-line flight, kill check."""
    from radar_sim.gpu.vec_missile import VecMissile

    m = VecMissile(num_envs=2, n_teams=2, speed_ms=244.4, kill_radius_m=500.0, device="cuda")
    m.reset()

    env_ids = torch.tensor([0, 1])
    start = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 10000.0, 0.0], [0.0, 5000.0, 0.0]])
    m.launch(env_ids, 0, start, target)

    assert m.in_flight[0, 0].item() == True
    assert m.in_flight[0, 1].item() == False  # team 1 not launched

    m.step(1.0)
    dy = m.missile_pos[0, 0, 1].item()
    assert 200 < dy < 300, f"Expected ~244m, got {dy}"

    # Enemy at 5000m — missile only traveled ~244m, well outside 500m kill radius
    enemy_pos = torch.tensor([[[0.0, 5000.0, 0.0]]]).expand(2, 1, 3).cuda()
    kills = m.check_kill(enemy_pos)
    assert kills[0, 0, 0].item() == False

    print("[PASS] test_missile_physics")
    return True


def test_kill_radius():
    """Test 2: kill within 500m, no kill beyond."""
    from radar_sim.gpu.vec_missile import VecMissile

    m = VecMissile(num_envs=2, n_teams=2, speed_ms=244.4, kill_radius_m=500.0, device="cuda")
    m.reset()

    env_ids = torch.tensor([0, 1])
    start = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 10000.0, 0.0], [0.0, 10000.0, 0.0]])
    m.launch(env_ids, 0, start, target)
    m.missile_pos[0, 0] = torch.tensor([0.0, 1000.0, 0.0])
    m.missile_pos[1, 0] = torch.tensor([0.0, 1000.0, 0.0])

    # Missile at (0, 1000), enemy at (0, 2000) — distance 1000m, no kill
    enemy_far = torch.tensor([[[0.0, 2000.0, 0.0]]]).expand(2, 1, 3).cuda()
    kills = m.check_kill(enemy_far)
    assert kills[0, 0, 0].item() == False

    # Missile at (0, 1000), enemy at (0, 1200) — distance 200m, kill!
    enemy_near = torch.tensor([[[0.0, 1200.0, 0.0]]]).expand(2, 1, 3).cuda()
    kills = m.check_kill(enemy_near)
    assert kills[0, 0, 0].item() == True

    print("[PASS] test_kill_radius")
    return True


def test_course_correction():
    """Test 3: update target mid-flight → missile changes direction."""
    from radar_sim.gpu.vec_missile import VecMissile

    m = VecMissile(num_envs=1, n_teams=2, speed_ms=244.4, device="cuda")
    m.reset()

    env_ids = torch.tensor([0])
    start = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 10000.0, 0.0]])
    m.launch(env_ids, 0, start, target)

    m.step(0.001)
    assert m.missile_vel[0, 0, 1].item() > 0

    new_target = torch.tensor([[10000.0, 0.0, 0.0]])
    m.update_target(env_ids, 0, new_target)

    m.step(0.001)
    vx = m.missile_vel[0, 0, 0].item()
    assert vx > 100, f"Expected vx > 100 after course correction, got {vx}"

    print("[PASS] test_course_correction")
    return True


def test_bpsk_batch_roundtrip():
    """Test 4: batch BPSK encode → modulate → demod_batch → decode_batch."""
    from radar_sim.gpu.waveform_gpu import (
        encode_bpsk, modulate_bpsk,
        demodulate_bpsk_batch, decode_bpsk_batch,
    )

    device = "cuda"
    fs = 200e6
    symbol_rate = 1e6
    n_samples = 10000
    n_env = 4

    signals = []
    original_xy = []
    for i in range(n_env):
        x = np.clip(i / (n_env - 1) * 2.0 - 1.0, -1, 1) if n_env > 1 else 0.5
        y = np.clip((n_env - 1 - i) / (n_env - 1) * 2.0 - 1.0, -1, 1) if n_env > 1 else -0.5
        original_xy.append((x, y))
        bits = encode_bpsk(float(x), float(y), device=device)
        sig = modulate_bpsk(bits, n_samples, fs, symbol_rate, device)
        signals.append(sig)

    batch = torch.stack(signals)
    bits_batch = demodulate_bpsk_batch(batch, symbol_rate, fs, n_bits=32)
    data_x, data_y, crc_ok = decode_bpsk_batch(bits_batch)

    for i in range(n_env):
        ox, oy = original_xy[i]
        assert crc_ok[i].item(), f"CRC failed for env {i}"
        assert abs(data_x[i].item() - ox) < 0.01, f"X error for env {i}"
        assert abs(data_y[i].item() - oy) < 0.01, f"Y error for env {i}"

    print("[PASS] test_bpsk_batch_roundtrip")
    return True


def test_env_step_with_commander():
    """Test 5: full env step with commander actions launches missile.

    Uses small config: 5x5=25 elements, 8 pulses to fit GPU.
    """
    from radar_sim.gpu.vec_mfar_env import MFARVecEnv

    N_in, N_out = 8, 4  # small latent dims for test
    env = MFARVecEnv(
        num_envs=1, n_radars=4, rows=5, cols=5,
        pulses_per_cpi=8, bandwidth=10e6, prf=10e3,
        num_input_length=N_in, num_output_length=N_out,
        device="cuda",
    )
    env.reset()

    actions = torch.zeros(1, 4, env.action_dim, device="cuda")

    # Commander action: [launch_flag, target_x, target_y, inst_0..., inst_1...]
    cmd_dim = 3 + 2 * N_out  # = 11
    commander_actions = torch.zeros(1, 2, cmd_dim, device="cuda")
    commander_actions[:, 0, 0] = 1.0  # Red launch
    commander_actions[:, 0, 1] = 0.0  # target x=0
    commander_actions[:, 0, 2] = 0.5  # target y=+5000
    commander_actions[:, 1, 0] = 1.0  # Blue launch
    commander_actions[:, 1, 2] = -0.5

    # Provide radar latents for commander obs
    radar_latents = torch.randn(1, 4, N_in, device="cuda")

    result = env.step(actions, commander_actions=commander_actions,
                      radar_latents=radar_latents)

    assert result["missile_pos"].shape == (1, 2, 3)
    assert result["dones"].shape == (1,)
    assert result["commander_obs"].shape == (1, 2, 4 + 2 * N_in), \
        f"Expected (1, 2, {4 + 2 * N_in}), got {result['commander_obs'].shape}"
    assert result["radar_instructions"].shape == (1, 4, N_out), \
        f"Expected (1, 4, {N_out}), got {result['radar_instructions'].shape}"
    assert result["radar_rewards"].shape == (1, 4)
    assert result["commander_rewards"].shape == (1, 2)
    assert result["kills"].shape == (1, 2, 2)

    assert env.battlefield.missile.in_flight[:, 0].all()
    assert env.battlefield.missile.in_flight[:, 1].all()

    # Verify instructions were extracted correctly
    inst = result["radar_instructions"]
    # Red radar 0 instruction should match commander_actions[:, 0, 3:3+N_out]
    assert torch.allclose(inst[0, 0], commander_actions[0, 0, 3:3 + N_out])

    print("[PASS] test_env_step_with_commander")
    return True


def test_win_condition():
    """Test 6: missile kills enemy radar → episode terminates."""
    from radar_sim.gpu.vec_mfar_env import MFARVecEnv

    env = MFARVecEnv(
        num_envs=1, n_radars=4, rows=5, cols=5,
        pulses_per_cpi=8, bandwidth=10e6, prf=10e3,
        device="cuda",
    )
    env.reset()

    # Place red missile near blue radar
    m = env.battlefield.missile
    m.in_flight[0, 0] = True
    m.launched[0, 0] = True
    m.missile_pos[0, 0] = torch.tensor([0.0, 5000.0, 0.0])
    m.target_pos[0, 0] = torch.tensor([0.0, 6000.0, 0.0])

    # Blue radar at (0, 5200, 0) — 200m away, within 500m kill radius
    env.radar_pos[0, 2] = torch.tensor([0.0, 5200.0, 0.0])

    actions = torch.zeros(1, 4, env.action_dim, device="cuda")
    result = env.step(actions)

    assert result["dones"][0].item() == True, "Episode should end"
    assert result["winners"][0].item() == 0, "Red team (0) should win"

    print("[PASS] test_win_condition")
    return True


def test_backward_compat():
    """Test 7: step without commander_actions works like before."""
    from radar_sim.gpu.vec_mfar_env import MFARVecEnv

    env = MFARVecEnv(
        num_envs=1, n_radars=4, rows=5, cols=5,
        pulses_per_cpi=8, bandwidth=10e6, prf=10e3,
        device="cuda",
    )
    env.reset()

    result = env.step()

    assert result["state"].shape[0] == 1
    assert result["state"].shape[1] == 4
    assert not torch.isnan(result["state"]).any()
    assert not torch.isinf(result["state"]).any()
    assert not result["dones"].any(), "No kills expected without missiles"
    assert result["commander_obs"].shape[2] == env.battlefield.commander_obs_dim
    assert result["radar_instructions"].shape[2] == env.num_output_length

    print("[PASS] test_backward_compat")
    return True


def test_state_dim():
    """Test 8: state_dim includes missile awareness + commander instruction."""
    from radar_sim.gpu.vec_mfar_env import MFARVecEnv

    N_out = 8
    env = MFARVecEnv(
        num_envs=1, n_radars=4, rows=5, cols=5,
        pulses_per_cpi=8, bandwidth=10e6, prf=10e3,
        num_output_length=N_out,
        device="cuda",
    )

    N = env.n_elem
    P = env.n_pulses
    B = env.n_bins
    expected = N * (P * B + 2 + 4) + 5 + 6 + env.n_teams * 3 + N_out
    assert env.state_dim == expected, f"Expected {expected}, got {env.state_dim}"

    env.reset()
    result = env.step()
    assert result["state"].shape[2] == expected

    print("[PASS] test_state_dim")
    return True


if __name__ == "__main__":
    torch.cuda.synchronize()

    tests = [
        test_missile_physics,
        test_kill_radius,
        test_course_correction,
        test_bpsk_batch_roundtrip,
        test_env_step_with_commander,
        test_win_condition,
        test_backward_compat,
        test_state_dim,
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

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
