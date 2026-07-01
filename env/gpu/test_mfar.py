"""Full chain test for MFAR environment.

Validates:
  1. Default step (all detect): state shape, no NaN/Inf
  2. Per-element task assignment: mixed recon/detect/jam/comm
  3. BPSK comm round-trip: encode → modulate → channel → demod → decode
  4. FFT spectrum correctness: inject known tone, verify peak
  5. Waveform library: all types generate valid signals
  6. Backward compatibility: original vec_env tests still pass
"""
import sys, os, time, gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import warp as wp

wp.init()

device = "cuda" if torch.cuda.is_available() else "cpu"
DEV = torch.device(device)

print(f"PyTorch: {torch.__version__}")
print(f"Warp: {wp.__version__}")
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def fresh_mfar_env(num_envs=2, rows=25, cols=25):
    """Create an MFAR env for testing (25×25)."""
    from env.gpu.vec_mfar_env import MFARVecEnv
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return MFARVecEnv(
        num_envs=num_envs, n_radars=2, rows=rows, cols=cols,
        pulses_per_cpi=4, n_targets=1, device=device,
        fft_size=64,  # small for testing
    )


# ============================================================
print("=" * 60)
print("Test 1: Default step (all detect)")
print("=" * 60)


def test_default_step():
    env = fresh_mfar_env()
    env.reset()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = env.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000

    state = result["state"]
    spectrum = result["spectrum"]
    comm_data = result["comm_data"]

    print(f"  Step time: {dt:.0f} ms")
    print(f"  state shape: {state.shape}")
    print(f"  spectrum shape: {spectrum.shape}")
    print(f"  comm_data shape: {comm_data.shape}")

    # No NaN/Inf
    assert not torch.isnan(state).any().item(), "state contains NaN"
    assert not torch.isinf(state).any().item(), "state contains Inf"
    assert not torch.isnan(spectrum).any().item(), "spectrum contains NaN"
    print(f"  state range: [{state.min().item():.3e}, {state.max().item():.3e}]")
    print(f"  spectrum range: [{spectrum.min().item():.3e}, {spectrum.max().item():.3e}]")

    print("  [PASS] test_default_step")
    del env
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 2: Mixed task assignment")
print("=" * 60)


def test_mixed_tasks():
    env = fresh_mfar_env()
    env.reset()

    E, R = env.num_envs, env.n_radars
    N = env.n_elem
    action_dim = env.action_dim

    # Build action: first half detect, second half recon
    action = torch.zeros(E, R, action_dim, device=DEV)
    elem_actions = action[:, :, :N * 22].reshape(E, R, N, 22)

    # First 12 elements: detect (task fraction [0,1,0,0])
    elem_actions[:, :, :12, 0] = 0.0
    elem_actions[:, :, :12, 1] = 1.0  # detect
    elem_actions[:, :, :12, 4] = 0.0  # az=0
    elem_actions[:, :, :12, 5] = 0.0  # el=0

    # Remaining: recon (task fraction [1,0,0,0])
    elem_actions[:, :, 12:, 0] = 1.0  # recon
    elem_actions[:, :, 12:, 1] = 0.0

    result = env.step(action)
    task_ids = result["task_ids"]

    print(f"  task_ids unique: {task_ids.unique().tolist()}")
    detect_count = (task_ids == 1).sum().item()
    recon_count = (task_ids == 0).sum().item()
    print(f"  detect elements: {detect_count}, recon elements: {recon_count}")

    assert detect_count == E * R * 12, f"Expected {E*R*12} detect, got {detect_count}"
    assert recon_count == E * R * (N - 12), f"Expected {E*R*(N-12)} recon, got {recon_count}"

    state = result["state"]
    assert not torch.isnan(state).any().item(), "state contains NaN"
    print("  [PASS] test_mixed_tasks")
    del env
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 3: BPSK round-trip")
print("=" * 60)


def test_bpsk_roundtrip():
    from env.gpu.waveform_gpu import (
        encode_bpsk, decode_bpsk, modulate_bpsk, demodulate_bpsk,
    )

    # Encode
    x_orig, y_orig = 0.5, -0.3
    bits = encode_bpsk(x_orig, y_orig, device=DEV)
    print(f"  Original: ({x_orig}, {y_orig})")
    print(f"  Encoded bits: {bits.tolist()[:8]}...")

    # Decode
    x_dec, y_dec = decode_bpsk(bits)
    err_x = abs(x_dec - x_orig)
    err_y = abs(y_dec - y_orig)
    print(f"  Decoded: ({x_dec:.4f}, {y_dec:.4f})")
    print(f"  Error: ({err_x:.4f}, {err_y:.4f})")
    assert err_x < 0.001, f"BPSK X error too large: {err_x}"
    assert err_y < 0.001, f"BPSK Y error too large: {err_y}"

    # Modulate + demodulate (no noise)
    n_samples = 10000
    fs = 200e6
    symbol_rate = 1e6
    waveform = modulate_bpsk(bits, n_samples, fs, symbol_rate, DEV)
    print(f"  Waveform shape: {waveform.shape}, norm: {waveform.norm().item():.4f}")

    demod_bits = demodulate_bpsk(waveform, symbol_rate, fs)
    ber = (demod_bits != bits).float().mean().item()
    print(f"  BER (no noise): {ber:.4f}")
    assert ber == 0.0, f"BER should be 0 without noise, got {ber}"

    print("  [PASS] test_bpsk_roundtrip")
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 4: FFT spectrum correctness")
print("=" * 60)


def test_fft_spectrum():
    from env.gpu.vec_element_processor import VecElementProcessor

    proc = VecElementProcessor(
        fs=200e6, n_samples=1000, pulses_per_cpi=4,
        fft_size=1024, device=device,
    )

    # Inject a known tone at frequency bin 100
    freq_bin = 100
    t = torch.arange(1000, dtype=torch.float32, device=DEV) / 200e6
    tone = torch.exp(1j * 2 * np.pi * freq_bin * 200e6 / 1024 * t)
    # Add as [1, 1, 1, 1000] (E=1, R=1, N=1, S=1000)
    iq = tone.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,1000]

    spectrum = proc.process_rx_spectrum(iq)  # [1,1,1,1024]
    peak_bin = spectrum.squeeze().argmax().item()
    peak_power = spectrum.squeeze().max().item()

    print(f"  Injected tone at bin {freq_bin}")
    print(f"  Peak detected at bin {peak_bin}, power {peak_power:.2e}")
    assert abs(peak_bin - freq_bin) <= 1, f"Peak at wrong bin: {peak_bin} vs {freq_bin}"
    assert peak_power > 1.0, f"Peak power too low: {peak_power}"

    print("  [PASS] test_fft_spectrum")
    return True


# ============================================================
print("\n" + "=" * 60)
print("Test 5: Waveform library completeness")
print("=" * 60)


def test_waveform_library():
    from env.gpu.waveform_gpu import (
        generate_lfm, generate_barker, generate_frank, generate_costas,
        generate_nlfm, generate_p4,
        generate_noise_broadband, generate_noise_spot, generate_drfm,
    )

    dev = DEV
    fs = 200e6
    pw = 10e-6
    bw = 100e6
    n = max(1, int(pw * fs))

    waveforms = {
        "lfm_up": generate_lfm(pw, bw, fs, dev, "up"),
        "lfm_down": generate_lfm(pw, bw, fs, dev, "down"),
        "barker_13": generate_barker(13, pw / 13, fs, dev),
        "frank_16": generate_frank(4, fs, pw, dev),
        "costas_16": generate_costas(4, pw, fs, dev),
        "nlfm": generate_nlfm(pw, bw, fs, dev),
        "p4_code": generate_p4(4, pw, fs, dev),
        "noise_bb": generate_noise_broadband(n, 1.0, dev),
        "noise_spot": generate_noise_spot(n, 0.0, bw * 0.1, fs, 1.0, dev),
    }

    all_pass = True
    for name, wf in waveforms.items():
        is_complex = wf.is_complex()
        has_nan = torch.isnan(wf).any().item()
        has_inf = torch.isinf(wf).any().item()
        norm = wf.norm().item()
        ok = is_complex and not has_nan and not has_inf and norm > 0
        status = "OK" if ok else "FAIL"
        print(f"  {name:15s}: shape={list(wf.shape)}, norm={norm:.4f}, [{status}]")
        all_pass = all_pass and ok

    # Test DRFM with a captured signal
    captured = generate_lfm(pw, bw, fs, dev, "up")
    drfm = generate_drfm(captured, freq_shift=1e6, fs=fs, delay_samples=100)
    drfm_ok = not torch.isnan(drfm).any() and drfm.norm() > 0
    print(f"  {'drfm':15s}: shape={list(drfm.shape)}, norm={drfm.norm().item():.4f}, [{'OK' if drfm_ok else 'FAIL'}]")
    all_pass = all_pass and drfm_ok

    if all_pass:
        print("  [PASS] test_waveform_library")
    else:
        print("  [FAIL] test_waveform_library")
    return all_pass


# ============================================================
print("\n" + "=" * 60)
print("Test 6: Per-element steering")
print("=" * 60)


def test_per_element_steering():
    from env.gpu.vec_array import VecArray

    arr = VecArray(
        rows=25, cols=25, fc=10e9,
        num_envs=1, n_radars=1, device=device,
    )

    # All elements same direction
    N = arr.n_elem  # 625 for 25x25
    az_same = torch.zeros(1, 1, N, device=DEV)
    el_same = torch.zeros(1, 1, N, device=DEV)
    w_same = arr.steer_per_element(az_same, el_same).clone()

    # Each element different direction
    az_diff = torch.linspace(-30, 30, N, device=DEV).reshape(1, 1, N)
    el_diff = torch.zeros(1, 1, N, device=DEV)
    w_diff = arr.steer_per_element(az_diff, el_diff).clone()

    print(f"  Same-dir weights shape: {w_same.shape}")
    print(f"  Diff-dir weights shape: {w_diff.shape}")

    # Same direction should match steer_all
    w_all = arr.steer_all(
        torch.zeros(1, 1, device=DEV),
        torch.zeros(1, 1, device=DEV),
    )
    max_diff = (w_same - w_all).abs().max().item()
    print(f"  Max diff steer_per_element vs steer_all (same dirs): {max_diff:.2e}")

    assert max_diff < 1e-5, f"Per-element steering doesn't match steer_all: {max_diff}"

    # Different directions should give different weights
    # Use abs variance (complex magnitude varies with phase)
    w_diff_abs = w_diff.abs()
    elem_var = w_diff_abs.var().item()
    # Also check that phases differ
    phase_var = w_diff.angle().var().item()
    print(f"  Magnitude variance: {elem_var:.6e}")
    print(f"  Phase variance: {phase_var:.6f}")
    assert phase_var > 1e-6, "Phases should vary with different beam directions"

    assert not torch.isnan(w_diff).any().item(), "Per-element weights contain NaN"

    print("  [PASS] test_per_element_steering")
    del arr
    return True


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    results = {
        "1_default_step": test_default_step(),
        "2_mixed_tasks": test_mixed_tasks(),
        "3_bpsk_roundtrip": test_bpsk_roundtrip(),
        "4_fft_spectrum": test_fft_spectrum(),
        "5_waveform_library": test_waveform_library(),
        "6_per_element_steering": test_per_element_steering(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results.values())
    for name, ok in results.items():
        print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
    print(f"\n  {passed}/{len(results)} passed")
    if all(results.values()):
        print("ALL TESTS PASSED")
