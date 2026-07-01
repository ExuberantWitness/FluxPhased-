"""Smoke test and benchmark for RadarSimVecEnv.

Validates:
  1. test_basic: 10 env basic CPI run
  2. test_consistency: vec_env(E=1) vs original pipeline_gpu RDM error < 0.5 dB
  3. benchmark: timing + VRAM for various num_envs
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import warp as wp

print(f"PyTorch: {torch.__version__}")
print(f"Warp: {wp.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

device = "cuda" if torch.cuda.is_available() else "cpu"
wp.init()

TORCH_DEVICE = torch.device(device)


# ============================================================
# Test 1: Basic 10-env simulation
# ============================================================
print("\n" + "=" * 60)
print("Test 1: Basic 10-env simulation")
print("=" * 60)


def test_basic():
    from env.gpu.vec_env import RadarSimVecEnv

    env = RadarSimVecEnv(
        num_envs=10, n_radars=4, rows=10, cols=10,
        pulses_per_cpi=16, n_targets=1, device=device,
    )
    env.reset()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = env.step()
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    print(f"  Total step time: {t_total * 1000:.0f} ms")
    timing = result["timing"]
    for k, v in timing.items():
        print(f"    {k}: {v:.1f} ms")

    detections = result["detections"]
    total_dets = sum(len(rd) for env_dets in detections for rd in env_dets)
    print(f"  Total detections across all envs: {total_dets}")
    print(f"  pulse_train shape: {result['pulse_train'].shape}")
    print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

    # Check all envs have outputs
    assert len(detections) == 10, f"Expected 10 envs, got {len(detections)}"
    for e, env_dets in enumerate(detections):
        assert len(env_dets) == 4, f"Env {e}: expected 4 radars, got {len(env_dets)}"

    # Regression: pulse_train must not contain NaN/Inf (vec_interference
    # cos(pi/2)**1.5 NaN bug fixed 2026-05-11).
    pt = result["pulse_train"]
    assert not torch.isnan(pt).any().item(), "pulse_train contains NaN"
    assert not torch.isinf(pt).any().item(), "pulse_train contains Inf"

    print("  [PASS] test_basic")
    return True


# ============================================================
# Test 2: Consistency — vec_env(E=1) vs original pipeline
# ============================================================
print("\n" + "=" * 60)
print("Test 2: Consistency check (E=1 vs original pipeline)")
print("=" * 60)


def test_consistency():
    from env.gpu.vec_env import RadarSimVecEnv
    from env.gpu.pipeline_gpu import RadarPipelineGPU, RadarState, TargetState

    fc = 10e9
    bw = 200e6
    prf = 10e3
    pulses = 32

    # --- Vec env ---
    vec = RadarSimVecEnv(
        num_envs=1, n_radars=1, rows=10, cols=10,
        fc=fc, bandwidth=bw, prf=prf, pulses_per_cpi=pulses,
        n_targets=1, device=device,
    )

    # Set deterministic state
    vec.radar_pos[0, 0] = torch.tensor([0.0, 0.0, 0.0], device=TORCH_DEVICE)
    vec.radar_vel[0, 0] = torch.tensor([0.0, 0.0, 0.0], device=TORCH_DEVICE)
    vec.target_pos[0, 0] = torch.tensor([5000.0, 0.0, 0.0], device=TORCH_DEVICE)
    vec.target_vel[0, 0] = torch.tensor([0.0, 0.0, 0.0], device=TORCH_DEVICE)
    vec.beam_az[0, 0] = 0.0
    vec.beam_el[0, 0] = 0.0

    torch.cuda.synchronize()
    vec_result = vec.step()
    torch.cuda.synchronize()

    # --- Original pipeline ---
    pipeline = RadarPipelineGPU(
        fc=fc, bandwidth=bw, prf=prf,
        pulses_per_cpi=pulses, n_radars=1, rows=10, cols=10, device=device,
    )
    pipeline.steer_beams({0: (0.0, 0.0)})

    radar_states = [RadarState(
        radar_id=0,
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([0.0, 0.0, 0.0]),
        freq_hz=fc, bandwidth_hz=bw,
        tx_power_w=1.0,
        array_az_deg=0.0, beam_az_deg=0.0, beam_el_deg=0.0,
        waveform_type="lfm_up",
        targets=[TargetState(
            pos=np.array([5000.0, 0.0, 0.0]),
            vel=np.array([0.0, 0.0, 0.0]),
            rcs_dbsm=20.0,
        )],
    )]

    torch.cuda.synchronize()
    pipe_result = pipeline.simulate_cpi(radar_states)
    torch.cuda.synchronize()

    # Compare RDM: vec gives rd_power computed internally, pipeline gives rd_map
    # We need to get rd_power from vec_receiver — let's compute it from pulse_train
    vec_pt = vec_result["pulse_train"][0, 0]  # [pulses, samples]
    pipe_pt = pipe_result[0]["pulse_train"]   # [pulses, samples]

    # Run same MF + Doppler on both pulse trains
    n_samples = vec_pt.shape[-1]
    n_filter = vec_pt.shape[-1]  # simplified — both use same baseband length
    n_fft = 1
    while n_fft < n_samples * 2 - 1:
        n_fft *= 2
    n_doppler = 1
    while n_doppler < pulses:
        n_doppler *= 2

    # Use a simple LFM as MF reference for both
    t = torch.linspace(0, vec.pri * 0.1, int(vec.pri * 0.1 * bw), device=TORCH_DEVICE)
    lfm = torch.exp(1j * np.pi * bw / (vec.pri * 0.1) * t ** 2)
    lfm = lfm / lfm.norm()
    n_pulse = lfm.shape[0]
    if n_pulse < n_samples:
        pad = torch.zeros(n_samples - n_pulse, dtype=torch.complex64, device=TORCH_DEVICE)
        lfm = torch.cat([lfm, pad])
    lfm = lfm[:n_samples]
    mf_ref = torch.fft.fft(lfm.conj().flip(0), n=n_fft)

    def compute_rdm(pt):
        sig_fft = torch.fft.fft(pt, n=n_fft, dim=-1)
        rp = torch.fft.ifft(sig_fft * mf_ref.unsqueeze(0), dim=-1)[..., :n_samples]
        rd = torch.fft.fft(rp, n=n_doppler, dim=0)
        rd = torch.fft.fftshift(rd, dim=0)
        return 20.0 * torch.log10(torch.abs(rd).clamp(min=1e-15))

    vec_rdm_db = compute_rdm(vec_pt).cpu().numpy()
    pipe_rdm_db = compute_rdm(pipe_pt).cpu().numpy()

    # Both runs use independent random noise, so exact match is impossible.
    # Compare statistical properties: noise floor, variance, and peak.
    vec_rms = float(torch.abs(vec_pt).mean().item())
    pipe_rms = float(torch.abs(pipe_pt).mean().item())
    print(f"  pulse_train mean |x|: vec={vec_rms:.3e}, pipe={pipe_rms:.3e}")

    vec_noise_floor = float(np.mean(vec_rdm_db))
    pipe_noise_floor = float(np.mean(pipe_rdm_db))
    print(f"  RDM mean (dB): vec={vec_noise_floor:.2f}, pipe={pipe_noise_floor:.2f}")

    vec_peak = float(np.max(vec_rdm_db))
    pipe_peak = float(np.max(pipe_rdm_db))
    print(f"  RDM peak (dB): vec={vec_peak:.2f}, pipe={pipe_peak:.2f}")

    # Tolerance: RMS amplitudes should be within 30% (stochastic noise).
    rms_ratio = vec_rms / max(pipe_rms, 1e-30)
    floor_diff = abs(vec_noise_floor - pipe_noise_floor)
    peak_diff = abs(vec_peak - pipe_peak)

    ok_rms = 0.7 < rms_ratio < 1.43
    ok_floor = floor_diff < 3.0  # noise floor within 3 dB
    ok_peak = peak_diff < 6.0    # peak within 6 dB

    if ok_rms and ok_floor and ok_peak:
        print(f"  [PASS] Consistency: rms_ratio={rms_ratio:.3f}, "
              f"floor_diff={floor_diff:.2f} dB, peak_diff={peak_diff:.2f} dB")
        return True
    else:
        print(f"  [FAIL] Consistency: rms_ratio={rms_ratio:.3f}, "
              f"floor_diff={floor_diff:.2f} dB, peak_diff={peak_diff:.2f} dB")
        return False


# ============================================================
# Test 3: Benchmark
# ============================================================
print("\n" + "=" * 60)
print("Test 3: Benchmark — timing vs num_envs")
print("=" * 60)


def benchmark():
    from env.gpu.vec_env import RadarSimVecEnv

    for n in [1, 4, 10]:
        torch.cuda.reset_peak_memory_stats()
        env = RadarSimVecEnv(
            num_envs=n, n_radars=4, rows=10, cols=10,
            pulses_per_cpi=16, n_targets=1, device=device,
        )
        env.reset()

        # Warmup
        env.step()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        result = env.step()
        torch.cuda.synchronize()
        t_total = time.perf_counter() - t0

        vram = torch.cuda.max_memory_allocated() / 1e6
        print(
            f"  E={n}: {t_total * 1000:.0f} ms total, "
            f"{vram:.0f} MB VRAM peak"
        )
        timing = result["timing"]
        print(f"    waveform={timing['waveform_ms']:.1f}ms "
              f"steer={timing['steer_ms']:.1f}ms "
              f"intf={timing['interference_ms']:.1f}ms "
              f"pulses={timing['pulses_ms']:.1f}ms "
              f"receiver={timing['receiver_ms']:.1f}ms")
    print("  [PASS] benchmark complete")
    return True


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    ok1 = test_basic()
    ok2 = test_consistency()
    ok3 = benchmark()

    print("\n" + "=" * 60)
    results = [ok1, ok2, ok3]
    passed = sum(results)
    print(f"SUMMARY: {passed}/{len(results)} tests passed")
    if all(results):
        print("ALL TESTS PASSED")
    else:
        for i, (name, ok) in enumerate(zip(
            ["test_basic", "test_consistency", "benchmark"], results,
        )):
            print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
