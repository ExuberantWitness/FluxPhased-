"""Baseline performance benchmark for FluxPhased GPU simulation.

Records VRAM usage, per-phase timing, and numerical fingerprints
under the 25x25 array configuration.

Usage:
    python baselines/benchmark_25x25.py
"""

import sys
import os
import gc
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV = torch.device(DEVICE)


def benchmark():
    print("=" * 60)
    print("FluxPhased GPU Performance Benchmark (25x25 baseline)")
    print("=" * 60)

    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram_total:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {DEVICE}")

    from env.gpu.vec_mfar_env import MFARVecEnv

    # --- Config ---
    config = {
        "num_envs": 2,
        "n_radars": 2,
        "rows": 25,
        "cols": 25,
        "pulses_per_cpi": 4,
        "n_targets": 1,
        "fft_size": 64,
        "device": DEVICE,
    }

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # --- Create env ---
    t_init_start = time.perf_counter()
    env = MFARVecEnv(**config)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_init = time.perf_counter() - t_init_start

    vram_after_init = 0
    if DEVICE == "cuda":
        vram_after_init = torch.cuda.memory_allocated() / 1e6  # MB

    print(f"\nInit time: {t_init * 1000:.0f} ms")
    print(f"VRAM after init: {vram_after_init:.1f} MB")

    # --- Reset ---
    torch.manual_seed(42)
    env.reset()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # --- Warmup step ---
    env.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # --- Measured steps ---
    N_STEPS = 5
    timing_sums = {
        "action_ms": 0, "tx_ms": 0, "interference_ms": 0,
        "pulses_ms": 0, "rx_ms": 0, "missile_ms": 0, "total_ms": 0,
    }
    vram_peaks = []
    vram_after_steps = []

    for step_i in range(N_STEPS):
        torch.manual_seed(42 + step_i)
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        result = env.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        t = result["timing"]
        for k in timing_sums:
            timing_sums[k] += t[k]

        if DEVICE == "cuda":
            vram_peaks.append(torch.cuda.max_memory_allocated() / 1e6)
            vram_after_steps.append(torch.cuda.memory_allocated() / 1e6)

    # --- Averages ---
    avg_timing = {k: v / N_STEPS for k, v in timing_sums.items()}
    avg_vram_peak = np.mean(vram_peaks)
    avg_vram_after = np.mean(vram_after_steps)
    delta_vram = np.mean(np.diff([vram_after_init] + vram_after_steps))

    print(f"\n--- Timing (avg over {N_STEPS} steps) ---")
    for k, v in avg_timing.items():
        print(f"  {k}: {v:.1f} ms")

    print(f"\n--- VRAM ---")
    print(f"  After init: {vram_after_init:.1f} MB")
    print(f"  Avg after step: {avg_vram_after:.1f} MB")
    print(f"  Avg peak per step: {avg_vram_peak:.1f} MB")
    print(f"  Delta per step: {delta_vram:.1f} MB")

    # --- Numerical fingerprint ---
    torch.manual_seed(100)
    env_fresh = MFARVecEnv(**config)
    env_fresh.reset()
    torch.manual_seed(100)
    result = env_fresh.step()

    spectrum = result["spectrum"]
    comm_data = result["comm_data"]
    state = result["state"]
    channel = result.get("channel_params", {})

    def to_list(t):
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy().round(6).tolist()
        return t

    fp = {
        "spectrum_shape": list(spectrum.shape),
        "spectrum_sample": to_list(spectrum[0, 0, :3, :2, :8]),
        "spectrum_range": [
            float(spectrum.min().item()),
            float(spectrum.max().item()),
        ],
        "comm_data_shape": list(comm_data.shape),
        "comm_data_sample": to_list(comm_data[0, 0, :3, :]),
        "state_shape": list(state.shape),
        "state_range": [
            float(state.min().item()),
            float(state.max().item()),
        ],
        "delay_sample": to_list(channel.get("delay_samples")),
        "doppler_hz": to_list(channel.get("doppler_hz")),
        "gain_linear": to_list(channel.get("gain_linear")),
    }

    print(f"\n--- Numerical Fingerprint ---")
    print(f"  spectrum shape: {fp['spectrum_shape']}")
    print(f"  spectrum range: [{fp['spectrum_range'][0]:.3e}, {fp['spectrum_range'][1]:.3e}]")
    print(f"  state shape: {fp['state_shape']}")
    print(f"  state range: [{fp['state_range'][0]:.3e}, {fp['state_range'][1]:.3e}]")

    # --- Save JSON ---
    output = {
        "config": config,
        "device_info": {
            "gpu": gpu_name if DEVICE == "cuda" else "cpu",
            "pytorch_version": torch.__version__,
        },
        "vram": {
            "after_init_mb": round(vram_after_init, 1),
            "avg_peak_step_mb": round(avg_vram_peak, 1),
            "avg_after_step_mb": round(avg_vram_after, 1),
            "delta_step_mb": round(delta_vram, 1),
        },
        "timing_ms": {k: round(v, 2) for k, v in avg_timing.items()},
        "numerical": fp,
    }

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_25x25.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")

    del env, env_fresh
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return output


if __name__ == "__main__":
    benchmark()
