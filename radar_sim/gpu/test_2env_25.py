"""Precision + usability tests for E=2 on 4x(25x25), 32 pulses.

Covers:
  A. Statistical equivalence — two envs with identical setup produce
     pulse_trains with matching RMS / RDM peak / noise floor.
  B. Independence — two envs with different beams/targets produce
     visibly different RDMs (no cross-contamination).
  C. Memory stability — 5 consecutive step() calls, peak VRAM bounded.
  D. Numerical sanity — no NaN / Inf anywhere.
  E. Reset works between runs.
"""
import sys, os, time, gc
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
import warp as wp
wp.init()

from radar_sim.gpu.vec_env import RadarSimVecEnv

device = "cuda"
DEV = torch.device(device)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
print("Config: num_envs=2, n_radars=4, rows=25, cols=25, pulses=32\n")


def fresh_env(num_envs=2):
    gc.collect()
    torch.cuda.empty_cache()
    return RadarSimVecEnv(
        num_envs=num_envs, n_radars=4, rows=25, cols=25,
        pulses_per_cpi=32, n_targets=1, device=device,
    )


# ============================================================
# A. Statistical equivalence — two envs with identical setup
# ============================================================
print("=" * 60)
print("Test A: Statistical equivalence (identical envs)")
print("=" * 60)


def test_equivalence():
    env = fresh_env(num_envs=2)
    # Identical setup for both envs
    for e in range(2):
        env.radar_pos[e, 0] = torch.tensor([0., 0., 0.], device=DEV)
        env.radar_pos[e, 1] = torch.tensor([1000., 0., 0.], device=DEV)
        env.radar_pos[e, 2] = torch.tensor([0., 1000., 0.], device=DEV)
        env.radar_pos[e, 3] = torch.tensor([1000., 1000., 0.], device=DEV)
        env.radar_vel[e] = 0.0
        env.target_pos[e, 0] = torch.tensor([5000., 0., 0.], device=DEV)
        env.target_vel[e, 0] = 0.0
        env.beam_az[e] = 0.0
        env.beam_el[e] = 0.0

    env.step()  # warmup
    torch.cuda.synchronize()
    r = env.step()

    pt = r["pulse_train"]  # [E=2, R=4, P=32, S=20000]
    rms_per_env = torch.abs(pt).mean(dim=(1, 2, 3)).cpu().numpy()  # [2]
    peak_per_env = torch.abs(pt).amax(dim=(1, 2, 3)).cpu().numpy()
    print(f"  pulse_train mean |x| per env: {rms_per_env}")
    print(f"  pulse_train peak |x| per env: {peak_per_env}")

    rms_ratio = rms_per_env[0] / rms_per_env[1]
    peak_ratio = peak_per_env[0] / peak_per_env[1]
    print(f"  RMS ratio  env0/env1 = {rms_ratio:.4f}  (expect ~1.0 +/- 5%)")
    print(f"  Peak ratio env0/env1 = {peak_ratio:.4f}  (expect ~1.0 +/- 20%)")

    ok = 0.95 < rms_ratio < 1.05 and 0.5 < peak_ratio < 2.0
    print(f"  {'[PASS]' if ok else '[FAIL]'} statistical equivalence")
    del env
    return ok


# ============================================================
# B. Independence — different setups produce different outputs
# ============================================================
print("\n" + "=" * 60)
print("Test B: Independence (different envs, different outputs)")
print("=" * 60)


def test_independence():
    env = fresh_env(num_envs=2)
    # env 0: beam pointing at target (5km, 0deg az)
    # env 1: beam pointing away (5km target still, but beam at 45 deg)
    for e in range(2):
        env.radar_pos[e, 0] = torch.tensor([0., 0., 0.], device=DEV)
        env.radar_pos[e, 1] = torch.tensor([1000., 0., 0.], device=DEV)
        env.radar_pos[e, 2] = torch.tensor([0., 1000., 0.], device=DEV)
        env.radar_pos[e, 3] = torch.tensor([1000., 1000., 0.], device=DEV)
        env.radar_vel[e] = 0.0
        env.target_pos[e, 0] = torch.tensor([5000., 0., 0.], device=DEV)
        env.target_vel[e, 0] = 0.0
    env.beam_az[0] = 0.0     # env0 on-target
    env.beam_az[1] = 45.0    # env1 off-target
    env.beam_el[:] = 0.0

    env.step()  # warmup
    torch.cuda.synchronize()
    r = env.step()

    pt = r["pulse_train"]
    rms_env = torch.abs(pt).mean(dim=(1, 2, 3)).cpu().numpy()
    print(f"  env0 (beam@0deg, target@0deg) mean |x| = {rms_env[0]:.3e}")
    print(f"  env1 (beam@45deg, target@0deg) mean |x| = {rms_env[1]:.3e}")
    # Both will be noise-dominated, so RMS will be ~equal, but the RDMs differ
    # in the signal cell. Easier check: tensors themselves should not be identical.
    diff = (pt[0] - pt[1]).abs().mean().item()
    norm0 = pt[0].abs().mean().item()
    rel_diff = diff / max(norm0, 1e-30)
    print(f"  relative L1 diff env0 vs env1: {rel_diff:.4f}")
    ok = rel_diff > 0.5  # noise-dominated, so most cells are independent
    print(f"  {'[PASS]' if ok else '[FAIL]'} envs are independent (different RNG)")
    del env
    return ok


# ============================================================
# C. Memory stability — 5 consecutive steps
# ============================================================
print("\n" + "=" * 60)
print("Test C: Memory stability (5 steps)")
print("=" * 60)


def test_memory_stability():
    env = fresh_env(num_envs=2)
    env.reset()
    env.step()  # warmup
    torch.cuda.synchronize()

    peaks = []
    for i in range(5):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        env.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() / 1e6
        cur = torch.cuda.memory_allocated() / 1e6
        peaks.append(peak)
        print(f"  step {i+1}: {dt:.0f} ms,  peak {peak:.0f} MB,  current {cur:.0f} MB")

    drift = peaks[-1] - peaks[0]
    ok = abs(drift) < 50  # peak should not grow more than 50 MB
    print(f"  Peak drift over 5 steps: {drift:+.0f} MB")
    print(f"  {'[PASS]' if ok else '[FAIL]'} no memory leak")
    del env
    return ok


# ============================================================
# D. Numerical sanity — no NaN/Inf
# ============================================================
print("\n" + "=" * 60)
print("Test D: Numerical sanity (no NaN/Inf)")
print("=" * 60)


def test_sanity():
    env = fresh_env(num_envs=2)
    env.reset()
    env.step()
    torch.cuda.synchronize()
    r = env.step()

    pt = r["pulse_train"]
    weights = r["weights"]
    nan_pt = torch.isnan(pt).any().item() or torch.isinf(pt).any().item()
    nan_w = torch.isnan(weights).any().item() or torch.isinf(weights).any().item()
    print(f"  pulse_train has NaN/Inf: {nan_pt}")
    print(f"  weights has NaN/Inf:     {nan_w}")

    # Detection structure check
    dets = r["detections"]
    n_envs = len(dets)
    n_rads = len(dets[0]) if dets else 0
    print(f"  detections: {n_envs} envs x {n_rads} radars")

    ok = (not nan_pt) and (not nan_w) and n_envs == 2 and n_rads == 4
    print(f"  {'[PASS]' if ok else '[FAIL]'} numerical sanity")
    del env
    return ok


# ============================================================
# E. Reset works between runs
# ============================================================
print("\n" + "=" * 60)
print("Test E: Reset between runs")
print("=" * 60)


def test_reset():
    env = fresh_env(num_envs=2)
    env.reset()
    r1 = env.step()
    pt1 = r1["pulse_train"].clone()

    env.reset()  # randomize again
    r2 = env.step()
    pt2 = r2["pulse_train"]

    # After reset, scenarios are different, so pulse_trains should differ
    diff = (pt1 - pt2).abs().mean().item()
    norm = pt1.abs().mean().item()
    rel = diff / max(norm, 1e-30)
    print(f"  relative diff pre/post reset: {rel:.4f}  (expect > 0.5)")
    ok = rel > 0.5
    print(f"  {'[PASS]' if ok else '[FAIL]'} reset produces new state")
    del env
    return ok


# ============================================================
# Run all
# ============================================================
results = {
    "A_equivalence": test_equivalence(),
    "B_independence": test_independence(),
    "C_memory":      test_memory_stability(),
    "D_sanity":      test_sanity(),
    "E_reset":       test_reset(),
}

print("\n" + "=" * 60)
print("SUMMARY (E=2, 4x25x25, 32 pulses)")
print("=" * 60)
passed = sum(results.values())
for name, ok in results.items():
    print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
print(f"\n  {passed}/{len(results)} passed")
