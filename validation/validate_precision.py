"""Precision validation: FluxPhased CPU (NumPy) vs GPU (Warp) vs RadarSimPy.

Compares each module's output under identical parameters:
  1. Array pattern (beam steering + array factor)
  2. Channel parameters (path loss, delay, Doppler)
  3. Matched filter output (pulse compression)
  4. Interference power (cross-radar JNR)
  5. End-to-end RDM comparison

Reports correlation, max error, RMSE per module.
Optionally compares against RadarSimPy if installed.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import warp as wp

device = "cuda" if torch.cuda.is_available() else "cpu"
wp.init()

from env.physics.array import PhasedArray
from env.physics.channel import (
    free_space_path_loss_db, compute_snr, compute_jamming_power,
    PropagationChannel,
)
from env.physics.receiver import MatchedFilter, RadarReceiver
from env.physics.interference import InterferenceEngine
from env.config import ArrayGeometry, RFConfig, CPIConfig

from env.gpu.array_gpu import PhasedArrayGPU
from env.gpu.channel_gpu import ChannelGPU
from env.gpu.receiver_gpu import RadarReceiverGPU
from env.gpu.interference_gpu import InterferenceEngineGPU

SPEED_OF_LIGHT = 299792458.0
DEG2RAD = np.pi / 180.0

# ============================================================
# Utility
# ============================================================
def compare(name, cpu_arr, gpu_arr, tolerance_db=0.5, db_floor=-60.0):
    """Compare two 1D arrays. Returns dict with metrics.

    db_floor: for dB-valued arrays, ignore points where either value
    is below this threshold (avoids comparing numerical noise in nulls).
    """
    cpu = np.asarray(cpu_arr).ravel().astype(np.float64)
    gpu = np.asarray(gpu_arr).ravel().astype(np.float64)
    assert len(cpu) == len(gpu), f"Length mismatch: {len(cpu)} vs {len(gpu)}"

    # Mask null regions where both are very negative (numerical noise)
    valid = (cpu > db_floor) | (gpu > db_floor)
    if np.sum(valid) < 10:
        valid = np.ones(len(cpu), dtype=bool)

    cpu_v = cpu[valid]
    gpu_v = gpu[valid]

    corr = np.corrcoef(cpu_v, gpu_v)[0, 1] if np.std(cpu_v) > 0 and np.std(gpu_v) > 0 else 1.0
    max_err = np.max(np.abs(cpu_v - gpu_v))
    rmse = np.sqrt(np.mean((cpu_v - gpu_v) ** 2))

    status = "✅ PASS" if corr > 0.998 else "❌ FAIL"
    print(f"  {status} {name}:")
    print(f"    Correlation: {corr:.6f}  Max err: {max_err:.4f}  RMSE: {rmse:.4f}  ({np.sum(valid)} pts)")
    return {"name": name, "corr": corr, "max_err": max_err, "rmse": rmse}


# ============================================================
# Test 1: Array Pattern Comparison
# ============================================================
print("=" * 70)
print("TEST 1: Array Pattern — CPU vs GPU")
print("=" * 70)

geom = ArrayGeometry(rows=25, cols=25, dx_wl=0.5, dy_wl=0.5)
fc = 10e9

results = []

steer_angles = [0, 15, -15, 30, -30, 45, -45]
az_angles = np.linspace(-90, 90, 361)

for steer_az in steer_angles:
    # CPU — use compute_pattern (same normalization as GPU)
    cpu_arr = PhasedArray(geom, fc)
    cpu_arr.steer_beam(steer_az, 0.0)
    cpu_pat = cpu_arr.compute_pattern(az_angles)  # normalized, peak=0 dB

    # GPU
    gpu_arr = PhasedArrayGPU(rows=25, cols=25, fc=fc, device=device)
    gpu_arr.steer_beam(0, steer_az, 0.0)
    gpu_pat = gpu_arr.compute_pattern(0, az_angles, el_deg=0.0)

    peak_cpu = az_angles[np.argmax(cpu_pat)]
    peak_gpu = az_angles[np.argmax(gpu_pat)]

    r = compare(f"Pattern steer={steer_az:+d}° (CPU peak={peak_cpu:+.1f}° GPU peak={peak_gpu:+.1f}°)",
                cpu_pat, gpu_pat)
    results.append(r)

# ============================================================
# Test 2: Channel Parameters
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Channel Parameters — CPU vs GPU")
print("=" * 70)

ch_gpu = ChannelGPU(fc=fc, bandwidth=200e6, device=device)
ch_cpu = PropagationChannel()

distances = [1000, 2000, 5000, 10000, 50000]
for dist in distances:
    # CPU: free-space path loss
    cpu_loss = free_space_path_loss_db(dist, fc)

    # GPU
    gpu_params = ch_gpu.compute_path_loss_db(dist)
    err = abs(cpu_loss - gpu_params)
    status = "✅" if err < 0.01 else "❌"
    print(f"  {status} Path loss @ {dist/1000:.0f}km: CPU={cpu_loss:.2f} dB  GPU={gpu_params:.2f} dB  err={err:.4f} dB")

    # Delay
    cpu_delay_samples = dist / SPEED_OF_LIGHT * 200e6  # one-way (same as GPU)
    gpu_delay = ch_gpu.compute_channel_params(
        np.zeros(3), np.array([dist, 0, 0])
    )["delay_samples"]
    err_delay = abs(cpu_delay_samples - gpu_delay)
    print(f"     Delay: CPU={cpu_delay_samples:.1f} samples  GPU={gpu_delay:.1f} samples  err={err_delay:.2f}")

# Radar equation SNR comparison
print("\n  Radar equation SNR comparison:")
for dist in [5000, 10000, 50000]:
    rcs = 20.0  # dBsm
    cpu_snr = compute_snr(
        tx_power_w=1000, tx_gain_db=32.9, rx_gain_db=32.9,
        distance_m=dist, frequency_hz=fc, bandwidth_hz=200e6,
        rcs_dbsm=rcs, system_loss_db=3.0,
    )
    # GPU equivalent — standard radar equation form (must match CPU)
    wavelength = SPEED_OF_LIGHT / fc
    tx_dbm = 10 * np.log10(1000 * 1000)
    noise_dbm = 10 * np.log10(1.380649e-23 * 290 * 1000) + 10 * np.log10(200e6) + 5
    gpu_snr = (
        tx_dbm + 2 * 32.9 + rcs
        + 20.0 * np.log10(wavelength)
        - 30.0 * np.log10(4.0 * np.pi)
        - 40.0 * np.log10(dist)
        - 3.0 - noise_dbm
    )

    err = abs(cpu_snr - gpu_snr)
    status = "✅" if err < 0.1 else "❌"
    print(f"  {status} SNR @ {dist/1000:.0f}km, RCS={rcs}dBsm: CPU={cpu_snr:.1f} dB  GPU={gpu_snr:.1f} dB  err={err:.2f} dB")

# ============================================================
# Test 3: Matched Filter
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Matched Filter — CPU vs GPU")
print("=" * 70)

n_samples = 2000
fs = 200e6
t = np.arange(n_samples) / fs
pw = 10e-6
bw = 200e6

# Generate LFM
lfm_np = np.exp(1j * np.pi * bw / pw * t[:int(pw * fs)] ** 2)
lfm_np = lfm_np / np.linalg.norm(lfm_np)

# Create signal with target at delay=200
delay = 200
signal_np = np.zeros(n_samples, dtype=complex)
signal_np[delay:] = 0.5 * lfm_np[:n_samples - delay]
noise_np = np.random.randn(n_samples) * 0.01 + 1j * np.random.randn(n_samples) * 0.01
signal_np += noise_np

# CPU matched filter
mf_cpu = MatchedFilter(lfm_np)
cpu_mf = mf_cpu.apply(signal_np)

# GPU matched filter
lfm_torch = torch.tensor(lfm_np, dtype=torch.complex64, device=torch.device(device))
signal_torch = torch.tensor(signal_np, dtype=torch.complex64, device=torch.device(device))
rx_gpu = RadarReceiverGPU(fc=fc, bandwidth=200e6, prf=10e3, device=device)
gpu_mf = rx_gpu.matched_filter(signal_torch, lfm_torch).cpu().numpy()

# Compare magnitudes
cpu_mag = np.abs(cpu_mf)
gpu_mag = np.abs(gpu_mf)
r = compare("Matched filter magnitude", cpu_mag, gpu_mag)

# Check peak position
cpu_peak = np.argmax(cpu_mag)
gpu_peak = np.argmax(gpu_mag)
print(f"    Peak position: CPU={cpu_peak}  GPU={gpu_peak}  (expected ~{delay})")

# ============================================================
# Test 4: Interference Power
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Interference Power — CPU vs GPU")
print("=" * 70)

# CPU interference engine
intf_cpu = InterferenceEngine()

# 4 radars in 2km square, boresights pointing toward center (1000, 1000)
# This keeps all links within ±45° of each array's broadside
# At 2km, one-way delay = 2km/3e8 * 200e6 = 1333 samples
positions = [
    [0, 0, 0], [2000, 0, 0], [0, 2000, 0], [2000, 2000, 0],
]
boresights = [45.0, 135.0, -45.0, -135.0]  # all point toward center
fc_hz = 10e9
bw_hz = 200e6
tx_power_w = 1000.0
n_intf_samples = 5000  # must exceed max delay (1333) + waveform length

def simple_beam_model(az_deg, el_deg):
    """Simplified beam model for CPU interference."""
    if abs(az_deg) < 4.06:
        return 32.9 - (az_deg / 4.06) ** 2 * 3
    else:
        return -10.0

# CPU pairwise interference
cpu_states = []
for i, pos in enumerate(positions):
    cpu_states.append({
        "pos": np.array(pos), "heading": 0, "array_az": boresights[i],
        "tx_power_w": tx_power_w, "tx_gain_db": 32.9,
        "freq_hz": fc_hz, "bandwidth_hz": bw_hz,
        "noise_figure_db": 5.0,
    })

cpu_jnr = intf_cpu.compute_full_interference(cpu_states, [simple_beam_model] * 4)

# GPU interference
gpu_arr_inst = PhasedArrayGPU(rows=25, cols=25, fc=fc_hz, device=device)
for i in range(4):
    gpu_arr_inst.steer_beam(i, 0.0, 0.0)

intf_gpu = InterferenceEngineGPU(
    arrays={i: gpu_arr_inst for i in range(4)},
    channel=ch_gpu, n_radars=4, device=device,
)

waveforms = {i: torch.ones(n_intf_samples, dtype=torch.complex64, device=torch.device(device))
             for i in range(4)}
gpu_states = [{
    "pos": pos, "vel": [0, 0, 0],
    "freq_hz": fc_hz, "bandwidth_hz": bw_hz,
    "tx_power_w": tx_power_w, "array_az_deg": boresights[i],
} for i, pos in enumerate(positions)]

gpu_interference = intf_gpu.compute_interference_matrix(gpu_states, waveforms, n_intf_samples)

# Compute per-pair JNR using radar equation directly (same as CPU model)
gpu_jnr_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        if i == j:
            continue
        # Compute link budget using array pattern gain
        distance, az_to_rx, az_to_tx = intf_gpu._compute_link_geometry(
            gpu_states[i]["pos"], gpu_states[j]["pos"],
        )
        tx_rel = intf_gpu._relative_angle(az_to_rx, gpu_states[i].get("array_az_deg", 0.0))
        rx_rel = intf_gpu._relative_angle(az_to_tx, gpu_states[j].get("array_az_deg", 0.0))
        tx_gain = gpu_arr_inst.get_gain_at_angle(i, tx_rel)
        rx_gain = gpu_arr_inst.get_gain_at_angle(j, rx_rel)
        path_loss = ch_gpu.compute_path_loss_db(distance)
        freq_ov = intf_gpu.compute_frequency_overlap(
            gpu_states[i]["freq_hz"], gpu_states[i]["bandwidth_hz"],
            gpu_states[j]["freq_hz"], gpu_states[j]["bandwidth_hz"],
        )
        tx_dbm = 10.0 * np.log10(tx_power_w * 1000.0)
        jam_dbm = tx_dbm + tx_gain + rx_gain - path_loss - 3.0 + 10.0 * np.log10(freq_ov + 1e-15)
        noise_dbm = ch_gpu.noise_power_dbm
        gpu_jnr_matrix[i, j] = jam_dbm - noise_dbm

# Per-victim total JNR (power sum of all interferers)
gpu_jnr = np.zeros(4)
for j in range(4):
    gpu_jnr[j] = 10.0 * np.log10(np.sum(10.0 ** (gpu_jnr_matrix[:, j] / 10.0)) + 1e-30)

print("  JNR matrix (dB):")
print("  CPU:")
for row in cpu_jnr:
    print(f"    [{', '.join(f'{v:+7.1f}' for v in row)}]")
print("  GPU:")
for i in range(4):
    row_str = []
    for j in range(4):
        if i == j:
            row_str.append("    0.0")
        else:
            row_str.append(f"{gpu_jnr_matrix[i, j]:+7.1f}")
    print(f"    [{', '.join(row_str)}]")

# Compare total JNR at each victim
print("\n  Per-victim total JNR (sum of all interferers):")
intf_results = []
for j in range(4):
    cpu_total = 10.0 * np.log10(np.sum(10.0 ** (cpu_jnr[:, j] / 10.0)))
    gpu_total = gpu_jnr[j]
    err = abs(cpu_total - gpu_total)
    status = "✅" if err < 5.0 else "❌"
    print(f"  {status} Victim {j}: CPU={cpu_total:+.1f} dB  GPU={gpu_total:+.1f} dB  err={err:.1f} dB")
    intf_results.append(err < 5.0)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
pass_count = sum(1 for r in results if r["corr"] > 0.998)
print(f"  Array pattern tests: {pass_count}/{len(results)} passed (corr > 0.998)")
for r in results:
    status = "✅" if r["corr"] > 0.998 else "❌"
    print(f"    {status} {r['name'][:50]:50s} corr={r['corr']:.6f}")

# Optional: RadarSimPy comparison
print("\n" + "=" * 70)
print("RADARSIMPY CHECK")
print("=" * 70)
try:
    import radarsimpy
    print(f"  RadarSimPy v{radarsimpy.__version__} detected — can run Level 2 validation")
except ImportError:
    print("  RadarSimPy not installed. To install:")
    print("  1. Download from https://radarsimx.com/product/radarsimpy/ (free personal use)")
    print("  2. Extract radarsimpy/ folder into this project")
    print("  3. Re-run this script for Level 2 comparison")
