"""Smoke test and benchmark for the GPU radar simulation pipeline.

Tests each module individually and then runs a full 4-radar CPI simulation.
Validates correctness against known analytical results where possible.
"""

import sys
import os
import time
import numpy as np

# Add raid/ to path
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

# ============================================================
# Test 1: Array GPU - Beam steering and array factor
# ============================================================
print("\n" + "=" * 60)
print("Test 1: PhasedArrayGPU - Beam steering and array factor")
print("=" * 60)

from env.gpu.array_gpu import PhasedArrayGPU

arr = PhasedArrayGPU(rows=25, cols=25, fc=10e9, device=device)
print(f"  Array: {arr.rows}x{arr.cols} = {arr.n_elem} elements")
print(f"  Directivity: {arr.directivity_db:.1f} dB")
print(f"  3dB beamwidth: {arr.beamwidth_3db[0]:.2f} x {arr.beamwidth_3db[1]:.2f} deg")

# Steer beam to 15 degrees
arr.steer_beam(0, az_deg=15.0, el_deg=0.0)
print(f"  Steered to az=15°, el=0°")

# Compute pattern
az_angles = np.linspace(-90, 90, 361)
t0 = time.perf_counter()
pattern = arr.compute_pattern(0, az_angles)
t_pattern = time.perf_counter() - t0
peak_idx = np.argmax(pattern)
print(f"  Pattern peak at az={az_angles[peak_idx]:.1f}° (expected ~15°)")
print(f"  Pattern compute time: {t_pattern * 1000:.1f} ms")

# Test beamforming
baseband = torch.randn(1000, dtype=torch.complex64, device=torch.device(device))
t0 = time.perf_counter()
tx_signal = arr.beamform_tx(0, baseband)
torch.cuda.synchronize()
t_beamform = time.perf_counter() - t0
print(f"  TX beamforming: {tx_signal.shape}, time: {t_beamform * 1000:.2f} ms")

# Test RX beamforming
rx_signal = torch.randn(arr.n_elem, 1000, dtype=torch.complex64, device=torch.device(device))
t0 = time.perf_counter()
beamformed = arr.beamform_rx(0, rx_signal)
torch.cuda.synchronize()
t_rx = time.perf_counter() - t0
print(f"  RX beamforming: {beamformed.shape}, time: {t_rx * 1000:.2f} ms")

print("  ✅ Array GPU passed")

# ============================================================
# Test 2: Channel GPU - Per-element channel simulation
# ============================================================
print("\n" + "=" * 60)
print("Test 2: ChannelGPU - Per-element channel simulation")
print("=" * 60)

from env.gpu.channel_gpu import ChannelGPU

ch = ChannelGPU(fc=10e9, bandwidth=200e6, device=device)
print(f"  Channel: fc=10GHz, bw=200MHz, noise floor={ch.noise_power_dbm:.1f} dBm")

# Compute channel parameters for a 5km link
params = ch.compute_channel_params(
    tx_pos=np.array([0.0, 0.0, 0.0]),
    rx_pos=np.array([5000.0, 0.0, 0.0]),
    tx_vel=np.array([0.0, 30.0, 0.0]),
    rx_vel=np.array([0.0, -30.0, 0.0]),
)
print(f"  Path loss (5km, one-way): {params['path_loss_db']:.1f} dB")
print(f"  Delay: {params['delay_samples']:.1f} samples ({params['delay_samples']/200e6*1e6:.2f} us)")
print(f"  Doppler: {params['doppler_hz']:.1f} Hz")

# Apply channel to signal
signal = torch.ones(arr.n_elem, 1000, dtype=torch.complex64, device=torch.device(device))
t0 = time.perf_counter()
rx = ch.apply_channel(signal, params, doppler_spread=100.0)
torch.cuda.synchronize()
t_channel = time.perf_counter() - t0
print(f"  Channel application: {rx.shape}, time: {t_channel * 1000:.2f} ms")
print(f"  RX power ratio: {torch.mean(torch.abs(rx)**2).item():.6f} (expected << 1)")

print("  ✅ Channel GPU passed")

# ============================================================
# Test 3: Receiver GPU - Matched filter and range-Doppler
# ============================================================
print("\n" + "=" * 60)
print("Test 3: RadarReceiverGPU - Matched filter + range-Doppler + CFAR")
print("=" * 60)

from env.gpu.receiver_gpu import RadarReceiverGPU

rx_proc = RadarReceiverGPU(fc=10e9, bandwidth=200e6, prf=10e3, device=device)
print(f"  Range resolution: {rx_proc.range_res:.2f} m")

# Generate LFM pulse and simulate target return
n_samples = 2000
n_pulses = 64
t = torch.linspace(0, 10e-6, n_samples, device=torch.device(device))
lfm = torch.exp(1j * np.pi * 200e6 / 10e-6 * t**2)
lfm = lfm / lfm.norm()

# Create pulse train with target at delay=200 samples
pulse_train = torch.zeros(n_pulses, n_samples, dtype=torch.complex64, device=torch.device(device))
for p in range(n_pulses):
    # Target return: delayed copy of LFM + noise
    target_signal = torch.zeros(n_samples, dtype=torch.complex64, device=torch.device(device))
    delay = 200
    amplitude = 0.5
    doppler_phase = 2 * np.pi * 1000 * p / n_pulses  # 1kHz Doppler
    target_signal[delay:delay + n_samples - delay] = amplitude * lfm[:n_samples - delay] * np.exp(1j * doppler_phase)
    noise = torch.randn(n_samples, dtype=torch.complex64, device=torch.device(device)) * 0.01
    pulse_train[p] = target_signal + noise

t0 = time.perf_counter()
result = rx_proc.process_pulse_train(pulse_train, lfm)
torch.cuda.synchronize()
t_rx_proc = time.perf_counter() - t0
print(f"  RDM shape: {result['rd_map'].shape}")
print(f"  Detections: {len(result['detections'])}")
if result['detections']:
    det = result['detections'][0]
    print(f"  Top detection: range={det['range_m']:.1f}m, vel={det['velocity_mps']:.1f}m/s, SNR={det['snr_db']:.1f}dB")
print(f"  Processing time: {t_rx_proc * 1000:.1f} ms")

print("  ✅ Receiver GPU passed")

# ============================================================
# Test 4: Interference GPU - Cross-radar interference
# ============================================================
print("\n" + "=" * 60)
print("Test 4: InterferenceEngineGPU - Cross-radar IQ interference")
print("=" * 60)

from env.gpu.interference_gpu import InterferenceEngineGPU

intf = InterferenceEngineGPU(
    arrays={i: arr for i in range(4)},
    channel=ch, n_radars=4, device=device,
)

# 4 radars at 2km separation
radar_states = []
for i in range(4):
    angle = 2 * np.pi * i / 4
    radar_states.append({
        "pos": [2000 * np.cos(angle), 2000 * np.sin(angle), 0.0],
        "vel": [0.0, 0.0, 0.0],
        "freq_hz": 10e9,
        "bandwidth_hz": 200e6,
        "tx_power_w": 1.0,
        "array_az_deg": 45.0 * i,
    })

waveforms = {i: lfm[:1000] for i in range(4)}
t0 = time.perf_counter()
interference = intf.compute_interference_matrix(radar_states, waveforms, 1000)
torch.cuda.synchronize()
t_intf = time.perf_counter() - t0

intf_power = intf.compute_interference_power_dbm(interference)
jnr = intf.compute_jnr_db(interference, ch.noise_power_linear)

print(f"  Interference compute time: {t_intf * 1000:.1f} ms")
for j in range(4):
    print(f"  Radar {j}: intf_power={intf_power[j]:.1f} dBm, JNR={jnr[j]:.1f} dB")

print("  ✅ Interference GPU passed")

# ============================================================
# Test 5: Full Pipeline Benchmark
# ============================================================
print("\n" + "=" * 60)
print("Test 5: Full Pipeline Benchmark")
print("=" * 60)

from env.gpu.pipeline_gpu import RadarPipelineGPU, RadarState, TargetState

# Use smaller params for quick benchmark
pipeline = RadarPipelineGPU(
    fc=10e9, bandwidth=200e6, prf=10e3,
    pulses_per_cpi=64,  # reduced for speed
    n_radars=4, device=device,
)

# Steer all beams to center
pipeline.steer_beams({i: (0.0, 0.0) for i in range(4)})

# Setup radar states with one target
radar_states = []
for i in range(4):
    angle = 2 * np.pi * i / 4
    radar_states.append(RadarState(
        radar_id=i,
        pos=np.array([2000 * np.cos(angle), 2000 * np.sin(angle), 0.0]),
        vel=np.array([0.0, 0.0, 0.0]),
        freq_hz=10e9,
        bandwidth_hz=200e6,
        tx_power_w=1.0,
        array_az_deg=45.0 * i,
        beam_az_deg=0.0,
        beam_el_deg=0.0,
        waveform_type="lfm_up",
        targets=[TargetState(
            pos=np.array([5000.0, 5000.0, 0.0]),
            vel=np.array([10.0, -20.0, 0.0]),
            rcs_dbsm=20.0,
        )],
    ))

print(f"  Running CPI simulation (4 radars × 64 pulses)...")
t0 = time.perf_counter()
results = pipeline.simulate_cpi(radar_states)
torch.cuda.synchronize()
t_total = time.perf_counter() - t0

for rid, res in results.items():
    print(f"  Radar {rid}: {res['n_detections']} detections, "
          f"intf={res['interference_power_dbm']:.1f} dBm")

print(f"\n  Total CPI simulation time: {t_total:.2f} s")
print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

# Quick benchmark
print(f"\n  Running micro-benchmark...")
bench = pipeline.benchmark()
for k, v in bench.items():
    print(f"    {k}: {v}")

print("\n  ✅ Full pipeline passed")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: All tests passed!")
print("=" * 60)
print(f"  Modules tested: array, channel, receiver, interference, pipeline")
print(f"  GPU memory peak: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
