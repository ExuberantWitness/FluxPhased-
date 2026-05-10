# FluxPhased

GPU-accelerated IQ-level signal simulation for mutual interference between four 25×25 phased array radars (200MHz bandwidth) using NVIDIA Warp + PyTorch.

## Architecture

```
radar_sim/
├── config.py            # System configuration (25x25 array, RF, waveform, battlefield)
├── physics/             # CPU physics baseline (NumPy)
│   ├── array.py         # Phased array model (25×25, beam steering, array factor)
│   ├── channel.py       # Propagation (path loss, Rayleigh fading, radar equation)
│   ├── waveform.py      # Waveform generation (LFM, Barker, Frank, Costas, NLFM, P4)
│   ├── receiver.py      # Receiver DSP (matched filter, CFAR, range-Doppler)
│   └── interference.py  # dB-level cross-radar interference
├── gpu/                 # GPU-accelerated simulation (Warp + PyTorch)
│   ├── array_gpu.py     # Warp: beam steering, array factor, per-element beamforming
│   ├── channel_gpu.py   # Warp: per-element delay/Doppler/fading
│   ├── receiver_gpu.py  # torch.fft: matched filter + Warp: 2D CA-CFAR
│   ├── interference_gpu.py  # Radar equation link budget + IQ-level interference
│   ├── pipeline_gpu.py  # Full 4-radar CPI orchestrator
│   ├── waveform_gpu.py  # PyTorch GPU waveform generation
│   └── test_gpu_pipeline.py  # Validation test suite
└── env/                 # Multi-agent battlefield environment (PettingZoo)
```

## Quick Start

```bash
conda activate env_isaacsim  # requires warp, torch with CUDA
python radar_sim/gpu/test_gpu_pipeline.py
```

## Precision Validation

CPU (NumPy float64) vs GPU (Warp float32 + torch.fft) under identical parameters:

```bash
python validation/validate_precision.py
```

### Results

| Module | Metric | Result |
|--------|--------|--------|
| Array pattern (7 steer angles: 0°, ±15°, ±30°, ±45°) | Correlation | **1.000000** |
| Array pattern | Max error (mainlobe) | **0.0024 dB** |
| Path loss (1–50 km) | Absolute error | **0.0000 dB** |
| Propagation delay | Absolute error | **0.00 samples** |
| Radar equation SNR | Absolute error | **0.00 dB** |
| Matched filter | Correlation | **1.000000** |
| Matched filter | Peak position | **Exact match** |
| Interference JNR (total per victim) | Absolute error | **0.1 dB** |

### Interference JNR matrix (4 radars, 2 km spacing, boresights toward center)

```
CPU (dB):                          GPU (dB):
  [+0.0,  +4.5,  +4.5, +87.3]      [  0.0, +14.7, +14.7, +87.3]
  [+4.5,  +0.0, +87.3,  +4.5]      [+14.7,   0.0, +87.3, +14.7]
  [+4.5, +87.3,  +0.0,  +4.5]      [+14.7, +87.3,   0.0, +14.7]
  [+87.3, +4.5,  +4.5,  +0.0]      [+87.3, +14.7, +14.7,   0.0]
```

Per-pair difference (~10 dB for adjacent links) comes from the CPU using a simplified beam model (fixed -10 dB sidelobe) vs GPU using the actual array pattern (sidelobe varies with angle). Total interference power matches within 0.1 dB.

## Tech Stack

- **NVIDIA Warp 1.7.2** — Custom CUDA kernels for per-element signal processing
- **PyTorch 2.5+** — GPU tensor ops and torch.fft (cuFFT backend)
- **Complex numbers**: Interleaved float32 (Warp lacks native complex64)

## Specs

| Parameter | Value |
|-----------|-------|
| Radars | 4 × 25×25 phased arrays |
| Bandwidth | 200 MHz (IQ-level) |
| Elements total | 4 × 625 = 2,500 |
| GPU memory | ~1.1 GB / 6.4 GB (RTX 2060) |

## Bug Fixes (from validation)

| Bug | File | Fix |
|-----|------|-----|
| Steer kernel sign error | `array_gpu.py` | `-taper*sin(phase)` → `+taper*sin(phase)` (pattern peak was at -az) |
| Channel delay direction | `channel_gpu.py` | `src = s + d_int` → `s - d_int` (was time-advance, not delay) |
| Interference missing TX directivity | `interference_gpu.py` | Rewrote to use Friis link budget with antenna gains |
