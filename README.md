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
│   ├── interference_gpu.py  # IQ-level cross-radar interference (12 links)
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
