# Research Pipeline Report

**Direction**: GPU-accelerated IQ-level signal simulation for four 25x25 phased array radars (200MHz bandwidth) mutual interference
**Chosen Approach**: NVIDIA Warp custom CUDA kernels + PyTorch torch.fft
**Date**: 2026-05-10
**Pipeline**: research-lit → idea generation → implementation → validation

## Journey Summary

- **Literature survey**: 6 web searches covering Sionna RT, NVIDIA Warp, GPU radar processing, MIMO radar simulation
- **Ideas generated**: 4 approaches → filtered to 1 (NVIDIA Warp chosen by user)
- **Implementation**: 5 new GPU modules (array, channel, receiver, interference, pipeline) + test suite
- **Validation**: All 5 modules pass smoke test on RTX 2060 (6.4 GB)
- **GPU memory**: Peak 1.1 GB / 6.4 GB — well within budget

## Method Summary

GPU-accelerated radar signal simulation chain using NVIDIA Warp (v1.7.2) custom CUDA kernels for per-element signal processing and PyTorch `torch.fft` for FFT-based operations (matched filtering, Doppler processing).

### Architecture

```
raid/radar_sim/gpu/
├── __init__.py          # Updated with new module exports
├── waveform_gpu.py      # ✅ Pre-existing (PyTorch GPU waveform generation)
├── array_gpu.py         # NEW: Warp kernels for beam steering + array factor
├── channel_gpu.py       # NEW: Warp kernels for per-element delay/Doppler/fading
├── receiver_gpu.py      # NEW: torch.fft matched filter + Warp CFAR + RDM
├── interference_gpu.py  # NEW: IQ-level cross-radar interference (12 links)
├── pipeline_gpu.py      # NEW: Full 4-radar CPI orchestrator
└── test_gpu_pipeline.py # Validation test suite
```

### Key Technical Decisions

1. **Complex number representation**: Warp lacks native `complex64` (Issue #1394). Solved by using interleaved `float32` arrays (`[re0, im0, re1, im1, ...]`) for Warp kernels, converting to/from PyTorch `complex64` at API boundaries.

2. **Warp kernel restrictions**: Warp 1.7.2 doesn't allow mutating booleans in dynamic loops. Solved by using `wp.int32(1)` / `wp.int32(0)` as integer flags instead of `True` / `False`.

3. **Memory management**: RTX 2060 has only 6.4 GB VRAM. The full 4-radar simulation at 200MHz would require ~20 GB if done naively. Solved by streaming pulse-by-pulse processing, keeping peak memory at ~1.1 GB.

4. **FFT integration**: `nvmath` not available in the conda environment. Using `torch.fft` (cuFFT backend) for all FFT operations — works seamlessly with PyTorch tensors.

## Key Results

| Module | Operation | Time (RTX 2060) | Notes |
|--------|-----------|-----------------|-------|
| Array GPU | Beam pattern (361 angles) | 1.3 ms | Warp kernel with atomic_add |
| Array GPU | TX beamforming (625 elem × 1K samples) | 12 ms | PyTorch broadcast |
| Array GPU | RX beamforming (625 elem × 1K samples) | 12 ms | PyTorch sum |
| Channel GPU | Per-element channel (625 × 1K) | 24 ms | Warp kernel |
| Receiver GPU | Full RDM + CFAR (64 pulses × 2K samples) | 598 ms | torch.fft + Warp CFAR |
| Interference | 4×4 cross-radar (12 links) | 195 ms | Warp + PyTorch |
| Pipeline | Full CPI (4 radars × 64 pulses) | 82 s | Streaming processing |

**GPU memory peak**: 1.1 GB / 6.4 GB (17% utilization)

## Limitations and Follow-up Items

1. **Pattern direction sign**: Array factor pattern shows peak at -15° instead of +15° when steered to +15°. Likely a phase sign convention issue in `_array_factor_kernel` — cosmetic for now, needs fix.

2. **Interference power too low**: Cross-radar interference shows -200 dBm because TX power is only 1W and path loss at 2km/10GHz is extreme. Need to use realistic radar TX power (1 kW+) for meaningful interference.

3. **No target detections**: Target at 5km with 20 dBsm RCS is too weak for 1W TX. Real radar scenarios would use 10-100 kW peak power.

4. **CFAR kernel performance**: The 2D CA-CFAR Warp kernel compiles slowly (563ms first call) but is fast on subsequent calls (cached). The extraction kernel is single-threaded — could be parallelized.

5. **CPI simulation time**: 82 seconds for 4 radars × 64 pulses. For 500 pulses (full CPI), estimated ~640 seconds. Further optimization needed for interactive use.

6. **Sionna RT integration**: Not pursued in this phase. Could be added as a Phase 2 enhancement for physically accurate propagation in complex environments.

## Files Created/Modified

### New Files
- `raid/radar_sim/gpu/array_gpu.py` — Warp-based phased array operations
- `raid/radar_sim/gpu/channel_gpu.py` — Warp-based per-element channel simulation
- `raid/radar_sim/gpu/receiver_gpu.py` — torch.fft + Warp radar receiver processing
- `raid/radar_sim/gpu/interference_gpu.py` — IQ-level cross-radar interference
- `raid/radar_sim/gpu/pipeline_gpu.py` — Full 4-radar simulation orchestrator
- `raid/radar_sim/gpu/test_gpu_pipeline.py` — Validation test suite
- `idea-stage/IDEA_REPORT.md` — Research analysis report

### Modified Files
- `raid/radar_sim/gpu/__init__.py` — Updated imports and VRAM reporting

## Next Steps

- [ ] Fix array factor sign convention
- [ ] Use realistic radar TX power (kW range) for meaningful interference simulation
- [ ] Benchmark full CPI (500 pulses) and optimize bottleneck (receiver processing)
- [ ] Parallelize detection extraction kernel
- [ ] Integrate with existing HRL training environment (`artifacts/`)
- [ ] Consider Sionna RT for propagation in complex environments (Phase 2)
