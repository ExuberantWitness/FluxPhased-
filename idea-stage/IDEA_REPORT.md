# Idea Discovery Report

**Direction**: GPU-accelerated IQ-level signal simulation for four 25x25 phased array radars (200MHz bandwidth) mutual interference
**Date**: 2026-05-10
**Pipeline**: research-lit (web + Sionna deep dive) → idea generation → feasibility analysis

## Executive Summary

Four technical approaches were evaluated for GPU-accelerating the existing `raid/radar_sim/` physics modules (array, channel, receiver, interference) from NumPy to full IQ-level signal simulation. The **recommended approach is a Hybrid PyTorch+CuPy pipeline** that leverages PyTorch's existing GPU waveform generation, adds CuPy for FFT-heavy signal processing (matched filter, range-Doppler, CFAR), and implements per-element channel simulation as batched PyTorch operations. This approach offers the best balance of performance, code reuse, and ecosystem compatibility, with estimated 50-100x speedup over NumPy for the full simulation chain.

## Literature Landscape

### NVIDIA Sionna RT (v2.0.1, Apr 2025)
- **Architecture**: RT module (Dr.Jit + Mitsuba 3), PHY module (PyTorch since v2.0), SYS module
- **Radar capability**: Cannot do IQ-level radar simulation. No radar waveforms, no range-Doppler, no CFAR, no RCS modeling
- **Array limitation**: Single TX array config + single RX array config per scene (GitHub Issue #673)
- **Usefulness**: Can provide propagation channel impulse responses between array element pairs via ray tracing. Partial building block only
- **Source**: [Sionna RT Technical Report (arXiv:2504.21719)](https://arxiv.org/abs/2504.21719), [NVIDIA Developer](https://developer.nvidia.com/sionna)

### NVIDIA Warp
- **Architecture**: Python JIT → CUDA kernels. Supports differentiable programming via `wp.kernel` decorators
- **Complex number support**: No native complex64/complex128 types (Issue #1394) — critical limitation for IQ signal processing
- **FFT**: Can invoke `nvmath` FFT from inside kernels, but complex number handling requires workarounds
- **Usefulness**: Best for custom physics kernels (e.g., per-element delay-Doppler computation). Not ideal for FFT-heavy signal processing due to complex number gaps
- **Source**: [NVIDIA Warp GitHub](https://github.com/nvidia/warp), [GTC 2026 Session](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81837/)

### GPU Radar Processing (Existing Work)
- **CUDA Range-Doppler** ([GitHub](https://github.com/NiclasEsser1/CUDARangeDopplerProcessing)): CUDA-based FMCW range-Doppler processing
- **MathWorks GPU**: Full radar chain (beamforming, pulse compression, Doppler, CFAR) on GPU via MATLAB
- **MIMO Radar CPU/GPU** ([MDPI](https://www.mdpi.com/1424-8220/22/1/396)): 13% improvement with CPU/GPU hybrid for MIMO radar
- **DiVA Portal thesis**: Real-time beamforming at 100 MHz is challenging even on GPUs

### Key Gap
**No existing open-source framework** provides all of: multi-array IQ-level simulation, GPU acceleration, radar-specific DSP (matched filter, CFAR, RDM), and cross-radar interference modeling.

## Ranked Ideas

### 🏆 Idea 1: Hybrid PyTorch + CuPy Pipeline — RECOMMENDED
**Pilot Signal**: STRONG POSITIVE | **Feasibility**: HIGH | **Estimated Speedup**: 50-100x

**Concept**: Extend the existing `raid/radar_sim/gpu/` module:
- **PyTorch** for per-element waveform generation, beamforming, array factor
- **CuPy** for FFT-heavy ops: matched filtering, range-Doppler maps, CFAR detection
- **Batched PyTorch ops** for per-element channel simulation (delay, Doppler, fading)

**Why**: PyTorch already in project; CuPy has drop-in `cupy.fft`; complex64 native; DLPack interop; 2500 elements × 200MHz = massive parallelism

**Architecture**:
```
raid/radar_sim/gpu/
├── waveform_gpu.py      # ✅ exists (PyTorch)
├── array_gpu.py         # NEW: batched array factor + steering (PyTorch)
├── channel_gpu.py       # NEW: per-element channel simulation (PyTorch)
├── receiver_gpu.py      # NEW: matched filter + RDM + CFAR (CuPy)
├── interference_gpu.py  # NEW: IQ-level cross-radar interference (PyTorch)
└── pipeline_gpu.py      # NEW: orchestrator for full 4-radar sim
```

**Compute estimate**: ~50-100ms per CPI on RTX 4090 (vs ~5-10s CPU)

**Risk**: Memory — 4 × 625 × 500 × 10000 complex64 ≈ 20GB. May need chunked processing.

---

### Idea 2: NVIDIA Warp Custom Kernels — BACKUP (Phase 2 optimization)
**Pilot Signal**: MODERATE | **Feasibility**: MEDIUM | **Estimated Speedup**: 100-200x theoretical

Fused CUDA kernels for maximum performance. Blocked by missing complex64 support (Issue #1394). Better as Phase 2 optimization after Idea 1 is working.

---

### Idea 3: Sionna RT + Custom PyTorch — NOT RECOMMENDED for this use case
**Pilot Signal**: WEAK | **Feasibility**: LOW-MEDIUM

Sionna cannot handle 4 different arrays in one scene, has no radar waveforms/processing. Overkill for open-terrain EW scenario where closed-form channel models suffice.

---

### Idea 4: Pure PyTorch (torch.fft) — SIMPLER ALTERNATIVE
**Pilot Signal**: MODERATE | **Feasibility**: HIGH | **Estimated Speedup**: 30-50x

Single framework, simpler deps. Less flexible for custom CFAR. Good fallback if CuPy problematic.

## Comparison Matrix

| Criterion | Idea 1 (PyTorch+CuPy) | Idea 2 (Warp) | Idea 3 (Sionna) | Idea 4 (Pure PyTorch) |
|-----------|----------------------|----------------|-----------------|-----------------------|
| Speedup | 50-100x | 100-200x | 10-30x | 30-50x |
| Feasibility | HIGH | MEDIUM | LOW-MEDIUM | HIGH |
| Code reuse | HIGH | LOW | MEDIUM | HIGH |
| Complex64 | Both support | Missing | PyTorch ok | Supported |
| FFT perf | CuPy (excellent) | nvmath (good) | N/A | torch.fft (good) |
| CFAR flex | CuPy (excellent) | Custom kernel | N/A | Limited |
| Eng effort | 2-3 weeks | 4-6 weeks | 3-4 weeks | 2 weeks |

## Implementation Plan (Idea 1)

1. **array_gpu.py** (3d): Batched array factor for 4 arrays × 625 elements, steering vectors in parallel
2. **channel_gpu.py** (3d): Per-element delay/Doppler/fading, batched over 2500 elements
3. **receiver_gpu.py** (5d): CuPy matched filter, Doppler FFT, RDM, CFAR
4. **interference_gpu.py** (3d): IQ-level 12 interference pairs, per-element received signal
5. **pipeline_gpu.py** (2d): Full chain orchestrator
6. **Validation** (2d): GPU vs CPU output comparison (correlation > 0.999)

## Next Steps
- [ ] Confirm approach with user (Gate 1)
- [ ] Implement core GPU modules
- [ ] Run validation experiments
- [ ] Consider Warp optimization (Idea 2) as Phase 2
