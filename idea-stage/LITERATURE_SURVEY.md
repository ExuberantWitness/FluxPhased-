# Literature Survey: GPU-Accelerated IQ-Level Phased Array Radar Interference Simulation

**Date**: 2026-05-10
**Direction**: 四个25x25相控阵雷达IQ级(200MHz)相互干扰GPU信号仿真

## Scale Analysis (Problem Sizing)

| Parameter | Value | Implication |
|-----------|-------|-------------|
| Radars | 4 × 25×25 = 2500 elements | 需element-level波束合成 |
| Sampling rate | 200 MSps (complex) | Nyquist采样 |
| CPI duration | 50 ms | 10M samples/element |
| Total samples per CPI | 2500 × 10M = 25B complex | 400 GB (complex128) / 200 GB (complex64) |
| GPU memory (RTX 4090) | 24 GB | 仅能容纳 ~6% 数据 (complex64) |
| Inter-element channels | 2500 × 2499 = 6.25M pairs | 完全交叉干扰矩阵 |

**结论**: 无法一次性加载所有IQ数据到GPU，必须采用分块/频域/压缩策略。

## Key References

### 1. Tensor-Core Beamformer (arXiv:2505.03269, May 2025)
- **URL**: https://arxiv.org/abs/2505.03269
- **核心**: 利用NVIDIA GPU张量核心(Tensor Cores)加速波束形成运算
- **启示**: beamforming可建模为矩阵乘法 → 利用MMA指令集

### 2. RadarSimPy (RadarSimX)
- **URL**: https://radarsimx.com/radarsimx/radarsimpy/
- **核心**: Python雷达仿真器，支持相控阵基带数据生成
- **局限**: 无GPU加速，大阵列速度慢

### 3. MIMO Radar CPU/GPU Parallel (PMC, 2022)
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC8749940/
- **核心**: CPU/GPU异构架构，模块合并策略减少数据传输

### 4. GPU Real-Time SDR Processing (FOSDEM 2024)
- **URL**: https://archive.fosdem.org/2024/events/attachments/fosdem-2024-1643-using-gpu-for-real-time-sdr-signal-processing/slides/22546/GPUforDSP_v2_lNrACIz.pdf
- **核心**: GPU实时雷达处理和数字波束形成

### 5. MathWorks GPU Radar Acceleration
- **URL**: https://www.mathworks.com/help/radar/ug/accelerating-radar-signal-processing-using-gpu.html
- **核心**: MATLAB GPU Coder加速雷达信号处理链

### 6. GPU Passive Bistatic Radar (MDPI, 2023)
- **URL**: https://www.mdpi.com/2072-4292/15/22/5421
- **核心**: GPU并行信号处理方案用于无源双基雷达

### 7. cuSignal (RAPIDS)
- **URL**: https://github.com/rapidsai/cusignal
- **核心**: 基于CuPy的GPU加速信号处理原语

### 8. NVIDIA GTC 2025: GPU/CUDA Radar Signal Processing
- **URL**: https://www.nvidia.com/en-us/on-demand/session/gtc25-s71459/
- **核心**: NVIDIA官方关于相控阵雷达GPU加速的最新指导

## Research Gaps

1. **无现有端到端工具**: 没有Python库能同时处理4个625阵元相控阵的IQ级互干扰
2. **内存墙**: 25B复数采样超出单GPU显存，需要novel的内存管理策略
3. **Element-level干扰**: 现有工作多在阵列级或链路预算级，element-level IQ干扰仿真几乎空白
4. **Python生态缺位**: 雷达IQ仿真多在MATLAB/CUDA C，Python(PyTorch/CuPy)的方案少
5. **多雷达互干扰**: 现有MIMO仿真多为单雷达多通道，非多雷达相互干扰场景

## Approach Landscape

```
Element-Level IQ仿真方案对比:

        ┌───────────────────────────────────────────────┐
        │           Memory-Compute Tradeoff              │
        │                                                │
  High  │  ┌─────────────┐                               │
  Mem   │  │ Time-Domain  │ ← 精确但不可行(400GB)       │
        │  │ Element-Level│                               │
        │  └──────┬───────┘                               │
        │         │                                       │
        │  ┌──────▼───────┐  ┌───────────────┐           │
        │  │ Freq-Domain  │  │ PyTorch       │           │
        │  │ Tiled FFT    │  │ Pulse-Stream  │           │
  Med   │  │ (推荐#1)     │  │ (推荐#4)      │           │
        │  └──────┬───────┘  └───────┬───────┘           │
        │         │                  │                    │
        │  ┌──────▼───────┐  ┌──────▼───────┐           │
        │  │ Subarray     │  │ Budget-Hybrid│           │
        │  │ Approximation│  │ IQ Inject    │           │
  Low   │  │ (方案#2)     │  │ (方案#3)     │           │
        │  └──────────────┘  └──────────────┘           │
        │                                                │
        └───────────────────────────────────────────────┘
```
