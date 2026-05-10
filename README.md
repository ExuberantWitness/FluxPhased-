# FluxPhased

GPU-accelerated IQ-level signal simulation for mutual interference between four 25×25 phased array radars (200MHz bandwidth) using NVIDIA Warp + PyTorch.

基于 NVIDIA Warp + PyTorch 的四部 25×25 相控阵雷达（200MHz 带宽）IQ 级互干扰信号 GPU 仿真。

---

## Architecture / 系统架构

```
radar_sim/
├── config.py            # System configuration / 系统配置 (25x25 阵列, 射频, 波形, 战场)
├── physics/             # CPU physics baseline (NumPy) / CPU 物理基线
│   ├── array.py         # Phased array model / 相控阵模型 (25×25, 波束指向, 阵列因子)
│   ├── channel.py       # Propagation / 信道传播 (路径损耗, 瑞利衰落, 雷达方程)
│   ├── waveform.py      # Waveform generation / 波形生成 (LFM, Barker, Frank, Costas, NLFM, P4)
│   ├── receiver.py      # Receiver DSP / 接收机信号处理 (匹配滤波, CFAR, 距离-多普勒)
│   └── interference.py  # dB-level cross-radar interference / dB 级互扰计算
├── gpu/                 # GPU-accelerated simulation / GPU 加速仿真 (Warp + PyTorch)
│   ├── array_gpu.py     # Warp: beam steering, array factor, per-element beamforming
│   ├── channel_gpu.py   # Warp: per-element delay/Doppler/fading / 逐元素延迟/多普勒/衰落
│   ├── receiver_gpu.py  # torch.fft: matched filter + Warp: 2D CA-CFAR
│   ├── interference_gpu.py  # Radar equation link budget + IQ-level interference / 雷达方程链路预算
│   ├── pipeline_gpu.py  # Full 4-radar CPI orchestrator / 四雷达 CPI 编排器
│   ├── waveform_gpu.py  # PyTorch GPU waveform generation / GPU 波形生成
│   └── test_gpu_pipeline.py  # Validation test suite / 功能测试套件
└── env/                 # Multi-agent battlefield environment / 多智能体战场环境 (PettingZoo)
```

## Quick Start / 快速开始

```bash
conda activate env_isaacsim  # requires warp, torch with CUDA
python radar_sim/gpu/test_gpu_pipeline.py
```

---

## Precision Validation / 精度校验

CPU (NumPy float64) vs GPU (Warp float32 + torch.fft) under identical parameters.

在相同参数下对比 CPU（NumPy float64）与 GPU（Warp float32 + torch.fft）的输出。

```bash
python validation/validate_precision.py
```

### Results / 校验结果

| Module / 模块 | Metric / 指标 | Result / 结果 |
|--------|--------|--------|
| Array pattern / 阵列方向图 (7 angles: 0°, ±15°, ±30°, ±45°) | Correlation / 相关系数 | **1.000000** |
| Array pattern / 阵列方向图 | Max error (mainlobe) / 主瓣最大误差 | **0.0024 dB** |
| Path loss / 路径损耗 (1–50 km) | Absolute error / 绝对误差 | **0.0000 dB** |
| Propagation delay / 传播延迟 | Absolute error / 绝对误差 | **0.00 samples** |
| Radar equation SNR / 雷达方程信噪比 | Absolute error / 绝对误差 | **0.00 dB** |
| Matched filter / 匹配滤波 | Correlation / 相关系数 | **1.000000** |
| Matched filter / 匹配滤波 | Peak position / 峰值位置 | **Exact match / 精确一致** |
| Interference JNR / 干扰干噪比 (total per victim) | Absolute error / 绝对误差 | **0.1 dB** |

### Interference JNR Matrix / 干扰 JNR 矩阵

4 radars at 2 km spacing, boresights pointing toward center / 四部雷达 2 km 间距，波束指向中心。

```
CPU (dB):                          GPU (dB):
  [+0.0,  +4.5,  +4.5, +87.3]      [  0.0, +14.7, +14.7, +87.3]
  [+4.5,  +0.0, +87.3,  +4.5]      [+14.7,   0.0, +87.3, +14.7]
  [+4.5, +87.3,  +0.0,  +4.5]      [+14.7, +87.3,   0.0, +14.7]
  [+87.3, +4.5,  +4.5,  +0.0]      [+87.3, +14.7, +14.7,   0.0]
```

Per-pair difference (~10 dB for adjacent links) comes from the CPU using a simplified beam model (fixed -10 dB sidelobe) vs GPU using the actual array pattern (sidelobe varies with angle). Total interference power matches within 0.1 dB.

逐对差异（相邻链路约 10 dB）源于 CPU 使用简化波束模型（旁瓣固定 -10 dB），GPU 使用真实阵列方向图（旁瓣随角度变化）。总干扰功率误差 0.1 dB。

---

## Tech Stack / 技术栈

- **NVIDIA Warp 1.7.2** — Custom CUDA kernels for per-element signal processing / 逐元素信号处理的自定义 CUDA 内核
- **PyTorch 2.5+** — GPU tensor ops and torch.fft (cuFFT backend) / GPU 张量运算与 FFT
- **Complex numbers / 复数表示**: Interleaved float32 (Warp lacks native complex64) / 交错 float32（Warp 无原生 complex64）

## Specs / 系统参数

| Parameter / 参数 | Value / 值 |
|-----------|-------|
| Radars / 雷达数量 | 4 × 25×25 phased arrays / 相控阵 |
| Bandwidth / 带宽 | 200 MHz (IQ-level / IQ 级) |
| Elements total / 总阵元数 | 4 × 625 = 2,500 |
| GPU memory / 显存占用 | ~1.1 GB / 6.4 GB (RTX 2060) |

---

## Bug Fixes / 缺陷修复

| Bug / 缺陷 | File / 文件 | Fix / 修复 |
|-----|------|-----|
| Steer kernel sign error / 导向核符号错误 | `array_gpu.py` | `-taper*sin(phase)` → `+taper*sin(phase)` (pattern peak was at -az / 方向图峰值偏移至 -az) |
| Channel delay direction / 信道延迟方向反转 | `channel_gpu.py` | `src = s + d_int` → `s - d_int` (was time-advance, not delay / 实现为时间超前而非延迟) |
| Missing TX directivity / 缺少发射空间指向性 | `interference_gpu.py` | Rewrote to use Friis link budget with antenna gains / 重写为 Friis 链路预算，加入天线增益 |
