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

Validated against analytical ground truth (closed-form radar physics formulas) and RadarSimPy v15.2.0 processing algorithms (range FFT, Doppler FFT, CA-CFAR).

对比解析真值（闭式雷达物理公式）与 RadarSimPy v15.2.0 信号处理算法（range FFT、Doppler FFT、CA-CFAR）进行校验。

```bash
python validation/validate_radarsimpy.py   # Ground truth + RadarSimPy processing
python validation/validate_precision.py    # CPU vs GPU internal consistency
```

### Results / 校验结果

| Test / 测试项 | Reference / 参考基准 | Result / 结果 |
|--------|--------|--------|
| Array factor (7 steer angles) / 阵列因子 | Analytical AF = sum(w*exp(jkru)) | **corr = 1.000000, max_err = 0.0024 dB** |
| Path loss (1–50 km) / 路径损耗 | Friis L = (4pid/lambda)^2 | **err = 0.000000 dB** |
| Radar SNR (4 ranges) / 雷达信噪比 | Analytical Pr = PtGtGr*lam^2*sigma/((4pi)^3*R^4*kTB) | **Linear = dB form (err < 0.01 dB)** |
| Matched filter / 匹配滤波 | Numpy FFT cross-correlation | **Peaks at exact delay positions / 精确延迟位置** |
| Range-Doppler map / 距离-多普勒图 | RadarSimPy doppler_fft | **Doppler bin exact match / 多普勒单元精确匹配** |
| CA-CFAR detection / CA-CFAR 检测 | RadarSimPy cfar_ca_2d | **Targets detected, noise rejected / 目标检出，噪声抑制** |
| Interference JNR / 干扰干噪比 | Analytical Friis link budget | **12/12 links validated, 0 dB path loss error** |
| Range resolution / 距离分辨率 | Theoretical dR = c/(2B) = 0.75 m | **Exact match / 精确一致** |
| Doppler velocity / 多普勒速度 | Analytical phase ramp | **err = 0.36 m/s (bin_res = 1.17 m/s)** |

### Interference JNR Matrix / 干扰 JNR 矩阵

4 radars at 2 km spacing, boresights pointing toward center / 四部雷达 2 km 间距，波束指向中心。

```
GPU + Analytical (dB):
  [  0.0, +14.7, +14.7, +87.3]
  [+14.7,   0.0, +87.3, +14.7]
  [+14.7, +87.3,   0.0, +14.7]
  [+87.3, +14.7, +14.7,   0.0]
```

Diagonal links (+87.3 dB) are boresight-to-boresight; side links (+14.7 dB) are sidelobe-to-sidelobe. All link budgets match Friis equation exactly.

对角链路（+87.3 dB）为波束正面耦合；侧链路（+14.7 dB）为旁瓣耦合。所有链路预算与 Friis 方程精确一致。

---

## Visualization / 可视化效果图

10 km 四雷达场景的 publication-quality 可视化，由 `validation/generate_plots.py` 生成。

6 张已完成的效果图位于 `validation/figures/`：

### 01 — Array Beam Pattern Overlay / 阵列方向图叠加
7 个指向角（0°, ±15°, ±30°, ±45°）的波束方向图，标注 3 dB 波束宽度与峰值方向性增益。
![01](validation/figures/01_array_pattern_overlay.png)

### 02 — Battlefield Top-Down View / 战场俯视图
10 km × 10 km 四雷达正方形部署，波束覆盖扇区，目标位置与距离标注。
![02](validation/figures/02_battlefield_topdown.png)

### 03 — Interference JNR Heatmap / 干扰干噪比热力图
4 × 4 互干扰 JNR 矩阵，基于 Friis 链路预算 + 阵列方向图增益计算 12 条链路。
![03](validation/figures/03_jnr_heatmap.png)

### 04 — Matched Filter Range Profile / 匹配滤波距离像
LFM 脉冲压缩，3 个目标（3 km / 6 km / 9 km），标注距离分辨率。
![04](validation/figures/04_matched_filter_range_profile.png)

### 08 — Waveform Comparison / 波形对比
6 种波形（LFM, Barker-13, Frank-16, Costas-16, NLFM, P4）的自相关（脉压）响应对比。
![08](validation/figures/08_waveform_comparison.png)

### 09 — JNR vs Distance / 干噪比-距离曲线
主瓣↔主瓣、旁瓣↔主瓣、旁瓣↔旁瓣三条 JNR 曲线，标注 10 km 工作范围与检测门限。
![09](validation/figures/09_jnr_vs_distance.png)

### 待完成 / In Progress
| Figure / 图号 | Content / 内容 | Status / 状态 |
|------|------|------|
| 05 | Range-Doppler Map / 距离-多普勒图 | Colormap scaling fix / 色标范围修正中 |
| 06 | CFAR Detection / CFAR 检测标记 | Colormap scaling fix / 色标范围修正中 |
| 07 | Interference Comparison / 干扰对比 | Colormap scaling fix / 色标范围修正中 |

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
