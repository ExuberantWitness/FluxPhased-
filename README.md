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
![01](validation/figures/01_array_pattern_overlay.png)

**条件：** 25×25 均匀矩形阵列（dx=dy=0.5λ），fc=10 GHz，分别指向 0°、±15°、±30°、±45° 共 7 个方位角，扫描范围 ±90°。

**说明的能力：** 系统能够在 GPU 上精确计算任意指向角下的阵列方向图（Warp 内核逐阵元相位叠加）。图中可观察到波束指向偏移时主瓣随动、旁瓣电平稳定在约 -13.3 dB（均匀加权理论值）、3 dB 波束宽度约 4.6° 均与解析公式一致，验证了波束形成与空间指向性建模的正确性。

**合理性：** 均匀矩形阵列的方向图具有可预测的主瓣/旁瓣结构，是相控阵雷达系统仿真的基础。该图证明系统在不同指向角下均能正确生成方向图，为后续干扰链路中的天线增益计算提供了可靠依据。

### 02 — Battlefield Top-Down View / 战场俯视图
![02](validation/figures/02_battlefield_topdown.png)

**条件：** 4 部雷达正方形部署于 (0,0)、(10km,0)、(0,10km)、(10km,10km)，波束分别指向 45°、135°、-45°、-135°（指向中心目标），目标位于 (5km, 5km, 0)，RCS=20 dBsm。

**说明的能力：** 系统能够在 10 km 量级的战术场景下进行多雷达部署与空间几何建模。波束覆盖扇区（±2.5° 锥形区域）清晰展示了各雷达的照射范围，4 条雷达-目标虚线连接标注了实际探测路径。这验证了系统在远距离场景下对雷达部署拓扑、波束指向与目标位置的空间关系建模能力。

**合理性：** 10 km 正方形间距、波束指向中心的部署方式符合典型的组网雷达协同探测场景。对角线距离 14.1 km 的标注也直接关联到互干扰链路预算的计算。

### 03 — Interference JNR Heatmap / 干扰干噪比热力图
![03](validation/figures/03_jnr_heatmap.png)

**条件：** 4 × 4 干扰链路矩阵，基于 Friis 传播方程 + 阵列方向图增益（含发射/接收波束指向性），TX=1 kW，fc=10 GHz，BW=200 MHz，噪声系数=5 dB。

**说明的能力：** 这是系统互干扰仿真的核心输出。12 条非对角线链路的 JNR（干噪比）由 GPU 上的 InterferenceEngine 计算得出，综合考虑了发射功率、收发阵列增益（含波束指向角偏移导致的旁瓣抑制）、自由空间路径损耗和接收机噪声。对角线链路（+87 dB）为波束正面耦合，侧链路（+14.7 dB）为旁瓣间耦合，量级差异反映了阵列空间选择性。

**合理性：** 10 km 间距下对角线链路的波束正面耦合 JNR 达 87 dB 是合理的——1 kW 发射功率经 25×25 阵列（~34 dBi）双程增益后，即使经过 14.1 km 自由空间衰减，残余功率仍远超接收机噪声底。旁瓣耦合仅 14.7 dB 则反映了阵列方向图在非主瓣方向的约 -20 dBi 抑制。这些数值直接验证了系统链路预算模型的物理正确性。

### 04 — Matched Filter Range Profile / 匹配滤波距离像
![04](validation/figures/04_matched_filter_range_profile.png)

**条件：** 10 μs LFM 脉冲（TB=2000），带宽 200 MHz，3 个点目标分别位于 3 km / 6 km / 9 km，幅度 0.9 / 0.5 / 0.3，叠加 AWGN 噪声（σ=0.01）。

**说明的能力：** 系统在 GPU 上通过 `torch.fft` 实现频域匹配滤波（脉压），正确完成发射波形与接收信号的互相关运算。图中 3 个目标峰精确出现在对应距离位置，脉压后的距离分辨率为 c/(2B)=0.75 m，与理论值一致。主旁瓣比约 13.3 dB（LFM 均匀加权）也符合理论预期。

**合理性：** 匹配滤波是雷达接收机的核心信号处理环节。该图证明系统的频域脉压实现正确，能够在多个目标同时存在的条件下分辨各目标的距离，且不引入虚假峰值或位置偏移。

### 08 — Waveform Comparison / 波形对比
![08](validation/figures/08_waveform_comparison.png)

**条件：** 6 种雷达波形的自相关（脉压）响应：LFM 线性调频、Barker-13 二相编码、Frank-16 多相编码、Costas-16 跳频、NLFM 非线性调频、P4 多相码，均在相同 TB 积（2000）条件下生成。

**说明的能力：** 系统的 `WaveformGeneratorGPU` 模块能够在 GPU 上生成多种典型雷达波形，并通过匹配滤波展示各自的脉压特性。图中清晰对比了不同波形的旁瓣结构差异：LFM 具有经典的 sinc 型旁瓣（-13.3 dB），Barker-13 具有均匀低旁瓣（-22.3 dB），Frank/Costas/P4 等编码波形展示了各自独特的旁瓣抑制特性。

**合理性：** 波形多样性是电子战与抗干扰研究的关键维度。该图证明系统不仅限于单一 LFM 波形，而是具备在统一框架下生成和评估多种波形脉压性能的能力，为后续波形选择与抗干扰策略研究奠定了基础。

### 09 — JNR vs Distance / 干噪比-距离曲线
![09](validation/figures/09_jnr_vs_distance.png)

**条件：** 发射功率 1 kW，25×25 阵列（峰值增益 34 dBi），旁瓣增益取 30° 偏轴方向值，扫描距离 0.5–14 km，fc=10 GHz。

**说明的能力：** 该图通过三条曲线（主瓣↔主瓣、旁瓣↔主瓣、旁瓣↔旁瓣）展示了互干扰强度随雷达间距的变化规律。主瓣对主瓣耦合在 1 km 处 JNR 超过 150 dB，即使在 14 km 处仍高于 100 dB——说明近距离主瓣对准是极端干扰场景。旁瓣对旁瓣在 10 km 工作范围处 JNR 约 0 dB，刚好处于检测门限临界点。

**合理性：** JNR 随距离以 20 dB/decade（1/d²）衰减，符合 Friis 自由空间传播规律。10 km 工作线处的 JNR 水平提示：旁瓣间干扰在实际工作距离上可能不会严重恶化检测性能，但主瓣耦合（如对角线部署）必须通过频率规划或波形正交化来规避。这为系统级电磁兼容设计提供了定量参考。

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
