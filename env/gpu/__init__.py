"""GPU-accelerated IQ-level phased array radar simulation using NVIDIA Warp + PyTorch.

Modules:
  - array_gpu: Warp kernels for beam steering and array factor
  - channel_gpu: Warp kernels for per-element channel simulation
  - waveform_gpu: PyTorch GPU waveform generation
  - receiver_gpu: torch.fft + Warp CFAR for radar receive processing
  - interference_gpu: IQ-level cross-radar interference
  - pipeline_gpu: Full 4-radar simulation orchestrator
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.complex64

print(f"[gpu] Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[gpu] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[gpu] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
