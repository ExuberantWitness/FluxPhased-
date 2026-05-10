"""GPU waveform generation: LFM, Barker, Frank, Costas, NLFM, P-code families.

All outputs are torch tensors on GPU, complex64.
"""

import torch
import numpy as np
from ..config import RFConfig, WaveformConfig, CPIConfig

SPEED_OF_LIGHT = 299792458.0


def _to_tensor(arr, device):
    return torch.tensor(arr, dtype=torch.complex64, device=device)


def generate_lfm(pulse_width, bandwidth, fs, device, direction="up"):
    n = max(1, int(pulse_width * fs))
    t = torch.arange(n, dtype=torch.float32, device=device) / fs
    k = bandwidth / pulse_width
    sign = 1.0 if direction == "up" else -1.0
    phase = sign * np.pi * k * t ** 2
    signal = torch.exp(1j * phase)
    signal = signal / signal.norm()
    return signal


def generate_barker(n_bits, chip_width, fs, device):
    codes = {
        5: [1, 1, 1, -1, 1],
        7: [1, 1, 1, -1, -1, 1, -1],
        11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
        13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
    }
    code = torch.tensor(codes.get(n_bits, codes[13]), dtype=torch.float32, device=device)
    spc = max(1, int(chip_width * fs))
    signal = code.repeat_interleave(spc).to(torch.complex64)
    signal = signal / signal.norm()
    return signal


def generate_frank(n_phases, fs, pulse_width, device):
    n = max(1, int(pulse_width * fs))
    idx = torch.arange(n_phases, device=device)
    i, j = torch.meshgrid(idx, idx, indexing="ij")
    phases = 2.0 * np.pi / n_phases * i * j
    phase_seq = torch.exp(1j * phases.ravel())
    indices = torch.linspace(0, len(phase_seq) - 1, n, device=device).long()
    signal = phase_seq[indices]
    signal = signal / signal.norm()
    return signal


def generate_costas(n_freqs, pulse_width, fs, device):
    seqs = {
        4: [1, 3, 2, 4], 5: [1, 3, 4, 2, 5],
        6: [1, 3, 2, 6, 4, 5], 7: [1, 3, 2, 6, 4, 5, 7],
        16: [2, 5, 10, 4, 6, 13, 9, 16, 3, 8, 2, 11, 7, 14, 12, 1],
    }
    seq = seqs.get(n_freqs, seqs[4])
    n = max(1, int(pulse_width * fs))
    n_chips = len(seq)
    signal = torch.zeros(n, dtype=torch.complex64, device=device)
    chip_len = n // n_chips
    for ci, fi in enumerate(seq):
        s = ci * chip_len
        e = min(s + chip_len, n)
        t_chip = torch.arange(e - s, dtype=torch.float32, device=device) / fs
        freq = fi / pulse_width
        signal[s:e] = torch.exp(1j * 2 * np.pi * freq * t_chip)
    signal = signal / signal.norm()
    return signal


def generate_nlfm(pulse_width, bandwidth, fs, device):
    n = max(1, int(pulse_width * fs))
    t = torch.arange(n, dtype=torch.float32, device=device) / fs
    k = bandwidth / pulse_width
    phase = np.pi * k * t ** 2 + 0.3 * np.pi * k / pulse_width * t ** 3
    signal = torch.exp(1j * phase)
    signal = signal / signal.norm()
    return signal


def generate_p4(n_stages, pulse_width, fs, device):
    n_pts = n_stages * n_stages
    n = max(1, int(pulse_width * fs))
    k = torch.arange(n_pts, dtype=torch.float32, device=device)
    phases = np.pi * k ** 2 / n_pts - np.pi * k
    phase_seq = torch.exp(1j * phases)
    indices = torch.linspace(0, n_pts - 1, n, device=device).long()
    signal = phase_seq[indices]
    signal = signal / signal.norm()
    return signal


class WaveformGeneratorGPU:
    def __init__(self, rf: RFConfig, wf_cfg: WaveformConfig, cpi: CPIConfig, device):
        self.rf = rf
        self.wf_cfg = wf_cfg
        self.cpi = cpi
        self.device = device

    def generate(self, waveform_type, params=None):
        params = params or {}
        fs = self.rf.fs
        pw = params.get("pulse_width", 50e-6)
        bw = params.get("bandwidth", self.rf.bandwidth)
        dev = self.device

        gens = {
            "lfm_up": lambda: generate_lfm(pw, bw, fs, dev, "up"),
            "lfm_down": lambda: generate_lfm(pw, bw, fs, dev, "down"),
            "barker_13": lambda: generate_barker(13, pw / 13, fs, dev),
            "frank_16": lambda: generate_frank(4, fs, pw, dev),
            "costas_16": lambda: generate_costas(4, pw, fs, dev),
            "nlfm": lambda: generate_nlfm(pw, bw, fs, dev),
            "p4_code": lambda: generate_p4(4, pw, fs, dev),
        }
        return gens.get(waveform_type, gens["lfm_up"])()

    def matched_filter(self, waveform):
        return torch.conj(waveform.flip(-1))
