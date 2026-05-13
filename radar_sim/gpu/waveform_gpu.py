"""GPU waveform generation: LFM, Barker, Frank, Costas, NLFM, P-code families,
BPSK modulation/demodulation, noise jamming, DRFM retransmission.

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


# ---------------------------------------------------------------------------
# BPSK modulation / demodulation (communication waveform)
# ---------------------------------------------------------------------------

def encode_bpsk(data_x: float, data_y: float, n_bits: int = 14,
                device="cpu") -> torch.Tensor:
    """Encode two floats into 32-bit BPSK payload (14+14 bits + 4-bit CRC).

    Layout: [X:14bits | Y:14bits | CRC:4bits]
    Args:
        data_x, data_y: values in [-1, 1], linearly mapped to 14-bit unsigned.
        device: torch device for output tensor.
    Returns:
        bits: [32] float32 tensor with values {0, 1}
    """
    x_int = int(max(0, min(2**14 - 1, (data_x + 1.0) / 2.0 * (2**14 - 1))))
    y_int = int(max(0, min(2**14 - 1, (data_y + 1.0) / 2.0 * (2**14 - 1))))
    data_28 = (x_int << 14) | y_int
    crc = 0
    val = data_28
    for _ in range(7):
        crc ^= (val & 0xF)
        val >>= 4
    word = (x_int << 18) | (y_int << 4) | (crc & 0xF)
    bits = torch.zeros(32, dtype=torch.float32, device=device)
    for i in range(32):
        bits[i] = float((word >> (31 - i)) & 1)
    return bits


def decode_bpsk(bits: torch.Tensor):
    """Decode 32-bit BPSK payload back to (data_x, data_y).

    Layout: [X:14bits | Y:14bits | CRC:4bits]
    Returns:
        (data_x, data_y) floats in [-1, 1], or (0.0, 0.0) on CRC failure.
    """
    if bits.numel() < 32:
        return 0.0, 0.0
    b = (bits[:32] > 0.5).int()
    word = 0
    for i in range(32):
        word = (word << 1) | int(b[i].item())
    # Extract fields
    x_int = (word >> 18) & ((1 << 14) - 1)
    y_int = (word >> 4) & ((1 << 14) - 1)
    crc_received = word & 0xF
    # CRC over X(14) + Y(14) = 28 bits = 7 nibbles
    data_28 = (x_int << 14) | y_int
    crc_computed = 0
    val = data_28
    for _ in range(7):
        crc_computed ^= (val & 0xF)
        val >>= 4
    if (crc_computed & 0xF) != crc_received:
        return 0.0, 0.0
    data_x = x_int / (2**14 - 1) * 2.0 - 1.0
    data_y = y_int / (2**14 - 1) * 2.0 - 1.0
    return data_x, data_y


def modulate_bpsk(bits: torch.Tensor, n_samples: int, fs: float,
                  symbol_rate: float, device) -> torch.Tensor:
    """BPSK modulate a bit sequence into baseband IQ waveform.

    Args:
        bits: [n_bits] float32 tensor with values {0, 1}
        n_samples: total output length
        fs: sampling rate (Hz)
        symbol_rate: symbols per second (Hz)
        device: torch device
    Returns:
        [n_samples] complex64 BPSK waveform
    """
    n_bits = bits.shape[0]
    samples_per_symbol = max(1, int(fs / symbol_rate))
    n = n_samples
    symbols = (2.0 * bits - 1.0).to(device=device, dtype=torch.complex64)
    # Upsample: repeat each symbol
    signal = symbols.repeat_interleave(samples_per_symbol)[:n]
    if signal.shape[0] < n:
        pad = torch.zeros(n - signal.shape[0], dtype=torch.complex64, device=signal.device)
        signal = torch.cat([signal, pad])
    norm = signal.norm()
    if norm > 0:
        signal = signal / norm
    return signal


def demodulate_bpsk(received: torch.Tensor, symbol_rate: float,
                    fs: float, n_bits: int = 32) -> torch.Tensor:
    """BPSK demodulate received IQ waveform to bits.

    Args:
        received: [n_samples] complex64 baseband after matched filtering
        symbol_rate: symbols per second
        fs: sampling rate
        n_bits: number of bits to decode
    Returns:
        [n_bits] float32 tensor with values {0, 1}
    """
    sps = max(1, int(fs / symbol_rate))
    # Sample at center of each symbol period
    indices = torch.arange(n_bits, device=received.device) * sps + sps // 2
    indices = indices.clamp(max=received.shape[0] - 1)
    symbols = received[indices]
    # Hard decision: Re > 0 → bit 1, else bit 0
    bits = (symbols.real > 0).float()
    return bits


def demodulate_bpsk_batch(received: torch.Tensor, symbol_rate: float,
                          fs: float, n_bits: int = 32) -> torch.Tensor:
    """Batched BPSK demodulate for [E, S] complex64 → [E, n_bits] float32."""
    sps = max(1, int(fs / symbol_rate))
    indices = torch.arange(n_bits, device=received.device) * sps + sps // 2
    indices = indices.clamp(max=received.shape[-1] - 1)
    # received: [E, S] → gather at indices → [E, n_bits]
    symbols = received[:, indices]
    bits = (symbols.real > 0).float()
    return bits


def decode_bpsk_batch(bits: torch.Tensor):
    """Vectorized BPSK decode for [E, 32] bits → (data_x, data_y, crc_ok).

    Returns:
        data_x: [E] float32 in [-1, 1], 0 on CRC fail
        data_y: [E] float32 in [-1, 1], 0 on CRC fail
        crc_ok: [E] bool
    """
    # Use long arithmetic to avoid float32 precision loss on 32-bit words
    powers = 2 ** torch.arange(31, -1, -1, dtype=torch.long, device=bits.device)
    words = (bits.long() * powers).sum(dim=-1)  # [E] long

    x_int = (words >> 18) & ((1 << 14) - 1)
    y_int = (words >> 4) & ((1 << 14) - 1)
    crc_received = words & 0xF

    data_28 = (x_int << 14) | y_int
    crc_computed = torch.zeros_like(data_28)
    for shift in range(7):
        crc_computed ^= ((data_28 >> (shift * 4)) & 0xF)
    crc_ok = (crc_computed & 0xF) == crc_received

    data_x = x_int.float() / (2**14 - 1) * 2.0 - 1.0
    data_y = y_int.float() / (2**14 - 1) * 2.0 - 1.0
    data_x = data_x * crc_ok.float()
    data_y = data_y * crc_ok.float()

    return data_x, data_y, crc_ok


# ---------------------------------------------------------------------------
# Noise jamming waveforms
# ---------------------------------------------------------------------------

def generate_noise_broadband(n_samples: int, power: float, device) -> torch.Tensor:
    """Broadband noise jamming waveform.

    Args:
        n_samples: output length
        power: relative power factor [0, 1]
        device: torch device
    Returns:
        [n_samples] complex64 noise waveform
    """
    signal = torch.randn(n_samples, dtype=torch.complex64, device=device)
    signal = signal / signal.norm() * (power ** 0.5)
    return signal


def generate_noise_spot(n_samples: int, center_freq: float, bandwidth: float,
                        fs: float, power: float, device) -> torch.Tensor:
    """Spot (narrowband) noise jamming centered at a frequency.

    Args:
        n_samples: output length
        center_freq: center frequency offset from carrier (Hz)
        bandwidth: noise bandwidth (Hz)
        fs: sampling rate
        power: relative power factor [0, 1]
        device: torch device
    Returns:
        [n_samples] complex64 narrowband noise
    """
    noise = torch.randn(n_samples, dtype=torch.complex64, device=device)
    # Filter to desired bandwidth via frequency domain
    spectrum = torch.fft.fft(noise)
    freqs = torch.fft.fftfreq(n_samples, 1.0 / fs, device=device)
    mask = (torch.abs(freqs - center_freq) < bandwidth / 2.0).float()
    spectrum = spectrum * mask
    signal = torch.fft.ifft(spectrum)
    norm = signal.norm()
    if norm > 0:
        signal = signal / norm * (power ** 0.5)
    return signal


def generate_drfm(captured: torch.Tensor, freq_shift: float, fs: float,
                  delay_samples: int = 0) -> torch.Tensor:
    """DRFM: frequency-shifted retransmission of captured signal.

    Args:
        captured: [n_samples] complex64 captured enemy signal
        freq_shift: frequency offset to apply (Hz)
        fs: sampling rate
        delay_samples: number of samples to delay (0 = no delay)
    Returns:
        [n_samples] complex64 retransmitted signal
    """
    n = captured.shape[0]
    t = torch.arange(n, dtype=torch.float32, device=captured.device) / fs
    shifted = captured * torch.exp(1j * 2.0 * np.pi * freq_shift * t)
    if delay_samples > 0 and delay_samples < n:
        shifted = torch.cat([
            torch.zeros(delay_samples, dtype=torch.complex64, device=captured.device),
            shifted[:-delay_samples],
        ])
    norm = shifted.norm()
    if norm > 0:
        shifted = shifted / norm
    return shifted


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
        n_samples = params.get("n_samples", max(1, int(pw * fs)))

        gens = {
            "lfm_up": lambda: generate_lfm(pw, bw, fs, dev, "up"),
            "lfm_down": lambda: generate_lfm(pw, bw, fs, dev, "down"),
            "barker_13": lambda: generate_barker(13, pw / 13, fs, dev),
            "frank_16": lambda: generate_frank(4, fs, pw, dev),
            "costas_16": lambda: generate_costas(4, pw, fs, dev),
            "nlfm": lambda: generate_nlfm(pw, bw, fs, dev),
            "p4_code": lambda: generate_p4(4, pw, fs, dev),
            "noise_broadband": lambda: generate_noise_broadband(
                n_samples, params.get("power", 1.0), dev),
            "noise_spot": lambda: generate_noise_spot(
                n_samples, params.get("center_freq", 0.0), bw * 0.1,
                fs, params.get("power", 1.0), dev),
        }
        return gens.get(waveform_type, gens["lfm_up"])()

    def matched_filter(self, waveform):
        return torch.conj(waveform.flip(-1))
