"""Waveform generation and pulse-level signal modeling.

Supports 8 waveform types with full pulse-compression matched filter generation.
Outputs complex baseband samples compatible with RadarSimPy's signal format.
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from ..config import WaveformConfig, RFConfig, CPIConfig


SPEED_OF_LIGHT = 299792458.0


def generate_lfm(
    pulse_width: float, bandwidth: float, fs: float, direction: str = "up"
) -> Tuple[NDArray, NDArray]:
    """Generate linear FM chirp.

    Returns (time_samples, complex_baseband_signal).
    """
    num_samples = max(1, int(pulse_width * fs))
    t = np.arange(num_samples) / fs
    k = bandwidth / pulse_width  # chirp rate (Hz/s)

    if direction == "up":
        phase = np.pi * k * t ** 2
    else:
        phase = -np.pi * k * t ** 2

    signal = np.exp(1j * phase)
    # Normalize to unit energy
    signal /= np.sqrt(np.sum(np.abs(signal) ** 2))
    return t, signal.astype(complex)


def generate_barker(n_bits: int, chip_width: float, fs: float) -> NDArray:
    """Generate Barker-coded pulse."""
    barker_codes = {
        5: np.array([1, 1, 1, -1, 1]),
        7: np.array([1, 1, 1, -1, -1, 1, -1]),
        11: np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]),
        13: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]),
    }
    code = barker_codes.get(n_bits, barker_codes[13])

    samples_per_chip = max(1, int(chip_width * fs))
    signal = np.repeat(code, samples_per_chip).astype(complex)
    signal /= np.sqrt(np.sum(np.abs(signal) ** 2))
    return signal


def generate_frank(n_phases: int, fs: float, pulse_width: float) -> NDArray:
    """Generate Frank polyphase code."""
    num_samples = max(1, int(pulse_width * fs))
    # Frank code: phase(i,j) = 2π/M * i * j
    phases = np.zeros((n_phases, n_phases))
    for i in range(n_phases):
        for j in range(n_phases):
            phases[i, j] = 2.0 * np.pi / n_phases * i * j
    phase_seq = np.exp(1j * phases.ravel())

    # Resample to desired number of samples
    indices = np.linspace(0, len(phase_seq) - 1, num_samples).astype(int)
    signal = phase_seq[indices]
    signal /= np.sqrt(np.sum(np.abs(signal) ** 2))
    return signal.astype(complex)


def generate_costas(n_freqs: int, pulse_width: float, fs: float) -> NDArray:
    """Generate Costas frequency-coded waveform."""
    num_samples = max(1, int(pulse_width * fs))
    # Use a Costas sequence (e.g., for n=4: [1,3,2,4], for n=5: [1,3,4,2,5])
    costas_sequences = {
        4: np.array([1, 3, 2, 4]),
        5: np.array([1, 3, 4, 2, 5]),
        6: np.array([1, 3, 2, 6, 4, 5]),
        7: np.array([1, 3, 2, 6, 4, 5, 7]),
        16: np.array([2, 5, 10, 4, 6, 13, 9, 16, 3, 8, 2, 11, 7, 14, 12, 1]),
    }
    seq = costas_sequences.get(n_freqs, costas_sequences[4])
    n = len(seq)

    t = np.linspace(0, pulse_width, num_samples)
    step = pulse_width / n
    signal = np.zeros(num_samples, dtype=complex)

    for i, freq_idx in enumerate(seq):
        t_start = int(i * num_samples / n)
        t_end = int((i + 1) * num_samples / n)
        t_chip = t[t_start:t_end] - t[t_start]
        freq = freq_idx / pulse_width
        signal[t_start:t_end] = np.exp(1j * 2.0 * np.pi * freq * t_chip)

    signal /= np.sqrt(np.sum(np.abs(signal) ** 2))
    return signal.astype(complex)


def generate_nlfm(pulse_width: float, bandwidth: float, fs: float) -> NDArray:
    """Generate non-linear FM (Taylor-weighted spectrum shaping)."""
    num_samples = max(1, int(pulse_width * fs))
    t = np.arange(num_samples) / fs

    # Use a cubic frequency modulation for NLFM
    k = bandwidth / pulse_width
    # Add cubic term for spectral shaping (reduces sidelobes without amplitude weighting)
    phase = np.pi * k * t ** 2 + 0.3 * np.pi * k / pulse_width * t ** 3
    signal = np.exp(1j * phase)
    signal /= np.sqrt(np.sum(np.abs(signal) ** 2))
    return signal.astype(complex)


def generate_p_code(n_stages: int, pulse_width: float, fs: float) -> NDArray:
    """Generate P1/P4 polyphase code."""
    num_samples = max(1, int(pulse_width * fs))
    # P1 code
    m = np.arange(n_stages)
    phases = np.zeros((n_stages, n_stages))
    for i in range(n_stages):
        for j in range(n_stages):
            # P1: phase = (π/M)*((M-(2j-1))*((j-1)*M+(i-1)))
            phases[i, j] = (np.pi / n_stages) * (n_stages - (2 * j + 1)) * (j * n_stages + i)
    phase_seq = np.exp(1j * phases.ravel())

    indices = np.linspace(0, len(phase_seq) - 1, num_samples).astype(int)
    signal = phase_seq[indices]
    signal /= np.sqrt(np.sum(np.abs(signal) ** 2))
    return signal.astype(complex)


def generate_p4_code(n_stages: int, pulse_width: float, fs: float) -> NDArray:
    """Generate P4 polyphase code."""
    num_samples = max(1, int(pulse_width * fs))
    n = n_stages * n_stages
    k = np.arange(n)
    # P4: phase(k) = π*k²/N - π*k
    phases = (np.pi * k ** 2 / n) - np.pi * k
    phase_seq = np.exp(1j * phases)

    indices = np.linspace(0, len(phase_seq) - 1, num_samples).astype(int)
    signal = phase_seq[indices]
    signal /= np.sqrt(np.sum(np.abs(signal) ** 2))
    return signal.astype(complex)


class WaveformGenerator:
    """Generates radar waveforms and matched filters for pulse compression."""

    def __init__(self, rf: RFConfig, wf_cfg: WaveformConfig, cpi: CPIConfig):
        self.rf = rf
        self.wf_cfg = wf_cfg
        self.cpi = cpi

    def generate(
        self,
        waveform_type: str,
        params: Optional[dict] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Generate transmit waveform.

        Args:
            waveform_type: One of the 8 supported types.
            params: Dict with optional overrides for pulse_width, bandwidth, etc.

        Returns:
            (time_samples, complex_baseband_waveform)
        """
        fs = self.rf.fs
        pulse_width = (params or {}).get("pulse_width", 50e-6)
        bandwidth = (params or {}).get("bandwidth", self.rf.bandwidth)

        generators = {
            "lfm_up": lambda: generate_lfm(pulse_width, bandwidth, fs, "up")[1],
            "lfm_down": lambda: generate_lfm(pulse_width, bandwidth, fs, "down")[1],
            "barker_13": lambda: generate_barker(13, pulse_width / 13, fs),
            "frank_16": lambda: generate_frank(4, fs, pulse_width),
            "costas_16": lambda: generate_costas(4, pulse_width, fs),
            "nlfm": lambda: generate_nlfm(pulse_width, bandwidth, fs),
            "p1_code": lambda: generate_p_code(4, pulse_width, fs),
            "p4_code": lambda: generate_p4_code(4, pulse_width, fs),
        }

        gen = generators.get(waveform_type, generators["lfm_up"])
        signal = gen()

        num_samples = len(signal)
        t = np.arange(num_samples) / fs
        return t, signal

    def matched_filter(self, waveform: NDArray) -> NDArray:
        """Generate matched filter (time-reversed conjugate)."""
        return np.conj(waveform[::-1])

    def ambiguity_function(
        self, waveform: NDArray, delays: NDArray, dopplers: NDArray
    ) -> NDArray:
        """Compute ambiguity function χ(τ, fd)."""
        n = len(waveform)
        af = np.zeros((len(dopplers), len(delays)), dtype=complex)

        for i, fd in enumerate(dopplers):
            doppler_shifted = waveform * np.exp(1j * 2.0 * np.pi * fd * np.arange(n) / self.rf.fs)
            mf = np.conj(doppler_shifted[::-1])
            corr = np.convolve(waveform, mf, mode="same")
            # Resample to delays grid
            center = len(corr) // 2
            for j, d in enumerate(delays):
                idx = center + d
                if 0 <= idx < len(corr):
                    af[i, j] = corr[idx]

        return np.abs(af) / np.max(np.abs(af))

    def generate_pulse_train(
        self,
        waveform_type: str,
        params: Optional[dict] = None,
    ) -> NDArray:
        """Generate a train of pulses for one CPI.

        Returns: [num_pulses, num_samples] complex baseband.
        """
        pulse_width = (params or {}).get("pulse_width", 50e-6)
        pri = (params or {}).get("pri", 1.0 / self.cpi.prf)
        num_pulses = self.cpi.pulses_per_cpi

        # Generate single pulse
        _, pulse = self.generate(waveform_type, params)
        pulse_samples = len(pulse)

        # Samples per PRI
        samples_per_pri = max(1, int(pri * self.rf.fs))

        pulse_train = np.zeros((num_pulses, samples_per_pri), dtype=complex)
        for p in range(num_pulses):
            if pulse_samples <= samples_per_pri:
                pulse_train[p, :pulse_samples] = pulse
            else:
                pulse_train[p, :] = pulse[:samples_per_pri]

        return pulse_train


def compute_range_resolution(bandwidth: float) -> float:
    """Range resolution (m) for given bandwidth."""
    return SPEED_OF_LIGHT / (2.0 * bandwidth)


def compute_velocity_resolution(fc: float, cpi_duration: float) -> float:
    """Velocity resolution (m/s) for given carrier and CPI duration."""
    wavelength = SPEED_OF_LIGHT / fc
    return wavelength / (2.0 * cpi_duration)


def compute_max_unambiguous_range(pri: float) -> float:
    """Maximum unambiguous range for given PRI."""
    return SPEED_OF_LIGHT * pri / 2.0


def compute_max_unambiguous_velocity(fc: float, pri: float) -> float:
    """Maximum unambiguous velocity for given carrier and PRI."""
    wavelength = SPEED_OF_LIGHT / fc
    return wavelength / (4.0 * pri)
