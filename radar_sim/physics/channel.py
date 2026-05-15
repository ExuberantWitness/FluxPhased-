"""Propagation channel models: path loss, fading, ground reflection.

Supports monostatic radar returns, bistatic paths (for communication and jamming),
and mutual interference between arrays. Computes SNR/SINR at the signal level.
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


SPEED_OF_LIGHT = 299792458.0
BOLTZMANN = 1.380649e-23  # J/K


def free_space_path_loss(distance_m: float, frequency_hz: float) -> float:
    """Free-space path loss (linear, not dB)."""
    if distance_m <= 0:
        return 1.0
    wavelength = SPEED_OF_LIGHT / frequency_hz
    return (wavelength / (4.0 * np.pi * distance_m)) ** 2


def free_space_path_loss_db(distance_m: float, frequency_hz: float) -> float:
    """Free-space path loss in dB."""
    if distance_m <= 0:
        return 0.0
    return 20.0 * np.log10(4.0 * np.pi * distance_m * frequency_hz / SPEED_OF_LIGHT)


def two_ray_path_loss(
    distance_m: float, frequency_hz: float,
    ht_m: float = 3.0, hr_m: float = 3.0
) -> float:
    """Two-ray ground reflection path loss (linear)."""
    wavelength = SPEED_OF_LIGHT / frequency_hz
    d0 = 4.0 * np.pi * ht_m * hr_m / wavelength

    if distance_m < d0:
        # Short range: free-space
        return free_space_path_loss(distance_m, frequency_hz)
    else:
        # Long range: 4th power law (ground reflection)
        return (ht_m * hr_m / distance_m ** 2) ** 2


def rayleigh_fading(num_samples: int, doppler_spread: float, fs: float,
                    rng: Optional[np.random.Generator] = None) -> NDArray:
    """Generate time-correlated Rayleigh fading coefficients."""
    if rng is None:
        rng = np.random.default_rng()

    # Jakes' model: sum-of-sinusoids
    n_paths = 48
    alpha = (2.0 * np.pi * np.arange(1, n_paths // 4 + 1) - 0.5) / (n_paths / 2)
    fd_max = doppler_spread
    t = np.arange(num_samples) / fs

    h = np.zeros(num_samples, dtype=complex)
    for n in range(n_paths // 4):
        phi_n = rng.uniform(0, 2.0 * np.pi)
        psi_n = rng.uniform(0, 2.0 * np.pi)
        h += np.exp(1j * (2.0 * np.pi * fd_max * np.cos(alpha[n]) * t + phi_n))
        h += np.exp(1j * (2.0 * np.pi * fd_max * np.cos(alpha[n]) * t + psi_n))

    # Normalize: E[|h|^2] = 1
    h /= np.sqrt(2.0 * n_paths // 2)
    return h


def compute_snr(
    tx_power_w: float,
    tx_gain_db: float,
    rx_gain_db: float,
    distance_m: float,
    frequency_hz: float,
    bandwidth_hz: float,
    noise_figure_db: float = 5.0,
    system_loss_db: float = 3.0,
    rcs_dbsm: float = 20.0,
) -> float:
    """Compute received SNR (dB) using radar range equation (monostatic).

    SNR = P_t + G_t + G_r + 2*G_λ + σ_RCS - L_sys - N - 40*log10(R)

    Uses the full radar equation with RCS.
    """
    wavelength = SPEED_OF_LIGHT / frequency_hz

    # Noise power (dBm)
    noise_power_dbm = 10.0 * np.log10(BOLTZMANN * 290.0 * 1000.0)  # -174 dBm/Hz
    noise_power_dbm += 10.0 * np.log10(bandwidth_hz) + noise_figure_db

    # Signal power using radar range equation (dB)
    tx_power_dbm = 10.0 * np.log10(tx_power_w * 1000.0)

    # Path loss (monostatic, 2-way): standard radar equation form
    # Pr = Pt + 2G + σ + 20·log10(λ) - 30·log10(4π) - 40·log10(R) - Lsys
    path_loss_db = (
        30.0 * np.log10(4.0 * np.pi)
        + 40.0 * np.log10(distance_m)
        - 20.0 * np.log10(wavelength)
    )

    # RCS + processing gain
    signal_dbm = (
        tx_power_dbm
        + tx_gain_db
        + rx_gain_db
        + rcs_dbsm
        - path_loss_db
        - system_loss_db
    )

    snr_db = signal_dbm - noise_power_dbm
    return snr_db


def compute_sinr(
    snr_db: float,
    interference_power_dbm: float,
    noise_power_dbm: float,
) -> float:
    """Compute SINR (dB) given SNR and interference power.

    SINR = Signal / (Noise + sum(Interference))
    """
    noise_w = 10.0 ** (noise_power_dbm / 10.0) * 1e-3
    interference_w = 10.0 ** (interference_power_dbm / 10.0) * 1e-3
    signal_w = noise_w * 10.0 ** (snr_db / 10.0)

    sinr_linear = signal_w / (noise_w + interference_w)
    return 10.0 * np.log10(max(sinr_linear, 1e-15))


def compute_jamming_power(
    jammer_power_w: float,
    jammer_gain_db: float,
    victim_rx_gain_db: float,
    distance_m: float,
    frequency_hz: float,
    bandwidth_hz: float,
    freq_overlap: float = 1.0,  # fraction of bandwidth overlap
    polarization_loss_db: float = 3.0,
) -> float:
    """Compute jamming power received at victim radar (one-way path).

    Jammer → Victim receiver: J = P_j + G_j + G_rx - L_path - L_pol

    Returns jamming power in dBm.
    """
    wavelength = SPEED_OF_LIGHT / frequency_hz

    # One-way path loss
    path_loss_db = 20.0 * np.log10(4.0 * np.pi * distance_m / wavelength)

    jammer_power_dbm = 10.0 * np.log10(jammer_power_w * 1000.0)

    jamming_dbm = (
        jammer_power_dbm
        + jammer_gain_db
        + victim_rx_gain_db
        - path_loss_db
        - polarization_loss_db
        + 10.0 * np.log10(freq_overlap + 1e-15)
    )

    return jamming_dbm


def compute_communication_snr(
    tx_power_w: float,
    tx_gain_db: float,
    rx_gain_db: float,
    distance_m: float,
    frequency_hz: float,
    bandwidth_hz: float,
    noise_figure_db: float = 5.0,
    system_loss_db: float = 3.0,
) -> float:
    """Compute comm link SNR (one-way path, Friis equation).

    SNR = P_t + G_t + G_r - L_path - N - L_sys
    """
    path_loss_db = free_space_path_loss_db(distance_m, frequency_hz)

    noise_power_dbm = 10.0 * np.log10(BOLTZMANN * 290.0 * 1000.0)
    noise_power_dbm += 10.0 * np.log10(bandwidth_hz) + noise_figure_db

    tx_power_dbm = 10.0 * np.log10(tx_power_w * 1000.0)

    signal_dbm = (
        tx_power_dbm
        + tx_gain_db
        + rx_gain_db
        - path_loss_db
        - system_loss_db
    )

    return signal_dbm - noise_power_dbm


def albersheim_detection_probability(snr_db: float, pfa: float = 1e-6, n_pulses: int = 1) -> float:
    """Detection probability from SNR using Albersheim's empirical formula.

    Standard Albersheim equation (Proc. IEEE 69(7), July 1981):
      SNR = -5·log10(N) + [6.2 + 4.54/√N]·log10(A) + 5·log10(B)
            + (4.6/N)·log10(A/B + 0.44·N)
    where A = ln(0.62/Pfa), B = ln(Pd/(1-Pd))

    This inverts the formula to compute Pd from SNR via bisection.
    Valid for Swerling 0 (non-fluctuating) targets.
    """
    N = max(n_pulses, 1)
    A = np.log(0.62 / pfa)

    def snr_for_pd(pd):
        if pd <= 0.0 or pd >= 1.0:
            return 1e10
        B = np.log(pd / (1.0 - pd))
        return (
            -5.0 * np.log10(N)
            + (6.2 + 4.54 / np.sqrt(N)) * np.log10(A)
            + 5.0 * np.log10(B)
            + (4.6 / N) * np.log10(A / B + 0.44 * N)
        )

    # Bisection to invert: find Pd such that snr_for_pd(Pd) ≈ snr_db
    lo, hi = pfa, 1.0 - pfa
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if snr_for_pd(mid) < snr_db:
            lo = mid
        else:
            hi = mid
    return float(np.clip((lo + hi) / 2.0, pfa, 0.9999))


def shannon_capacity_bps(snr_linear: float, bandwidth_hz: float) -> float:
    """Shannon channel capacity (bps)."""
    if snr_linear <= 0:
        return 0.0
    return bandwidth_hz * np.log2(1.0 + snr_linear)


def detection_snr_required(pfa: float = 1e-6, pd: float = 0.9, n_pulses: int = 1) -> float:
    """Required SNR (dB) for given P_d and P_fa using Albersheim's formula.

    Standard Albersheim equation (Proc. IEEE 69(7), July 1981).
    """
    N = max(n_pulses, 1)
    A = np.log(0.62 / pfa)
    B = np.log(pd / (1.0 - pd))
    snr = (
        -5.0 * np.log10(N)
        + (6.2 + 4.54 / np.sqrt(N)) * np.log10(A)
        + 5.0 * np.log10(B)
        + (4.6 / N) * np.log10(A / B + 0.44 * N)
    )
    return float(snr)


class PropagationChannel:
    """Models signal propagation for radar, jamming, and communication paths."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def radar_path(
        self,
        tx_power_w: float,
        tx_gain_db: float,
        rx_gain_db: float,
        distance_m: float,
        frequency_hz: float,
        bandwidth_hz: float,
        rcs_dbsm: float,
        noise_figure_db: float = 5.0,
        system_loss_db: float = 3.0,
    ) -> dict:
        """Compute monostatic radar return parameters.

        Returns:
            dict with snr_db, path_loss_db, received_power_dbm, noise_power_dbm,
                 range_delay_s, doppler_shift_hz
        """
        path_loss_db = 40.0 * np.log10(4.0 * np.pi * distance_m * frequency_hz / SPEED_OF_LIGHT)

        noise_power_dbm = 10.0 * np.log10(BOLTZMANN * 290.0 * 1000.0)
        noise_power_dbm += 10.0 * np.log10(bandwidth_hz) + noise_figure_db

        snr_db = compute_snr(
            tx_power_w, tx_gain_db, rx_gain_db,
            distance_m, frequency_hz, bandwidth_hz,
            noise_figure_db, system_loss_db, rcs_dbsm,
        )

        tx_power_dbm = 10.0 * np.log10(tx_power_w * 1000.0)
        rx_power_dbm = tx_power_dbm + tx_gain_db + rx_gain_db + rcs_dbsm - path_loss_db - system_loss_db

        range_delay = 2.0 * distance_m / SPEED_OF_LIGHT

        return {
            "snr_db": snr_db,
            "path_loss_db": path_loss_db,
            "received_power_dbm": rx_power_dbm,
            "noise_power_dbm": noise_power_dbm,
            "range_delay_s": range_delay,
            "doppler_shift_hz": 0.0,  # caller updates based on radial velocity
        }

    def jamming_path(
        self,
        jammer_power_w: float,
        jammer_gain_db: float,
        victim_rx_gain_db: float,
        distance_m: float,
        frequency_hz: float,
        freq_overlap: float = 1.0,
    ) -> float:
        """Compute jamming power received at victim.

        Returns jamming power in dBm.
        """
        return compute_jamming_power(
            jammer_power_w, jammer_gain_db, victim_rx_gain_db,
            distance_m, frequency_hz,
            bandwidth_hz=1e6,  # placeholder
            freq_overlap=freq_overlap,
        )

    def comm_path(
        self,
        tx_power_w: float,
        tx_gain_db: float,
        rx_gain_db: float,
        distance_m: float,
        frequency_hz: float,
        bandwidth_hz: float,
        noise_figure_db: float = 5.0,
    ) -> dict:
        """Compute communication link parameters.

        Returns:
            dict with snr_db, path_loss_db, capacity_bps
        """
        snr_db = compute_communication_snr(
            tx_power_w, tx_gain_db, rx_gain_db,
            distance_m, frequency_hz, bandwidth_hz, noise_figure_db,
        )
        path_loss_db = free_space_path_loss_db(distance_m, frequency_hz)
        snr_linear = 10.0 ** (snr_db / 10.0)

        return {
            "snr_db": snr_db,
            "path_loss_db": path_loss_db,
            "capacity_bps": shannon_capacity_bps(snr_linear, bandwidth_hz),
            "propagation_delay_s": distance_m / SPEED_OF_LIGHT,
        }
