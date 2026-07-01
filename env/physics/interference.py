"""Inter-system interference and jamming analysis.

Computes mutual interference between all 4 radar arrays based on:
- Beam pointing directions
- Frequency channel overlap
- Spatial separation
- Transmit power and antenna gains
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .channel import compute_jamming_power, free_space_path_loss_db


class InterferenceEngine:
    """Computes interference between all radar systems in the environment."""

    def __init__(self):
        pass

    def compute_pairwise(
        self,
        interferer: dict,   # {pos, heading, array_az, tx_power_w, tx_gain_db, freq_hz, bandwidth_hz}
        victim: dict,       # {pos, heading, array_az, rx_gain_db, freq_hz, bandwidth_hz, noise_figure_db}
        beam_model,         # callable beam_model(az_deg, el_deg) -> gain_db
    ) -> dict:
        """Compute interference from one radar to another.

        Returns dict with jamming_power_dbm, jnr_db, effective_gain_db.
        """
        # Relative geometry
        dx = victim["pos"][0] - interferer["pos"][0]
        dy = victim["pos"][1] - interferer["pos"][1]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        if distance < 1.0:
            distance = 1.0  # avoid singularity

        # Angle from interferer to victim (in global frame)
        az_to_victim = np.degrees(np.arctan2(dy, dx))
        # Angle relative to interferer's array boresight
        interferer_boresight = interferer["array_az"]  # array rotation in global frame
        rel_az_interferer = az_to_victim - interferer_boresight
        rel_az_interferer = ((rel_az_interferer + 180) % 360) - 180  # wrap to [-180, 180]

        # Effective transmit gain in victim direction
        if callable(beam_model):
            tx_gain_effective = beam_model(rel_az_interferer, 0.0)
        else:
            tx_gain_effective = interferer.get("tx_gain_db", 30.0)

        # Angle from victim to interferer (for receive gain)
        az_to_interferer = np.degrees(np.arctan2(-dy, -dx))
        victim_boresight = victim["array_az"]
        rel_az_victim = az_to_interferer - victim_boresight
        rel_az_victim = ((rel_az_victim + 180) % 360) - 180

        if callable(beam_model):
            rx_gain_effective = beam_model(rel_az_victim, 0.0)
        else:
            rx_gain_effective = victim.get("rx_gain_db", 30.0)

        # Frequency overlap
        f1 = interferer["freq_hz"]
        bw1 = interferer["bandwidth_hz"]
        f2 = victim["freq_hz"]
        bw2 = victim["bandwidth_hz"]

        freq_overlap = _compute_frequency_overlap(f1, bw1, f2, bw2)

        # Jamming power at victim
        jamming_dbm = compute_jamming_power(
            jammer_power_w=interferer["tx_power_w"],
            jammer_gain_db=tx_gain_effective,
            victim_rx_gain_db=rx_gain_effective,
            distance_m=distance,
            frequency_hz=(f1 + f2) / 2.0,
            bandwidth_hz=victim["bandwidth_hz"],
            freq_overlap=freq_overlap,
        )

        # Noise floor of victim
        BOLTZMANN = 1.380649e-23
        noise_power_dbm = 10.0 * np.log10(BOLTZMANN * 290.0 * 1000.0)
        noise_power_dbm += 10.0 * np.log10(victim["bandwidth_hz"])
        noise_power_dbm += victim.get("noise_figure_db", 5.0)

        # JNR (jam-to-noise ratio) at victim receiver
        jnr_db = jamming_dbm - noise_power_dbm

        return {
            "jamming_power_dbm": jamming_dbm,
            "jnr_db": jnr_db,
            "tx_gain_effective_db": tx_gain_effective,
            "rx_gain_effective_db": rx_gain_effective,
            "distance_m": distance,
            "freq_overlap": freq_overlap,
            "rel_az_interferer_deg": rel_az_interferer,
            "rel_az_victim_deg": rel_az_victim,
        }

    def compute_full_interference(
        self,
        radar_states: List[dict],
        beam_models: List,
    ) -> NDArray:
        """Compute interference matrix between all radars.

        Args:
            radar_states: List of per-radar state dicts
            beam_models: List of per-radar beam model callables

        Returns:
            interference_matrix: [n_radar, n_radar] where [i,j] is JNR (dB) from i→j
        """
        n = len(radar_states)
        mat = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                result = self.compute_pairwise(
                    radar_states[i], radar_states[j], beam_models[i]
                )
                mat[i, j] = result["jnr_db"]

        return mat


def _compute_frequency_overlap(f1: float, bw1: float, f2: float, bw2: float) -> float:
    """Compute fraction of victim bandwidth overlapped by interferer.

    Returns [0, 1] where 1 = complete overlap.
    """
    lo1, hi1 = f1 - bw1 / 2, f1 + bw1 / 2
    lo2, hi2 = f2 - bw2 / 2, f2 + bw2 / 2

    overlap_lo = max(lo1, lo2)
    overlap_hi = min(hi1, hi2)

    if overlap_lo >= overlap_hi:
        return 0.0

    overlap_bw = overlap_hi - overlap_lo
    return min(overlap_bw / bw2, 1.0)


def compute_spectrum_occupancy(
    radar_states: List[dict],
    freq_range: Tuple[float, float],
    n_channels: int = 64,
) -> NDArray:
    """Compute spectrum occupancy across all radars.

    Returns array [n_channels] with power level per channel (dBm).
    """
    f_min, f_max = freq_range
    channel_edges = np.linspace(f_min, f_max, n_channels + 1)
    channel_centers = (channel_edges[:-1] + channel_edges[1:]) / 2.0

    occupancy = np.full(n_channels, -120.0)  # noise floor dBm

    for rs in radar_states:
        fc = rs["freq_hz"]
        bw = rs["bandwidth_hz"]
        power_dbm = 10.0 * np.log10(rs["tx_power_w"] * 1000.0) + rs.get("tx_gain_db", 30.0)

        for i, (lo, hi) in enumerate(zip(channel_edges[:-1], channel_edges[1:])):
            if fc - bw / 2 < hi and fc + bw / 2 > lo:
                # Interferer occupies this channel
                overlap = _compute_frequency_overlap(fc, bw, channel_centers[i], hi - lo)
                if overlap > 0:
                    channel_power = power_dbm + 10.0 * np.log10(overlap + 1e-15)
                    occupancy[i] = _power_sum_dbm(occupancy[i], channel_power)

    return occupancy


def _power_sum_dbm(p1_dbm: float, p2_dbm: float) -> float:
    """Sum two powers in dBm."""
    if p1_dbm < -200:
        return p2_dbm
    if p2_dbm < -200:
        return p1_dbm
    p1 = 10.0 ** (p1_dbm / 10.0)
    p2 = 10.0 ** (p2_dbm / 10.0)
    return 10.0 * np.log10(p1 + p2)
