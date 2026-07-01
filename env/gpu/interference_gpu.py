"""GPU-accelerated cross-radar interference simulation.

Computes the interfering IQ signal that each radar receives from every
other radar using the radar equation link budget:

  P_rx = P_tx + G_tx(θ) + G_rx(θ') - L_path - L_pol

For 4 radars: 4×4 - 4 = 12 interference links.
IQ signals are generated with correct amplitude from the link budget,
then distributed across the victim's antenna elements for downstream
receiver processing (matched filter, CFAR, etc.).
"""

import numpy as np
import torch
from typing import Dict, List

from .array_gpu import PhasedArrayGPU
from .channel_gpu import ChannelGPU

SPEED_OF_LIGHT = 299792458.0
POLARIZATION_LOSS_DB = 3.0


class InterferenceEngineGPU:
    """IQ-level mutual interference between phased array radars on GPU.

    For each pair (i, j) where i != j:
    1. Compute link budget: TX gain + RX gain - path loss
    2. Generate interference IQ signal with correct amplitude
    3. Apply propagation delay (sample shift)
    4. Distribute across victim's antenna elements
    """

    def __init__(
        self,
        arrays: Dict[int, PhasedArrayGPU],
        channel: ChannelGPU,
        n_radars: int = 4,
        device: str = "cuda",
    ):
        self.arrays = arrays
        self.channel = channel
        self.n_radars = n_radars
        self.device = device

    def compute_frequency_overlap(
        self,
        freq1_hz: float, bw1_hz: float,
        freq2_hz: float, bw2_hz: float,
    ) -> float:
        """Fraction of victim bandwidth overlapped by interferer."""
        lo1, hi1 = freq1_hz - bw1_hz / 2, freq1_hz + bw1_hz / 2
        lo2, hi2 = freq2_hz - bw2_hz / 2, freq2_hz + bw2_hz / 2
        overlap_lo = max(lo1, lo2)
        overlap_hi = min(hi1, hi2)
        if overlap_lo >= overlap_hi:
            return 0.0
        return min((overlap_hi - overlap_lo) / bw2_hz, 1.0)

    def _compute_link_geometry(self, tx_pos, rx_pos):
        """Compute distance and relative angles between two radars."""
        dx = rx_pos[0] - tx_pos[0]
        dy = rx_pos[1] - tx_pos[1]
        distance = float(np.sqrt(dx ** 2 + dy ** 2))
        if distance < 1.0:
            distance = 1.0
        # Global azimuth from TX to RX
        az_global_tx_to_rx = np.degrees(np.arctan2(dy, dx))
        # Global azimuth from RX to TX (opposite direction)
        az_global_rx_to_tx = np.degrees(np.arctan2(-dy, -dx))
        return distance, az_global_tx_to_rx, az_global_rx_to_tx

    def _relative_angle(self, az_global, boresight_deg):
        """Compute angle relative to array boresight, wrapped to [-180, 180]."""
        rel = az_global - boresight_deg
        return ((rel + 180) % 360) - 180

    def compute_interference_matrix(
        self,
        radar_states: List[dict],
        waveforms: Dict[int, torch.Tensor],
        n_samples: int,
    ) -> Dict[int, torch.Tensor]:
        """Compute IQ-level interference at each radar from all others.

        Uses the radar equation link budget to set correct signal amplitude,
        then applies propagation delay and distributes across RX elements.

        Args:
            radar_states: List of per-radar state dicts with keys:
                - pos: [3] position in meters
                - vel: [3] velocity in m/s
                - freq_hz: carrier frequency
                - bandwidth_hz: signal bandwidth
                - tx_power_w: transmit power
                - array_az_deg: array boresight azimuth
            waveforms: {radar_id: [n_samples] complex64 baseband waveform}
            n_samples: number of IQ samples per pulse
        Returns:
            {victim_id: [n_elem, n_samples] complex64 total interference}
        """
        array_0 = self.arrays[0] if 0 in self.arrays else list(self.arrays.values())[0]
        n_elem = array_0.n_elem

        # Initialize per-radar interference accumulators
        interference = {}
        for j in range(self.n_radars):
            interference[j] = torch.zeros(
                n_elem, n_samples, dtype=torch.complex64, device=self.device,
            )

        # Compute pairwise interference using radar equation
        for i in range(self.n_radars):
            for j in range(self.n_radars):
                if i == j:
                    continue

                # Link geometry
                distance, az_to_rx, az_to_tx = self._compute_link_geometry(
                    radar_states[i]["pos"], radar_states[j]["pos"],
                )

                # TX antenna gain in victim direction
                tx_boresight = radar_states[i].get("array_az_deg", 0.0)
                tx_rel_az = self._relative_angle(az_to_rx, tx_boresight)
                tx_gain_db = self.arrays[i].get_gain_at_angle(i, tx_rel_az)

                # RX antenna gain in interferer direction
                rx_boresight = radar_states[j].get("array_az_deg", 0.0)
                rx_rel_az = self._relative_angle(az_to_tx, rx_boresight)
                rx_gain_db = self.arrays[j].get_gain_at_angle(j, rx_rel_az)

                # Path loss (one-way)
                path_loss_db = self.channel.compute_path_loss_db(distance)

                # Frequency overlap
                freq_overlap = self.compute_frequency_overlap(
                    radar_states[i]["freq_hz"], radar_states[i]["bandwidth_hz"],
                    radar_states[j]["freq_hz"], radar_states[j]["bandwidth_hz"],
                )
                if freq_overlap <= 0:
                    continue
                freq_overlap_db = 10.0 * np.log10(freq_overlap)

                # TX power in dBm
                tx_power_w = radar_states[i].get("tx_power_w", 1.0)
                tx_power_dbm = 10.0 * np.log10(tx_power_w * 1000.0)

                # Received interference power (dBm) via Friis equation
                rx_power_dbm = (
                    tx_power_dbm
                    + tx_gain_db
                    + rx_gain_db
                    - path_loss_db
                    - POLARIZATION_LOSS_DB
                    + freq_overlap_db
                )

                # Convert to linear amplitude (V into 50 ohm)
                rx_power_w = 10.0 ** ((rx_power_dbm - 30.0) / 10.0)
                rx_amplitude = np.sqrt(rx_power_w)

                # Propagation delay in samples
                delay_samples = int(distance / SPEED_OF_LIGHT * self.channel.fs)

                # Generate per-element interference signal
                # Distribute the received signal across elements with RX phase
                # (far-field: all elements see same signal, different phase)
                waveform = waveforms[i]  # [n_samples]
                if waveform.shape[0] < n_samples:
                    padded = torch.zeros(n_samples, dtype=torch.complex64, device=self.device)
                    padded[:waveform.shape[0]] = waveform
                    waveform = padded

                # Apply delay: shift waveform by delay_samples
                delayed = torch.zeros(n_samples, dtype=torch.complex64, device=self.device)
                src_end = min(n_samples, waveform.shape[0])
                dst_start = min(delay_samples, n_samples)
                dst_end = min(dst_start + src_end, n_samples)
                copy_len = dst_end - dst_start
                if copy_len > 0:
                    delayed[dst_start:dst_end] = waveform[:copy_len]

                # Scale by received amplitude
                delayed = delayed * rx_amplitude

                # Distribute across elements (uniform for far-field)
                interference[j] += delayed.unsqueeze(0).expand(n_elem, -1)

        return interference

    def compute_interference_power_dbm(
        self,
        interference: Dict[int, torch.Tensor],
    ) -> np.ndarray:
        """Compute per-radar total interference power in dBm."""
        power_dbm = np.zeros(self.n_radars)
        for j in range(self.n_radars):
            total_power = torch.sum(torch.abs(interference[j]) ** 2).item()
            if total_power > 0:
                power_dbm[j] = 10.0 * np.log10(total_power * 1000.0)
            else:
                power_dbm[j] = -200.0
        return power_dbm

    def compute_jnr_db(
        self,
        interference: Dict[int, torch.Tensor],
        noise_power_linear: float,
    ) -> np.ndarray:
        """Compute JNR (Jam-to-Noise Ratio) per radar in dB.

        After RX beamforming (coherent sum across elements), noise reduces
        by N (incoherent averaging) while signal is preserved.
        """
        jnr_db = np.zeros(self.n_radars)
        for j in range(self.n_radars):
            # Beamformed interference: average across elements
            beamformed = torch.mean(interference[j], dim=0)
            jam_power = torch.mean(torch.abs(beamformed) ** 2).item()
            # Noise after beamforming: reduced by N elements
            n_elem = interference[j].shape[0]
            noise_bf = noise_power_linear / n_elem
            if jam_power > 0 and noise_bf > 0:
                jnr_db[j] = 10.0 * np.log10(jam_power / noise_bf)
            else:
                jnr_db[j] = -200.0
        return jnr_db

    def _angle_to_interferer(
        self, victim_pos, interferer_pos, victim_boresight_deg,
    ) -> float:
        """Compute angle from victim to interferer relative to boresight."""
        dx = interferer_pos[0] - victim_pos[0]
        dy = interferer_pos[1] - victim_pos[1]
        az_global = np.degrees(np.arctan2(dy, dx))
        rel_az = az_global - victim_boresight_deg
        rel_az = ((rel_az + 180) % 360) - 180
        return rel_az
