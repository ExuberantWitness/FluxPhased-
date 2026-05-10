"""GPU-accelerated IQ-level cross-radar interference simulation using Warp.

Computes the actual interfering IQ signal that each radar receives from
every other radar (not just dB-level power). This is the key differentiator
from the existing interference.py which only computes JNR in dB.

For 4 radars: 4×4 - 4 = 12 interference links to compute.
Each link: TX waveform → TX beam pattern → channel → RX beam pattern → victim RX.
"""

import numpy as np
import warp as wp
import torch
from typing import Dict, List, Tuple, Optional

from .array_gpu import PhasedArrayGPU
from .channel_gpu import ChannelGPU

SPEED_OF_LIGHT = 299792458.0


@wp.kernel
def _accumulate_interference_kernel(
    # Interfering signal from one source [n_elem, 2*n_samples] (interleaved)
    interference: wp.array2d(dtype=wp.float32),
    # Victim's accumulated interference [n_elem, 2*n_samples]
    accumulated: wp.array2d(dtype=wp.float32),
    n_samples: wp.int32,
):
    """Add interfering signal to victim's accumulated interference buffer."""
    e = wp.tid()
    for s in range(n_samples):
        accumulated[e, 2 * s] += interference[e, 2 * s]
        accumulated[e, 2 * s + 1] += interference[e, 2 * s + 1]


@wp.kernel
def _compute_frequency_overlap_kernel(
    freq_overlap: wp.float32,
    # Signal [n_elem, 2*n_samples]
    signal_in: wp.array2d(dtype=wp.float32),
    signal_out: wp.array2d(dtype=wp.float32),
    n_samples: wp.int32,
):
    """Scale signal by frequency overlap factor."""
    e = wp.tid()
    for s in range(n_samples):
        signal_out[e, 2 * s] = signal_in[e, 2 * s] * freq_overlap
        signal_out[e, 2 * s + 1] = signal_in[e, 2 * s + 1] * freq_overlap


class InterferenceEngineGPU:
    """IQ-level mutual interference between phased array radars on GPU.

    For each pair (i, j) where i ≠ j:
    1. Generate TX signal from radar i (per-element, with TX beamforming)
    2. Apply propagation channel from radar i to radar j (delay, path loss, fading)
    3. The interfering signal arrives at each element of radar j
    4. Sum all interfering signals into victim j's receive buffer

    The total interference at radar j = sum of signals from i=0,1,2,3 (i≠j)
    after channel propagation and frequency overlap filtering.
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

    def compute_interference_matrix(
        self,
        radar_states: List[dict],
        waveforms: Dict[int, torch.Tensor],
        n_samples: int,
    ) -> Dict[int, torch.Tensor]:
        """Compute IQ-level interference at each radar from all others.

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
        n_elem = self.arrays[0].n_elem if 0 in self.arrays else 625

        # Initialize per-radar interference accumulators
        interference = {}
        for j in range(self.n_radars):
            interference[j] = torch.zeros(
                n_elem, n_samples, dtype=torch.complex64, device=self.device,
            )

        # Compute pairwise interference
        for i in range(self.n_radars):
            for j in range(self.n_radars):
                if i == j:
                    continue

                # TX signal from radar i (per-element, beamformed)
                tx_signal = self.arrays[i].beamform_tx(i, waveforms[i])  # [n_elem, n_samples]

                # Apply TX power scaling
                tx_power_w = radar_states[i].get("tx_power_w", 1.0)
                tx_scale = np.sqrt(tx_power_w)
                tx_signal = tx_signal * tx_scale

                # Propagation channel from i to j
                channel_params = self.channel.compute_channel_params(
                    tx_pos=np.array(radar_states[i]["pos"]),
                    rx_pos=np.array(radar_states[j]["pos"]),
                    tx_vel=np.array(radar_states[i].get("vel", [0, 0, 0])),
                    rx_vel=np.array(radar_states[j].get("vel", [0, 0, 0])),
                )

                # Apply channel effects (delay, path loss, fading)
                rx_interference = self.channel.apply_channel(
                    tx_signal, channel_params, doppler_spread=0.0,
                )

                # Apply frequency overlap scaling
                freq_overlap = self.compute_frequency_overlap(
                    radar_states[i]["freq_hz"], radar_states[i]["bandwidth_hz"],
                    radar_states[j]["freq_hz"], radar_states[j]["bandwidth_hz"],
                )
                rx_interference *= freq_overlap

                # Apply victim's RX antenna gain in direction of interferer
                victim_az = self._angle_to_interferer(
                    radar_states[j]["pos"], radar_states[i]["pos"],
                    radar_states[j].get("array_az_deg", 0.0),
                )
                # Simple gain model: mainlobe = directivity, sidelobe = -20dB
                if abs(victim_az) < 5.0:
                    rx_gain = 1.0
                else:
                    rx_gain = 0.1 ** (abs(victim_az) / 60.0)  # rolloff
                rx_interference *= rx_gain

                # Accumulate into victim j's interference buffer
                interference[j] += rx_interference

        return interference

    def compute_interference_power_dbm(
        self,
        interference: Dict[int, torch.Tensor],
    ) -> np.ndarray:
        """Compute per-radar total interference power in dBm.

        Returns:
            [n_radars] array of interference power in dBm
        """
        power_dbm = np.zeros(self.n_radars)
        for j in range(self.n_radars):
            total_power = torch.sum(torch.abs(interference[j]) ** 2).item()
            if total_power > 0:
                # Convert to dBm (assuming 50 ohm load)
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

        Args:
            interference: {radar_id: [n_elem, n_samples] complex64}
            noise_power_linear: noise power per sample (linear)
        Returns:
            [n_radars] JNR in dB
        """
        jnr_db = np.zeros(self.n_radars)
        for j in range(self.n_radars):
            jam_power = torch.mean(torch.abs(interference[j]) ** 2).item()
            if jam_power > 0 and noise_power_linear > 0:
                jnr_db[j] = 10.0 * np.log10(jam_power / noise_power_linear)
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
