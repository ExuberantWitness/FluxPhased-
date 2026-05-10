"""Full 4-radar GPU simulation pipeline orchestrator.

Coordinates the complete IQ-level simulation chain:
  Waveform → TX Beamforming → Channel → Interference → RX Beamforming
  → Matched Filter → Range-Doppler → CFAR → Detections

Processes in a streaming fashion to fit within RTX 2060 (6.4 GB) VRAM:
  - Per-pulse chunking for memory efficiency
  - Accumulates interference across all 12 cross-radar links
  - Outputs per-radar range-Doppler maps and detections
"""

import numpy as np
import torch
import warp as wp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .array_gpu import PhasedArrayGPU
from .channel_gpu import ChannelGPU
from .waveform_gpu import WaveformGeneratorGPU
from .receiver_gpu import RadarReceiverGPU
from .interference_gpu import InterferenceEngineGPU

SPEED_OF_LIGHT = 299792458.0


@dataclass
class RadarState:
    """State for one radar in the simulation."""
    radar_id: int
    pos: np.ndarray  # [3] position (m)
    vel: np.ndarray  # [3] velocity (m/s)
    freq_hz: float  # carrier frequency
    bandwidth_hz: float  # signal bandwidth
    tx_power_w: float  # total TX power
    array_az_deg: float  # array boresight azimuth
    beam_az_deg: float  # current beam steering azimuth
    beam_el_deg: float  # current beam steering elevation
    waveform_type: str  # "lfm_up", "lfm_down", etc.
    # Target parameters (for radar return simulation)
    targets: list = field(default_factory=list)


@dataclass
class TargetState:
    """State for one target."""
    pos: np.ndarray  # [3] position (m)
    vel: np.ndarray  # [3] velocity (m/s)
    rcs_dbsm: float  # radar cross section (dBsm)


class RadarPipelineGPU:
    """Full 4-radar IQ-level simulation pipeline on GPU.

    Usage:
        pipeline = RadarPipelineGPU(config)
        pipeline.steer_beams({0: (15.0, 0.0), 1: (-30.0, 5.0), ...})
        results = pipeline.simulate_cpi(radar_states, targets)

    Memory budget (RTX 2060, 6.4 GB):
      - Per-pulse chunk: 4 radars × 625 elem × 10K samples × 8B ≈ 200 MB
      - RDM: 4 radars × 500 pulses × 10K bins × 8B ≈ 160 MB
      - Waveforms + buffers: ~100 MB
      - Total peak: ~500 MB (well within budget)
    """

    def __init__(
        self,
        fc: float = 10e9,
        bandwidth: float = 200e6,
        prf: float = 10e3,
        pulses_per_cpi: int = 500,
        n_radars: int = 4,
        rows: int = 25,
        cols: int = 25,
        device: str = "cuda",
    ):
        self.fc = fc
        self.bandwidth = bandwidth
        self.fs = bandwidth
        self.prf = prf
        self.pulses_per_cpi = pulses_per_cpi
        self.n_radars = n_radars
        self.device = device

        # Initialize subsystems
        self.arrays = {}
        for i in range(n_radars):
            self.arrays[i] = PhasedArrayGPU(
                rows=rows, cols=cols, fc=fc, device=device,
            )

        self.channel = ChannelGPU(fc=fc, bandwidth=bandwidth, device=device)
        self.receiver = RadarReceiverGPU(
            fc=fc, bandwidth=bandwidth, prf=prf,
            pulses_per_cpi=pulses_per_cpi, device=device,
        )
        self.interference_engine = InterferenceEngineGPU(
            arrays=self.arrays, channel=self.channel,
            n_radars=n_radars, device=device,
        )

        # Waveform generator
        rf_config = type('RFConfig', (), {
            'fs': self.fs, 'bandwidth': self.bandwidth,
        })()
        wf_config = type('WFConfig', (), {})()
        cpi_config = type('CPIConfig', (), {
            'prf': self.prf, 'pulses_per_cpi': self.pulses_per_cpi,
        })()
        self.waveform_gen = WaveformGeneratorGPU(
            rf_config, wf_config, cpi_config,
            torch.device(device),
        )

        # Samples per pulse
        self.pri = 1.0 / prf
        self.n_samples_per_pulse = int(self.pri * self.fs)

    def steer_beams(self, beam_dirs: Dict[int, Tuple[float, float]]):
        """Set beam steering for all radars.

        Args:
            beam_dirs: {radar_id: (az_deg, el_deg)}
        """
        for rid, (az, el) in beam_dirs.items():
            self.arrays[rid].steer_beam(rid, az, el)

    def generate_pulse(self, waveform_type: str = "lfm_up",
                       pulse_width: float = None) -> torch.Tensor:
        """Generate a single baseband pulse on GPU.

        Returns:
            [n_samples] complex64
        """
        if pulse_width is None:
            pulse_width = self.pri * 0.1  # 10% duty cycle

        pulse = self.waveform_gen.generate(
            waveform_type,
            {"pulse_width": pulse_width, "bandwidth": self.bandwidth},
        )
        # Zero-pad to full PRI
        n_pulse = pulse.shape[0]
        n_pri = self.n_samples_per_pulse
        if n_pulse < n_pri:
            padding = torch.zeros(n_pri - n_pulse, dtype=torch.complex64,
                                  device=torch.device(self.device))
            pulse = torch.cat([pulse, padding])
        return pulse[:n_pri]

    def simulate_pulse(
        self,
        radar_states: List[RadarState],
        pulse_idx: int,
        baseband: torch.Tensor,
        interference_accum: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Simulate one pulse across all radars.

        For each radar:
        1. TX beamforming: baseband → per-element signals
        2. Generate radar returns from targets (per-element)
        3. Add cross-radar interference
        4. RX beamforming: per-element → single beam output
        5. Return received pulse for this pulse index

        Args:
            radar_states: List of RadarState for each radar
            pulse_idx: current pulse index (for Doppler phase)
            baseband: [n_samples] baseband pulse
            interference_accum: optional pre-computed interference
        Returns:
            {radar_id: [n_samples] complex64 received pulse}
        """
        n_samples = baseband.shape[0]
        n_elem = self.arrays[0].n_elem
        received = {}

        # Compute interference if not provided
        if interference_accum is None:
            waveforms = {}
            states_dicts = []
            for rs in radar_states:
                waveforms[rs.radar_id] = baseband
                states_dicts.append({
                    "pos": rs.pos.tolist(),
                    "vel": rs.vel.tolist(),
                    "freq_hz": rs.freq_hz,
                    "bandwidth_hz": rs.bandwidth_hz,
                    "tx_power_w": rs.tx_power_w,
                    "array_az_deg": rs.array_az_deg,
                })
            interference_accum = self.interference_engine.compute_interference_matrix(
                states_dicts, waveforms, n_samples,
            )

        for rs in radar_states:
            rid = rs.radar_id

            # Simulate radar returns from targets
            rx_signal = torch.zeros(n_elem, n_samples, dtype=torch.complex64,
                                    device=torch.device(self.device))

            for tgt in rs.targets:
                tgt_pos = np.array(tgt.pos)
                tgt_vel = np.array(tgt.vel)
                rcs_dbsm = tgt.rcs_dbsm

                # Distance and angle from radar to target
                rel = tgt_pos - rs.pos
                distance = float(np.linalg.norm(rel))
                if distance < 1.0:
                    distance = 1.0

                # RCS in linear
                rcs_linear = 10.0 ** (rcs_dbsm / 10.0)

                # Two-way path loss
                wavelength = SPEED_OF_LIGHT / rs.freq_hz
                path_loss_db = 40.0 * np.log10(
                    4.0 * np.pi * distance / wavelength
                )
                # Radar equation: received power includes RCS
                rx_power_dbm = (
                    10 * np.log10(rs.tx_power_w * 1000)
                    + self.arrays[rid].directivity_db * 2  # TX + RX gain
                    + rcs_dbsm
                    - path_loss_db
                    - 3.0  # system loss
                )
                rx_power_linear = 10.0 ** (rx_power_dbm / 10.0) * 1e-3
                rx_amplitude = np.sqrt(rx_power_linear)

                if rx_amplitude < 1e-15:
                    continue

                # Channel params for radar return
                channel_params = self.channel.compute_channel_params(
                    tx_pos=rs.pos, rx_pos=tgt_pos,
                    tx_vel=rs.vel, rx_vel=tgt_vel,
                )
                # Two-way delay and Doppler
                channel_params["delay_samples"] *= 2  # round trip
                channel_params["doppler_hz"] = (
                    2.0 * np.dot(rs.vel - tgt_vel, rel) / distance
                    * rs.freq_hz / SPEED_OF_LIGHT
                )

                # Scale baseband by TX amplitude
                tx_scaled = baseband * rx_amplitude

                # Apply channel to per-element signal
                tgt_return = self.channel.apply_channel(
                    self.arrays[rid].beamform_tx(rid, tx_scaled),
                    channel_params, doppler_spread=0.0,
                )
                rx_signal += tgt_return

            # Add cross-radar interference
            if rid in interference_accum:
                rx_signal += interference_accum[rid]

            # Add thermal noise
            noise_figure_db = 5.0
            noise_power_dbm = (
                10 * np.log10(1.380649e-23 * 290 * 1000)
                + 10 * np.log10(rs.bandwidth_hz)
                + noise_figure_db
            )
            noise_power_linear = 10.0 ** (noise_power_dbm / 10.0) * 1e-3
            noise_std = np.sqrt(noise_power_linear / 2.0)
            noise = (torch.randn(n_elem, n_samples, dtype=torch.complex64,
                                 device=torch.device(self.device))
                     * noise_std)
            rx_signal += noise

            # RX beamforming: weight and sum across elements
            received[rid] = self.arrays[rid].beamform_rx(rid, rx_signal)

        return received

    def simulate_cpi(
        self,
        radar_states: List[RadarState],
        pulse_width: float = None,
    ) -> Dict[int, dict]:
        """Simulate a full CPI (Coherent Processing Interval) for all radars.

        Processes pulses one at a time (streaming) to manage memory,
        then assembles the pulse train for each radar and runs
        range-Doppler processing.

        Args:
            radar_states: List of RadarState per radar
            pulse_width: pulse width in seconds
        Returns:
            {radar_id: dict with rd_map, detections, snr, interference_power}
        """
        n_samples = self.n_samples_per_pulse
        n_pulses = self.pulses_per_cpi

        # Pre-compute interference matrix (same for all pulses in static scenario)
        baseband = self.generate_pulse(
            radar_states[0].waveform_type if radar_states else "lfm_up",
            pulse_width,
        )

        states_dicts = []
        for rs in radar_states:
            states_dicts.append({
                "pos": rs.pos.tolist(),
                "vel": rs.vel.tolist(),
                "freq_hz": rs.freq_hz,
                "bandwidth_hz": rs.bandwidth_hz,
                "tx_power_w": rs.tx_power_w,
                "array_az_deg": rs.array_az_deg,
            })

        interference = self.interference_engine.compute_interference_matrix(
            states_dicts,
            {rs.radar_id: baseband for rs in radar_states},
            n_samples,
        )

        # Accumulate pulse train for each radar
        pulse_trains = {}
        for rs in radar_states:
            pulse_trains[rs.radar_id] = torch.zeros(
                n_pulses, n_samples, dtype=torch.complex64,
                device=torch.device(self.device),
            )

        # Process each pulse
        for p in range(n_pulses):
            received = self.simulate_pulse(
                radar_states, p, baseband, interference,
            )
            for rid, pulse in received.items():
                pulse_trains[rid][p, :] = pulse

            if (p + 1) % 100 == 0:
                print(f"  [pipeline] Pulse {p + 1}/{n_pulses} processed")

        # Run receiver processing for each radar
        results = {}
        for rs in radar_states:
            rid = rs.radar_id
            rx_result = self.receiver.process_pulse_train(
                pulse_trains[rid], baseband,
            )

            # Interference power at this radar
            if rid in interference:
                int_power = torch.mean(torch.abs(interference[rid]) ** 2).item()
                int_power_dbm = 10 * np.log10(max(int_power * 1000, 1e-30))
            else:
                int_power_dbm = -200.0

            results[rid] = {
                "rd_map": rx_result["rd_map"],
                "range_profile": rx_result["range_profile"],
                "detections": rx_result["detections"],
                "interference_power_dbm": int_power_dbm,
                "n_detections": len(rx_result["detections"]),
                "pulse_train": pulse_trains[rid],
            }

        return results

    def benchmark(self, n_pulses: int = 10) -> dict:
        """Quick benchmark: time one CPI simulation cycle."""
        import time

        radar_states = []
        for i in range(self.n_radars):
            angle = 2 * np.pi * i / self.n_radars
            radar_states.append(RadarState(
                radar_id=i,
                pos=np.array([3000 * np.cos(angle), 3000 * np.sin(angle), 0.0]),
                vel=np.array([0.0, 0.0, 0.0]),
                freq_hz=self.fc,
                bandwidth_hz=self.bandwidth,
                tx_power_w=1.0,
                array_az_deg=45.0 * i,
                beam_az_deg=0.0,
                beam_el_deg=0.0,
                waveform_type="lfm_up",
                targets=[],
            ))

        self.steer_beams({i: (0.0, 0.0) for i in range(self.n_radars)})

        # Time waveform generation
        t0 = time.perf_counter()
        baseband = self.generate_pulse("lfm_up")
        torch.cuda.synchronize()
        t_waveform = time.perf_counter() - t0

        # Time interference computation
        t0 = time.perf_counter()
        states_dicts = [{
            "pos": rs.pos.tolist(), "vel": rs.vel.tolist(),
            "freq_hz": rs.freq_hz, "bandwidth_hz": rs.bandwidth_hz,
            "tx_power_w": rs.tx_power_w, "array_az_deg": rs.array_az_deg,
        } for rs in radar_states]
        interference = self.interference_engine.compute_interference_matrix(
            states_dicts,
            {i: baseband for i in range(self.n_radars)},
            self.n_samples_per_pulse,
        )
        torch.cuda.synchronize()
        t_interference = time.perf_counter() - t0

        return {
            "waveform_gen_ms": t_waveform * 1000,
            "interference_compute_ms": t_interference * 1000,
            "n_samples_per_pulse": self.n_samples_per_pulse,
            "n_radars": self.n_radars,
            "n_elements_per_array": self.arrays[0].n_elem,
            "gpu_mem_used_mb": torch.cuda.memory_allocated() / 1e6,
        }
