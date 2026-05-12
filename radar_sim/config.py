"""Configuration for multi-agent phased array EW simulation."""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import numpy as np


@dataclass
class ArrayGeometry:
    """25x25 planar phased array geometry."""
    rows: int = 25
    cols: int = 25
    dx: float = 0.5        # element spacing in wavelengths (x)
    dy: float = 0.5        # element spacing in wavelengths (y)
    taper: str = "uniform" # amplitude taper: uniform, taylor, chebyshev
    taper_param: float = 30.0  # sidelobe level (dB) for Taylor/Chebyshev

    @property
    def num_elements(self) -> int:
        return self.rows * self.cols

    @property
    def aperture_size(self) -> Tuple[float, float]:
        return (self.cols * self.dx, self.rows * self.dy)


@dataclass
class RFConfig:
    """RF front-end parameters for one radar."""
    fc: float = 10.0e9           # carrier frequency (Hz), default X-band
    bandwidth: float = 100e6     # instantaneous bandwidth (Hz)
    tx_power_dbm: float = 60.0   # total transmit power (dBm) - sum of all elements
    noise_figure: float = 5.0    # receiver noise figure (dB)
    rf_gain: float = 30.0        # RF gain (dB)
    baseband_gain: float = 20.0  # baseband gain (dB)
    fs: float = 200e6            # sampling rate (sps), >= 2*bandwidth for complex
    load_resistor: float = 50.0  # ohms
    bb_type: str = "complex"
    system_loss: float = 3.0     # system losses (dB)


@dataclass
class WaveformConfig:
    """Waveform parameter ranges for the 6-dim action space."""
    # Frequency dimension: 64 channels across the band
    num_freq_channels: int = 64
    freq_range: Tuple[float, float] = (8.0e9, 12.0e9)  # full tunable range

    # Time dimension: PRI / pulse width control
    pri_range: Tuple[float, float] = (10e-6, 500e-6)   # 10us - 500us
    pulse_width_range: Tuple[float, float] = (1e-6, 100e-6)
    duty_cycle_max: float = 0.3

    # Spatial dimension: beam steering + array rotation
    beam_az_range: Tuple[float, float] = (-60.0, 60.0)   # degrees from boresight
    beam_el_range: Tuple[float, float] = (-45.0, 45.0)

    # Power dimension
    power_range: Tuple[float, float] = (0.01, 1.0)  # fraction of max tx_power

    # Waveform dimension: 8 types
    waveform_types: Tuple[str, ...] = (
        "lfm_up", "lfm_down", "barker_13", "frank_16",
        "costas_16", "nlfm", "p1_code", "p4_code"
    )

    # Code dimension: 16 coding schemes
    num_code_schemes: int = 16


@dataclass
class VehicleConfig:
    """Vehicle mobility parameters."""
    max_speed: float = 8.33       # m/s (30 km/h)
    max_accel: float = 2.0        # m/s^2
    max_rotation_speed: float = 60.0  # deg/s (array rotation)
    length: float = 6.0           # vehicle length (m)
    rcs_dbsm: float = 20.0        # vehicle radar cross-section (dBsm)


@dataclass
class ArtilleryConfig:
    """Artillery parameters."""
    flight_time: float = 30.0          # shell flight time (s)
    kill_radius_min: float = 350.0     # m
    kill_radius_max: float = 360.0     # m
    cooldown: float = 60.0             # seconds between shots
    max_range: float = 15000.0         # m
    cep: float = 50.0                  # circular error probable (m)


@dataclass
class MissileConfig:
    """Cruise missile parameters."""
    speed_ms: float = 244.4            # 880 km/h in m/s
    kill_radius_m: float = 500.0       # lethal radius (m)
    rcs_dbsm: float = 10.0             # average RCS (dBsm), used as radar equation reference
    red_launch_pos: Tuple[float, float] = (0.0, -10000.0)   # red baseline center
    blue_launch_pos: Tuple[float, float] = (0.0, 10000.0)   # blue baseline center
    max_per_team: int = 1              # only 1 missile per team at a time
    interceptable: bool = False        # cannot be intercepted

    # Aspect-angle dependent RCS model
    rcs_nose_dbsm: float = -5.0        # nose-on (head-on, θ≈180°): ~0.3 m²
    rcs_side_dbsm: float = 12.0        # broadside (θ≈90°): fuselage+wings ~16 m²
    rcs_tail_dbsm: float = 3.0         # tail-on (chasing, θ≈0°): engine+tail ~2 m²

    # Swerling fluctuation model: 0=none, 1=slow exp, 2=fast exp, 3=slow χ²(4), 4=fast χ²(4)
    swerling_model: int = 3            # cruise missile: 1 dominant + many small scatterers

    # Hierarchical agent latent dimensions
    num_input_length: int = 32         # radar encoder → commander (uplink latent width)
    num_output_length: int = 16        # commander → radar decoder (downlink instruction width)


@dataclass
class CPIConfig:
    """CPI-level timing parameters."""
    cpi_duration: float = 0.05    # 50ms per CPI
    prf: float = 10.0e3           # 10 kHz → 500 pulses per CPI
    pulses_per_cpi: int = 500

    @property
    def pri(self) -> float:
        return 1.0 / self.prf


@dataclass
class BattlefieldConfig:
    """Battlefield scenario configuration."""
    map_size: Tuple[float, float] = (20000.0, 20000.0)  # 20km x 20km, origin at center

    # Red team initial positions (y < 0 half)
    red_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-2000.0, -5000.0),   # red radar 0
        (-1000.0, -3000.0),   # red radar 1
        (-1500.0, -7000.0),   # red artillery
    ])
    red_headings: List[float] = field(default_factory=lambda: [0.0, 0.0])

    # Blue team initial positions (y > 0 half)
    blue_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (2000.0, 5000.0),     # blue radar 0
        (1000.0, 3000.0),     # blue radar 1
        (1500.0, 7000.0),     # blue artillery
    ])
    blue_headings: List[float] = field(default_factory=lambda: [180.0, 180.0])

    # Boundary handling
    boundary_margin: float = 100.0


@dataclass
class EnvConfig:
    """Top-level environment configuration."""
    array: ArrayGeometry = field(default_factory=ArrayGeometry)
    rf: RFConfig = field(default_factory=RFConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    artillery: ArtilleryConfig = field(default_factory=ArtilleryConfig)
    missile: MissileConfig = field(default_factory=MissileConfig)
    cpi: CPIConfig = field(default_factory=CPIConfig)
    battlefield: BattlefieldConfig = field(default_factory=BattlefieldConfig)

    # Simulation
    seed: int = 42
    max_steps: int = 10000
    use_gpu: bool = False  # placeholder for future RadarSimPy GPU

    # Agent IDs
    red_radar_agents: Tuple[str, str] = ("red_radar_0", "red_radar_1")
    blue_radar_agents: Tuple[str, str] = ("blue_radar_0", "blue_radar_1")
    red_commander: str = "red_commander"
    blue_commander: str = "blue_commander"

    @property
    def all_agents(self) -> List[str]:
        return (list(self.red_radar_agents) + list(self.blue_radar_agents)
                + [self.red_commander, self.blue_commander])


# Default config instance
default_config = EnvConfig()
