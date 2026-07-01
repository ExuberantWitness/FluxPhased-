"""Configuration dataclasses for FluxPhased GPU simulation.

All defaults match the GPU module constructors (vec_mfar_env, vec_channel, etc.)
as of the baseline recording date. This is the single source of truth for
parameter schemas used by YAML config loading.
"""

from dataclasses import dataclass, field
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Global array geometry defaults — single source of truth.
# Change these to switch the entire codebase to a new array size
# (e.g. 24×26, 32×26, etc.). All GPU constructors and config dataclasses
# read from these constants.
# ---------------------------------------------------------------------------
DEFAULT_ROWS: int = 25
DEFAULT_COLS: int = 25


@dataclass
class ArrayGeometry:
    rows: int = DEFAULT_ROWS
    cols: int = DEFAULT_COLS
    dx_wl: float = 0.5
    dy_wl: float = 0.5
    taper: str = "uniform"
    taper_param: float = -30.0

    @property
    def num_elements(self) -> int:
        return self.rows * self.cols


@dataclass
class RFConfig:
    fc: float = 10.0e9
    bandwidth: float = 200e6
    tx_power_w: float = 50000.0
    noise_figure_db: float = 5.0
    system_loss_db: float = 3.0


@dataclass
class CPIConfig:
    prf: float = 10.0e3
    pulses_per_cpi: int = 32

    @property
    def pri(self) -> float:
        return 1.0 / self.prf


@dataclass
class MissileConfig:
    speed_ms: float = 244.4
    kill_radius_m: float = 500.0
    rcs_dbsm: float = 10.0
    rcs_nose_dbsm: float = -5.0
    rcs_side_dbsm: float = 12.0
    rcs_tail_dbsm: float = 3.0
    swerling_model: int = 3
    red_launch_pos: Tuple[float, float] = (0.0, -10000.0)
    blue_launch_pos: Tuple[float, float] = (0.0, 10000.0)


@dataclass
class BattlefieldConfig:
    map_size: Tuple[float, float] = (20000.0, 20000.0)
    n_radars: int = 4
    n_teams: int = 2
    n_targets: int = 1
    target_rcs_dbsm: float = 20.0


@dataclass
class ResetConfig:
    position_spread_x: float = 8000.0
    position_spread_y: float = 6000.0
    y_center_offset: float = 5000.0
    velocity_range: float = 20.0
    heading_range: float = 360.0
    speed_range: float = 8.0
    array_rotation_range: float = 120.0
    target_distance_min: float = 5000.0
    target_distance_max: float = 15000.0
    target_velocity_range: float = 30.0


@dataclass
class InterferenceConfig:
    tx_power_dbm: float = 30.0
    polarization_loss_db: float = 3.0


@dataclass
class RewardConfig:
    kill_bonus: float = 10.0
    death_penalty: float = -10.0
    emission_cost: float = -0.001
    urgency_penalty: float = -0.01
    radar_kill_share: float = 1.0
    radar_death_share: float = -1.0


@dataclass
class CommConfig:
    symbol_rate: float = 1e6
    n_bits: int = 32


@dataclass
class PhysicsConfig:
    """All physical simulation parameters."""
    array: ArrayGeometry = field(default_factory=ArrayGeometry)
    rf: RFConfig = field(default_factory=RFConfig)
    cpi: CPIConfig = field(default_factory=CPIConfig)
    missile: MissileConfig = field(default_factory=MissileConfig)
    battlefield: BattlefieldConfig = field(default_factory=BattlefieldConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    comm: CommConfig = field(default_factory=CommConfig)


@dataclass
class WaveformConfig:
    """Waveform parameter ranges (legacy, used by physics/ and waveform_gpu)."""
    num_freq_channels: int = 64
    freq_range: Tuple[float, float] = (8.0e9, 12.0e9)
    pri_range: Tuple[float, float] = (10e-6, 500e-6)
    pulse_width_range: Tuple[float, float] = (1e-6, 100e-6)
    duty_cycle_max: float = 0.3
    beam_az_range: Tuple[float, float] = (-60.0, 60.0)
    beam_el_range: Tuple[float, float] = (-45.0, 45.0)
    power_range: Tuple[float, float] = (0.01, 1.0)
    waveform_types: Tuple[str, ...] = (
        "lfm_up", "lfm_down", "barker_13", "frank_16",
        "costas_16", "nlfm", "p1_code", "p4_code",
    )
    num_code_schemes: int = 16


@dataclass
class VehicleConfig:
    """Vehicle mobility parameters (legacy)."""
    max_speed: float = 8.33
    max_accel: float = 2.0
    max_rotation_speed: float = 60.0
    length: float = 6.0
    rcs_dbsm: float = 20.0


@dataclass
class EnvConfig:
    """Top-level environment configuration (legacy alias for PhysicsConfig)."""
    array: ArrayGeometry = field(default_factory=ArrayGeometry)
    rf: RFConfig = field(default_factory=RFConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    missile: MissileConfig = field(default_factory=MissileConfig)
    cpi: CPIConfig = field(default_factory=CPIConfig)
    battlefield: BattlefieldConfig = field(default_factory=BattlefieldConfig)
    seed: int = 42
    max_steps: int = 10000
    use_gpu: bool = False

    @property
    def all_agents(self) -> List[str]:
        return [
            "red_radar_0", "red_radar_1",
            "blue_radar_0", "blue_radar_1",
            "red_commander", "blue_commander",
        ]


default_config = EnvConfig()


@dataclass
class AlgorithmConfig:
    """Algorithm and training parameters (no physical meaning)."""
    num_input_length: int = 32
    num_output_length: int = 16
    fft_size: int = 0
    max_steps: int = 10000

    # Training hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048

    # Evaluation
    eval_confidence: float = 0.95
    eval_half_width: float = 0.05
    eval_max_episodes: int = 500
    eval_min_episodes: int = 20
