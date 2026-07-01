"""YAML config loader for FluxPhased simulation.

Loads physics.yaml and algorithm.yaml into dataclass instances,
and provides a bridge to MFARVecEnv constructor kwargs.
"""

import os
import yaml
from dataclasses import fields, asdict
from typing import Tuple

from .config import PhysicsConfig, AlgorithmConfig


def _dict_to_dataclass(cls, d: dict):
    """Recursively populate a dataclass from a dict, using defaults for missing keys."""
    if d is None:
        return cls()
    kwargs = {}
    for f in fields(cls):
        if f.name in d:
            val = d[f.name]
            # Handle tuple fields (YAML loads as list)
            if hasattr(f.type, '__origin__'):
                pass
            if isinstance(val, list) and f.name.endswith('_pos') or f.name == 'map_size':
                val = tuple(val)
            # Check if field type is a dataclass
            field_type = f.type
            if isinstance(field_type, type) and hasattr(field_type, '__dataclass_fields__'):
                val = _dict_to_dataclass(field_type, val)
            kwargs[f.name] = val
        # else: dataclass default will be used
    return cls(**kwargs)


def _resolve_config_dir() -> str:
    """Find the configs/ directory."""
    # Try relative to this file's parent
    candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
    if os.path.isdir(candidate):
        return candidate
    # Try CWD
    if os.path.isdir('configs'):
        return 'configs'
    return 'configs'


def load_physics_config(path: str = None) -> PhysicsConfig:
    """Load physics config from YAML file."""
    if path is None:
        path = os.path.join(_resolve_config_dir(), 'physics.yaml')
    if not os.path.exists(path):
        return PhysicsConfig()
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return _dict_to_dataclass(PhysicsConfig, data)


def load_algorithm_config(path: str = None) -> AlgorithmConfig:
    """Load algorithm config from YAML file."""
    if path is None:
        path = os.path.join(_resolve_config_dir(), 'algorithm.yaml')
    if not os.path.exists(path):
        return AlgorithmConfig()
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return _dict_to_dataclass(AlgorithmConfig, data)


def config_to_env_kwargs(
    physics: PhysicsConfig,
    algorithm: AlgorithmConfig,
    num_envs: int = 2,
    device: str = "cuda",
) -> dict:
    """Flatten config hierarchy into MFARVecEnv constructor kwargs."""
    return dict(
        num_envs=num_envs,
        n_radars=physics.battlefield.n_radars,
        rows=physics.array.rows,
        cols=physics.array.cols,
        fc=physics.rf.fc,
        bandwidth=physics.rf.bandwidth,
        prf=physics.cpi.prf,
        pulses_per_cpi=physics.cpi.pulses_per_cpi,
        n_targets=physics.battlefield.n_targets,
        tx_power_w=physics.rf.tx_power_w,
        target_rcs_dbsm=physics.battlefield.target_rcs_dbsm,
        fft_size=algorithm.fft_size,
        symbol_rate=physics.comm.symbol_rate,
        num_input_length=algorithm.num_input_length,
        num_output_length=algorithm.num_output_length,
        n_teams=physics.battlefield.n_teams,
        device=device,
        # Newly exposed params (will be wired in Phase 4)
        dx_wl=physics.array.dx_wl,
        dy_wl=physics.array.dy_wl,
        noise_figure_db=physics.rf.noise_figure_db,
        map_size=physics.battlefield.map_size,
        speed_ms=physics.missile.speed_ms,
        kill_radius_m=physics.missile.kill_radius_m,
        missile_rcs_dbsm=physics.missile.rcs_dbsm,
        rcs_nose_dbsm=physics.missile.rcs_nose_dbsm,
        rcs_side_dbsm=physics.missile.rcs_side_dbsm,
        rcs_tail_dbsm=physics.missile.rcs_tail_dbsm,
        swerling_model=physics.missile.swerling_model,
        red_launch_pos=physics.missile.red_launch_pos,
        blue_launch_pos=physics.missile.blue_launch_pos,
        polarization_loss_db=physics.interference.polarization_loss_db,
        reset_config=asdict(physics.reset),
        reward_config=asdict(physics.reward),
    )
