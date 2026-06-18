"""Laser precise-kill task module for FluxLeague.

Extracted from `training/train_laser.py` (lines 228-280, 377-599, 640-707) so the
laser task can run under the AlphaStar-style FluxLeague + TeamPPOTrainer framework
without duplicating logic.

Components:
  - KalmanTracker / fused_sensing / add_sensing_noise: multi-radar anisotropic
    measurement fusion with optional Kalman tracking. Critical for sub-meter
    aim precision (the 0.2m kill radius).
  - LaserRewardShaper: laser-specific reward terms ((1/r²)×t⁴ beam reward,
    fire-gated illumination, kill bonus, misfire penalty, EW race terms).
    Duck-typed to match DenseRewardShaper's __call__ interface.
"""

from .sensing import (
    KalmanTracker,
    fused_sensing,
    add_sensing_noise,
    enforce_radar_baseline,
)
from .reward import LaserRewardShaper

__all__ = [
    "KalmanTracker",
    "fused_sensing",
    "add_sensing_noise",
    "enforce_radar_baseline",
    "LaserRewardShaper",
]
