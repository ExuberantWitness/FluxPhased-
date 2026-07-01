"""Trigger source library for evaluation scenario generation.

Three tables mapping to document 表1/表2/3, adapted to FluxPhased parameters.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TriggerSource:
    """A single configurable parameter for evaluation scenarios."""
    name: str
    category: str       # "perception" | "analysis" | "game"
    param_name: str     # MFARVecEnv constructor kwarg
    value_range: tuple  # (min, max)
    default: float
    unit: str
    description: str


# 表1: 感知场景触发源
PERCEPTION_TRIGGERS: List[TriggerSource] = [
    TriggerSource(
        "target_rcs", "perception", "target_rcs_dbsm",
        (-10, 30), 20.0, "dBsm",
        "目标 RCS — 低 RCS 等效低 SNR",
    ),
    TriggerSource(
        "bandwidth", "perception", "bandwidth",
        (50e6, 400e6), 200e6, "Hz",
        "信号带宽 — 影响距离分辨率和 SNR",
    ),
    TriggerSource(
        "prf", "perception", "prf",
        (1e3, 50e3), 10e3, "Hz",
        "脉冲重频 — 影响 Doppler 模糊距离和多普勒分辨率",
    ),
    TriggerSource(
        "pulses_per_cpi", "perception", "pulses_per_cpi",
        (2, 64), 32, "pulses",
        "每 CPI 脉冲数 — 影响积累增益和 Doppler 分辨率",
    ),
    TriggerSource(
        "tx_power", "perception", "tx_power_w",
        (0.01, 10.0), 1.0, "W",
        "发射功率 — 直接影响 SNR",
    ),
]

# 表2: 分析场景触发源
ANALYSIS_TRIGGERS: List[TriggerSource] = [
    TriggerSource(
        "swerling", "analysis", "swerling_model",
        (0, 4), 3, "index",
        "Swerling 起伏模型 — 影响 RCS 统计特性",
    ),
    TriggerSource(
        "rows", "analysis", "rows",
        (2, 32), 25, "elements",
        "阵列行数 — 影响天线增益和波束宽度",
    ),
    TriggerSource(
        "cols", "analysis", "cols",
        (2, 32), 25, "elements",
        "阵列列数 — 影响天线增益和波束宽度",
    ),
]

# 表3: 干扰博弈触发源
GAME_TRIGGERS: List[TriggerSource] = [
    TriggerSource(
        "missile_speed", "game", "speed_ms",
        (100, 500), 244.4, "m/s",
        "导弹速度 — 影响决策时间压力",
    ),
    TriggerSource(
        "kill_radius", "game", "kill_radius_m",
        (100, 1000), 500.0, "m",
        "击杀半径 — 影响制导精度需求",
    ),
    TriggerSource(
        "n_radars", "game", "n_radars",
        (2, 8), 4, "count",
        "雷达数量 — 影响团队规模和协同复杂度",
    ),
    TriggerSource(
        "map_size", "game", "map_size",
        (5000, 40000), 20000.0, "m",
        "战场尺寸 — 影响探测距离和机动空间",
    ),
]

ALL_TRIGGERS = PERCEPTION_TRIGGERS + ANALYSIS_TRIGGERS + GAME_TRIGGERS
