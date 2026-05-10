"""PettingZoo ParallelEnv for multi-agent phased array EW simulation.

6 agents: red_radar_0, red_radar_1, red_commander, blue_radar_0, blue_radar_1, blue_commander

Radar agents control a 25x25 phased array (6-dim deep regulation) + vehicle movement.
Commander agents fuse intel and direct artillery strikes.

All agents act simultaneously each CPI (~50ms). Red vs Blue zero-sum.
"""

from typing import Dict, List, Optional, Tuple, Any
import functools
import copy
import numpy as np
from numpy.typing import NDArray

import gymnasium
from gymnasium import spaces

from ..config import EnvConfig, default_config
from ..physics.array import PhasedArray
from ..physics.waveform import WaveformGenerator
from ..physics.channel import (
    PropagationChannel, albersheim_detection_probability,
    compute_snr, compute_communication_snr, shannon_capacity_bps,
)
from ..physics.interference import InterferenceEngine, compute_spectrum_occupancy
from ..physics.receiver import RadarReceiver, DetectionAggregator
from .battlefield import Battlefield, VehicleState, ArtilleryState


SPEED_OF_LIGHT = 299792458.0
DEG2RAD = np.pi / 180.0


# ─── Helper: convert action vector to physical parameters ───

def _decode_radar_action(
    action: NDArray,
    cfg: EnvConfig,
) -> dict:
    """Decode flat radar action vector into physical parameters.

    Action layout (dim=14):
        [0]:   frequency_channel (0..63 continuous index)
        [1]:   pri_fraction (0..1 → pri min..max)
        [2]:   pulse_width_fraction (0..1 → pw min..max)
        [3]:   beam_az (0..1 → -60..60 deg)
        [4]:   beam_el (0..1 → -45..45 deg)
        [5]:   array_rotation (-1..1 → relative rotation rate)
        [6]:   tx_power_fraction (0..1 → 1%..100% max power)
        [7]:   waveform_index (discrete 0..7)
        [8]:   code_index (discrete 0..15)
        [9]:   function_mode (discrete: 0=detect, 1=recon, 2=jam, 3=comm)
        [10]:  speed (0..1 → 0..max_speed)
        [11]:  heading_change (-1..1 → turn rate)
        [12]:  beam_az_2 (0..1 → for multi-beam / 2nd function)
        [13]:  function_mode_2 (discrete: secondary function, 0=none)
    """
    wf = cfg.waveform
    rf = cfg.rf

    # Frequency: pick center frequency within the band
    freq_channel = float(np.clip(action[0], 0.0, 1.0))
    freq_idx = int(freq_channel * (wf.num_freq_channels - 1))
    channel_bw = (wf.freq_range[1] - wf.freq_range[0]) / wf.num_freq_channels
    fc = wf.freq_range[0] + freq_idx * channel_bw + channel_bw / 2

    # Time: PRI and pulse width
    pri_frac = float(np.clip(action[1], 0.0, 1.0))
    pri = wf.pri_range[0] + pri_frac * (wf.pri_range[1] - wf.pri_range[0])

    pw_frac = float(np.clip(action[2], 0.0, 1.0))
    pulse_width = wf.pulse_width_range[0] + pw_frac * (wf.pulse_width_range[1] - wf.pulse_width_range[0])
    # Enforce duty cycle limit
    if pulse_width > pri * wf.duty_cycle_max:
        pulse_width = pri * wf.duty_cycle_max

    # Space: beam steering + array rotation
    beam_az_norm = float(np.clip(action[3], 0.0, 1.0))
    beam_az = wf.beam_az_range[0] + beam_az_norm * (wf.beam_az_range[1] - wf.beam_az_range[0])

    beam_el_norm = float(np.clip(action[4], 0.0, 1.0))
    beam_el = wf.beam_el_range[0] + beam_el_norm * (wf.beam_el_range[1] - wf.beam_el_range[0])

    array_rotation_rate = float(np.clip(action[5], -1.0, 1.0))

    # Power
    power_frac = float(np.clip(action[6], 0.01, 1.0))
    tx_power = power_frac * (10.0 ** (rf.tx_power_dbm / 10.0) / 1000.0)  # watts

    # Waveform and code
    waveform_idx = int(np.clip(action[7], 0, 7))
    code_idx = int(np.clip(action[8], 0, 15))

    # Function mode
    func_mode = int(np.clip(action[9], 0, 3))

    # Movement
    speed = float(np.clip(action[10], 0.0, 1.0)) * cfg.vehicle.max_speed
    heading_change = float(np.clip(action[11], -1.0, 1.0))

    # Multi-function: secondary beam
    beam_az_2_norm = float(np.clip(action[12], 0.0, 1.0))
    beam_az_2 = wf.beam_az_range[0] + beam_az_2_norm * (wf.beam_az_range[1] - wf.beam_az_range[0])
    func_mode_2 = int(np.clip(action[13], 0, 3))  # 0 = none

    return {
        "fc": fc,
        "bandwidth": rf.bandwidth,
        "pri": pri,
        "pulse_width": pulse_width,
        "beam_az": beam_az,
        "beam_el": beam_el,
        "array_rotation": array_rotation_rate,
        "tx_power_w": tx_power,
        "waveform_type": wf.waveform_types[waveform_idx],
        "code_scheme": code_idx,
        "function_mode": func_mode,
        "speed": speed,
        "heading_change": heading_change,
        "beam_az_2": beam_az_2,
        "function_mode_2": func_mode_2,
    }


def _decode_commander_action(
    action: NDArray,
    cfg: EnvConfig,
) -> dict:
    """Decode commander action. Layout: [fire_flag, target_x_norm, target_y_norm]."""
    fire = action[0] > 0.5
    target_x = float(action[1]) * cfg.battlefield.map_size[0]
    target_y = float(action[2]) * cfg.battlefield.map_size[1]
    return {"fire": fire, "target_x": target_x, "target_y": target_y}


# ─── Observation builders ───

def _build_radar_obs(
    agent_id: str,
    team: str,
    vehicle: VehicleState,
    radar_params: dict,
    detections: list,
    spectrum: NDArray,
    sinr_beams: NDArray,
    comm_rx: dict,
    friendly_radar: VehicleState,
    friendly_arty: ArtilleryState,
    enemy_vehicles: List[Tuple[str, VehicleState]],
    cfg: EnvConfig,
) -> NDArray:
    """Build flat radar agent observation vector.

    Observation layout (107-dim):
        [0:2]    own position (x, y) normalized by map size
        [2]      own heading (deg) / 360
        [3]      own speed / max_speed
        [4]      array_bearing / 360
        [5]      alive flag
        [6:70]   spectrum occupancy (64 channels, normalized)
        [70:78]  beams SINR per beam (8 beams, dB normalized)
        [78:80]  own beam direction (az, el) normalized
        [80:84]  primary target info (range, az, velocity, snr) or zeros
        [84:90]  6-dim resource usage (freq, time, space, power, wf, code)
        [90:92]  friendly radar relative position
        [92:94]  nearest enemy relative position
        [94]     function mode one-hot encoded → flattened
        [95:99]  artillery status
        [99:107] comm_rx data (8-dim)
    """
    ms = cfg.battlefield.map_size

    obs = np.zeros(107, dtype=np.float32)

    # Own state
    obs[0] = vehicle.x / ms[0]
    obs[1] = vehicle.y / ms[1]
    obs[2] = vehicle.heading / 360.0
    obs[3] = vehicle.speed / cfg.vehicle.max_speed
    obs[4] = vehicle.array_bearing / 360.0
    obs[5] = float(vehicle.alive)

    # Spectrum (64 channels, dBm normalized to [0,1])
    spectrum_norm = np.clip((spectrum + 120.0) / 80.0, 0.0, 1.0)  # -120..-40 dBm → 0..1
    obs[6:70] = spectrum_norm[:64]

    # Beam SINR (8 beams, clamped -20..40 dB → 0..1)
    sinr_norm = np.clip((sinr_beams + 20.0) / 60.0, 0.0, 1.0)
    obs[70:78] = sinr_norm[:8]

    # Current beam direction
    obs[78] = (radar_params.get("beam_az", 0.0) + 60.0) / 120.0
    obs[79] = (radar_params.get("beam_el", 0.0) + 45.0) / 90.0

    # Best detection
    if detections:
        best = detections[0]
        obs[80] = min(best.get("range_m", 0.0) / 20000.0, 1.0)
        obs[81] = (best.get("bearing_deg", 0.0) + 180.0) / 360.0
        obs[82] = (best.get("velocity_mps", 0.0) + 100.0) / 200.0
        obs[83] = np.clip((best.get("snr_db", -20.0) + 20.0) / 60.0, 0.0, 1.0)

    # 6-dim resource usage (one-hot for discrete dimensions)
    obs[84] = (radar_params.get("fc", 10e9) - cfg.waveform.freq_range[0]) / (
        cfg.waveform.freq_range[1] - cfg.waveform.freq_range[0]
    )
    obs[85] = radar_params.get("pri", 100e-6) / cfg.waveform.pri_range[1]
    obs[86] = (radar_params.get("beam_az", 0.0) + 60.0) / 120.0
    obs[87] = radar_params.get("tx_power_w", 1.0) / (10.0 ** (cfg.rf.tx_power_dbm / 10.0) / 1000.0)
    obs[88] = radar_params.get("_wf_idx", 0) / 7.0
    obs[89] = radar_params.get("_code_idx", 0) / 15.0

    # Friendly radar relative position
    dx_f = (friendly_radar.x - vehicle.x) / ms[0]
    dy_f = (friendly_radar.y - vehicle.y) / ms[1]
    obs[90] = (dx_f + 1.0) / 2.0
    obs[91] = (dy_f + 1.0) / 2.0

    # Nearest enemy
    if enemy_vehicles:
        nearest = min(enemy_vehicles, key=lambda v: np.sqrt(
            (v[1].x - vehicle.x)**2 + (v[1].y - vehicle.y)**2
        ))
        obs[92] = (nearest[1].x - vehicle.x) / ms[0] * 0.5 + 0.5
        obs[93] = (nearest[1].y - vehicle.y) / ms[1] * 0.5 + 0.5

    # Function mode as one-hot across 4 bins
    fm = radar_params.get("function_mode", 0)
    obs[94] = fm / 3.0  # 0, 1/3, 2/3, 1

    # Artillery status
    obs[95] = float(friendly_arty.ready)
    obs[96] = friendly_arty.cooldown_remaining / cfg.artillery.cooldown
    obs[97] = len(friendly_arty.shells_in_flight) / 10.0
    obs[98] = 0.0  # reserved

    # Comm RX data
    for i in range(min(8, len(comm_rx.get("data", [])))):
        obs[99 + i] = comm_rx["data"][i] if "data" in comm_rx else 0.0

    return obs.astype(np.float32)


def _build_commander_obs(
    team: str,
    own_vehicles: List[Tuple[str, VehicleState]],
    enemy_vehicles: List[Tuple[str, VehicleState]],
    radar_detections: Dict[str, list],
    arty: ArtilleryState,
    shell_impacts: List[dict],
    cfg: EnvConfig,
) -> NDArray:
    """Build commander observation vector.

    Observation layout (50-dim):
        [0:2]    own_radar_0 position (normalized)
        [2:4]    own_radar_1 position
        [4:8]    fused detection 0 (range, bearing, velocity, confidence)
        [8:12]   fused detection 1
        [12:16]  fused detection 2
        [16:18]  estimated enemy 0 position
        [18:20]  estimated enemy 1 position
        [20:22]  artillery position
        [22]     artillery ready
        [23]     artillery cooldown fraction
        [24]     num shells in flight / 10
        [25:29]  shell impact info (next impact time, x, y, team)
        [29:50]  reserved
    """
    ms = cfg.battlefield.map_size
    obs = np.zeros(50, dtype=np.float32)

    # Own radar positions
    for i, (aid, v) in enumerate(own_vehicles[:2]):
        obs[i * 2] = v.x / ms[0]
        obs[i * 2 + 1] = v.y / ms[1]

    # Fused detections (top 3)
    all_dets = []
    for dets in radar_detections.values():
        all_dets.extend(dets)
    all_dets.sort(key=lambda d: d.get("snr_db", -99), reverse=True)

    for i in range(min(3, len(all_dets))):
        base = 4 + i * 4
        d = all_dets[i]
        obs[base] = min(d.get("range_m", 0) / 20000.0, 1.0)
        obs[base + 1] = (d.get("bearing_deg", 0) + 180.0) / 360.0
        obs[base + 2] = (d.get("velocity_mps", 0) + 100.0) / 200.0
        obs[base + 3] = d.get("snr_db", -20) / 40.0 if d.get("snr_db", -20) > -20 else 0.0

    # Estimated enemy positions (best guess from detections)
    for i, (eid, ev) in enumerate(enemy_vehicles[:2]):
        obs[16 + i * 2] = ev.x / ms[0]
        obs[16 + i * 2 + 1] = ev.y / ms[1]

    # Artillery state
    obs[20] = arty.x / ms[0]
    obs[21] = arty.y / ms[1]
    obs[22] = float(arty.ready)
    obs[23] = arty.cooldown_remaining / max(cfg.artillery.cooldown, 1.0)
    obs[24] = min(len(arty.shells_in_flight) / 10.0, 1.0)

    # Shell impacts
    for i, si in enumerate(shell_impacts[:1]):
        obs[25 + i * 4] = si.get("time_to_impact", 0) / cfg.artillery.flight_time
        obs[26 + i * 4] = si.get("impact_x", 0) / ms[0]
        obs[27 + i * 4] = si.get("impact_y", 0) / ms[1]

    return obs.astype(np.float32)


# ─── PettingZoo ParallelEnv ───

class PhasedArrayEWEnv:
    """PettingZoo-compatible ParallelEnv for multi-agent phased array EW.

    This follows the PettingZoo Parallel API conventions.
    Not subclassing pettingzoo.ParallelEnv to avoid dependency issues,
    but the API is fully compatible.
    """

    metadata = {"name": "PhasedArrayEW-v0"}

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or default_config
        self.cfg = self.config

        # Agent IDs
        self.possible_agents = list(self.cfg.all_agents)
        self.agents = list(self.possible_agents)

        # Initialize physics engines (one per radar)
        self.arrays: Dict[str, PhasedArray] = {}
        self.waveforms: Dict[str, WaveformGenerator] = {}
        self.receivers: Dict[str, RadarReceiver] = {}
        self.detection_aggregators: Dict[str, DetectionAggregator] = {}
        self.channel = PropagationChannel(seed=self.cfg.seed)
        self.interference = InterferenceEngine()

        # Per-team communication state
        self._comm_buffers: Dict[str, dict] = {}

        # Battlefield
        self.battlefield: Optional[Battlefield] = None

        # Step counter
        self.step_count: int = 0

        # RNG
        self._rng = np.random.default_rng(self.cfg.seed)

        # Build observation and action spaces
        self._build_spaces()

    def _build_spaces(self):
        """Define observation and action spaces for all agents."""

        # Radar agent action space (14-dim)
        self.action_spaces = {}
        self.action_spaces["radar"] = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 7, 15, 3, 1, 1, 1, 3], dtype=np.float32),
            dtype=np.float32,
        )

        # Commander agent action space (3-dim)
        self.action_spaces["commander"] = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Radar agent observation space (107-dim)
        self.observation_spaces = {}
        self.observation_spaces["radar"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(107,), dtype=np.float32
        )

        # Commander agent observation space (50-dim)
        self.observation_spaces["commander"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32
        )

    def _get_agent_team(self, agent_id: str) -> str:
        return "red" if agent_id.startswith("red") else "blue"

    def _get_agent_type(self, agent_id: str) -> str:
        return "commander" if "commander" in agent_id else "radar"

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        """Reset the environment. Returns (observations, infos)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.cfg.seed = seed

        self.step_count = 0
        self.agents = list(self.possible_agents)

        # Initialize battlefield
        self.battlefield = Battlefield(
            self.cfg.vehicle, self.cfg.artillery,
            self.cfg.battlefield, self.cfg.cpi,
            seed=self.cfg.seed,
        )

        # Initialize arrays, waveforms, receivers for each radar
        for aid in self.possible_agents:
            if self._get_agent_type(aid) == "radar":
                self.arrays[aid] = PhasedArray(self.cfg.array, self.cfg.rf.fc)
                self.waveforms[aid] = WaveformGenerator(self.cfg.rf, self.cfg.waveform, self.cfg.cpi)
                self.receivers[aid] = RadarReceiver(self.cfg.rf, self.cfg.waveform, self.cfg.cpi)
                self.detection_aggregators[aid] = DetectionAggregator(track_lifetime=40)

        # Init comm buffers
        self._comm_buffers = {team: {"data": np.zeros(8), "snr_db": -50.0} for team in ["red", "blue"]}

        # Build initial observations
        obs = {}
        infos = {}
        for aid in self.agents:
            obs[aid], infos[aid] = self._build_observation(aid)

        return obs, infos

    def step(self, actions: Dict[str, NDArray]) -> Tuple[dict, dict, dict, dict, dict]:
        """Execute one CPI step.

        Args:
            actions: {agent_id: action_array}

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        self.step_count += 1

        # 1. Execute all radar physics
        radar_results = {}
        for aid in self.agents:
            if self._get_agent_type(aid) == "radar":
                action = actions.get(aid, np.zeros(14))
                radar_results[aid] = self._execute_radar_step(aid, action)
            else:
                radar_results[aid] = self._execute_commander_step(
                    aid, actions.get(aid, np.zeros(3))
                )

        # 2. Compute mutual interference
        self._compute_interference(radar_results)

        # 3. Execute battlefield dynamics
        self.battlefield.step_time()

        kills = self.battlefield.check_shell_impacts()

        # 4. Build observations, rewards, terminations
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        game_over = self.battlefield.check_game_over()
        truncated = self.step_count >= self.cfg.max_steps

        for aid in self.agents:
            obs, info = self._build_observation(aid)
            observations[aid] = obs
            rewards[aid] = self._compute_reward(aid, radar_results, kills)
            terminations[aid] = game_over is not None
            truncations[aid] = truncated
            infos[aid] = info

        # Remove dead agents
        for aid in list(self.agents):
            agent_type = self._get_agent_type(aid)
            if agent_type == "radar":
                v = self.battlefield.get_vehicle(aid)
                if not v.alive:
                    self.agents.remove(aid)

        return observations, rewards, terminations, truncations, infos

    def _execute_radar_step(self, agent_id: str, action: NDArray) -> dict:
        """Execute one CPI of radar physics for a radar agent."""
        cfg = self.cfg
        team = self._get_agent_team(agent_id)
        params = _decode_radar_action(action, cfg)

        # Store waveform/code indices for observation
        wf_types = cfg.waveform.waveform_types
        params["_wf_idx"] = int(np.clip(action[7], 0, 7))
        params["_code_idx"] = int(np.clip(action[8], 0, 15))

        array = self.arrays[agent_id]
        array.steer_beam(params["beam_az"], params["beam_el"])

        # Multi-beam for multi-function
        if params["function_mode_2"] > 0:
            array.steer_multi_beam([
                (params["beam_az"], params["beam_el"], 0.7),
                (params["beam_az_2"], params["beam_el"], 0.3),
            ])

        # Get array gain in beam direction
        tx_gain_db = array.get_antenna_gain(params["beam_az"], params["beam_el"])
        rx_gain_db = tx_gain_db  # Same array for tx and rx

        # Update vehicle state
        self.battlefield.update_vehicle(
            agent_id, params["speed"], params["heading_change"], params["array_rotation"]
        )

        return {
            "params": params,
            "tx_gain_db": tx_gain_db,
            "rx_gain_db": rx_gain_db,
            "agent_id": agent_id,
            "team": team,
        }

    def _execute_commander_step(self, agent_id: str, action: NDArray) -> dict:
        """Execute commander action (artillery fire decision)."""
        params = _decode_commander_action(action, self.cfg)
        team = self._get_agent_team(agent_id)
        fired = False

        if params["fire"]:
            fired = self.battlefield.fire_artillery(
                team, params["target_x"], params["target_y"]
            )

        return {
            "params": params,
            "fired": fired,
            "agent_id": agent_id,
            "team": team,
        }

    def _compute_interference(self, radar_results: dict):
        """Compute mutual interference between all radar pairs."""
        # Build radar state list for interference engine
        states = []
        beam_models = []

        radar_agents = [aid for aid in self.agents if self._get_agent_type(aid) == "radar"]
        for aid in radar_agents:
            v = self.battlefield.get_vehicle(aid)
            res = radar_results.get(aid, {})
            params = res.get("params", {})
            array = self.arrays[aid]

            states.append({
                "pos": (v.x, v.y),
                "heading": v.heading,
                "array_az": v.array_bearing,
                "tx_power_w": params.get("tx_power_w", 1.0),
                "tx_gain_db": res.get("tx_gain_db", 30.0),
                "rx_gain_db": res.get("rx_gain_db", 30.0),
                "freq_hz": params.get("fc", self.cfg.rf.fc),
                "bandwidth_hz": params.get("bandwidth", self.cfg.rf.bandwidth),
                "noise_figure_db": self.cfg.rf.noise_figure,
            })
            beam_models.append(lambda az, el, a=array: a.get_antenna_gain(float(az), float(el)))

        if len(states) >= 2:
            interference_mat = self.interference.compute_full_interference(states, beam_models)

            # Update SINR for each radar
            for i, aid in enumerate(radar_agents):
                total_jnr_linear = 0.0
                for j in range(len(states)):
                    if i != j:
                        jnr_db = interference_mat[j, i]  # interference from j to i
                        total_jnr_linear += 10.0 ** (jnr_db / 10.0)

                # Store in radar results
                radar_results[aid]["interference_jnr_linear"] = total_jnr_linear
                radar_results[aid]["interference_mat"] = interference_mat

    def _build_observation(self, agent_id: str) -> Tuple[NDArray, dict]:
        """Build observation for a single agent."""
        cfg = self.cfg
        agent_type = self._get_agent_type(agent_id)
        team = self._get_agent_team(agent_id)

        if agent_type == "commander":
            return self._build_commander_obs(agent_id, team)
        else:
            return self._build_radar_agent_obs(agent_id, team)

    def _build_radar_agent_obs(self, agent_id: str, team: str) -> Tuple[NDArray, dict]:
        """Build full radar agent observation."""
        cfg = self.cfg
        vehicle = self.battlefield.get_vehicle(agent_id)

        # Determine radar parameters from last action (or defaults)
        # For observation, we use the actual physical state
        radar_params = {
            "fc": cfg.rf.fc,
            "beam_az": self.arrays[agent_id].beam_az,
            "beam_el": self.arrays[agent_id].beam_el,
            "pri": 100e-6,
            "tx_power_w": 10.0 ** (cfg.rf.tx_power_dbm / 10.0) / 1000.0,
            "_wf_idx": 0,
            "_code_idx": 0,
            "function_mode": 0,
        }

        # Spectrum occupancy
        radar_states = []
        for aid in self.agents:
            if self._get_agent_type(aid) == "radar" and aid != agent_id:
                v = self.battlefield.get_vehicle(aid)
                radar_states.append({
                    "pos": (v.x, v.y),
                    "tx_power_w": 10.0 ** (cfg.rf.tx_power_dbm / 10.0) / 1000.0,
                    "tx_gain_db": 30.0,
                    "freq_hz": cfg.rf.fc,
                    "bandwidth_hz": cfg.rf.bandwidth,
                })

        spectrum = compute_spectrum_occupancy(
            radar_states, cfg.waveform.freq_range, cfg.waveform.num_freq_channels
        )

        # Beam SINR for 8 scan directions
        beam_angles = np.linspace(-60, 60, 8)
        sinr_beams = np.zeros(8)
        for i, az in enumerate(beam_angles):
            gain_db = self.arrays[agent_id].get_antenna_gain(float(az), 0.0)
            sinr_beams[i] = gain_db - 80.0  # approximate SINR based on gain minus path loss

        # Detections: check each enemy vehicle
        detections = []
        enemy_vehicles = self.battlefield.get_enemy_vehicles(team)
        enemy_vehicles_list = [(eid, ev) for eid, ev in enemy_vehicles if ev.alive]

        for eid, ev in enemy_vehicles_list:
            distance = np.sqrt((vehicle.x - ev.x) ** 2 + (vehicle.y - ev.y) ** 2)
            bearing = np.degrees(np.arctan2(ev.y - vehicle.y, ev.x - vehicle.x))
            rel_bearing = ((bearing - vehicle.array_bearing + 180) % 360) - 180

            # Check if enemy is within our beam
            bw_az, _ = self.arrays[agent_id].beamwidth_3db()
            if abs(rel_bearing) < bw_az * 3:  # within 3x beamwidth
                # Compute detection SNR
                snr_db = compute_snr(
                    tx_power_w=radar_params["tx_power_w"],
                    tx_gain_db=self.arrays[agent_id].get_antenna_gain(float(rel_bearing), 0.0),
                    rx_gain_db=self.arrays[agent_id].get_antenna_gain(float(rel_bearing), 0.0),
                    distance_m=distance,
                    frequency_hz=cfg.rf.fc,
                    bandwidth_hz=cfg.rf.bandwidth,
                    noise_figure_db=cfg.rf.noise_figure,
                    system_loss_db=cfg.rf.system_loss,
                    rcs_dbsm=cfg.vehicle.rcs_dbsm,
                )

                # Consider jamming interference
                jnr_linear_last = 0.0
                # (would be updated from _compute_interference)

                # Detection probability
                pd = albersheim_detection_probability(snr_db, pfa=1e-6, n_pulses=cfg.cpi.pulses_per_cpi)

                # Probabilistic detection
                if self._rng.random() < pd:
                    # Add measurement noise
                    range_noise = self._rng.normal(0, SPEED_OF_LIGHT / (2 * cfg.rf.bandwidth) / np.sqrt(2))
                    bearing_noise = self._rng.normal(0, bw_az / 3.0)

                    detections.append({
                        "range_m": distance + range_noise,
                        "bearing_deg": rel_bearing + bearing_noise,
                        "velocity_mps": 0.0,  # would need Doppler processing
                        "snr_db": snr_db,
                        "pd": pd,
                        "enemy_id": eid,
                    })

        detections.sort(key=lambda d: d["snr_db"], reverse=True)

        # Friendly radar info
        friendly_radars = self.battlefield.get_team_vehicles(team)
        friendly_radar = None
        for fid, fv in friendly_radars:
            if fid != agent_id:
                friendly_radar = fv
                break
        if friendly_radar is None:
            friendly_radar = vehicle  # fallback

        # Comm RX
        comm_rx = self._comm_buffers.get(team, {"data": np.zeros(8)})

        # Enemy vehicles
        enemy_list = [(eid, ev) for eid, ev in enemy_vehicles if ev.alive]
        if not enemy_list:
            enemy_list = [("none", VehicleState(0, 0, 0, 0, 0, False))]

        # Artillery
        arty = self.battlefield.artillery[team]

        obs = _build_radar_obs(
            agent_id, team, vehicle, radar_params,
            detections, spectrum, sinr_beams,
            comm_rx, friendly_radar, arty,
            enemy_list, cfg,
        )

        info = {
            "detections": detections,
            "alive": vehicle.alive,
            "position": (vehicle.x, vehicle.y),
            "team": team,
        }

        return obs, info

    def _build_commander_obs(self, agent_id: str, team: str) -> Tuple[NDArray, dict]:
        """Build commander agent observation."""
        cfg = self.cfg
        own_vehicles = self.battlefield.get_team_vehicles(team)
        enemy_vehicles = self.battlefield.get_enemy_vehicles(team)

        own_list = [(aid, v) for aid, v in own_vehicles if v.alive]
        enemy_list = [(eid, ev) for eid, ev in enemy_vehicles if ev.alive]

        # Collect radar detections for this team
        radar_detections = {}
        for aid in self.agents:
            if self._get_agent_type(aid) == "radar" and self._get_agent_team(aid) == team:
                # Build detections for this radar
                v = self.battlefield.get_vehicle(aid)
                enemy_list_local = [(eid, ev) for eid, ev in enemy_vehicles if ev.alive]
                dets = []
                for eid, ev in enemy_list_local:
                    distance = np.sqrt((v.x - ev.x) ** 2 + (v.y - ev.y) ** 2)
                    bearing = np.degrees(np.arctan2(ev.y - v.y, ev.x - v.x))
                    rel_bearing = ((bearing - v.array_bearing + 180) % 360) - 180
                    bw_az, _ = self.arrays.get(aid, PhasedArray(cfg.array, cfg.rf.fc)).beamwidth_3db()
                    if abs(rel_bearing) < bw_az * 3:
                        dets.append({
                            "range_m": distance,
                            "bearing_deg": rel_bearing,
                            "velocity_mps": 0.0,
                            "snr_db": 15.0,
                        })
                radar_detections[aid] = dets

        arty = self.battlefield.artillery[team]

        shell_impacts = [
            {
                "time_to_impact": s.time_to_impact(self.battlefield.current_time),
                "impact_x": s.impact_x,
                "impact_y": s.impact_y,
            }
            for s in arty.shells_in_flight
        ]

        obs = _build_commander_obs(
            team, own_list, enemy_list, radar_detections,
            arty, shell_impacts, cfg,
        )

        info = {
            "team": team,
            "artillery_ready": arty.ready,
            "shells_in_flight": len(arty.shells_in_flight),
        }

        return obs, info

    def _compute_reward(
        self,
        agent_id: str,
        radar_results: dict,
        kills: List[Tuple[str, str]],
    ) -> float:
        """Compute per-agent reward. Zero-sum between teams."""
        agent_type = self._get_agent_type(agent_id)
        team = self._get_agent_team(agent_id)
        enemy_team = "blue" if team == "red" else "red"
        reward = 0.0

        if agent_type == "radar":
            v = self.battlefield.get_vehicle(agent_id)
            if not v.alive:
                return -10.0  # destroyed → large penalty

            # Small reward for successful detection
            enemy_vehicles = self.battlefield.get_enemy_vehicles(team)
            for eid, ev in enemy_vehicles:
                if ev.alive:
                    dist = np.sqrt((v.x - ev.x) ** 2 + (v.y - ev.y) ** 2)
                    bearing = np.degrees(np.arctan2(ev.y - v.y, ev.x - v.x))
                    rel_bearing = ((bearing - v.array_bearing + 180) % 360) - 180
                    bw_az, _ = self.arrays[agent_id].beamwidth_3db()
                    if abs(rel_bearing) < bw_az:
                        pd = albersheim_detection_probability(
                            20.0 - 10 * np.log10(max(dist, 1)), pfa=1e-6
                        )
                        reward += 0.01 * pd

            # Small penalty for emitting (being detectable)
            reward -= 0.001

        elif agent_type == "commander":
            # Commander reward: team's kills are positive
            for killed_id, killer_team in kills:
                if killer_team == team:
                    reward += 5.0
                else:
                    reward -= 5.0

            # Small reward for keeping artillery ready
            arty = self.battlefield.artillery[team]
            if arty.ready:
                reward += 0.001

        # Check win/loss
        winner = self.battlefield.check_game_over()
        if winner == team:
            reward += 10.0
        elif winner == enemy_team:
            reward -= 10.0

        return reward

    def render(self) -> Optional[NDArray]:
        """Return current battlefield state for external rendering."""
        if self.battlefield is None:
            return None
        return self.battlefield.get_state_snapshot()

    def close(self):
        pass

    def observation_space(self, agent_id: str) -> spaces.Space:
        agent_type = self._get_agent_type(agent_id)
        return self.observation_spaces[agent_type]

    def action_space(self, agent_id: str) -> spaces.Space:
        agent_type = self._get_agent_type(agent_id)
        return self.action_spaces[agent_type]


# ─── PettingZoo wrapper for compatibility ───

try:
    from pettingzoo import ParallelEnv as PZParallelEnv

    class PettingZooPhasedArrayEW(PZParallelEnv):
        """Official PettingZoo ParallelEnv wrapper for PhasedArrayEW."""

        metadata = {"name": "PhasedArrayEW-v0"}

        def __init__(self, config: Optional[EnvConfig] = None):
            super().__init__()
            self._env = PhasedArrayEWEnv(config)
            self.possible_agents = self._env.possible_agents
            self.agents = self._env.agents

        def reset(self, seed=None, options=None):
            self.agents = list(self.possible_agents)
            return self._env.reset(seed, options)

        def step(self, actions):
            return self._env.step(actions)

        def render(self):
            return self._env.render()

        def close(self):
            self._env.close()

        def observation_space(self, agent):
            return self._env.observation_space(agent)

        def action_space(self, agent):
            return self._env.action_space(agent)

except ImportError:
    # PettingZoo not installed, use standalone
    PettingZooPhasedArrayEW = PhasedArrayEWEnv
