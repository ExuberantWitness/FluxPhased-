"""Pure physics engine for MFAR phased array radar simulation.

Accepts raw TX IQ, applies physical channel (delay + Doppler + gain + noise +
interference), returns raw RX IQ. No signal processing, no action decoding,
no state assembly. All signal processing happens in the Policy.

Architecture: E parallel environments × R radars × N elements × S samples.
One step = one pulse = 100μs = 20,000 IQ samples at 200 MHz.
"""

import time
import numpy as np
import torch

from .vec_array import VecArray
from .vec_channel import VecChannel
from .vec_interference import VecInterference
from .vec_battlefield import VecBattlefield
from .damage import (
    apply_weibull_clutter,
    apply_multipath_2ray,
    clamp_beam_slew,
)
from ..config import DEFAULT_ROWS, DEFAULT_COLS

SPEED_OF_LIGHT = 299792458.0
DEG2RAD = np.pi / 180.0


class MFARVecEnv:
    """Pure physics engine: TX IQ in → channel → RX IQ out.

    Each step processes one pulse (S=20,000 IQ samples at 200 MHz = 100μs).
    The env manages physical state (positions, velocities, channel, noise,
    interference, drone, laser) but does NOT manage signal processing or
    action decoding — that's the Policy's job.
    """

    def __init__(
        self,
        num_envs: int = 2,
        n_radars: int = 4,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        fc: float = 10e9,
        bandwidth: float = 200e6,
        prf: float = 10e3,
        pulses_per_cpi: int = 4,
        n_targets: int = 1,
        tx_power_w: float = 1.0,
        target_rcs_dbsm: float = 20.0,
        fft_size: int = 0,
        symbol_rate: float = 1e6,
        num_input_length: int = 32,
        num_output_length: int = 16,
        n_teams: int = 2,
        device: str = "cuda",
        dx_wl: float = 0.5,
        dy_wl: float = 0.5,
        noise_figure_db: float = 5.0,
        map_size: tuple = (20000.0, 20000.0),
        kill_radius_m: float = 0.2,
        illumination_time_s: float = 0.002,
        drone_altitude_m: float = 3000.0,
        polarization_loss_db: float = 3.0,
        tx_rx_isolation_db: float = 200.0,
        rx_beamforming: bool = True,
        reset_config: dict = None,
        reward_config: dict = None,
        vehicle_speed_ms: float = 20.0,
        damage_config: dict = None,
    ):
        self.num_envs = num_envs
        self.n_radars = n_radars
        self.n_teams = n_teams
        self.rows = rows
        self.cols = cols
        self.n_elem = rows * cols
        self.n_pulses = pulses_per_cpi
        self.n_targets = n_targets
        self.fc = fc
        self.bandwidth = bandwidth
        self.fs = bandwidth
        self.prf = prf
        self.pri = 1.0 / prf
        self.n_samples = max(1, int(self.pri * self.fs))
        self.tx_power_w = tx_power_w
        self.target_rcs_dbsm = target_rcs_dbsm
        self.num_input_length = num_input_length
        self.num_output_length = num_output_length
        self.device = device
        self.dx_wl = dx_wl
        self.dy_wl = dy_wl
        self.noise_figure_db = noise_figure_db
        self.map_size = map_size
        self.tx_rx_isolation_db = tx_rx_isolation_db
        self.rx_beamforming = rx_beamforming
        self.reset_config = reset_config or {}
        self.reward_config = reward_config or {}
        self.vehicle_speed_ms = vehicle_speed_ms
        # WP3.2 damage-injection config (None → all damages disabled)
        self.damage_config = damage_config or {}
        self._init_damage()

        E, R, N = num_envs, n_radars, self.n_elem
        S = self.n_samples
        dev = torch.device(device)

        # --- Subsystems ---
        self.array = VecArray(
            rows=rows, cols=cols, fc=fc,
            num_envs=num_envs, n_radars=n_radars, device=device,
            dx_wl=dx_wl, dy_wl=dy_wl,
        )
        self.channel = VecChannel(
            fc=fc, bandwidth=bandwidth,
            num_envs=num_envs, n_radars=n_radars,
            n_elem=N, n_samples=S, device=device,
            noise_figure_db=noise_figure_db,
        )
        self.interference = VecInterference(
            fc=fc, bandwidth=bandwidth, rows=rows, cols=cols,
            num_envs=num_envs, n_radars=n_radars,
            n_elem=N, device=device,
            polarization_loss_db=polarization_loss_db,
            dx_wl=dx_wl, dy_wl=dy_wl,
        )

        self.n_bins = fft_size if fft_size > 0 else S

        # --- Pre-allocated GPU buffers ---
        self._buf_rx = torch.zeros(E, R, N, S, dtype=torch.complex64, device=dev)
        self._buf_noise = torch.zeros(E, R, N, S, dtype=torch.complex64, device=dev)
        self._buf_intf = torch.zeros(E, R, N, S, dtype=torch.complex64, device=dev)

        # --- State tensors ---
        self.radar_pos = torch.zeros(E, R, 3, device=dev)
        self.radar_vel = torch.zeros(E, R, 3, device=dev)
        self.radar_heading = torch.zeros(E, R, device=dev)
        self.radar_speed = torch.zeros(E, R, device=dev)
        self.array_rotation = torch.zeros(E, R, device=dev)
        self.target_pos = torch.zeros(E, n_targets, 3, device=dev)
        self.target_vel = torch.zeros(E, n_targets, 3, device=dev)

        # Element positions
        dx_m = dx_wl * self.array.wavelength
        dy_m = dy_wl * self.array.wavelength
        x_pos = (np.arange(cols) - (cols - 1) / 2.0) * dx_m
        y_pos = (np.arange(rows) - (rows - 1) / 2.0) * dy_m
        X, Y = np.meshgrid(x_pos, y_pos)
        self.elem_x = torch.tensor(X.ravel().astype(np.float32), device=dev)
        self.elem_y = torch.tensor(Y.ravel().astype(np.float32), device=dev)

        # --- Battlefield (drone + laser combat + game state) ---
        self.battlefield = VecBattlefield(
            num_envs=num_envs, n_radars=n_radars, n_teams=n_teams,
            fs=self.fs, symbol_rate=symbol_rate, device=device,
            num_input_length=num_input_length,
            num_output_length=num_output_length,
            map_size=map_size,
            kill_radius_m=kill_radius_m,
            illumination_time_s=illumination_time_s,
            drone_altitude_m=drone_altitude_m,
            reward_config=reward_config,
            comm_rate_bps=float((damage_config or {}).get("comm_rate_bps", 0.0)),
            pri=self.pri,
        )

        # Pulse counter
        self._pulse_count = torch.zeros(E, dtype=torch.long, device=dev)

    def _init_damage(self):
        """Initialize WP3.2 damage-injection state from damage_config dict.

        Reads optional fields:
          clutter_model: "none" | "weibull"
          clutter_shape_k, clutter_scale_lambda, clutter_cnr_db
          multipath_model: "none" | "2ray"
          multipath_delay_spread_ns, multipath_attenuation_db
          max_slew_rate_deg_per_s  (beam steer rate cap)
          duty_cycle_max           (tx duty cap, 0..1)

        All defaults are no-op. Pre-allocates scratch buffers lazily on first
        use to keep zero-damage runs cost-free.
        """
        cfg = self.damage_config
        dev = torch.device(self.device)
        E, R, N = self.num_envs, self.n_radars, self.n_elem

        # --- Clutter (Weibull) ---
        self.clutter_model = str(cfg.get("clutter_model", "none")).lower()
        self.clutter_shape_k = float(cfg.get("clutter_shape_k", 0.0))
        self.clutter_scale_lambda = float(cfg.get("clutter_scale_lambda", 0.0))
        self.clutter_cnr_db = float(cfg.get("clutter_cnr_db", 0.0))
        self._buf_clutter = None  # lazy alloc

        # --- Multipath (2-ray) ---
        self.multipath_model = str(cfg.get("multipath_model", "none")).lower()
        delay_ns = float(cfg.get("multipath_delay_spread_ns", 0.0))
        self.multipath_delay_samples = int(delay_ns * 1e-9 * self.fs)
        self.multipath_attenuation_db = float(
            cfg.get("multipath_attenuation_db", 0.0))

        # --- Slew rate cap (action-level; tracked in episode runner) ---
        # Stored here for reference / scheduling; the actual clamp happens in
        # training/laser/episode.py where raw radar_actions are visible.
        self.max_slew_rate_deg_per_s = float(
            cfg.get("max_slew_rate_deg_per_s", 0.0))

        # --- Control delay (action queue depth, episode runner) ---
        self.control_delay_steps = int(cfg.get("control_delay_steps", 0))

        # --- Duty cycle cap (average-power interpretation) ---
        # duty_cycle_max < 1 → scale tx_signal power by duty_cycle_max so the
        # radar's effective average radiated power matches a pulsed duty cycle
        # without requiring an explicit transmit on/off action.
        self.duty_cycle_max = float(cfg.get("duty_cycle_max", 0.0))
        if 0.0 < self.duty_cycle_max < 1.0:
            # amplitude scale = sqrt(power_scale) for IQ
            self._duty_amp_scale = float(np.sqrt(self.duty_cycle_max))
        else:
            self._duty_amp_scale = 1.0

    @property
    def state_dim(self) -> int:
        return self.n_elem * (self.n_pulses * self.n_bins + 2 + 4) + 5 + 8 + self.num_output_length

    @property
    def action_dim(self) -> int:
        return self.n_elem * 22 + 3

    def destroy(self):
        bufs = [
            "_buf_rx", "_buf_noise", "_buf_intf",
            "radar_pos", "radar_vel", "radar_heading", "radar_speed",
            "array_rotation", "target_pos", "target_vel",
            "elem_x", "elem_y",
        ]
        for name in bufs:
            obj = getattr(self, name, None)
            if obj is not None:
                setattr(self, name, None)
        for sub in ["array", "channel", "interference", "battlefield"]:
            setattr(self, sub, None)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        E = len(env_ids)
        dev = torch.device(self.device)

        r_per_team = self.n_radars // self.n_teams
        rc = self.reset_config
        pos_spread_x = rc.get('position_spread_x', 8000.0)
        pos_spread_y = rc.get('position_spread_y', 6000.0)
        y_offset = rc.get('y_center_offset', 5000.0)
        tgt_dist_min = rc.get('target_distance_min', 5000.0)
        tgt_dist_range = rc.get('target_distance_max', 15000.0) - tgt_dist_min
        tgt_vel_range = rc.get('target_velocity_range', 30.0)

        for t in range(self.n_teams):
            r_start = t * r_per_team
            r_end = r_start + r_per_team
            n_r = r_end - r_start
            y_center = (t * 2 - 1) * y_offset
            self.radar_pos[env_ids, r_start:r_end, 0] = (torch.rand(E, n_r, device=dev) - 0.5) * pos_spread_x
            self.radar_pos[env_ids, r_start:r_end, 1] = y_center + (torch.rand(E, n_r, device=dev) - 0.5) * pos_spread_y
            self.radar_pos[env_ids, r_start:r_end, 2] = 0.0

        self.radar_vel[env_ids] = 0.0
        self.radar_heading[env_ids] = 0.0
        self.radar_speed[env_ids] = 0.0
        self.array_rotation[env_ids] = 0.0

        r = tgt_dist_min + torch.rand(E, self.n_targets, device=dev) * tgt_dist_range
        ang = torch.rand(E, self.n_targets, device=dev) * 2 * np.pi
        self.target_pos[env_ids, :, 0] = r * torch.cos(ang)
        self.target_pos[env_ids, :, 1] = r * torch.sin(ang)
        self.target_pos[env_ids, :, 2] = 0.0
        self.target_vel[env_ids] = (torch.rand(E, self.n_targets, 3, device=dev) - 0.5) * tgt_vel_range

        self.battlefield.reset(env_ids)
        self._pulse_count[env_ids] = 0

    def step(
        self,
        tx_signal: torch.Tensor,
        commander_actions: torch.Tensor = None,
        vehicle_actions: torch.Tensor = None,
        beam_az: torch.Tensor = None,
        beam_el: torch.Tensor = None,
    ) -> dict:
        """Process one pulse through the physical channel.

        Args:
            tx_signal: [E, R, N, S] complex64 — pre-assembled TX signal (from policy)
            commander_actions: [E, n_teams, 5] — commander→drone perfect link
                [..., 0] = fire_on_off, [..., 1:4] = aim(x,y,z), [..., 4] = reserved
            vehicle_actions: [E, R, 3] optional — vehicle movement
                [..., 0] = speed [0,1], [..., 1] = heading_delta [0,1], [..., 2] = array_rot [0,1]
            beam_az: [E, R] optional — per-radar mean beam azimuth (degrees) from policy.
                If None, falls back to boresight (zeros) — backward compat for tasks
                that don't use beam steering.
            beam_el: [E, R] optional — per-radar mean beam elevation (degrees).
        Returns:
            dict with keys: rx_iq, events, kills, dones, winners, ...
        """
        E, R, N, S = tx_signal.shape
        dev = torch.device(self.device)
        dt = self.pri

        # WP3.2 damage: duty cycle cap (average-power interpretation)
        if self._duty_amp_scale < 1.0:
            tx_signal = tx_signal * self._duty_amp_scale

        t0 = time.perf_counter()

        # --- 1. Compute channel params for static targets ---
        bw_az = float(np.degrees(0.886 / (self.cols * 0.5)))
        bw_el = float(np.degrees(0.886 / (self.rows * 0.5)))

        self._buf_rx.zero_()

        # Default to boresight if policy beam direction not provided.
        if beam_az is None:
            beam_az = torch.zeros(E, R, device=dev)
        if beam_el is None:
            beam_el = torch.zeros(E, R, device=dev)

        for t_idx in range(self.n_targets):
            delay_s, doppler_hz, gain = self.channel.compute_params_batch(
                self.radar_pos, self.radar_vel,
                self.target_pos[:, t_idx], self.target_vel[:, t_idx],
                tx_power_w=self.tx_power_w,
                rcs_dbsm=self.target_rcs_dbsm,
                array_directivity_db=self.array.directivity_db,
                n_elem=self.n_elem,
                beam_az=beam_az,
                beam_el=beam_el,
                array_rotation=self.array_rotation,
                bw_az_deg=bw_az, bw_el_deg=bw_el,
            )
            target_return = self.channel.apply_batch(
                tx_signal, delay_s, doppler_hz, gain,
            )
            self._buf_rx += target_return

        # --- 2. Enemy radar echoes (combat channel) ---
        r_per_team = R // self.n_teams
        for t in range(self.n_teams):
            enemy_t = 1 - t
            own_idx = self.battlefield.team_radar_indices[t]
            enemy_idx = self.battlefield.team_radar_indices[enemy_t]
            for ei in enemy_idx:
                enemy_alive = self.battlefield.alive[:, ei]
                if not enemy_alive.any():
                    continue
                alive_mask = enemy_alive.float()
                enemy_pos = self.radar_pos[:, ei:ei+1, :].expand(-1, R, -1)
                enemy_vel = self.radar_vel[:, ei:ei+1, :].expand(-1, R, -1)
                delay_s, doppler_hz, gain = self.channel.compute_params_batch(
                    self.radar_pos, self.radar_vel,
                    enemy_pos[:, 0, :], enemy_vel[:, 0, :],
                    tx_power_w=self.tx_power_w,
                    rcs_dbsm=20.0,  # radar vehicle RCS
                    array_directivity_db=self.array.directivity_db,
                    n_elem=self.n_elem,
                    beam_az=beam_az,
                    beam_el=beam_el,
                    array_rotation=self.array_rotation,
                    bw_az_deg=bw_az, bw_el_deg=bw_el,
                )
                gain = gain * alive_mask.unsqueeze(1)
                echo = self.channel.apply_batch(tx_signal, delay_s, doppler_hz, gain)
                self._buf_rx += echo

        # --- 3. Interference ---
        self._buf_intf.zero_()
        avg_az = torch.zeros(E, R, device=dev)
        avg_el = torch.zeros(E, R, device=dev)
        baseband = tx_signal.mean(dim=2)
        weights = self.array.steer_all(avg_az, avg_el)
        self.interference.compute(
            self.radar_pos, avg_az, avg_el,
            weights, baseband[0, 0, :],
            out=self._buf_intf,
        )
        self._buf_rx += self._buf_intf

        # --- 4. Noise ---
        self.channel.generate_noise(out=self._buf_noise)
        self._buf_rx += self._buf_noise

        # --- 4b. WP3.2 damage: Weibull clutter (post-noise, pre-detection) ---
        if self.clutter_model == "weibull" and self.clutter_shape_k > 0:
            if self._buf_clutter is None:
                self._buf_clutter = torch.zeros_like(self._buf_rx)
            apply_weibull_clutter(
                self._buf_rx, self._buf_clutter,
                shape_k=self.clutter_shape_k,
                scale_lambda=self.clutter_scale_lambda,
                cnr_db=self.clutter_cnr_db,
                noise_power_linear=self.channel.noise_power_linear,
            )

        # --- 4c. WP3.2 damage: 2-ray multipath (FIR on aggregated rx) ---
        if self.multipath_model == "2ray" and self.multipath_delay_samples > 0:
            atten_lin = 10.0 ** (-abs(self.multipath_attenuation_db) / 20.0)
            apply_multipath_2ray(
                self._buf_rx,
                delay_samples=self.multipath_delay_samples,
                attenuation_linear=atten_lin,
            )

        # --- 5. Vehicle movement ---
        if vehicle_actions is not None:
            self._apply_vehicle_actions(vehicle_actions, dt)

        # --- 6. Drone BPSK comm ---
        self.battlefield.process_radar_comm(tx_signal, self.radar_pos, self.channel)

        # --- 7. Commander perfect link ---
        if commander_actions is not None:
            self.battlefield.process_commander_actions(commander_actions)

        # --- 8. Laser kill check ---
        kills = self.battlefield.step_lasers(dt, self.radar_pos)
        self.battlefield.update_alive(kills)
        dones, winners = self.battlefield.check_win()
        rewards = self.battlefield.compute_rewards(kills, dones, winners)

        # --- 9. Counter ---
        self._pulse_count += 1
        cpi_complete = (self._pulse_count % self.n_pulses == 0)
        pulses_per_control = 5  # 500μs / 100μs = 5 pulses = 2kHz NN rate
        nn_control_step = (self._pulse_count % pulses_per_control == 0)

        elapsed = time.perf_counter() - t0

        return {
            "rx_iq": self._buf_rx.clone(),
            "kills": kills,
            "dones": dones,
            "winners": winners,
            "radar_rewards": rewards["radar_rewards"],
            "commander_rewards": rewards["commander_rewards"],
            "illumination_progress": self.battlefield.laser.get_illumination_progress(),
            "cpi_complete": cpi_complete,
            "nn_control_step": nn_control_step,
            "pulse_count": self._pulse_count.clone(),
            "alive": self.battlefield.alive.clone(),
            "radar_pos": self.radar_pos.clone(),
            "timing_ms": elapsed * 1000,
        }

    def _apply_vehicle_actions(self, vehicle_actions: torch.Tensor, dt: float):
        """Update vehicle positions from action [E, R, 3].
        [..., 0] = speed [0,1], [..., 1] = heading_delta [0,1], [..., 2] = array_rot [0,1]
        """
        max_speed = self.vehicle_speed_ms
        self.radar_speed = vehicle_actions[..., 0].clamp(0, 1) * max_speed
        self.radar_heading = (self.radar_heading + vehicle_actions[..., 1] * 60.0) % 360.0
        self.array_rotation = (self.array_rotation + vehicle_actions[..., 2] * 60.0) % 360.0

        hdg_rad = torch.deg2rad(self.radar_heading)
        self.radar_vel[..., 0] = self.radar_speed * torch.cos(hdg_rad)
        self.radar_vel[..., 1] = self.radar_speed * torch.sin(hdg_rad)
        self.radar_vel[..., 2] = 0.0
        self.radar_pos = self.radar_pos + self.radar_vel * dt

        half_x = self.map_size[0] / 2.0
        half_y = self.map_size[1] / 2.0
        self.radar_pos[..., 0] = self.radar_pos[..., 0].clamp(-half_x, half_x)
        self.radar_pos[..., 1] = self.radar_pos[..., 1].clamp(-half_y, half_y)
