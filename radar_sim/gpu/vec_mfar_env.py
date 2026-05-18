"""MFAR (Multi-Function Phased Array Radar) vectorized environment.

Orchestrates per-element independent control for 4 tasks:
reconnaissance, detection, jamming, communication.

Architecture: E parallel environments × R radars × N elements × 4 tasks.
Each element independently chooses its task, beam direction, and waveform.

State  = [E, R, 625×(32×N_bins+2) + 5 + 12 + N_out]  (FFT spectra + comm + vehicle + missile + commander instruction)
Action = [E, R, 13753]                              (per-element 22-dim + vehicle 3-dim)
Commander obs   = [E, n_teams, 4 + 2*N_in]          (positions + radar latents)
Commander action = [E, n_teams, 3 + 2*N_out]         (launch + target + radar instructions)
"""

import time
import numpy as np
import torch

from .vec_array import VecArray
from .vec_channel import VecChannel
from .vec_interference import VecInterference
from .vec_element_processor import VecElementProcessor
from .vec_battlefield import VecBattlefield
from .vec_missile import swerling_gain_multiplier

SPEED_OF_LIGHT = 299792458.0
DEG2RAD = np.pi / 180.0

# Action layout per element (22 dims)
# [0:4]   task fractions (recon, detect, jam, comm) — argmax selects task
# [4:12]  beam steering (az, el) × 4 tasks
# [12:15] detect TX params (carrier_freq, BW, pulse_width)
# [15:18] jam TX params (BW, power, freq_shift)
# [18:22] comm TX params (carrier_freq, symbol_rate, data_X, data_Y)
ACTION_PER_ELEM = 22
# Vehicle action: [speed, heading_change, array_rotation]
ACTION_VEHICLE = 3
# Note: actual action_dim is computed per-instance as n_elem * ACTION_PER_ELEM + ACTION_VEHICLE


class MFARVecEnv:
    """Vectorized MFAR environment with per-element independent control."""

    def __init__(
        self, num_envs: int = 2, n_radars: int = 4,
        rows: int = 25, cols: int = 25,
        fc: float = 10e9, bandwidth: float = 200e6,
        prf: float = 10e3, pulses_per_cpi: int = 32,
        n_targets: int = 1, tx_power_w: float = 1.0,
        target_rcs_dbsm: float = 20.0,
        fft_size: int = 0, symbol_rate: float = 1e6,
        num_input_length: int = 32,
        num_output_length: int = 16,
        n_teams: int = 2,
        device: str = "cuda",
        # Configurable physical parameters (previously hardcoded):
        dx_wl: float = 0.5, dy_wl: float = 0.5,
        noise_figure_db: float = 5.0,
        map_size: tuple = (20000.0, 20000.0),
        speed_ms: float = 244.4,
        kill_radius_m: float = 500.0,
        missile_rcs_dbsm: float = 10.0,
        rcs_nose_dbsm: float = -5.0,
        rcs_side_dbsm: float = 12.0,
        rcs_tail_dbsm: float = 3.0,
        swerling_model: int = 3,
        red_launch_pos: tuple = (0.0, -10000.0),
        blue_launch_pos: tuple = (0.0, 10000.0),
        polarization_loss_db: float = 3.0,
        tx_rx_isolation_db: float = 25.0,
        reset_config: dict = None,
        reward_config: dict = None,
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
        self.reset_config = reset_config or {}
        self.reward_config = reward_config or {}

        self.pri = 1.0 / prf
        self.n_samples = max(1, int(self.pri * self.fs))
        E, R, N = num_envs, n_radars, self.n_elem
        S = self.n_samples

        dev_torch = torch.device(device)

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
        self.processor = VecElementProcessor(
            fs=self.fs, n_samples=S,
            pulses_per_cpi=pulses_per_cpi,
            fft_size=fft_size, symbol_rate=symbol_rate,
            device=device,
        )

        self.n_bins = self.processor.n_bins

        # --- Pre-allocated GPU buffers ---
        # Per-element per-pulse IQ: [E, R, N, P, S]
        self._buf_rx_signal = torch.zeros(
            E, R, N, S, dtype=torch.complex64, device=dev_torch,
        )
        self._buf_tx = torch.zeros(
            E, R, N, S, dtype=torch.complex64, device=dev_torch,
        )
        self._buf_noise = torch.zeros(
            E, R, N, S, dtype=torch.complex64, device=dev_torch,
        )
        self._buf_intf = torch.zeros(
            E, R, N, S, dtype=torch.complex64, device=dev_torch,
        )
        # CPI buffer: per-element per-pulse IQ [E, R, N, P, S]
        self._buf_cpi = torch.zeros(
            E, R, N, self.n_pulses, S, dtype=torch.complex64, device=dev_torch,
        )

        # Spectrum and comm output buffers (pre-allocated, reused each step)
        self._buf_spectrum = torch.zeros(
            E, R, N, self.n_pulses, self.n_bins, dtype=torch.float32, device=dev_torch,
        )
        self._buf_comm_data = torch.zeros(
            E, R, N, 2, dtype=torch.float32, device=dev_torch,
        )

        # DRFM capture buffer: [E, R, S] complex64
        self._captured_signal = torch.zeros(
            E, R, S, dtype=torch.complex64, device=dev_torch,
        )

        # --- State tensors ---
        self.radar_pos = torch.zeros(E, R, 3, device=dev_torch)
        self.radar_vel = torch.zeros(E, R, 3, device=dev_torch)
        self.radar_heading = torch.zeros(E, R, device=dev_torch)
        self.radar_speed = torch.zeros(E, R, device=dev_torch)
        self.array_rotation = torch.zeros(E, R, device=dev_torch)

        self.target_pos = torch.zeros(E, n_targets, 3, device=dev_torch)
        self.target_vel = torch.zeros(E, n_targets, 3, device=dev_torch)

        # Commander instruction buffer (set from commander action each step)
        self._commander_instructions = torch.zeros(
            E, R, num_output_length, device=dev_torch,
        )

        # Element positions (shared, computed from array geometry)
        dx_m = self.dx_wl * self.array.wavelength
        dy_m = self.dy_wl * self.array.wavelength
        x_pos = (np.arange(cols) - (cols - 1) / 2.0) * dx_m
        y_pos = (np.arange(rows) - (rows - 1) / 2.0) * dy_m
        X, Y = np.meshgrid(x_pos, y_pos)
        self.elem_x = torch.tensor(
            X.ravel().astype(np.float32), device=dev_torch,
        )
        self.elem_y = torch.tensor(
            Y.ravel().astype(np.float32), device=dev_torch,
        )

        # --- Battlefield (missile combat + game state) ---
        self.battlefield = VecBattlefield(
            num_envs=num_envs, n_radars=n_radars, n_teams=n_teams,
            fs=self.fs, symbol_rate=symbol_rate, device=device,
            num_input_length=num_input_length,
            num_output_length=num_output_length,
            map_size=map_size,
            speed_ms=speed_ms,
            kill_radius_m=kill_radius_m,
            missile_rcs_dbsm=missile_rcs_dbsm,
            rcs_nose_dbsm=rcs_nose_dbsm,
            rcs_side_dbsm=rcs_side_dbsm,
            rcs_tail_dbsm=rcs_tail_dbsm,
            swerling_model=swerling_model,
            red_launch_pos=red_launch_pos,
            blue_launch_pos=blue_launch_pos,
            reward_config=reward_config,
        )

    @property
    def state_dim(self) -> int:
        missile_dims = 6 + self.n_teams * 3  # own missile 6 + all missiles awareness 6
        return (self.n_elem * (self.n_pulses * self.n_bins + 2 + 4)
                + 5 + missile_dims + self.num_output_length)

    @property
    def action_dim(self) -> int:
        return self.n_elem * ACTION_PER_ELEM + ACTION_VEHICLE

    def reset(self, env_ids=None):
        """Randomize positions for specified envs (or all).

        Team layout (20km map, origin at center):
          Red (radars 0,1): y < 0 half
          Blue (radars 2,3): y > 0 half
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        E = len(env_ids)
        dev = torch.device(self.device)

        # Team-based positions: Red in y<0, Blue in y>0
        r_per_team = self.n_radars // self.n_teams
        rc = self.reset_config
        pos_spread_x = rc.get('position_spread_x', 8000.0)
        pos_spread_y = rc.get('position_spread_y', 6000.0)
        y_offset = rc.get('y_center_offset', 5000.0)
        vel_range = rc.get('velocity_range', 20.0)
        hdg_range = rc.get('heading_range', 360.0)
        spd_range = rc.get('speed_range', 8.0)
        arr_rot_range = rc.get('array_rotation_range', 120.0)
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

        self.radar_vel[env_ids] = (torch.rand(E, self.n_radars, 3, device=dev) - 0.5) * vel_range
        self.radar_heading[env_ids] = torch.rand(E, self.n_radars, device=dev) * hdg_range
        self.radar_speed[env_ids] = torch.rand(E, self.n_radars, device=dev) * spd_range
        self.array_rotation[env_ids] = (torch.rand(E, self.n_radars, device=dev) - 0.5) * arr_rot_range

        r = tgt_dist_min + torch.rand(E, self.n_targets, device=dev) * tgt_dist_range
        ang = torch.rand(E, self.n_targets, device=dev) * 2 * np.pi
        self.target_pos[env_ids, :, 0] = r * torch.cos(ang)
        self.target_pos[env_ids, :, 1] = r * torch.sin(ang)
        self.target_pos[env_ids, :, 2] = 0.0
        self.target_vel[env_ids] = (torch.rand(E, self.n_targets, 3, device=dev) - 0.5) * tgt_vel_range

        self.battlefield.reset(env_ids)

    def step(self, actions: torch.Tensor = None,
             commander_actions: torch.Tensor = None,
             radar_latents: torch.Tensor = None) -> dict:
        """Run one CPI for all envs.

        Args:
            actions: [E, R, action_dim] float32. If None, uses default (all detect).
            commander_actions: [E, n_teams, commander_action_dim] float32.
                Layout: [launch_flag, target_x, target_y, inst_0..., inst_1...]
                If None, no missile launch.
            radar_latents: [E, R, num_input_length] float32 from radar NN encoder.
                If None, commander_obs will be zero-filled.
        Returns:
            dict with keys: state, spectrum, comm_data, task_ids, timing, tx_signal,
                            commander_obs, radar_instructions, radar_rewards,
                            commander_rewards, dones, winners, missile_pos, kills
        """
        E, R, N = self.num_envs, self.n_radars, self.n_elem
        S, P = self.n_samples, self.n_pulses
        dev = torch.device(self.device)

        t0 = time.perf_counter()

        # --- Phase 0: Commander actions (missile launch + radar instructions) ---
        if commander_actions is not None:
            self.battlefield.process_commander_actions(
                commander_actions, self.radar_pos,
            )
            self._commander_instructions = self.battlefield.extract_radar_instructions(
                commander_actions,
            )
        else:
            self._commander_instructions.zero_()

        # --- Phase 1: Decode radar actions ---
        if actions is not None:
            task_ids, beam_az, beam_el, wf_types, detect_params, jam_params, comm_params = (
                self._decode_actions(actions)
            )
            self._apply_vehicle_actions(actions)
        else:
            # Default: all elements detect with LFM, beam at 0°
            task_ids = torch.ones(E, R, N, dtype=torch.long, device=dev)
            beam_az = torch.zeros(E, R, N, device=dev)
            beam_el = torch.zeros(E, R, N, device=dev)
            wf_types = torch.zeros(E, R, N, dtype=torch.long, device=dev)
            detect_params = torch.zeros(E, R, N, 3, device=dev)
            jam_params = torch.zeros(E, R, N, 3, device=dev)
            comm_params = torch.zeros(E, R, N, 3, device=dev)

        t_action = time.perf_counter()

        # --- Phase 2: Assemble TX signals ---
        ex = self._get_elem_x()
        ey = self._get_elem_y()

        tx_signal = self.processor.assemble_tx_per_element(
            task_ids, beam_az, beam_el, wf_types,
            detect_params, jam_params, comm_params,
            ex, ey, self.array.wavelength, S,
            captured_signal=self._captured_signal,
        )
        t_tx = time.perf_counter()

        # --- Interference ---
        avg_az = beam_az.float().mean(dim=-1)
        avg_el = beam_el.float().mean(dim=-1)
        baseband = tx_signal.mean(dim=2)
        self._buf_intf.zero_()
        weights_for_intf = self.array.steer_all(avg_az, avg_el)
        self.interference.compute(
            self.radar_pos, avg_az, avg_el,
            weights_for_intf, baseband[0, 0, :],
            out=self._buf_intf,
        )

        t_intf = time.perf_counter()

        # --- Phase 3: CPI pulse loop (with missile targets) ---
        waveform_refs = self._build_waveform_refs(task_ids, wf_types, detect_params, comm_params)

        missile = self.battlefield.missile

        # Pre-compute aspect RCS correction [E, n_teams, R] (constant within CPI)
        aspect_correction = missile.compute_aspect_rcs_correction(self.radar_pos)

        # Pre-compute slow Swerling multiplier (constant within CPI)
        swerling_slow = None
        if missile.swerling_model in (1, 3):
            swerling_slow = swerling_gain_multiplier(
                (E, self.n_teams, R), missile.swerling_model, dev,
            )

        # Pre-compute channel params outside the pulse loop (positions constant within CPI)
        static_params = []
        for t_idx in range(self.n_targets):
            delay_s, doppler_hz, gain = self.channel.compute_params_batch(
                self.radar_pos, self.radar_vel,
                self.target_pos[:, t_idx], self.target_vel[:, t_idx],
                tx_power_w=self.tx_power_w,
                rcs_dbsm=self.target_rcs_dbsm,
                array_directivity_db=self.array.directivity_db,
                n_elem=self.n_elem,
            )
            static_params.append((delay_s, doppler_hz, gain))

        missile_params = []
        for team_idx in range(self.n_teams):
            flying = missile.in_flight[:, team_idx]
            if not flying.any():
                missile_params.append(None)
                continue
            m_pos = missile.missile_pos[:, team_idx]
            m_vel = missile.missile_vel[:, team_idx]
            delay_s, doppler_hz, gain = self.channel.compute_params_batch(
                self.radar_pos, self.radar_vel, m_pos, m_vel,
                tx_power_w=self.tx_power_w,
                rcs_dbsm=missile.rcs_dbsm,
                array_directivity_db=self.array.directivity_db,
                n_elem=self.n_elem,
            )
            gain = gain * aspect_correction[:, team_idx]
            if swerling_slow is not None:
                gain = gain * swerling_slow[:, team_idx]
            gain = gain * flying.float().unsqueeze(1)
            missile_params.append((delay_s, doppler_hz, gain))

        # Track last-computed channel params for evaluation
        last_channel_params = None
        if static_params:
            last_channel_params = {
                "delay_samples": static_params[0][0].detach(),
                "doppler_hz": static_params[0][1].detach(),
                "gain_linear": static_params[0][2].detach(),
            }

        for p in range(P):
            self._buf_rx_signal.zero_()

            # Static targets (use pre-computed params)
            for delay_s, doppler_hz, gain in static_params:
                target_return = self.channel.apply_batch(
                    tx_signal, delay_s, doppler_hz, gain,
                )
                self._buf_rx_signal += target_return

            # Missile targets (pre-computed params, Swerling 2/4 re-draw per pulse)
            for team_idx, params in enumerate(missile_params):
                if params is None:
                    continue
                delay_s, doppler_hz, gain = params
                if missile.swerling_model in (2, 4):
                    fast_mult = swerling_gain_multiplier(
                        (E, R), missile.swerling_model, dev,
                    )
                    gain = gain * fast_mult

                target_return = self.channel.apply_batch(
                    tx_signal, delay_s, doppler_hz, gain,
                )
                self._buf_rx_signal += target_return

            self._buf_rx_signal += self._buf_intf

            # Self-interference: TX→RX leakage within same array
            if self.tx_rx_isolation_db < 200.0:
                coupling = 10.0 ** (-self.tx_rx_isolation_db / 20.0)
                tx_active = (task_ids != 0)   # non-recon elements transmit
                rx_active = (task_ids != 2)   # non-jam elements receive
                si = tx_signal * coupling      # [E, R, N, S]
                self._buf_rx_signal += si * rx_active.unsqueeze(-1).float()

            self.channel.generate_noise(out=self._buf_noise)
            self._buf_rx_signal += self._buf_noise
            self._buf_cpi[:, :, :, p, :] = self._buf_rx_signal

        t_pulses = time.perf_counter()

        # --- Phase 3.5: DRFM signal capture ---
        # Store the mean RX signal per radar for DRFM retransmission next step
        self._captured_signal = self._buf_cpi[:, :, :, -1, :].mean(dim=2).detach()

        # --- Phase 4: RX processing ---
        spectrum = self._buf_spectrum.zero_()
        comm_data = self._buf_comm_data.zero_()

        # Build waveform refs dict for single-pass FFT
        wf_refs = {}
        recon_mask = (task_ids == 0)
        detect_mask = (task_ids == 1)
        comm_mask = (task_ids == 3)

        if recon_mask.any():
            wf_refs[0] = None
        if detect_mask.any():
            wf_refs[1] = self._build_detect_ref(wf_types, detect_params)
        if comm_mask.any():
            comm_ref = self._build_comm_ref(comm_params)
            wf_refs[3] = comm_ref

        if wf_refs:
            spec_results = self.processor.process_rx_cpi_unified(
                self._buf_cpi, wf_refs,
            )
            for task_id, spec in spec_results.items():
                mask = (task_ids == task_id)
                spectrum = torch.where(
                    mask.unsqueeze(-1).unsqueeze(-1),
                    spec, spectrum,
                )

        if comm_mask.any():
            comm_xy = self.processor.process_rx_comm(self._buf_cpi, wf_refs[3])
            comm_data = torch.where(
                comm_mask.unsqueeze(-1),
                comm_xy, comm_data,
            )

        # Recon intelligence: extract signal parameters from recon spectrum
        recon_intel = torch.zeros(
            E, R, self.n_elem, 4, dtype=torch.float32, device=dev,
        )
        if recon_mask.any():
            recon_spec = spectrum.clone()
            ri = self.processor.process_rx_recon(recon_spec)
            recon_intel = torch.where(
                recon_mask.unsqueeze(-1),
                ri, recon_intel,
            )

        t_rx = time.perf_counter()

        # --- Phase 5: Radar→missile BPSK communication ---
        self.battlefield.process_missile_comm(
            self.radar_pos, task_ids, comm_params, tx_signal,
            self.channel, self.processor,
        )

        # --- Phase 6: Missile physics + kill check ---
        dt = self.pri * self.n_pulses
        kills = self.battlefield.step_missiles(dt, self.radar_pos)
        self.battlefield.update_alive(kills)
        dones, winners = self.battlefield.check_win()
        rewards = self.battlefield.compute_rewards(kills, dones, winners)

        t_missile = time.perf_counter()

        # --- Phase 7: Assemble state ---
        state = self._assemble_state(spectrum, comm_data, recon_intel)

        if radar_latents is not None:
            commander_obs = self.battlefield.get_commander_observation(
                self.radar_pos, radar_latents,
            )
        else:
            commander_obs = torch.zeros(
                E, self.n_teams, self.battlefield.commander_obs_dim, device=dev,
            )

        timing = {
            "action_ms": (t_action - t0) * 1000,
            "tx_ms": (t_tx - t_action) * 1000,
            "interference_ms": (t_intf - t_tx) * 1000,
            "pulses_ms": (t_pulses - t_intf) * 1000,
            "rx_ms": (t_rx - t_pulses) * 1000,
            "missile_ms": (t_missile - t_rx) * 1000,
            "total_ms": (t_missile - t0) * 1000,
        }

        # TC-DAMS task-allocation fingerprint per team: [E, n_teams, 4]
        # entry [e, t, k] = fraction of elements on team t assigned task k.
        # task_ids in {0=recon, 1=detect, 2=jam, 3=comm}, shape [E, R, N].
        r_per_team = R // self.n_teams
        one_hot = torch.nn.functional.one_hot(
            task_ids.clamp(0, 3), num_classes=4,
        ).to(torch.float32)  # [E, R, N, 4]
        # Sum over N (elements) then over radars within each team, normalize.
        team_counts = one_hot.sum(dim=2)  # [E, R, 4]
        team_counts = team_counts.view(E, self.n_teams, r_per_team, 4).sum(dim=2)
        task_fingerprint = team_counts / max(r_per_team * N, 1)  # [E, n_teams, 4]

        return {
            "state": state,
            "spectrum": spectrum,
            "comm_data": comm_data,
            "task_ids": task_ids,
            "task_fingerprint": task_fingerprint,
            "commander_obs": commander_obs,
            "radar_instructions": self._commander_instructions,
            "radar_rewards": rewards["radar_rewards"],
            "commander_rewards": rewards["commander_rewards"],
            "dones": dones,
            "winners": winners,
            "missile_pos": missile.missile_pos,
            "kills": kills,
            "timing": timing,
            "tx_signal": tx_signal,
            "channel_params": last_channel_params,
            "steering_weights": weights_for_intf,
            "detect_params": detect_params,
            "jam_params": jam_params,
            "comm_crc_ok": getattr(self.battlefield, "_last_comm_crc_ok", None),
        }

    def _decode_actions(self, actions: torch.Tensor):
        """Decode [E, R, 13753] actions into per-element parameters."""
        E, R = actions.shape[:2]
        N = self.n_elem
        dev = torch.device(self.device)

        # Per-element actions: [E, R, 13753] → [E, R, N, 22]
        elem_actions = actions[:, :, :N * ACTION_PER_ELEM].reshape(E, R, N, ACTION_PER_ELEM)

        # Task assignment: argmax of first 4 dims
        task_ids = elem_actions[..., 0:4].argmax(dim=-1)  # [E, R, N]

        # Beam steering per element: select (az, el) for the assigned task
        # [4:12] = (az, el) × 4 tasks — pick the one matching task_id
        all_az = elem_actions[..., 4:12:2]  # [E, R, N, 4]
        all_el = elem_actions[..., 5:12:2]  # [E, R, N, 4]
        task_idx = task_ids.unsqueeze(-1)     # [E, R, N, 1]
        beam_az = all_az.gather(-1, task_idx).squeeze(-1) * 60.0   # scale to ±60°
        beam_el = all_el.gather(-1, task_idx).squeeze(-1) * 45.0   # scale to ±45°

        # Waveform type (shared across tasks, used as index)
        wf_types = (elem_actions[..., 4:6].argmax(dim=-1)).long()  # simplified

        # Task-specific params
        detect_params = elem_actions[..., 12:15]  # [E, R, N, 3]
        jam_params = elem_actions[..., 15:18]     # [E, R, N, 3]
        comm_params = elem_actions[..., 18:22]    # [E, R, N, 4] → take first 3

        return task_ids, beam_az, beam_el, wf_types, detect_params, jam_params, comm_params[..., :3]

    def _apply_vehicle_actions(self, actions: torch.Tensor):
        """Update vehicle state from action's last 3 dims."""
        E, R = actions.shape[:2]
        vehicle_action = actions[:, :, -ACTION_VEHICLE:]  # [E, R, 3]
        self.radar_speed = vehicle_action[..., 0].clamp(0, 8.33) * 8.33
        self.radar_heading = (self.radar_heading + vehicle_action[..., 1] * 60.0) % 360.0
        self.array_rotation = (self.array_rotation + vehicle_action[..., 2] * 60.0) % 360.0

    def _build_waveform_refs(self, task_ids, wf_types, detect_params, comm_params):
        """Build waveform reference tensors for matched filtering."""
        # Returns dict mapping task_id → reference waveform tensor
        # For simplicity, use LFM for all detect elements
        E, R, N = task_ids.shape
        dev = torch.device(self.device)

        detect_mask = (task_ids == 1)
        if detect_mask.any():
            # Generate a default LFM reference
            from .waveform_gpu import generate_lfm
            ref = generate_lfm(
                self.pri * 0.1, self.bandwidth, self.fs, dev, "up",
            )
            # Pad to n_samples
            if ref.shape[0] < self.n_samples:
                pad = torch.zeros(
                    self.n_samples - ref.shape[0],
                    dtype=torch.complex64, device=dev,
                )
                ref = torch.cat([ref, pad])
            ref = ref[:self.n_samples]
            return ref
        return None

    def _build_detect_ref(self, wf_types, detect_params):
        """Build matched filter reference for detection elements."""
        E, R, N = wf_types.shape
        dev = torch.device(self.device)

        from .waveform_gpu import generate_lfm
        ref = generate_lfm(self.pri * 0.1, self.bandwidth, self.fs, dev, "up")
        if ref.shape[0] < self.n_samples:
            pad = torch.zeros(
                self.n_samples - ref.shape[0],
                dtype=torch.complex64, device=dev,
            )
            ref = torch.cat([ref, pad])
        return ref[:self.n_samples]

    def _build_comm_ref(self, comm_params):
        """Build BPSK reference for communication elements."""
        E, R, N = comm_params.shape[:3]
        dev = torch.device(self.device)

        from .waveform_gpu import encode_bpsk, modulate_bpsk
        bits = encode_bpsk(0.0, 0.0)
        ref = modulate_bpsk(bits, self.n_samples, self.fs, self.processor.symbol_rate, dev)
        return ref

    def _assemble_state(self, spectrum, comm_data, recon_intel=None):
        """Assemble state vector from spectrum, comm data, recon intel, vehicle, and missile awareness.

        Args:
            spectrum: [E, R, N, P, n_bins] float32
            comm_data: [E, R, N, 2] float32
            recon_intel: [E, R, N, 4] float32 or None
        Returns:
            [E, R, state_dim] float32
        """
        E, R, N, P, B = spectrum.shape

        spec_flat = spectrum.reshape(E, R, N * P * B)
        comm_flat = comm_data.reshape(E, R, N * 2)

        if recon_intel is None:
            recon_flat = torch.zeros(E, R, N * 4, dtype=torch.float32, device=spectrum.device)
        else:
            recon_flat = recon_intel.reshape(E, R, N * 4)

        vehicle = torch.stack([
            self.radar_pos[..., 0],
            self.radar_pos[..., 1],
            self.radar_heading,
            self.radar_speed,
            self.array_rotation,
        ], dim=-1)

        missile_state = self._build_missile_state_per_radar()

        parts = [spec_flat, comm_flat, recon_flat, vehicle, missile_state]
        if self.num_output_length > 0:
            parts.append(self._commander_instructions)

        return torch.cat(parts, dim=-1)

    def _build_missile_state_per_radar(self) -> torch.Tensor:
        """Build per-radar missile awareness tensor [E, R, 6 + n_teams*3].

        Per radar:
          [0:6] own team missile: pos_x, pos_y, pos_z, in_flight, target_x, target_y
          [6:6+n_teams*3] all missiles: pos_x, pos_y, in_flight per team
        """
        E, R = self.num_envs, self.n_radars
        n_t = self.n_teams
        dev = torch.device(self.device)
        m = self.battlefield.missile

        result = torch.zeros(E, R, 6 + n_t * 3, device=dev)

        for r in range(R):
            team = self.battlefield.team_id[r].item()
            # Own team missile
            result[:, r, 0] = m.missile_pos[:, team, 0]
            result[:, r, 1] = m.missile_pos[:, team, 1]
            result[:, r, 2] = m.missile_pos[:, team, 2]
            result[:, r, 3] = m.in_flight[:, team].float()
            result[:, r, 4] = m.target_pos[:, team, 0]
            result[:, r, 5] = m.target_pos[:, team, 1]
            # All missiles awareness
            for t in range(n_t):
                base = 6 + t * 3
                result[:, r, base] = m.missile_pos[:, t, 0]
                result[:, r, base + 1] = m.missile_pos[:, t, 1]
                result[:, r, base + 2] = m.in_flight[:, t].float()

        return result

    def _get_elem_x(self):
        return self.elem_x

    def _get_elem_y(self):
        return self.elem_y
