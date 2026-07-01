"""Complete radar policy: signal processing (DSP) + neural network (NN).

Encapsulates the entire signal processing chain and decision-making
into a single module. Internally iterable: v1 (DSP+NN) → v2 (end-to-end)
→ v3 (deploy to real radar).

This module sits between the training loop and the physics engine:
  - Input:  events dict from env (rx_iq, kills, done, ...)
  - Output: tx_pulse [E,R,N,S] complex64 + commander_actions [E,T,5]
"""

import torch
import torch.nn as nn
import numpy as np

from radar_sim.gpu.vec_element_processor import VecElementProcessor
from radar_sim.gpu.waveform_gpu import encode_bpsk, modulate_bpsk

from .ppo.actor_critic import (
    SubArrayRadarActorCritic,
    CommanderActorCritic,
    create_team_policy,
)


class CPIAccumulator:
    """Accumulates RX pulses into CPI buffers for FFT processing."""

    def __init__(self, num_envs, n_radars, n_elem, n_pulses, n_samples, device="cuda"):
        self.num_envs = num_envs
        self.n_radars = n_radars
        self.n_elem = n_elem
        self.n_pulses = n_pulses
        self.n_samples = n_samples
        self.device = device
        dev = torch.device(device)
        # [E, R, N, P, S] complex64
        self.buffer = torch.zeros(
            num_envs, n_radars, n_elem, n_pulses, n_samples,
            dtype=torch.complex64, device=dev,
        )
        self.pulse_idx = 0

    def append(self, rx_pulse: torch.Tensor):
        """Append one pulse [E, R, N, S] to the buffer."""
        idx = self.pulse_idx % self.n_pulses
        self.buffer[:, :, :, idx, :] = rx_pulse
        self.pulse_idx += 1

    def is_complete(self) -> bool:
        return self.pulse_idx > 0 and self.pulse_idx % self.n_pulses == 0

    def data(self) -> torch.Tensor:
        return self.buffer

    def reset(self):
        self.buffer.zero_()
        self.pulse_idx = 0


class RadarPolicy:
    """Complete radar policy: DSP chain + ActorCritic NN.

    v1: fixed DSP (FFT, matched filter, beamforming) + ActorCritic NN — current
    v2: end-to-end NN → raw IQ — future
    v3: deploy to real radar — sim-to-real

    Usage:
        policy = RadarPolicy(env_config)
        for pulse in range(max_pulses):
            tx, cmd = policy.act(events)
            rx, events = env.step(tx, cmd)
            policy.receive_rx(rx, events)
    """

    def __init__(
        self,
        num_envs: int = 2,
        n_radars: int = 4,
        n_teams: int = 2,
        n_elem: int = 625,
        n_pulses: int = 4,
        n_samples: int = 20000,
        n_bins: int = 20000,
        fft_size: int = 0,
        fs: float = 200e6,
        symbol_rate: float = 1e6,
        pulses_per_control: int = 5,
        sub_array_size: int = 5,
        num_input_length: int = 32,
        num_output_length: int = 16,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.n_radars = n_radars
        self.n_teams = n_teams
        self.n_elem = n_elem
        self.n_pulses = n_pulses
        self.n_samples = n_samples
        self.n_bins = fft_size if fft_size > 0 else n_samples
        self.fs = fs
        self.symbol_rate = symbol_rate
        self.pulses_per_control = pulses_per_control
        self.device = device
        self.num_input_length = num_input_length
        self.num_output_length = num_output_length

        E, R, N = num_envs, n_radars, n_elem
        dev = torch.device(device)

        # CPI accumulator
        self.cpi_buffer = CPIAccumulator(
            num_envs, n_radars, n_elem, n_pulses, n_samples, device=device,
        )

        # Signal processor (fixed, not learnable)
        self.processor = VecElementProcessor(
            fs=fs, n_samples=n_samples,
            pulses_per_cpi=n_pulses,
            fft_size=fft_size, symbol_rate=symbol_rate,
            device=device,
        )

        # Neural networks (learnable)
        self.radar_ac = SubArrayRadarActorCritic(
            n_elem=n_elem, n_pulses=n_pulses, n_bins=self.n_bins,
            sub_array_size=sub_array_size,
            commander_instr_dim=num_output_length,
        ).to(dev)

        self.commander_ac = CommanderActorCritic(
            obs_dim=4 + 2 * num_input_length + 8,
            act_dim=5,
        ).to(dev)

        # Current actions (reused between control steps)
        self._current_radar_actions = None
        self._current_cmd_actions = torch.zeros(E, n_teams, 5, device=dev)
        self._current_tx = torch.zeros(E, R, N, n_samples, dtype=torch.complex64, device=dev)

        # Detection results (from last CPI processing)
        self._spectrum = None
        self._detections = None

        # Pulse counter for control step timing
        self._pulse_count = 0

    def act(self, events: dict):
        """Generate TX pulse and commander actions.

        Called every pulse. Internally manages CPI accumulation and
        NN control timing.

        Args:
            events: dict from env.step() — may contain cpi_complete, nn_control_step
        Returns:
            tx_pulse: [E, R, N, S] complex64
            commander_actions: [E, n_teams, 5]
            vehicle_actions: [E, R, 3]
        """
        E, R, N = self.num_envs, self.n_radars, self.n_elem
        S = self.n_samples
        dev = torch.device(self.device)

        # CPI processing: if CPI complete, run FFT+MF+BF
        cpi_complete = events.get('cpi_complete', False)
        if isinstance(cpi_complete, torch.Tensor):
            cpi_complete = cpi_complete.any()
        if cpi_complete:
            self._process_cpi()

        # NN control step: if due, run NN decision
        nn_control = events.get('nn_control_step', False)
        if isinstance(nn_control, torch.Tensor):
            nn_control = nn_control.any()
        if nn_control:
            self._nn_decide(events)

        self._pulse_count += 1

        return (
            self._current_tx,
            self._current_cmd_actions,
            self._get_vehicle_actions(),
        )

    def receive_rx(self, rx_pulse: torch.Tensor, events: dict):
        """Accumulate received pulse into CPI buffer."""
        self.cpi_buffer.append(rx_pulse)

    def _process_cpi(self):
        """Run FFT + matched filter + beamforming on accumulated CPI data."""
        cpi_data = self.cpi_buffer.data()  # [E, R, N, P, S]

        # FFT across samples dimension
        spectrum = torch.fft.fft(cpi_data, n=self.n_bins, dim=-1)  # [E, R, N, P, n_bins]
        self._spectrum = spectrum.abs().float()  # magnitude for NN input

        # Reset CPI buffer for next accumulation
        self.cpi_buffer.reset()

    def _nn_decide(self, events: dict):
        """Build observations and run NN decision."""
        dev = torch.device(self.device)
        E, R = self.num_envs, self.n_radars

        # Build radar observation from spectrum
        if self._spectrum is not None:
            # Flatten spectrum: [E, R, N * P * n_bins]
            spec_flat = self._spectrum.reshape(E, R, -1)
            # For SubArrayRadarActorCritic, we need the full state vector
            # For now, use zeros for non-spectrum parts (will be filled by training loop)
            comm_flat = torch.zeros(E, R, self.n_elem * 2, device=dev)
            recon_flat = torch.zeros(E, R, self.n_elem * 4, device=dev)
            vehicle = torch.zeros(E, R, 5, device=dev)
            laser_state = torch.zeros(E, R, 12, device=dev)  # matches SubArrayRadarActorCritic missile_dim=12
            cmd_instr = torch.zeros(E, R, self.num_output_length, device=dev)

            state = torch.cat([spec_flat, comm_flat, recon_flat, vehicle, laser_state, cmd_instr], dim=-1)
        else:
            # No CPI data yet — use zeros
            state = torch.zeros(E, R, self.radar_ac.spectrum_flat_dim +
                                self.radar_ac.comm_flat_dim +
                                self.radar_ac.recon_flat_dim +
                                self.radar_ac.other_dim, device=dev)

        # Radar actions — flatten [E, R, state_dim] → [E*R, state_dim] for NN
        state_flat = state.reshape(E * R, -1)
        radar_action, _, _, _ = self.radar_ac.get_action(state_flat)
        self._current_radar_actions = radar_action.reshape(E, R, -1)

        # Assemble TX signal from radar actions
        self._current_tx = self._assemble_tx(self._current_radar_actions)

        # Commander actions — reshape [E, T, 76] → [E*T, 76]
        commander_obs = events.get('commander_obs', None)
        if commander_obs is None:
            commander_obs = torch.zeros(E, self.n_teams, 76, device=dev)
        cmd_flat = commander_obs.reshape(E * self.n_teams, -1)
        cmd_action, _, _, _ = self.commander_ac.get_action(cmd_flat)
        self._current_cmd_actions = cmd_action.reshape(E, self.n_teams, -1)

    def _assemble_tx(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert high-level actions [E, R, action_dim] to TX signal [E, R, N, S].

        Decodes per-element actions (beam, task, waveform) and assembles
        the TX signal using the element processor.
        """
        E, R = actions.shape[:2]
        N = self.n_elem
        S = self.n_samples
        dev = torch.device(self.device)
        ACTION_PER_ELEM = 22

        # Decode actions
        elem_actions = actions[:, :, :N * ACTION_PER_ELEM].reshape(E, R, N, ACTION_PER_ELEM)

        # Task assignment
        task_ids = elem_actions[..., 0:4].argmax(dim=-1)  # [E, R, N]

        # Beam steering
        all_az = elem_actions[..., 4:12:2]  # [E, R, N, 4]
        all_el = elem_actions[..., 5:12:2]
        task_idx = task_ids.unsqueeze(-1)
        beam_az = all_az.gather(-1, task_idx).squeeze(-1) * 60.0
        beam_el = all_el.gather(-1, task_idx).squeeze(-1) * 45.0

        # Waveform + params
        wf_types = elem_actions[..., 4:6].argmax(dim=-1).long()
        detect_params = elem_actions[..., 12:15]
        jam_params = elem_actions[..., 15:18]
        comm_params = elem_actions[..., 18:22]

        # BPSK comm payload: radar NN outputs position estimate
        # For comm elements, use NN's estimated position (dims 20:22 of action)
        # The BPSK encode uses these as normalized coordinates

        # Assemble TX signal using processor
        ex = torch.zeros(N, device=dev)
        ey = torch.zeros(N, device=dev)
        wavelength = 3e8 / 10e9  # fc=10GHz

        tx_signal = self.processor.assemble_tx_per_element(
            task_ids, beam_az, beam_el, wf_types,
            detect_params, jam_params, comm_params[..., :3],
            ex, ey, wavelength, S,
        )

        return tx_signal

    def _get_vehicle_actions(self) -> torch.Tensor:
        """Extract vehicle actions from current radar actions."""
        if self._current_radar_actions is None:
            return torch.zeros(
                self.num_envs, self.n_radars, 3,
                device=torch.device(self.device),
            )
        return self._current_radar_actions[:, :, -3:]

    def get_obs(self, events: dict) -> dict:
        """Build observations for PPO training.

        Returns:
            {"radar_obs": [E, R, state_dim], "commander_obs": [E, n_teams, 76]}
        """
        dev = torch.device(self.device)
        E, R = self.num_envs, self.n_radars

        if self._spectrum is not None:
            spec_flat = self._spectrum.reshape(E, R, -1)
        else:
            spec_flat = torch.zeros(E, R, self.n_elem * self.n_pulses * self.n_bins, device=dev)

        comm_flat = torch.zeros(E, R, self.n_elem * 2, device=dev)
        recon_flat = torch.zeros(E, R, self.n_elem * 4, device=dev)
        vehicle = torch.zeros(E, R, 5, device=dev)
        laser_state = torch.zeros(E, R, 12, device=dev)  # matches missile_dim=12
        cmd_instr = torch.zeros(E, R, self.num_output_length, device=dev)

        if 'radar_pos' in events:
            vehicle[:, :, 0] = events['radar_pos'][:, :, 0]
            vehicle[:, :, 1] = events['radar_pos'][:, :, 1]

        radar_obs = torch.cat([spec_flat, comm_flat, recon_flat, vehicle, laser_state, cmd_instr], dim=-1)

        commander_obs = events.get('commander_obs',
            torch.zeros(E, self.n_teams, 76, device=dev))

        return {"radar_obs": radar_obs, "commander_obs": commander_obs}
