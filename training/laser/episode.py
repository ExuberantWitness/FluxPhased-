"""Laser episode runner: pulse-level loop adapter between FluxLeague and MFARVecEnv.

Why this exists:
  - `MFARVecEnv.step()` runs ONE pulse and returns `rx_iq` (raw CPI sample).
  - Radar policy needs an FFT-processed CPI buffer (N pulses accumulated).
  - The existing `FluxLeague._train_against` / `PayoffMatrix.evaluate_pair`
    assumed `env.step(actions=, commander_actions=)` and synchronous state
    query via `env._assemble_state()` — neither exists in the current env.
  - This runner owns the pulse loop (run `pulses_per_control` env.step()
    calls, accumulate CPI, FFT, then query both trainers' policies once).

The pattern mirrors `train_laser.py:869-960` but is decoupled from any
specific trainer — both `_train_against` and `evaluate_pair` consume it.

Lifecycle per episode:
  1. reset(env_ids=None) → env.reset + cpi_buffer.reset + zero cached actions
  2. step_control(red_trainer, blue_trainer, deterministic) → runs N pulses,
     processes CPI, queries trainers, caches new actions for the NEXT control
     step. Returns (result, red_transition_prev, blue_transition_prev).
  3. Caller calls `red_trainer.store_transition(env, result, red_transition_prev, 0)`
     to credit the PREVIOUS control step's actions with the current reward.

The first control step has no prior actions; we run N random pulses so the
CPI is populated before the first real policy query.
"""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import Optional, Dict, Any

from training.radar_policy import CPIAccumulator


__all__ = ["LaserEpisodeRunner"]


class LaserEpisodeRunner:
    """Pulse-level episode runner for the laser task.

    Owns:
      - CPI accumulator (one per env, shared across all radars/teams)
      - Cached actions from the previous control step (for the next N pulses)
      - pulses_per_control config (default 5)

    The runner is env-attached: callers should construct one per env and
    call `reset()` at the start of each episode.
    """

    def __init__(
        self,
        env,
        pulses_per_control: int = 5,
        device: str = "cuda",
    ):
        self.env = env
        self.pulses_per_control = int(pulses_per_control)
        self.device = torch.device(device)

        E = env.num_envs
        R = env.n_radars
        N = env.n_elem
        P = env.n_pulses
        S = env.n_samples
        self.E, self.R, self.N, self.P, self.S = E, R, N, P, S
        self.n_bins = env.n_bins

        self.cpi_buffer = CPIAccumulator(E, R, N, P, S, device=device)

        # Cached physical-layer tensors (rebuilt on reset; first control step uses random)
        self._cached_tx: Optional[torch.Tensor] = None
        self._cached_cmd: Optional[torch.Tensor] = None
        self._cached_veh: Optional[torch.Tensor] = None

        # Cached previous-step transitions (for store_transition timing)
        self._prev_red_transition: Optional[dict] = None
        self._prev_blue_transition: Optional[dict] = None

        # Element positions / wavelength for _assemble_tx
        self.elem_x = env.elem_x
        self.elem_y = env.elem_y
        self.wavelength = float(env.array.wavelength)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_ids=None, red_trainer=None, blue_trainer=None):
        """Reset env + CPI buffer + clear cached actions + per-episode trainer state."""
        self.env.reset(env_ids)
        self.cpi_buffer.reset()
        self._cached_tx = None
        self._cached_cmd = None
        self._cached_veh = None
        self._prev_red_transition = None
        self._prev_blue_transition = None
        # Laser per-episode state: reward shaper _jam_level/_beam_hit_time
        # and KalmanTracker _initialized. Reset on both trainers if attached.
        E = self.env.num_envs
        n_teams = self.env.n_teams
        for trainer in (red_trainer, blue_trainer):
            if trainer is None:
                continue
            shaper = getattr(trainer, "reward_shaper", None)
            if shaper is not None and hasattr(shaper, "reset_episode"):
                shaper.reset_episode(E, n_teams)
            kalman = getattr(trainer, "kalman_tracker", None)
            if kalman is not None and hasattr(kalman, "reset"):
                kalman.reset()

    # ------------------------------------------------------------------
    # State building
    # ------------------------------------------------------------------

    def _process_cpi(self) -> torch.Tensor:
        """FFT the accumulated CPI → spectrum magnitude [E, R, N*P*n_bins]."""
        cpi = self.cpi_buffer.data()  # [E, R, N, P, S]
        spectrum = torch.fft.fft(cpi, n=self.n_bins, dim=-1)  # [E, R, N, P, n_bins]
        spec_mag = spectrum.abs().float()
        self.cpi_buffer.reset()
        return spec_mag

    def _build_radar_obs(self, spectrum: torch.Tensor, events: dict) -> torch.Tensor:
        """Build per-radar observation from spectrum + events.

        Mirrors train_laser.py:_build_radar_obs. Layout:
          [spec_flat, comm_flat, recon_flat, vehicle, laser_state, cmd_instr]

        spectrum is the only signal-carrying component for now; the others
        are zero placeholders for parity with the radar actor-critic input dim.
        """
        dev = self.device
        E, R, N = self.E, self.R, self.N
        P = self.P
        n_bins = self.n_bins

        spec_flat = spectrum.reshape(E, R, -1) if spectrum is not None else \
            torch.zeros(E, R, N * P * n_bins, device=dev)

        comm_flat = torch.zeros(E, R, N * 2, device=dev)
        recon_flat = torch.zeros(E, R, N * 4, device=dev)
        vehicle = torch.zeros(E, R, 5, device=dev)
        laser_state = torch.zeros(E, R, 12, device=dev)
        cmd_instr = torch.zeros(E, R, 16, device=dev)

        if "radar_pos" in events:
            vehicle[:, :, 0] = events["radar_pos"][:, :, 0]
            vehicle[:, :, 1] = events["radar_pos"][:, :, 1]

        return torch.cat(
            [spec_flat, comm_flat, recon_flat, vehicle, laser_state, cmd_instr],
            dim=-1,
        )

    def _build_commander_obs(self, events: dict) -> torch.Tensor:
        """Build commander obs from env battlefield state."""
        env = self.env
        dev = self.device
        radar_latents = torch.zeros(env.num_envs, env.n_radars, 32, device=dev)
        obs = env.battlefield.get_commander_observation(env.radar_pos, radar_latents)
        return obs  # caller applies sensing

    # ------------------------------------------------------------------
    # TX assembly (action → physical signal)
    # ------------------------------------------------------------------

    def assemble_tx(self, radar_actions_global: torch.Tensor) -> torch.Tensor:
        """Convert flat radar actions [E, R, action_dim] → TX signal [E, R, N, S].

        Lifted from train_laser.py:_assemble_tx. Decodes per-element actions
        (task_id, beam_az/el, waveform, jam, comm) into a beamformed LFM chirp.
        """
        E, R, N, S = self.E, self.R, self.N, self.S
        dev = self.device
        ACTION_PER_ELEM = 22

        actions = radar_actions_global.reshape(E, R, -1)
        elem_actions = actions[:, :, :N * ACTION_PER_ELEM].reshape(E, R, N, ACTION_PER_ELEM)
        beam_az = elem_actions[..., 4] * 60.0
        beam_el = elem_actions[..., 5] * 45.0

        k = 2.0 * math.pi / self.wavelength
        DEG2RAD = math.pi / 180.0
        az_rad = beam_az.clamp(-90, 90) * DEG2RAD
        el_rad = beam_el.clamp(-90, 90) * DEG2RAD
        u = torch.sin(az_rad) * torch.cos(el_rad)
        v = torch.sin(el_rad)
        phase = -k * (self.elem_x.view(1, 1, N) * u + self.elem_y.view(1, 1, N) * v)
        weights = torch.exp(1j * phase)  # [E, R, N]

        t = torch.linspace(0, S / 200e6, S, device=dev)
        bw = 200e6 * 0.5
        chirp = torch.exp(1j * 2 * math.pi * bw * t ** 2 / (S / 200e6))  # [S]

        tx = weights.unsqueeze(-1) * chirp.view(1, 1, 1, S)
        return tx.to(torch.complex64)

    # ------------------------------------------------------------------
    # One control step (pulses_per_control pulses + one policy query)
    # ------------------------------------------------------------------

    def step_control(
        self,
        red_trainer,
        blue_trainer,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """Run one control step: N pulses, then policy query.

        Returns dict with:
          - result: last env.step() output (rewards/dones/winners)
          - red_transition_prev / blue_transition_prev: transitions from the
            PREVIOUS control step's actions (to be passed to store_transition).
            None on the very first control step (no prior action).
          - red_transition_new / blue_transition_new: freshly queried
            transitions, which become cached for the NEXT step_control call.
          - first_step: True if this was the first control step (no prior
            actions to credit).
        """
        env = self.env
        dev = self.device
        E = self.E
        R = self.R
        n_teams = env.n_teams

        # Phase 1: use cached actions for N pulses; accumulate CPI.
        result = None
        for pulse in range(self.pulses_per_control):
            if self._cached_tx is None:
                # First control step: random physical actions so the CPI is populated.
                rand_actions = torch.rand(E, R, env.action_dim, device=dev) * 2 - 1
                self._cached_tx = self.assemble_tx(rand_actions)
                self._cached_cmd = torch.zeros(E, n_teams, env.battlefield.commander_action_dim, device=dev)
                self._cached_veh = torch.zeros(E, R, 3, device=dev)
            result = env.step(
                self._cached_tx, self._cached_cmd, self._cached_veh,
            )
            self.cpi_buffer.append(result["rx_iq"])
            if result["dones"].any():
                break

        first_step = self._prev_red_transition is None

        # Phase 2: FFT the CPI into a spectrum.
        spectrum = self._process_cpi()
        events = {
            "radar_pos": env.radar_pos,
            "alive": env.battlefield.alive,
            "spectrum": spectrum,
        }

        # Phase 3: query both trainers' policies for the NEXT control step.
        red_new = red_trainer.get_own_actions(
            env, team=0, deterministic=deterministic,
            spectrum=spectrum, events=events,
        )
        blue_new = blue_trainer.get_own_actions(
            env, team=1, deterministic=deterministic,
            spectrum=spectrum, events=events,
        )

        # Phase 4: combine per-team actions into global tensors; cache.
        global_radar = torch.zeros(E, R, env.action_dim, device=dev)
        for i, r in enumerate(range(red_new["r_start"], red_new["r_end"])):
            global_radar[:, r, :] = red_new["radar_actions"][i]
        for i, r in enumerate(range(blue_new["r_start"], blue_new["r_end"])):
            global_radar[:, r, :] = blue_new["radar_actions"][i]

        global_cmd = torch.zeros(
            E, n_teams, env.battlefield.commander_action_dim, device=dev,
        )
        global_cmd[:, 0, :] = red_new["commander_action"]
        global_cmd[:, 1, :] = blue_new["commander_action"]

        global_veh = global_radar[..., -3:]  # last 3 dims are vehicle (speed/heading/rot)

        self._cached_tx = self.assemble_tx(global_radar)
        self._cached_cmd = global_cmd
        self._cached_veh = global_veh

        # Phase 5: rotate transitions — previous becomes "to credit", new becomes "previous".
        red_to_credit = self._prev_red_transition
        blue_to_credit = self._prev_blue_transition
        self._prev_red_transition = red_new["transition"]
        self._prev_blue_transition = blue_new["transition"]

        return {
            "result": result,
            "first_step": first_step,
            "red_transition_prev": red_to_credit,
            "blue_transition_prev": blue_to_credit,
            "red_transition_new": red_new["transition"],
            "blue_transition_new": blue_new["transition"],
        }
