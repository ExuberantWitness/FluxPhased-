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

from algo._shared.radar_policy import CPIAccumulator


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
        max_slew_rate_deg_per_s: float = None,
        control_delay_steps: int = None,
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

        # Cached beam direction (degrees, [E, R]) from last assemble_tx call.
        # Used to thread policy beam steer into env.step() for RX channel.
        self._last_beam_az: Optional[torch.Tensor] = None
        self._last_beam_el: Optional[torch.Tensor] = None

        # Cached previous-step transitions (for store_transition timing)
        self._prev_red_transition: Optional[dict] = None
        self._prev_blue_transition: Optional[dict] = None

        # WP3.2 damage: auto-inherit from env if not explicitly provided
        if max_slew_rate_deg_per_s is None:
            max_slew_rate_deg_per_s = getattr(env, "max_slew_rate_deg_per_s", 0.0)
        if control_delay_steps is None:
            control_delay_steps = getattr(env, "control_delay_steps", 0)

        # WP3.2 damage: beam slew rate cap (per-pulse deg)
        self.max_slew_rate_deg_per_s = float(max_slew_rate_deg_per_s)
        self._max_slew_per_pulse = self.max_slew_rate_deg_per_s * env.pri
        # Track previous beam az/el in [-1, 1] normalized action space, [E, R, 2]
        if self._max_slew_per_pulse > 0:
            bw_az_deg = float(np.degrees(0.886 / (env.cols * 0.5)))
            bw_el_deg = float(np.degrees(0.886 / (env.rows * 0.5)))
            # action[4:6] is in [-1,1] mapping to ±(beam_az_max_deg / 2)
            # Convert per-pulse deg cap → action units
            self._slew_cap_az = self._max_slew_per_pulse / max(bw_az_deg, 1e-6)
            self._slew_cap_el = self._max_slew_per_pulse / max(bw_el_deg, 1e-6)
            self._prev_beam = torch.zeros(E, R, 2, device=self.device)
            self._slew_initialized = False
        else:
            self._slew_cap_az = 0.0
            self._slew_cap_el = 0.0
            self._prev_beam = None
            self._slew_initialized = False

        # WP3.2 damage: control delay (queue length N)
        self.control_delay_steps = int(control_delay_steps)
        if self.control_delay_steps > 0:
            # FIFO of (tx, cmd, veh) tuples; oldest is what gets executed
            self._action_queue_tx = []
            self._action_queue_cmd = []
            self._action_queue_veh = []
        else:
            self._action_queue_tx = None
            self._action_queue_cmd = None
            self._action_queue_veh = None

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
        # FIX(win_rate=0.5): enforce the radar baseline AT RESET — before the first
        # commander_obs is built and before the Kalman warm-start. Mirrors
        # train_laser._enforce_radar_baseline (applied at reset). Without this the
        # warm-start locks the tracker onto the un-spread near-collinear geometry →
        # degenerate fused anchor → aim at map centre → progress=0 → win_rate=0.5.
        from algo._shared.laser.sensing import enforce_radar_baseline
        for _trainer in (red_trainer, blue_trainer):
            _base = float(getattr(_trainer, "min_radar_baseline_m", 0.0)) if _trainer else 0.0
            if _base > 0.0:
                enforce_radar_baseline(self.env, _base)  # spreads ALL teams in one call
                break
        self.cpi_buffer.reset()
        self._cached_tx = None
        self._cached_cmd = None
        self._cached_veh = None
        self._last_beam_az = None
        self._last_beam_el = None
        self._prev_red_transition = None
        self._prev_blue_transition = None
        # WP3.2: clear damage-injection state for new episode
        if self._action_queue_tx is not None:
            self._action_queue_tx.clear()
            self._action_queue_cmd.clear()
            self._action_queue_veh.clear()
        if self._prev_beam is not None:
            self._prev_beam.zero_()
            self._slew_initialized = False
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

        Also caches the per-radar mean beam_az/beam_el (in degrees) into
        self._last_beam_az / self._last_beam_el so downstream code (env.step)
        can use the same beam direction for RX channel modeling.
        """
        E, R, N, S = self.E, self.R, self.N, self.S
        dev = self.device
        ACTION_PER_ELEM = 22

        actions = radar_actions_global.reshape(E, R, -1)
        elem_actions = actions[:, :, :N * ACTION_PER_ELEM].reshape(E, R, N, ACTION_PER_ELEM)
        beam_az = elem_actions[..., 4] * 60.0
        beam_el = elem_actions[..., 5] * 45.0

        # Cache mean beam direction per (env, radar) for RX channel modeling.
        # Channel expects [E, R] in degrees; mean across N elements since the
        # channel uses a single beam direction per radar.
        self._last_beam_az = beam_az.mean(dim=-1).detach()  # [E, R]
        self._last_beam_el = beam_el.mean(dim=-1).detach()  # [E, R]

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
                beam_az=self._last_beam_az,
                beam_el=self._last_beam_el,
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

        # WP3.2 damage: clamp beam az/el slew rate (action indices 4,5 per elem).
        # action layout per element: [task_id(4), beam_az(1), beam_el(1), wf(8), jam(4), comm(4)]
        if self._prev_beam is not None and self._max_slew_per_pulse > 0:
            n_elem = env.n_elem
            ACTION_PER_ELEM = 22
            elem_view = global_radar[:, :, :n_elem * ACTION_PER_ELEM].reshape(
                E, R, n_elem, ACTION_PER_ELEM,
            )
            cur_az = elem_view[:, :, :, 4]
            cur_el = elem_view[:, :, :, 5]
            # First-step init: prev := current (no clamp on first action)
            if not getattr(self, "_slew_initialized", False):
                self._prev_beam[..., 0] = cur_az.mean(dim=-1)
                self._prev_beam[..., 1] = cur_el.mean(dim=-1)
                self._slew_initialized = True
            prev_az = self._prev_beam[..., 0].unsqueeze(-1)  # [E, R, 1]
            prev_el = self._prev_beam[..., 1].unsqueeze(-1)
            # Clamp Δ to ±cap, broadcast across elements
            new_az = prev_az + (cur_az - prev_az).clamp(
                -self._slew_cap_az, self._slew_cap_az)
            new_el = prev_el + (cur_el - prev_el).clamp(
                -self._slew_cap_el, self._slew_cap_el)
            elem_view[:, :, :, 4] = new_az
            elem_view[:, :, :, 5] = new_el
            self._prev_beam[..., 0] = new_az.mean(dim=-1)
            self._prev_beam[..., 1] = new_el.mean(dim=-1)

        global_cmd = torch.zeros(
            E, n_teams, env.battlefield.commander_action_dim, device=dev,
        )
        global_cmd[:, 0, :] = red_new["commander_action"]
        global_cmd[:, 1, :] = blue_new["commander_action"]

        global_veh = global_radar[..., -3:]  # last 3 dims are vehicle (speed/heading/rot)

        new_tx = self.assemble_tx(global_radar)
        new_cmd = global_cmd
        new_veh = global_veh

        # WP3.2 damage: control_delay_steps FIFO queue.
        # If delay > 0, push new actions to back; pop oldest from front for execution.
        if self.control_delay_steps > 0 and self._action_queue_tx is not None:
            self._action_queue_tx.append(new_tx)
            self._action_queue_cmd.append(new_cmd)
            self._action_queue_veh.append(new_veh)
            # While queue not full, use oldest (which is the only one for first N steps)
            if len(self._action_queue_tx) > self.control_delay_steps + 1:
                self._action_queue_tx.pop(0)
                self._action_queue_cmd.pop(0)
                self._action_queue_veh.pop(0)
            # Execute the oldest queued action (FIFO front)
            self._cached_tx = self._action_queue_tx[0]
            self._cached_cmd = self._action_queue_cmd[0]
            self._cached_veh = self._action_queue_veh[0]
        else:
            self._cached_tx = new_tx
            self._cached_cmd = new_cmd
            self._cached_veh = new_veh

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
