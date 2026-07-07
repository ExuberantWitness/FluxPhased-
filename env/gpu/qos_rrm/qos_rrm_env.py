"""QoS-RRM env layer: asymmetric-team adapter around MFARVecEnv.

Roles:
  - Red team = 4-function cognitive radar (full 22-dim/elem actions, surfaced
    spectrum for detect/track/comm/jam QoS scoring).
  - Blue team = adaptive jammer (single `jam_level` per step, drives the
    `jam_mul = 1 + jam_gain × enemy_jam` coupling inside fused_sensing).

The wrapped MFARVecEnv is constructed with `qos_rrm_mode=True` so its step()
already surfaces `task_ids` (argmax over per-elem 4-dim task head) and
`comm_crc_ok` (BPSK decoder result). This wrapper adds:
  - `spectrum`: FFT of accumulated CPI buffer (P pulses) → [E,R,N,P,n_bins] complex
  - `jsr_db`: jam-to-signal ratio proxy at red's receiver, per team
  - `trace_P_norm`: normalized Kalman track covariance per team (if tracker set)
  - `qos`: aggregate per-team QoS satisfaction dict (4 functions)

The wrapper does NOT apply jam coupling — that's the trainer's job via
`fused_sensing(jam_level=...)`. The wrapper only records jam_level for metric
computation.
"""

from __future__ import annotations

import torch
from typing import Optional, Dict, Any

from ..vec_mfar_env import MFARVecEnv
from .spectrum_metrics import (
    pd_at_pfa,
    trace_P_norm,
    crc_pass_rate,
    jam_power_on_victim_db,
    qos_satisfaction,
)


class QoSRRMEnv:
    """Asymmetric Concerto-RRM env adapter.

    Wraps an MFARVecEnv configured with qos_rrm_mode=True. Adds CPI buffering
    for spectrum surfacing and per-step QoS metric computation.

    The wrapper is a MEASUREMENT + ORCHESTRATION layer; it does not own policy
    logic. Callers (ConcertoTrainer, ClassicalQoSRRM) supply actions and read
    metrics from the returned dict.
    """

    def __init__(
        self,
        env: MFARVecEnv,
        kalman_tracker=None,
        jam_gain: float = 8.0,
        exposure_gain: float = 50.0,
        team_radar_indices=None,
        pfa: float = 1e-4,
        qos_thresholds: Optional[Dict[str, float]] = None,
    ):
        if not getattr(env, "qos_rrm_mode", False):
            raise ValueError(
                "QoSRRMEnv requires env constructed with qos_rrm_mode=True "
                "so that task_ids and comm_crc_ok are surfaced by step()."
            )
        self.env = env
        self.kalman_tracker = kalman_tracker
        self.jam_gain = float(jam_gain)
        self.exposure_gain = float(exposure_gain)
        self.team_radar_indices = team_radar_indices or self._default_team_indices()
        self.pfa = float(pfa)
        defaults = dict(pd_thresh=0.9, trace_thresh=0.6, crc_thresh=0.7,
                        jsr_target_db=6.0)
        if qos_thresholds:
            defaults.update(qos_thresholds)
        self.qos_thresholds = defaults

        # CPI buffer for spectrum (one CPI = P pulses)
        E, R, N = env.num_envs, env.n_radars, env.n_elem
        P, S = env.n_pulses, env.n_samples
        dev = torch.device(env.device)
        self._cpi_buf = torch.zeros(E, R, N, P, S, dtype=torch.complex64, device=dev)
        self._cpi_idx = 0
        self._P = P
        self._n_bins = env.n_bins

    def _default_team_indices(self):
        n_teams = self.env.n_teams
        r_per = self.env.n_radars // n_teams
        return [list(range(t * r_per, (t + 1) * r_per)) for t in range(n_teams)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, env_ids=None):
        self.env.reset(env_ids)
        self._cpi_buf.zero_()
        self._cpi_idx = 0
        if self.kalman_tracker is not None:
            self.kalman_tracker.reset()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(
        self,
        red_tx: torch.Tensor,
        red_actions_raw: Optional[torch.Tensor] = None,
        beam_az: Optional[torch.Tensor] = None,
        beam_el: Optional[torch.Tensor] = None,
        blue_jam_level: Optional[torch.Tensor] = None,
        commander_actions: Optional[torch.Tensor] = None,
        vehicle_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Step the wrapped env and augment with QoS metrics.

        Args:
            red_tx: [E,R,N,S] complex64 TX IQ for the red (cognitive radar) team.
            red_actions_raw: [E,R,action_dim] raw per-element actions BEFORE
                beamforming — needed for task_id decoding (already decoded by
                env.qos_rrm_mode surface, but passed to env.step for completeness).
            beam_az, beam_el: [E,R] beam direction (degrees) for red radars.
            blue_jam_level: [E, n_teams] jam level per team. Team t's jam is
                what team (1-t) receives. If None, treated as zeros.
            commander_actions, vehicle_actions: optional, passed through.
        Returns:
            dict with all env.step keys PLUS:
              spectrum: [E,R,N,P,n_bins] complex (None until CPI complete)
              cpi_complete: bool whether spectrum is fresh this step
              jsr_db: [E, n_teams] JSR proxy at each team's receiver
              trace_P_norm: [E, n_teams] normalized Kalman cov trace (or None)
              qos: dict of per-team QoS scores {detect,track,comm,jam,aggregate}
              blue_jam_level: echoed input for downstream logging
        """
        out = self.env.step(
            tx_signal=red_tx,
            commander_actions=commander_actions,
            vehicle_actions=vehicle_actions,
            beam_az=beam_az,
            beam_el=beam_el,
            radar_actions_raw=red_actions_raw,
        )

        # Append rx_iq to CPI buffer (one pulse per step).
        E, R, N, S = out["rx_iq"].shape
        self._cpi_buf[:, :, :, self._cpi_idx % self._P, :] = out["rx_iq"]
        self._cpi_idx += 1

        cpi_complete = (self._cpi_idx % self._P == 0)
        if cpi_complete:
            spectrum = torch.fft.fft(self._cpi_buf, n=self._n_bins, dim=-1)
            out["spectrum"] = spectrum
        else:
            out["spectrum"] = None
        out["cpi_complete"] = cpi_complete

        # --- QoS metrics (always computable; spectrum may be None between CPIs) ---
        if blue_jam_level is None:
            blue_jam_level = torch.zeros(
                self.env.num_envs, self.env.n_teams, device=out["rx_iq"].device,
            )
        out["blue_jam_level"] = blue_jam_level
        out["jsr_db"] = jam_power_on_victim_db(
            blue_jam_level, jam_gain=self.jam_gain,
        )

        traceP = self.kalman_tracker.trace_P if self.kalman_tracker is not None else None
        out["trace_P_norm"] = trace_P_norm(traceP) if traceP is not None else None

        # Per-function QoS — needs spectrum for Pd; skip until first CPI complete.
        if cpi_complete and out.get("task_ids") is not None:
            pd = pd_at_pfa(out["spectrum"], out["task_ids"], pfa=self.pfa)
        else:
            pd = torch.ones(self.env.num_envs, self.env.n_radars, device=out["rx_iq"].device)
        crc = crc_pass_rate(out.get("comm_crc_ok", torch.zeros(
            self.env.num_envs, self.env.n_teams,
            dtype=torch.bool, device=out["rx_iq"].device,
        )))
        out["qos"] = qos_satisfaction(
            pd=pd,
            trace_norm=out["trace_P_norm"],
            crc_rate=crc,
            jsr_db=out["jsr_db"],
            team_radar_indices=self.team_radar_indices,
            n_teams=self.env.n_teams,
            **self.qos_thresholds,
        )
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def n_radars(self) -> int:
        return self.env.n_radars

    @property
    def n_teams(self) -> int:
        return self.env.n_teams

    @property
    def device(self) -> str:
        return self.env.device

    def destroy(self):
        self._cpi_buf = None
        self.env.destroy()
