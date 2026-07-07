"""Classical QoS-RRM scheduler (strong classical baseline, no learning).

Per the Concerto-RRM plan §2.1: a rule-based water-fill scheduler that
distributes 25 aperture elements (5×5) across the 4 radar functions (detect /
track / comm / jam) based on per-function QoS margin. Designed to be near-
optimal in low-difficulty (L0, no EW) and degrade gracefully under adaptive EW
(L1/L3) — providing a *credible* classical baseline that RL must beat.

API mirrors ClassicalMPC.get_own_actions so the Concerto trainer / runner can
dispatch to it interchangeably.

Per-element action layout (22 dims):
  [task_id(4), beam_az(1), beam_el(1), waveform(8), jam(4), comm(4)]

The scheduler writes:
  - task_id: one-hot 4-dim (which function this element serves)
  - beam_az, beam_el: per-function beam direction
  - waveform: task-dependent fixed waveform (chirp for detect, CW for track, etc.)
  - jam: jam amplitude for jam-tasked elems (scaled by team's jam_level)
  - comm: comm symbol for comm-tasked elems (BPSK modulated)

The other 22-dim slots are zero (no learning → no neural modulation).
"""

from __future__ import annotations

import math
import torch
from typing import Dict, Optional

from algo._shared.laser.sensing import KalmanTracker, fused_sensing


TASK_RECON = 0
TASK_DETECT = 1
TASK_JAM = 2
TASK_COMM = 3
N_TASKS = 4
ACTION_PER_ELEM = 22


class ClassicalQoSRRM:
    """Rule-based QoS water-fill scheduler.

    Maintains a Kalman tracker (same as ClassicalMPC) for fused enemy anchor +
    trace_P. Each step:
      1. Reads current QoS signals from `events` arg (Pd, crc, trace_P, jsr).
      2. Computes per-function margin = max(0, target - current).
      3. Allocates elements: floor N_floor per function, then water-fills the
         remainder by priority × margin.
      4. Sets per-element task_id, beam direction, waveform, jam, comm.

    Near-optimal under L0 (no jam): with all margins ≈ 0 except detect+track,
    the scheduler pours elements into detect/track → high QoS by construction.
    """

    def __init__(
        self,
        env,
        team: int,
        qos_floor_per_fn: int = 3,
        priorities: Optional[Dict[str, float]] = None,
        targets: Optional[Dict[str, float]] = None,
        min_radar_baseline_m: float = 5000.0,
        range_sigma_m: float = 0.05,
        crossrange_factor: float = 7.4e-5,
        track_q_m: float = 0.02,
        track_burnin: int = 120,
        half_map_m: float = 10000.0,
        jam_gain: float = 8.0,
        exposure_gain: float = 50.0,
        jam_amplitude: float = 0.5,
        comm_amplitude: float = 0.5,
        jam_spread_deg: float = 15.0,
    ):
        self.env = env
        self.team = int(team)
        self.qos_floor_per_fn = int(qos_floor_per_fn)
        self.priorities = dict(detect=1.0, track=0.9, comm=0.6, jam=0.4)
        if priorities:
            self.priorities.update(priorities)
        self.targets = dict(pd=0.9, trace_norm=0.4, crc=0.7, jsr_db=6.0)
        if targets:
            self.targets.update(targets)

        self.min_radar_baseline_m = float(min_radar_baseline_m)
        self.range_sigma_m = float(range_sigma_m)
        self.crossrange_factor = float(crossrange_factor)
        self.track_q_m = float(track_q_m)
        self.track_burnin = int(track_burnin)
        self.half_map_m = float(half_map_m)
        self.jam_gain = float(jam_gain)
        self.exposure_gain = float(exposure_gain)
        self.jam_amplitude = float(jam_amplitude)
        self.comm_amplitude = float(comm_amplitude)
        self.jam_spread_deg = float(jam_spread_deg)

        E = env.num_envs
        R_team = env.n_radars // env.n_teams
        self.r_start = team * R_team
        self.r_end = (team + 1) * R_team
        self.R_team = R_team

        self.kalman_tracker = KalmanTracker(
            track_q_m=self.track_q_m, track_burnin=self.track_burnin,
        )
        self.kalman_tracker.ensure_alloc(E, env.n_teams, torch.device(env.device))
        self.kalman_tracker._initialized = True
        self.jam_level = None  # set externally (team's own jam broadcast)

    def reset_episode(self, E: int, n_teams: int):
        self.kalman_tracker.reset()
        self.kalman_tracker.ensure_alloc(E, n_teams, torch.device(self.env.device))
        self.kalman_tracker._initialized = True

    # ------------------------------------------------------------------
    # Water-fill core
    # ------------------------------------------------------------------
    def _compute_alloc(
        self,
        pd_team: torch.Tensor,         # [E]
        trace_norm_team: torch.Tensor, # [E]
        crc_team: torch.Tensor,        # [E]
        jsr_team: torch.Tensor,        # [E] (my jam on enemy)
        n_elem: int,
    ) -> torch.Tensor:
        """Per-env per-function element allocation. Returns [E, 4] long.

        Step 1: floor N_floor to each function.
        Step 2: water-fill remainder by priority × margin.
        """
        E = pd_team.shape[0]
        dev = pd_team.device
        floor = self.qos_floor_per_fn
        targets = self.targets

        # Per-function margins [E, 4] in order [recon, detect, track, comm, jam]
        # but recon is implicit (leftover). We compute 4: detect, track, comm, jam.
        m_detect = (targets["pd"] - pd_team).clamp(min=0.0) * self.priorities["detect"]
        # track: high trace_norm = bad → margin grows
        m_track = (trace_norm_team - targets["trace_norm"]).clamp(min=0.0) * self.priorities["track"]
        m_comm = (targets["crc"] - crc_team).clamp(min=0.0) * self.priorities["comm"]
        m_jam = (targets["jsr_db"] - jsr_team).clamp(min=0.0) * self.priorities["jam"]

        margins = torch.stack([m_detect, m_track, m_comm, m_jam], dim=-1)  # [E, 4]
        margins = margins + 1e-6  # avoid all-zero ties → proportional split

        # Initial allocation: floor to each function (4 functions × floor)
        alloc = torch.full((E, N_TASKS), floor, dtype=torch.long, device=dev)
        # Recon gets whatever is left after main 4 + water-fill (can be 0)
        used = alloc.sum(dim=-1)  # [E]
        remaining = n_elem - used  # [E] — distribute to detect/track/comm/jam

        # Distribute remaining by proportional water-fill (no recon —
        # recon is implicit via left-over bandwidth in detect/track beams).
        weights = margins / margins.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        # Expected allocation (float), then round with floor to int
        extra = (weights * remaining.float().unsqueeze(-1)).floor().long()
        alloc = alloc + extra
        # Fix rounding remainder: give to argmax margin
        leftover = n_elem - alloc.sum(dim=-1)
        idx_max = margins.argmax(dim=-1)  # [E]
        for e in range(E):
            lv = int(leftover[e].item())
            if lv > 0:
                alloc[e, int(idx_max[e].item())] += lv

        # Clamp: each fn ≤ n_elem - 3 (ensure at least 3 elem for the others)
        # This is satisfied by construction since floor=3 each.
        return alloc  # [E, 4] in order [detect, track, comm, jam]

    def _alloc_to_task_ids(self, alloc: torch.Tensor, n_elem: int) -> torch.Tensor:
        """Expand [E, 4] allocation to [E, n_elem] task_ids.

        Task order: detect first, then track, comm, jam. (Order within doesn't
        matter for the env; it's just for layout consistency.)
        """
        E = alloc.shape[0]
        dev = alloc.device
        task_ids = torch.full((E, n_elem), TASK_RECON, dtype=torch.long, device=dev)
        # task_ids[:, :n_detect] = DETECT, etc.
        n_detect = alloc[:, 0]
        n_track = alloc[:, 1]
        n_comm = alloc[:, 2]
        n_jam = alloc[:, 3]
        for e in range(E):
            d = int(n_detect[e].item())
            t = int(n_track[e].item())
            c = int(n_comm[e].item())
            j = int(n_jam[e].item())
            idx = 0
            task_ids[e, idx:idx + d] = TASK_DETECT; idx += d
            task_ids[e, idx:idx + t] = TASK_RECON  # use RECON slot for track (task_id=0 used as recon/track umbrella)
            idx += t
            task_ids[e, idx:idx + c] = TASK_COMM; idx += c
            task_ids[e, idx:idx + j] = TASK_JAM; idx += j
        return task_ids

    # ------------------------------------------------------------------
    # API: get_own_actions
    # ------------------------------------------------------------------
    def get_own_actions(
        self,
        env,
        team: int = None,
        deterministic: bool = True,
        spectrum: torch.Tensor = None,
        events: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        if team is None:
            team = self.team

        # Stash the latest spectrum for downstream QoS computation.
        self._last_spectrum = spectrum

        dev = torch.device(env.device)
        E = env.num_envs
        R_team = self.R_team
        n_elem = env.n_elem

        # --- Fused sensing: write fused enemy xy into cmd_obs[68:72] ---
        radar_latents = torch.zeros(E, env.n_radars, 32, device=dev)
        cmd_obs = env.battlefield.get_commander_observation(
            env.radar_pos, radar_latents,
        )  # [E, n_teams, 76]
        fused_sensing(
            cmd_obs,
            half_x=self.half_map_m, half_y=self.half_map_m,
            range_sigma_m=self.range_sigma_m,
            crossrange_factor=self.crossrange_factor,
            tracker=self.kalman_tracker,
            jam_gain=self.jam_gain,
            exposure_gain=self.exposure_gain,
            jam_level=self.jam_level,
        )

        team_obs = cmd_obs[:, team, :]  # [E, 76]
        enemy_x_norm = team_obs[:, 68]
        enemy_y_norm = team_obs[:, 69]

        # --- Current QoS signals (from events; default to "all met" if missing) ---
        if events is not None:
            pd_team = events.get("pd_team", torch.full((E,), self.targets["pd"], device=dev))
            trace_norm = events.get("trace_P_norm_team",
                                     torch.zeros(E, device=dev))
            crc_team = events.get("crc_team", torch.full((E,), self.targets["crc"], device=dev))
            # jsr_team = MY jam impact on enemy (we want this HIGH → flip enemy's received jam)
            my_jam = (self.jam_level[:, team] if self.jam_level is not None
                      else torch.zeros(E, device=dev))
            jsr_team = events.get("jsr_team",
                                   10.0 * torch.log10((self.jam_gain * my_jam).clamp(min=1e-10)))
        else:
            pd_team = torch.full((E,), self.targets["pd"], device=dev)
            trace_norm = torch.zeros(E, device=dev)
            crc_team = torch.full((E,), self.targets["crc"], device=dev)
            my_jam = (self.jam_level[:, team] if self.jam_level is not None
                      else torch.zeros(E, device=dev))
            jsr_team = 10.0 * torch.log10((self.jam_gain * my_jam).clamp(min=1e-10))

        # --- Water-fill per env (vectorized) ---
        alloc = self._compute_alloc(pd_team, trace_norm, crc_team, jsr_team, n_elem)
        # alloc is [E, 4] in order [detect, track, comm, jam]
        task_ids_per_env = self._alloc_to_task_ids(alloc, n_elem)  # [E, n_elem]

        # --- Broadcast to all radars in team (same allocation per radar) ---
        radar_actions = torch.zeros(E, R_team, env.action_dim, device=dev)
        elem_actions = radar_actions[:, :, :n_elem * ACTION_PER_ELEM].reshape(
            E, R_team, n_elem, ACTION_PER_ELEM,
        )
        # Cache per-env task_ids for downstream QoS scoring. Pd metric needs
        # to know which elems are TASK_DETECT — env.step doesn't see raw actions
        # through the runner, so the scheduler stashes them.
        self._last_task_ids = task_ids_per_env.detach()  # [E, n_elem]

        # Set task_id one-hot
        task_ids_per_env_exp = task_ids_per_env.unsqueeze(1).expand(-1, R_team, -1)  # [E, R, N]
        task_onehot = torch.zeros(E, R_team, n_elem, 4, device=dev)
        task_onehot.scatter_(-1, task_ids_per_env_exp.unsqueeze(-1), 1.0)
        elem_actions[..., :4] = task_onehot

        # Beam direction per task
        # detect/track/recon → enemy anchor; jam → enemy with spread; comm → ally
        # We compute per-elem beam based on its task assignment
        for e in range(E):
            # Per-elem beam_az/el based on task
            tids = task_ids_per_env[e]  # [N]
            beam_az = torch.where(
                tids == TASK_JAM,
                torch.full_like(tids, float(enemy_x_norm[e]), dtype=torch.float),
                torch.full_like(tids, float(enemy_x_norm[e]), dtype=torch.float),
            )
            # Add ±spread for jam elems (per-elem phase offset)
            n_jam_e = int((tids == TASK_JAM).sum().item())
            if n_jam_e > 0:
                jam_idx = torch.where(tids == TASK_JAM)[0]
                spread = torch.linspace(-self.jam_spread_deg, self.jam_spread_deg,
                                         steps=len(jam_idx), device=dev) / 180.0 * math.pi
                beam_az_jam = float(enemy_x_norm[e]) + torch.sin(spread) * 0.1
                beam_az = beam_az.clone()
                beam_az[jam_idx] = beam_az_jam
            elem_actions[e, :, :, 4] = beam_az.unsqueeze(0)  # beam_az
            elem_actions[e, :, :, 5] = float(enemy_y_norm[e])  # beam_el

            # Waveform: simple markers (chirp=1 for detect, CW=0.5 for track,
            # noise=0.3 for jam, BPSK=±1 for comm)
            wf_detect = 1.0
            wf_track = 0.5
            wf_jam = 0.3
            wf_comm = 0.7
            wf = torch.zeros(n_elem, 8, device=dev)
            wf[tids == TASK_DETECT, 0] = wf_detect
            wf[tids == TASK_RECON, 0] = wf_track
            wf[tids == TASK_JAM, 0] = wf_jam
            wf[tids == TASK_COMM, 0] = wf_comm
            elem_actions[e, :, :, 6:14] = wf.unsqueeze(0)

            # Jam amplitude (only for jam-tasked elems)
            jam_vec = torch.zeros(n_elem, 4, device=dev)
            jam_vec[tids == TASK_JAM, 0] = self.jam_amplitude
            elem_actions[e, :, :, 14:18] = jam_vec.unsqueeze(0)

            # Comm symbol (BPSK alternating for comm-tasked elems)
            comm_vec = torch.zeros(n_elem, 4, device=dev)
            comm_idx = torch.where(tids == TASK_COMM)[0]
            for k, ci in enumerate(comm_idx.tolist()):
                comm_vec[ci, 0] = self.comm_amplitude * (1.0 if k % 2 == 0 else -1.0)
            elem_actions[e, :, :, 18:22] = comm_vec.unsqueeze(0)

        radar_actions_list = [radar_actions[:, i, :] for i in range(R_team)]

        # Commander action: fire if enemy localized reasonably (always in this baseline)
        commander_action = torch.zeros(E, 5, device=dev)
        commander_action[:, 0] = 1.0
        commander_action[:, 1] = enemy_x_norm
        commander_action[:, 2] = enemy_y_norm
        commander_action[:, 3] = 0.0

        # Cache last alloc so the Concerto driver / jammer can read it.
        self._last_alloc = alloc.detach()

        return {
            "r_start": self.r_start,
            "r_end": self.r_end,
            "radar_actions": radar_actions_list,
            "commander_action": commander_action,
            "transition": None,
            "qos_alloc": alloc,  # [E, 4] in order [detect, track, comm, jam]
        }
