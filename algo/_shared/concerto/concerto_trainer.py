"""Concerto pilot driver: composes classical + RL under an orchestrator.

Two pieces:

  1. ConcertoTrainerAdapter — wraps (ClassicalQoSRRM, RL trainer, composer)
     into a single object with the standard get_own_actions API. This lets
     LaserEpisodeRunner drive it interchangeably with any other trainer.

     The adapter asks the composer for owner each step; if RL, dispatches to
     the RL trainer; if classical, to ClassicalQoSRRM. It also stashes the
     chosen owner in `last_owner` and `last_owner_per_env` for downstream
     metric collection.

  2. ConcertoPilotDriver — orchestrates an entire pilot cell: builds the env,
     jammer, scheduler, composer (per method + difficulty), runs N eval
     episodes, returns QoS metrics. Used by run_pilot.py (A11).

The adapter approach keeps the env/runner untouched — the runner just sees a
trainer with get_own_actions; all the Concerto logic (owner dispatch, jam
coupling, metric collection) lives in the adapter / driver.
"""

from __future__ import annotations

import math
import time
import torch
from typing import Optional, Dict, Any, Callable, List

from env.gpu.qos_rrm import QoSRRMEnv, make_jammer
from env.gpu.qos_rrm.spectrum_metrics import (
    pd_at_pfa, trace_P_norm, crc_pass_rate, jam_power_on_victim_db, qos_satisfaction,
)
from algo._shared.concerto.composer import (
    ComposerV1, ComposerV2, OWNER_CLASSICAL, OWNER_RL,
)
from algo._shared.baselines.classical_qos_rrm import ClassicalQoSRRM


# ---------------------------------------------------------------------------
# ConcertoTrainerAdapter — singletrainer-API wrapper
# ---------------------------------------------------------------------------

class ConcertoTrainerAdapter:
    """Quacks like a trainer (get_own_actions), dispatches classical vs RL.

    Required for Concerto v1/v2: LaserEpisodeRunner.step_control calls
    trainer.get_own_actions(...) once per team per step. We intercept that
    call, ask the composer for owner, and dispatch accordingly.

    The adapter also exposes:
      - last_owner          : "rl" / "classical" (majority across envs)
      - last_owner_per_env  : [E] long
      - kalman_tracker      : the classical scheduler's tracker (for trace_P)
      - jam_level           : passthrough (set by runner / driver externally)
    """

    def __init__(
        self,
        classical: ClassicalQoSRRM,
        rl_trainer,
        composer,
        team: int,
    ):
        self.classical = classical
        self.rl_trainer = rl_trainer
        self.composer = composer
        self.team = int(team)
        self.r_start = classical.r_start
        self.r_end = classical.r_end
        self.last_owner = "classical"
        self.last_owner_per_env = None
        # Expose kalman_tracker so the runner can read trace_P
        self.kalman_tracker = classical.kalman_tracker
        # jam_level passthrough (set externally)
        self.jam_level = None
        # Eagerly init ComposerV2 countdown so owner() works on first call
        # even before reset_episode is invoked.
        if isinstance(composer, ComposerV2):
            dev = torch.device(classical.env.device)
            composer.reset(num_envs=classical.env.num_envs, device=dev)

    def reset_episode(self, E: int, n_teams: int):
        self.classical.reset_episode(E, n_teams)
        if hasattr(self.rl_trainer, "reset_episode"):
            self.rl_trainer.reset_episode(E, n_teams)
        if isinstance(self.composer, ComposerV2):
            dev = torch.device(self.classical.env.device)
            self.composer.reset(num_envs=E, device=dev)
        else:
            self.composer.reset()

    def _compute_trigger_signals(self, env, events: Optional[dict]) -> Dict[str, torch.Tensor]:
        """Build the signals the composer reads.

        jsr_db:        JSR at each team's receiver from its enemy (defensive —
                       high value means I am being jammed → trigger RL).
        trace_P_norm:  from classical scheduler's Kalman tracker
        qos_margin_min: from events if provided (per-team QoS aggregate)
        """
        dev = torch.device(env.device)
        E = env.num_envs
        n_teams = env.n_teams

        # trace_P_norm from classical scheduler's tracker (always available
        # because classical fuses every step regardless of owner — we just
        # may not ACT on it)
        trP = self.classical.kalman_tracker.trace_P
        tr_norm = trace_P_norm(trP) if trP is not None else None

        # jsr_db[t] = 10*log10(jam_gain × enemy_jam_on_t) — JSR at team t's
        # receiver from its enemy. enemy_jam_on_t = jam_level[:, 1-t].
        if self.jam_level is not None:
            enemy_jam = self.jam_level.flip(-1)  # [E, n_teams], team t sees (1-t)'s jam
            jsr = 10.0 * torch.log10(
                (self.classical.jam_gain * enemy_jam).clamp(min=1e-10))
        else:
            jsr = torch.full((E, n_teams), -20.0, device=dev)

        # qos_margin_min: from events
        margin = None
        if events is not None and "qos_margin_min" in events:
            margin = events["qos_margin_min"]
        return {
            "jsr_db": jsr,
            "trace_P_norm": tr_norm,
            "qos_margin_min": margin,
        }

    def get_own_actions(
        self,
        env,
        team: int = None,
        deterministic: bool = True,
        spectrum: torch.Tensor = None,
        events: Optional[dict] = None,
    ) -> Dict[str, Any]:
        if team is None:
            team = self.team

        # Always run classical fused_sensing first — it populates the Kalman
        # tracker and gives us the enemy anchor. The cost is one fused_sensing
        # call per step (already cheap). Classical's get_own_actions does this.
        # To avoid wasted work when owner==RL, we'd ideally factor fused_sensing
        # out; for now we just call classical always and discard its action
        # when owner==RL. (Pilot budget tolerates this 2× cost.)
        classical_out = self.classical.get_own_actions(
            env, team=team, deterministic=deterministic,
            spectrum=spectrum, events=events,
        )

        sig = self._compute_trigger_signals(env, events)
        owner = self.composer.owner(
            jsr_db=sig["jsr_db"],
            trace_P_norm=sig["trace_P_norm"],
            qos_margin_min=sig["qos_margin_min"],
        )  # [E] long

        majority_rl = bool((owner == OWNER_RL).float().mean().item() > 0.5)
        if majority_rl:
            rl_out = self.rl_trainer.get_own_actions(
                env, team=team, deterministic=deterministic,
                spectrum=spectrum, events=events,
            )
            self.last_owner = "rl"
            self.last_owner_per_env = owner
            # Carry qos_alloc from classical (for downstream dwell tracking)
            rl_out["qos_alloc"] = classical_out.get("qos_alloc")
            rl_out["owner"] = "rl"
            rl_out["owner_per_env"] = owner
            return rl_out
        else:
            self.last_owner = "classical"
            self.last_owner_per_env = owner
            classical_out["owner"] = "classical"
            classical_out["owner_per_env"] = owner
            return classical_out


# ---------------------------------------------------------------------------
# ConcertoPilotDriver — end-to-end evaluation
# ---------------------------------------------------------------------------

class ConcertoPilotDriver:
    """Run N eval episodes for one (method, difficulty, seed) cell.

    Builds the env + jammer + scheduler + composer + RL trainer (if any),
    then calls LaserEpisodeRunner.step_control repeatedly. QoS metrics are
    computed from the runner's outputs (spectrum from CPI buffer, task_ids /
    comm_crc_ok from qos_rrm_mode env step output, trace_P from classical
    scheduler's Kalman tracker, jsr_db from jam_level).

    Methods supported:
      "classical"    : red = ClassicalQoSRRM only (no RL).
      "mappo"        : red = pre-trained MAPPO policy only.
      "concerto_v1"  : red = ConcertoTrainerAdapter(classical, mappo, V1).
      "concerto_v2"  : red = ConcertoTrainerAdapter(classical, mappo, V2).
    """

    def __init__(
        self,
        env_qos: QoSRRMEnv,
        method: str,
        difficulty: str,
        seed: int,
        red_rl_trainer=None,
        classical_scheduler: Optional[ClassicalQoSRRM] = None,
        composer=None,
        jammer=None,
        max_steps: int = 100,
        team: int = 0,
        n_eval_episodes: int = 50,
        pulses_per_control: int = 5,
    ):
        self.qenv = env_qos
        self.method = method.lower()
        self.difficulty = difficulty.upper()
        self.seed = int(seed)
        self.red_rl = red_rl_trainer
        self.classical = classical_scheduler
        self.composer = composer
        self.jammer = jammer
        self.max_steps = int(max_steps)
        self.team = int(team)
        self.n_eval_episodes = int(n_eval_episodes)
        self.pulses_per_control = int(pulses_per_control)

        self._validate()

    def _validate(self):
        if self.method == "classical" and self.classical is None:
            raise ValueError("method='classical' requires classical_scheduler")
        if self.method == "mappo" and self.red_rl is None:
            raise ValueError("method='mappo' requires red_rl_trainer")
        if self.method.startswith("concerto"):
            if self.red_rl is None or self.classical is None or self.composer is None:
                raise ValueError(
                    f"method='{self.method}' requires red_rl_trainer + "
                    "classical_scheduler + composer")

    def _build_red_trainer(self):
        """Return the trainer object that LaserEpisodeRunner.step_control calls."""
        if self.method == "classical":
            return self.classical
        if self.method == "mappo":
            return self.red_rl
        return ConcertoTrainerAdapter(
            classical=self.classical, rl_trainer=self.red_rl,
            composer=self.composer, team=self.team,
        )

    def _build_blue_trainer(self):
        """For 1v1 asymmetric: blue is a no-op trainer (jammer runs separately)."""
        # The jammer is applied by externally setting jam_level on red's sensing.
        # Blue team's "trainer" is a stub that emits zeros (no real radars).
        return _StubBlueTrainer(self.qenv.env, team=1 - self.team)

    def run(self) -> Dict[str, Any]:
        """Run n_eval_episodes episodes. Return per-episode + aggregate metrics."""
        from algo._shared.laser.episode import LaserEpisodeRunner
        from algo._shared.laser.sensing import enforce_radar_baseline

        env = self.qenv.env
        runner = LaserEpisodeRunner(env, pulses_per_control=self.pulses_per_control)
        red_trainer = self._build_red_trainer()
        blue_trainer = self._build_blue_trainer()

        per_episode = []
        n_rl_total = 0
        n_classical_total = 0
        wallclock_start = time.perf_counter()

        for ep in range(self.n_eval_episodes):
            torch.manual_seed(self.seed * 1000 + ep)
            # Explicit reset of trainer per-episode state (runner.reset doesn't
            # call trainer.reset_episode — only reward_shaper/kalman_tracker).
            if hasattr(red_trainer, "reset_episode"):
                red_trainer.reset_episode(env.num_envs, env.n_teams)
            if hasattr(blue_trainer, "reset_episode"):
                blue_trainer.reset_episode(env.num_envs, env.n_teams)
            runner.reset(red_trainer=red_trainer, blue_trainer=blue_trainer)
            min_baseline = self.classical.min_radar_baseline_m if self.classical else 0.0
            if min_baseline > 0:
                enforce_radar_baseline(env, min_baseline)
            if self.jammer is not None:
                self.jammer.reset(env.num_envs, env.n_teams, env.device)
            # Set initial jam_level on red trainer (and classical) to zero.
            if hasattr(red_trainer, "jam_level"):
                red_trainer.jam_level = torch.zeros(
                    env.num_envs, env.n_teams, device=env.device)
            if self.classical is not None:
                self.classical.jam_level = torch.zeros(
                    env.num_envs, env.n_teams, device=env.device)

            ep_qos = {k: 0.0 for k in ["detect", "track", "comm", "jam", "aggregate"]}
            ep_dwell = {k: 0.0 for k in ["detect", "track", "comm", "jam"]}
            qos_count = 0
            ep_rl, ep_classical = 0, 0

            for step in range(self.max_steps):
                # Inject jam from jammer BEFORE red's sensing call:
                # build a jam_level tensor for both teams (symmetric: blue's
                # jam is what red receives; red's jam is what blue receives).
                # Here we use last allocation to inform jammer.
                last_alloc = getattr(self.classical, "_last_alloc", None) if self.classical else None
                if self.jammer is not None and last_alloc is not None:
                    n_elem = env.n_elem
                    hist = (last_alloc.float() / max(1, n_elem)).clamp(0.0, 1.0)
                    hist = hist.unsqueeze(1).expand(-1, env.n_teams, -1).contiguous()
                    jam = self.jammer.step(hist, None)  # [E, n_teams]
                    # team 1 (blue) emits jam; team 0 (red) emits 0.
                    jam_level = torch.zeros_like(jam)
                    jam_level[:, 1] = jam[:, 1]
                    if hasattr(red_trainer, "jam_level"):
                        red_trainer.jam_level = jam_level
                    if self.classical is not None:
                        self.classical.jam_level = jam_level

                out = runner.step_control(
                    red_trainer, blue_trainer, deterministic=True,
                )
                # Update red's emitted jam_level based on its allocation.
                # The scheduler emits jam proportional to its jam-element count.
                _alloc = getattr(self.classical, "_last_alloc", None) if self.classical else None
                if _alloc is not None:
                    n_elem = env.n_elem
                    n_jam = _alloc[:, 3].float()  # [E]
                    my_jam = (n_jam / max(1, n_elem)).clamp(0.0, 1.0) * 0.5  # cap 0.5
                    if self.classical.jam_level is None:
                        self.classical.jam_level = torch.zeros(
                            env.num_envs, env.n_teams, device=env.device)
                    self.classical.jam_level[:, self.team] = my_jam
                    if isinstance(red_trainer, ConcertoTrainerAdapter):
                        red_trainer.jam_level = self.classical.jam_level
                # Track owner (only meaningful for ConcertoTrainerAdapter)
                if isinstance(red_trainer, ConcertoTrainerAdapter):
                    if red_trainer.last_owner == "rl":
                        ep_rl += 1
                    else:
                        ep_classical += 1
                else:
                    # Non-Concerto: all steps are the method's owner
                    if self.method == "classical":
                        ep_classical += 1
                    else:
                        ep_rl += 1

                # Compute QoS metrics from this step's spectrum + env outputs
                result = out.get("result")
                if result is None:
                    break
                qos_step = self._compute_step_qos(result, runner, red_trainer)
                if qos_step is not None:
                    for k in ep_qos:
                        ep_qos[k] += qos_step[k]
                    qos_count += 1
                # Dwell from current allocation
                _dwell_alloc = getattr(self.classical, "_last_alloc", None) if self.classical else None
                if _dwell_alloc is not None:
                    n_elem = env.n_elem
                    ep_dwell["detect"] += float(_dwell_alloc[:, 0].sum().item()) / (env.num_envs * n_elem)
                    ep_dwell["track"] += float(_dwell_alloc[:, 1].sum().item()) / (env.num_envs * n_elem)
                    ep_dwell["comm"] += float(_dwell_alloc[:, 2].sum().item()) / (env.num_envs * n_elem)
                    ep_dwell["jam"] += float(_dwell_alloc[:, 3].sum().item()) / (env.num_envs * n_elem)

                if result["dones"].any():
                    break

            n = max(1, qos_count)
            # dwell normalizer: actual control steps taken (ep_classical + ep_rl),
            # NOT max_steps — episodes that end early (enemy killed) shouldn't
            # have their dwell_avg diluted by steps that never happened.
            n_steps = max(1, ep_classical + ep_rl)
            ep_metrics = {
                "qos_satisfaction": ep_qos["aggregate"] / n,
                "qos_per_function": {k: ep_qos[k] / n for k in
                                      ["detect", "track", "comm", "jam"]},
                "dwell_frac": {k: ep_dwell[k] / n_steps for k in ep_dwell},
                "n_rl_steps": ep_rl,
                "n_classical_steps": ep_classical,
                "qos_count": qos_count,
                "winners": result["winners"].tolist() if result else [],
            }
            per_episode.append(ep_metrics)
            n_rl_total += ep_rl
            n_classical_total += ep_classical

        wallclock = time.perf_counter() - wallclock_start
        return self._aggregate(per_episode, n_rl_total, n_classical_total, wallclock)

    def _compute_step_qos(
        self,
        result: Dict[str, Any],
        runner,
        red_trainer,
    ) -> Optional[Dict[str, float]]:
        """Compute per-step QoS from result + runner state + trainer tracker."""
        env = self.qenv.env
        dev = torch.device(env.device)
        E = env.num_envs
        # Spectrum: from runner's CPI buffer (last accumulated CPI)
        spectrum = self._last_spectrum(runner)
        if spectrum is None:
            return None
        # task_ids: prefer scheduler's stash (the runner doesn't pass
        # radar_actions_raw through to env.step, so result["task_ids"] is
        # always zeros). Fall back to result if scheduler stash unavailable.
        task_ids = None
        if self.classical is not None:
            tids_env = getattr(self.classical, "_last_task_ids", None)  # [E, N]
            if tids_env is not None:
                # Expand to [E, R, N] — same allocation for all team radars.
                R_team = self.classical.R_team
                rs = self.classical.r_start
                task_ids = torch.zeros(E, env.n_radars, env.n_elem,
                                        dtype=tids_env.dtype, device=dev)
                task_ids[:, rs:rs + R_team, :] = tids_env.unsqueeze(1).expand(-1, R_team, -1)
        if task_ids is None:
            task_ids = result.get("task_ids")
        if task_ids is None:
            task_ids = torch.zeros(E, env.n_radars, env.n_elem, dtype=torch.long, device=dev)
        pd = pd_at_pfa(spectrum, task_ids, pfa=1e-4)
        # trace_P from classical scheduler (shared with adapter)
        tracker = getattr(red_trainer, "kalman_tracker", None)
        if tracker is None and self.classical is not None:
            tracker = self.classical.kalman_tracker
        trP = tracker.trace_P if tracker is not None else None
        tr_norm = trace_P_norm(trP) if trP is not None else None
        # Comm CRC
        crc = result.get("comm_crc_ok")
        if crc is None:
            crc = torch.zeros(E, env.n_teams, dtype=torch.bool, device=dev)
        crc_rate = crc_pass_rate(crc)
        # JSR: team t's received jam = enemy's jam level
        jl = getattr(red_trainer, "jam_level", None)
        if jl is None and self.classical is not None:
            jl = self.classical.jam_level
        if jl is None:
            jl = torch.zeros(E, env.n_teams, device=dev)
        jsr_db = jam_power_on_victim_db(jl, jam_gain=8.0)

        # ew_degradation[t] = enemy's jam on team t = flip jl so team t sees (1-t)'s emission
        ew_deg = jl.flip(-1).clamp(0.0, 1.0) if jl is not None else None
        # Pull thresholds from QoS env wrapper (pilot-lenient defaults)
        thresholds = getattr(self.qenv, "qos_thresholds", None) or {}
        qos = qos_satisfaction(
            pd=pd, trace_norm=tr_norm, crc_rate=crc_rate, jsr_db=jsr_db,
            team_radar_indices=[list(range(t * (env.n_radars // env.n_teams),
                                            (t + 1) * (env.n_radars // env.n_teams)))
                                 for t in range(env.n_teams)],
            n_teams=env.n_teams,
            ew_degradation=ew_deg,
            pd_thresh=thresholds.get("pd_thresh", 0.3),
            trace_thresh=thresholds.get("trace_thresh", 0.6),
            crc_thresh=thresholds.get("crc_thresh", 0.4),
            jsr_target_db=thresholds.get("jsr_target_db", 3.0),
        )
        # Pilot is asymmetric: red (team=self.team) is the cognitive radar we
        # care about; blue (team=1-self.team) is a jammer-only stub. Report
        # only red's QoS so the aggregate isn't dragged down by blue's zero.
        return {k: float(v[:, self.team].mean().item()) for k, v in qos.items()}

    def _last_spectrum(self, runner):
        """Read the latest spectrum from the classical scheduler's stash.

        LaserEpisodeRunner._process_cpi resets the CPI buffer after FFT, so we
        can't read it from runner.cpi_buffer post-step. Instead, the classical
        scheduler (which is called every step with spectrum as an arg) stashes
        it on `self._last_spectrum`. This works for ALL methods because the
        ConcertoTrainerAdapter always calls classical.get_own_actions() (even
        when owner==RL) to populate the Kalman tracker.
        """
        sched = self.classical
        spec = getattr(sched, "_last_spectrum", None) if sched is not None else None
        return spec

    def _aggregate(self, per_episode, n_rl_total, n_classical_total, wallclock):
        if not per_episode:
            return {"error": "no episodes"}
        n = len(per_episode)
        # Mean across episodes
        agg_qos = sum(ep["qos_satisfaction"] for ep in per_episode) / n
        agg_per_fn = {
            k: sum(ep["qos_per_function"][k] for ep in per_episode) / n
            for k in ["detect", "track", "comm", "jam"]
        }
        agg_dwell = {
            k: sum(ep["dwell_frac"][k] for ep in per_episode) / n
            for k in ["detect", "track", "comm", "jam"]
        }
        # Variance for CI
        var_qos = sum((ep["qos_satisfaction"] - agg_qos) ** 2 for ep in per_episode) / max(1, n - 1)
        std_qos = math.sqrt(var_qos)
        return {
            "method": self.method,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "qos_satisfaction_mean": agg_qos,
            "qos_satisfaction_std": std_qos,
            "qos_per_function_mean": agg_per_fn,
            "dwell_frac_mean": agg_dwell,
            "min_dwell_frac": min(agg_dwell.values()),
            "n_episodes": n,
            "n_rl_steps_total": n_rl_total,
            "n_classical_steps_total": n_classical_total,
            "wallclock_s": wallclock,
        }


# ---------------------------------------------------------------------------
# Stub blue trainer — emits zeros for asymmetric 1v1
# ---------------------------------------------------------------------------

class _StubBlueTrainer:
    """No-op trainer for the blue team (asymmetric pilot: blue is jammer-only).

    Returns zero radar_actions + zero commander_action so the runner doesn't
    crash. The actual blue jamming is injected via jam_level on red's sensing.
    """

    def __init__(self, env, team: int):
        self.env = env
        self.team = int(team)
        R_team = env.n_radars // env.n_teams
        self.r_start = team * R_team
        self.r_end = (team + 1) * R_team
        self.R_team = R_team
        self.jam_level = None

    def reset_episode(self, E: int, n_teams: int):
        pass

    def get_own_actions(
        self,
        env,
        team: int = None,
        deterministic: bool = True,
        spectrum: torch.Tensor = None,
        events: Optional[dict] = None,
    ) -> Dict[str, Any]:
        if team is None:
            team = self.team
        dev = torch.device(env.device)
        E = env.num_envs
        R_team = self.R_team
        radar_actions = [torch.zeros(E, env.action_dim, device=dev)
                          for _ in range(R_team)]
        commander_action = torch.zeros(E, 5, device=dev)
        return {
            "r_start": self.r_start,
            "r_end": self.r_end,
            "radar_actions": radar_actions,
            "commander_action": commander_action,
            "transition": None,
        }
