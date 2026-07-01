"""Payoff matrix computation for PSRO.

Evaluates win rates between all pairs of policies by running games
in the vectorized MFARVecEnv. Supports batch evaluation using
parallel environments.

Also accumulates per-policy *task fingerprints* — the long-run
mean fraction of array elements assigned to each of the 4 tasks
(recon/detect/jam/comm) during evaluation games. These are
consumed by TC-DAMS to bias the meta-solver toward task-axis
diversity. Fingerprints are recorded per (policy_id, team) since
a policy plays a single team during any evaluation game.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from .opponent_pool import OpponentPool

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PayoffMatrix:
    """Compute and store empirical payoff matrix between policy populations."""

    def __init__(
        self,
        opponent_pool: OpponentPool,
        n_eval_games: int = 50,
        device: str = "cuda",
        max_steps_per_game: int = 200,
        task_type: str = "generic",
        pulses_per_control: int = 5,
    ):
        self.pool = opponent_pool
        self.n_eval_games = n_eval_games
        self.device = device
        # Hard cap on env.step() calls per evaluation game. If no team
        # wins by this many steps, the game is scored as a draw (0.5).
        # Default 200 keeps a single PSRO eval bounded; raise it if you
        # genuinely need episodes to run to natural termination.
        self.max_steps_per_game = max_steps_per_game
        self.task_type = task_type
        self.pulses_per_control = int(pulses_per_control)
        # matrix[i][j] = win rate of policy i against policy j
        self.matrix: Dict[Tuple[str, str], float] = {}
        # fingerprint[i] = [P(recon), P(detect), P(jam), P(comm)] averaged
        # over all evaluation games where policy i played.
        self.fingerprints: Dict[str, np.ndarray] = {}
        self._fp_counts: Dict[str, int] = {}
        # Last-iteration kill-rate signal (laser task). Updated by evaluate_pair:
        # fraction of games that ended in a decisive outcome (not a step-cap draw).
        # Used by FluxLeague._anneal_kill_radius for success-gated curriculum.
        self.last_kill_rate: float = 0.0
        # Cached illumination_progress [E, n_teams] from the most recent step.
        # Used as a tiebreaker when timeout games need scoring: team closer to
        # a kill (higher progress) wins. Without this, every timeout collapses
        # to 0.5 and the league's payoff matrix becomes uninformative.
        self._last_step_progress = None
        # F7: periodic re-evaluation of frozen pairs. Without this, PFSP/Elo
        # priorities ossify at first-observation values — AlphaStar uses
        # dedicated evaluator workers to keep the matrix current. Set to 0
        # to disable (legacy behavior); 3 = re-eval every 3 iters.
        self.re_eval_interval: int = 3
        self._eval_iter: Dict[Tuple[str, str], int] = {}
        self._iteration_counter: int = 0

    def _accumulate_fingerprint(self, policy_id: str, fp: np.ndarray) -> None:
        """Update running mean of policy_id's task fingerprint with one sample fp [4]."""
        fp = np.asarray(fp, dtype=np.float64).reshape(4)
        n = self._fp_counts.get(policy_id, 0)
        if n == 0:
            self.fingerprints[policy_id] = fp.copy()
        else:
            self.fingerprints[policy_id] = (
                self.fingerprints[policy_id] * n + fp
            ) / (n + 1)
        self._fp_counts[policy_id] = n + 1

    def evaluate_pair(
        self,
        red_policy_id: str,
        blue_policy_id: str,
        env,  # MFARVecEnv
        red_trainer,  # TeamPPOTrainer
        blue_trainer,  # TeamPPOTrainer
    ) -> float:
        """Evaluate win rate of red vs blue over multiple games.

        Uses deterministic policy inference for both sides.
        Resets all envs in parallel; extra envs beyond n_eval_games are
        harmless — we just count the first n_eval_games completions.
        """
        red_wins = 0
        total = 0
        n_step_cap_draws = 0  # games that hit max_steps_per_game without a winner
        remaining = self.n_eval_games
        E = env.num_envs
        live_envs = set()

        # Pulse-level runner: env.step takes (tx_signal, commander_actions,
        # vehicle_actions) and runs ONE pulse. We need pulses_per_control
        # pulses to fill a CPI before the radar policy can read state.
        from algo._shared.laser.episode import LaserEpisodeRunner
        runner = LaserEpisodeRunner(
            env, pulses_per_control=self.pulses_per_control, device=self.device,
        )

        while remaining > 0:
            batch = min(E, remaining)
            runner.reset(red_trainer=red_trainer, blue_trainer=blue_trainer)
            live_envs = set(range(batch))

            for step in range(self.max_steps_per_game):
                if not live_envs:
                    break
                if step % 100 == 0 and step > 0:
                    wr_sofar = red_wins / max(total, 1)
                    print(f"    step {step}: alive={len(live_envs)} "
                          f"red_wins={red_wins}/{total} wr_sofar={wr_sofar:.2f}",
                          flush=True)
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({
                            f"eval_match/{red_policy_id}_vs_{blue_policy_id}/step": step,
                            f"eval_match/{red_policy_id}_vs_{blue_policy_id}/wr_sofar": wr_sofar,
                            f"eval_match/{red_policy_id}_vs_{blue_policy_id}/alive": len(live_envs),
                            f"eval_match/{red_policy_id}_vs_{blue_policy_id}/games_done": total,
                        })
                with torch.no_grad():
                    # LaserEpisodeRunner handles the N-pulse loop, builds global
                    # tx_signal from both trainers' per-team actions, and calls
                    # env.step(tx_signal, commander_actions, vehicle_actions).
                    step_out = runner.step_control(
                        red_trainer, blue_trainer, deterministic=True,
                    )
                result = step_out["result"]
                if result is None:
                    break
                # Cache illumination_progress for timeout tiebreaker.
                # Shape [E, n_teams], values in [0, 1] (fraction of dwell done).
                if self.task_type == "laser":
                    self._last_step_progress = result.get("illumination_progress")
                    # Surface progress on the final step so we can diagnose why
                    # timeout tiebreaker might still produce 0.5 (e.g., both
                    # teams genuinely making zero progress).
                    if step == self.max_steps_per_game - 1 and \
                            self._last_step_progress is not None:
                        p = self._last_step_progress
                        print(f"    [{red_policy_id} vs {blue_policy_id}] "
                              f"final-step illumination_progress: "
                              f"team0={p[:, 0].mean().item():.4f} "
                              f"team1={p[:, 1].mean().item():.4f} "
                              f"(max t0={p[:, 0].max().item():.4f} "
                              f"t1={p[:, 1].max().item():.4f})",
                              flush=True)

                if result["dones"].any():
                    for e in sorted(live_envs):
                        if result["dones"][e]:
                            live_envs.discard(e)
                            if result["winners"][e] == 0:
                                red_wins += 1
                            total += 1
                            remaining -= 1

            # Score any still-live envs (step cap reached). For laser task,
            # use illumination_progress as tiebreaker so the team closer to a
            # kill wins; falls back to 0.5 draw for generic/missile tasks or
            # when both sides made zero progress.
            last_progress = self._last_step_progress
            for e in sorted(live_envs):
                total += 1
                n_step_cap_draws += 1
                remaining -= 1
                if last_progress is not None and self.task_type == "laser" \
                        and e < last_progress.shape[0]:
                    p0 = float(last_progress[e, 0])
                    p1 = float(last_progress[e, 1])
                    # Threshold: progress diff < 1% of dwell-requirement → draw.
                    # Otherwise the team with higher progress wins outright.
                    if p0 - p1 > 0.01:
                        red_wins += 1.0
                    elif p1 - p0 > 0.01:
                        red_wins += 0.0
                    else:
                        red_wins += 0.5
                else:
                    red_wins += 0.5
            live_envs.clear()
            self._last_step_progress = None

        win_rate = red_wins / max(total, 1)
        # Laser kill signal: fraction of games that ended in a real win
        # (someone died), as opposed to running out the step clock. This is
        # the success-gate for kill_radius curriculum annealing.
        n_decisive = max(total - n_step_cap_draws, 0)
        pair_kill_rate = n_decisive / max(total, 1)
        self.last_kill_rate = pair_kill_rate

        self.matrix[(red_policy_id, blue_policy_id)] = win_rate
        self.matrix[(blue_policy_id, red_policy_id)] = 1.0 - win_rate

        self.pool.update_win_rate(red_policy_id, blue_policy_id, win_rate >= 0.5)
        self.pool.update_win_rate(blue_policy_id, red_policy_id, win_rate < 0.5)

        return win_rate

    def evaluate_all(self, env, trainers: dict):
        """Evaluate all cross-team pairs.

        Args:
            env: MFARVecEnv
            trainers: {policy_id: TeamPPOTrainer}
        """
        red_policies = [
            pid for pid, rec in self.pool.policies.items()
            if rec.team == 0 and rec.is_active
        ]
        blue_policies = [
            pid for pid, rec in self.pool.policies.items()
            if rec.team == 1 and rec.is_active
        ]

        total_pairs = len(red_policies) * len(blue_policies)
        n_done = 0
        # F3: reset iteration-level kill signal at top of evaluate_all so a
        # zero-kill iter no longer inherits the previous iter's stale value.
        max_kill_rate = 0.0
        self._iteration_counter += 1
        for r_id in red_policies:
            for b_id in blue_policies:
                key = (r_id, b_id)
                # F7: re-evaluate pairs whose last eval was > re_eval_interval
                # iters ago, so PFSP priorities don't ossify at first-observation
                # values. AlphaStar uses dedicated evaluator workers; we approximate.
                should_eval = key not in self.matrix
                if not should_eval and self.re_eval_interval > 0:
                    last_eval_iter = self._eval_iter.get(key, 0)
                    if self._iteration_counter - last_eval_iter >= self.re_eval_interval:
                        should_eval = True
                if should_eval:
                    r_trainer = trainers.get(r_id)
                    b_trainer = trainers.get(b_id)
                    if r_trainer and b_trainer:
                        print(f"  [payoff] {r_id} vs {b_id} ({n_done + 1}/{total_pairs})...",
                              end=" ", flush=True)
                        t0 = time.time()
                        self.evaluate_pair(r_id, b_id, env, r_trainer, b_trainer)
                        elapsed = time.time() - t0
                        win_rate = self.matrix.get(key, 0.5)
                        print(f"win_rate={win_rate:.2f} ({elapsed:.1f}s)", flush=True)
                        max_kill_rate = max(max_kill_rate, self.last_kill_rate)
                        self._eval_iter[key] = self._iteration_counter
                n_done += 1
        # F3: always update last_kill_rate (was conditional on max > 0,
        # which made it sticky and triggered spurious annealing decisions).
        self.last_kill_rate = max_kill_rate

    def get_submatrix(self, team: int, exclude_roles: Optional[List[str]] = None) -> np.ndarray:
        """Get payoff submatrix for one team's perspective.

        Args:
            team: 0 (red) or 1 (blue)
            exclude_roles: optional list of role strings to exclude (e.g. ["mutant"])
        Returns:
            payoff: [K, K_opponent] numpy array, K = policies for this team
            own_policies: list of K policy_ids
            opp_policies: list of K_opponent policy_ids
        """
        own_policies = [
            pid for pid, rec in self.pool.policies.items()
            if rec.team == team and rec.is_active
            and (exclude_roles is None or rec.role not in exclude_roles)
        ]
        opp_policies = [
            pid for pid, rec in self.pool.policies.items()
            if rec.team != team and rec.is_active
            and (exclude_roles is None or rec.role not in exclude_roles)
        ]

        n_own = len(own_policies)
        n_opp = len(opp_policies)
        payoff = np.full((n_own, n_opp), 0.5)

        for i, own_id in enumerate(own_policies):
            for j, opp_id in enumerate(opp_policies):
                payoff[i, j] = self.matrix.get((own_id, opp_id), 0.5)

        return payoff, own_policies, opp_policies

    def get_fingerprints(self, policy_ids: List[str]) -> np.ndarray:
        """Stack task fingerprints for the given policies as [K, 4].

        Missing policies (no fingerprint observed yet) get a uniform
        prior [0.25, 0.25, 0.25, 0.25] so TC-DAMS sees them as neutral.
        """
        K = len(policy_ids)
        F = np.full((K, 4), 0.25, dtype=np.float64)
        for i, pid in enumerate(policy_ids):
            if pid in self.fingerprints:
                F[i] = self.fingerprints[pid]
        return F

    def to_array(self) -> np.ndarray:
        """Export full payoff matrix as numpy array."""
        active = [
            pid for pid, rec in self.pool.policies.items() if rec.is_active
        ]
        n = len(active)
        mat = np.full((n, n), 0.5)
        for i, p1 in enumerate(active):
            for j, p2 in enumerate(active):
                if p1 != p2:
                    mat[i, j] = self.matrix.get((p1, p2), 0.5)
        return mat, active
