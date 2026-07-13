"""Adaptive jammer opponents for Concerto-RRM pilot (L0 / L1 / L3).

Three difficulty levels (per EAAI_RESEARCH_PLAN.md §4.1):

  L0 StaticJammer   — fixed jam_level (e.g. 0.3). Zero adaptive behavior.
                      Baseline: classical scheduler should be near-optimal.
  L1 ReactiveJammer — maintains EMA of red's task histogram over the last τ
                      steps; each step picks red's dominant function and
                      concentrates jam on that function's band. τ = reaction
                      delay (steps). At τ=∞ collapses to L0.
  L3 LearnedJammer  — small MLP (input: red task histogram + own jam history;
                      output: jam_level + jam_band onehot). Trained via PPO
                      self-play against league snapshots of the red policy.

All adversaries share the same external API:
    jammer.reset(num_envs, n_teams, device)
    jammer.step(red_task_hist, jam_history) → jam_level [E, n_teams]

Where:
    red_task_hist: [E, n_teams, 4] — fraction of red elements on each function
                   in the previous step. (Each team's "red" is the enemy team.)
    jam_history:   [E, n_teams] — this jammer's own jam_level last step.

The jam_level tensor is consumed by:
  - fused_sensing (jam coupling: noise multiplier on red's sensing)
  - spectrum_metrics.jam_power_on_victim_db (JSR computation)
  - ClassicalQoSRRM (sees JSR via events dict → margin computation)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional


# ---------------------------------------------------------------------------
# L0: Static jammer
# ---------------------------------------------------------------------------

class StaticJammer:
    """L0: constant jam_level. Zero adaptive behavior.

    Near-zero compute. Used to verify the classical QoS scheduler is near-
    optimal under L0 (Pilot criterion 1 sanity floor: classical QoS > 0.9).
    """

    def __init__(self, jam_level: float = 0.3):
        self.jam_level_value = float(jam_level)
        self._buf = None

    def reset(self, num_envs: int, n_teams: int, device):
        self._buf = torch.full(
            (num_envs, n_teams), self.jam_level_value,
            dtype=torch.float32, device=device,
        )

    def step(
        self,
        red_task_hist: Optional[torch.Tensor] = None,
        jam_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._buf


# ---------------------------------------------------------------------------
# L1: Reactive jammer (EMA of red task histogram, τ-step delay)
# ---------------------------------------------------------------------------

class ReactiveJammer:
    """L1: τ-delayed reactive jammer.

    Each step:
      1. Update EMA of red's task histogram (per team).
      2. Identify red's dominant function (argmax of EMA).
      3. Set jam_level proportional to (1 - red_dominant_fraction): more
         concentrated red is → more damage if we jam that band.
      4. Output band-targeted jam_level.

    Band targeting is implicit: the jam is applied multiplicatively to the
    sensing noise floor (jam_mul = 1 + jam_gain × enemy_jam in sensing.py).
    A higher jam_level on a more concentrated red distribution causes more
    denial because the few detect/track elems all get degraded.

    τ controls the reaction delay: EMA decay = 1 - 1/τ. At τ→∞, EMA never
    updates → equivalent to L0. At τ=1, EMA = current step (instant).
    """

    def __init__(
        self,
        tau: int = 8,
        base_jam: float = 0.3,
        max_jam: float = 1.0,
        adaptivity: float = 0.7,
    ):
        self.tau = max(1, int(tau))
        self.alpha = 1.0 / self.tau  # EMA update weight
        self.base_jam = float(base_jam)
        self.max_jam = float(max_jam)
        self.adaptivity = float(adaptivity)
        self._ema = None
        self._buf = None

    def reset(self, num_envs: int, n_teams: int, device):
        # Initialize EMA to uniform [0.25, 0.25, 0.25, 0.25]
        self._ema = torch.full(
            (num_envs, n_teams, 4), 0.25,
            dtype=torch.float32, device=device,
        )
        self._buf = torch.zeros(num_envs, n_teams, dtype=torch.float32, device=device)

    def step(
        self,
        red_task_hist: Optional[torch.Tensor],
        jam_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if red_task_hist is None:
            return self._buf
        # Update EMA
        self._ema = (1 - self.alpha) * self._ema + self.alpha * red_task_hist.float()
        # Concentration: how massed red's distribution is (1=fully concentrated)
        # Use (max - 1/4) scaled — 0 if uniform, up to 0.75 if all on one fn
        concentration = (self._ema.max(dim=-1).values - 0.25).clamp(min=0.0) / 0.75
        # jam_level: base + adaptivity × concentration
        jam = self.base_jam + self.adaptivity * concentration * (self.max_jam - self.base_jam)
        self._buf = jam.clamp(0.0, self.max_jam)
        return self._buf


# ---------------------------------------------------------------------------
# L3: Learned jammer (small MLP, PPO-trained via self-play)
# ---------------------------------------------------------------------------

class _JammerPolicy(nn.Module):
    """Tiny MLP: red_task_hist[E,T,4] + own_last_jam[E,T,1] → jam_mean[E,T].

    Output is a scalar jam_level in [0, 1] via sigmoid. PPO trainer wraps this
    with a learned std for exploration (not used in eval).
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, red_task_hist: torch.Tensor, own_jam: torch.Tensor) -> torch.Tensor:
        x = torch.cat([red_task_hist, own_jam.unsqueeze(-1)], dim=-1)  # [E, T, 5]
        return torch.sigmoid(self.net(x)).squeeze(-1)  # [E, T]


class LearnedJammer:
    """L3: small MLP trained via PPO self-play against league snapshots.

    For the pilot, we provide a *randomly-initialized* policy by default (no
    actual training yet — that requires the league infrastructure which is the
    WP-A phase). The pilot's L3 cell will either:
      (a) Train the jammer for a fixed wallclock against the current red policy
          (mini-league with 1 snapshot); or
      (b) Fall back to ReactiveJammer(tau=1) if training doesn't converge in
          budget (per Pilot risk R4).

    The class still implements the same step() API so the trainer doesn't care.
    """

    def __init__(
        self,
        base_jam: float = 0.3,
        hidden: int = 64,
        device: str = "cuda",
        policy_path: Optional[str] = None,
    ):
        self.base_jam = float(base_jam)
        self.device = torch.device(device)
        self.policy = _JammerPolicy(hidden=hidden).to(self.device)
        if policy_path is not None:
            sd = torch.load(policy_path, map_location=self.device)
            self.policy.load_state_dict(sd)
        self._buf = None
        self._last_jam = None

    def reset(self, num_envs: int, n_teams: int, device):
        device = torch.device(device)
        self._buf = torch.full(
            (num_envs, n_teams), self.base_jam,
            dtype=torch.float32, device=device,
        )
        self._last_jam = self._buf.clone()

    @torch.no_grad()
    def step(
        self,
        red_task_hist: Optional[torch.Tensor],
        jam_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if red_task_hist is None:
            return self._buf
        own = jam_history if jam_history is not None else self._last_jam
        jam = self.policy(red_task_hist, own)
        self._buf = jam.clamp(0.0, 1.0)
        self._last_jam = self._buf.clone()
        return self._buf

    # API for self-play training (used by future WP-A league)
    def parameters(self):
        return self.policy.parameters()

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_jammer(level: str = "L0", **kwargs):
    """Factory for the difficulty levels.

    Args:
        level: "L0" / "L1" / "L1-tau{N}" / "L3" / "L3-trained" (case-insensitive).
            L1-tau{N} selects ReactiveJammer with explicit τ ∈ {16,8,4,2,1}.
            "L3-trained" is an alias for "L3" with a required policy_path kwarg.
        **kwargs: passed to the underlying jammer constructor.
    """
    level_norm = level.upper()
    # Parse L1-tau{N} variant
    if level_norm.startswith("L1-TAU"):
        try:
            tau = int(level_norm.split("TAU")[1])
        except (IndexError, ValueError):
            tau = kwargs.get("tau", 8)
        return ReactiveJammer(
            tau=tau,
            base_jam=kwargs.get("base_jam", 0.3),
            max_jam=kwargs.get("max_jam", 1.0),
            adaptivity=kwargs.get("adaptivity", 0.7),
        )
    if level_norm == "L0":
        return StaticJammer(jam_level=kwargs.get("jam_level", 0.3))
    elif level_norm == "L1":
        return ReactiveJammer(
            tau=kwargs.get("tau", 8),
            base_jam=kwargs.get("base_jam", 0.3),
            max_jam=kwargs.get("max_jam", 1.0),
            adaptivity=kwargs.get("adaptivity", 0.7),
        )
    elif level_norm in ("L3", "L3-TRAINED"):
        policy_path = kwargs.get("policy_path", None)
        # "L3-trained" without policy_path = error (don't silently random-init)
        if level_norm == "L3-TRAINED" and policy_path is None:
            raise ValueError("L3-trained requires policy_path kwarg (don't use random init).")
        return LearnedJammer(
            base_jam=kwargs.get("base_jam", 0.3),
            hidden=kwargs.get("hidden", 64),
            device=kwargs.get("device", "cuda"),
            policy_path=policy_path,
        )
    else:
        raise ValueError(f"Unknown jammer level: {level}")
