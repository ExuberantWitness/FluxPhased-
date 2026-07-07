"""Concerto-RRM orchestrator: decides who owns each control slot.

Two modes (per Concerto-RRM plan §3):
  V1 (fixed alternation, baseline reproducing CRL2RT arXiv:2502.10429):
      owner = RL  iff  (t % (N+1)) == N
      i.e. classical owns N slots, then RL owns 1, repeating.
  V2 (event-triggered, this work's contribution):
      When ANY of {JSR > θ1, trace_P_norm > θ2, any QoS margin < ε} fires,
      the next K slots are RL-owned. Otherwise classical-owned.

Both composers are DETERMINISTIC given the inputs (V2 uses seeded torch
Generator for any randomization, but the trigger logic is purely thresholded).

The composer is a stateless function (V1) or a small stateful class (V2 needs
the K-step countdown after a trigger).
"""

from __future__ import annotations

import torch
from typing import Optional


OWNER_CLASSICAL = 0
OWNER_RL = 1


class ComposerV1:
    """Fixed N:1 classical→RL alternation (CRL2RT baseline).

    Slot t: owner = RL if t % (N+1) == N else CLASSICAL.
    With N=3: CCCR CCCR CCCR ...  → ~25% RL slots.
    """

    def __init__(self, n_classical_per_rl: int = 3):
        self.N = max(1, int(n_classical_per_rl))
        self._t = 0

    def reset(self):
        self._t = 0

    def owner(
        self,
        jsr_db: Optional[torch.Tensor] = None,
        trace_P_norm: Optional[torch.Tensor] = None,
        qos_margin_min: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns owner per env: [E] long in {OWNER_CLASSICAL, OWNER_RL}.

        Extra args (jsr/trace/margin) are ignored in V1 — kept for API
        compatibility with V2.
        """
        E = jsr_db.shape[0] if jsr_db is not None else (
            trace_P_norm.shape[0] if trace_P_norm is not None else
            (qos_margin_min.shape[0] if qos_margin_min is not None else 1))
        dev = (jsr_db.device if jsr_db is not None else
               (trace_P_norm.device if trace_P_norm is not None else
                qos_margin_min.device if qos_margin_min is not None else torch.device("cpu")))
        is_rl = (self._t % (self.N + 1) == self.N)
        owner = torch.full((E,), OWNER_RL if is_rl else OWNER_CLASSICAL,
                            dtype=torch.long, device=dev)
        self._t += 1
        return owner


class ComposerV2:
    """Event-triggered: classical by default, RL for K slots after any trigger.

    Triggers (any one suffices, per env):
      jsr_db      > θ1 (default 10 dB)
      trace_P_norm > θ2 (default 0.6)
      qos_margin_min < ε (default 0.2)

    When any trigger fires for env e, the next K slots for env e are RL-owned
    (regardless of subsequent triggers — K-slot commitment). After K slots,
    revert to classical unless a new trigger fires.

    Per-env state: countdown[E] = remaining RL slots.
    """

    def __init__(
        self,
        theta1_jsr_db: float = 10.0,
        theta2_trace: float = 0.6,
        epsilon_margin: float = 0.2,
        k_commitment: int = 5,
    ):
        self.theta1 = float(theta1_jsr_db)
        self.theta2 = float(theta2_trace)
        self.epsilon = float(epsilon_margin)
        self.K = max(1, int(k_commitment))
        self._countdown = None
        self._dev = None
        self._E = None

    def reset(self, num_envs: int = None, device=None):
        if num_envs is not None and device is not None:
            self._countdown = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._dev = torch.device(device)
            self._E = num_envs
        elif self._countdown is not None:
            self._countdown.zero_()

    def owner(
        self,
        jsr_db: Optional[torch.Tensor] = None,
        trace_P_norm: Optional[torch.Tensor] = None,
        qos_margin_min: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns [E] long in {OWNER_CLASSICAL, OWNER_RL}.

        On entry: countdown[e] = remaining RL slots from prior triggers.
        We then evaluate new triggers; if any fires and countdown==0, set
        countdown = K. Finally, owner = RL if countdown > 0.
        """
        if self._countdown is None:
            raise RuntimeError("ComposerV2.reset(num_envs, device) must be called first.")
        E = self._E
        dev = self._dev

        # Evaluate triggers (per env, OR'd). A trigger fires if ANY signal
        # crosses its threshold for that env.
        trig = torch.zeros(E, dtype=torch.bool, device=dev)
        if jsr_db is not None:
            # jsr_db: [E, T] per-team → take max over teams (worst-case jam)
            j = jsr_db.max(dim=-1).values if jsr_db.dim() > 1 else jsr_db
            trig = trig | (j > self.theta1)
        if trace_P_norm is not None:
            t = trace_P_norm.max(dim=-1).values if trace_P_norm.dim() > 1 else trace_P_norm
            trig = trig | (t > self.theta2)
        if qos_margin_min is not None:
            # qos_margin_min: [E] already-reduced min margin across functions
            trig = trig | (qos_margin_min < self.epsilon)

        # If currently in classical (countdown==0) and a trigger fires,
        # commit K RL slots.
        not_in_rl = (self._countdown == 0)
        new_trigger = trig & not_in_rl
        self._countdown = torch.where(
            new_trigger,
            torch.full_like(self._countdown, self.K),
            self._countdown,
        )

        owner = torch.where(
            self._countdown > 0, OWNER_RL, OWNER_CLASSICAL,
        ).long()
        # Decrement countdown for next step (clamp at 0).
        self._countdown = (self._countdown - 1).clamp(min=0)
        return owner

    @property
    def state(self) -> dict:
        return {
            "countdown": self._countdown.tolist() if self._countdown is not None else None,
            "theta1": self.theta1, "theta2": self.theta2,
            "epsilon": self.epsilon, "K": self.K,
        }


def make_composer(variant: str = "v1", **kwargs):
    """Factory: 'v1' → ComposerV1, 'v2' → ComposerV2."""
    variant = variant.lower()
    if variant in ("v1", "fixed"):
        return ComposerV1(n_classical_per_rl=kwargs.get("n_classical_per_rl", 3))
    elif variant in ("v2", "event"):
        return ComposerV2(
            theta1_jsr_db=kwargs.get("theta1_jsr_db", 10.0),
            theta2_trace=kwargs.get("theta2_trace", 0.6),
            epsilon_margin=kwargs.get("epsilon_margin", 0.2),
            k_commitment=kwargs.get("k_commitment", 5),
        )
    else:
        raise ValueError(f"Unknown composer variant: {variant}")
