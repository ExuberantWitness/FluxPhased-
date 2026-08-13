"""Generic N-head actor for MultiDiscrete / mixed action spaces.

This module generalizes S2's two-head `MultiDiscreteActor` (Categorical base +
Categorical beam) into an N-head framework driven by a list of `HeadSpec`s,
so that S3 can add a third Bernoulli(5) cell-binding head by *registration*
instead of rewriting the actor / KL / sampling logic.

Design goals:
  1. Bit-exact equivalence to MultiDiscreteActor when head_specs =
     [HeadSpec("base","categorical",3), HeadSpec("beam","categorical",5)].
     Same trunk (fc1 -> fc2 -> tanh), same head Linear layers, same masking.
  2. Head-kind polymorphism: "categorical" (masked, inverse-CDF sampling) and
     "bernoulli" (per-cell independent 0/1, used by S3 cell binding).
  3. Independence assumption preserved: joint logp / entropy / KL = sum over
     heads. This is the same assumption S2 makes; S3's 3-head joint space
     (3 x 5 x 2^5 = 480) is large enough that independence remains the
     pragmatic choice (a full joint Categorical would blow up the output dim).

S2 usage (equivalent to MultiDiscreteActor):
    specs = [
        HeadSpec("base", "categorical", N_ACTIONS_BASE),
        HeadSpec("beam", "categorical", N_ACTIONS_BEAM),
    ]
    actor = MultiHeadActor(obs_dim, specs)

S3 usage (planned, requires env-side cell binding — see S3_HEAD_REGISTRATION.md):
    specs = [
        HeadSpec("base", "categorical", N_ACTIONS_BASE),
        HeadSpec("beam", "categorical", N_ACTIONS_BEAM),
        HeadSpec("cell", "bernoulli",   N_CELLS),   # 5 independent 0/1
    ]
    actor = MultiHeadActor(obs_dim, specs)

The independence assumption means joint KL = sum of per-head KL even for
bernoulli heads; the per-head KL implementations are dispatched by kind.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Head specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadSpec:
    """Describes one action head.

    name:       identifier, used as dict key in distribution/action/logp dicts
    kind:       "categorical" (mutually-exclusive actions, masked logits)
                or "bernoulli" (n_actions independent 0/1 cells)
    n_actions:  number of output logits:
                  categorical -> number of discrete actions (e.g. 3, 5)
                  bernoulli   -> number of independent cells (e.g. 5)
    """
    name: str
    kind: str
    n_actions: int

    def __post_init__(self):
        if self.kind not in ("categorical", "bernoulli"):
            raise ValueError(f"HeadSpec kind must be 'categorical' or 'bernoulli', got {self.kind!r}")
        if self.n_actions <= 0:
            raise ValueError(f"HeadSpec n_actions must be > 0, got {self.n_actions}")


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class MultiHeadActor(nn.Module):
    """Shared-trunk actor with N independent heads dispatched by HeadSpec.kind.

    Architecture (identical to MultiDiscreteActor when specs match S2's two
    categorical heads):
        obs -> Linear(obs_dim, hidden) -> tanh -> Linear(hidden, hidden) -> tanh -> h
        h -> head_{name}: Linear(hidden, n_actions)   (one per spec)

    All heads share the trunk fc1/fc2 (same as S2). Heads are stored in an
    nn.ModuleDict keyed by spec.name so they appear in state_dict / parameters().
    """

    def __init__(self, obs_dim: int, head_specs: Sequence[HeadSpec], hidden: int = 128):
        super().__init__()
        if not head_specs:
            raise ValueError("head_specs must be non-empty")
        names = [s.name for s in head_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"head spec names must be unique, got {names}")
        self.head_specs: tuple[HeadSpec, ...] = tuple(head_specs)
        self.head_names: tuple[str, ...] = tuple(names)

        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.heads = nn.ModuleDict({
            s.name: nn.Linear(hidden, s.n_actions) for s in self.head_specs
        })

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns {head_name: logits[E, n_actions]}.

        Logits are RAW (pre-mask). Callers apply masking via distribution().
        Bernoulli logits are interpreted as log-odds of cell=1.
        """
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        return {name: self.heads[name](h) for name in self.head_names}

    def distribution(
        self, obs: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.distributions.Distribution]:
        """Returns {head_name: Distribution}.

        masks: {head_name: bool tensor[E, n_actions]}.
          - categorical: True = legal action; masked logits set to -inf.
          - bernoulli:   mask shape [E, n_cells]; True = cell may be on.
                         (Typically all-True; kept for API symmetry. Masked
                         cells forced off by setting logit to -inf.)

        Dispatches on HeadSpec.kind to build the right Distribution type.
        """
        logits = self.forward(obs)
        dists: dict[str, torch.distributions.Distribution] = {}
        for spec in self.head_specs:
            head_logits = logits[spec.name]
            mask = masks[spec.name].bool()
            if spec.kind == "categorical":
                masked = head_logits.masked_fill(~mask, float("-inf"))
                dists[spec.name] = torch.distributions.Categorical(logits=masked)
            else:  # bernoulli
                masked = head_logits.masked_fill(~mask, float("-inf"))
                dists[spec.name] = torch.distributions.Bernoulli(logits=masked)
        return dists

    def joint_log_prob(
        self, obs: torch.Tensor,
        masks: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Joint log-prob = sum of per-head log-probs (independence assumption).

        actions: {head_name: tensor[E]}.
          - categorical: int64 action indices in [0, n_actions)
          - bernoulli:   float in {0.0, 1.0} per cell, shape [E, n_cells]
        Returns [E].
        """
        dists = self.distribution(obs, masks)
        total = None
        for spec in self.head_specs:
            lp = dists[spec.name].log_prob(actions[spec.name])
            if spec.kind == "bernoulli":
                # torch's Bernoulli.log_prob (BCE-with-logits) returns NaN at
                # masked cells (logit=-inf: -inf * 0 = NaN). Force masked
                # cells to contribute exactly 0, then sum over cells so every
                # head contributes a per-env scalar [E], matching categorical.
                mask = masks[spec.name].bool()
                lp = torch.where(mask, lp, torch.zeros_like(lp)).sum(dim=-1)
            total = lp if total is None else total + lp
        return total

    def joint_entropy(
        self, obs: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Per-head entropy + their sum.

        Returns {"<name>": head_entropy[E], ..., "_sum": total[E]} so callers
        can apply per-head entropy coefficients without recomputing.
        """
        dists = self.distribution(obs, masks)
        out: dict[str, torch.Tensor] = {}
        total = None
        for spec in self.head_specs:
            if spec.kind == "categorical":
                ent = dists[spec.name].entropy()
            else:
                # Bernoulli entropy computed manually: torch's
                # Bernoulli.entropy() (BCE-with-logits) returns NaN when a
                # cell's logit is -inf (masked off), because -inf * 0 = NaN.
                # Masked cells contribute exactly 0 entropy (mirroring
                # bernoulli_kl's masked-cell handling).
                d = dists[spec.name]
                p = d.probs.clamp(1e-12, 1.0 - 1e-12)
                ent = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p))
                mask = masks[spec.name].bool()
                ent = torch.where(mask, ent, torch.zeros_like(ent))
                # per-cell entropy [E, n_cells] -> per-env scalar [E]
                ent = ent.sum(dim=-1)
            out[spec.name] = ent
            total = ent if total is None else total + ent
        out["_sum"] = total
        return out


# ---------------------------------------------------------------------------
# KL divergence (independence assumption: joint KL = sum of per-head KL)
# ---------------------------------------------------------------------------

def categorical_kl(
    logits_old: torch.Tensor, logits_new: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """KL(old || new) per sample for masked categoricals. Identical to S2's."""
    lo = logits_old.masked_fill(~mask.bool(), float("-inf"))
    ln = logits_new.masked_fill(~mask.bool(), float("-inf"))
    log_po = F.log_softmax(lo, dim=-1)
    log_pn = F.log_softmax(ln, dim=-1)
    po = log_po.exp()
    safe = po > 0
    contrib = torch.where(safe, po * (log_po - log_pn), torch.zeros_like(po))
    return contrib.sum(dim=-1)


def bernoulli_kl(
    logits_old: torch.Tensor, logits_new: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """KL(old || new) per sample for independent Bernoullis.

    For a single Bernoulli: KL = p*log(p/q) + (1-p)*log((1-p)/(1-q)), summed
    over cells. Masked cells (mask=False) contribute 0 (p forced to the value
    implied by logit=-inf on both sides, so KL term = 0).

    Args match categorical_kl for dispatch symmetry. logits are raw log-odds.
    """
    mask = mask.bool()
    lo = logits_old.masked_fill(~mask, float("-inf"))
    ln = logits_new.masked_fill(~mask, float("-inf"))
    # probs under old / new (clamped to avoid log(0))
    p = torch.sigmoid(lo).clamp(1e-12, 1.0 - 1e-12)
    q = torch.sigmoid(ln).clamp(1e-12, 1.0 - 1e-12)
    kl = p * (torch.log(p) - torch.log(q)) + (1.0 - p) * (torch.log1p(-p) - torch.log1p(-q))
    # masked cells: p is either ~0 or ~1 and q matches -> kl ~ 0, but force exact 0
    kl = torch.where(mask, kl, torch.zeros_like(kl))
    return kl.sum(dim=-1)


def joint_kl_multihead(
    actor: MultiHeadActor,
    obs: torch.Tensor,
    masks: dict[str, torch.Tensor],
    logits_old: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Joint KL(old || new) under independence = sum of per-head KL.

    Dispatches per-head KL by HeadSpec.kind. logits_old is the snapshot of
    {head_name: raw logits[E, n_actions]} taken before the update step.
    """
    logits_new = actor.forward(obs)
    total = None
    for spec in actor.head_specs:
        kl_fn = categorical_kl if spec.kind == "categorical" else bernoulli_kl
        kl = kl_fn(logits_old[spec.name], logits_new[spec.name], masks[spec.name])
        total = kl if total is None else total + kl
    return total


# ---------------------------------------------------------------------------
# Sampling (inverse-CDF, one uniform per head — deterministic via generator)
# ---------------------------------------------------------------------------

def sample_multihead(
    actor: MultiHeadActor,
    obs: torch.Tensor,
    masks: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Sample one action per head using inverse-CDF (categorical) / threshold (bernoulli).

    Uses a single torch.rand per head per env (deterministic given `generator`),
    matching MultiDiscreteActor._sample_actions' RNG contract so that a
    2-categorical-head MultiHeadActor reproduces S2's action stream exactly.

    Returns (actions, joint_logp):
      actions[head_name]:
        categorical -> [E] int64
        bernoulli   -> [E, n_cells] float32 in {0., 1.}
      joint_logp: [E]
    """
    dists = actor.distribution(obs, masks)
    E = obs.shape[0]
    device = obs.device
    actions: dict[str, torch.Tensor] = {}
    logp_total = torch.zeros(E, device=device)

    for spec in actor.head_specs:
        d = dists[spec.name]
        if spec.kind == "categorical":
            u = torch.rand(E, generator=generator, device=device)
            probs = d.probs.clamp(min=1e-12)
            cdf = torch.cumsum(probs, dim=-1)
            action = (u.unsqueeze(-1) < cdf).float().argmax(dim=-1).long()
            actions[spec.name] = action
            logp_total = logp_total + d.log_prob(action)
        else:  # bernoulli: one uniform per cell, independent
            n = spec.n_actions
            u = torch.rand(E, n, generator=generator, device=device)
            probs1 = d.probs.clamp(min=1e-12, max=1.0 - 1e-12)  # [E, n_cells]
            action = (u < probs1).to(torch.float32)             # [E, n_cells]
            actions[spec.name] = action
            # log_prob for Bernoulli: per-cell [E, n_cells]; NaN at masked
            # cells (BCE -inf*0), so zero them before summing to [E].
            lp_cell = d.log_prob(action)
            mask = masks[spec.name].bool()
            lp_cell = torch.where(mask, lp_cell, torch.zeros_like(lp_cell))
            logp_total = logp_total + lp_cell.sum(dim=-1)

    return actions, logp_total
