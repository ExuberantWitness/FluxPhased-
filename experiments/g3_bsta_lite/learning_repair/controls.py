"""R2 controls — every required control for Gate 3 and R3 (PREREGISTRATION §10).

Each control is a callable policy of the same shape as the trained actor:
given obs + mask, return an int64 action. They are evaluated under the
SAME evaluation harness (evaluate_actor equivalent) so the comparison
is policy-only, not harness-only.

Controls required by §10:

  random_untrained      uniform-over-mask sampled actions, fixed seed
  scratch_init          iteration = -1 snapshot of the scratch PPO actor
  scratch_trained       validation-selected checkpoint of scratch PPO
  pristine_bc           F3 DAgger actor, no PPO fine-tuning
  residual_ppo          stop-gradient BC logits + alpha * residual, with
                        KL-to-BC constraint
  shuffled_observation  trained actor with obs channels shuffled by a
                        fixed permutation
  time_only             actor sees only [step_idx / horizon]
  no_update             initial policy evaluated under the same harness
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.gpu.g3_bsta_lite import (
    EnvConfig,
    G3BstaLiteVecEnv,
    N_ACTIONS,
    OBS_DIM,
)


def _action_from_logits_masked(
    logits: torch.Tensor, mask: torch.Tensor, *,
    sample: bool, gen: Optional[torch.Generator],
) -> torch.Tensor:
    if sample:
        probs = torch.softmax(logits.masked_fill(~mask.bool(), float("-inf")), dim=-1)
        probs = probs.clamp(min=1e-12)
        u = torch.rand(probs.shape[0], generator=gen, device=probs.device)
        cdf = torch.cumsum(probs, dim=-1)
        return (u.unsqueeze(-1) < cdf).float().argmax(dim=-1)
    return logits.masked_fill(~mask.bool(), float("-inf")).argmax(dim=-1)


# ---------------------------------------------------------------------------
# random_untrained
# ---------------------------------------------------------------------------

class RandomUntrainedPolicy:
    """Uniform-over-mask sampler with a dedicated RNG."""

    def __init__(self, *, seed: int, device: str = "cpu"):
        # RNG lives on CPU; only the chosen action index is moved to the
        # policy device. torch.randint does not accept a CUDA generator.
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.device = device

    def act(self, obs: torch.Tensor, mask: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        E = obs.shape[0]
        actions = torch.zeros(E, dtype=torch.int64, device=self.device)
        for e in range(E):
            legal = torch.nonzero(mask[e]).flatten().tolist()
            r = int(torch.randint(0, len(legal), (1,), generator=self.gen).item())
            actions[e] = legal[r]
        return actions


# ---------------------------------------------------------------------------
# no_update / frozen policy
# ---------------------------------------------------------------------------

class FrozenPolicy:
    """Wraps any actor and never updates weights."""

    def __init__(self, actor: nn.Module, *, device: str = "cpu",
                 seed: int = 0):
        self.actor = actor.to(device).eval()
        self.gen = torch.Generator(device=device).manual_seed(int(seed))
        self.device = device

    def act(self, obs: torch.Tensor, mask: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        with torch.no_grad():
            logits = self.actor.forward(obs)
        return _action_from_logits_masked(logits, mask, sample=sample, gen=self.gen)


# ---------------------------------------------------------------------------
# shuffled_observation control
# ---------------------------------------------------------------------------

class ShuffledObservationPolicy:
    """Trained actor but with observation channels permuted by a fixed
    random permutation.

    Tests whether the actor relies on the causal structure of its input
    or is just memorising a generic pattern. If shuffled == trained in
    performance, the actor is not actually using the obs.
    """

    def __init__(self, actor: nn.Module, *, perm: torch.Tensor,
                 device: str = "cpu", seed: int = 0):
        self.actor = actor.to(device).eval()
        self.perm = perm.to(device)
        self.gen = torch.Generator(device=device).manual_seed(int(seed))
        self.device = device

    @classmethod
    def with_random_perm(cls, actor: nn.Module, *, seed: int, device: str = "cpu"):
        # torch.randperm requires a CPU generator; the resulting perm is
        # moved to the policy device inside __init__.
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(OBS_DIM, generator=g)
        return cls(actor=actor, perm=perm, device=device, seed=seed)

    def act(self, obs: torch.Tensor, mask: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        with torch.no_grad():
            obs_shuffled = obs.index_select(-1, self.perm)
            logits = self.actor.forward(obs_shuffled)
        return _action_from_logits_masked(logits, mask, sample=sample, gen=self.gen)


# ---------------------------------------------------------------------------
# time_only control
# ---------------------------------------------------------------------------

class TimeOnlyPolicy:
    """Actor sees only [step_idx / horizon].

    This is the degenerate baseline: a policy that can only condition on
    time. It cannot beat a real scheduler unless the optimal schedule is
    purely time-based. Implemented as a small MLP over a 1-dim input.
    """

    def __init__(self, *, hidden: int = 64, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(int(seed))
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, N_ACTIONS),
        ).to(device).eval()
        self.gen = torch.Generator(device=device).manual_seed(int(seed))
        self.device = device

    def act(self, obs: torch.Tensor, mask: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        # obs[:, 1] is rem_t in both profiles (channels 0..1 are rem_E, rem_t).
        rem_t = obs[:, 1:2]
        with torch.no_grad():
            logits = self.net(rem_t)
        return _action_from_logits_masked(logits, mask, sample=sample, gen=self.gen)


# ---------------------------------------------------------------------------
# residual_ppo policy (BC logits + alpha * residual)
# ---------------------------------------------------------------------------

class ResidualPolicy(nn.Module):
    """stop-gradient BC logits + alpha * residual_logits.

    alpha is grown 0 -> 1 over training (caller-controlled here; the
    residual PPO trainer would handle the schedule). At alpha=0 this is
    pure BC.
    """

    def __init__(self, bc_actor: nn.Module, residual_actor: nn.Module, *,
                 alpha: float = 0.0):
        super().__init__()
        self.bc_actor = bc_actor
        self.residual_actor = residual_actor
        self.alpha = float(alpha)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            bc_logits = self.bc_actor.forward(obs)
        residual_logits = self.residual_actor.forward(obs)
        return bc_logits + self.alpha * residual_logits

    def distribution(self, obs: torch.Tensor, mask: torch.Tensor):
        logits = self.forward(obs)
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return torch.distributions.Categorical(logits=logits)


# ---------------------------------------------------------------------------
# Policy evaluation harness — same for every control
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy,
    *,
    env_cfg: EnvConfig,
    scenario_seeds: list[int],
    n_action_reps: int = 4,
    sample: bool = True,
    action_seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Per-scenario macro drop_ratio and raw per-rep rows.

    Same shape as trainer.evaluate_actor so cross-policy comparisons
    are harness-invariant.
    """
    gen = torch.Generator(device=device).manual_seed(action_seed)
    # Note: do NOT overwrite policy.gen here. Each policy's RNG device is
    # chosen at construction (CPU for randint-based samplers, device-matched
    # for tensor samplers); overwriting would break the randint path on CUDA.
    per_seed_drops: list[float] = []
    raw_rows: list[dict] = []
    for sd in scenario_seeds:
        rep_drops: list[float] = []
        for rep in range(n_action_reps):
            env = G3BstaLiteVecEnv(env_cfg)
            env.reset(seed=sd)
            for t in range(env_cfg.horizon):
                obs = env._build_observation()
                mask = env._compute_mask()
                action = policy.act(obs, mask, sample=sample)
                if action.dim() == 0:
                    action = action.unsqueeze(0)
                env.step(action)
            rep_drop = float(env.drop_ratio()[0])
            rep_drops.append(rep_drop)
            n_eligible = int(env.counters.n_eligible[0].item())
            raw_rows.append({
                "seed": int(sd), "rep": int(rep),
                "drop_ratio": rep_drop,
                "n_eligible": n_eligible,
                "ledger_residual": int(env.ledger_identity_residual()),
                "accounting_residual": int(env.accounting_residual()[0].item()),
            })
        per_seed_drops.append(sum(rep_drops) / len(rep_drops))
    macro = sum(per_seed_drops) / len(per_seed_drops) if per_seed_drops else float("nan")
    return {
        "per_seed_drops": per_seed_drops,
        "macro_mean_drop": macro,
        "n_seeds": len(scenario_seeds),
        "n_action_reps": n_action_reps,
        "sample": sample,
        "raw_rows": raw_rows,
    }


# ---------------------------------------------------------------------------
# Adapter so MaskedCategoricalActor (from trainer.py) plugs into the
# shared evaluate_policy harness.
# ---------------------------------------------------------------------------

class ActorAdapter:
    """Adapt a MaskedCategoricalActor to the policy.act interface."""

    def __init__(self, actor: nn.Module, *, seed: int = 0, device: str = "cpu"):
        self.actor = actor.to(device).eval()
        self.gen = torch.Generator(device=device).manual_seed(int(seed))
        self.device = device

    def act(self, obs: torch.Tensor, mask: torch.Tensor, *, sample: bool = True) -> torch.Tensor:
        with torch.no_grad():
            if sample:
                dist = self.actor.distribution(obs, mask)
                probs = dist.probs.clamp(min=1e-12)
                u = torch.rand(probs.shape[0], generator=self.gen,
                                device=probs.device)
                cdf = torch.cumsum(probs, dim=-1)
                return (u.unsqueeze(-1) < cdf).float().argmax(dim=-1)
            logits = self.actor.forward(obs)
        return _action_from_logits_masked(logits, mask, sample=False, gen=self.gen)
