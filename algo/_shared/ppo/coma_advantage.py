"""COMA counterfactual advantage — sample-based approximation (COMA-S).

Approximates:
    A_i(s, a) = Q(s, a) − b_i(s, a_{−i})
    b_i(s, a_{−i}) = E_{a_i' ~ π_i}[Q(s, a_{−i}, a_i')]

Why sampling instead of exact marginalize:
  The original Foerster 2018 paper marginalizes over each agent's full action
  vector. For our radar actor that's a 25-sub-array × Categorical(4) multi-
  discrete + 200-dim continuous + 3-dim vehicle, exact joint marginalize
  needs 4^25 ≈ 1e15 evaluations — infeasible. The classical COMA engineering
  practice for mixed discrete/continuous actions is sample-based estimation
  (sample K counterfactual action vectors per agent, average Q). This file
  implements that sample-based variant.

  This is sometimes called "COMA-S" in the literature. It is conceptually
  faithful to the original (counterfactual baseline for credit assignment)
  while being tractable on multi-discrete + continuous action spaces.

Bit-exact reproducibility:
  Continuous sampling draws from the actor's current policy via
  actor.get_action(obs). The caller is responsible for setting
  torch.manual_seed(step_seed) before invoking this function —
  the formula is documented at the call site (train_laser.py _ppo_update
  COMA branch). Inside this function we make no RNG calls other than
  through the actor's standard get_action, so the seed set by the caller
  fully determines the result.

Joint action layout (matches coma_critic.py, 1222 dims):
    offset  size  source
    0       5     commander team-0   [fire_pm1, aim_x, aim_y, aim_z, jam]
    5       5     commander team-1
    10      303   radar team-0 radar-0
    313     303   radar team-0 radar-1
    616     303   radar team-1 radar-0
    919     303   radar team-1 radar-1
                                       total = 1222

Per-radar compact block layout (303 dims = 25 sub-arrays × 12 + 3 vehicle):
    [0:25*4]      task_frac       one-hot, [B, 25, 4] flattened to [B, 100]
    [100:300]     params          [B, 25, 8] flattened to [B, 200]
    [300:303]     vehicle         [B, 3]

Caller (train_laser.py) is responsible for compressing the actor's
element-level 13753-dim radar action into the 303-dim compact block via
SubArrayRadarActorCritic._extract_sub_from_elem followed by the layout
documented above. See train_laser._build_coma_joint_action for the canonical
implementation.
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


JOINT_ACTION_DIM = 1222
CMD_OFFSETS = (0, 5)              # team-0 cmd [0:5], team-1 cmd [5:10]
RADAR_OFFSETS = (10, 313, 616, 919)  # 4 radars in joint_action order
                                      #   idx 0,1 = team-0 radar-0, radar-1
                                      #   idx 2,3 = team-1 radar-0, radar-1
RADAR_BLOCK_DIM = 303
CMD_BLOCK_DIM = 5


def _cmd_offset(team_idx: int) -> int:
    return CMD_OFFSETS[team_idx]


def _radar_offset(agent_idx_in_team: int, team_idx: int) -> int:
    """Offset of a specific radar's 303-dim block in joint_action."""
    radar_joint_idx = team_idx * 2 + agent_idx_in_team  # 0,1,2,3
    return RADAR_OFFSETS[radar_joint_idx]


def coma_counterfactual_advantage(
    critic: nn.Module,
    team_state: torch.Tensor,           # [B, 104]
    joint_action: torch.Tensor,         # [B, 1222]
    agent_kind: str,                    # "commander" | "radar"
    team_idx: int,                      # 0 or 1
    agent_idx: int,                     # 0 for commander; 0/1 for radars in team
    actor: nn.Module,                   # CommanderActorCritic or SubArrayRadarActorCritic
    obs: torch.Tensor,                  # agent's local obs [B, obs_dim]
    n_samples: int = 8,
    advantage_clip_sigma: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute COMA-Sample counterfactual advantage.

    Returns:
        advantage: [B]   — Q(s, a) − baseline_i(s, a_{−i})
        baseline:  [B]   — counterfactual baseline (mean Q over K samples)

    Caller-side RNG contract: caller sets `torch.manual_seed(step_seed)`
    before calling. This function consumes RNG only through
    `actor.get_action(obs)` (K calls). The K samples are therefore
    deterministic given step_seed.

    Args:
        critic: COMACritic
        agent_kind: "commander" or "radar"
        team_idx: 0 or 1 (which team this agent is on)
        agent_idx: 0 for commander (always); 0 or 1 for radars within a team
        actor: must expose `get_action(obs)` returning `(action, ...)` tuple.
               For commander: action shape [B, 5].
               For radar: action shape [B, 13753] (element-level flat) and
               must also expose `compact_joint_block(action)` → [B, 303]
               (or `to_joint_block` for short). See train_laser for the helper
               that does the same compression at storage time.
        obs: agent's local observation
        n_samples: K, number of counterfactual action draws per sample.
                   Default 8 (paper-illustrative; 4 also fine).
        advantage_clip_sigma: if >0, clip advantage to ±N*std before return.
    """
    B = joint_action.shape[0]
    Q_actual = critic(team_state, joint_action).squeeze(-1)  # [B]

    Q_samples = []
    for _ in range(n_samples):
        with torch.no_grad():
            sampled = actor.get_action(obs)
        a_sample = sampled[0] if isinstance(sampled, tuple) else sampled

        cf = joint_action.clone()
        if agent_kind == "commander":
            offset = _cmd_offset(team_idx)
            cf[:, offset:offset + CMD_BLOCK_DIM] = a_sample
        elif agent_kind == "radar":
            offset = _radar_offset(agent_idx, team_idx)
            # Radar actor returns element-level 13753-dim action; compress to
            # 303-dim compact block via actor's helper method.
            if a_sample.shape[-1] == RADAR_BLOCK_DIM:
                a_compact = a_sample
            elif hasattr(actor, "compact_joint_block"):
                a_compact = actor.compact_joint_block(a_sample)
            elif hasattr(actor, "to_sub_array_form"):
                a_compact = actor.to_sub_array_form(a_sample)
            else:
                raise RuntimeError(
                    f"COMA: radar actor returned {a_sample.shape[-1]}-dim "
                    f"action; needs compact_joint_block() or "
                    f"to_sub_array_form() to compress to {RADAR_BLOCK_DIM} dims."
                )
            cf[:, offset:offset + RADAR_BLOCK_DIM] = a_compact
        else:
            raise ValueError(f"COMA: unknown agent_kind={agent_kind!r}")

        Q_k = critic(team_state, cf).squeeze(-1)  # [B]
        Q_samples.append(Q_k)

    Q_stack = torch.stack(Q_samples, dim=0)  # [K, B]
    baseline = Q_stack.mean(dim=0)            # [B]
    advantage = Q_actual - baseline           # [B]

    if advantage_clip_sigma > 0:
        std = advantage.std() + 1e-8
        advantage = advantage.clamp(
            -advantage_clip_sigma * std, advantage_clip_sigma * std,
        )

    return advantage, baseline
