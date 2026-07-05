"""COMA centralized Q-critic (Foerster et al. 2018).

Q(team_state, all_agents_actions) → scalar. Counterpart to TeamCritic
(MAPPO V critic) but conditions on the full joint action vector so the
counterfactual baseline can be computed by marginalizing one agent's
action dim at a time.

Used by coma_advantage.coma_counterfactual_advantage — see that file
for the marginalize / sampling logic.

Joint action layout (1222 dims — compact sub-array param form):
    Two teams of 1 commander + 2 deployed radars.

    offset  size  source
    0       5     commander team-0 action  [fire_pm1, aim_x, aim_y, aim_z, jam]
    5       5     commander team-1 action  (opponent; same layout)
    10      303   radar team-0 radar-0     (K=25 sub-arrays × 12 + 3 vehicle)
    313     303   radar team-0 radar-1
    616     303   radar team-1 radar-0
    919     303   radar team-1 radar-1
                                       total = 1222

Per-radar sub-array compact block (12 dims per sub-array × 25 + 3 vehicle):
    [0:4]   task_frac       one-hot of Categorical(4)
    [4:12]  params          (continuous: beam_az, beam_el, detect×3, jam×3)
    + vehicle [3]           (continuous, global per radar)

Per-dim semantics (so the critic sees consistent real-valued inputs):
  fire          ∈ {-1, +1}      (squashed from Bernoulli)
  aim_xyz       ∈ [-1, 1]       (tanh)
  jam           ∈ [-1, 1]       (tanh)
  task_frac     ∈ {0, 1}        (one-hot per sub-array)
  params        ∈ [-1, 1]       (tanh; from actor_critic.py (az-0.5)*2 transform)
  vehicle       ∈ [-1, 1]       (tanh)

The first Linear has LayerNorm to absorb the heterogeneous scale mix
(fire can be ±1 while sigmoid-derived params stay in [-1, 1]); without
it, orthogonal init on a 1222-dim row can produce Q magnitudes that
blow up downstream in the advantage computation.
"""

import torch
import torch.nn as nn


# Joint action vector dimensionality. Hardcoded because the COMA layout is
# part of the contract between COMACritic, coma_advantage, and train_laser
# (any change here must be mirrored in all three places). See module docstring
# for the per-block breakdown.
JOINT_ACTION_DIM = 1222
TEAM_STATE_DIM = 104


class COMACritic(nn.Module):
    """Centralized Q critic for COMA.

    Forward: Linear(1326→256) → LayerNorm → ReLU
             → Linear(256→256) → ReLU
             → Linear(256→1)

    Input layout: concat([team_state[B,104], joint_action[B,1222]], dim=-1).
    Output: Q(s, a) scalar, shape [B, 1].
    """

    def __init__(
        self,
        team_state_dim: int = 104,
        joint_action_dim: int = 1222,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.team_state_dim = team_state_dim
        self.joint_action_dim = joint_action_dim
        self.input_dim = team_state_dim + joint_action_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # Orthogonal init on hidden layers (matches TeamCritic / actor_critic
        # convention); small init on the final layer so initial Q ≈ 0 doesn't
        # explode advantage variance on the first minibatch.
        for m in [self.net[0], self.net[3]]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
        last = self.net[5]
        nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.constant_(last.bias, 0.0)

    def forward(
        self,
        team_state: torch.Tensor,    # [B, 104]
        joint_action: torch.Tensor,  # [B, 1222] or [B*, 1222] (broadcastable)
    ) -> torch.Tensor:
        """Return Q(s, a) of shape [B, 1] (or matching leading dim)."""
        if team_state.shape[0] != joint_action.shape[0]:
            K = joint_action.shape[0] // team_state.shape[0]
            assert team_state.shape[0] * K == joint_action.shape[0], (
                f"COMACritic forward: cannot broadcast team_state "
                f"{team_state.shape} vs joint_action {joint_action.shape}"
            )
            team_state = team_state.repeat_interleave(K, dim=0)
        x = torch.cat([team_state, joint_action], dim=-1)
        return self.net(x)

