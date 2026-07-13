"""Two-team learning commander (MAPPO/CTDE Actor-Critic) for WP1 BR + WP2 self-play.

Per TWOTEAM_MULTIFUNCTION_PLAN.md §WP2 + plan snuggly-exploring-parrot.md Step 2.

Key design:
  - Single AC, per-team forward (two teams symmetric → share params)
  - Action heads:
      * task_alloc: Dirichlet(α) per aperture (4-way simplex, soft fractions)
                    — spec D2=A mandates continuous fractions (NOT Categorical 1-of-4)
      * beam_target: Categorical(2) per aperture (enemy radar 0 or 1)
      * laser_target: Categorical(2) (enemy radar 0 or 1)
      * emission_on: Bernoulli per aperture
  - Critic: dual trunk
      * central_trunk: obs + privileged → value (CTDE)
      * local_trunk: obs only → value_local (IPPO ablation)
  - α_eff blend is computed in trainer (NOT here) to keep priv[:,4] bug-isolation

Action layout (per team, per env):
  task_alloc[E, R=2, n_fn=4]      Dirichlet samples (sums to 1 over n_fn)
  beam_target[E, R=2]             long ∈ {0, 1}
  laser_target[E]                 long ∈ {0, 1}
  emission_on[E, R=2]             float ∈ [0, 1] (Bernoulli sample)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class TwoTeamCommanderActorCritic(nn.Module):
    """Per-team learning commander. Forward signature accepts [E, T, ...] or [E, ...]."""

    def __init__(
        self,
        obs_dim: int = 36,
        privileged_dim: int = 8,
        hidden: int = 256,
        n_fn: int = 4,
        n_aperture: int = 2,
        n_enemy: int = 2,
        dirichlet_min_concentration: float = 0.5,   # avoid degenerate Dirichlet
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.privileged_dim = int(privileged_dim)
        self.hidden = int(hidden)
        self.n_fn = int(n_fn)
        self.n_aperture = int(n_aperture)
        self.n_enemy = int(n_enemy)
        self.dirichlet_min_concentration = float(dirichlet_min_concentration)

        # --- Shared actor trunk ---
        self.actor_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Dirichlet concentration per aperture: [hidden → n_aperture * n_fn]
        self.task_alloc_head = nn.Linear(hidden, n_aperture * n_fn)
        # Beam target per aperture: [hidden → n_aperture * n_enemy]
        self.beam_target_head = nn.Linear(hidden, n_aperture * n_enemy)
        # Laser target: [hidden → n_enemy]
        self.laser_target_head = nn.Linear(hidden, n_enemy)
        # Emission on per aperture: [hidden → n_aperture]
        self.emission_on_head = nn.Linear(hidden, n_aperture)

        # --- Critics (CTDE central + IPPO local) ---
        self.central_trunk = nn.Sequential(
            nn.Linear(obs_dim + privileged_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.local_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,            # [B, obs_dim] (B = E or E*T flattened)
        privileged: Optional[torch.Tensor] = None,   # [B, priv_dim]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward for a single team's batch. Returns action_dict, log_prob, value, value_local.

        Action dict shapes (per team):
          task_alloc: [B, n_aperture, n_fn]   (Dirichlet sample, sums to 1 over n_fn)
          beam_target: [B, n_aperture]         (Categorical sample)
          laser_target: [B]                    (Categorical sample)
          emission_on: [B, n_aperture]         (Bernoulli sample)
        """
        B = obs.shape[0]
        h = self.actor_trunk(obs)   # [B, hidden]

        # --- Dirichlet parameters for task_alloc ---
        # raw [B, n_aperture, n_fn] → positive concentrations via softplus + floor
        raw_task = self.task_alloc_head(h).reshape(B, self.n_aperture, self.n_fn)
        alpha = F.softplus(raw_task) + self.dirichlet_min_concentration
        task_dist = torch.distributions.Dirichlet(alpha)
        task_sample = task_dist.rsample()   # [B, n_aperture, n_fn], sums to 1
        task_logp = task_dist.log_prob(task_sample)   # [B, n_aperture]

        # --- Beam target (Categorical per aperture) ---
        beam_logits = self.beam_target_head(h).reshape(B, self.n_aperture, self.n_enemy)
        beam_dist = torch.distributions.Categorical(logits=beam_logits)
        beam_sample = beam_dist.sample()   # [B, n_aperture]
        beam_logp = beam_dist.log_prob(beam_sample)   # [B, n_aperture]

        # --- Laser target (Categorical) ---
        laser_logits = self.laser_target_head(h)   # [B, n_enemy]
        laser_dist = torch.distributions.Categorical(logits=laser_logits)
        laser_sample = laser_dist.sample()   # [B]
        laser_logp = laser_dist.log_prob(laser_sample)   # [B]

        # --- Emission on (Bernoulli per aperture) ---
        emit_logits = self.emission_on_head(h)   # [B, n_aperture]
        emit_dist = torch.distributions.Bernoulli(logits=emit_logits)
        emit_sample = emit_dist.sample().float()   # [B, n_aperture]
        emit_logp = emit_dist.log_prob(emit_sample)   # [B, n_aperture]

        # --- Joint log_prob (sum across all heads) ---
        # task_logp: [B, n_aperture], sum over apertures
        # beam_logp: [B, n_aperture], sum
        # laser_logp: [B]
        # emit_logp: [B, n_aperture], sum
        log_prob = (task_logp.sum(dim=-1) + beam_logp.sum(dim=-1)
                    + laser_logp + emit_logp.sum(dim=-1))   # [B]

        # --- Critics ---
        value = self.central_trunk(
            torch.cat([obs, privileged], dim=-1)).squeeze(-1)   # [B]
        value_local = self.local_trunk(obs).squeeze(-1)   # [B]

        action = {
            "task_alloc": task_sample,                # [B, n_aperture, n_fn]
            "beam_target": beam_sample.long(),        # [B, n_aperture]
            "laser_target": laser_sample.long(),      # [B]
            "emission_on": emit_sample,               # [B, n_aperture]
        }
        return action, log_prob, value, value_local

    # ------------------------------------------------------------------
    # Evaluate actions (for PPO update)
    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs: torch.Tensor,                # [B, obs_dim]
        action: Dict[str, torch.Tensor],  # action dict
        privileged: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns log_prob, value, value_local, entropy for PPO update."""
        B = obs.shape[0]
        h = self.actor_trunk(obs)

        # Dirichlet
        raw_task = self.task_alloc_head(h).reshape(B, self.n_aperture, self.n_fn)
        alpha = F.softplus(raw_task) + self.dirichlet_min_concentration
        task_dist = torch.distributions.Dirichlet(alpha)
        task_logp = task_dist.log_prob(action["task_alloc"])   # [B, n_aperture]
        task_entropy = task_dist.entropy()   # [B, n_aperture]

        # Beam
        beam_logits = self.beam_target_head(h).reshape(B, self.n_aperture, self.n_enemy)
        beam_dist = torch.distributions.Categorical(logits=beam_logits)
        beam_logp = beam_dist.log_prob(action["beam_target"])   # [B, n_aperture]
        beam_entropy = beam_dist.entropy()   # [B, n_aperture]

        # Laser
        laser_logits = self.laser_target_head(h)
        laser_dist = torch.distributions.Categorical(logits=laser_logits)
        laser_logp = laser_dist.log_prob(action["laser_target"])   # [B]
        laser_entropy = laser_dist.entropy()   # [B]

        # Emission
        emit_logits = self.emission_on_head(h)
        emit_dist = torch.distributions.Bernoulli(logits=emit_logits)
        emit_logp = emit_dist.log_prob(action["emission_on"])   # [B, n_aperture]
        emit_entropy = emit_dist.entropy()   # [B, n_aperture]

        log_prob = (task_logp.sum(dim=-1) + beam_logp.sum(dim=-1)
                    + laser_logp + emit_logp.sum(dim=-1))
        entropy = (task_entropy.sum(dim=-1) + beam_entropy.sum(dim=-1)
                   + laser_entropy + emit_entropy.sum(dim=-1)) / (
                   self.n_aperture * 3 + 1)   # normalize by num heads

        value = self.central_trunk(
            torch.cat([obs, privileged], dim=-1)).squeeze(-1)
        value_local = self.local_trunk(obs).squeeze(-1)

        return log_prob, value, value_local, entropy

    # ------------------------------------------------------------------
    # Deterministic / sampled action for env stepping (per-team slice)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action_for_env(
        self,
        obs_team: torch.Tensor,             # [E, obs_dim]
        privileged_team: torch.Tensor,      # [E, priv_dim]
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get per-team action slice for env.step.

        Returns (action_dict_per_team, log_prob).
        action_dict matches ExtremeCommander.get_action shape:
          task_alloc: [E, n_aperture, n_fn]
          beam_target: [E, n_aperture]
          laser_target: [E]
          emission_on: [E, n_aperture]
        """
        action, log_prob, _, _ = self.forward(obs_team, privileged_team)
        if deterministic:
            # Use mean of distributions instead of sample
            B = obs_team.shape[0]
            h = self.actor_trunk(obs_team)
            raw_task = self.task_alloc_head(h).reshape(B, self.n_aperture, self.n_fn)
            alpha = F.softplus(raw_task) + self.dirichlet_min_concentration
            task_mean = alpha / alpha.sum(dim=-1, keepdim=True)
            beam_logits = self.beam_target_head(h).reshape(B, self.n_aperture, self.n_enemy)
            beam_argmax = beam_logits.argmax(dim=-1)
            laser_logits = self.laser_target_head(h)
            laser_argmax = laser_logits.argmax(dim=-1)
            emit_logits = self.emission_on_head(h)
            emit_round = (torch.sigmoid(emit_logits) > 0.5).float()
            action = {
                "task_alloc": task_mean,
                "beam_target": beam_argmax.long(),
                "laser_target": laser_argmax.long(),
                "emission_on": emit_round,
            }
        return action, log_prob
