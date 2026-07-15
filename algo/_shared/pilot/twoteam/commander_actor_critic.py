"""Two-team learning commander (MAPPO/CTDE Actor-Critic) for WP1 BR + WP2 self-play.

Per TWOTEAM_MULTIFUNCTION_PLAN.md §WP2 + plan snuggly-exploring-parrot.md Step 2
+ TWOTEAM_ENV_FIX_SPEC.md (2026-07-14 FIX 1).

Key design:
  - Single AC, per-team forward (two teams symmetric → share params)
  - Action heads:
      * task_alloc: Dirichlet(α) per aperture (4-way simplex, soft fractions)
                    — spec D2=A mandates continuous fractions (NOT Categorical 1-of-4)
      * beam_target: Categorical(2) per aperture (enemy radar 0 or 1)
      * laser_target: Categorical(2) (enemy radar 0 or 1)
      * emission_on: Bernoulli per aperture
      * freq_hop_rate: Beta(α,β) per aperture → sample ∈ [0,1] → rescale to [1, freq_hop_max]
                    — FIX 1: anti-jam skill dimension. Real-radar frequency agility.
  - Critic: dual trunk
      * central_trunk: obs + privileged → value (CTDE)
      * local_trunk: obs only → value_local (IPPO ablation)
  - α_eff blend is computed in trainer (NOT here) to keep priv[:,4] bug-isolation

Action layout (per team, per env):
  task_alloc[E, R=2, n_fn=4]      Dirichlet samples (sums to 1 over n_fn)
  beam_target[E, R=2]             long ∈ {0, 1}
  laser_target[E]                 long ∈ {0, 1}
  emission_on[E, R=2]             float ∈ [0, 1] (Bernoulli sample)
  freq_hop_rate[E, R=2]           float ∈ [1, freq_hop_max] (Beta sample rescaled)
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
        obs_dim: int = 40,   # WP-A: env.obs_dim 36 → 40 (4 freq-channel slots added)
        privileged_dim: int = 8,
        hidden: int = 256,
        n_fn: int = 4,
        n_aperture: int = 2,
        n_enemy: int = 2,
        dirichlet_min_concentration: float = 0.5,
        freq_hop_max: float = 8.0,   # FIX 1: matches env.freq_hop_max
        beta_min_concentration: float = 0.5,   # FIX 1: avoid degenerate Beta
        n_channels: int = 8,   # WP-C R3: per-aperture channel select (coordination lever)
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.privileged_dim = int(privileged_dim)
        self.hidden = int(hidden)
        self.n_fn = int(n_fn)
        self.n_aperture = int(n_aperture)
        self.n_enemy = int(n_enemy)
        self.dirichlet_min_concentration = float(dirichlet_min_concentration)
        self.freq_hop_max = float(freq_hop_max)
        self.beta_min_concentration = float(beta_min_concentration)
        self.n_channels = int(n_channels)

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
        # FIX 1: freq_hop Beta per aperture: [hidden → n_aperture * 2] (α,β per aperture)
        self.freq_hop_head = nn.Linear(hidden, n_aperture * 2)
        # WP-C R3: channel select per aperture: [hidden → n_aperture * n_channels]
        self.channel_select_head = nn.Linear(hidden, n_aperture * n_channels)

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
        obs: torch.Tensor,
        privileged: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward for a single team's batch."""
        B = obs.shape[0]
        h = self.actor_trunk(obs)

        # --- Dirichlet for task_alloc ---
        raw_task = self.task_alloc_head(h).reshape(B, self.n_aperture, self.n_fn)
        alpha = F.softplus(raw_task) + self.dirichlet_min_concentration
        task_dist = torch.distributions.Dirichlet(alpha)
        task_sample = task_dist.rsample()
        task_logp = task_dist.log_prob(task_sample)

        # --- Beam target ---
        beam_logits = self.beam_target_head(h).reshape(B, self.n_aperture, self.n_enemy)
        beam_dist = torch.distributions.Categorical(logits=beam_logits)
        beam_sample = beam_dist.sample()
        beam_logp = beam_dist.log_prob(beam_sample)

        # --- Laser target ---
        laser_logits = self.laser_target_head(h)
        laser_dist = torch.distributions.Categorical(logits=laser_logits)
        laser_sample = laser_dist.sample()
        laser_logp = laser_dist.log_prob(laser_sample)

        # --- Emission on ---
        emit_logits = self.emission_on_head(h)
        emit_dist = torch.distributions.Bernoulli(logits=emit_logits)
        emit_sample = emit_dist.sample().float()
        emit_logp = emit_dist.log_prob(emit_sample)

        # --- FIX 1: freq_hop Beta → rescale to [1, freq_hop_max] ---
        raw_fh = self.freq_hop_head(h).reshape(B, self.n_aperture, 2)
        fh_alpha = F.softplus(raw_fh[..., 0]) + self.beta_min_concentration
        fh_beta = F.softplus(raw_fh[..., 1]) + self.beta_min_concentration
        fh_dist = torch.distributions.Beta(fh_alpha, fh_beta)
        fh_uniform = fh_dist.rsample()   # [B, n_aperture] ∈ [0, 1]
        fh_logp = fh_dist.log_prob(fh_uniform)   # [B, n_aperture]
        # Rescale to [1, freq_hop_max]
        freq_hop_sample = fh_uniform * (self.freq_hop_max - 1.0) + 1.0   # [B, n_aperture]

        # --- WP-C R3: channel_select Categorical per aperture ---
        chan_logits = self.channel_select_head(h).reshape(B, self.n_aperture, self.n_channels)
        chan_dist = torch.distributions.Categorical(logits=chan_logits)
        chan_sample = chan_dist.sample()   # [B, n_aperture]
        chan_logp = chan_dist.log_prob(chan_sample)   # [B, n_aperture]

        # --- Joint log_prob ---
        log_prob = (task_logp.sum(dim=-1) + beam_logp.sum(dim=-1)
                    + laser_logp + emit_logp.sum(dim=-1)
                    + fh_logp.sum(dim=-1)
                    + chan_logp.sum(dim=-1))

        # --- Critics ---
        value = self.central_trunk(
            torch.cat([obs, privileged], dim=-1)).squeeze(-1)
        value_local = self.local_trunk(obs).squeeze(-1)

        action = {
            "task_alloc": task_sample,
            "beam_target": beam_sample.long(),
            "laser_target": laser_sample.long(),
            "emission_on": emit_sample,
            "freq_hop_rate": freq_hop_sample,   # FIX 1
            "channel_select": chan_sample.long(),   # WP-C R3
        }
        return action, log_prob, value, value_local

    # ------------------------------------------------------------------
    # Evaluate actions (for PPO update)
    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: Dict[str, torch.Tensor],
        privileged: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns log_prob, value, value_local, entropy for PPO update."""
        B = obs.shape[0]
        h = self.actor_trunk(obs)

        # Dirichlet
        raw_task = self.task_alloc_head(h).reshape(B, self.n_aperture, self.n_fn)
        alpha = F.softplus(raw_task) + self.dirichlet_min_concentration
        task_dist = torch.distributions.Dirichlet(alpha)
        task_logp = task_dist.log_prob(action["task_alloc"])
        task_entropy = task_dist.entropy()

        # Beam
        beam_logits = self.beam_target_head(h).reshape(B, self.n_aperture, self.n_enemy)
        beam_dist = torch.distributions.Categorical(logits=beam_logits)
        beam_logp = beam_dist.log_prob(action["beam_target"])
        beam_entropy = beam_dist.entropy()

        # Laser
        laser_logits = self.laser_target_head(h)
        laser_dist = torch.distributions.Categorical(logits=laser_logits)
        laser_logp = laser_dist.log_prob(action["laser_target"])
        laser_entropy = laser_dist.entropy()

        # Emission
        emit_logits = self.emission_on_head(h)
        emit_dist = torch.distributions.Bernoulli(logits=emit_logits)
        emit_logp = emit_dist.log_prob(action["emission_on"])
        emit_entropy = emit_dist.entropy()

        # FIX 1: freq_hop Beta — need to convert action["freq_hop_rate"] back to [0,1]
        raw_fh = self.freq_hop_head(h).reshape(B, self.n_aperture, 2)
        fh_alpha = F.softplus(raw_fh[..., 0]) + self.beta_min_concentration
        fh_beta = F.softplus(raw_fh[..., 1]) + self.beta_min_concentration
        fh_dist = torch.distributions.Beta(fh_alpha, fh_beta)
        # Inverse rescale: freq_hop ∈ [1, freq_hop_max] → uniform ∈ [0, 1]
        fh_uniform = (action["freq_hop_rate"] - 1.0) / (self.freq_hop_max - 1.0)
        fh_uniform = fh_uniform.clamp(1e-4, 1 - 1e-4)   # avoid Beta boundary singularities
        fh_logp = fh_dist.log_prob(fh_uniform)
        fh_entropy = fh_dist.entropy()

        # WP-C R3: channel_select Categorical
        chan_logits = self.channel_select_head(h).reshape(B, self.n_aperture, self.n_channels)
        chan_dist = torch.distributions.Categorical(logits=chan_logits)
        chan_logp = chan_dist.log_prob(action["channel_select"])
        chan_entropy = chan_dist.entropy()

        log_prob = (task_logp.sum(dim=-1) + beam_logp.sum(dim=-1)
                    + laser_logp + emit_logp.sum(dim=-1)
                    + fh_logp.sum(dim=-1)
                    + chan_logp.sum(dim=-1))
        entropy = (task_entropy.sum(dim=-1) + beam_entropy.sum(dim=-1)
                   + laser_entropy + emit_entropy.sum(dim=-1)
                   + fh_entropy.sum(dim=-1)
                   + chan_entropy.sum(dim=-1)) / (
                   self.n_aperture * 4 + 1 + self.n_aperture)   # +n_aperture channel heads

        value = self.central_trunk(
            torch.cat([obs, privileged], dim=-1)).squeeze(-1)
        value_local = self.local_trunk(obs).squeeze(-1)

        return log_prob, value, value_local, entropy

    # ------------------------------------------------------------------
    # Deterministic / sampled action for env stepping
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action_for_env(
        self,
        obs_team: torch.Tensor,
        privileged_team: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get per-team action slice for env.step."""
        action, log_prob, _, _ = self.forward(obs_team, privileged_team)
        if deterministic:
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
            # FIX 1: Beta mean = α/(α+β), rescale to [1, freq_hop_max]
            raw_fh = self.freq_hop_head(h).reshape(B, self.n_aperture, 2)
            fh_alpha = F.softplus(raw_fh[..., 0]) + self.beta_min_concentration
            fh_beta = F.softplus(raw_fh[..., 1]) + self.beta_min_concentration
            fh_mean_uniform = fh_alpha / (fh_alpha + fh_beta)
            fh_mean = fh_mean_uniform * (self.freq_hop_max - 1.0) + 1.0
            # WP-C R3: channel select argmax
            chan_logits = self.channel_select_head(h).reshape(B, self.n_aperture, self.n_channels)
            chan_argmax = chan_logits.argmax(dim=-1)
            action = {
                "task_alloc": task_mean,
                "beam_target": beam_argmax.long(),
                "laser_target": laser_argmax.long(),
                "emission_on": emit_round,
                "freq_hop_rate": fh_mean,
                "channel_select": chan_argmax.long(),
            }
        return action, log_prob
