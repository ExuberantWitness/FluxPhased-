"""Two-team learning commander (MAPPO/CTDE Actor-Critic) — WP-3 blind version.

Per FLUXPH_BLIND_ADVERSARIAL_SPEC.md §4. WP-3 M0:
  - Removed legacy `beam_target_head` (god-view Categorical over enemy index)
  - Added permutation-invariant detection-list encoder (DeepSets mean-pool)
    over K_max detections: (z_x, z_y, snr_db, is_fa, mask) → 32-d embedding
  - Action heads (all blind — derived from obs/belief, no enemy truth):
      * task_alloc: Dirichlet(α) per aperture (4-way simplex, continuous fractions)
      * beam_direction: Beta(α,β) per aperture → azimuth ∈ [-π, π]
      * laser_target: Categorical(2) (slot id, NOT enemy id — env does belief check)
      * emission_on: Bernoulli per aperture
      * freq_hop_rate: Beta(α,β) per aperture → [1, freq_hop_max]
      * channel_select: Categorical per aperture (ECCM)
  - Critics: dual trunk
      * central_trunk: obs + detect_emb + privileged → value (CTDE)
      * local_trunk: obs + detect_emb → value_local (IPPO ablation)
  - α_eff blend is computed in trainer (NOT here) to keep priv[:,4] bug-isolation

Action layout (per team, per env):
  task_alloc[E, R=2, n_fn=4]      Dirichlet samples (sums to 1 over n_fn)
  beam_direction[E, R=2]          float ∈ [-π, π] (Beta sample rescaled)
  laser_target[E]                 long ∈ {0, 1}  (slot id; env checks belief)
  emission_on[E, R=2]             float ∈ [0, 1] (Bernoulli sample)
  freq_hop_rate[E, R=2]           float ∈ [1, freq_hop_max] (Beta sample rescaled)
  channel_select[E, R=2]          long ∈ {0, n_channels-1}
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class TwoTeamCommanderActorCritic(nn.Module):
    """Per-team learning commander. Forward signature accepts [E, T, ...] or [E, ...]."""

    def __init__(
        self,
        obs_dim: int = 44,
        privileged_dim: int = 8,
        hidden: int = 256,
        n_fn: int = 4,
        n_aperture: int = 2,
        n_enemy: int = 2,   # kept for laser_target head arity (slot count, post-WP-2)
        k_max: int = 5,
        detect_emb_dim: int = 32,
        dirichlet_min_concentration: float = 0.5,
        freq_hop_max: float = 8.0,
        beta_min_concentration: float = 0.5,
        n_channels: int = 8,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.privileged_dim = int(privileged_dim)
        self.hidden = int(hidden)
        self.n_fn = int(n_fn)
        self.n_aperture = int(n_aperture)
        self.n_enemy = int(n_enemy)
        self.k_max = int(k_max)
        self.detect_emb_dim = int(detect_emb_dim)
        self.dirichlet_min_concentration = float(dirichlet_min_concentration)
        self.freq_hop_max = float(freq_hop_max)
        self.beta_min_concentration = float(beta_min_concentration)
        self.n_channels = int(n_channels)

        # --- WP-3 M0: Detection-list encoder (DeepSets mean-pool) ---
        # Per-detection features: (z_x, z_y, snr_db, is_fa_float, mask_float) = 5 dims
        self.detect_mlp = nn.Sequential(
            nn.Linear(5, self.detect_emb_dim), nn.Tanh(),
            nn.Linear(self.detect_emb_dim, self.detect_emb_dim), nn.Tanh(),
        )

        trunk_in = self.obs_dim + self.detect_emb_dim

        # --- Shared actor trunk ---
        self.actor_trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Dirichlet concentration per aperture: [hidden → n_aperture * n_fn]
        self.task_alloc_head = nn.Linear(hidden, n_aperture * n_fn)
        # beam_direction per aperture (NEW, no god-view). Beta(α,β) → azimuth ∈ [-π, π].
        # Output shape [B, n_aperture, 2] (α,β per aperture).
        self.beam_direction_head = nn.Linear(hidden, n_aperture * 2)
        # Laser target: [hidden → n_enemy]  (n_enemy = slot count after WP-2)
        self.laser_target_head = nn.Linear(hidden, n_enemy)
        # Emission on per aperture: [hidden → n_aperture]
        self.emission_on_head = nn.Linear(hidden, n_aperture)
        # freq_hop Beta per aperture: [hidden → n_aperture * 2] (α,β per aperture)
        self.freq_hop_head = nn.Linear(hidden, n_aperture * 2)
        # WP-C R3: channel select per aperture: [hidden → n_aperture * n_channels]
        self.channel_select_head = nn.Linear(hidden, n_aperture * n_channels)

        # --- Critics (CTDE central + IPPO local) ---
        self.central_trunk = nn.Sequential(
            nn.Linear(trunk_in + privileged_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.local_trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _detect_embedding(self, detect_list: torch.Tensor) -> torch.Tensor:
        """Per-team detection embedding via DeepSets mean-pool.

        Args:
            detect_list: [..., K_max, 5]  (leading dims arbitrary)
        Returns:
            embedding: [..., detect_emb_dim]
        """
        # MLP per detection, then mean over K_max (permutation-invariant)
        return self.detect_mlp(detect_list).mean(dim=-2)

    def _trunk_input(
        self, obs: torch.Tensor, detect_list: torch.Tensor
    ) -> torch.Tensor:
        """Concat obs + detect_emb: [..., obs_dim + detect_emb_dim]."""
        detect_emb = self._detect_embedding(detect_list)
        return torch.cat([obs, detect_emb], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        obs: torch.Tensor,
        detect_list: torch.Tensor,
        privileged: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward for a single team's batch."""
        B = obs.shape[0]
        trunk_in = self._trunk_input(obs, detect_list)
        h = self.actor_trunk(trunk_in)

        # --- Dirichlet for task_alloc ---
        raw_task = self.task_alloc_head(h).reshape(B, self.n_aperture, self.n_fn)
        alpha = F.softplus(raw_task) + self.dirichlet_min_concentration
        task_dist = torch.distributions.Dirichlet(alpha)
        task_sample = task_dist.rsample()
        task_logp = task_dist.log_prob(task_sample)

        # --- beam_direction (Beta → azimuth ∈ [-π, π]) ---
        raw_bd = self.beam_direction_head(h).reshape(B, self.n_aperture, 2)
        bd_alpha = F.softplus(raw_bd[..., 0]) + self.beta_min_concentration
        bd_beta = F.softplus(raw_bd[..., 1]) + self.beta_min_concentration
        bd_dist = torch.distributions.Beta(bd_alpha, bd_beta)
        bd_uniform = bd_dist.rsample()                            # [B, n_aperture] ∈ [0, 1]
        bd_logp = bd_dist.log_prob(bd_uniform)                    # [B, n_aperture]
        # Rescale [0, 1] → [-π, π]
        beam_direction_sample = bd_uniform * 2.0 * math.pi - math.pi   # [B, n_aperture]

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

        # --- freq_hop Beta → rescale to [1, freq_hop_max] ---
        raw_fh = self.freq_hop_head(h).reshape(B, self.n_aperture, 2)
        fh_alpha = F.softplus(raw_fh[..., 0]) + self.beta_min_concentration
        fh_beta = F.softplus(raw_fh[..., 1]) + self.beta_min_concentration
        fh_dist = torch.distributions.Beta(fh_alpha, fh_beta)
        fh_uniform = fh_dist.rsample()   # [B, n_aperture] ∈ [0, 1]
        fh_logp = fh_dist.log_prob(fh_uniform)   # [B, n_aperture]
        freq_hop_sample = fh_uniform * (self.freq_hop_max - 1.0) + 1.0

        # --- channel_select Categorical per aperture ---
        chan_logits = self.channel_select_head(h).reshape(B, self.n_aperture, self.n_channels)
        chan_dist = torch.distributions.Categorical(logits=chan_logits)
        chan_sample = chan_dist.sample()
        chan_logp = chan_dist.log_prob(chan_sample)

        # --- Joint log_prob ---
        log_prob = (task_logp.sum(dim=-1) + bd_logp.sum(dim=-1)
                    + laser_logp + emit_logp.sum(dim=-1)
                    + fh_logp.sum(dim=-1)
                    + chan_logp.sum(dim=-1))

        # --- Critics ---
        value = self.central_trunk(
            torch.cat([trunk_in, privileged], dim=-1)).squeeze(-1)
        value_local = self.local_trunk(trunk_in).squeeze(-1)

        action = {
            "task_alloc": task_sample,
            "beam_direction": beam_direction_sample,
            "laser_target": laser_sample.long(),
            "emission_on": emit_sample,
            "freq_hop_rate": freq_hop_sample,
            "channel_select": chan_sample.long(),
        }
        return action, log_prob, value, value_local

    # ------------------------------------------------------------------
    # Evaluate actions (for PPO update)
    # ------------------------------------------------------------------
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        detect_list: torch.Tensor,
        action: Dict[str, torch.Tensor],
        privileged: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns log_prob, value, value_local, entropy for PPO update."""
        B = obs.shape[0]
        trunk_in = self._trunk_input(obs, detect_list)
        h = self.actor_trunk(trunk_in)

        # Dirichlet
        raw_task = self.task_alloc_head(h).reshape(B, self.n_aperture, self.n_fn)
        alpha = F.softplus(raw_task) + self.dirichlet_min_concentration
        task_dist = torch.distributions.Dirichlet(alpha)
        task_logp = task_dist.log_prob(action["task_alloc"])
        task_entropy = task_dist.entropy()

        # beam_direction Beta — inverse-rescale action from [-π,π] to [0,1]
        raw_bd = self.beam_direction_head(h).reshape(B, self.n_aperture, 2)
        bd_alpha = F.softplus(raw_bd[..., 0]) + self.beta_min_concentration
        bd_beta = F.softplus(raw_bd[..., 1]) + self.beta_min_concentration
        bd_dist = torch.distributions.Beta(bd_alpha, bd_beta)
        # Inverse: azimuth ∈ [-π, π] → uniform ∈ [0, 1]
        bd_uniform = (action["beam_direction"] + math.pi) / (2.0 * math.pi)
        bd_uniform = bd_uniform.clamp(1e-4, 1 - 1e-4)
        bd_logp = bd_dist.log_prob(bd_uniform)
        bd_entropy = bd_dist.entropy()

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

        # freq_hop Beta — inverse-rescale action from [1, freq_hop_max] to [0,1]
        raw_fh = self.freq_hop_head(h).reshape(B, self.n_aperture, 2)
        fh_alpha = F.softplus(raw_fh[..., 0]) + self.beta_min_concentration
        fh_beta = F.softplus(raw_fh[..., 1]) + self.beta_min_concentration
        fh_dist = torch.distributions.Beta(fh_alpha, fh_beta)
        fh_uniform = (action["freq_hop_rate"] - 1.0) / (self.freq_hop_max - 1.0)
        fh_uniform = fh_uniform.clamp(1e-4, 1 - 1e-4)
        fh_logp = fh_dist.log_prob(fh_uniform)
        fh_entropy = fh_dist.entropy()

        # channel_select Categorical
        chan_logits = self.channel_select_head(h).reshape(B, self.n_aperture, self.n_channels)
        chan_dist = torch.distributions.Categorical(logits=chan_logits)
        chan_logp = chan_dist.log_prob(action["channel_select"])
        chan_entropy = chan_dist.entropy()

        log_prob = (task_logp.sum(dim=-1) + bd_logp.sum(dim=-1)
                    + laser_logp + emit_logp.sum(dim=-1)
                    + fh_logp.sum(dim=-1)
                    + chan_logp.sum(dim=-1))
        entropy = (task_entropy.sum(dim=-1) + bd_entropy.sum(dim=-1)
                   + laser_entropy + emit_entropy.sum(dim=-1)
                   + fh_logp.sum(dim=-1)   # placeholder, replaced below
                   + chan_entropy.sum(dim=-1)) / (
                   self.n_aperture * 4 + 1 + self.n_aperture
                   + self.n_aperture   # beam_direction
                   + self.n_aperture)   # channel heads
        # Fix entropy denominator (was double-counting freq_hop placeholder)
        entropy = (task_entropy.sum(dim=-1) + bd_entropy.sum(dim=-1)
                   + laser_entropy + emit_entropy.sum(dim=-1)
                   + fh_entropy.sum(dim=-1)
                   + chan_entropy.sum(dim=-1)) / (
                   self.n_aperture   # task_alloc (4-way simplex per aperture)
                   + self.n_aperture   # beam_direction per aperture
                   + 1   # laser_target
                   + self.n_aperture   # emission_on per aperture
                   + self.n_aperture   # freq_hop per aperture
                   + self.n_aperture)  # channel_select per aperture

        value = self.central_trunk(
            torch.cat([trunk_in, privileged], dim=-1)).squeeze(-1)
        value_local = self.local_trunk(trunk_in).squeeze(-1)

        return log_prob, value, value_local, entropy

    # ------------------------------------------------------------------
    # Deterministic / sampled action for env stepping
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action_for_env(
        self,
        obs_team: torch.Tensor,
        detect_list_team: torch.Tensor,
        privileged_team: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get per-team action slice for env.step."""
        action, log_prob, _, _ = self.forward(obs_team, detect_list_team, privileged_team)
        if deterministic:
            B = obs_team.shape[0]
            trunk_in = self._trunk_input(obs_team, detect_list_team)
            h = self.actor_trunk(trunk_in)
            raw_task = self.task_alloc_head(h).reshape(B, self.n_aperture, self.n_fn)
            alpha = F.softplus(raw_task) + self.dirichlet_min_concentration
            task_mean = alpha / alpha.sum(dim=-1, keepdim=True)
            # beam_direction deterministic = Beta mean rescaled to [-π, π]
            raw_bd = self.beam_direction_head(h).reshape(B, self.n_aperture, 2)
            bd_alpha = F.softplus(raw_bd[..., 0]) + self.beta_min_concentration
            bd_beta = F.softplus(raw_bd[..., 1]) + self.beta_min_concentration
            bd_mean_uniform = bd_alpha / (bd_alpha + bd_beta)
            bd_mean = bd_mean_uniform * 2.0 * math.pi - math.pi   # [B, n_aperture] ∈ [-π, π]
            laser_logits = self.laser_target_head(h)
            laser_argmax = laser_logits.argmax(dim=-1)
            emit_logits = self.emission_on_head(h)
            emit_round = (torch.sigmoid(emit_logits) > 0.5).float()
            # Beta mean = α/(α+β), rescale to [1, freq_hop_max]
            raw_fh = self.freq_hop_head(h).reshape(B, self.n_aperture, 2)
            fh_alpha = F.softplus(raw_fh[..., 0]) + self.beta_min_concentration
            fh_beta = F.softplus(raw_fh[..., 1]) + self.beta_min_concentration
            fh_mean_uniform = fh_alpha / (fh_alpha + fh_beta)
            fh_mean = fh_mean_uniform * (self.freq_hop_max - 1.0) + 1.0
            chan_logits = self.channel_select_head(h).reshape(B, self.n_aperture, self.n_channels)
            chan_argmax = chan_logits.argmax(dim=-1)
            action = {
                "task_alloc": task_mean,
                "beam_direction": bd_mean,
                "laser_target": laser_argmax.long(),
                "emission_on": emit_round,
                "freq_hop_rate": fh_mean,
                "channel_select": chan_argmax.long(),
            }
        return action, log_prob
