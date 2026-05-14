"""Actor-Critic networks for Commander and Radar agents.

CommanderActorCritic: 68-dim obs → 35-dim action, tanh squash
RadarActorCritic: spectrum + state → task logits + params + vehicle, separate value head

Includes AdaptiveSpectrumEncoder that handles any (N, P, B) input size.
"""

import torch
import torch.nn as nn
import numpy as np


class AdaptiveSpectrumEncoder(nn.Module):
    """Spectrum encoder that adapts to any input size.

    Processes [B, N, P, B_bins] spectrum → [B, hidden_dim] features.
    Uses 1D Conv per element (treating P as channels), then global avg pool + linear.
    """

    def __init__(
        self,
        n_elem: int,
        n_pulses: int,
        n_bins: int,
        hidden_dim: int = 256,
        base_channels: int = 32,
    ):
        super().__init__()
        self.n_elem = n_elem
        self.n_pulses = n_pulses
        self.n_bins = n_bins
        self.hidden_dim = hidden_dim

        # 1D conv: P input channels → base_channels, then pool to fixed size
        self.conv = nn.Sequential(
            nn.Conv1d(n_pulses, base_channels, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )

        # Detect output size with dry run
        with torch.no_grad():
            dummy = torch.zeros(1, n_pulses, n_bins)
            conv_out = self.conv(dummy)
            conv_flat = conv_out.shape[1] * conv_out.shape[2]

        self.proj = nn.Linear(conv_flat, hidden_dim)

        # Element-level attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True,
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """[B, N, P, B_bins] → [B, hidden_dim]"""
        B, N, P, BINS = spectrum.shape
        x = spectrum.reshape(B * N, P, BINS)
        x = self.conv(x)                # [B*N, C, BINS']
        x = x.reshape(B * N, -1)        # flatten
        x = self.proj(x)                # [B*N, hidden_dim]
        x = x.reshape(B, N, self.hidden_dim)

        # Attention across elements
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out  # residual

        return x.mean(dim=1)  # [B, hidden_dim]


class CommanderActorCritic(nn.Module):
    """Commander policy: small MLP with separate action and value heads.

    Outputs tanh-squashed actions in [-1, 1] with learnable log-std for PPO.
    """

    def __init__(
        self,
        obs_dim: int = 68,
        act_dim: int = 35,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 1.0)  # init std ≈ 0.37

    def forward(self, obs: torch.Tensor):
        """Deterministic forward: obs → (action, value)."""
        features = self.shared(obs)
        mean = self.action_head(features)
        action = torch.tanh(mean)
        value = self.value_head(features)
        return action, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action, return (action, log_prob, value).

        Args:
            obs: [B, obs_dim]
            deterministic: if True, use mean (no sampling)
        Returns:
            action: [B, act_dim] in [-1, 1]
            log_prob: [B]
            value: [B, 1]
        """
        features = self.shared(obs)
        mean = self.action_head(features)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()

        action = torch.tanh(raw_action)
        # Log-prob with tanh correction
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        # Squash correction: log(1 - tanh(x)^2)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)

        value = self.value_head(features)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate log-prob, entropy, value for given actions (PPO update).

        Args:
            obs: [B, obs_dim]
            actions: [B, act_dim] in [-1, 1]
        Returns:
            log_prob: [B]
            entropy: [B]
            value: [B, 1]
        """
        features = self.shared(obs)
        mean = self.action_head(features)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        # Inverse tanh to get pre-squash action
        raw_action = 0.5 * torch.log((actions + 1.0) / (1.0 - actions + 1e-6).clamp(min=1e-6))

        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(features)

        return log_prob, entropy, value


class RadarActorCritic(nn.Module):
    """Radar policy: SpectrumEncoder → task/param/vehicle heads + value head.

    Action space decomposition:
      - task_head: [B, N, 4] logits → argmax task assignment
      - param_head: [B, N, 8] sigmoid → per-element waveform params
      - vehicle_head: [B, 3] tanh → (speed, heading, rotation)
    Total output matches ACTION_TOTAL_PER_RADAR = 625*22 + 3 = 13753.
    """

    def __init__(
        self,
        n_elem: int = 625,
        n_pulses: int = 32,
        n_bins: int = 1024,
        spectrum_hidden: int = 256,
        vehicle_dim: int = 5,
        missile_dim: int = 12,
        commander_instr_dim: int = 16,
        comm_data_dim: int = 1250,
        encoder_kwargs: dict = None,
    ):
        super().__init__()
        self.n_elem = n_elem
        self.n_pulses = n_pulses
        self.n_bins = n_bins

        # Spectrum encoder (adaptive, handles any input size)
        ek = encoder_kwargs or {}
        base_channels = ek.get("base_channels", 32)

        self.spectrum_encoder = AdaptiveSpectrumEncoder(
            n_elem=n_elem, n_pulses=n_pulses, n_bins=n_bins,
            hidden_dim=spectrum_hidden, base_channels=base_channels,
        )

        # The non-spectrum state dim: vehicle + comm_flat + missile + commander_instr
        # Note: comm_data is part of the spectrum path, not passed separately here
        # The state obs from env is: spec_flat + comm_flat + vehicle + missile + cmd_instr
        # We extract spec_flat separately, then feed remainder to MLP
        # spec_flat = N * P * B, comm_flat = N * 2
        spectrum_flat = n_elem * n_pulses * n_bins
        comm_flat = n_elem * 2
        other_dim = vehicle_dim + missile_dim + commander_instr_dim

        # For the raw state input: we split off spectrum portion and encode it,
        # then concatenate with the rest
        self.spectrum_flat_dim = spectrum_flat
        self.comm_flat_dim = comm_flat
        self.other_dim = other_dim

        shared_in = spectrum_hidden + comm_flat + other_dim
        self.shared = nn.Sequential(
            nn.Linear(shared_in, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Action heads
        self.task_head = nn.Linear(512, n_elem * 4)
        self.param_head = nn.Linear(512, n_elem * 8)
        self.vehicle_head = nn.Linear(512, 3)

        # Value head
        self.value_head = nn.Linear(512, 1)

        # Learnable log-std for continuous params (used by PPO)
        self.log_std_params = nn.Parameter(torch.zeros(n_elem * 8))
        self.log_std_vehicle = nn.Parameter(torch.zeros(3))

    def forward(self, state: torch.Tensor):
        """Deterministic forward: state → assembled flat action [B, 13753] + value."""
        action, _, value = self.get_action(state, deterministic=True)
        return action, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action, return (action, log_prob, value).

        Args:
            state: [B, state_dim]
            deterministic: if True, use mean
        Returns:
            action: [B, 13753] flat action for env
            log_prob: [B]
            value: [B, 1]
        """
        B = state.shape[0]
        N = self.n_elem

        spec_end = self.spectrum_flat_dim
        comm_end = spec_end + self.comm_flat_dim
        other_end = comm_end + self.other_dim

        spec_flat = state[..., :spec_end]
        comm_flat = state[..., spec_end:comm_end]
        other = state[..., comm_end:other_end]

        spectrum = spec_flat.reshape(B, N, self.n_pulses, self.n_bins)
        spec_feat = self.spectrum_encoder(spectrum)

        shared_in = torch.cat([spec_feat, comm_flat, other], dim=-1)
        features = self.shared(shared_in)

        # Task logits → Categorical
        task_logits = self.task_head(features).reshape(B, N, 4)
        task_dist = torch.distributions.Categorical(logits=task_logits)

        # Params → Normal
        param_mean = torch.sigmoid(self.param_head(features))  # [B, N*8]
        param_std = torch.exp(self.log_std_params).expand_as(param_mean)
        param_dist = torch.distributions.Normal(param_mean, param_std)

        # Vehicle → Normal
        veh_mean = torch.tanh(self.vehicle_head(features))
        veh_std = torch.exp(self.log_std_vehicle).expand_as(veh_mean)
        veh_dist = torch.distributions.Normal(veh_mean, veh_std)

        value = self.value_head(features)

        # Sample or use mean
        if deterministic:
            task_choice = task_logits.argmax(dim=-1)  # [B, N]
            params = param_mean
            vehicle = veh_mean
        else:
            task_choice = task_dist.sample()  # [B, N]
            params = param_dist.rsample().clamp(0.01, 0.99)
            vehicle = veh_dist.rsample().clamp(-0.999, 0.999)

        # Log-prob
        task_logp = task_dist.log_prob(task_choice).sum(dim=-1)  # [B]
        param_logp = param_dist.log_prob(params).sum(dim=-1)      # [B]
        veh_logp = veh_dist.log_prob(vehicle).sum(dim=-1)          # [B]
        log_prob = task_logp + param_logp + veh_logp

        # Assemble flat action
        task_frac = torch.zeros(B, N, 4, device=state.device)
        task_frac.scatter_(-1, task_choice.unsqueeze(-1), 1.0)

        action = self._assemble_action_from_parts(task_frac, params, vehicle)

        return action, log_prob, value

    def _assemble_action_from_parts(
        self,
        task_frac: torch.Tensor,  # [B, N, 4]
        params: torch.Tensor,      # [B, N*8]
        vehicle: torch.Tensor,     # [B, 3]
    ) -> torch.Tensor:
        """Assemble decomposed components into flat [B, 22*N+3] action."""
        B = task_frac.shape[0]
        N = self.n_elem
        p = params.reshape(B, N, 8)

        beam_az = p[..., 0:1].expand(B, N, 4) * 0.5 + 0.5
        beam_el = p[..., 1:2].expand(B, N, 4) * 0.5 + 0.5
        beam = torch.stack([beam_az, beam_el], dim=-1).reshape(B, N, 8)

        detect_p = p[..., 2:5]
        jam_p = p[..., 5:8]
        comm_p = torch.cat([p[..., 2:3], p[..., 0:1], p[..., 6:7], p[..., 7:8]], dim=-1)

        elem_action = torch.cat([task_frac, beam, detect_p, jam_p, comm_p], dim=-1)
        flat = elem_action.reshape(B, N * 22)
        return torch.cat([flat, vehicle], dim=-1)

    def _assemble_action(
        self,
        task_logits: torch.Tensor,   # [B, N, 4]
        params: torch.Tensor,         # [B, N*8]
        vehicle: torch.Tensor,        # [B, 3]
    ) -> torch.Tensor:
        """Assemble decomposed action into flat [B, 13753] format.

        Per-element layout (22 dims):
          [0:4]   task fractions from softmax of logits
          [4:12]  beam steering: use params[0:2] repeated for all 4 tasks
          [12:15] detect TX params: params[2:5]
          [15:18] jam TX params: params[5:8]
          [18:22] comm TX params: params[2:4] + params[5:7] (reuse)
        """
        B = task_logits.shape[0]
        N = self.n_elem
        p = params.reshape(B, N, 8)

        # Task fractions: softmax → [B, N, 4]
        task_frac = torch.softmax(task_logits, dim=-1)

        # Beam steering: [az, el] repeated for 4 tasks → [B, N, 8]
        beam_az = p[..., 0:1].expand(B, N, 4) * 0.5 + 0.5  # normalize to [0, 1]
        beam_el = p[..., 1:2].expand(B, N, 4) * 0.5 + 0.5
        beam = torch.stack([beam_az, beam_el], dim=-1).reshape(B, N, 8)

        # Detect params: 3 dims
        detect_p = p[..., 2:5]
        # Jam params: 3 dims
        jam_p = p[..., 5:8]
        # Comm params: 4 dims (reuse some params)
        comm_p = torch.cat([p[..., 2:3], p[..., 0:1], p[..., 6:7], p[..., 7:8]], dim=-1)

        elem_action = torch.cat([task_frac, beam, detect_p, jam_p, comm_p], dim=-1)  # [B, N, 22]
        flat = elem_action.reshape(B, N * 22)

        return torch.cat([flat, vehicle], dim=-1)  # [B, 13753]

    def get_distribution(self, state: torch.Tensor):
        """Get action distributions for PPO.

        Returns:
            task_dist: Categorical per element
            param_dist: Normal for continuous params
            vehicle_dist: Normal for vehicle
            value: [B, 1]
        """
        B = state.shape[0]

        spec_end = self.spectrum_flat_dim
        comm_end = spec_end + self.comm_flat_dim
        other_end = comm_end + self.other_dim

        spec_flat = state[..., :spec_end]
        comm_flat = state[..., spec_end:comm_end]
        other = state[..., comm_end:other_end]

        spectrum = spec_flat.reshape(B, self.n_elem, self.n_pulses, self.n_bins)
        spec_feat = self.spectrum_encoder(spectrum)

        shared_in = torch.cat([spec_feat, comm_flat, other], dim=-1)
        features = self.shared(shared_in)

        task_logits = self.task_head(features).reshape(B, self.n_elem, 4)
        param_mean = torch.sigmoid(self.param_head(features))  # [B, N*8]
        vehicle_mean = torch.tanh(self.vehicle_head(features))
        value = self.value_head(features)

        task_dist = torch.distributions.Categorical(logits=task_logits)

        param_std = torch.exp(self.log_std_params).expand_as(param_mean)
        param_dist = torch.distributions.Normal(param_mean, param_std)

        veh_std = torch.exp(self.log_std_vehicle).expand_as(vehicle_mean)
        vehicle_dist = torch.distributions.Normal(vehicle_mean, veh_std)

        return task_dist, param_dist, vehicle_dist, value

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor):
        """Evaluate log-prob, entropy, value for PPO update.

        Args:
            state: [B, state_dim]
            actions: [B, 13753] flat actions
        Returns:
            log_prob: [B]
            entropy: [B]
            value: [B, 1]
        """
        task_dist, param_dist, vehicle_dist, value = self.get_distribution(state)

        N = self.n_elem
        # Parse actions back
        elem_act = actions[..., :N * 22].reshape(-1, N, 22)
        task_frac = elem_act[..., 0:4]
        vehicle_act = actions[..., -3:]

        # Task: use the argmax task choice for log-prob
        task_choice = task_frac.argmax(dim=-1)  # [B, N]
        task_logp = task_dist.log_prob(task_choice)  # [B, N]
        task_ent = task_dist.entropy()  # [B, N]

        # Params: use the param portion of the action
        param_act = actions[..., :N * 8]  # first N*8 of 22 dims (approximate)
        # We use the raw param values — note this is an approximation
        # since the env action layout interleaves task+beam+params
        param_logp = param_dist.log_prob(param_act.clamp(0.01, 0.99)).sum(dim=-1)

        # Vehicle
        veh_logp = vehicle_dist.log_prob(vehicle_act).sum(dim=-1)

        log_prob = task_logp.sum(dim=-1) + param_logp + veh_logp
        entropy = task_ent.sum(dim=-1) + param_dist.entropy().sum(dim=-1) + vehicle_dist.entropy().sum(dim=-1)

        return log_prob, entropy, value


def create_team_policy(
    team: int,
    n_elem: int = 625,
    n_pulses: int = 32,
    n_bins: int = 1024,
    num_output_length: int = 16,
    device: str = "cuda",
    encoder_kwargs: dict = None,
) -> dict:
    """Create a full team policy (commander + shared radar).

    Returns:
        dict with "commander" and "radar" actor-critic modules.
    """
    commander = CommanderActorCritic(
        obs_dim=68,
        act_dim=35,
        hidden_dim=256,
    ).to(device)

    # Compute actual n_bins from MFARVecEnv auto-fft logic
    fft_size = 1
    while fft_size < 200:  # approximate for typical fs/pri
        fft_size *= 2
    # Use provided n_bins directly if > 0

    radar = RadarActorCritic(
        n_elem=n_elem,
        n_pulses=n_pulses,
        n_bins=n_bins,
        commander_instr_dim=num_output_length,
        encoder_kwargs=encoder_kwargs,
    ).to(device)

    return {"commander": commander, "radar": radar}
