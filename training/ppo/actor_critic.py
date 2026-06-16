"""Actor-Critic networks for Commander and Radar agents.

CommanderActorCritic: 76-dim obs → 35-dim action, tanh squash
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

    CTDE (Centralized Training, Decentralized Execution):
    - value_head: deployment critic, takes shared features only
    - privileged_value_head: training critic, sees shared features + privileged_info
      (beam_hit_time, etc.) — gives more accurate GAE advantages during training.
      At deployment, only value_head is used (privileged_info unavailable).
    """

    def __init__(
        self,
        obs_dim: int = 76,
        act_dim: int = 35,
        hidden_dim: int = 256,
        value_init: float = 2.8,
        privileged_dim: int = 0,
        hybrid_fire: bool = False,
        decouple_value: bool = False,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.privileged_dim = privileged_dim
        # When True, action dim 0 is a discrete fire trigger modeled by a Bernoulli
        # head (its own log-prob/entropy), while dims 1: stay continuous tanh-Gaussian.
        # This gives the fire bit a real policy-gradient signal instead of being a
        # threshold applied outside the distribution (which carries no log-prob).
        self.hybrid_fire = hybrid_fire
        # When True, the value head gets its OWN trunk (value_shared) so the value-loss
        # gradient no longer churns the policy trunk — the aim/residual head can then
        # converge its mean to ~0 (sub-meter aim) instead of being perturbed to ~1-2m.
        self.decouple_value = decouple_value
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if decouple_value:
            self.value_shared = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        self.action_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 1.0)  # init std ≈ 0.37

        # Optional privileged critic (CTDE) — takes shared features + privileged info
        if privileged_dim > 0:
            self.privileged_value_head = nn.Sequential(
                nn.Linear(hidden_dim + privileged_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.privileged_value_head = None

        # Orthogonal init — PyTorch default kaiming_uniform is suboptimal for PPO
        # (ICLR Blog Track 2022 "37 Implementation Details of PPO")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Value head: smaller gain, bias to expected return mean (skips cold-start
        # where critic outputs ~0 while returns are ~2.8 → advantages all-positive
        # → PPO has no useful gradient direction)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, value_init)
        if self.privileged_value_head is not None:
            # Init privileged critic's output bias to same scale
            last_layer = self.privileged_value_head[-1]
            nn.init.orthogonal_(last_layer.weight, gain=1.0)
            nn.init.constant_(last_layer.bias, value_init)
        # Zero-init the AIM output (dims 1:) for residual aiming: the policy starts with
        # residual ≈ 0 (aim exactly on the sensing anchor) and only learns tiny corrections.
        if self.hybrid_fire:
            with torch.no_grad():
                self.action_head.weight[1:].mul_(0.01)
                self.action_head.bias[1:].zero_()

    def _vfeat(self, obs, features):
        """Features for the value head: a decoupled trunk if enabled (so value-loss
        gradient doesn't churn the policy trunk), else the shared policy features."""
        return self.value_shared(obs) if self.decouple_value else features

    def forward(self, obs: torch.Tensor, privileged_info: torch.Tensor = None):
        """Deterministic forward: obs → (action, value)."""
        features = self.shared(obs)
        mean = self.action_head(features)
        action = torch.tanh(mean)
        value = self.value_head(self._vfeat(obs, features))
        return action, value

    def _compute_privileged_value(self, features: torch.Tensor, privileged_info: torch.Tensor):
        """Compute privileged critic value. Returns regular value if no privileged head."""
        if self.privileged_value_head is None or privileged_info is None:
            return self.value_head(features)
        priv_input = torch.cat([features, privileged_info], dim=-1)
        return self.privileged_value_head(priv_input)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False,
                   privileged_info: torch.Tensor = None):
        """Sample action, return (action, log_prob, value, privileged_value).

        Args:
            obs: [B, obs_dim]
            deterministic: if True, use mean (no sampling)
            privileged_info: [B, privileged_dim] or None — used for privileged_value
                             (training-time centralized critic).
        Returns:
            action: [B, act_dim] in [-1, 1]
            log_prob: [B]
            value: [B, 1] (deployment critic)
            privileged_value: [B, 1] (training critic; = value if no privileged head)
        """
        features = self.shared(obs)
        mean = self.action_head(features)

        if self.hybrid_fire:
            fire_logit = mean[:, 0:1]
            aim_mean = mean[:, 1:]
            aim_std = torch.exp(self.log_std[1:]).expand_as(aim_mean)
            aim_dist = torch.distributions.Normal(aim_mean, aim_std)
            fire_dist = torch.distributions.Bernoulli(logits=fire_logit)

            if deterministic:
                fire = (torch.sigmoid(fire_logit) > 0.5).float()
                aim_raw = aim_mean
            else:
                fire = fire_dist.sample()
                aim_raw = aim_dist.rsample()

            aim = torch.tanh(aim_raw)
            # Map fire {0,1} → {-1,+1} so the env's (action[0] > 0.5) decode fires iff fire=1
            action = torch.cat([fire * 2.0 - 1.0, aim], dim=-1)
            log_prob = fire_dist.log_prob(fire).sum(dim=-1)
            log_prob += aim_dist.log_prob(aim_raw).sum(dim=-1)
            log_prob -= torch.log(1.0 - aim.pow(2) + 1e-6).sum(dim=-1)
        else:
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

        vfeat = self._vfeat(obs, features)
        value = self.value_head(vfeat)
        privileged_value = self._compute_privileged_value(vfeat, privileged_info)
        return action, log_prob, value, privileged_value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         privileged_info: torch.Tensor = None):
        """Evaluate log-prob, entropy, value for given actions (PPO update).

        Args:
            obs: [B, obs_dim]
            actions: [B, act_dim] in [-1, 1]
            privileged_info: [B, privileged_dim] or None
        Returns:
            log_prob: [B]
            entropy: [B]
            value: [B, 1] (deployment critic)
            privileged_value: [B, 1] (training critic)
        """
        features = self.shared(obs)
        mean = self.action_head(features)

        if self.hybrid_fire:
            fire_logit = mean[:, 0:1]
            aim_mean = mean[:, 1:]
            aim_std = torch.exp(self.log_std[1:]).expand_as(aim_mean)
            aim_dist = torch.distributions.Normal(aim_mean, aim_std)
            fire_dist = torch.distributions.Bernoulli(logits=fire_logit)

            # Recover the Bernoulli sample from the stored action (dim 0 ∈ {-1,+1})
            fire = (actions[:, 0:1] > 0.0).float()
            aim = actions[:, 1:]
            aim_raw = 0.5 * torch.log((aim + 1.0) / (1.0 - aim + 1e-6).clamp(min=1e-6))

            log_prob = fire_dist.log_prob(fire).sum(dim=-1)
            log_prob += aim_dist.log_prob(aim_raw).sum(dim=-1)
            log_prob -= torch.log(1.0 - aim.pow(2) + 1e-6).sum(dim=-1)
            entropy = fire_dist.entropy().sum(dim=-1) + aim_dist.entropy().sum(dim=-1)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
            dist = torch.distributions.Normal(mean, std)

            # Inverse tanh to get pre-squash action
            raw_action = 0.5 * torch.log((actions + 1.0) / (1.0 - actions + 1e-6).clamp(min=1e-6))

            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            log_prob -= torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        vfeat = self._vfeat(obs, features)
        value = self.value_head(vfeat)
        privileged_value = self._compute_privileged_value(vfeat, privileged_info)

        return log_prob, entropy, value, privileged_value


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

    def forward(self, state: torch.Tensor, privileged_info: torch.Tensor = None):
        """Deterministic forward: state → assembled flat action [B, 13753] + value."""
        action, _, value, _ = self.get_action(state, deterministic=True)
        return action, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False,
                   privileged_info: torch.Tensor = None):
        """Sample action, return (action, log_prob, value, privileged_value).

        Args:
            state: [B, state_dim]
            deterministic: if True, use mean
        Returns:
            action: [B, 13753] flat action for env
            log_prob: [B]
            value: [B, 1]
            privileged_value: [B, 1] (same as value for base class — no asymmetric critic)
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

        return action, log_prob, value, value

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

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor,
                         privileged_info: torch.Tensor = None):
        """Evaluate log-prob, entropy, value for PPO update.

        Args:
            state: [B, state_dim]
            actions: [B, 13753] flat actions
        Returns:
            log_prob: [B]
            entropy: [B]
            value: [B, 1]
            privileged_value: [B, 1] (same as value for base class)
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

        return log_prob, entropy, value, value


class SubArrayRadarActorCritic(nn.Module):
    """Sub-array decomposed radar policy for large arrays (25x25+).

    Groups n_elem elements into n_sub sub-arrays (sub_size x sub_size blocks).
    Internally produces per-sub-array actions (K*22+3 dims), then broadcasts
    to per-element actions (n_elem*22+3 dims). External interface identical
    to RadarActorCritic — drop-in replacement requiring no other code changes.

    State layout (from MFARVecEnv._assemble_state):
      spec_flat   [n_elem * P * B_bins]
      comm_flat   [n_elem * 2]
      recon_flat  [n_elem * 4]
      vehicle     [5]
      missile     [12]
      cmd_instr   [num_output_length]
    """

    def __init__(
        self,
        n_elem: int = 625,
        n_pulses: int = 32,
        n_bins: int = 1024,
        sub_array_size: int = 5,
        spectrum_hidden: int = 256,
        vehicle_dim: int = 5,
        missile_dim: int = 12,
        commander_instr_dim: int = 16,
    ):
        super().__init__()
        self.n_elem = n_elem
        self.n_pulses = n_pulses
        self.n_bins = n_bins
        self.sub_size = sub_array_size
        self.elem_per_sub = sub_array_size * sub_array_size
        self.n_sub = n_elem // self.elem_per_sub
        assert n_elem % self.elem_per_sub == 0, (
            f"n_elem={n_elem} must be divisible by sub_array_size^2={self.elem_per_sub}"
        )

        # State layout dimensions
        self.spectrum_flat_dim = n_elem * n_pulses * n_bins
        self.comm_flat_dim = n_elem * 2
        self.recon_flat_dim = n_elem * 4
        self.other_dim = vehicle_dim + missile_dim + commander_instr_dim

        # Sub-array spectrum encoder: Conv1D per sub-array
        self.conv = nn.Sequential(
            nn.Conv1d(n_pulses, 32, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_pulses, n_bins)
            conv_out = self.conv(dummy)
            conv_flat = conv_out.shape[1] * conv_out.shape[2]
        self.proj = nn.Linear(conv_flat, spectrum_hidden)

        # Cross sub-array attention
        self.attention = nn.MultiheadAttention(
            embed_dim=spectrum_hidden, num_heads=4, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(spectrum_hidden)

        # Per-sub-array feature MLP (spectrum_hidden + comm(2) + recon(4))
        per_sub_in = spectrum_hidden + 2 + 4
        self.shared = nn.Sequential(
            nn.Linear(per_sub_in, 256),
            nn.ReLU(),
        )

        # Action heads (applied per sub-array)
        self.task_head = nn.Linear(256, 4)
        self.param_head = nn.Linear(256, 8)

        # Global heads (from pooled sub-array features)
        self.vehicle_head = nn.Linear(spectrum_hidden + self.other_dim, 3)
        self.value_head = nn.Linear(spectrum_hidden + self.other_dim, 1)

        # Log-std for continuous params (shared across all sub-arrays)
        self.log_std_params = nn.Parameter(torch.zeros(8))
        self.log_std_vehicle = nn.Parameter(torch.zeros(3))

        # --- Privileged critic: sees full element-level spectrum (no SADP pooling)
        # plus privileged info only available during centralized training.
        # n_teams is 2 for the standard adversarial setting.
        _n_teams = 2
        self.privileged_spec_encoder = AdaptiveSpectrumEncoder(
            n_elem=n_elem, n_pulses=n_pulses, n_bins=n_bins,
            hidden_dim=256,
        )
        # comm pooled (mean+max over elements → 2+2=4)
        # recon pooled (mean+max over elements → 4+4=8)
        # + other (vehicle+missile+cmd) + privileged extra
        priv_extra_dim = _n_teams * 4 + _n_teams * 3 + 2  # task_fingerprint + intercept + target
        priv_in = 256 + 4 + 8 + self.other_dim + priv_extra_dim
        self.privileged_head = nn.Sequential(
            nn.Linear(priv_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _parse_state(self, state: torch.Tensor):
        """Parse flat state into structured components.

        Returns:
            spectrum: [B, n_elem, P, B_bins]
            comm: [B, n_elem, 2]
            recon: [B, n_elem, 4]
            other: [B, other_dim] (vehicle + missile + cmd_instr)
        """
        B = state.shape[0]
        off = 0

        spec_flat = state[..., off:off + self.spectrum_flat_dim]
        off += self.spectrum_flat_dim
        spectrum = spec_flat.reshape(B, self.n_elem, self.n_pulses, self.n_bins)

        comm_flat = state[..., off:off + self.comm_flat_dim]
        off += self.comm_flat_dim
        comm = comm_flat.reshape(B, self.n_elem, 2)

        recon_flat = state[..., off:off + self.recon_flat_dim]
        off += self.recon_flat_dim
        recon = recon_flat.reshape(B, self.n_elem, 4)

        other = state[..., off:off + self.other_dim]

        return spectrum, comm, recon, other

    def _encode_sub_arrays(self, spectrum, comm, recon):
        """Encode spectrum/comm/recon into per-sub-array features.

        Args:
            spectrum: [B, n_elem, P, B_bins]
            comm: [B, n_elem, 2]
            recon: [B, n_elem, 4]
        Returns:
            sub_features: [B, n_sub, spectrum_hidden + 6]
        """
        B = spectrum.shape[0]
        K = self.n_sub
        S2 = self.elem_per_sub
        P, BINS = self.n_pulses, self.n_bins

        # Reshape to sub-arrays and average pool
        sub_spec = spectrum.reshape(B, K, S2, P, BINS).mean(dim=2)  # [B, K, P, BINS]
        sub_comm = comm.reshape(B, K, S2, 2).mean(dim=2)            # [B, K, 2]
        sub_recon = recon.reshape(B, K, S2, 4).mean(dim=2)          # [B, K, 4]

        # Conv1D per sub-array
        x = sub_spec.reshape(B * K, P, BINS)
        x = self.conv(x)
        x = x.reshape(B * K, -1)
        x = self.proj(x)  # [B*K, hidden]
        x = x.reshape(B, K, -1)

        # Cross sub-array attention with residual + norm
        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)  # [B, K, hidden]

        # Concatenate comm + recon per sub-array
        sub_features = torch.cat([x, sub_comm, sub_recon], dim=-1)  # [B, K, hidden+6]

        return sub_features, x  # sub_features, pooled_spectrum

    def _get_distributions(self, state: torch.Tensor, privileged_info: torch.Tensor = None):
        """Build action distributions from state.

        Returns:
            task_dist, param_dist, vehicle_dist, value, privileged_value
        """
        B = state.shape[0]
        K = self.n_sub

        spectrum, comm, recon, other = self._parse_state(state)
        sub_feat, pooled_spec = self._encode_sub_arrays(spectrum, comm, recon)

        # Global features for vehicle/value heads
        global_spec = pooled_spec.mean(dim=1)  # [B, hidden]
        global_feat = torch.cat([global_spec, other], dim=-1)  # [B, hidden+other]

        # Per-sub-array action features
        shared_feat = self.shared(sub_feat)  # [B, K, 256]

        # Task distribution per sub-array
        task_logits = self.task_head(shared_feat)  # [B, K, 4]
        task_dist = torch.distributions.Categorical(logits=task_logits)

        # Param distribution per sub-array (shared log_std)
        param_mean = torch.sigmoid(self.param_head(shared_feat))  # [B, K, 8]
        param_std = torch.exp(self.log_std_params)  # [8]
        param_dist = torch.distributions.Normal(param_mean, param_std)

        # Vehicle distribution
        veh_mean = torch.tanh(self.vehicle_head(global_feat))  # [B, 3]
        veh_std = torch.exp(self.log_std_vehicle)  # [3]
        vehicle_dist = torch.distributions.Normal(veh_mean, veh_std)

        value = self.value_head(global_feat)  # [B, 1]

        # Privileged critic value: full element-level encoding + privileged info
        privileged_value = self._compute_privileged_value(
            spectrum, comm, recon, other, privileged_info,
        )

        return task_dist, param_dist, vehicle_dist, value, privileged_value

    def _compute_privileged_value(self, spectrum, comm, recon, other, privileged_info=None):
        """Compute value from privileged (full-resolution) critic.

        The privileged critic sees the full element-level spectrum (not SADP-
        pooled) plus optional privileged info (task_fingerprint, intercept,
        target positions) that is only available during centralized training.

        When privileged_info is None (deployment), falls back to a degraded
        estimate using only full-resolution spectrum + comm + recon.
        """
        B = spectrum.shape[0]

        # Full element-level spectrum encoding (no SADP pooling)
        priv_spec = self.privileged_spec_encoder(spectrum)  # [B, 256]

        # Pool comm and recon across elements
        comm_pooled = torch.cat([
            comm.mean(dim=1),
            comm.max(dim=1).values,
        ], dim=-1)  # [B, 4]
        recon_pooled = torch.cat([
            recon.mean(dim=1),
            recon.max(dim=1).values,
        ], dim=-1)  # [B, 8]

        priv_input = torch.cat([priv_spec, comm_pooled, recon_pooled, other], dim=-1)

        if privileged_info is not None:
            priv_input = torch.cat([priv_input, privileged_info], dim=-1)
        else:
            # Zero-pad for missing privileged info
            pad_dim = self.privileged_head[0].in_features - priv_input.shape[-1]
            priv_input = torch.cat([
                priv_input,
                torch.zeros(B, pad_dim, device=spectrum.device),
            ], dim=-1)

        return self.privileged_head(priv_input)  # [B, 1]

    def _expand_to_elements(
        self,
        task_frac: torch.Tensor,  # [B, K, 4]
        params: torch.Tensor,      # [B, K, 8]
        vehicle: torch.Tensor,     # [B, 3]
    ) -> torch.Tensor:
        """Broadcast sub-array actions to element-level flat action [B, 13753]."""
        B = task_frac.shape[0]
        K = self.n_sub
        S2 = self.elem_per_sub
        N = self.n_elem

        p = params  # [B, K, 8]

        # Beam steering: same for all tasks per sub-array
        beam_az = p[..., 0:1].expand(B, K, 4) * 0.5 + 0.5
        beam_el = p[..., 1:2].expand(B, K, 4) * 0.5 + 0.5
        beam = torch.stack([beam_az, beam_el], dim=-1).reshape(B, K, 8)

        detect_p = p[..., 2:5]
        jam_p = p[..., 5:8]
        comm_p = torch.cat([p[..., 2:3], p[..., 0:1], p[..., 6:7], p[..., 7:8]], dim=-1)

        sub_action = torch.cat([task_frac, beam, detect_p, jam_p, comm_p], dim=-1)  # [B, K, 22]

        # Broadcast: [B, K, 22] → [B, K, S2, 22] → [B, N, 22]
        elem_action = sub_action.unsqueeze(2).expand(B, K, S2, 22).reshape(B, N, 22)

        flat = elem_action.reshape(B, N * 22)
        return torch.cat([flat, vehicle], dim=-1)

    def _extract_sub_from_elem(self, actions: torch.Tensor):
        """Extract sub-array actions from element-level flat action [B, 13753].

        Since all elements in a sub-array share the same action (broadcast),
        we take the first element of each sub-array.
        """
        B = actions.shape[0]
        K = self.n_sub
        S2 = self.elem_per_sub
        N = self.n_elem

        elem_act = actions[..., :N * 22].reshape(B, N, 22)
        # Take first element of each sub-array
        indices = torch.arange(K, device=actions.device) * S2
        sub_act = elem_act[:, indices, :]  # [B, K, 22]

        task_frac = sub_act[..., 0:4]  # [B, K, 4]
        vehicle = actions[..., -3:]

        # Reconstruct params from the action layout
        # Layout per element: [task(4), beam(8), detect(3), jam(3), comm(4)]
        # Beam is interleaved: [az0, el0, az1, el1, az2, el2, az3, el3]
        # All 4 beam pairs are identical (same az/el for all tasks).
        # Beam stored as: p[0]*0.5+0.5 (az), p[1]*0.5+0.5 (el).
        # Invert: p[0] = (az - 0.5) * 2
        params = torch.cat([
            (sub_act[..., 4:5] - 0.5) * 2,   # beam_az0 → param[0]
            (sub_act[..., 5:6] - 0.5) * 2,   # beam_el0 → param[1]
            sub_act[..., 12:15],               # detect params → param[2:5]
            sub_act[..., 15:18],               # jam params → param[5:8]
        ], dim=-1)  # [B, K, 8]

        return task_frac, params, vehicle

    def forward(self, state: torch.Tensor, privileged_info: torch.Tensor = None):
        """Deterministic forward: state → (action [B, 13753], value, privileged_value)."""
        action, _, value, privileged_value = self.get_action(
            state, deterministic=True, privileged_info=privileged_info,
        )
        return action, value, privileged_value

    def get_action(self, state: torch.Tensor, deterministic: bool = False,
                   privileged_info: torch.Tensor = None):
        """Sample action, return (action, log_prob, value, privileged_value).

        Returns:
            action: [B, n_elem*22+3] flat action for env (drop-in compatible)
            log_prob: [B]
            value: [B, 1]
            privileged_value: [B, 1]
        """
        B = state.shape[0]
        K = self.n_sub

        task_dist, param_dist, vehicle_dist, value, privileged_value = \
            self._get_distributions(state, privileged_info)

        if deterministic:
            task_choice = task_dist.logits.argmax(dim=-1)  # [B, K]
            params = param_dist.mean  # [B, K, 8]
            vehicle = vehicle_dist.mean  # [B, 3]
        else:
            task_choice = task_dist.sample()  # [B, K]
            params = param_dist.rsample().clamp(0.01, 0.99)  # [B, K, 8]
            vehicle = vehicle_dist.rsample().clamp(-0.999, 0.999)  # [B, 3]

        # Log-prob (per sub-array, not per element)
        task_logp = task_dist.log_prob(task_choice).sum(dim=-1)  # [B]
        param_logp = param_dist.log_prob(params).sum(-1).sum(-1)  # [B]
        veh_logp = vehicle_dist.log_prob(vehicle).sum(dim=-1)     # [B]
        log_prob = task_logp + param_logp + veh_logp

        # One-hot task assignment
        task_frac = torch.zeros(B, K, 4, device=state.device)
        task_frac.scatter_(-1, task_choice.unsqueeze(-1), 1.0)

        # Expand to element-level action
        action = self._expand_to_elements(task_frac, params, vehicle)

        return action, log_prob, value, privileged_value

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor,
                         privileged_info: torch.Tensor = None):
        """Evaluate log-prob, entropy, value for given actions (PPO update).

        Args:
            state: [B, state_dim]
            actions: [B, n_elem*22+3] flat actions (element-level)
        Returns:
            log_prob: [B]
            entropy: [B]
            value: [B, 1]
            privileged_value: [B, 1]
        """
        task_dist, param_dist, vehicle_dist, value, privileged_value = \
            self._get_distributions(state, privileged_info)

        # Extract sub-array actions from element-level action
        task_frac, params, vehicle = self._extract_sub_from_elem(actions)

        task_choice = task_frac.argmax(dim=-1)  # [B, K]

        task_logp = task_dist.log_prob(task_choice)  # [B, K]
        task_ent = task_dist.entropy()               # [B, K]

        param_logp = param_dist.log_prob(params.clamp(0.01, 0.99)).sum(-1).sum(-1)  # [B]
        veh_logp = vehicle_dist.log_prob(vehicle.clamp(-0.999, 0.999)).sum(dim=-1)  # [B]

        log_prob = task_logp.sum(dim=-1) + param_logp + veh_logp
        entropy = (task_ent.sum(dim=-1)
                   + param_dist.entropy().sum(-1).sum(-1)
                   + vehicle_dist.entropy().sum(dim=-1))

        return log_prob, entropy, value, privileged_value

    def get_distribution(self, state: torch.Tensor, privileged_info: torch.Tensor = None):
        """Get action distributions for PPO.

        Returns:
            task_dist, param_dist, vehicle_dist, value, privileged_value
        """
        return self._get_distributions(state, privileged_info)


def create_team_policy(
    team: int,
    n_elem: int = 625,
    n_pulses: int = 32,
    n_bins: int = 1024,
    num_output_length: int = 16,
    device: str = "cuda",
    encoder_kwargs: dict = None,
    sub_array_size: int = 0,
    commander_privileged_dim: int = 0,
) -> dict:
    """Create a full team policy (commander + shared radar).

    Args:
        sub_array_size: If > 0, use SubArrayRadarActorCritic with this sub-array
                       block size (e.g., 5 for 5×5 sub-array blocks in a 25×25 array).
        commander_privileged_dim: If > 0, add CTDE privileged critic to commander.
    Returns:
        dict with "commander" and "radar" actor-critic modules.
    """
    commander = CommanderActorCritic(
        obs_dim=76,
        act_dim=5,
        hidden_dim=256,
        privileged_dim=commander_privileged_dim,
    ).to(device)

    if sub_array_size > 0:
        radar = SubArrayRadarActorCritic(
            n_elem=n_elem,
            n_pulses=n_pulses,
            n_bins=n_bins,
            sub_array_size=sub_array_size,
            commander_instr_dim=num_output_length,
        ).to(device)
    else:
        radar = RadarActorCritic(
            n_elem=n_elem,
            n_pulses=n_pulses,
            n_bins=n_bins,
            commander_instr_dim=num_output_length,
            encoder_kwargs=encoder_kwargs,
        ).to(device)

    return {"commander": commander, "radar": radar}


class TeamCritic(nn.Module):
    """Team-level critic for hierarchical advantage estimation.

    Takes a compact team-state summary (104 dims) and predicts a single
    scalar value V_team(s).  Used alongside per-agent critics to give
    long-horizon strategic feedback (launch → 600-step kill).

    Input (104 dims):
      commander_obs      76   — mean commander battlefield observation
      task_fingerprint    8   — [n_teams * 4] per-team task fractions
      avg_snr             4   — per-radar mean detect SNR
      alive               2   — per-team alive flags (any radar alive)
      missile_pos         6   — [pos_x, pos_y, pos_z] per team
      missile_in_flight   2   — per-team in-flight flag
      missile_target      6   — [target_x, target_y, target_z] per team
                           = 76 + 8 + 4 + 2 + 6 + 2 + 6 = 104
    """

    def __init__(self, input_dim: int = 104, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B, 1] value estimate."""
        return self.net(x)


def build_team_state(
    commander_obs: torch.Tensor,      # [E, n_teams, 76] or [B, 76]
    task_fingerprint: torch.Tensor,   # [E, n_teams, 4] or None → zeros
    avg_snr: torch.Tensor,            # [E, n_radars] or None → zeros
    alive: torch.Tensor,              # [E, n_radars] bool
    missile_pos: torch.Tensor,        # [E, n_teams, 3]
    missile_in_flight: torch.Tensor,  # [E, n_teams] bool
    missile_target: torch.Tensor,     # [E, n_teams, 3]
) -> torch.Tensor:
    """Build compact team-state vector [B, 104] for TeamCritic.

    Handles both batch [B, ...] and env [E, ...] shapes — the first dim
    is always the batch dimension.
    """
    dev = commander_obs.device
    B = commander_obs.shape[0]

    # Commander obs: take team 0's observation as summary (or mean if both)
    if commander_obs.dim() == 3:  # [B, n_teams, 76]
        cmd = commander_obs.mean(dim=1)  # [B, 76]
    else:
        cmd = commander_obs  # [B, 76]

    # Task fingerprint: flatten [B, n_teams, 4] → [B, 8]
    if task_fingerprint is not None and task_fingerprint.numel() > 0:
        fp = task_fingerprint.reshape(B, -1)
        if fp.shape[-1] < 8:
            fp = torch.cat([fp, torch.zeros(B, 8 - fp.shape[-1], device=dev)], dim=-1)
        fp = fp[:, :8]
    else:
        fp = torch.zeros(B, 8, device=dev)

    # Avg SNR per radar [B, n_radars] → [B, 4] (pad/truncate to 4)
    if avg_snr is not None and avg_snr.numel() > 0:
        snr = avg_snr.reshape(B, -1)[:, :4]
        if snr.shape[-1] < 4:
            snr = torch.cat([snr, torch.zeros(B, 4 - snr.shape[-1], device=dev)], dim=-1)
    else:
        snr = torch.zeros(B, 4, device=dev)

    # Alive: per-team (any radar alive)
    if alive is not None and alive.numel() > 0:
        al = alive.float().reshape(B, -1)
        # 4 radars → 2 teams: group by n_radars//2
        n_radars = al.shape[-1]
        n_per_team = max(n_radars // 2, 1)
        team_alive = torch.stack([
            al[:, :n_per_team].max(dim=-1).values,
            al[:, n_per_team:2*n_per_team].max(dim=-1).values,
        ], dim=-1)  # [B, 2]
        al_out = team_alive
    else:
        al_out = torch.ones(B, 2, device=dev)

    # Missile state: [B, n_teams, 3] pos → 6, [B, n_teams] in_flight → 2, target → 6
    m_pos = missile_pos.reshape(B, -1)[:, :6] if missile_pos is not None else torch.zeros(B, 6, device=dev)
    if m_pos.shape[-1] < 6:
        m_pos = torch.cat([m_pos, torch.zeros(B, 6 - m_pos.shape[-1], device=dev)], dim=-1)

    m_flight = missile_in_flight.float().reshape(B, -1)[:, :2] if missile_in_flight is not None else torch.zeros(B, 2, device=dev)
    if m_flight.shape[-1] < 2:
        m_flight = torch.cat([m_flight, torch.zeros(B, 2 - m_flight.shape[-1], device=dev)], dim=-1)

    m_tgt = missile_target.reshape(B, -1)[:, :6] if missile_target is not None else torch.zeros(B, 6, device=dev)
    if m_tgt.shape[-1] < 6:
        m_tgt = torch.cat([m_tgt, torch.zeros(B, 6 - m_tgt.shape[-1], device=dev)], dim=-1)

    return torch.cat([cmd, fp, snr, al_out, m_pos, m_flight, m_tgt], dim=-1)
