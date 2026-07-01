"""openevolve integration for neural architecture search.

Searches for optimal CNN/Transformer architecture to process the
[E, R, N, P, n_bins] spectrum input for the MFAR RL agent.

openevolve repo: https://github.com/algorithmicsuperintelligence/openevolve

Search space:
  1. CNN layers, kernel sizes, channel counts
  2. 3D convolution vs decomposed 2D+1D
  3. Attention mechanisms (spatial vs frequency)
  4. Residual connection strategy
  5. Pre-processing: Doppler FFT vs raw temporal stack
"""

import torch
import torch.nn as nn
import numpy as np


class SpectrumEncoder(nn.Module):
    """Configurable CNN encoder for [N, P, n_bins] spectrum input.

    This is the search target for openevolve: the architecture parameters
    are evolved to find the best feature extraction strategy.
    """

    def __init__(
        self,
        n_elem: int = 625,
        n_pulses: int = 32,
        n_bins: int = 1024,
        hidden_dim: int = 256,
        # Evolvable hyperparameters
        n_conv_layers: int = 3,
        base_channels: int = 32,
        kernel_freq: int = 7,
        kernel_time: int = 3,
        use_3d_conv: bool = False,
        use_attention: bool = True,
        use_residual: bool = True,
        use_doppler_fft: bool = False,
    ):
        super().__init__()
        self.n_elem = n_elem
        self.n_pulses = n_pulses
        self.n_bins = n_bins
        self.hidden_dim = hidden_dim
        self.use_doppler_fft = use_doppler_fft

        if use_doppler_fft:
            # Pre-process: Doppler FFT compresses time axis to frequency
            self.freq_bins_time = n_pulses  # after FFT
        else:
            self.freq_bins_time = n_pulses  # keep raw temporal

        if use_3d_conv:
            # 3D convolution: space × time × frequency jointly
            self.encoder = self._build_3d_encoder(
                n_conv_layers, base_channels, kernel_freq, kernel_time,
            )
        else:
            # Decomposed: frequency CNN per element, then spatial aggregation
            self.encoder = self._build_decomposed_encoder(
                n_conv_layers, base_channels, kernel_freq,
            )

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, batch_first=True,
            )
        else:
            self.attention = None

        self.use_residual = use_residual
        self.proj = nn.Linear(base_channels * (n_bins // (2 ** n_conv_layers)), hidden_dim)

    def _build_3d_encoder(self, n_layers, base_ch, k_freq, k_time):
        layers = []
        in_ch = 1
        for i in range(n_layers):
            out_ch = base_ch * (2 ** i)
            layers.append(nn.Conv3d(
                in_ch, out_ch,
                kernel_size=(1, k_time, k_freq),
                padding=(0, k_time // 2, k_freq // 2),
                stride=(1, 2, 2),
            ))
            layers.append(nn.BatchNorm3d(out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
        return nn.Sequential(*layers)

    def _build_decomposed_encoder(self, n_layers, base_ch, k_freq):
        layers = []
        in_ch = self.n_pulses  # treat pulses as channels
        for i in range(n_layers):
            out_ch = base_ch * (2 ** i)
            layers.append(nn.Conv1d(
                in_ch, out_ch, kernel_size=k_freq,
                padding=k_freq // 2, stride=2,
            ))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Encode spectrum to hidden features.

        Args:
            spectrum: [B, N, P, n_bins] float32 power spectra
        Returns:
            [B, hidden_dim] float32 feature vector
        """
        B, N, P, BINS = spectrum.shape

        if self.use_doppler_fft:
            # Doppler FFT across pulses: [B, N, P, BINS] → [B, N, P, BINS]
            spectrum = torch.abs(torch.fft.fft(spectrum, dim=2)) ** 2

        # Frequency CNN per element: [B*N, P, BINS]
        x = spectrum.reshape(B * N, P, BINS)
        x = self.encoder(x)  # [B*N, C, BINS']
        x = x.mean(dim=-1)   # [B*N, C] global average pooling

        features = self.proj(x)  # [B*N, hidden_dim]
        features = features.reshape(B, N, self.hidden_dim)

        # Attention across elements
        if self.attention is not None:
            attn_out, _ = self.attention(features, features, features)
            if self.use_residual:
                features = features + attn_out
            else:
                features = attn_out

        # Aggregate across elements
        return features.mean(dim=1)  # [B, hidden_dim]


class MFARPolicyNetwork(nn.Module):
    """Full policy network: SpectrumEncoder + MLP → Action.

    Used by RL agent (PPO/SAC) to map spectrum observation to actions.
    """

    def __init__(
        self,
        n_elem: int = 625,
        n_pulses: int = 32,
        n_bins: int = 1024,
        hidden_dim: int = 256,
        action_dim: int = 13753,
        vehicle_state_dim: int = 5,
        commander_latent_dim: int = 0,
        comm_data_dim: int = 1250,  # 625 × 2
        # Encoder architecture params (openevolve search targets)
        **encoder_kwargs,
    ):
        super().__init__()

        self.spectrum_encoder = SpectrumEncoder(
            n_elem=n_elem, n_pulses=n_pulses, n_bins=n_bins,
            hidden_dim=hidden_dim, **encoder_kwargs,
        )

        # MLP head: [spectrum_feat + vehicle + comm + commander] → action
        mlp_input_dim = hidden_dim + vehicle_state_dim + comm_data_dim + commander_latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(
        self, spectrum: torch.Tensor, vehicle_state: torch.Tensor,
        comm_data: torch.Tensor, commander_latent: torch.Tensor = None,
    ) -> torch.Tensor:
        """Map observation to action.

        Args:
            spectrum: [B, N, P, n_bins] float32
            vehicle_state: [B, 5] float32
            comm_data: [B, 625×2] float32
            commander_latent: [B, L] float32 or None
        Returns:
            [B, action_dim] float32
        """
        features = self.spectrum_encoder(spectrum)  # [B, hidden_dim]

        parts = [features, vehicle_state, comm_data]
        if commander_latent is not None:
            parts.append(commander_latent)

        mlp_input = torch.cat(parts, dim=-1)
        return self.mlp(mlp_input)


# ---------------------------------------------------------------------------
# openevolve integration
# ---------------------------------------------------------------------------

def create_search_space():
    """Define the search space for openevolve architecture search."""
    return {
        "n_conv_layers": {"type": "int", "min": 1, "max": 6},
        "base_channels": {"type": "choice", "options": [16, 32, 64, 128]},
        "kernel_freq": {"type": "choice", "options": [3, 5, 7, 11, 15]},
        "kernel_time": {"type": "choice", "options": [1, 3, 5]},
        "use_3d_conv": {"type": "bool"},
        "use_attention": {"type": "bool"},
        "use_residual": {"type": "bool"},
        "use_doppler_fft": {"type": "bool"},
        "hidden_dim": {"type": "choice", "options": [128, 256, 512]},
        "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-3, "log": True},
    }


def build_encoder_from_config(config: dict, n_elem=625, n_pulses=32, n_bins=1024):
    """Build SpectrumEncoder from openevolve config dict."""
    return SpectrumEncoder(
        n_elem=n_elem, n_pulses=n_pulses, n_bins=n_bins,
        n_conv_layers=config.get("n_conv_layers", 3),
        base_channels=config.get("base_channels", 32),
        kernel_freq=config.get("kernel_freq", 7),
        kernel_time=config.get("kernel_time", 3),
        use_3d_conv=config.get("use_3d_conv", False),
        use_attention=config.get("use_attention", True),
        use_residual=config.get("use_residual", True),
        use_doppler_fft=config.get("use_doppler_fft", False),
        hidden_dim=config.get("hidden_dim", 256),
    )


def evaluate_architecture(config: dict, env=None, n_episodes: int = 3):
    """Evaluate an architecture config on the MFAR environment.

    Returns a fitness score (higher = better) for openevolve to maximize.
    """
    n_elem = 625
    n_pulses = 32
    n_bins = 1024

    encoder = build_encoder_from_config(config, n_elem, n_pulses, n_bins)
    policy = MFARPolicyNetwork(
        n_elem=n_elem, n_pulses=n_pulses, n_bins=n_bins,
        hidden_dim=config.get("hidden_dim", 256),
        **{k: v for k, v in config.items()
           if k in {"n_conv_layers", "base_channels", "kernel_freq", "kernel_time",
                     "use_3d_conv", "use_attention", "use_residual", "use_doppler_fft"}},
    )

    # Count parameters
    n_params = sum(p.numel() for p in policy.parameters())

    # Forward pass test
    dummy_spectrum = torch.randn(2, n_elem, n_pulses, n_bins)
    dummy_vehicle = torch.randn(2, 5)
    dummy_comm = torch.randn(2, 1250)

    try:
        action = policy(dummy_spectrum, dummy_vehicle, dummy_comm)
        forward_ok = action.shape == (2, policy.mlp[-1].out_features)
    except Exception as e:
        return {"fitness": -100.0, "error": str(e), "n_params": n_params}

    # Fitness: smaller network + valid forward pass = better
    # Real fitness would use RL training reward
    fitness = -n_params / 1e6  # penalize size
    if forward_ok:
        fitness += 1.0

    return {"fitness": fitness, "n_params": n_params, "forward_ok": forward_ok}
