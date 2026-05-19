#!/usr/bin/env python3
"""train_league.py — FluxPhased League Training (self-contained, single-file)

Consolidates the full 4-phase curriculum training pipeline:
  Phase A: Single-task pre-training (recon / detect / jam)
  Phase B: Multi-task integration with reward shaping
  Phase C: PSRO population training with meta-solver (Nash / TC-DAMS)
  Phase D: League exploiter refinement + final evaluation

Usage:
    # Single ablation cell, Phase A only (fastest dry-run)
    python train_league.py --cells R0 --phase a --seed 42

    # Single cell, full A→D pipeline
    python train_league.py --cells R0 --seed 42

    # Full ablation matrix (3 cells, R0/R1/R3)
    python train_league.py --cells R0 R1 R3 --seed 42

Requires: fluxphased conda environment with torch, warp-lang, pettingzoo,
          gym, gymnasium, numpy, scipy, matplotlib, pyyaml

The single external import is the radar simulation environment:
    from radar_sim.gpu.vec_mfar_env import MFARVecEnv
Everything else (PPO, league manager, self-play,
meta-solvers, curriculum) is contained in this file.
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linprog

# ── Flush every print immediately so progress lines land in log files ──
print = functools.partial(builtins.print, flush=True)

# ── Single external dependency: the GPU radar simulation environment ──
from radar_sim.gpu.vec_mfar_env import MFARVecEnv

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

TASK_RECON   = 0
TASK_DETECT  = 1
TASK_JAM     = 2
TASK_COMM    = 3

ROLE_MAIN              = "main"
ROLE_MAIN_EXPLOITER    = "main_exploiter"
ROLE_LEAGUE_EXPLOITER  = "league_exploiter"

DEFAULT_ELO = 1500.0
DEFAULT_K   = 24.0

_EPS = 1e-12

# ═══════════════════════════════════════════════════════════════════════
#  Production 25×25 defaults (streaming mode) — replaces YAML configs
# ═══════════════════════════════════════════════════════════════════════

ENV_DEFAULTS = {
    "rows": 25, "cols": 25, "num_envs": 1, "n_radars": 4,
    "pulses_per_cpi": 1, "fft_size": 32768, "device": "cuda",
    "tx_power_w": 50000,
    "cpi_preallocate": False,  # False=streaming (~3GB), True=batch (~15GB, needs 96GB)
}

PPO_DEFAULTS = {
    "commander_lr": 3e-4, "radar_lr": 1e-4,
    "gamma": 0.99, "gae_lambda": 0.95,
    "commander_clip": 0.2, "radar_clip": 0.1,
    "commander_entropy": 0.01, "radar_entropy": 0.02,
    "value_coef": 0.5, "max_grad_norm": 0.5,
    "n_epochs": 5, "batch_size": 16,
    "buffer_size_commander": 64, "buffer_size_radar": 16,
}

LEAGUE_DEFAULTS = {
    "population_cap": 4,
    "n_eval_games": 5,
    "pfsp_temperature": 1.0,
    "exploiter_reset_prob": 0.1,
    "episodes_per_training": 50,
    "max_steps_per_episode": 50,
}

CURRICULUM_DEFAULTS = {
    "phase_a_episodes": 50,
    "phase_b_episodes": 30,
    "phase_c_iterations": 3,
    "phase_c_episodes_per_iter": 30,
    "phase_d_episodes": 40,
}

ABLATION_CELLS = {
    "R0": {
        "description": "Nash baseline (no TC-DAMS, no Elo-band)",
        "meta_solver": "nash", "tcdams_lambda": 0.3, "use_elo_band": False,
    },
    "R1": {
        "description": "TC-DAMS (task-diversity meta-solver, lambda=0.3)",
        "meta_solver": "tc_dams", "tcdams_lambda": 0.3, "use_elo_band": False,
    },
    "R3": {
        "description": "TC-DAMS + Elo-band PFSP",
        "meta_solver": "tc_dams", "tcdams_lambda": 0.3, "use_elo_band": True,
    },
}


# =======================================================================
#  Section 1: Rollout Buffer
# =======================================================================

class RolloutBuffer:
    """On-policy rollout buffer for PPO training."""

    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 device: str = "cuda"):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        self.obs = torch.zeros(self.buffer_size, self.obs_dim, dtype=torch.float32)
        self.actions = torch.zeros(self.buffer_size, self.act_dim, dtype=torch.float32)
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.values = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.advantages = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, value, log_prob):
        assert self.ptr < self.buffer_size, (
            f"RolloutBuffer overflow at ptr={self.ptr} "
            f"(buffer_size={self.buffer_size}); caller must update() "
            f"before adding past capacity."
        )
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    @property
    def near_full(self) -> bool:
        return self.ptr >= self.buffer_size - 1

    def compute_returns(self, last_value: float = 0.0):
        gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            delta = (self.rewards[t] + self.gamma * next_value * next_non_terminal
                     - self.values[t])
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_minibatches(self, batch_size: int):
        indices = torch.randperm(self.ptr)
        n = self.ptr
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                "obs": self.obs[idx].to(self.device),
                "actions": self.actions[idx].to(self.device),
                "old_log_probs": self.log_probs[idx].to(self.device),
                "advantages": self.advantages[idx].to(self.device),
                "returns": self.returns[idx].to(self.device),
            }

    @property
    def full(self):
        return self.ptr >= self.buffer_size

    @property
    def size(self):
        return self.ptr


# =======================================================================
#  Section 2: Reward Shaping
# =======================================================================

class DenseRewardShaper:
    """Compute dense intermediate rewards from MFARVecEnv step() output dict."""

    def __init__(self, detect_snr_weight: float = 0.1,
                 detect_coverage_weight: float = 0.05,
                 jam_effectiveness_weight: float = 0.1,
                 comm_reliability_weight: float = 0.05,
                 recon_intel_weight: float = 0.03,
                 beam_accuracy_weight: float = 0.02,
                 snr_threshold_db: float = 10.0,
                 device: str = "cuda"):
        self.detect_snr_weight = detect_snr_weight
        self.detect_coverage_weight = detect_coverage_weight
        self.jam_effectiveness_weight = jam_effectiveness_weight
        self.comm_reliability_weight = comm_reliability_weight
        self.recon_intel_weight = recon_intel_weight
        self.beam_accuracy_weight = beam_accuracy_weight
        self.snr_threshold_db = snr_threshold_db
        self.device = device

    def __call__(self, step_output: dict) -> dict:
        spectrum = step_output["spectrum"]
        task_ids = step_output["task_ids"]
        detect_reward = self._detect_reward(spectrum, task_ids)
        jam_reward = self._jam_reward(spectrum, task_ids)
        comm_reward = self._comm_reward(step_output, task_ids)
        recon_reward = self._recon_reward(spectrum, task_ids)
        beam_acc = self._beam_accuracy_reward(step_output)
        total = (detect_reward * self.detect_snr_weight
                 + jam_reward * self.jam_effectiveness_weight
                 + comm_reward * self.comm_reliability_weight
                 + recon_reward * self.recon_intel_weight
                 + beam_acc * self.beam_accuracy_weight)
        return {"detect_reward": detect_reward, "jam_reward": jam_reward,
                "comm_reward": comm_reward, "recon_reward": recon_reward,
                "beam_accuracy": beam_acc,
                "total_shaped": total}

    def _beam_accuracy_reward(self, step_output: dict):
        """Gaussian reward for pointing beam toward target direction.

        World_beam = array_local_beam + array_rotation.
        Reward = exp(-0.5 * (off_bore / sigma)^2) per radar.
        """
        beam_az = step_output.get("beam_az")
        beam_el = step_output.get("beam_el")
        tgt_az = step_output.get("target_az")
        tgt_el = step_output.get("target_el")
        arr_rot = step_output.get("array_rotation")
        if beam_az is None or tgt_az is None:
            return torch.tensor(0.0, device=self.device)

        # World-frame beam direction
        world_az = beam_az + (arr_rot if arr_rot is not None else 0.0)

        # Off-boresight (wrap azimuth)
        d_az = world_az - tgt_az
        d_az = torch.atan2(torch.sin(d_az * np.pi / 180.0),
                            torch.cos(d_az * np.pi / 180.0)) * (180.0 / np.pi)
        d_el = beam_el - (tgt_el if tgt_el is not None else 0.0)

        # Gaussian beam penalty (σ ≈ 1.5× BW for smooth gradient)
        sigma = 6.0  # degrees, gives broad reward catchment
        r = torch.exp(-0.5 * (d_az / sigma)**2 - 0.5 * (d_el / sigma)**2)
        return r.mean().to(self.device)

    def _detect_reward(self, spectrum, task_ids):
        E, R, N, P, B = spectrum.shape
        detect_mask = (task_ids == TASK_DETECT)
        n_detect = detect_mask.sum(dim=-1).clamp(min=1).float()
        detect_spectrum = spectrum * detect_mask.unsqueeze(-1).unsqueeze(-1).float()
        peak_power = detect_spectrum.amax(dim=-1).amax(dim=-1)
        noise_floor = spectrum.median(dim=-1).values.median(dim=-1).values.clamp(min=1e-30)
        snr_db = 10.0 * torch.log10(peak_power.clamp(min=1e-30) / noise_floor.clamp(min=1e-30))
        snr_db = snr_db * detect_mask.float()
        above_thresh = (snr_db > self.snr_threshold_db).float()
        coverage = above_thresh.sum(dim=-1) / n_detect
        avg_snr = (snr_db - self.snr_threshold_db).clamp(min=0).sum(dim=-1) / (n_detect * 20.0)
        return coverage * 0.5 + avg_snr * 0.5

    def _jam_reward(self, spectrum, task_ids):
        E, R, N, P, B = spectrum.shape
        jam_mask = (task_ids == TASK_JAM).float()
        jam_fraction = jam_mask.sum(dim=-1) / N
        jam_energy = (spectrum.mean(dim=-1).mean(dim=-1) * jam_mask).sum(dim=-1)
        jam_energy_norm = jam_energy / (jam_energy.amax() + 1e-10)
        return 0.3 * jam_fraction + 0.7 * jam_energy_norm

    def _comm_reward(self, step_output, task_ids):
        E, R, N = task_ids.shape
        dev = task_ids.device
        crc_ok = step_output.get("comm_crc_ok")
        if crc_ok is None:
            return torch.zeros(E, R, device=dev)
        team_id = torch.tensor([i // (R // 2) for i in range(R)], device=dev)
        crc_per_radar = crc_ok[:, team_id].float()
        comm_mask = (task_ids == TASK_COMM).float()
        comm_fraction = comm_mask.sum(dim=-1) / task_ids.shape[-1]
        return crc_per_radar * (0.5 + 0.5 * comm_fraction)

    def _recon_reward(self, spectrum, task_ids):
        E, R, N, P, B = spectrum.shape
        recon_mask = (task_ids == TASK_RECON).float()
        recon_energy = (spectrum.mean(dim=-1).mean(dim=-1) * recon_mask).sum(dim=-1)
        max_energy = recon_energy.amax() + 1e-10
        return (recon_energy / max_energy).clamp(0, 1)


# =======================================================================
#  Section 3: Actor-Critic Networks
# =======================================================================

class AdaptiveSpectrumEncoder(nn.Module):
    """Spectrum encoder that adapts to any input size."""

    def __init__(self, n_elem: int, n_pulses: int, n_bins: int,
                 hidden_dim: int = 256, base_channels: int = 32):
        super().__init__()
        self.n_elem = n_elem; self.n_pulses = n_pulses; self.n_bins = n_bins
        self.hidden_dim = hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(n_pulses, base_channels, kernel_size=7, padding=3, stride=2), nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2, stride=2), nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, stride=2), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_pulses, n_bins)
            conv_out = self.conv(dummy)
            conv_flat = conv_out.shape[1] * conv_out.shape[2]
        self.proj = nn.Linear(conv_flat, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        B, N, P, BINS = spectrum.shape
        x = spectrum.reshape(B * N, P, BINS)
        x = self.conv(x)
        x = x.reshape(B * N, -1)
        x = self.proj(x)
        x = x.reshape(B, N, self.hidden_dim)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        return x.mean(dim=1)


class CommanderActorCritic(nn.Module):
    """Commander policy: MLP (obs_dim→256→256), tanh action, learnable log_std."""

    def __init__(self, obs_dim: int = 68, act_dim: int = 35, hidden_dim: int = 256):
        super().__init__()
        self.act_dim = act_dim; self.obs_dim = obs_dim
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 1.0)

    def forward(self, obs):
        features = self.shared(obs)
        action = torch.tanh(self.action_head(features))
        value = self.value_head(features)
        return action, value

    def get_action(self, obs, deterministic=False):
        features = self.shared(obs)
        mean = self.action_head(features)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.exp(self.log_std.clamp(-20, 2)).expand_as(mean).clamp(1e-4, 1e4)
        dist = torch.distributions.Normal(mean, std)
        raw_action = mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        value = self.value_head(features)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        features = self.shared(obs)
        mean = self.action_head(features)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        std = torch.exp(self.log_std.clamp(-20, 2)).expand_as(mean).clamp(1e-4, 1e4)
        dist = torch.distributions.Normal(mean, std)
        raw_action = 0.5 * torch.log((actions + 1.0) / (1.0 - actions + 1e-6).clamp(min=1e-6))
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(features)
        return log_prob, entropy, value


class RadarActorCritic(nn.Module):
    """Radar policy: SpectrumEncoder → task/param/vehicle heads + value head."""

    def __init__(self, n_elem: int = 625, n_pulses: int = 32, n_bins: int = 1024,
                 spectrum_hidden: int = 256, vehicle_dim: int = 5, missile_dim: int = 12,
                 commander_instr_dim: int = 16, encoder_kwargs: dict = None):
        super().__init__()
        self.n_elem = n_elem; self.n_pulses = n_pulses; self.n_bins = n_bins
        ek = encoder_kwargs or {}
        base_channels = ek.get("base_channels", 32)
        self.spectrum_encoder = AdaptiveSpectrumEncoder(
            n_elem=n_elem, n_pulses=n_pulses, n_bins=n_bins,
            hidden_dim=spectrum_hidden, base_channels=base_channels)
        self.spectrum_flat_dim = n_elem * n_pulses * n_bins
        self.comm_flat_dim = n_elem * 2
        self.other_dim = vehicle_dim + missile_dim + commander_instr_dim
        shared_in = spectrum_hidden + self.comm_flat_dim + self.other_dim
        self.shared = nn.Sequential(
            nn.Linear(shared_in, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU())
        self.task_head = nn.Linear(512, n_elem * 4)
        self.param_head = nn.Linear(512, n_elem * 8)
        self.vehicle_head = nn.Linear(512, 3)
        self.value_head = nn.Linear(512, 1)
        self.log_std_params = nn.Parameter(torch.zeros(n_elem * 8))
        self.log_std_vehicle = nn.Parameter(torch.zeros(3))

    def forward(self, state):
        action, _, value = self.get_action(state, deterministic=True)
        return action, value

    def get_action(self, state, deterministic=False):
        B = state.shape[0]; N = self.n_elem
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
        task_logits = self.task_head(features).reshape(B, N, 4)
        task_dist = torch.distributions.Categorical(logits=task_logits)
        param_mean = torch.sigmoid(self.param_head(features))
        param_std = torch.exp(self.log_std_params).expand_as(param_mean)
        param_dist = torch.distributions.Normal(param_mean, param_std)
        veh_mean = torch.tanh(self.vehicle_head(features))
        veh_std = torch.exp(self.log_std_vehicle).expand_as(veh_mean)
        veh_dist = torch.distributions.Normal(veh_mean, veh_std)
        value = self.value_head(features)
        if deterministic:
            task_choice = task_logits.argmax(dim=-1)
            params = param_mean; vehicle = veh_mean
        else:
            task_choice = task_dist.sample()
            params = param_dist.rsample().clamp(0.01, 0.99)
            vehicle = veh_dist.rsample().clamp(-0.999, 0.999)
        task_logp = task_dist.log_prob(task_choice).sum(dim=-1)
        param_logp = param_dist.log_prob(params).sum(dim=-1)
        veh_logp = veh_dist.log_prob(vehicle).sum(dim=-1)
        log_prob = task_logp + param_logp + veh_logp
        task_frac = torch.zeros(B, N, 4, device=state.device)
        task_frac.scatter_(-1, task_choice.unsqueeze(-1), 1.0)
        action = self._assemble_action_from_parts(task_frac, params, vehicle)
        return action, log_prob, value

    def _assemble_action_from_parts(self, task_frac, params, vehicle):
        B = task_frac.shape[0]; N = self.n_elem
        p = params.reshape(B, N, 8)
        beam_az = p[..., 0:1].expand(B, N, 4) * 0.5 + 0.5
        beam_el = p[..., 1:2].expand(B, N, 4) * 0.5 + 0.5
        beam = torch.stack([beam_az, beam_el], dim=-1).reshape(B, N, 8)
        detect_p = p[..., 2:5]; jam_p = p[..., 5:8]
        comm_p = torch.cat([p[..., 2:3], p[..., 0:1], p[..., 6:7], p[..., 7:8]], dim=-1)
        elem_action = torch.cat([task_frac, beam, detect_p, jam_p, comm_p], dim=-1)
        flat = elem_action.reshape(B, N * 22)
        return torch.cat([flat, vehicle], dim=-1)

    def get_distribution(self, state):
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
        features = torch.nan_to_num(features, nan=0.0)
        task_logits = self.task_head(features).reshape(B, self.n_elem, 4)
        param_mean = torch.sigmoid(self.param_head(features))
        vehicle_mean = torch.tanh(self.vehicle_head(features))
        value = self.value_head(features)
        task_dist = torch.distributions.Categorical(logits=task_logits)
        param_std = torch.exp(self.log_std_params.clamp(-20, 2)).clamp(1e-4, 1e4).expand_as(param_mean)
        param_dist = torch.distributions.Normal(param_mean, param_std)
        veh_std = torch.exp(self.log_std_vehicle).expand_as(vehicle_mean)
        vehicle_dist = torch.distributions.Normal(vehicle_mean, veh_std)
        return task_dist, param_dist, vehicle_dist, value

    def evaluate_actions(self, state, actions):
        task_dist, param_dist, vehicle_dist, value = self.get_distribution(state)
        N = self.n_elem
        elem_act = actions[..., :N * 22].reshape(-1, N, 22)
        task_frac = elem_act[..., 0:4]
        vehicle_act = actions[..., -3:]
        task_choice = task_frac.argmax(dim=-1)
        task_logp = task_dist.log_prob(task_choice)
        task_ent = task_dist.entropy()
        param_act = actions[..., :N * 8]
        param_logp = param_dist.log_prob(param_act.clamp(0.01, 0.99)).sum(dim=-1)
        veh_logp = vehicle_dist.log_prob(vehicle_act).sum(dim=-1)
        log_prob = task_logp.sum(dim=-1) + param_logp + veh_logp
        entropy = task_ent.sum(dim=-1) + param_dist.entropy().sum(dim=-1) + vehicle_dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class SubArrayRadarActorCritic(nn.Module):
    """Sub-array decomposed radar policy (drop-in replacement for RadarActorCritic).

    Supports compact observation encoding: raw state (5M dim) → compact (~6.5K dim)
    before storing in the rollout buffer, reducing buffer memory by ~780x.
    """

    def __init__(self, n_elem: int = 625, n_pulses: int = 32, n_bins: int = 1024,
                 sub_array_size: int = 5, spectrum_hidden: int = 256,
                 vehicle_dim: int = 5, missile_dim: int = 12, commander_instr_dim: int = 16):
        super().__init__()
        self.n_elem = n_elem; self.n_pulses = n_pulses; self.n_bins = n_bins
        self.sub_size = sub_array_size
        self.elem_per_sub = sub_array_size * sub_array_size
        self.n_sub = n_elem // self.elem_per_sub
        assert n_elem % self.elem_per_sub == 0
        self.spectrum_flat_dim = n_elem * n_pulses * n_bins
        self.comm_flat_dim = n_elem * 2
        self.recon_flat_dim = n_elem * 4
        self.other_dim = vehicle_dim + missile_dim + commander_instr_dim
        # Compact obs: encoded_spec [K*256] + sub_comm [K*2] + sub_recon [K*4] + other [33]
        self.compact_dim = self.n_sub * (spectrum_hidden + 2 + 4) + self.other_dim
        # Frequency compressor: strided conv stack to compress n_bins → compact features
        # Replaces the old conv+proj for large-n_bins spectrum preprocessing
        self.freq_compressor = nn.Sequential(
            nn.Conv1d(n_pulses, 16, kernel_size=15, stride=8, padding=7), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=8, padding=7), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=8, padding=7), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=8, padding=7), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(128 * 16, spectrum_hidden), nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(embed_dim=spectrum_hidden, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(spectrum_hidden)
        per_sub_in = spectrum_hidden + 2 + 4
        self.sub_norm = nn.LayerNorm(per_sub_in)
        self.shared = nn.Sequential(nn.Linear(per_sub_in, 256), nn.ReLU())
        self.global_norm = nn.LayerNorm(spectrum_hidden + self.other_dim)
        self.task_head = nn.Linear(256, 4); self.param_head = nn.Linear(256, 8)
        self.vehicle_head = nn.Linear(spectrum_hidden + self.other_dim, 3)
        self.value_head = nn.Linear(spectrum_hidden + self.other_dim, 1)
        self.log_std_params = nn.Parameter(torch.zeros(8))
        self.log_std_vehicle = nn.Parameter(torch.zeros(3))

    def _parse_state(self, state):
        B = state.shape[0]; off = 0
        spec_flat = state[..., off:off + self.spectrum_flat_dim]; off += self.spectrum_flat_dim
        spectrum = spec_flat.reshape(B, self.n_elem, self.n_pulses, self.n_bins)
        comm_flat = state[..., off:off + self.comm_flat_dim]; off += self.comm_flat_dim
        comm = comm_flat.reshape(B, self.n_elem, 2)
        recon_flat = state[..., off:off + self.recon_flat_dim]; off += self.recon_flat_dim
        recon = recon_flat.reshape(B, self.n_elem, 4)
        other = state[..., off:off + self.other_dim]
        return spectrum, comm, recon, other

    def _encode_sub_arrays(self, spectrum, comm, recon):
        B = spectrum.shape[0]; K = self.n_sub; S2 = self.elem_per_sub
        P, BINS = self.n_pulses, self.n_bins
        # max preserves signal peaks for detection; mean for comm/recon link quality
        sub_spec = spectrum.reshape(B, K, S2, P, BINS).max(dim=2).values
        sub_comm = comm.reshape(B, K, S2, 2).max(dim=2).values
        sub_recon = recon.reshape(B, K, S2, 4).max(dim=2).values
        x = sub_spec.reshape(B * K, P, BINS)
        x = self.freq_compressor(x); x = x.reshape(B, K, -1)
        attn_out, _ = self.attention(x, x, x); x = self.attn_norm(x + attn_out)
        sub_features = torch.cat([x, sub_comm, sub_recon], dim=-1)
        return sub_features, x

    def encode_obs(self, state):
        """Compress raw state (~5.1M dim) → compact features (~6.5K dim) for buffer storage."""
        spectrum, comm, recon, other = self._parse_state(state)
        sub_feat, pooled_spec = self._encode_sub_arrays(spectrum, comm, recon)
        # sub_feat: [B, K, 262], pooled_spec: [B, K, 256], other: [B, 33]
        B = state.shape[0]; K = self.n_sub
        return torch.cat([
            pooled_spec.reshape(B, K * 256),      # 6400
            sub_feat[..., 256:258].reshape(B, K * 2),  # sub_comm: 50
            sub_feat[..., 258:262].reshape(B, K * 4),  # sub_recon: 100
            other,                                     # 33
        ], dim=-1)  # total: 6583

    def _decode_obs(self, compact):
        """Reverse encode_obs for network consumption."""
        B = compact.shape[0]; K = self.n_sub
        off = 0
        n_spec = K * 256; pooled_spec = compact[:, off:off + n_spec].reshape(B, K, 256); off += n_spec
        n_comm = K * 2; sub_comm = compact[:, off:off + n_comm].reshape(B, K, 2); off += n_comm
        n_recon = K * 4; sub_recon = compact[:, off:off + n_recon].reshape(B, K, 4); off += n_recon
        other = compact[:, off:off + self.other_dim]
        return pooled_spec, sub_comm, sub_recon, other

    def _get_distributions(self, state):
        B = state.shape[0]; K = self.n_sub
        spectrum, comm, recon, other = self._parse_state(state)
        sub_feat, pooled_spec = self._encode_sub_arrays(spectrum, comm, recon)
        global_spec = pooled_spec.mean(dim=1)
        global_feat = torch.cat([global_spec, other], dim=-1)
        global_feat = self.global_norm(global_feat)
        sub_feat = self.sub_norm(sub_feat)
        shared_feat = self.shared(sub_feat)
        task_logits = torch.nan_to_num(self.task_head(shared_feat), nan=0.0)
        task_dist = torch.distributions.Categorical(logits=task_logits)
        param_mean = torch.sigmoid(torch.nan_to_num(self.param_head(shared_feat), nan=0.0))
        param_std = torch.exp(self.log_std_params.clamp(-20, 2)).clamp(1e-4, 1e4)
        param_dist = torch.distributions.Normal(param_mean, param_std)
        veh_mean = torch.tanh(torch.nan_to_num(self.vehicle_head(global_feat), nan=0.0))
        veh_std = torch.exp(self.log_std_vehicle)
        vehicle_dist = torch.distributions.Normal(veh_mean, veh_std)
        value = self.value_head(global_feat)
        return task_dist, param_dist, vehicle_dist, value

    def _expand_to_elements(self, task_frac, params, vehicle):
        B = task_frac.shape[0]; K = self.n_sub; S2 = self.elem_per_sub; N = self.n_elem
        p = params
        beam_az = p[..., 0:1].expand(B, K, 4) * 2.0 - 1.0
        beam_el = p[..., 1:2].expand(B, K, 4) * 2.0 - 1.0
        beam = torch.stack([beam_az, beam_el], dim=-1).reshape(B, K, 8)
        detect_p = p[..., 2:5]; jam_p = p[..., 5:8]
        comm_p = torch.cat([p[..., 2:3], p[..., 0:1], p[..., 6:7], p[..., 7:8]], dim=-1)
        sub_action = torch.cat([task_frac, beam, detect_p, jam_p, comm_p], dim=-1)
        elem_action = sub_action.unsqueeze(2).expand(B, K, S2, 22).reshape(B, N, 22)
        flat = elem_action.reshape(B, N * 22)
        return torch.cat([flat, vehicle], dim=-1)

    def _extract_sub_from_elem(self, actions):
        B = actions.shape[0]; K = self.n_sub; S2 = self.elem_per_sub; N = self.n_elem
        elem_act = actions[..., :N * 22].reshape(B, N, 22)
        indices = torch.arange(K, device=actions.device) * S2
        sub_act = elem_act[:, indices, :]
        task_frac = sub_act[..., 0:4]; vehicle = actions[..., -3:]
        params = torch.cat([
            (sub_act[..., 4:5] - 0.5) * 2, (sub_act[..., 5:6] - 0.5) * 2,
            sub_act[..., 12:15], sub_act[..., 15:18]], dim=-1)
        return task_frac, params, vehicle

    def forward(self, state):
        action, _, value = self.get_action(state, deterministic=True)
        return action, value

    def get_action(self, state, deterministic=False):
        B = state.shape[0]; K = self.n_sub
        task_dist, param_dist, vehicle_dist, value = self._get_distributions(state)
        if deterministic:
            task_choice = task_dist.logits.argmax(dim=-1)
            params = param_dist.mean; vehicle = vehicle_dist.mean
        else:
            task_choice = task_dist.sample()
            params = param_dist.rsample().clamp(0.01, 0.99)
            vehicle = vehicle_dist.rsample().clamp(-0.999, 0.999)
        task_logp = task_dist.log_prob(task_choice).sum(dim=-1)
        param_logp = param_dist.log_prob(params).sum(-1).sum(-1)
        veh_logp = vehicle_dist.log_prob(vehicle).sum(dim=-1)
        log_prob = task_logp + param_logp + veh_logp
        task_frac = torch.zeros(B, K, 4, device=state.device)
        task_frac.scatter_(-1, task_choice.unsqueeze(-1), 1.0)
        action = self._expand_to_elements(task_frac, params, vehicle)
        return action, log_prob, value

    def evaluate_actions(self, state, actions):
        task_dist, param_dist, vehicle_dist, value = self._get_distributions(state)
        task_frac, params, vehicle = self._extract_sub_from_elem(actions)
        task_choice = task_frac.argmax(dim=-1)
        task_logp = task_dist.log_prob(task_choice)
        task_ent = task_dist.entropy()
        param_logp = param_dist.log_prob(params.clamp(0.01, 0.99)).sum(-1).sum(-1)
        veh_logp = vehicle_dist.log_prob(vehicle.clamp(-0.999, 0.999)).sum(dim=-1)
        log_prob = task_logp.sum(dim=-1) + param_logp + veh_logp
        entropy = (task_ent.sum(dim=-1) + param_dist.entropy().sum(-1).sum(-1)
                   + vehicle_dist.entropy().sum(dim=-1))
        return log_prob, entropy, value

    def get_distribution(self, state):
        return self._get_distributions(state)


def create_team_policy(team: int, n_elem: int = 625, n_pulses: int = 32,
                       n_bins: int = 1024, num_output_length: int = 16,
                       device: str = "cuda", encoder_kwargs: dict = None,
                       sub_array_size: int = 0) -> dict:
    commander = CommanderActorCritic(obs_dim=68, act_dim=35, hidden_dim=256).to(device)
    if sub_array_size > 0:
        radar = SubArrayRadarActorCritic(
            n_elem=n_elem, n_pulses=n_pulses, n_bins=n_bins,
            sub_array_size=sub_array_size, commander_instr_dim=num_output_length).to(device)
    else:
        radar = RadarActorCritic(
            n_elem=n_elem, n_pulses=n_pulses, n_bins=n_bins,
            commander_instr_dim=num_output_length, encoder_kwargs=encoder_kwargs).to(device)
    return {"commander": commander, "radar": radar}


# =======================================================================
#  Section 4: PPO Trainer
# =======================================================================

class PPOTrainer:
    """PPO training loop for one agent (commander or radar)."""

    def __init__(self, actor_critic: nn.Module, lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_range: float = 0.2,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 max_grad_norm: float = 0.5, n_epochs: int = 10,
                 batch_size: int = 64, buffer_size: int = 2048, device: str = "cuda"):
        self.ac = actor_critic
        self.gamma = gamma; self.gae_lambda = gae_lambda
        self.clip_range = clip_range; self.entropy_coef = entropy_coef
        self.value_coef = value_coef; self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs; self.batch_size = batch_size; self.device = device
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

    def update(self, buffer: RolloutBuffer) -> dict:
        total_policy_loss = 0.0; total_value_loss = 0.0; total_entropy = 0.0; n_updates = 0
        for _ in range(self.n_epochs):
            for batch in buffer.get_minibatches(self.batch_size):
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                log_prob, entropy, value = self.ac.evaluate_actions(batch["obs"], batch["actions"])
                ratio = torch.exp(log_prob - batch["old_log_probs"])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((value.squeeze() - batch["returns"]) ** 2).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        return {"policy_loss": total_policy_loss / max(n_updates, 1),
                "value_loss": total_value_loss / max(n_updates, 1),
                "entropy": total_entropy / max(n_updates, 1)}


class TeamPPOTrainer:
    """Manages PPO training for a full team (1 commander + shared radar)."""

    def __init__(self, commander: CommanderActorCritic, radar: RadarActorCritic,
                 commander_lr: float = 3e-4, radar_lr: float = 1e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 commander_clip: float = 0.2, radar_clip: float = 0.1,
                 commander_entropy: float = 0.01, radar_entropy: float = 0.02,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 n_epochs: int = 10, batch_size: int = 64,
                 buffer_size_commander: int = 256, buffer_size_radar: int = 64,
                 device: str = "cuda"):
        self.commander_trainer = PPOTrainer(
            commander, lr=commander_lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=commander_clip, entropy_coef=commander_entropy,
            value_coef=value_coef, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
            buffer_size=buffer_size_commander, device=device)
        self.radar_trainer = PPOTrainer(
            radar, lr=radar_lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=radar_clip, entropy_coef=radar_entropy,
            value_coef=value_coef, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
            buffer_size=buffer_size_radar, device=device)
        self.device = device; self.gamma = gamma; self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.buffer_size_commander = buffer_size_commander
        self.buffer_size_radar = buffer_size_radar
        self.commander_buffer = None; self.radar_buffer = None
        self.reward_shaper = DenseRewardShaper(device=device)

    def init_buffers(self, env_state_dim: int, env_action_dim: int):
        self.commander_buffer = RolloutBuffer(
            self.buffer_size_commander, obs_dim=68, act_dim=35,
            gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device)
        self.radar_buffer = RolloutBuffer(
            self.buffer_size_radar, obs_dim=env_state_dim, act_dim=env_action_dim,
            gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device)

    def _get_observations(self, env):
        state = env._assemble_state(env._buf_spectrum, env._buf_comm_data)
        comm_input = torch.zeros(env.num_envs, env.n_radars, env.num_input_length,
                                 device=self.device)
        commander_obs = env.battlefield.get_commander_observation(env.radar_pos, comm_input)
        return state, commander_obs

    def get_own_actions(self, env, team: int, deterministic: bool = False):
        r_per_team = env.n_radars // env.n_teams
        r_start = team * r_per_team; r_end = r_start + r_per_team
        state, commander_obs = self._get_observations(env)
        with torch.no_grad():
            cmd_obs = commander_obs[:, team, :]
            cmd_action, cmd_logp, cmd_val = self.commander_trainer.ac.get_action(
                cmd_obs, deterministic=deterministic)
            radar_actions = []; rep_logp = rep_val = rep_obs = rep_action = None
            for r in range(r_start, r_end):
                r_obs = state[:, r, :]
                r_act, r_logp, r_val = self.radar_trainer.ac.get_action(
                    r_obs, deterministic=deterministic)
                radar_actions.append(r_act)
                if r == r_start:
                    rep_obs = r_obs; rep_action = r_act
                    rep_logp = r_logp; rep_val = r_val.squeeze(-1)
        return {"radar_actions": radar_actions, "commander_action": cmd_action,
                "transition": {"cmd_obs": cmd_obs, "cmd_action": cmd_action,
                               "cmd_logp": cmd_logp, "cmd_val": cmd_val.squeeze(-1),
                               "radar_obs": rep_obs, "radar_action": rep_action,
                               "radar_logp": rep_logp, "radar_val": rep_val},
                "r_start": r_start, "r_end": r_end}

    def store_transition(self, env, result: dict, transition: dict, team: int):
        shaped = self.reward_shaper(result)
        r_per_team = env.n_radars // env.n_teams; r_start = team * r_per_team
        total_radar_reward = shaped["total_shaped"] + result["radar_rewards"]
        cmd_reward = result["commander_rewards"]
        radar_obs = transition["radar_obs"]
        radar_action = transition["radar_action"]
        for e in range(env.num_envs):
            done = float(result["dones"][e].item())
            self.commander_buffer.add(
                obs=transition["cmd_obs"][e].cpu(),
                action=transition["cmd_action"][e].cpu(),
                reward=cmd_reward[e, team].item(), done=done,
                value=transition["cmd_val"][e].item(),
                log_prob=transition["cmd_logp"][e].item())
            radar_reward = total_radar_reward[e, r_start:r_start + r_per_team].sum().item()
            self.radar_buffer.add(
                obs=radar_obs[e].cpu(),
                action=radar_action[e].cpu(),
                reward=radar_reward, done=done,
                value=transition["radar_val"][e].item(),
                log_prob=transition["radar_logp"][e].item())
        return {"radar_reward": total_radar_reward, "commander_reward": cmd_reward,
                "shaped_rewards": shaped}

    def update(self) -> dict:
        cmd_metrics = {}; radar_metrics = {}
        if self.commander_buffer and self.commander_buffer.size > self.batch_size:
            self.commander_buffer.compute_returns()
            cmd_metrics = self.commander_trainer.update(self.commander_buffer)
            self.commander_buffer.reset()
        if self.radar_buffer and self.radar_buffer.size > self.batch_size:
            self.radar_buffer.compute_returns()
            radar_metrics = self.radar_trainer.update(self.radar_buffer)
            self.radar_buffer.reset()
        return {"commander": cmd_metrics, "radar": radar_metrics}

    def save(self, path: str):
        torch.save({"commander": self.commander_trainer.ac.state_dict(),
                     "radar": self.radar_trainer.ac.state_dict(),
                     "commander_optimizer": self.commander_trainer.optimizer.state_dict(),
                     "radar_optimizer": self.radar_trainer.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.commander_trainer.ac.load_state_dict(ckpt["commander"])
        self.radar_trainer.ac.load_state_dict(ckpt["radar"])
        self.commander_trainer.optimizer.load_state_dict(ckpt["commander_optimizer"])
        self.radar_trainer.optimizer.load_state_dict(ckpt["radar_optimizer"])


# =======================================================================
#  Section 5: Opponent Pool
# =======================================================================

@dataclass
class PolicyRecord:
    policy_id: str; team: int; role: str; checkpoint_path: str; generation: int
    win_rates: dict = field(default_factory=dict)
    parent_id: Optional[str] = None; is_active: bool = True


class OpponentPool:
    """Pool of past policy checkpoints for self-play training."""

    def __init__(self, pool_dir: str = "checkpoints/pool", population_cap: int = 20,
                 pfsp_temperature: float = 1.0):
        self.pool_dir = pool_dir; self.population_cap = population_cap
        self.pfsp_temperature = pfsp_temperature
        self.policies: dict = {}; self._next_id = 0
        os.makedirs(pool_dir, exist_ok=True)

    def add_policy(self, team: int, role: str, checkpoint_path: str,
                   generation: int, parent_id: str = None) -> str:
        policy_id = f"p{self._next_id:04d}"; self._next_id += 1
        record = PolicyRecord(policy_id=policy_id, team=team, role=role,
                              checkpoint_path=checkpoint_path, generation=generation,
                              parent_id=parent_id)
        self.policies[policy_id] = record
        if len(self.policies) > self.population_cap:
            self._evict_oldest()
        return policy_id

    def get_opponent_team(self, team: int) -> List[str]:
        return [pid for pid, rec in self.policies.items()
                if rec.team != team and rec.is_active]

    def get_team_policies(self, team: int, role: str = None) -> List[str]:
        return [pid for pid, rec in self.policies.items()
                if rec.team == team and rec.is_active
                and (role is None or rec.role == role)]

    def sample_pfsp(self, current_policy_id: str, n_samples: int = 1) -> List[str]:
        record = self.policies[current_policy_id]
        opponents = self.get_opponent_team(record.team)
        if not opponents: return []
        win_rates = np.array([record.win_rates.get(opp, 0.5) for opp in opponents])
        loss_rates = 1.0 - win_rates
        if loss_rates.sum() < 1e-8:
            probs = np.ones(len(opponents)) / len(opponents)
        else:
            logits = loss_rates / self.pfsp_temperature
            logits = logits - logits.max()
            probs = np.exp(logits); probs = probs / probs.sum()
        indices = np.random.choice(len(opponents), size=min(n_samples, len(opponents)),
                                   replace=False, p=probs)
        return [opponents[i] for i in indices]

    def sample_uniform(self, current_policy_id: str, n_samples: int = 1) -> List[str]:
        record = self.policies[current_policy_id]
        opponents = self.get_opponent_team(record.team)
        if not opponents: return []
        indices = np.random.choice(len(opponents), size=min(n_samples, len(opponents)),
                                   replace=False)
        return [opponents[i] for i in indices]

    def update_win_rate(self, policy_id: str, opponent_id: str, win: bool):
        record = self.policies[policy_id]; key = opponent_id
        if key not in record.win_rates: record.win_rates[key] = 0.5
        alpha = 0.1
        record.win_rates[key] = (1 - alpha) * record.win_rates[key] + alpha * float(win)

    def get_active_main(self, team: int) -> Optional[str]:
        mains = [pid for pid, rec in self.policies.items()
                 if rec.team == team and rec.role == "main" and rec.is_active]
        if not mains: return None
        return max(mains, key=lambda pid: self.policies[pid].generation)

    def load_policy(self, policy_id: str) -> dict:
        path = self.policies[policy_id].checkpoint_path
        return torch.load(path, map_location="cpu", weights_only=False)

    def _evict_oldest(self):
        candidates = [(pid, rec) for pid, rec in self.policies.items()
                      if rec.role != "main"]
        if not candidates: return
        oldest = min(candidates, key=lambda x: x[1].generation)
        self.policies[oldest[0]].is_active = False

    def save_metadata(self):
        import json
        meta = {}
        for pid, rec in self.policies.items():
            meta[pid] = {"team": rec.team, "role": rec.role,
                         "checkpoint_path": rec.checkpoint_path,
                         "generation": rec.generation, "parent_id": rec.parent_id,
                         "is_active": rec.is_active,
                         "win_rates": {k: float(v) for k, v in rec.win_rates.items()}}
        path = os.path.join(self.pool_dir, "pool_metadata.json")
        with open(path, "w") as f: json.dump(meta, f, indent=2)

    def load_metadata(self):
        import json
        path = os.path.join(self.pool_dir, "pool_metadata.json")
        if not os.path.exists(path): return
        with open(path) as f: meta = json.load(f)
        for pid, data in meta.items():
            rec = PolicyRecord(policy_id=pid, team=data["team"], role=data["role"],
                               checkpoint_path=data["checkpoint_path"],
                               generation=data["generation"],
                               parent_id=data.get("parent_id"),
                               is_active=data.get("is_active", True),
                               win_rates=data.get("win_rates", {}))
            self.policies[pid] = rec
            self._next_id = max(self._next_id, int(pid[1:]) + 1)


# =======================================================================
#  Section 6: Payoff Matrix
# =======================================================================

class PayoffMatrix:
    """Compute and store empirical payoff matrix between policy populations."""

    def __init__(self, opponent_pool: OpponentPool, n_eval_games: int = 50,
                 device: str = "cuda", max_steps_per_game: int = 200):
        self.pool = opponent_pool; self.n_eval_games = n_eval_games
        self.device = device; self.max_steps_per_game = max_steps_per_game
        self.matrix: Dict[Tuple[str, str], float] = {}
        self.fingerprints: Dict[str, np.ndarray] = {}
        self._fp_counts: Dict[str, int] = {}

    def _accumulate_fingerprint(self, policy_id: str, fp: np.ndarray):
        fp = np.asarray(fp, dtype=np.float64).reshape(4)
        n = self._fp_counts.get(policy_id, 0)
        if n == 0: self.fingerprints[policy_id] = fp.copy()
        else: self.fingerprints[policy_id] = (self.fingerprints[policy_id] * n + fp) / (n + 1)
        self._fp_counts[policy_id] = n + 1

    def evaluate_pair(self, red_policy_id: str, blue_policy_id: str,
                      env, red_trainer, blue_trainer) -> float:
        red_wins = 0; total = 0; remaining = self.n_eval_games; E = env.num_envs
        while remaining > 0:
            batch = min(E, remaining); env.reset(); game_ended = False
            for step in range(self.max_steps_per_game):
                with torch.no_grad():
                    red = red_trainer.get_own_actions(env, team=0, deterministic=True)
                    blue = blue_trainer.get_own_actions(env, team=1, deterministic=True)
                    actions = torch.zeros(batch, env.n_radars, env.action_dim, device=self.device)
                    for i, r in enumerate(range(red["r_start"], red["r_end"])):
                        actions[:, r, :] = red["radar_actions"][i]
                    for i, r in enumerate(range(blue["r_start"], blue["r_end"])):
                        actions[:, r, :] = blue["radar_actions"][i]
                    commander_actions = torch.zeros(
                        batch, env.n_teams, env.battlefield.commander_action_dim, device=self.device)
                    commander_actions[:, 0, :] = red["commander_action"]
                    commander_actions[:, 1, :] = blue["commander_action"]
                result = env.step(actions=actions, commander_actions=commander_actions)
                fp_t = result.get("task_fingerprint", None)
                if fp_t is not None:
                    fp_red = fp_t[:batch, 0].mean(dim=0).detach().cpu().numpy()
                    fp_blue = fp_t[:batch, 1].mean(dim=0).detach().cpu().numpy()
                    self._accumulate_fingerprint(red_policy_id, fp_red)
                    self._accumulate_fingerprint(blue_policy_id, fp_blue)
                if result["dones"].any():
                    for e in range(batch):
                        if result["dones"][e]:
                            if result["winners"][e] == 0: red_wins += 1
                            total += 1
                    game_ended = True; break
            if not game_ended:
                red_wins += 0.5 * batch; total += batch
            remaining -= batch
        win_rate = red_wins / max(total, 1)
        self.matrix[(red_policy_id, blue_policy_id)] = win_rate
        self.matrix[(blue_policy_id, red_policy_id)] = 1.0 - win_rate
        self.pool.update_win_rate(red_policy_id, blue_policy_id, win_rate >= 0.5)
        self.pool.update_win_rate(blue_policy_id, red_policy_id, win_rate < 0.5)
        return win_rate

    def evaluate_all(self, env, trainers: dict):
        red_policies = [pid for pid, rec in self.pool.policies.items()
                        if rec.team == 0 and rec.is_active]
        blue_policies = [pid for pid, rec in self.pool.policies.items()
                         if rec.team == 1 and rec.is_active]
        for r_id in red_policies:
            for b_id in blue_policies:
                if (r_id, b_id) not in self.matrix:
                    r_trainer = trainers.get(r_id); b_trainer = trainers.get(b_id)
                    if r_trainer and b_trainer:
                        self.evaluate_pair(r_id, b_id, env, r_trainer, b_trainer)

    def get_submatrix(self, team: int) -> Tuple[np.ndarray, list, list]:
        own_policies = [pid for pid, rec in self.pool.policies.items()
                        if rec.team == team and rec.is_active]
        opp_policies = [pid for pid, rec in self.pool.policies.items()
                        if rec.team != team and rec.is_active]
        n_own, n_opp = len(own_policies), len(opp_policies)
        payoff = np.full((n_own, n_opp), 0.5)
        for i, own_id in enumerate(own_policies):
            for j, opp_id in enumerate(opp_policies):
                payoff[i, j] = self.matrix.get((own_id, opp_id), 0.5)
        return payoff, own_policies, opp_policies

    def get_fingerprints(self, policy_ids: List[str]) -> np.ndarray:
        K = len(policy_ids); F = np.full((K, 4), 0.25, dtype=np.float64)
        for i, pid in enumerate(policy_ids):
            if pid in self.fingerprints: F[i] = self.fingerprints[pid]
        return F

    def to_array(self) -> Tuple[np.ndarray, list]:
        active = [pid for pid, rec in self.pool.policies.items() if rec.is_active]
        n = len(active); mat = np.full((n, n), 0.5)
        for i, p1 in enumerate(active):
            for j, p2 in enumerate(active):
                if p1 != p2: mat[i, j] = self.matrix.get((p1, p2), 0.5)
        return mat, active


# =======================================================================
#  Section 7: Meta-Solvers  (Nash LP + TC-DAMS Frank-Wolfe)
# =======================================================================

def solve_nash(payoff_matrix: np.ndarray) -> np.ndarray:
    """2-player zero-sum Nash equilibrium via linear programming."""
    K, K_opp = payoff_matrix.shape
    if K == 0 or K_opp == 0: return np.array([])
    if K == 1: return np.array([1.0])
    n_vars = K + 1; c = np.zeros(n_vars); c[-1] = -1.0
    A_ub = np.zeros((K_opp, n_vars))
    for j in range(K_opp):
        A_ub[j, :K] = -payoff_matrix[:, j]; A_ub[j, K] = 1.0
    b_ub = np.zeros(K_opp)
    A_eq = np.zeros((1, n_vars)); A_eq[0, :K] = 1.0; b_eq = np.array([1.0])
    bounds = [(0, None)] * K + [(None, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
    if result.success:
        weights = np.maximum(result.x[:K], 0.0); weights /= weights.sum() + 1e-10
        return weights
    return np.ones(K) / K


def solve_uniform(K: int) -> np.ndarray:
    if K == 0: return np.array([])
    return np.ones(K) / K


def solve_rectified_nash(payoff_matrix: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    weights = solve_nash(payoff_matrix); weights[weights < threshold] = 0.0
    total = weights.sum()
    if total > 0: weights /= total
    else: weights = np.ones(len(weights)) / len(weights)
    return weights


def nash_conv(payoff_matrix: np.ndarray, meta_strategy: np.ndarray) -> float:
    if len(meta_strategy) == 0: return 0.0
    expected = payoff_matrix @ np.ones(payoff_matrix.shape[1]) / payoff_matrix.shape[1]
    br_value = expected.max(); current_value = (meta_strategy * expected).sum()
    return br_value - current_value


# ── TC-DAMS ──

def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64); mask = p > _EPS
    return float(-(p[mask] * np.log(p[mask])).sum())


def _entropy_grad_wrt_sigma(sigma: np.ndarray, F: np.ndarray) -> np.ndarray:
    F_bar = sigma @ F; safe = np.clip(F_bar, _EPS, 1.0)
    log_term = -(1.0 + np.log(safe))
    return F @ log_term


def _lp_nash_with_linear_bonus(payoff_matrix: np.ndarray, bonus: np.ndarray | None = None) -> np.ndarray:
    K, K_opp = payoff_matrix.shape
    if K == 0 or K_opp == 0: return np.array([])
    if K == 1: return np.array([1.0])
    n_vars = K + 1; c = np.zeros(n_vars); c[-1] = -1.0
    if bonus is not None: c[:K] = -bonus
    A_ub = np.zeros((K_opp, n_vars))
    for j in range(K_opp):
        A_ub[j, :K] = -payoff_matrix[:, j]; A_ub[j, K] = 1.0
    b_ub = np.zeros(K_opp)
    A_eq = np.zeros((1, n_vars)); A_eq[0, :K] = 1.0; b_eq = np.array([1.0])
    bounds = [(0.0, None)] * K + [(None, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
    if not result.success: return np.ones(K) / K
    weights = np.maximum(result.x[:K], 0.0); s = weights.sum()
    return weights / s if s > 0 else np.ones(K) / K


def solve_tc_dams(payoff_matrix: np.ndarray, fingerprints: np.ndarray | None = None,
                  lambda_diversity: float = 0.3, n_iters: int = 25,
                  tol: float = 1e-6) -> np.ndarray:
    K, K_opp = payoff_matrix.shape
    if K == 0 or K_opp == 0: return np.array([])
    if K == 1: return np.array([1.0])
    if fingerprints is None or lambda_diversity <= 0.0:
        return solve_nash(payoff_matrix)
    fingerprints = np.asarray(fingerprints, dtype=np.float64)
    if fingerprints.shape[0] != K: return solve_nash(payoff_matrix)
    row_sums = fingerprints.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > _EPS, row_sums, 1.0)
    F = fingerprints / row_sums
    sigma = solve_nash(payoff_matrix)
    if sigma.size == 0: return sigma
    for k in range(n_iters):
        grad_H = _entropy_grad_wrt_sigma(sigma, F)
        bonus = lambda_diversity * grad_H; bonus = bonus - bonus.mean()
        vertex = _lp_nash_with_linear_bonus(payoff_matrix, bonus=bonus)
        if vertex.size == 0: break
        step = 2.0 / (k + 2.0)
        new_sigma = (1.0 - step) * sigma + step * vertex
        if np.linalg.norm(new_sigma - sigma) < tol:
            sigma = new_sigma; break
        sigma = new_sigma
    sigma = np.maximum(sigma, 0.0); s = sigma.sum()
    return sigma / s if s > 0 else np.ones(K) / K


def task_fingerprint_entropy(sigma: np.ndarray, fingerprints: np.ndarray) -> float:
    if sigma.size == 0 or fingerprints.size == 0: return 0.0
    F = np.asarray(fingerprints, dtype=np.float64)
    if F.shape[0] != sigma.size: return 0.0
    F_bar = sigma @ F; F_bar = np.clip(F_bar, 0.0, None)
    s = F_bar.sum()
    return _entropy(F_bar / s) if s > 0 else 0.0


def effective_population_size(sigma: np.ndarray) -> float:
    if sigma.size == 0: return 0.0
    return float(np.exp(_entropy(sigma)))


# =======================================================================
#  Section 8: Elo-band Sampler
# =======================================================================

class EloBandSampler:
    """Elo-banded PFSP wrapper on top of OpponentPool."""

    def __init__(self, pool: OpponentPool, initial_elo: float = DEFAULT_ELO,
                 k_factor: float = DEFAULT_K, band_init: float = 400.0,
                 band_final: float = 100.0, anneal_iters: int = 15):
        self.pool = pool; self.initial_elo = initial_elo; self.k_factor = k_factor
        self.band_init = band_init; self.band_final = band_final
        self.anneal_iters = max(1, anneal_iters); self.elo: Dict[str, float] = {}

    def _get(self, policy_id: str) -> float:
        if policy_id not in self.elo: self.elo[policy_id] = self.initial_elo
        return self.elo[policy_id]

    def update_from_match(self, winner_id: str, loser_id: str, draw: bool = False):
        ra = self._get(winner_id); rb = self._get(loser_id)
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0)); eb = 1.0 - ea
        sa = 0.5 if draw else 1.0; sb = 0.5 if draw else 0.0
        self.elo[winner_id] = ra + self.k_factor * (sa - ea)
        self.elo[loser_id] = rb + self.k_factor * (sb - eb)

    def update_from_payoff_matrix(self, payoff: dict):
        for (a, b), w in payoff.items():
            if a not in self.pool.policies or b not in self.pool.policies: continue
            ra = self._get(a); rb = self._get(b)
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            self.elo[a] = ra + self.k_factor * (w - ea)
            self.elo[b] = rb + self.k_factor * ((1.0 - w) - (1.0 - ea))

    def current_band(self, iteration: int) -> float:
        if iteration >= self.anneal_iters: return self.band_final
        alpha = iteration / self.anneal_iters
        return (1.0 - alpha) * self.band_init + alpha * self.band_final

    def sample(self, current_policy_id: str, iteration: int, n_samples: int = 1) -> List[str]:
        if current_policy_id not in self.pool.policies: return []
        record = self.pool.policies[current_policy_id]
        all_opponents = self.pool.get_opponent_team(record.team)
        if not all_opponents: return []
        self_elo = self._get(current_policy_id); band = self.current_band(iteration)
        in_band = [opp for opp in all_opponents
                   if abs(self._get(opp) - self_elo) <= band]
        if not in_band:
            return self.pool.sample_pfsp(current_policy_id, n_samples=n_samples)
        win_rates = np.array([record.win_rates.get(opp, 0.5) for opp in in_band])
        loss_rates = 1.0 - win_rates
        if loss_rates.sum() < 1e-8:
            probs = np.ones(len(in_band)) / len(in_band)
        else:
            logits = loss_rates / max(self.pool.pfsp_temperature, 1e-6)
            logits = logits - logits.max(); probs = np.exp(logits)
            probs = probs / probs.sum()
        n = min(n_samples, len(in_band))
        idx = np.random.choice(len(in_band), size=n, replace=False, p=probs)
        return [in_band[i] for i in idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({k: float(v) for k, v in self.elo.items()}, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path): return
        with open(path) as f: data = json.load(f)
        self.elo = {k: float(v) for k, v in data.items()}


# =======================================================================
#  Section 9: FluxLeague Manager
# =======================================================================

class FluxLeague:
    """Full 3-role league training manager (AlphaStar-style PSRO)."""

    def __init__(self, n_elem: int = 625, n_pulses: int = 32, n_bins: int = 1024,
                 num_output_length: int = 16, n_teams: int = 2, population_cap: int = 20,
                 n_eval_games: int = 50, meta_solver: str = "nash",
                 pfsp_temperature: float = 1.0, exploiter_reset_prob: float = 0.1,
                 episodes_per_training: int = 1000, max_steps_per_episode: int = 1000,
                 checkpoint_dir: str = "checkpoints/league", device: str = "cuda",
                 sub_array_size: int = 0, tcdams_lambda: float = 0.3,
                 use_elo_band: bool = False, elo_band_init: float = 400.0,
                 elo_band_final: float = 100.0, elo_anneal_iters: int = 15,
                 buffer_size_commander: int = 2048, buffer_size_radar: int = 64,
                 commander_lr: float = 3e-4, radar_lr: float = 1e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 commander_clip: float = 0.2, radar_clip: float = 0.1,
                 commander_entropy: float = 0.01, radar_entropy: float = 0.02,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 n_epochs: int = 10, batch_size: int = 64):
        self.n_elem = n_elem; self.n_pulses = n_pulses; self.n_bins = n_bins
        self.num_output_length = num_output_length; self.n_teams = n_teams
        self.population_cap = population_cap; self.n_eval_games = n_eval_games
        self.meta_solver_name = meta_solver; self.sub_array_size = sub_array_size
        self.pfsp_temperature = pfsp_temperature
        self.exploiter_reset_prob = exploiter_reset_prob
        self.episodes_per_training = episodes_per_training
        self.max_steps_per_episode = max_steps_per_episode
        self.checkpoint_dir = checkpoint_dir; self.device = device
        self.ppo_config = dict(
            commander_lr=commander_lr, radar_lr=radar_lr, gamma=gamma,
            gae_lambda=gae_lambda, commander_clip=commander_clip, radar_clip=radar_clip,
            commander_entropy=commander_entropy, radar_entropy=radar_entropy,
            value_coef=value_coef, max_grad_norm=max_grad_norm,
            n_epochs=n_epochs, batch_size=batch_size,
            buffer_size_commander=buffer_size_commander,
            buffer_size_radar=buffer_size_radar,
            device=device)
        self.pool = OpponentPool(pool_dir=os.path.join(checkpoint_dir, "pool"),
                                 population_cap=population_cap,
                                 pfsp_temperature=pfsp_temperature)
        self.payoff = None; self.trainers: Dict[str, TeamPPOTrainer] = {}
        self.meta_strategies: Dict[int, np.ndarray] = {}
        self.tcdams_lambda = tcdams_lambda; self.use_elo_band = use_elo_band
        self.elo_sampler = EloBandSampler(
            self.pool, band_init=elo_band_init, band_final=elo_band_final,
            anneal_iters=elo_anneal_iters) if use_elo_band else None
        self.diag_history: list = []; self.iteration = 0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def initialize(self, env):
        self.payoff = PayoffMatrix(self.pool, self.n_eval_games, self.device,
                                   max_steps_per_game=self.max_steps_per_episode)
        for team in range(self.n_teams):
            for role in [ROLE_MAIN, ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER]:
                policy_dict = create_team_policy(
                    team=team, n_elem=self.n_elem, n_pulses=self.n_pulses,
                    n_bins=self.n_bins, num_output_length=self.num_output_length,
                    device=self.device, sub_array_size=self.sub_array_size)
                trainer = TeamPPOTrainer(
                    commander=policy_dict["commander"], radar=policy_dict["radar"],
                    **self.ppo_config)
                trainer.init_buffers(env.state_dim, env.action_dim)
                ckpt_name = f"{role}_team{team}_gen0.pt"
                ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
                trainer.save(ckpt_path)
                policy_id = self.pool.add_policy(
                    team=team, role=role, checkpoint_path=ckpt_path, generation=0)
                self.trainers[policy_id] = trainer

    def psro_iteration(self, env):
        metrics = {"iteration": self.iteration}; t0 = time.time()
        print(f"[League] Iteration {self.iteration}: Evaluating payoff matrix...")
        self.payoff.evaluate_all(env, self.trainers)
        if self.elo_sampler is not None:
            self.elo_sampler.update_from_payoff_matrix(self.payoff.matrix)
        iter_diag = {"iteration": self.iteration, "teams": {}}
        for team in range(self.n_teams):
            payoff_mat, own_ids, opp_ids = self.payoff.get_submatrix(team)
            F = self.payoff.get_fingerprints(own_ids)
            if self.meta_solver_name == "nash":
                sigma = solve_nash(payoff_mat)
            elif self.meta_solver_name == "rectified_nash":
                sigma = solve_rectified_nash(payoff_mat)
            elif self.meta_solver_name == "tc_dams":
                sigma = solve_tc_dams(payoff_mat, fingerprints=F,
                                      lambda_diversity=self.tcdams_lambda)
            else:
                K = len(own_ids); sigma = np.ones(K) / max(K, 1)
            self.meta_strategies[team] = sigma
            nc = nash_conv(payoff_mat, sigma)
            H_task = task_fingerprint_entropy(sigma, F)
            eff_K = effective_population_size(sigma)
            print(f"  Team {team} sigma={sigma.round(3)} NashConv={nc:.4f} "
                  f"H_task={H_task:.3f} effK={eff_K:.2f}")
            iter_diag["teams"][team] = dict(
                sigma=sigma.tolist(), nash_conv=float(nc),
                task_entropy=float(H_task), effective_K=float(eff_K),
                fingerprints=F.tolist(), own_ids=list(own_ids))
        self.diag_history.append(iter_diag)
        active_ids = list(self.trainers.keys())
        for policy_id in active_ids:
            trainer = self.trainers[policy_id]
            record = self.pool.policies[policy_id]
            if not record.is_active: continue
            if record.role == ROLE_MAIN:
                opponents = self._sample_opponents(policy_id, n_samples=1)
            elif record.role == ROLE_MAIN_EXPLOITER:
                opp_main = self.pool.get_active_main(1 - record.team)
                opponents = [opp_main] if opp_main else []
            elif record.role == ROLE_LEAGUE_EXPLOITER:
                opponents = self._sample_opponents(policy_id, n_samples=1)
            else:
                opponents = self.pool.sample_uniform(policy_id, n_samples=1)
            if not opponents: continue
            if record.role in [ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER]:
                if np.random.random() < self.exploiter_reset_prob:
                    self._maybe_reset(policy_id, trainer)
            opp_id = opponents[0]
            print(f"  Training {record.role} (team {record.team}, {policy_id}) "
                  f"against {opp_id}...")
            train_metrics = self._train_against(env, trainer, opp_id, policy_id)
            metrics[f"{policy_id}_train"] = train_metrics
            ckpt_name = f"{record.role}_team{record.team}_gen{self.iteration + 1}.pt"
            ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
            trainer.save(ckpt_path)
            new_id = self.pool.add_policy(
                team=record.team, role=record.role, checkpoint_path=ckpt_path,
                generation=self.iteration + 1, parent_id=policy_id)
            self.trainers[new_id] = trainer
        elapsed = time.time() - t0; metrics["elapsed_s"] = elapsed
        print(f"[League] Iteration {self.iteration} complete in {elapsed:.1f}s")
        self.iteration += 1; self.pool.save_metadata()
        return metrics

    def _train_against(self, env, trainer, opponent_id, own_policy_id):
        opp_trainer = self.trainers.get(opponent_id)
        record = self.pool.policies[own_policy_id]
        team = record.team; opp_team = 1 - team
        total_rewards = 0.0; wins = 0; episodes = 0
        for ep in range(self.episodes_per_training):
            env.reset(); episode_reward = 0.0
            for step in range(self.max_steps_per_episode):
                with torch.no_grad():
                    own = trainer.get_own_actions(env, team)
                    actions = torch.zeros(env.num_envs, env.n_radars, env.action_dim,
                                          device=self.device)
                    for i, r in enumerate(range(own["r_start"], own["r_end"])):
                        actions[:, r, :] = own["radar_actions"][i]
                    if opp_trainer:
                        opp = opp_trainer.get_own_actions(env, opp_team)
                        for i, r in enumerate(range(opp["r_start"], opp["r_end"])):
                            actions[:, r, :] = opp["radar_actions"][i]
                    else:
                        opp_r_start = opp_team * (env.n_radars // env.n_teams)
                        opp_r_end = opp_r_start + (env.n_radars // env.n_teams)
                        actions[:, opp_r_start:opp_r_end, :] = torch.rand(
                            env.num_envs, opp_r_end - opp_r_start, env.action_dim,
                            device=self.device)
                    commander_actions = torch.zeros(
                        env.num_envs, env.n_teams, env.battlefield.commander_action_dim,
                        device=self.device)
                    commander_actions[:, team, :] = own["commander_action"]
                    if opp_trainer:
                        commander_actions[:, opp_team, :] = opp["commander_action"]
                    else:
                        commander_actions[:, opp_team, :] = (
                            torch.rand(env.num_envs, env.battlefield.commander_action_dim,
                                       device=self.device) * 2 - 1)
                result = env.step(actions=actions, commander_actions=commander_actions)
                reward_info = trainer.store_transition(env, result, own["transition"], team)
                episode_reward += reward_info["radar_reward"][
                    :, own["r_start"]:own["r_end"]].sum().item()
                if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
                   (trainer.radar_buffer and trainer.radar_buffer.near_full):
                    trainer.update()
                if result["dones"].any():
                    if result["winners"][0] == team: wins += 1
                    break
            total_rewards += episode_reward; episodes += 1
            if episodes % 10 == 0: trainer.update()
        return {"episodes": episodes, "win_rate": wins / max(episodes, 1),
                "avg_reward": total_rewards / max(episodes, 1)}

    def _maybe_reset(self, policy_id: str, trainer: TeamPPOTrainer):
        record = self.pool.policies[policy_id]
        if record.parent_id and record.parent_id in self.pool.policies:
            parent = self.pool.policies[record.parent_id]
            if os.path.exists(parent.checkpoint_path):
                print(f"  Resetting {policy_id} to parent checkpoint")
                trainer.load(parent.checkpoint_path)

    def _sample_opponents(self, policy_id: str, n_samples: int = 1) -> list:
        if self.elo_sampler is not None:
            return self.elo_sampler.sample(policy_id, iteration=self.iteration,
                                           n_samples=n_samples)
        return self.pool.sample_pfsp(policy_id, n_samples=n_samples)

    def get_final_agent(self, team: int) -> str:
        if team in self.meta_strategies:
            own_ids = [pid for pid, rec in self.pool.policies.items()
                       if rec.team == team and rec.is_active]
            weights = self.meta_strategies[team]
            if len(own_ids) == len(weights):
                best_idx = np.argmax(weights); return own_ids[best_idx]
        return self.pool.get_active_main(team)

    def save(self):
        state = {"iteration": self.iteration,
                 "meta_strategies": {k: v.tolist() for k, v in self.meta_strategies.items()},
                 "ppo_config": self.ppo_config, "tcdams_lambda": self.tcdams_lambda,
                 "use_elo_band": self.use_elo_band, "meta_solver_name": self.meta_solver_name}
        torch.save(state, os.path.join(self.checkpoint_dir, "league_state.pt"))
        self.pool.save_metadata()
        if self.elo_sampler is not None:
            self.elo_sampler.save(os.path.join(self.checkpoint_dir, "elo.json"))
        try:
            with open(os.path.join(self.checkpoint_dir, "diag_history.json"), "w") as f:
                json.dump(self.diag_history, f, indent=2)
        except Exception as exc:
            print(f"[League] WARN: failed to write diag_history: {exc}")

    def load(self):
        state_path = os.path.join(self.checkpoint_dir, "league_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.iteration = state["iteration"]
            self.meta_strategies = {int(k): np.array(v) for k, v
                                    in state["meta_strategies"].items()}
        self.pool.load_metadata()
        if self.elo_sampler is not None:
            self.elo_sampler.load(os.path.join(self.checkpoint_dir, "elo.json"))


# =======================================================================
#  Section 10: Phased Trainer  (4-phase curriculum orchestrator)
# =======================================================================

class PhasedTrainer:
    """Orchestrates the 4-phase curriculum for FluxLeague training."""

    def __init__(self, env_factory, league: FluxLeague, phase_a_episodes: int = 5000,
                 phase_b_episodes: int = 10000, phase_c_iterations: int = 20,
                 phase_c_episodes_per_iter: int = 1000, phase_d_episodes: int = 10000,
                 log_dir: str = "logs/curriculum", device: str = "cuda"):
        self.env_factory = env_factory; self.league = league
        self.phase_a_episodes = phase_a_episodes; self.phase_b_episodes = phase_b_episodes
        self.phase_c_iterations = phase_c_iterations
        self.phase_c_episodes_per_iter = phase_c_episodes_per_iter
        self.phase_d_episodes = phase_d_episodes; self.log_dir = log_dir
        self.device = device; os.makedirs(log_dir, exist_ok=True)

    def run_all(self):
        import gc
        for name, method in [("A", self.run_phase_a), ("B", self.run_phase_b),
                              ("C", self.run_phase_c), ("D", self.run_phase_d)]:
            print("=" * 60); print(f"Phase {name}"); print("=" * 60); method()
            gc.collect(); torch.cuda.empty_cache()
        self.league.save(); print("\nTraining complete. Final league saved.")

    def run_phase_a(self):
        env = self.env_factory()
        tasks = [0, 1, 2]; task_names = ["recon", "detect", "jam"]
        for task_idx, task_name in zip(tasks, task_names):
            print(f"\n  Pre-training task: {task_name}")
            policy = create_team_policy(
                0, device=self.device, n_elem=self.league.n_elem,
                n_pulses=self.league.n_pulses, n_bins=self.league.n_bins,
                num_output_length=self.league.num_output_length,
                sub_array_size=self.league.sub_array_size)
            trainer = TeamPPOTrainer(commander=policy["commander"],
                                     radar=policy["radar"], **self.league.ppo_config)
            trainer.init_buffers(env.state_dim, env.action_dim)
            max_steps = getattr(self.league, "max_steps_per_episode", 1000)
            for ep in range(self.phase_a_episodes // 3):
                env.reset()
                for step in range(max_steps):
                    with torch.no_grad():
                        state, commander_obs = trainer._get_observations(env)
                        actions = torch.zeros(env.num_envs, env.n_radars, env.action_dim,
                                              device=self.device)
                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim, device=self.device)
                        for team in range(env.n_teams):
                            cmd_obs = commander_obs[:, team, :]
                            cmd_act, _, _ = trainer.commander_trainer.ac.get_action(cmd_obs)
                            commander_actions[:, team, :] = cmd_act
                        rep_obs = rep_action = None
                        for r in range(env.n_radars):
                            radar_obs = state[:, r, :]
                            radar_act, _, _ = trainer.radar_trainer.ac.get_action(radar_obs)
                            for e in range(env.n_elem):
                                base = e * 22
                                radar_act[:, base:base + 4].zero_()
                                radar_act[:, base + task_idx] = 1.0
                            actions[:, r, :] = radar_act
                            if r == 0: rep_obs = radar_obs; rep_action = radar_act.clone()
                    result = env.step(actions=actions, commander_actions=commander_actions)
                    with torch.no_grad():
                        rep_logp, _, rep_val = trainer.radar_trainer.ac.evaluate_actions(
                            rep_obs, rep_action)
                        cmd_obs_t0 = commander_obs[:, 0, :]
                        cmd_act_t0 = commander_actions[:, 0, :]
                        cmd_logp, _, cmd_val = trainer.commander_trainer.ac.evaluate_actions(
                            cmd_obs_t0, cmd_act_t0)
                    transition = {"cmd_obs": cmd_obs_t0, "cmd_action": cmd_act_t0,
                                  "cmd_logp": cmd_logp, "cmd_val": cmd_val.squeeze(-1),
                                  "radar_obs": rep_obs, "radar_action": rep_action,
                                  "radar_logp": rep_logp, "radar_val": rep_val.squeeze(-1)}
                    trainer.store_transition(env, result, transition, team=0)
                    if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
                       (trainer.radar_buffer and trainer.radar_buffer.near_full):
                        trainer.update()
                    if result["dones"].any(): break
                if ep % 5 == 0:
                    metrics = trainer.update()
                    print(f"    Episode {ep}: {metrics}")
            ckpt_path = os.path.join(self.league.checkpoint_dir, f"pretrain_{task_name}.pt")
            trainer.save(ckpt_path)

    def run_phase_b(self):
        env = self.env_factory(); self.league.initialize(env)
        for policy_id, trainer in self.league.trainers.items():
            record = self.league.pool.policies[policy_id]
            if not record.is_active: continue
            team = record.team; opp_team = 1 - team
            r_per_team = env.n_radars // env.n_teams
            opp_r_start = opp_team * r_per_team; opp_r_end = opp_r_start + r_per_team
            print(f"\n  Training {record.role} team {team} ({policy_id})")
            wins = 0
            for ep in range(self.phase_b_episodes):
                env.reset()
                for step in range(getattr(self.league, "max_steps_per_episode", 1000)):
                    with torch.no_grad():
                        own = trainer.get_own_actions(env, team)
                        actions = torch.zeros(env.num_envs, env.n_radars, env.action_dim,
                                              device=self.device)
                        for i, r in enumerate(range(own["r_start"], own["r_end"])):
                            actions[:, r, :] = own["radar_actions"][i]
                        actions[:, opp_r_start:opp_r_end, :] = torch.rand(
                            env.num_envs, opp_r_end - opp_r_start, env.action_dim,
                            device=self.device)
                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim, device=self.device)
                        commander_actions[:, team, :] = own["commander_action"]
                        commander_actions[:, opp_team, :] = (
                            torch.rand(env.num_envs, env.battlefield.commander_action_dim,
                                       device=self.device) * 2 - 1)
                    result = env.step(actions=actions, commander_actions=commander_actions)
                    trainer.store_transition(env, result, own["transition"], team)
                    if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
                       (trainer.radar_buffer and trainer.radar_buffer.near_full):
                        trainer.update()
                    if result["dones"].any():
                        if result["winners"][0] == team: wins += 1
                        break
                if ep > 0 and ep % 10 == 0: trainer.update()
                if ep % 500 == 0:
                    wr = wins / max(ep + 1, 1)
                    print(f"    Episode {ep}: win_rate={wr:.3f}")
            ckpt_path = os.path.join(self.league.checkpoint_dir,
                                     f"{record.role}_team{team}_phaseB.pt")
            trainer.save(ckpt_path)

    def run_phase_c(self):
        env = self.env_factory()
        for it in range(self.phase_c_iterations):
            print(f"\n  PSRO Iteration {it}/{self.phase_c_iterations}")
            metrics = self.league.psro_iteration(env)
            log_path = os.path.join(self.log_dir, f"psro_iter_{it:03d}.json")
            with open(log_path, "w") as f: json.dump(metrics, f, indent=2, default=str)
            if it % 5 == 0: self.league.save()

    def run_phase_d(self):
        env = self.env_factory()
        for policy_id, trainer in self.league.trainers.items():
            record = self.league.pool.policies[policy_id]
            if not record.is_active: continue
            if record.role not in [ROLE_MAIN_EXPLOITER, ROLE_LEAGUE_EXPLOITER]: continue
            team = record.team; opp_team = 1 - team
            r_per_team = env.n_radars // env.n_teams
            opp_r_start = opp_team * r_per_team; opp_r_end = opp_r_start + r_per_team
            opp_main_id = self.league.pool.get_active_main(opp_team)
            opp_trainer = self.league.trainers.get(opp_main_id) if opp_main_id else None
            print(f"\n  Refining {record.role} team {team} ({policy_id})")
            wins = 0
            for ep in range(self.phase_d_episodes):
                env.reset()
                for step in range(getattr(self.league, "max_steps_per_episode", 1000)):
                    with torch.no_grad():
                        own = trainer.get_own_actions(env, team)
                        actions = torch.zeros(env.num_envs, env.n_radars, env.action_dim,
                                              device=self.device)
                        for i, r in enumerate(range(own["r_start"], own["r_end"])):
                            actions[:, r, :] = own["radar_actions"][i]
                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim, device=self.device)
                        commander_actions[:, team, :] = own["commander_action"]
                        if opp_trainer:
                            opp = opp_trainer.get_own_actions(env, opp_team)
                            for i, r in enumerate(range(opp["r_start"], opp["r_end"])):
                                actions[:, r, :] = opp["radar_actions"][i]
                            commander_actions[:, opp_team, :] = opp["commander_action"]
                        else:
                            actions[:, opp_r_start:opp_r_end, :] = torch.rand(
                                env.num_envs, opp_r_end - opp_r_start, env.action_dim,
                                device=self.device)
                            commander_actions[:, opp_team, :] = (
                                torch.rand(env.num_envs, env.battlefield.commander_action_dim,
                                           device=self.device) * 2 - 1)
                    result = env.step(actions=actions, commander_actions=commander_actions)
                    trainer.store_transition(env, result, own["transition"], team)
                    if (trainer.commander_buffer and trainer.commander_buffer.near_full) or \
                       (trainer.radar_buffer and trainer.radar_buffer.near_full):
                        trainer.update()
                    if result["dones"].any():
                        if result["winners"][0] == team: wins += 1
                        break
                if ep > 0 and ep % 10 == 0: trainer.update()
                if ep % 500 == 0:
                    wr = wins / max(ep + 1, 1)
                    print(f"    Episode {ep}: win_rate={wr:.3f}")
        print("\n  Final evaluation..."); self._final_evaluation(env)

    def _final_evaluation(self, env):
        r_per_team = env.n_radars // env.n_teams
        for team in range(self.league.n_teams):
            agent_id = self.league.get_final_agent(team)
            trainer = self.league.trainers.get(agent_id)
            if not trainer:
                print(f"\n  Team {team} final agent: {agent_id} (no trainer)"); continue
            opp_team = 1 - team
            opp_r_start = opp_team * r_per_team; opp_r_end = opp_r_start + r_per_team
            print(f"\n  Team {team} final agent: {agent_id}")
            wins = 0; n_games = 100
            for game in range(n_games):
                env.reset()
                for step in range(getattr(self.league, "max_steps_per_episode", 1000)):
                    with torch.no_grad():
                        own = trainer.get_own_actions(env, team, deterministic=True)
                        actions = torch.zeros(env.num_envs, env.n_radars, env.action_dim,
                                              device=self.device)
                        for i, r in enumerate(range(own["r_start"], own["r_end"])):
                            actions[:, r, :] = own["radar_actions"][i]
                        actions[:, opp_r_start:opp_r_end, :] = torch.rand(
                            env.num_envs, opp_r_end - opp_r_start, env.action_dim,
                            device=self.device)
                        commander_actions = torch.zeros(
                            env.num_envs, env.n_teams,
                            env.battlefield.commander_action_dim, device=self.device)
                        commander_actions[:, team, :] = own["commander_action"]
                        commander_actions[:, opp_team, :] = (
                            torch.rand(env.num_envs, env.battlefield.commander_action_dim,
                                       device=self.device) * 2 - 1)
                    result = env.step(actions=actions, commander_actions=commander_actions)
                    if result["dones"].any():
                        if result["winners"][0] == team: wins += 1
                        break
            print(f"  Win rate: {wins}/{n_games} = {wins / n_games:.2%}")


# =======================================================================
#  Section 11: CLI Entry Point
# =======================================================================

def main():
    # Ensure cwd is this script's directory so relative checkpoint paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description="FluxPhased League Training (consolidated single-file)")
    parser.add_argument("--cells", nargs="+", default=["R0"],
                        choices=list(ABLATION_CELLS.keys()),
                        help="Ablation cells to run (default: R0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["a", "b", "c", "d", "all"],
                        help="Run specific phase only (default: all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    for cell in args.cells:
        cell_spec = ABLATION_CELLS[cell]
        ckpt_dir = f"checkpoints/league_{cell}_seed{args.seed}"
        print(f"\n{'='*60}")
        print(f"Running ablation cell {cell}: {cell_spec['description']}")
        print(f"Checkpoint: {ckpt_dir} | Seed: {args.seed} | Phase: {args.phase}")
        print(f"{'='*60}")

        def make_env():
            return MFARVecEnv(
                num_envs=ENV_DEFAULTS["num_envs"],
                n_radars=ENV_DEFAULTS["n_radars"],
                rows=ENV_DEFAULTS["rows"],
                cols=ENV_DEFAULTS["cols"],
                pulses_per_cpi=ENV_DEFAULTS["pulses_per_cpi"],
                fft_size=ENV_DEFAULTS["fft_size"],
                device=args.device,
                tx_power_w=ENV_DEFAULTS["tx_power_w"],
                cpi_preallocate=ENV_DEFAULTS["cpi_preallocate"],
            )

        env = make_env()
        n_bins = env.n_bins

        league = FluxLeague(
            n_elem=env.n_elem, n_pulses=env.n_pulses, n_bins=n_bins,
            num_output_length=env.num_output_length, n_teams=env.n_teams,
            population_cap=LEAGUE_DEFAULTS["population_cap"],
            n_eval_games=LEAGUE_DEFAULTS["n_eval_games"],
            meta_solver=cell_spec["meta_solver"],
            pfsp_temperature=LEAGUE_DEFAULTS["pfsp_temperature"],
            exploiter_reset_prob=LEAGUE_DEFAULTS["exploiter_reset_prob"],
            episodes_per_training=LEAGUE_DEFAULTS["episodes_per_training"],
            max_steps_per_episode=LEAGUE_DEFAULTS["max_steps_per_episode"],
            checkpoint_dir=ckpt_dir, device=args.device, sub_array_size=5,
            tcdams_lambda=cell_spec["tcdams_lambda"],
            use_elo_band=cell_spec["use_elo_band"],
            elo_band_init=400.0, elo_band_final=100.0, elo_anneal_iters=6,
            commander_lr=PPO_DEFAULTS["commander_lr"],
            radar_lr=PPO_DEFAULTS["radar_lr"],
            gamma=PPO_DEFAULTS["gamma"],
            gae_lambda=PPO_DEFAULTS["gae_lambda"],
            commander_clip=PPO_DEFAULTS["commander_clip"],
            radar_clip=PPO_DEFAULTS["radar_clip"],
            commander_entropy=PPO_DEFAULTS["commander_entropy"],
            radar_entropy=PPO_DEFAULTS["radar_entropy"],
            value_coef=PPO_DEFAULTS["value_coef"],
            max_grad_norm=PPO_DEFAULTS["max_grad_norm"],
            n_epochs=PPO_DEFAULTS["n_epochs"],
            batch_size=PPO_DEFAULTS["batch_size"],
            buffer_size_commander=PPO_DEFAULTS["buffer_size_commander"],
            buffer_size_radar=PPO_DEFAULTS["buffer_size_radar"],
        )

        trainer = PhasedTrainer(
            env_factory=make_env, league=league,
            phase_a_episodes=CURRICULUM_DEFAULTS["phase_a_episodes"],
            phase_b_episodes=CURRICULUM_DEFAULTS["phase_b_episodes"],
            phase_c_iterations=CURRICULUM_DEFAULTS["phase_c_iterations"],
            phase_c_episodes_per_iter=CURRICULUM_DEFAULTS["phase_c_episodes_per_iter"],
            phase_d_episodes=CURRICULUM_DEFAULTS["phase_d_episodes"],
            device=args.device,
        )

        if args.phase == "a":
            trainer.run_phase_a()
        elif args.phase == "b":
            trainer.run_phase_b()
        elif args.phase == "c":
            if not os.path.exists(os.path.join(ckpt_dir, "pool", "pool_metadata.json")):
                print("[init] Phase C needs initialized league; running Phase B first.")
                trainer.run_phase_b()
            trainer.run_phase_c()
        elif args.phase == "d":
            trainer.run_phase_d()
        else:
            trainer.run_all()

    print("\nAll cells complete.")


if __name__ == "__main__":
    main()
