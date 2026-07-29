"""Supervised imitation for G3-BSTA-lite (F3, Gate 2).

Per MODIFICATION_PLAN W4 + Gate 2:
  - Generate ~10k planner-development samples + ~2k held-out samples.
  - Labels are actions produced by the executable clairvoyant oracle
    (which is at least as strong as the causal witness).
  - Train a small masked-categorical actor.
  - Verify on held-out:
      * mask-valid actions: 100%
      * tie-aware top-1 accuracy: >= 90%
      * normalized oracle regret: <= 10%
      * held-out rollouts recover >= 90% of witness-vs-random gap

Labels use only actions available to the causal witness (same obs
channels, same mask), proving the observation/action representation can
express the witness without privileged inputs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.gpu.g3_bsta_lite import (
    EnvConfig,
    G3BstaLiteVecEnv,
    N_ACTIONS,
    generate_paired_manifest,
)
from .baselines import CausalReactiveOrEDF


@dataclass
class ImitationSample:
    obs: torch.Tensor       # [OBS_DIM]
    mask: torch.Tensor      # [N_ACTIONS]
    action: torch.Tensor    # scalar int64


@dataclass
class ImitationDataset:
    obs: torch.Tensor         # [N, OBS_DIM]
    mask: torch.Tensor        # [N, N_ACTIONS]
    action: torch.Tensor      # [N]
    scenario_seed: torch.Tensor  # [N] for grouped eval


def generate_imitation_dataset(
    *,
    n_scenarios: int,
    cfg: EnvConfig,
    seed_offset: int = 0,
) -> ImitationDataset:
    """Run causal witness on `n_scenarios`; collect (obs, mask, action).

    Per MODIFICATION_PLAN W4: "Labels may use only actions available to
    the causal witness." We label with the witness (CausalReactiveOrEDF)
    itself, which consumes only actor-visible channels. This makes the
    Gate 2 question well-posed: can a small MLP express the witness from
    the same observation?

    Each scenario produces `cfg.horizon` samples. Returns stacked tensors.
    """
    manifest = generate_paired_manifest(
        base_seed=20260729 + seed_offset,
        n_scenarios=n_scenarios,
        horizon=cfg.horizon,
        n_services=cfg.n_services,
        arrival_rate_per_service=cfg.arrival_rate_per_service,
        baseline_snr_db=cfg.baseline_snr_db,
        device=cfg.device,
    )

    obs_list, mask_list, act_list, seed_list = [], [], [], []
    for sce in manifest:
        env = G3BstaLiteVecEnv(cfg)
        witness = CausalReactiveOrEDF()
        env.reset(seed=sce.seed)
        witness.reset(env, seed=sce.seed)
        for t in range(cfg.horizon):
            obs = env._build_observation()           # [E, OBS_DIM]
            mask = env._compute_mask()               # [E, N_ACTIONS]
            actions = witness.act(obs, mask, step_idx=t)
            for e in range(env.E):
                obs_list.append(obs[e].clone())
                mask_list.append(mask[e].clone())
                act_list.append(actions[e].clone())
                seed_list.append(torch.tensor(sce.seed, dtype=torch.int64))
            env.step(actions)

    return ImitationDataset(
        obs=torch.stack(obs_list),
        mask=torch.stack(mask_list),
        action=torch.stack(act_list),
        scenario_seed=torch.stack(seed_list),
    )


class ImitationActor(nn.Module):
    """Masked-categorical actor: obs [OBS_DIM] -> logits [N_ACTIONS].

    Architecture: 2 x 128 Tanh MLP (matches MODIFICATION_PLAN W5 actor
    spec, used here for supervised pre-training).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(obs))
        h = torch.tanh(self.fc2(h))
        return self.head(h)

    def log_prob(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Log-prob with mask: illegal actions get -inf logits."""
        logits = self.forward(obs)
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return F.log_softmax(logits, dim=-1)


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 3e-4
    epochs: int = 30
    seed: int = 0
    device: str = "cpu"


def train_imitation(
    train: ImitationDataset,
    model: ImitationActor,
    cfg: TrainConfig,
) -> dict:
    """Train actor with cross-entropy on oracle actions (mask-aware).

    Returns metrics dict.
    """
    torch.manual_seed(cfg.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    N = train.obs.shape[0]
    idx = torch.randperm(N)
    history = []
    for epoch in range(cfg.epochs):
        # Shuffle and batch.
        perm = idx[torch.randperm(N)]
        total_loss = 0.0
        n_batches = 0
        correct = 0
        for s in range(0, N, cfg.batch_size):
            bi = perm[s:s + cfg.batch_size]
            obs = train.obs[bi]
            mask = train.mask[bi]
            act = train.action[bi]
            logp = model.log_prob(obs, mask)
            loss = F.nll_loss(logp, act.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
            pred = logp.argmax(dim=-1)
            correct += int((pred == act.long()).sum().item())
        avg_loss = total_loss / max(1, n_batches)
        acc = correct / N
        history.append({"epoch": epoch, "loss": avg_loss, "acc": acc})
    return {"history": history, "final_loss": history[-1]["loss"], "final_acc": history[-1]["acc"]}


def evaluate_imitation(
    model: ImitationActor,
    ds: ImitationDataset,
    *,
    batch_size: int = 256,
) -> dict:
    """Compute held-out metrics:
      - mask_valid: P(predicted action is in mask)
      - top1_tie_aware: P(predicted in {tied best actions set})

    Tie-aware: an action is "correct" if its logit-prob equals the oracle
    action's prob (within epsilon), AND the prediction is in the tied set.
    For supervised learning from a deterministic oracle, ties are rare
    (oracle picks a single action); tie-aware thus reduces to standard
    top-1.
    """
    model.eval()
    N = ds.obs.shape[0]
    correct = 0
    mask_violations = 0
    with torch.no_grad():
        for s in range(0, N, batch_size):
            obs = ds.obs[s:s + batch_size]
            mask = ds.mask[s:s + batch_size]
            act = ds.action[s:s + batch_size].long()
            logits = model.forward(obs)
            # Mask-aware argmax.
            masked_logits = logits.masked_fill(~mask.bool(), float("-inf"))
            pred = masked_logits.argmax(dim=-1)
            correct += int((pred == act).sum().item())
            mask_violations += int((~mask[torch.arange(len(pred)), pred]).sum().item())
    return {
        "n": N,
        "top1_acc": correct / N,
        "mask_valid": 1.0 - (mask_violations / N),
    }


def rollout_with_dagger_labels(
    *,
    model: ImitationActor | None,
    witness_factory: Callable,
    cfg: EnvConfig,
    n_scenarios: int,
    seed_offset: int,
    device: str,
    mix_witness_prob: float = 0.5,
):
    """Roll out (model or witness) on scenarios; label every visited
    (obs, mask) with the witness action.

    ``mix_witness_prob`` fraction of steps use the witness action (so the
    trajectory covers near-on-distribution obs); the rest use the model
    action (covers off-distribution obs for DAgger).
    """
    manifest = generate_paired_manifest(
        base_seed=20260729 + seed_offset,
        n_scenarios=n_scenarios,
        horizon=cfg.horizon,
        n_services=cfg.n_services,
        arrival_rate_per_service=cfg.arrival_rate_per_service,
        baseline_snr_db=cfg.baseline_snr_db,
        device=device,
    )
    obs_list, mask_list, act_list, seed_list = [], [], [], []
    for sce in manifest:
        env = G3BstaLiteVecEnv(cfg)
        witness = witness_factory()
        env.reset(seed=sce.seed)
        witness.reset(env, seed=sce.seed)
        for t in range(cfg.horizon):
            obs = env._build_observation()
            mask = env._compute_mask()
            wit_a = witness.act(obs, mask, step_idx=t)
            if model is None:
                a = wit_a
            else:
                # Coin flip per env: witness or model.
                with torch.no_grad():
                    logits = model.forward(obs)
                    masked_logits = logits.masked_fill(~mask.bool(), float("-inf"))
                    model_a = masked_logits.argmax(dim=-1)
                use_wit = torch.rand(env.E) < mix_witness_prob
                a = torch.where(use_wit, wit_a, model_a)
            for e in range(env.E):
                obs_list.append(obs[e].clone())
                mask_list.append(mask[e].clone())
                act_list.append(wit_a[e].clone())  # witness action as label
                seed_list.append(torch.tensor(sce.seed, dtype=torch.int64))
            env.step(a)
    return ImitationDataset(
        obs=torch.stack(obs_list),
        mask=torch.stack(mask_list),
        action=torch.stack(act_list),
        scenario_seed=torch.stack(seed_list),
    )


def train_imitation_dagger(
    *,
    cfg: EnvConfig,
    n_initial_scenarios: int = 128,
    n_dagger_rounds: int = 3,
    n_dagger_scenarios_per_round: int = 64,
    epochs_per_round: int = 25,
    seed: int = 0,
    device: str = "cpu",
    hidden: int = 128,
) -> tuple[ImitationActor, dict]:
    """DAgger: iteratively collect data with current model + witness
    mixture, label with witness, retrain. Fixes covariate shift so the
    actor imitates the witness under its own trajectory distribution.
    """
    torch.manual_seed(seed)
    model = ImitationActor(obs_dim=11, n_actions=N_ACTIONS, hidden=hidden)
    from .baselines import CausalReactiveOrEDF
    wf = CausalReactiveOrEDF

    history = {"rounds": []}
    # Initial dataset: pure witness rollout.
    dev = generate_imitation_dataset(
        n_scenarios=n_initial_scenarios, cfg=cfg, seed_offset=0,
    )
    all_obs, all_mask, all_act = [dev.obs], [dev.mask], [dev.action]

    for r in range(n_dagger_rounds + 1):
        train_ds = ImitationDataset(
            obs=torch.cat(all_obs),
            mask=torch.cat(all_mask),
            action=torch.cat(all_act),
            scenario_seed=torch.zeros(torch.cat(all_obs).shape[0]),
        )
        tcfg = TrainConfig(batch_size=256, lr=3e-4, epochs=epochs_per_round,
                            seed=seed + r, device=device)
        info = train_imitation(train_ds, model, tcfg)
        history["rounds"].append({
            "round": r, "train_size": train_ds.obs.shape[0],
            "final_loss": info["final_loss"], "final_acc": info["final_acc"],
        })

        if r == n_dagger_rounds:
            break

        # Collect more data using current model + witness mixture.
        ds = rollout_with_dagger_labels(
            model=model, witness_factory=wf, cfg=cfg,
            n_scenarios=n_dagger_scenarios_per_round,
            seed_offset=2000 + r * 100, device=device,
            mix_witness_prob=0.5,
        )
        all_obs.append(ds.obs); all_mask.append(ds.mask); all_act.append(ds.action)

    return model, history


def rollout_imitation(
    model: ImitationActor,
    *,
    cfg: EnvConfig,
    n_scenarios: int,
    seed_offset: int = 0,
    device: str = "cpu",
) -> float:
    """Macro-mean drop_ratio of the imitation actor over `n_scenarios`."""
    manifest = generate_paired_manifest(
        base_seed=20260729 + seed_offset,
        n_scenarios=n_scenarios,
        horizon=cfg.horizon,
        n_services=cfg.n_services,
        arrival_rate_per_service=cfg.arrival_rate_per_service,
        baseline_snr_db=cfg.baseline_snr_db,
        device=device,
    )
    drops = []
    model.eval()
    with torch.no_grad():
        for sce in manifest:
            env = G3BstaLiteVecEnv(cfg)
            env.reset(seed=sce.seed)
            for t in range(cfg.horizon):
                obs = env._build_observation()
                mask = env._compute_mask()
                logits = model.forward(obs)
                masked_logits = logits.masked_fill(~mask.bool(), float("-inf"))
                action = masked_logits.argmax(dim=-1)
                env.step(action)
            drops.append(float(env.drop_ratio()[0]))
    return sum(drops) / len(drops) if drops else float("nan")
