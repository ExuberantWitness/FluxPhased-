"""Behavioral-cloning pretrainer for two-team learning commander (WP1 BC → PPO).

Implements the AlphaStar SL → RL paradigm (Vinyals 2019, Nature):
  1. Use StrongRule as expert, collect (obs, priv, action) triples
  2. NLL supervised training on AC's multi-head distribution
  3. Hand off BC-pretrained AC to PPO fine-tune (TwoTeamBRTrainer, unchanged)

Why BC:
  on-policy PPO from random init in 13 continuous + 5 discrete action space must
  simultaneously "learn to play" AND "find exploit" — the former eats most samples.
  BC starts PPO at rule's local optimum, freeing PPO to explore exploit structure.

Loss: NLL via existing ac.evaluate_actions (multi-head log density already correct).
  task_alloc   Dirichlet
  beam_target  Categorical
  laser_target Categorical
  emission_on  Bernoulli
  freq_hop     Beta (with [1, freq_hop_max] → [0,1] inverse rescale in evaluate_actions)

Critic NOT pretrained — PPO fine-tune learns value from scratch via GAE.
"""

from __future__ import annotations
import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Optional

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions, STRATEGIES
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic


class TwoTeamBCPretrainer:
    """BC pretrainer: collect expert demos from StrongRule, NLL-fit AC's actor."""

    def __init__(
        self,
        ac: TwoTeamCommanderActorCritic,
        lr: float = 1e-3,
        batch_size: int = 256,
        val_split: float = 0.1,
        device: str = "cuda",
    ):
        self.ac = ac
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.val_split = float(val_split)
        self.device = torch.device(device)
        # Only actor params — critic left for PPO fine-tune
        actor_params = (
            list(self.ac.actor_trunk.parameters())
            + list(self.ac.task_alloc_head.parameters())
            + list(self.ac.beam_target_head.parameters())
            + list(self.ac.laser_target_head.parameters())
            + list(self.ac.emission_on_head.parameters())
            + list(self.ac.freq_hop_head.parameters())
            + list(self.ac.channel_select_head.parameters())
        )
        self.opt = torch.optim.Adam(actor_params, lr=self.lr)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def collect_samples(
        self,
        env,
        rule,
        n_samples: int = 50000,
        opponent_strategies: Optional[List[str]] = None,
        episode_steps: int = 200,
        base_seed: int = 1000,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run rule vs ExtremeCommander episodes, collect (obs, priv, rule_action).

        Per-episode:
          - Pick an ExtremeCommander opponent (cycling through diversified pool)
          - rule plays team 0 (collect rule's transitions), then team 1 (symmetric aug)
          - Each step records (obs[:, team], priv[:, team], rule_action) per env

        Stop when total samples >= n_samples.
        """
        if opponent_strategies is None:
            opponent_strategies = [
                "pure_track", "pure_jam", "pure_comm", "pure_detect",
                "balanced", "balanced_jam_heavy", "track_agile",
            ]

        E = env.E
        R = env.n_radars_per_team
        n_fn = env.n_fn
        obs_dim = env.obs_dim
        priv_dim = env.privileged_dim

        obs_buf = []
        priv_buf = []
        task_alloc_buf = []
        beam_target_buf = []
        laser_target_buf = []
        emission_on_buf = []
        freq_hop_rate_buf = []
        channel_select_buf = []   # WP-C R3: from env state (rule doesn't output)

        total = 0
        ep = 0
        t0 = time.time()
        while total < n_samples:
            opp_name = opponent_strategies[ep % len(opponent_strategies)]
            opp = STRATEGIES[opp_name]

            # Phase 1: rule = team 0
            env.seed = base_seed + ep * 2
            env._reset_count = ep * 2
            env.reset()
            for step in range(episode_steps):
                obs_dict = env.get_obs()
                a_rule = rule.get_action(env, 0)
                a_opp = opp.get_action(env, 1)
                # Record rule's transition (obs BEFORE step)
                obs_buf.append(obs_dict["obs"][:, 0].clone())
                priv_buf.append(obs_dict["privileged"][:, 0].clone())
                task_alloc_buf.append(a_rule["task_alloc"].clone())
                beam_target_buf.append(a_rule["beam_target"].clone())
                laser_target_buf.append(a_rule["laser_target"].clone())
                emission_on_buf.append(a_rule["emission_on"].clone())
                freq_hop_rate_buf.append(a_rule["freq_hop_rate"].clone())
                # WP-C R3: rule doesn't output channel_select — derive from env state
                # (wrapper-set orthogonal config => BC learns "fixed orth" as the
                # pre-PPO starting point; PPO fine-tune learns dynamic on top).
                ch_idx_t0 = ((env.radar_freq_hz[:, 0, :] - env.fc_hz)
                             / env.channel_spacing_hz).round().long().clamp(0, env.n_channels - 1)
                channel_select_buf.append(ch_idx_t0.clone())
                total += E
                if total >= n_samples:
                    break
                action = combine_team_actions(env, a_rule, a_opp)
                env.step(action)

            # Phase 2: rule = team 1 (symmetric augmentation)
            if total < n_samples:
                env.seed = base_seed + ep * 2 + 1
                env._reset_count = ep * 2 + 1
                env.reset()
                for step in range(episode_steps):
                    obs_dict = env.get_obs()
                    a_opp = opp.get_action(env, 0)
                    a_rule = rule.get_action(env, 1)
                    obs_buf.append(obs_dict["obs"][:, 1].clone())
                    priv_buf.append(obs_dict["privileged"][:, 1].clone())
                    task_alloc_buf.append(a_rule["task_alloc"].clone())
                    beam_target_buf.append(a_rule["beam_target"].clone())
                    laser_target_buf.append(a_rule["laser_target"].clone())
                    emission_on_buf.append(a_rule["emission_on"].clone())
                    freq_hop_rate_buf.append(a_rule["freq_hop_rate"].clone())
                    # WP-C R3: channel_select from env state (team 1 row)
                    ch_idx_t1 = ((env.radar_freq_hz[:, 1, :] - env.fc_hz)
                                 / env.channel_spacing_hz).round().long().clamp(0, env.n_channels - 1)
                    channel_select_buf.append(ch_idx_t1.clone())
                    total += E
                    if total >= n_samples:
                        break
                    action = combine_team_actions(env, a_opp, a_rule)
                    env.step(action)

            ep += 1
            if verbose and (ep % 5 == 0 or total >= n_samples):
                elapsed = time.time() - t0
                print(f"  [BC collect] ep={ep} total={total}/{n_samples} "
                      f"opp={opp_name} t={elapsed:.1f}s", flush=True)

        # Trim to exactly n_samples
        def trim(buf_list, target_shape, dtype):
            cat = torch.cat([b.reshape(-1, *target_shape[1:]) if len(target_shape) > 1
                             else b.reshape(-1) for b in buf_list], dim=0)
            return cat[:n_samples].to(dtype)

        samples = {
            "obs": torch.cat([b for b in obs_buf], dim=0)[:n_samples].to(torch.float32),
            "priv": torch.cat([b for b in priv_buf], dim=0)[:n_samples].to(torch.float32),
            "task_alloc": torch.cat([b for b in task_alloc_buf], dim=0)[:n_samples].to(torch.float32),
            "beam_target": torch.cat([b for b in beam_target_buf], dim=0)[:n_samples].to(torch.long),
            "laser_target": torch.cat([b for b in laser_target_buf], dim=0)[:n_samples].to(torch.long),
            "emission_on": torch.cat([b for b in emission_on_buf], dim=0)[:n_samples].to(torch.float32),
            "freq_hop_rate": torch.cat([b for b in freq_hop_rate_buf], dim=0)[:n_samples].to(torch.float32),
            "channel_select": torch.cat([b for b in channel_select_buf], dim=0)[:n_samples].to(torch.long),
        }
        if verbose:
            print(f"  [BC collect] DONE: {samples['obs'].shape[0]} samples in "
                  f"{(time.time()-t0):.1f}s ({ep} episodes)", flush=True)
        return samples

    # ------------------------------------------------------------------
    # NLL loss
    # ------------------------------------------------------------------
    def _bc_loss(self, obs_b, priv_b, action_b):
        """NLL on rule's action under AC's current distribution."""
        log_prob, _, _, _ = self.ac.evaluate_actions(obs_b, action_b, priv_b)
        return -log_prob.mean()

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def train(
        self,
        samples: Dict[str, torch.Tensor],
        n_epochs: int = 10,
        early_stop_patience: int = 3,
        log_every: int = 1,
    ) -> List[Dict]:
        """NLL supervised training. Returns per-epoch history.

        Early stops if val_loss doesn't improve for `early_stop_patience` epochs.
        """
        dev = self.device
        N = samples["obs"].shape[0]

        # Move to device once
        obs = samples["obs"].to(dev)
        priv = samples["priv"].to(dev)
        action_full = {
            "task_alloc": samples["task_alloc"].to(dev),
            "beam_target": samples["beam_target"].to(dev),
            "laser_target": samples["laser_target"].to(dev),
            "emission_on": samples["emission_on"].to(dev),
            "freq_hop_rate": samples["freq_hop_rate"].to(dev),
            "channel_select": samples["channel_select"].to(dev),
        }

        # Train/val split
        perm = torch.randperm(N, device=dev)
        n_val = max(1, int(N * self.val_split))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        obs_tr, obs_val = obs[train_idx], obs[val_idx]
        priv_tr, priv_val = priv[train_idx], priv[val_idx]
        act_tr = {k: v[train_idx] for k, v in action_full.items()}
        act_val = {k: v[val_idx] for k, v in action_full.items()}

        n_train = train_idx.shape[0]
        history: List[Dict] = []
        best_val = float("inf")
        patience_left = early_stop_patience

        for epoch in range(n_epochs):
            self.ac.train()
            # Shuffle training indices each epoch
            shuf = torch.randperm(n_train, device=dev)
            train_loss_accum = 0.0
            n_batches = 0
            for i in range(0, n_train, self.batch_size):
                b = shuf[i:i + self.batch_size]
                if b.numel() < 8:
                    continue
                obs_b = obs_tr[b]
                priv_b = priv_tr[b]
                act_b = {k: v[b] for k, v in act_tr.items()}

                loss = self._bc_loss(obs_b, priv_b, act_b)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.opt.step()

                train_loss_accum += loss.item()
                n_batches += 1

            train_loss = train_loss_accum / max(1, n_batches)

            # Val (no grad)
            self.ac.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                n_val_batches = 0
                for i in range(0, val_idx.shape[0], self.batch_size):
                    b = slice(i, min(i + self.batch_size, val_idx.shape[0]))
                    obs_b = obs_val[b]
                    priv_b = priv_val[b]
                    act_b = {k: v[b] for k, v in act_val.items()}
                    vl = self._bc_loss(obs_b, priv_b, act_b)
                    val_loss_accum += vl.item()
                    n_val_batches += 1
                val_loss = val_loss_accum / max(1, n_val_batches)

            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            })

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                patience_left = early_stop_patience
            else:
                patience_left -= 1

            if epoch % log_every == 0 or epoch == n_epochs - 1:
                print(f"  [BC train] epoch={epoch+1}/{n_epochs} "
                      f"train_loss={train_loss:+.3f} val_loss={val_loss:+.3f} "
                      f"best={best_val:+.3f} patience={patience_left}", flush=True)

            if patience_left <= 0:
                print(f"  [BC train] early-stop at epoch {epoch+1} "
                      f"(val_loss no improvement for {early_stop_patience} epochs)", flush=True)
                break

            # NaN guard
            if any(torch.isnan(p).any().item() for p in self.ac.parameters()):
                print(f"  ❌ NaN detected in AC params at epoch {epoch}, aborting", flush=True)
                break

        return history

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, path: str, history: List[Dict]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "ac_state": self.ac.state_dict(),
            "bc_history": history,
        }, path)
