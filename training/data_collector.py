"""Rollout data collector for critic and behavior cloning pre-training.

Uses scripted (heuristic) policies to generate (state, action, reward,
next_state, done) trajectories.  The collected data is used to:
  1. Pre-train the critic via supervised regression on Monte Carlo returns
  2. Pre-train the actor via behavior cloning (BC) on scripted actions

All computation stays on GPU — CPU transfers only happen at the final return.
"""

import torch
import numpy as np
import functools
import builtins
from typing import Dict

print = functools.partial(builtins.print, flush=True)


class RolloutDataCollector:
    """Collect trajectories from scripted policies for critic/actor pre-training.

    Fully GPU-native: tensors stay on device during collection, minimal CPU sync.
    """

    def __init__(self, device: str = "cuda", augment_noise: float = 0.0):
        self.device = device
        self.augment_noise = augment_noise  # 0.0=off, 1.0=full augmentation

    def collect(
        self,
        env,           # MFARVecEnv
        scripted_policy,     # callable: (env, team) -> actions [E, R, act_dim]
        scripted_commander,  # callable: (env, team, crc_ok) -> cmd_actions [E, cmd_dim]
        team: int,
        n_episodes: int,
        max_steps: int = 1000,
        gamma: float = 0.99,
    ) -> dict:
        """Collect rollout data for one team.

        Runs n_episodes with scripted policies for BOTH teams (so the env
        is fully populated).  Stores trajectories for the specified team.

        Returns dict with keys:
            obs: [T, obs_dim]            — radar observations (representative radar)
            actions: [T, act_dim]        — radar actions (representative radar)
            rewards: [T]                 — summed per-step radar reward
            dones: [T]                   — episode done flag
            values: [T]                  — placeholder (zero)
            privileged_infos: [T, 16]    — privileged info for critic
            returns: [T]                 — Monte Carlo discounted returns
            commander_obs: [T, 76]       — commander observations
            commander_actions: [T, 35]   — commander actions
            commander_rewards: [T]       — commander rewards
            commander_returns: [T]       — commander MC returns
        """
        n_teams = 2
        r_per_team = env.n_radars // n_teams
        other_team = 1 - team
        E = env.num_envs
        dev = env.device

        obs_list, act_list, rew_list, done_list = [], [], [], []
        priv_list = []
        cmd_obs_list, cmd_act_list, cmd_rew_list = [], [], []

        episodes_done = 0

        # ── Track scene coverage + task variance for quality checks ──
        scene_counts = {"recon": 0, "detect": 0, "jam": 0, "comm": 0}
        launch_count = 0
        # Accumulate per-episode task fingerprint (averaged over envs) for
        # task-allocation variance check: std/mean across episodes > 10%.
        task_fp_episodes = []  # list of [4] tensors (one per episode)

        while episodes_done < n_episodes:
            env.reset()

            # ── Augmentation: perturb target positions (±2000m Gaussian) ──
            if self.augment_noise > 0:
                tgt_orig = env.target_pos.clone()
                perturb = torch.randn_like(tgt_orig) * 2000.0 * self.augment_noise
                env.target_pos = tgt_orig + perturb

            # ── Augmentation: scale channel noise [0.5×, 2.0×] ──
            noise_scale = 1.0
            if self.augment_noise > 0 and hasattr(env, '_channel'):
                noise_scale = 0.5 + torch.rand(1).item() * 1.5  # [0.5, 2.0]
                env._channel.noise_std *= noise_scale

            ep_obs, ep_act, ep_rew, ep_done = [], [], [], []
            ep_priv = []
            ep_cmd_obs, ep_cmd_act, ep_cmd_rew = [], [], []

            active = torch.ones(E, dtype=torch.bool, device=dev)
            crc_ok_cache = None
            ep_launched = False
            ep_scenes = set()

            for step in range(max_steps):
                if not active.any():
                    break

                with torch.no_grad():
                    # Get scripted actions for BOTH teams
                    own_actions = scripted_policy(env, team)        # [E, R, act_dim]
                    opp_actions = scripted_policy(env, other_team)  # [E, R, act_dim]

                    actions = torch.zeros(E, env.n_radars, env.action_dim, device=dev)
                    r0_own, r1_own = team * r_per_team, (team + 1) * r_per_team
                    r0_opp, r1_opp = other_team * r_per_team, (other_team + 1) * r_per_team
                    actions[:, r0_own:r1_own, :] = own_actions[:, r0_own:r1_own, :]
                    actions[:, r0_opp:r1_opp, :] = opp_actions[:, r0_opp:r1_opp, :]

                    # Commander actions for both teams
                    cmd_own = scripted_commander(env, team, crc_ok_cache)
                    cmd_opp = scripted_commander(env, other_team, crc_ok_cache)

                    cmd_actions = torch.zeros(
                        E, n_teams, env.battlefield.commander_action_dim,
                        device=dev,
                    )
                    cmd_actions[:, team, :] = cmd_own
                    cmd_actions[:, other_team, :] = cmd_opp

                    # ── NO force-launch during data collection ──
                    # HPEDF commander uses CRC-based launch; demo data must
                    # reflect realistic tactical decisions, not forced kills.

                result = env.step(actions=actions, commander_actions=cmd_actions)

                # ── Scene tracking for coverage checks ──
                if cmd_own[:, 0].max() > 0.5:
                    ep_launched = True
                # Classify scene from task fingerprint
                fp = result.get("task_fingerprint")  # [E, n_teams, 4]
                if fp is not None:
                    dominant = fp[:, team, :].mean(dim=0).argmax().item()
                    scene_names = ["recon", "detect", "jam", "comm"]
                    ep_scenes.add(scene_names[dominant])

                crc_ok_cache = result.get("comm_crc_ok")

                r_rep = team * r_per_team  # first radar of the team

                # Keep on GPU — active mask via boolean indexing
                state = result["state"][:, r_rep, :][active]  # [active_E, state_dim]
                own_act = actions[:, r_rep, :][active]        # [active_E, act_dim]
                radar_rew = result["radar_rewards"][:, r_rep][active]  # [active_E]
                cmd_obs = result["commander_obs"][:, team, :][active]  # [active_E, cmd_obs_dim]
                cmd_rew = result["commander_rewards"][:, team][active]  # [active_E]
                done_flag = result["dones"][active]  # [active_E]

                priv = self._build_privileged(env, team, r_per_team)[active]  # [active_E, 16]

                ep_obs.append(state)
                ep_act.append(own_act)
                ep_rew.append(radar_rew)
                ep_done.append(done_flag)
                ep_priv.append(priv)
                ep_cmd_obs.append(cmd_obs)
                ep_cmd_act.append(cmd_own[active])
                ep_cmd_rew.append(cmd_rew)

                active = active & ~result["dones"]

            episodes_done += E

            # ── Accumulate coverage stats ──
            for s in ep_scenes:
                scene_counts[s] += 1
            if ep_launched:
                launch_count += 1
            # Per-episode mean task fingerprint (averaged over envs)
            fp_mean = result["task_fingerprint"][:, team, :].mean(dim=0)  # [4]
            task_fp_episodes.append(fp_mean.cpu())

            if episodes_done % E == 0:
                print(f"  [collector] {episodes_done}/{n_episodes} episodes...", flush=True)

            # Concatenate episode data on GPU
            ep_obs_t = torch.cat(ep_obs, dim=0)   # [T_ep, obs_dim]
            ep_act_t = torch.cat(ep_act, dim=0)
            ep_rew_t = torch.cat(ep_rew, dim=0)
            ep_done_t = torch.cat(ep_done, dim=0)
            ep_priv_t = torch.cat(ep_priv, dim=0)
            ep_cmd_obs_t = torch.cat(ep_cmd_obs, dim=0)
            ep_cmd_act_t = torch.cat(ep_cmd_act, dim=0)
            ep_cmd_rew_t = torch.cat(ep_cmd_rew, dim=0)

            # GPU-vectorized Monte Carlo returns
            returns = self._compute_returns(ep_rew_t, ep_done_t, gamma)
            cmd_returns = self._compute_returns(ep_cmd_rew_t, ep_done_t, gamma)

            obs_list.append(ep_obs_t)
            act_list.append(ep_act_t)
            rew_list.append(ep_rew_t)
            done_list.append(ep_done_t)
            priv_list.append(ep_priv_t)
            cmd_obs_list.append(ep_cmd_obs_t)
            cmd_act_list.append(ep_cmd_act_t)
            cmd_rew_list.append(ep_cmd_rew_t)

            if len(obs_list) == 1:
                returns_list = [returns]
                cmd_returns_list = [cmd_returns]
            else:
                returns_list.append(returns)
                cmd_returns_list.append(cmd_returns)

        result = {
            "obs": torch.cat(obs_list, dim=0),
            "actions": torch.cat(act_list, dim=0),
            "rewards": torch.cat(rew_list, dim=0),
            "dones": torch.cat(done_list, dim=0),
            "privileged_infos": torch.cat(priv_list, dim=0),
            "returns": torch.cat(returns_list, dim=0),
            "commander_obs": torch.cat(cmd_obs_list, dim=0),
            "commander_actions": torch.cat(cmd_act_list, dim=0),
            "commander_rewards": torch.cat(cmd_rew_list, dim=0),
            "commander_returns": torch.cat(cmd_returns_list, dim=0),
        }

        T = result["obs"].shape[0]
        print(f"  [collector] Collected {T} transitions from {n_episodes} episodes", flush=True)

        # ── Compute task allocation variance ──
        if len(task_fp_episodes) > 1:
            fp_stack = torch.stack(task_fp_episodes)  # [n_episodes, 4]
            fp_mean = fp_stack.mean(dim=0)
            fp_std = fp_stack.std(dim=0, correction=1)  # sample std (ddof=1)
            # Coefficient of variation per task, max across tasks
            task_cv = (fp_std / fp_mean.clamp(min=1e-6)).max().item()
        else:
            task_cv = 0.0

        # ── Coverage check ──
        min_per_scene = min(scene_counts.values()) if scene_counts else 0
        launch_pct = 100 * launch_count / max(n_episodes, 1)
        coverage_ok = True

        print(f"  [collector] Scene coverage: {scene_counts}", flush=True)
        print(f"  [collector] Launch rate: {launch_count}/{n_episodes} "
              f"({launch_pct:.0f}%)", flush=True)
        print(f"  [collector] Task CV: {task_cv:.3f} (need >0.10 for diversity)",
              flush=True)

        if min_per_scene < 5:
            print(f"  [collector] ⚠ FAIL: only {min_per_scene} episodes for "
                  f"rarest scene (need ≥5)", flush=True)
            coverage_ok = False
        if launch_pct < 30.0:
            print(f"  [collector] ⚠ FAIL: launch rate {launch_pct:.0f}% < 30%",
                  flush=True)
            coverage_ok = False
        if task_cv < 0.10:
            print(f"  [collector] ⚠ FAIL: task CV {task_cv:.3f} < 0.10 "
                  f"(task allocation too uniform)", flush=True)
            coverage_ok = False

        if coverage_ok:
            print(f"  [collector] ✓ Coverage check PASSED", flush=True)

        # Attach coverage stats for auto-retry logic
        result["coverage"] = {
            "ok": coverage_ok,
            "scene_counts": scene_counts,
            "launch_count": launch_count,
            "launch_pct": launch_pct,
            "task_cv": task_cv,
            "n_episodes": n_episodes,
        }

        return result

    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor,
                         gamma: float) -> torch.Tensor:
        """Compute Monte Carlo discounted returns (CPU, fast for T<100k).

        Avoids GPU kernel-launch overhead by running the scalar loop on CPU.
        Typical T=8000 takes <1ms on CPU vs seconds with per-element GPU ops.

        G_t = r_t + gamma * (1 - done_t) * G_{t+1}
        """
        T = rewards.shape[0]
        r = rewards.cpu().numpy()
        d = dones.cpu().numpy().astype(bool)
        result = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in range(T - 1, -1, -1):
            if d[t]:
                running = r[t]
            else:
                running = r[t] + gamma * running
            result[t] = running
        return torch.from_numpy(result).to(rewards.device)

    def _build_privileged(self, env, team: int, r_per_team: int) -> torch.Tensor:
        """Build privileged info tensor [E, 16] for the asymmetric critic.

        Layout matching TeamPPOTrainer._build_privileged_info():
          - task_fingerprint: [E, n_teams * 4] = [E, 8]
          - cross_team_intercept_flat: [E, n_teams * 3] = [E, 6]
          - target_direction: [E, 2]
        """
        E = env.num_envs
        dev = env.device
        n_teams = 2

        fp = getattr(env, "_cached_task_fingerprint", None)
        if fp is not None:
            fp = fp.reshape(E, n_teams * 4)
        else:
            fp = torch.zeros(E, n_teams * 4, device=dev)

        intercept = getattr(env, "_cached_cross_team_intercept", None)
        if intercept is not None and isinstance(intercept, dict):
            parts = []
            for t in range(n_teams):
                detail = intercept.get(f"team{t}_intercept_detail")
                if detail is not None:
                    parts.append(detail)
                else:
                    parts.append(torch.zeros(E, 3, device=dev))
            intercept_flat = torch.cat(parts, dim=-1)
        else:
            intercept_flat = torch.zeros(E, n_teams * 3, device=dev)

        r0 = team * r_per_team
        target_pos = env.target_pos[:, 0, :]  # [E, 3]
        radar_pos = env.radar_pos[:, r0, :]    # [E, 3]
        diff = target_pos - radar_pos
        dx, dy, dz = diff[:, 0], diff[:, 1], diff[:, 2]
        dist_xy = torch.sqrt(dx ** 2 + dy ** 2).clamp(min=1.0)
        az = torch.atan2(dy, dx)
        el = torch.atan2(dz, dist_xy)
        target_dir = torch.stack([az, el], dim=-1)  # [E, 2]

        return torch.cat([fp, intercept_flat, target_dir], dim=-1)  # [E, 16]
