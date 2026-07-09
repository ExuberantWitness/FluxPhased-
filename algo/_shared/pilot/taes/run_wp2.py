"""WP2 main driver: train RL commander + evaluate vs classical.

Pipeline:
  Phase A — train (curriculum L0 → L1 → L3, 100 + 200 + 200 iters)
  Phase B — evaluate trained RL vs static + FP classical across the 6-cell grid
  Phase C — compute exploitability (U[π vs static L0 jammer] - U[π vs BR jammer])
            BR jammer = small MLP trained via PPO against π_RL

Outputs:
  /home/ubuntu/CODE/FluxPhased-/experiments/wp12_results/wp2_train.csv
  /home/ubuntu/CODE/FluxPhased-/experiments/wp12_results/wp2_eval.csv
  /home/ubuntu/CODE/FluxPhased-/experiments/wp12_results/wp2_exploitability.csv
  /home/ubuntu/CODE/FluxPhased-/experiments/wp12_results/WP2_VERDICT.md
"""

from __future__ import annotations

import os
import sys
import csv
import time
import json
import argparse
import torch
import numpy as np

# Repo path bootstrap
ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.taes.taes_env import TAESVecEnv
from env.gpu.qos_rrm.adversary import make_jammer, LearnedJammer
from algo._shared.pilot.taes.taes_actor_critic import TaesCommanderActorCritic, build_privileged
from algo._shared.pilot.taes.taes_ppo import TaesPPOTrainer
from algo._shared.baselines.taes_classical_commander import TaesClassicalCommander
from algo._shared.baselines.taes_fp_classical_commander import TaesFictitiousPlayCommander


# -----------------------------------------------------------------------------
# Eval helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def eval_method(method_fn, jammer_level: str, n_envs: int, n_targets: int,
                episode_steps: int, seed: int, device: str = "cuda",
                n_episodes_offset: int = 0):
    """Run one eval episode on each of n_envs parallel envs.

    method_fn(env) → action_dict (matches env.step expected keys).
    Returns per-env metrics (averaged over envs as floats).
    """
    torch.manual_seed(seed + n_episodes_offset)
    env = TAESVecEnv(n_envs=n_envs, n_targets=n_targets, device=device,
                    seed=seed + n_episodes_offset, episode_steps=episode_steps)
    jammer = make_jammer(jammer_level, device=device)
    jammer.reset(env.E, 1, env.device)
    obs_dict = env.reset()
    ep_kills = torch.zeros(env.E, device=device)
    ep_homejam = torch.zeros(env.E, device=device)
    ep_exposure = torch.zeros(env.E, device=device)
    ep_trackloss = torch.zeros(env.E, device=device)
    first_kill = torch.full((env.E,), float(episode_steps), device=device)
    ep_len = torch.zeros(env.E, device=device)
    ep_return = torch.zeros(env.E, device=device)
    alive_at_end = torch.ones(env.E, dtype=torch.bool, device=device)

    for step in range(episode_steps):
        action = method_fn(env)
        obs_dict, reward, done, info = env.step(action, jammer=jammer)
        ep_kills += info["n_kills_step"]
        ep_homejam += info["homejam_death"]
        ep_exposure = info["exposure"].clone()
        ep_trackloss += info["track_loss_rate"]
        # First-kill tracking
        any_new_kill = info["n_kills_step"] > 0
        not_yet = first_kill >= episode_steps
        update = any_new_kill & not_yet
        first_kill = torch.where(update, torch.full_like(first_kill, step + 1),
                                  first_kill)
        ep_len += (~done).float()
        ep_return += reward
        alive_at_end = alive_at_end & ~done.bool()
        if done.all():
            break

    n_actual = float(env.N)
    return {
        "kill_count": float(ep_kills.mean()),
        "kill_rate": float((ep_kills >= n_actual).float().mean()),
        "ttk_first": float(first_kill[ep_kills > 0].mean() if (ep_kills > 0).any() else episode_steps),
        "survival_rate": float((ep_homejam < 0.5).float().mean()),  # survived if no homejam death
        "survival_steps": float(ep_len.mean()),
        "homejam_count": float(ep_homejam.sum()),
        "exposure_final": float(ep_exposure.mean()),
        "trackloss_mean": float(ep_trackloss.mean() / max(1.0, ep_len.mean())),
        "ep_return": float(ep_return.mean()),
    }


# -----------------------------------------------------------------------------
# Method wrappers
# -----------------------------------------------------------------------------

def make_classical_static():
    cmd = TaesClassicalCommander()
    def fn(env):
        return cmd.step(env)
    return fn


def make_classical_fp():
    cmd = TaesFictitiousPlayCommander()
    def fn(env):
        return cmd.step(env)
    return fn


def make_rl(ac: TaesCommanderActorCritic, deterministic: bool = True):
    def fn(env):
        obs_dict = env.get_obs()
        obs = obs_dict["obs"]
        action = ac.get_action_for_env(obs, deterministic=deterministic)
        return action
    return fn


# -----------------------------------------------------------------------------
# Train (Phase A)
# -----------------------------------------------------------------------------

def train_rl_curriculum(save_dir: str, log_prefix: str = "taes_rl",
                         n_iters_l0: int = 100, n_iters_l1: int = 200,
                         n_iters_l3: int = 200, horizon: int = 600,
                         n_envs: int = 16, n_targets: int = 4,
                         seed: int = 42, device: str = "cuda"):
    """Curriculum: L0 → L1 → L3."""
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)

    env = TAESVecEnv(n_envs=n_envs, n_targets=n_targets, device=device,
                    seed=seed, episode_steps=horizon)
    ac = TaesCommanderActorCritic(obs_dim=env.obs_dim, n_targets_max=env.N_max,
                                   privileged_dim=10).to(device)
    trainer = TaesPPOTrainer(env, ac, horizon=horizon, n_epochs=4,
                              minibatch_size=128, lr_actor=3e-4, lr_critic=1e-3,
                              device=device)

    csv_path = os.path.join(save_dir, f"{log_prefix}_train.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "iter", "ep_rew", "n_kills_total", "trackloss",
                    "exposure", "homejam", "value_loss", "entropy", "approx_kl"])

    for phase, (level, n_iters) in enumerate(
        [("L0", n_iters_l0), ("L1", n_iters_l1), ("L3", n_iters_l3)]
    ):
        print(f"\n=== Phase {phase}: curriculum jammer={level} for {n_iters} iters ===",
              flush=True)
        jammer = make_jammer(level, device=device)
        jammer.reset(env.E, 1, env.device)
        # Reset env at phase start to break any state leakage
        obs_dict = env.reset()
        trainer._last_obs = obs_dict["obs"]

        for it in range(n_iters):
            rm = trainer.collect_rollout(jammer=jammer)
            um = trainer.update()
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([level, it, f"{rm['ep_rew_mean']:.3f}",
                            f"{rm['n_kills_total']:.1f}",
                            f"{rm['trackloss_mean']:.4f}",
                            f"{rm['exposure_mean']:.4f}",
                            f"{rm['homejam_total']:.0f}",
                            f"{um['value_loss']:.4f}",
                            f"{um['entropy']:.4f}",
                            f"{um['approx_kl']:.5f}"])
            if it % 10 == 0:
                print(f"  [{level}] it={it:4d} rew={rm['ep_rew_mean']:+.2f} "
                      f"kill={rm['n_kills_total']:.1f} "
                      f"trackloss={rm['trackloss_mean']:.3f} "
                      f"v_loss={um['value_loss']:.3f} H={um['entropy']:.3f} "
                      f"kl={um['approx_kl']:.4f}", flush=True)

        # Save per-phase checkpoint
        ckpt_path = os.path.join(save_dir, f"{log_prefix}_phase{phase}_{level}.pt")
        torch.save(ac.state_dict(), ckpt_path)
        print(f"  saved {ckpt_path}", flush=True)

    final_path = os.path.join(save_dir, f"{log_prefix}_final.pt")
    torch.save(ac.state_dict(), final_path)
    print(f"Saved final checkpoint: {final_path}")
    return ac


# -----------------------------------------------------------------------------
# Evaluate (Phase B)
# -----------------------------------------------------------------------------

def run_eval_grid(ac: TaesCommanderActorCritic, save_dir: str,
                  seeds=(42, 43, 44, 45, 46)):
    csv_path = os.path.join(save_dir, "wp2_eval.csv")
    methods = {
        "static_classical": make_classical_static(),
        "fp_classical": make_classical_fp(),
        "rl_commander": make_rl(ac, deterministic=True),
    }
    cells = [(1, "L0"), (4, "L0"), (4, "L1"), (4, "L3"),
             (8, "L0"), (8, "L3")]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_targets", "jam_level", "seed",
                    "kill_count", "kill_rate", "ttk_first", "survival_rate",
                    "survival_steps", "homejam_count", "exposure_final",
                    "trackloss_mean", "ep_return"])

    for (n_t, level) in cells:
        for seed in seeds:
            for mname, mfn in methods.items():
                metrics = eval_method(mfn, level, n_envs=8, n_targets=n_t,
                                       episode_steps=600, seed=seed)
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    row = [mname, n_t, level, seed]
                    row += [f"{metrics[k]:.4f}" for k in
                            ["kill_count", "kill_rate", "ttk_first",
                             "survival_rate", "survival_steps",
                             "homejam_count", "exposure_final",
                             "trackloss_mean", "ep_return"]]
                    w.writerow(row)
                print(f"  {mname:18s} n{n_t}_{level} seed={seed} "
                      f"kill={metrics['kill_count']:.2f}/{n_t} "
                      f"surv={metrics['survival_rate']:.2f} "
                      f"ttk={metrics['ttk_first']:.0f}", flush=True)

    print(f"\nEval complete → {csv_path}")
    return csv_path


# -----------------------------------------------------------------------------
# Exploitability (Phase C)
# -----------------------------------------------------------------------------

class JammerPPOTrainer:
    """Tiny PPO trainer for the LearnedJammer against a fixed commander.

    Reward = -commander_reward (zero-sum proxy).
    State: [red_task_hist(4), own_last_jam(1)] per env.
    Action: jam_mean ∈ [0,1] via sigmoid; learned std for exploration.
    """

    def __init__(self, jammer_module, lr=1e-3, gamma=0.99,
                 gae_lambda=0.95, clip=0.2, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5, n_epochs=4,
                 minibatch_size=128, device="cuda"):
        from env.gpu.qos_rrm.adversary import _JammerPolicy
        self.jammer = jammer_module  # already-instantiated LearnedJammer
        self.policy = jammer_module.policy  # _JammerPolicy
        # Value head (separate small MLP)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(5, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        ).to(device)
        # Std head (log_std for Gaussian around sigmoid mean)
        # Action distribution: Beta(alpha, beta) for [0,1] support
        # Use log_std on logit-normal as a simpler alternative:
        # logit = log(μ/(1-μ)); sample = sigmoid(logit + std*N(0,1))
        self.log_std = torch.nn.Parameter(torch.zeros(1, device=device))
        self.opt = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_head.parameters())
            + [self.log_std], lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.mb = minibatch_size
        self.device = torch.device(device)

    @torch.no_grad()
    def collect_rollout(self, env, commander_fn, horizon):
        """Step env with commander + jammer for `horizon` steps."""
        device = self.device
        H = horizon
        E = env.E
        # Storage
        obs_buf = torch.zeros(H, E, 5, device=device)
        act_buf = torch.zeros(H, E, device=device)
        logp_buf = torch.zeros(H, E, device=device)
        val_buf = torch.zeros(H, E, device=device)
        rew_buf = torch.zeros(H, E, device=device)
        done_buf = torch.zeros(H, E, device=device)
        adv_buf = torch.zeros(H, E, device=device)
        ret_buf = torch.zeros(H, E, device=device)

        # Reset env + jammer
        env.reset()
        self.jammer.reset(E, 1, device)
        # Reset env._last_jam (force fresh)
        env._last_jam = torch.zeros(E, device=device)

        for t in range(H):
            # Commander acts first
            action = commander_fn(env)
            # Jammer observes red_task_hist (= action["task_alloc"]) + last_jam
            red_task_hist = action["task_alloc"].unsqueeze(1)  # [E,1,4]
            own_last = env._last_jam.unsqueeze(1) if env._last_jam is not None \
                       else torch.zeros(E, 1, device=device)
            # Forward jammer policy (with exploration)
            mean = self.jammer.policy(red_task_hist, own_last).squeeze(1)  # [E]
            std = self.log_std.exp().expand_as(mean)
            # Sample via logit-normal
            eps = torch.randn_like(mean)
            logit_mean = torch.log(mean.clamp(1e-6, 1-1e-6) /
                                    (1 - mean.clamp(1e-6, 1-1e-6)))
            logit_sample = logit_mean + std * eps
            jam_sample = torch.sigmoid(logit_sample)
            # log_prob of sample under logit-normal:
            # Easy approximation: use Bernoulli-style with mean
            # For stability we use Normal log_prob on logit then correct via
            # change of variables — but simpler: use Normal(logit_mean, std).log_prob(logit_sample)
            log_prob = torch.distributions.Normal(logit_mean, std).log_prob(logit_sample)
            # State value
            obs_5 = torch.cat([red_task_hist.squeeze(1),
                                own_last.squeeze(-1).unsqueeze(-1)], dim=-1)  # [E,5]
            value = self.value_head(obs_5).squeeze(-1)

            # Step env with this jam_sample
            # Manually call env.step but bypass jammer.step() (use sample directly)
            # Monkey-patch by setting jammer._buf
            self.jammer._buf = jam_sample.unsqueeze(1)
            self.jammer._last_jam = jam_sample
            obs_dict_new, reward, done, info = env.step(action, jammer=self.jammer)

            # Reward: negated commander reward (zero-sum proxy on per-step reward)
            # We can't recover commander reward from info; use -reward (env reward
            # is the commander's reward). i.e. jammer's reward = -reward.
            jam_reward = -reward

            obs_buf[t] = obs_5
            act_buf[t] = logit_sample  # store logit (action in unconstrained space)
            logp_buf[t] = log_prob
            val_buf[t] = value
            rew_buf[t] = jam_reward
            done_buf[t] = done.float()

            if done.all():
                env.reset()
                self.jammer.reset(E, 1, device)
                env._last_jam = torch.zeros(E, device=device)

        # Bootstrap last value
        with torch.no_grad():
            last_val = self.value_head(obs_buf[-1]).squeeze(-1)
        # GAE
        adv = torch.zeros(E, device=device)
        for t in reversed(range(H)):
            non_term = 1.0 - done_buf[t]
            next_v = last_val if t == H - 1 else val_buf[t + 1]
            delta = rew_buf[t] + self.gamma * next_v * non_term - val_buf[t]
            adv = delta + self.gamma * self.gae_lambda * non_term * adv
            adv_buf[t] = adv
            ret_buf[t] = adv + val_buf[t]
        adv_flat = adv_buf.reshape(-1)
        adv_buf = (adv_buf - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        return dict(obs=obs_buf, act=act_buf, logp_old=logp_buf, val_old=val_buf,
                    ret=ret_buf, adv=adv_buf)

    def update(self, buf):
        H, E = buf["obs"].shape[0], buf["obs"].shape[1]
        N = H * E
        device = self.device
        obs_flat = buf["obs"].reshape(N, -1)
        act_flat = buf["act"].reshape(-1)
        lp_old = buf["logp_old"].reshape(-1)
        v_old = buf["val_old"].reshape(-1)
        ret_flat = buf["ret"].reshape(-1)
        adv_flat = buf["adv"].reshape(-1)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0,
                   "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0
        for epoch in range(self.n_epochs):
            idx = torch.randperm(N, device=device)
            for i in range(0, N, self.mb):
                b = idx[i:i + self.mb]
                if b.numel() < 8: continue
                obs_b = obs_flat[b]
                # Reconstruct red_task_hist + own_last from obs_b (first 4 + last 1)
                rth = obs_b[..., :4].unsqueeze(1)  # [B,1,4]
                own = obs_b[..., 4:5].transpose(0, 1).squeeze(0)  # [B]
                # Better:
                own = obs_b[..., 4]  # [B]
                rth_2 = obs_b[..., :4]  # [B,4]
                # policy expects [E,T,4]; T=1 here
                mean = self.jammer.policy(rth_2.unsqueeze(1),
                                          own.unsqueeze(1)).squeeze(1)  # [B]
                std = self.log_std.exp().expand_as(mean)
                logit_mean = torch.log(mean.clamp(1e-6, 1-1e-6) /
                                        (1 - mean.clamp(1e-6, 1-1e-6)))
                logit_sample = act_flat[b]
                log_prob = torch.distributions.Normal(logit_mean, std).log_prob(logit_sample)
                value = self.value_head(obs_b).squeeze(-1)
                entropy = torch.distributions.Normal(logit_mean, std).entropy()

                ratio = torch.exp(log_prob - lp_old[b])
                surr1 = ratio * adv_flat[b]
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv_flat[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (value - ret_flat[b]).pow(2).mean()
                loss = policy_loss - self.entropy_coef * entropy.mean() \
                        + self.value_coef * value_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                params = list(self.policy.parameters()) + \
                         list(self.value_head.parameters()) + [self.log_std]
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.opt.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["approx_kl"] += (lp_old[b] - log_prob).mean().item()
                n_updates += 1

        for k in metrics: metrics[k] /= max(1, n_updates)
        return metrics

    def train(self, env, commander_fn, n_iterations, horizon):
        for it in range(n_iterations):
            buf = self.collect_rollout(env, commander_fn, horizon)
            metrics = self.update(buf)
            if it % 10 == 0:
                # Summarize jammer reward (mean over rollout)
                r_mean = buf["ret"].mean().item()  # = adv + val
                print(f"  [BR-jammer] it={it:3d} jam_reward_mean={r_mean:+.3f} "
                      f"v_loss={metrics['value_loss']:.3f} "
                      f"kl={metrics['approx_kl']:.4f}", flush=True)


def compute_exploitability(ac: TaesCommanderActorCritic, save_dir: str,
                            seeds=(42, 43, 44),
                            br_train_iters: int = 100,
                            horizon: int = 300,
                            n_envs: int = 8,
                            device: str = "cuda"):
    """Compute exploitability for: RL commander, static classical, FP classical.

    Exploitability(π) = U(π vs L0_static_jammer) - U(π vs BR_jammer(π))
    where U is commander's ep_return (mean over episodes × envs).

    BR_jammer(π) is trained via PPO for `br_train_iters` iterations against π
    frozen, then evaluated.
    """
    csv_path = os.path.join(save_dir, "wp2_exploitability.csv")
    rows = []

    for mname, mfn_factory in [
        ("rl_commander", lambda: make_rl(ac, deterministic=True)),
        ("static_classical", lambda: make_classical_static()),
        ("fp_classical", lambda: make_classical_fp()),
    ]:
        for seed in seeds:
            print(f"\n--- Exploitability: {mname} seed={seed} ---", flush=True)
            torch.manual_seed(seed)
            env = TAESVecEnv(n_envs=n_envs, n_targets=4, device=device,
                            seed=seed, episode_steps=horizon)
            # Step 1: U(π vs L0 static)
            mfn = mfn_factory()
            u_static = eval_method(mfn, "L0", n_envs=n_envs, n_targets=4,
                                    episode_steps=horizon, seed=seed)["ep_return"]
            # Step 2: Train BR jammer vs π
            jammer = LearnedJammer(base_jam=0.3, device=device)
            jammer.reset(env.E, 1, env.device)
            trainer = JammerPPOTrainer(jammer, lr=1e-3, n_epochs=4,
                                        minibatch_size=128, device=device)
            mfn_for_br = mfn_factory()
            trainer.train(env, mfn_for_br, br_train_iters, horizon)
            # Step 3: Eval U(π vs BR jammer) — use the *trained* jammer
            u_br = _eval_method_with_jammer(mfn_for_br, jammer,
                                              n_envs=n_envs, n_targets=4,
                                              episode_steps=horizon, seed=seed)
            exploit = u_static - u_br
            rows.append([mname, seed, f"{u_static:.3f}", f"{u_br:.3f}",
                          f"{exploit:.3f}"])
            with open(csv_path, "a" if os.path.exists(csv_path) else "w",
                      newline="") as f:
                w = csv.writer(f)
                if f.tell() == 0:
                    w.writerow(["method", "seed", "u_vs_static_L0",
                                "u_vs_br_jammer", "exploitability"])
                w.writerow(rows[-1])
            print(f"  {mname} seed={seed}: U(L0)={u_static:+.3f}  "
                  f"U(BR)={u_br:+.3f}  exploit={exploit:+.3f}", flush=True)

    return csv_path


@torch.no_grad()
def _eval_method_with_jammer(method_fn, jammer, n_envs: int, n_targets: int,
                              episode_steps: int, seed: int,
                              device: str = "cuda"):
    """Eval with a pre-instantiated jammer (BR)."""
    torch.manual_seed(seed)
    env = TAESVecEnv(n_envs=n_envs, n_targets=n_targets, device=device,
                    seed=seed, episode_steps=episode_steps)
    jammer.reset(env.E, 1, env.device)
    env._last_jam = torch.zeros(env.E, device=device)
    obs_dict = env.reset()
    ep_return = torch.zeros(env.E, device=device)
    for step in range(episode_steps):
        action = method_fn(env)
        obs_dict, reward, done, info = env.step(action, jammer=jammer)
        ep_return += reward
        if done.all():
            break
    return float(ep_return.mean())


# -----------------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------------

def summarize(csv_path: str, out_md: str):
    """Aggregate eval CSV → per-(method, n_targets, jam_level) means."""
    from collections import defaultdict
    agg = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["method"], int(row["n_targets"]), row["jam_level"])
            agg[key].append({k: float(v) for k, v in row.items()
                             if k not in ("method", "n_targets", "jam_level", "seed")})

    with open(out_md, "w") as f:
        f.write("# WP2 Verdict: RL Commander vs Classical\n\n")
        f.write("## Eval grid (means over 5 seeds × 8 envs = 40 episodes per cell)\n\n")
        f.write("| method | n_t | jam | kill | surv | ttk | trackloss |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for key in sorted(agg.keys()):
            m, n_t, level = key
            rows = agg[key]
            kill = np.mean([r["kill_count"] for r in rows])
            surv = np.mean([r["survival_rate"] for r in rows])
            ttk = np.mean([r["ttk_first"] for r in rows])
            tl = np.mean([r["trackloss_mean"] for r in rows])
            f.write(f"| {m} | {n_t} | {level} | {kill:.2f} | {surv:.2f} | "
                    f"{ttk:.0f} | {tl:.3f} |\n")

        f.write("\n## G1 verdict (hard regime n4_L3)\n\n")
        # Compare RL vs FP classical at hardest cell
        rl_l3 = np.mean([r["kill_count"] for r in agg.get(("rl_commander", 4, "L3"), [{"kill_count":0}])])
        fp_l3 = np.mean([r["kill_count"] for r in agg.get(("fp_classical", 4, "L3"), [{"kill_count":0}])])
        st_l3 = np.mean([r["kill_count"] for r in agg.get(("static_classical", 4, "L3"), [{"kill_count":0}])])
        rl_l1 = np.mean([r["kill_count"] for r in agg.get(("rl_commander", 4, "L1"), [{"kill_count":0}])])
        fp_l1 = np.mean([r["kill_count"] for r in agg.get(("fp_classical", 4, "L1"), [{"kill_count":0}])])

        f.write(f"- Static classical n4_L3: kill={st_l3:.2f}/4\n")
        f.write(f"- FP classical     n4_L3: kill={fp_l3:.2f}/4\n")
        f.write(f"- RL commander     n4_L3: kill={rl_l3:.2f}/4\n")
        f.write(f"- FP classical     n4_L1: kill={fp_l1:.2f}/4\n")
        f.write(f"- RL commander     n4_L1: kill={rl_l1:.2f}/4\n\n")
        if rl_l3 > fp_l3 + 0.5:
            f.write("**G1 PASS**: RL commander beats FP classical at n4_L3 by "
                    f"{rl_l3 - fp_l3:+.2f} kills. → Proceed to WP3.\n")
        elif rl_l1 > fp_l1 + 0.3:
            f.write("**G1 PARTIAL**: RL commander beats FP classical at n4_L1 "
                    f"({rl_l1 - fp_l1:+.2f} kills) but not at n4_L3. "
                    "→ Tighten L3 coupling OR strengthen RL training.\n")
        else:
            f.write("**G1 FAIL**: RL commander does not beat FP classical. "
                    "→ Either tighten L3 coupling or pursue IET fallback.\n")

    print(f"\nVerdict written: {out_md}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "eval", "exploit", "all"],
                        default="all")
    parser.add_argument("--save-dir",
                        default="/home/ubuntu/CODE/FluxPhased-/checkpoints/taes_mainline")
    parser.add_argument("--results-dir",
                        default="/home/ubuntu/CODE/FluxPhased-/experiments/wp12_results")
    parser.add_argument("--n-iters-l0", type=int, default=100)
    parser.add_argument("--n-iters-l1", type=int, default=200)
    parser.add_argument("--n-iters-l3", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--br-train-iters", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--br-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--ckpt", default=None,
                        help="if set, skip training and load this checkpoint for eval")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = "cuda"

    # Phase A: train
    if args.ckpt is not None and args.phase in ("eval", "exploit"):
        ac = TaesCommanderActorCritic(obs_dim=95, n_targets_max=8,
                                       privileged_dim=10).to(device)
        ac.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded checkpoint: {args.ckpt}")
    elif args.phase in ("train", "all"):
        ac = train_rl_curriculum(
            save_dir=args.save_dir,
            log_prefix="taes_rl",
            n_iters_l0=args.n_iters_l0, n_iters_l1=args.n_iters_l1,
            n_iters_l3=args.n_iters_l3,
            horizon=args.horizon, n_envs=args.n_envs, n_targets=4,
            seed=42, device=device,
        )
    else:
        # Try default checkpoint
        default_ckpt = os.path.join(args.save_dir, "taes_rl_final.pt")
        if not os.path.exists(default_ckpt):
            print(f"No checkpoint at {default_ckpt}. Run with --phase train first.")
            sys.exit(1)
        ac = TaesCommanderActorCritic(obs_dim=95, n_targets_max=8,
                                       privileged_dim=10).to(device)
        ac.load_state_dict(torch.load(default_ckpt, map_location=device))

    # Phase B: eval
    if args.phase in ("eval", "all"):
        eval_csv = run_eval_grid(ac, args.results_dir, seeds=args.seeds)
        summarize(eval_csv, os.path.join(args.results_dir, "WP2_VERDICT.md"))

    # Phase C: exploitability
    if args.phase in ("exploit", "all"):
        exploit_csv = compute_exploitability(
            ac, args.results_dir, seeds=args.br_seeds,
            br_train_iters=args.br_train_iters, horizon=args.horizon,
            n_envs=args.n_envs, device=device,
        )
        print(f"Exploitability CSV: {exploit_csv}")


if __name__ == "__main__":
    main()
