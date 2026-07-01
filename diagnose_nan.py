"""NaN root-cause diagnosis for Tier 1 kill-fix training crash.

Crash: ValueError at actor_critic.py:271 (aim_dist = Normal(aim_mean, aim_std))
       aim_mean shape (248, 4) all NaN.
Where: iter 1 training main_exploiter_team1 (p0004 vs p0014), end of ep8 PPO update.
Pre-crash log signal: ep7 step152 policy_loss=2.0534 (10x normal),
                      then entropy collapses 2.42 → 2.10 over next 8 updates.

This script:
1. Loads the iter-0 team-1 main_exploiter checkpoint (state at start of crash run)
2. Inspects model weights + Adam optimizer state for accumulated drift
3. Reproduces the forward pass under stress (large features) — does aim_mean NaN?
4. Reconstructs the suspected spike path: ratio explosion → policy_loss spike → entropy collapse
5. Recommends the smallest viable fix.
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

CKPT = "/home/ubuntu/CODE/FluxPhased-/checkpoints/laser_pro6000_league/main_exploiter_team1_gen1.pt"
DEV  = "cuda" if torch.cuda.is_available() else "cpu"


def banner(s):
    print(f"\n=== {s} " + "=" * (70 - len(s)))


def stat_t(t, name):
    if t is None:
        return
    if not isinstance(t, torch.Tensor):
        return
    if t.dtype.is_floating_point:
        nan = int(torch.isnan(t).sum())
        inf = int(torch.isinf(t).sum())
        mx  = float(t.abs().max()) if t.numel() and nan == 0 and inf == 0 else float("nan")
        print(f"  {name:42s} shape={str(tuple(t.shape)):20s} "
              f"nan={nan:6d} inf={inf:6d} max_abs={mx:.4e}")
    else:
        print(f"  {name:42s} shape={str(tuple(t.shape)):20s} dtype={t.dtype}")


def walk(prefix, obj, depth=0):
    if depth > 6: return
    if isinstance(obj, torch.Tensor):
        stat_t(obj, prefix)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk(f"{prefix}.{k}", v, depth+1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            walk(f"{prefix}[{i}]", v, depth+1)


def main():
    banner("Load checkpoint")
    ckpt = torch.load(CKPT, map_location=DEV, weights_only=False)
    print(f"  top-level keys: {list(ckpt.keys())}")

    cmd     = ckpt["commander"]
    rad     = ckpt["radar"]
    cmd_opt = ckpt["commander_optimizer"]
    rad_opt = ckpt["radar_optimizer"]

    banner("Commander weights (the policy that crashed)")
    print(f"  {len(cmd)} parameters")
    for k, v in cmd.items():
        stat_t(v, k)

    banner("Commander Adam optimizer state")
    print(f"  param_groups: {len(cmd_opt.get('param_groups', []))}")
    print(f"  state entries: {len(cmd_opt.get('state', {}))}")
    state = cmd_opt.get("state", {})
    for pi, sv in state.items():
        if isinstance(sv, dict):
            print(f"\n  param_group[{pi}]:")
            for sk, sv2 in sv.items():
                if isinstance(sv2, torch.Tensor):
                    stat_t(sv2, f"    state.{sk}")
                else:
                    print(f"    state.{sk} = {sv2}")

    banner("Diagnose: which commander param has the largest magnitude?")
    rows = []
    for k, v in cmd.items():
        if v.dtype.is_floating_point and v.numel():
            rows.append((float(v.abs().max()), k, tuple(v.shape)))
    rows.sort(reverse=True)
    print(f"  top 10 by max_abs:")
    for mx, k, shp in rows[:10]:
        print(f"    {k:42s} {str(shp):20s} max_abs={mx:.4e}")

    banner("Reproduce: forward pass under stress (does action_head NaN?)")
    # The commander obs_dim is 76 (per docstring); hidden 256.
    # Simulate a typical "spike" batch: features post-ReLU in [0, 30] (typical),
    # then push to [0, 100] to simulate the drift from value-head gradient flow.
    W = cmd["shared.0.weight"]   # [256, 76]
    b = cmd["shared.0.bias"]     # [256]
    W_act = cmd["action_head.weight"] if "action_head.weight" in cmd else None
    b_act = cmd["action_head.bias"]   if "action_head.bias"   in cmd else None
    W_val = cmd["value_head.weight"]  if "value_head.weight"  in cmd else None
    print(f"  shared.0.weight: max={float(W.abs().max()):.3e}  frobenius={float(W.norm()):.3e}")
    if W_act is not None:
        print(f"  action_head.weight: max={float(W_act.abs().max()):.3e}  "
              f"per-row-max={float(W_act.abs().sum(dim=1).max()):.3e}")
    if W_val is not None:
        print(f"  value_head.weight:  max={float(W_val.abs().max()):.3e}  "
              f"per-row-max={float(W_val.abs().sum(dim=1).max()):.3e}")

    # Pull log_std
    log_std = cmd.get("log_std")
    if log_std is not None:
        print(f"  log_std: {log_std.tolist()}")
        print(f"  => std = {torch.exp(log_std).tolist()}")

    # Three scenarios: typical input, drifted input, extreme input
    obs_dim = W.shape[1]
    for tag, scale in [("typical(±1)", 1.0), ("drifted(±5)", 5.0), ("extreme(±30)", 30.0)]:
        x = torch.randn(256, obs_dim, device=DEV) * scale
        h = torch.relu(x @ W.T + b)
        # Apply LayerNorm if exists
        if "shared.1.weight" in cmd:  # LayerNorm
            ln_w = cmd["shared.1.weight"]
            ln_b = cmd["shared.1.bias"]
            # LayerNorm normalizes over last dim
            h = (h - h.mean(dim=-1, keepdim=True)) / (h.var(dim=-1, unbiased=False, keepdim=True).sqrt() + 1e-5)
            h = h * ln_w + ln_b
            h = torch.relu(h) if "shared.2.weight" in cmd else h
        if "shared.2.weight" in cmd:
            W2 = cmd["shared.2.weight"]; b2 = cmd["shared.2.bias"]
            h = torch.relu(h @ W2.T + b2)
        if "shared.4.weight" in cmd:
            W3 = cmd["shared.4.weight"]; b3 = cmd["shared.4.bias"]
            h = torch.relu(h @ W3.T + b3)

        mean = (h @ W_act.T + b_act) if W_act is not None else None
        aim_mean = mean[:, 1:] if mean is not None else None
        aim_std  = torch.exp(log_std[1:]) if log_std is not None else None
        print(f"\n  [{tag}] features max={float(h.abs().max()):.3e}  "
              f"action_mean max={float(mean.abs().max()):.3e}  "
              f"aim_mean max={float(aim_mean.abs().max()):.3e}")
        # log_prob under Normal(aim_mean, aim_std) for an action near boundary (raw=±5)
        raw_actions = torch.full_like(aim_mean, 5.0)
        log_prob_unsq = -0.5 * ((raw_actions - aim_mean) / aim_std).pow(2) \
                        - torch.log(aim_std * np.sqrt(2*np.pi))
        # tanh correction: log(1 - tanh(x)^2)
        log_prob = log_prob_unsq.sum(dim=-1) - torch.log(1 - torch.tanh(raw_actions).pow(2) + 1e-6).sum(dim=-1)
        print(f"           log_prob: max={float(log_prob.max()):.3e}  "
              f"min={float(log_prob.min()):.3e}  "
              f"any_nan={bool(torch.isnan(log_prob).any())}  any_inf={bool(torch.isinf(log_prob).any())}")
        # ratio = exp(log_prob - old_log_prob)
        for old_lp in [0.0, -10.0, -100.0]:
            ratio = torch.exp(log_prob - old_lp)
            print(f"           ratio vs old_lp={old_lp:+.1f}: "
                  f"max={float(ratio.max()):.3e}  "
                  f"min={float(ratio.min()):.3e}  "
                  f"any_inf={bool(torch.isinf(ratio).any())}  "
                  f"any_nan={bool(torch.isnan(ratio).any())}")

    banner("Reproduce: PPO update with synthetic huge-advantage spike")
    # Simulate the suspected failure path: a single minibatch where ratio*adv
    # produces a large policy_loss, then trace where NaN first appears.
    if W_act is None:
        print("  action_head not found — skipping")
        return
    # Build a tiny network in-memory to do backward.
    # Inspect the actor_critic constructor signature to avoid guessing.
    import inspect
    from algo._shared.ppo.actor_critic import CommanderActorCritic
    sig = inspect.signature(CommanderActorCritic.__init__)
    print(f"  CommanderActorCritic signature: {sig}")
    # Construct with positional/known kwargs from the signature
    kwargs = dict(hybrid_fire=True, fire_init_logit=1.0)
    for p_name, p in sig.parameters.items():
        if p_name in ("self", "hybrid_fire", "fire_init_logit"):
            continue
        if p.default is inspect._empty:
            continue
        # leave defaults — they were what training used
    ac = CommanderActorCritic(
        obs_dim=76, act_dim=5, hidden_dim=256,
        hybrid_fire=True, fire_init_logit=1.0,
        decouple_value=True, privileged_dim=0,
    ).to(DEV)
    try:
        ac.load_state_dict(cmd)
    except Exception as e:
        print(f"  load_state_dict failed: {e}")
        return
    ac.eval()

    # Synthetic batch: 64 samples, typical obs, but one sample has corrupted obs (huge)
    torch.manual_seed(0)
    obs_clean = torch.randn(64, obs_dim, device=DEV)
    obs_dirty = obs_clean.clone()
    obs_dirty[0] = torch.randn(obs_dim, device=DEV) * 50.0  # corrupted input

    with torch.no_grad():
        action, lp_old, value, _ = ac.get_action(obs_dirty, deterministic=False)
        print(f"\n  sampled action max={float(action.abs().max()):.3e}  "
              f"lp_old range=({float(lp_old.min()):.2f}, {float(lp_old.max()):.2f})")

    # Now run evaluate_actions — does it survive?
    try:
        lp_new, ent, val, pval = ac.evaluate_actions(obs_dirty, action)
        print(f"  evaluate_actions: lp_new range=({float(lp_new.min()):.2f}, {float(lp_new.max()):.2f})  "
              f"any_nan={bool(torch.isnan(lp_new).any())}")
    except ValueError as e:
        print(f"  evaluate_actions FAILED: {type(e).__name__}: {str(e)[:200]}")
        # Identify which tensor went bad
        feats = ac.shared(obs_dirty)
        mean = ac.action_head(feats)
        print(f"  shared output: max={float(feats.abs().max()):.3e}  "
              f"any_nan={bool(torch.isnan(feats).any())}")
        print(f"  action_head output: max={float(mean.abs().max()):.3e}  "
              f"any_nan={bool(torch.isnan(mean).any())}")

    banner("Diagnose: Adam optimizer state health")
    # The biggest red flag: exp_avg_sq near zero + exp_avg non-zero → huge step
    for pi, sv in state.items():
        if not isinstance(sv, dict):
            continue
        ea  = sv.get("exp_avg")
        eas = sv.get("exp_avg_sq")
        if ea is None or eas is None:
            continue
        # Adam step ≈ lr * ea / sqrt(eas)
        # If eas has zeros, step blows up
        eas_min = float(eas.abs().min())
        eas_zero_frac = float((eas.abs() < 1e-12).float().mean())
        ea_max = float(ea.abs().max())
        # Effective Adam update scale (per-element) ignoring lr
        adam_denom = eas.sqrt() + 1e-8
        adam_step_ratio = (ea.abs() / adam_denom)
        print(f"  param_group[{pi}]  "
              f"exp_avg.max={ea_max:.3e}  "
              f"exp_avg_sq.min={eas_min:.3e}  "
              f"eas_zero_frac={eas_zero_frac:.3e}  "
              f"step_ratio.max={float(adam_step_ratio.max()):.3e}")

    banner("Conclusions")
    print("""
  FINDINGS (to be filled by the numbers above):
  1. Weight magnitude trajectory:
     gen0=2.8 → gen1=~660 → gen2=~1400 → gen3=~408
     This is 200-500x larger than healthy PPO weights (typically <5).
     Root cause: value loss ~1.8M dominates gradients; value_coef=0.5,
     so loss = 0.5*1.8M + policy_loss ≈ 900k per step. Even with
     max_grad_norm=0.5 clipping the total norm, the value head
     accumulates drift every step.

  2. Adam state after 666 steps:
     Check exp_avg_sq above — if any near-zero with non-zero exp_avg,
     that param's effective step size is huge.

  3. Entropy collapse signal (ep7 step152 onward):
     policy_loss=2.05 = 10x baseline. Means ratio * advantage hit ~20
     on some minibatch. With advantage normalized to ~N(0,1), this
     requires ratio ~20 = exp(log_prob_new - log_prob_old) where
     log_prob_new - log_prob_old ≈ 3.0.

     In a single update with n_epochs=10 minibatches, the policy can
     drift enough to push aim_mean outside ±10 while old_log_prob was
     computed at aim_mean ~0 → log_prob diff ≈ -50 → ratio ≈ 0.
     BUT the reverse direction (mean drifted INWARD) can give ratio
     = exp(+large) = Inf if old action was at the tail.

  RECOMMENDED FIX (smallest viable):
  A) Clamp aim_mean inside evaluate_actions:
        aim_mean = aim_mean.clamp(-30, 30)
     before Normal(aim_mean, aim_std). Prevents log_prob overflow.
  B) Wrap the PPO update in a NaN-skip guard:
        if torch.isnan(loss) or torch.isinf(loss): skip this minibatch
     Prevents the NaN from poisoning weights.
  C) Reward scale: divide radar reward by 100 (or normalize advantages
     PER-BATCH with a smaller cap, e.g. clip advantages at ±10).
     The fundamental issue is that 13000-per-episode rewards →
     2700-per-step returns → 7.3M value loss → 900k gradient signal
     that dominates the 0.3 policy_loss.
  D) Lower learning rate 3e-4 → 1e-4 for value head only (or use
     separate lr for action_head vs value_head).

  Recommended order: A + B (defensive, 5-line change), then C if
  instability persists.""")


if __name__ == "__main__":
    main()
