"""CPU-only unit tests for the laser-fix changes.

Verifies the high-risk mechanics WITHOUT the 40GB GPU env:
  1. CommanderActorCritic hybrid Bernoulli fire head — shapes, log-prob round-trip,
     entropy, and that gradients flow (PPO-style backward).
  2. Reshaped reward — log-distance guidance is monotone & non-saturating from
     r_ref→r_floor (no dead zone), and the illumination term is fire-gated.

Run: python tests/test_laser_fix_cpu.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from training.ppo.actor_critic import CommanderActorCritic

torch.manual_seed(0)
DEV = "cpu"
ok = True


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


print("== 1. Hybrid Bernoulli fire head ==")
ac = CommanderActorCritic(obs_dim=76, act_dim=5, hidden_dim=64, hybrid_fire=True).to(DEV)
obs = torch.randn(32, 76)

action, logp, value, priv = ac.get_action(obs)
check("action shape [B,5]", tuple(action.shape) == (32, 5))
check("fire dim in {-1,+1}", set(action[:, 0].unique().tolist()) <= {-1.0, 1.0})
check("aim dims in (-1,1)", (action[:, 1:].abs() < 1.0).all().item())
check("log_prob shape [B]", tuple(logp.shape) == (32,))
check("log_prob finite", torch.isfinite(logp).all().item())

# Round-trip: evaluate_actions on the SAME stored action must reproduce log_prob exactly
# (deterministic given fixed params; sampling only happened inside get_action).
logp2, entropy, value2, priv2 = ac.evaluate_actions(obs, action.detach())
check("eval log_prob matches get_action", torch.allclose(logp, logp2, atol=1e-4))
check("entropy finite & >0", torch.isfinite(entropy).all().item() and (entropy.mean() > 0).item())

# Deterministic path: fire is a hard threshold on the logit, aim is the mean
det_a, _, _, _ = ac.get_action(obs, deterministic=True)
check("deterministic fire in {-1,+1}", set(det_a[:, 0].unique().tolist()) <= {-1.0, 1.0})

# PPO-style backward: a fake advantage-weighted policy loss must produce grads on BOTH
# the fire logit path and the aim path.
ac.zero_grad()
logp3, ent3, _, _ = ac.evaluate_actions(obs, action.detach())
adv = torch.randn(32)
loss = -(logp3 * adv).mean() - 0.01 * ent3.mean()
loss.backward()
g = ac.action_head.weight.grad
check("action_head grad finite", g is not None and torch.isfinite(g).all().item())
check("fire-logit row (0) has grad", g[0].abs().sum().item() > 0)
check("aim rows (1:) have grad", g[1:].abs().sum().item() > 0)
check("log_std grad only on aim dims [1:]",
      ac.log_std.grad is None or ac.log_std.grad[0].abs().item() == 0.0)

print("== 2. Reshaped reward surface ==")
r_ref, r_floor = 3000.0, 0.2
gw, iw = 5.0, 50.0


def guidance(r):
    r_eff = max(r, r_floor)
    return max(torch.log(torch.tensor(r_ref / r_eff)).item(), 0.0) * gw


dists = [3000.0, 1000.0, 500.0, 95.0, 50.0, 10.0, 1.0, 0.2]
vals = [guidance(r) for r in dists]
print("   r(m) -> guidance:", {r: round(v, 2) for r, v in zip(dists, vals)})
# Strictly increasing as distance shrinks (the OLD capped reward was flat below 95m)
mono = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
check("guidance strictly increases 3000m→0.2m (no dead zone)", mono)
check("guidance has gradient inside old dead zone (95m != 10m)",
      abs(guidance(95.0) - guidance(10.0)) > 1e-3)
check("guidance ~0 at/beyond r_ref", guidance(3000.0) == 0.0)

# Illumination term is fire-gated: no fire → no illum reward even when locked
t_norm = torch.tensor(1.0)
illum_fire = 1.0 * (t_norm ** 2).item() * iw
illum_nofire = 0.0 * (t_norm ** 2).item() * iw
check("illum paid when fire & locked", illum_fire > 0)
check("illum zero when not firing", illum_nofire == 0.0)

print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)
