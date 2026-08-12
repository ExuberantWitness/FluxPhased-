"""Verify MultiHeadActor is bit-exact equivalent to MultiDiscreteActor for S2's
two-categorical-head configuration.

This is the correctness gate for Step 1 of the N-head refactor: if this passes,
the generic framework is numerically identical to the hard-coded two-head actor,
and S2 amend02 results remain reproducible through the new path.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))

import torch

from env.gpu.array_face_s2 import N_ACTIONS_BASE, N_ACTIONS_BEAM
from experiments.array_face_s2.learning_repair.trainer import (
    MultiDiscreteActor, categorical_kl as s2_cat_kl, joint_kl as s2_joint_kl,
)
from experiments.array_face_s2.learning_repair.actor_heads import (
    HeadSpec, MultiHeadActor, categorical_kl as mh_cat_kl, joint_kl_multihead,
    sample_multihead,
)


def copy_weights(src: MultiDiscreteActor, dst: MultiHeadActor):
    """Copy trunk + base/beam heads from the 2-head actor to the multihead actor."""
    dst.fc1.weight.data.copy_(src.fc1.weight.data)
    dst.fc1.bias.data.copy_(src.fc1.bias.data)
    dst.fc2.weight.data.copy_(src.fc2.weight.data)
    dst.fc2.bias.data.copy_(src.fc2.bias.data)
    dst.heads["base"].weight.data.copy_(src.head_base.weight.data)
    dst.heads["base"].bias.data.copy_(src.head_base.bias.data)
    dst.heads["beam"].weight.data.copy_(src.head_beam.weight.data)
    dst.heads["beam"].bias.data.copy_(src.head_beam.bias.data)


def main():
    torch.manual_seed(42)
    obs_dim = 21
    E = 16
    dev = "cpu"

    specs = [
        HeadSpec("base", "categorical", N_ACTIONS_BASE),
        HeadSpec("beam", "categorical", N_ACTIONS_BEAM),
    ]
    old = MultiDiscreteActor(obs_dim).to(dev)
    new = MultiHeadActor(obs_dim, specs).to(dev)
    copy_weights(old, new)

    # random obs + masks (all-True so sampled actions are always legal,
    # otherwise log_prob over masked-out logits is NaN-by-design on both sides)
    obs = torch.randn(E, obs_dim, device=dev)
    mask_base = torch.ones(E, N_ACTIONS_BASE, dtype=torch.bool, device=dev)
    mask_beam = torch.ones(E, N_ACTIONS_BEAM, dtype=torch.bool, device=dev)
    masks_new = {"base": mask_base, "beam": mask_beam}

    ok = True
    tol = 0.0  # require exact equality

    # --- forward ---
    lb_old, lbm_old = old.forward(obs)
    fwd_new = new.forward(obs)
    d_fwd_base = (lb_old - fwd_new["base"]).abs().max().item()
    d_fwd_beam = (lbm_old - fwd_new["beam"]).abs().max().item()
    print(f"[forward] base max|d|={d_fwd_base:.2e}  beam max|d|={d_fwd_beam:.2e}", end="")
    if d_fwd_base <= tol and d_fwd_beam <= tol:
        print("  OK")
    else:
        print("  FAIL"); ok = False

    # --- distribution log_prob ---
    action_base = torch.randint(0, N_ACTIONS_BASE, (E,), device=dev)
    action_beam = torch.randint(0, N_ACTIONS_BEAM, (E,), device=dev)
    lp_old = old.joint_log_prob(obs, mask_base, mask_beam, action_base, action_beam)
    lp_new = new.joint_log_prob(obs, masks_new, {"base": action_base, "beam": action_beam})
    d_lp = (lp_old - lp_new).abs().max().item()
    print(f"[log_prob] max|d|={d_lp:.2e}", end="")
    if d_lp <= tol:
        print("  OK")
    else:
        print("  FAIL"); ok = False

    # --- entropy ---
    ent_old = old.joint_entropy(obs, mask_base, mask_beam)
    ent_new_dict = new.joint_entropy(obs, masks_new)
    ent_new = ent_new_dict["_sum"]
    d_ent = (ent_old - ent_new).abs().max().item()
    print(f"[entropy]  max|d|={d_ent:.2e}", end="")
    if d_ent <= tol:
        print("  OK")
    else:
        print("  FAIL"); ok = False

    # --- categorical_kl (functional) ---
    logits_a = torch.randn(E, N_ACTIONS_BASE, device=dev)
    logits_b = torch.randn(E, N_ACTIONS_BASE, device=dev)
    kl_old = s2_cat_kl(logits_a, logits_b, mask_base)
    kl_new = mh_cat_kl(logits_a, logits_b, mask_base)
    d_kl = (kl_old - kl_new).abs().max().item()
    print(f"[cat_kl]   max|d|={d_kl:.2e}", end="")
    if d_kl <= tol:
        print("  OK")
    else:
        print("  FAIL"); ok = False

    # --- joint_kl (actor-level) ---
    with torch.no_grad():
        kl_old_full = s2_joint_kl(old, obs, mask_base, mask_beam,
                                  *old.forward(obs))
        lb_old_snap, lbm_old_snap = old.forward(obs)
        kl_new_full = joint_kl_multihead(new, obs, masks_new,
                                         {"base": lb_old_snap, "beam": lbm_old_snap})
    d_jkl = (kl_old_full - kl_new_full).abs().max().item()
    print(f"[joint_kl] max|d|={d_jkl:.2e}", end="")
    if d_jkl <= tol:
        print("  OK")
    else:
        print("  FAIL"); ok = False

    # --- sampling stream equivalence ---
    # Both must consume identical RNG and produce identical actions, given the
    # same generator and identical logit inputs. The RNG contract: one rand()
    # per head per env, base first then beam (matches MultiDiscreteActor order
    # _sample_actions: u_base then u_beam).
    g_old = torch.Generator(device=dev).manual_seed(123)
    g_new = torch.Generator(device=dev).manual_seed(123)

    # S2 trainer's _sample_actions inlined (we don't have a trainer instance here,
    # but the sampling is purely a function of distribution + generator, so we
    # replicate it on the old actor):
    d_base_o, d_beam_o = old.distribution(obs, mask_base, mask_beam)
    u_bo = torch.rand(E, generator=g_old, device=dev)
    u_bm = torch.rand(E, generator=g_old, device=dev)
    a_bo = (u_bo.unsqueeze(-1) < torch.cumsum(d_base_o.probs.clamp(min=1e-12), dim=-1)).float().argmax(dim=-1).long()
    a_bm = (u_bm.unsqueeze(-1) < torch.cumsum(d_beam_o.probs.clamp(min=1e-12), dim=-1)).float().argmax(dim=-1).long()

    # new path: sample_multihead iterates head_specs order = [base, beam]
    acts_new, _ = sample_multihead(new, obs, masks_new, g_new)

    d_ab = (a_bo - acts_new["base"]).abs().max().item()
    d_am = (a_bm - acts_new["beam"]).abs().max().item()
    print(f"[sample]   base max|d|={d_ab:.2e}  beam max|d|={d_am:.2e}", end="")
    if d_ab <= tol and d_am <= tol:
        print("  OK")
    else:
        print("  FAIL"); ok = False

    # --- bernoulli head smoke (forward only — no S2 env to compare against) ---
    # Verify a bernoulli head produces finite logits / log_prob / entropy / kl.
    specs3 = [
        HeadSpec("base", "categorical", N_ACTIONS_BASE),
        HeadSpec("beam", "categorical", N_ACTIONS_BEAM),
        HeadSpec("cell", "bernoulli", 5),
    ]
    actor3 = MultiHeadActor(obs_dim, specs3).to(dev)
    mask_cell = torch.ones(E, 5, dtype=torch.bool, device=dev)
    masks3 = {"base": mask_base, "beam": mask_beam, "cell": mask_cell}
    obs3 = torch.randn(E, obs_dim, device=dev)
    fwd3 = actor3.forward(obs3)
    assert torch.isfinite(fwd3["cell"]).all(), "bernoulli logits not finite"
    cell_action = torch.randint(0, 2, (E, 5), device=dev).float()
    lp3 = actor3.joint_log_prob(obs3, masks3, {"base": action_base, "beam": action_beam, "cell": cell_action})
    assert torch.isfinite(lp3).all(), "bernoulli joint_log_prob not finite"
    ent3 = actor3.joint_entropy(obs3, masks3)
    assert torch.isfinite(ent3["_sum"]).all(), "bernoulli entropy not finite"
    # bernoulli KL finite
    logits_old3 = actor3.forward(obs3)
    logits_new3 = actor3.forward(obs3 + 0.01)  # slightly perturbed
    kl3 = joint_kl_multihead(actor3, obs3, masks3, logits_old3)
    kl3_perturbed = joint_kl_multihead(actor3, obs3, masks3, logits_new3)
    assert torch.isfinite(kl3).all() and (kl3 >= -1e-6).all(), "bernoulli kl not finite/non-negative"
    print(f"[bernoulli head smoke] logits/logp/entropy/kl all finite, kl>=0  OK")

    # --- sampling with bernoulli head ---
    g3 = torch.Generator(device=dev).manual_seed(999)
    acts3, lp3s = sample_multihead(actor3, obs3, masks3, g3)
    assert acts3["cell"].shape == (E, 5), "bernoulli action shape wrong"
    assert set(acts3["cell"].unique().tolist()) <= {0.0, 1.0}, "bernoulli action not binary"
    assert torch.isfinite(lp3s).all(), "bernoulli sampling logp not finite"
    print(f"[bernoulli sample] shape={tuple(acts3['cell'].shape)} binary=OK logp finite  OK")

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
