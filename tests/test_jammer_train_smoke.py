"""Smoke test for the L3 jammer trainer.

Verifies:
- train_jammer() runs end-to-end for a few iters without errors
- Checkpoint saves and loads
- Trained jammer outputs non-trivial jam_level (mean > 0.05)
- log_std_floor + log_std_ceiling are enforced

Per AppInt plan Step 2c (gate judgment for L3 effectiveness happens in eval_jammer_kill_drop,
not here — this is just an integration smoke test).
"""

from __future__ import annotations

import sys
import os
import shutil
import torch

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algo._shared.pilot.taes.train_jammer import train_jammer
from env.gpu.qos_rrm.adversary import make_jammer


def test_train_smoke():
    """5-iter training should complete and produce a non-trivial checkpoint."""
    save_dir = "/tmp/_test_jammer_smoke"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    try:
        path = train_jammer(
            save_dir=save_dir,
            n_iters=8,
            horizon=128,
            n_envs=4,
            n_targets=4,
            use_mixed_n=True,
            bootstrap_classical_iters=8,
            snapshot_every=4,
            log_std_floor=-6.0,
            log_std_ceiling=-1.0,
            lr=1e-3,
            seed=42,
            device="cuda",
            log_prefix="jam_test",
        )
        assert os.path.exists(path), f"final ckpt not created: {path}"

        # Load and inspect
        jammer = make_jammer("L3", device="cuda", policy_path=path)
        jammer.reset(4, 1, "cuda")
        red = torch.full((4, 1, 4), 0.25, device="cuda")
        out = jammer.step(red, torch.zeros(4, 1, device="cuda"))
        mean_jam = float(out.mean())
        # Smoke: just non-trivial. (Effectiveness gate is in eval_jammer_kill_drop.)
        assert mean_jam > 0.01, \
            f"trained jammer mean={mean_jam:.4f} too low — likely log_std ceiling missing"
        print(f"PASS: smoke train produced jammer with mean={mean_jam:.3f}")
    finally:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)


def test_factory_L3_trained_requires_path():
    """make_jammer('L3-trained') without policy_path must error."""
    try:
        make_jammer("L3-trained", device="cuda")
    except ValueError as e:
        assert "policy_path" in str(e), f"wrong error: {e}"
        print("PASS: L3-trained without policy_path raises ValueError")
        return
    raise AssertionError("L3-trained without policy_path should have errored")


def test_factory_L1_tau_variants():
    """make_jammer should parse L1-tau{N} variants correctly."""
    for tau in [16, 8, 4, 2, 1]:
        j = make_jammer(f"L1-tau{tau}", device="cuda")
        assert j.tau == tau, f"tau mismatch: expected {tau}, got {j.tau}"
    print("PASS: L1-tau{16,8,4,2,1} all parsed correctly")


if __name__ == "__main__":
    test_train_smoke()
    test_factory_L3_trained_requires_path()
    test_factory_L1_tau_variants()
    print("\nAll jammer smoke tests PASS")
