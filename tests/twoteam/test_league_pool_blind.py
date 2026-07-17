"""WP-3 M3 verification tests: league pool contains BlindClassical + health monitor.

Tests:
  1. test_pool_includes_blind_classical: initialize_pool() returns pool with BlindClassical.
  2. test_pool_size_is_13: pool has 13 records (rule + blind_classical + 7 extreme + 3 exploit + 1 BC).
  3. test_pool_blind_classical_produces_beam_direction: factory() returns commander whose action
     dict contains beam_direction (NOT legacy beam_target).
  4. test_make_factory_commander_blind: make_factory_commander("blind_classical") works.
"""

from __future__ import annotations
import sys
import os
import torch

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.pilot.twoteam.run_wp2_league import initialize_pool, make_factory_commander


def _setup_bc_ckpt(tmp_path):
    """Create a minimal BC checkpoint for pool seed."""
    os.makedirs(tmp_path, exist_ok=True)
    ckpt_path = os.path.join(tmp_path, "iter000_bc.pt")
    torch.save({"ac_state": {}, "iter": 0}, ckpt_path)
    return ckpt_path


def test_pool_includes_blind_classical(tmp_path=None):
    """initialize_pool() returns pool containing a record named 'blind_classical'."""
    import tempfile
    tmp = tmp_path or tempfile.mkdtemp(prefix="wp3_pool_test_")
    ckpt = _setup_bc_ckpt(tmp)
    pool = initialize_pool(bc_ckpt_path=ckpt, bc_iter=0, population_cap=30, seed=42)
    names = [r.name for r in pool.all_records()]
    assert "blind_classical" in names, f"blind_classical missing from pool: {names}"
    bc_rec = pool.get("blind_classical")
    assert bc_rec.kind == "rule"
    print(f"pool includes 'blind_classical' record (kind={bc_rec.kind})")


def test_pool_size_is_13():
    """Pool has 13 records: rule + blind_classical + 7 extreme + 3 exploit + 1 BC."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="wp3_pool_test_")
    ckpt = _setup_bc_ckpt(tmp)
    pool = initialize_pool(bc_ckpt_path=ckpt, bc_iter=0, population_cap=30, seed=42)
    assert pool.num_records() == 13, (
        f"pool size {pool.num_records()} != 13 (rule×2 + 7 extreme + 3 exploit + 1 BC)")
    print(f"pool size = {pool.num_records()} (rule + blind_classical + 7 extreme + 3 exploit + 1 BC)")


def test_pool_blind_classical_produces_beam_direction():
    """Factory for 'blind_classical' produces commander whose action has beam_direction (no beam_target)."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=10,
                        geometry=RANDOM_GEOMETRY, seed=42)
    env.reset()
    factory = make_factory_commander("blind_classical")
    cmd = factory()
    assert isinstance(cmd, BlindClassicalCommander), (
        f"factory returned {type(cmd)}, expected BlindClassicalCommander")
    action = cmd.get_action(env, team=0)
    assert "beam_direction" in action, "BlindClassical action missing beam_direction"
    assert "beam_target" not in action, "BlindClassical should NOT emit beam_target"
    print("BlindClassical pool factory returns blind commander (beam_direction, no beam_target)")


def test_make_factory_commander_blind():
    """make_factory_commander dispatches 'blind_classical' correctly."""
    factory = make_factory_commander("blind_classical")
    cmd = factory()
    assert isinstance(cmd, BlindClassicalCommander)
    # Strong rule still works (legacy ablation path)
    from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
    sr_factory = make_factory_commander("strong_rule")
    sr_cmd = sr_factory()
    assert isinstance(sr_cmd, TwoTeamStrongRuleCommander)
    print("make_factory_commander dispatches both 'blind_classical' and 'strong_rule'")


if __name__ == "__main__":
    test_pool_includes_blind_classical()
    test_pool_size_is_13()
    test_pool_blind_classical_produces_beam_direction()
    test_make_factory_commander_blind()
    print("\nAll WP-3 M3 league-pool-blind tests PASS")
