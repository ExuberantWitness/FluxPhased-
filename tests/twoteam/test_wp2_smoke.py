"""Smoke tests for WP2 league pipeline.

Tests:
  1. test_pool_mixed_kinds: TwoTeamOpponentPool accepts rule/extreme/script/checkpoint records
  2. test_candidate_exploits_action_format: 3 exploit classes produce valid action dicts
  3. test_league_loop_minimal: league loop runs 3 iters end-to-end, no crash, snapshot saved
"""

from __future__ import annotations
import sys
import os
import shutil
import torch
import numpy as np

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.extreme_commanders import STRATEGIES
from algo._shared.pilot.twoteam.opponent_pool import (
    TwoTeamOpponentPool, PolicyRecord, build_opponent_action_fn,
)


def test_pool_mixed_kinds():
    """TwoTeamOpponentPool accepts all 4 record kinds without crash."""
    pool = TwoTeamOpponentPool(population_cap=10, rng_seed=42)
    pool.add(PolicyRecord(name="rule", kind="rule",
                          factory=lambda: TwoTeamStrongRuleCommander()))
    pool.add(PolicyRecord(name="extreme/track", kind="extreme",
                          factory=lambda: STRATEGIES["pure_track"]))
    pool.add(PolicyRecord(name="exploit/jam_spread", kind="script",
                          factory=lambda: STRATEGIES["jam_spread"]))
    pool.add(PolicyRecord(name="ckpt_fake", kind="checkpoint",
                          checkpoint_path="/tmp/_wp2_fake_ckpt.pt"))
    assert pool.num_records() == 4

    # PFSP sample should return one of them
    r = pool.sample_pfsp()
    assert r is not None
    assert r.kind in ("rule", "extreme", "script", "checkpoint")

    # EMA update
    pool.update_win_rate("rule", True)
    pool.update_win_rate("rule", True)
    pool.update_win_rate("rule", False)
    rec = pool.get("rule")
    assert rec.win_rate_vs_current is not None
    assert 0.0 <= rec.win_rate_vs_current <= 1.0
    assert rec.games_played_vs_current == 3

    print(f"✅ pool_mixed_kinds OK — sampled {r.name} (kind={r.kind}); "
          f"rule EMA after 2W/1L = {rec.win_rate_vs_current:.3f}")


def test_candidate_exploits_action_format():
    """All 3 exploit commanders produce action dicts with required keys + shapes."""
    env = TwoTeamVecEnv(n_envs=2, device="cuda", episode_steps=10,
                        geometry=RANDOM_GEOMETRY, seed=42)
    env.reset()
    E = env.E

    for name in ["jam_spread", "hard_jam_focus", "track_heavy_agile"]:
        cmd = STRATEGIES[name]
        a = cmd.get_action(env, team=0)
        # Required keys
        required = {"task_alloc", "beam_target", "laser_target",
                    "emission_on", "freq_hop_rate"}
        assert set(a.keys()) >= required, f"{name} missing keys: {required - set(a.keys())}"
        # Shapes (per-team slice: [E, 2_radars, ...])
        assert a["task_alloc"].shape == (E, 2, 4), \
            f"{name} task_alloc shape {a['task_alloc'].shape} != ({E}, 2, 4)"
        assert a["beam_target"].shape == (E, 2), \
            f"{name} beam_target shape {a['beam_target'].shape}"
        assert a["laser_target"].shape == (E,), \
            f"{name} laser_target shape {a['laser_target'].shape}"
        assert a["emission_on"].shape == (E, 2), \
            f"{name} emission_on shape {a['emission_on'].shape}"
        assert a["freq_hop_rate"].shape == (E, 2), \
            f"{name} freq_hop_rate shape {a['freq_hop_rate'].shape}"
        # task_alloc normalizes to a simplex (sums to 1)
        ta_sum = a["task_alloc"].sum(dim=-1)
        assert torch.allclose(ta_sum, torch.ones_like(ta_sum), atol=1e-3), \
            f"{name} task_alloc doesn't sum to 1: {ta_sum}"
        # No NaN
        for k, v in a.items():
            assert not torch.isnan(v).any(), f"{name} {k} has NaN"

    print(f"✅ candidate_exploits_action_format OK — all 3 exploits emit valid actions")


def test_league_loop_minimal():
    """League loop runs 3 iters end-to-end without crash, produces ≥1 snapshot."""
    import tempfile
    tmp_ckpt = tempfile.mkdtemp(prefix="wp2_smoke_ckpt_")
    tmp_out = tempfile.mktemp(prefix="wp2_smoke_report_", suffix=".md")
    try:
        # Use subprocess-style invocation by importing main() directly
        import argparse
        from algo._shared.pilot.twoteam.run_wp2_league import run_league

        # Build minimal args (WP-3 M3: include blind_teacher + entropy_coef_min)
        args = argparse.Namespace(
            n_iters=3, snapshot_every=2, log_every=1, pfsp_hardness=1.0,
            pfsp_var_mix=0.0, ema_var_uniform_floor=0.0,
            entropy_gate_on_kill=False, self_play_frac=0.0,
            population_cap=15,
            bc_samples=500, bc_epochs=1, bc_batch_size=64, bc_lr=1e-3,
            ppo_lr_actor=1e-4, ppo_lr_critic=1e-3, ppo_entropy_coef=0.01,
            ppo_entropy_coef_min=0.001,
            blind_teacher=True,
            log_std_floor=-6.0,
            # WP-3 dense shaping defaults (off) + WP-3.1 Fix A
            shape_track_bonus=0.0,
            shape_exposure_penalty=0.0,
            shape_dwell_bonus=0.0,
            shape_kill_bonus=0.0,
            shape_init_bonus=0.0,
            shape_detect_in_beam_bonus=0.0,
            shape_belief_bonus=0.0,
            curriculum_p_start=0.0,
            curriculum_anneal_iters=0,
            n_envs=2, horizon=30, n_eval_episodes=2,
            ckpt_dir=tmp_ckpt, out=tmp_out, seed=42,
        )

        run_league(args)

        # Check artifacts
        assert os.path.isfile(os.path.join(tmp_ckpt, "iter000_bc.pt")), \
            f"BC checkpoint missing in {tmp_ckpt}"
        assert os.path.isfile(os.path.join(tmp_ckpt, "iter_final.pt")), \
            f"final checkpoint missing in {tmp_ckpt}"
        assert os.path.isfile(os.path.join(tmp_ckpt, "pool_metadata.json")), \
            f"pool metadata missing in {tmp_ckpt}"
        assert os.path.isfile(tmp_out), f"report missing: {tmp_out}"

        # At least one snapshot (every snapshot_every=2 → 1 snapshot at iter 2)
        snapshots = [f for f in os.listdir(tmp_ckpt)
                     if f.startswith("iter") and f != "iter000_bc.pt" and f != "iter_final.pt"]
        assert len(snapshots) >= 1, f"expected ≥1 snapshot, got {snapshots}"

        print(f"✅ league_loop_minimal OK — 3 iters ran, {len(snapshots)} snapshot(s) saved")

    finally:
        shutil.rmtree(tmp_ckpt, ignore_errors=True)
        if os.path.isfile(tmp_out):
            os.remove(tmp_out)


if __name__ == "__main__":
    print("--- test 1: pool_mixed_kinds ---")
    test_pool_mixed_kinds()
    print("\n--- test 2: candidate_exploits_action_format ---")
    test_candidate_exploits_action_format()
    print("\n--- test 3: league_loop_minimal ---")
    test_league_loop_minimal()
    print("\n✅ All WP2 smoke tests passed")
