"""Smoke test for EloBandSampler."""

import os
import shutil
import tempfile
import numpy as np

from training.self_play.opponent_pool import OpponentPool
from training.self_play.elo_band_sampler import EloBandSampler


def _build_pool(tmp):
    pool = OpponentPool(pool_dir=tmp, population_cap=20, pfsp_temperature=1.0)
    # 4 red, 4 blue
    for t in range(2):
        for i in range(4):
            pool.add_policy(team=t, role="main",
                            checkpoint_path=os.path.join(tmp, f"t{t}_p{i}.pt"),
                            generation=i)
    return pool


def test_band_anneal_and_fallback():
    tmp = tempfile.mkdtemp()
    try:
        pool = _build_pool(tmp)
        s = EloBandSampler(pool, band_init=400.0, band_final=100.0, anneal_iters=10)
        # All Elo start at 1500 -> any band contains everyone.
        sample0 = s.sample("p0000", iteration=0, n_samples=1)
        assert sample0 and pool.policies[sample0[0]].team != pool.policies["p0000"].team
        # Spread Elo wide; iteration 0 with band=400 should still find some;
        # iteration 50 with band=100 should narrow.
        s.elo = {pid: 1500.0 + 250.0 * (int(pid[1:]) - 4) for pid in pool.policies}
        # p0000 is team 0; opponents are p0004..p0007 (team 1).
        # Elo: p0000=1500-1000=500; opps = 1500..2250.
        # band=400 at iter 0 -> none in band -> fallback.
        sample_fallback = s.sample("p0000", iteration=0, n_samples=1)
        assert len(sample_fallback) == 1  # fallback delivered something
        # Update from a fake payoff matrix and verify Elo moved.
        before = s.elo["p0000"]
        s.update_from_payoff_matrix({("p0000", "p0004"): 0.9})  # we beat them
        after = s.elo["p0000"]
        assert after > before, f"Elo should rise after winning, {before} -> {after}"
        # Persistence round-trip.
        path = os.path.join(tmp, "elo.json")
        s.save(path)
        s2 = EloBandSampler(pool)
        s2.load(path)
        assert abs(s2.elo["p0000"] - s.elo["p0000"]) < 1e-6
        print("[PASS] band anneal, fallback, Elo update, persistence")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    np.random.seed(0)
    test_band_anneal_and_fallback()
    print("All EloBandSampler tests PASSED")
