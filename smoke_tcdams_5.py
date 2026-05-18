"""Smoke test: 5x5 + P=4 + bins=64 + num_envs=1 with TC-DAMS league.

Same end-to-end checks as smoke_tcdams_25.py, but on a small 5x5 grid so
the (CPU-bound) payoff matrix evaluation finishes in minutes instead of
hours. Purpose: verify TC-DAMS solver, fingerprints, Elo-band sampler,
and diag_history round-trip on the real PSRO loop.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import time
import torch
import numpy as np

# Ensure project root is importable when run with `python smoke_tcdams_5.py`.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from radar_sim.gpu.vec_mfar_env import MFARVecEnv
from training.flux_league import FluxLeague


CKPT_DIR = "checkpoints/smoke_tcdams_5"


def main():
    if os.path.exists(CKPT_DIR):
        shutil.rmtree(CKPT_DIR)

    print("[smoke] Building 5x5 P=4 bins=64 num_envs=1 env...")
    t0 = time.time()
    env = MFARVecEnv(
        num_envs=1, n_radars=4, rows=5, cols=5,
        pulses_per_cpi=4, fft_size=64, device="cuda",
    )
    print(f"[smoke] Env built in {time.time()-t0:.1f}s "
          f"(E={env.num_envs}, R={env.n_radars}, N={env.n_elem}, "
          f"P={env.n_pulses}, bins={env.n_bins}, state_dim={env.state_dim}, "
          f"action_dim={env.action_dim})")

    # ----- Check 1: env returns task_fingerprint -----
    env.reset()
    actions = torch.rand(env.num_envs, env.n_radars, env.action_dim,
                         device="cuda") * 2 - 1
    cmd_actions = torch.rand(env.num_envs, env.n_teams,
                             env.battlefield.commander_action_dim,
                             device="cuda") * 2 - 1
    out = env.step(actions=actions, commander_actions=cmd_actions)
    assert "task_fingerprint" in out, "env.step() must return 'task_fingerprint'"
    fp = out["task_fingerprint"]
    assert tuple(fp.shape) == (env.num_envs, env.n_teams, 4), (
        f"expected [{env.num_envs}, {env.n_teams}, 4], got {tuple(fp.shape)}"
    )
    row_sums = fp.sum(dim=-1).detach().cpu().numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-4), (
        f"fingerprint rows must sum to 1, got sums={row_sums}"
    )
    print(f"[smoke] [PASS] task_fingerprint shape={tuple(fp.shape)}, "
          f"sample row={fp[0,0].detach().cpu().numpy().round(3)}")

    # ----- Check 2/3: tiny FluxLeague PSRO iteration with TC-DAMS -----
    league = FluxLeague(
        n_elem=env.n_elem,
        n_pulses=env.n_pulses,
        n_bins=env.n_bins,
        num_output_length=env.num_output_length,
        n_teams=env.n_teams,
        population_cap=4,
        n_eval_games=1,                   # MINIMAL: 1 game per pair
        meta_solver="tc_dams",
        tcdams_lambda=0.3,
        use_elo_band=True,
        episodes_per_training=1,          # MINIMAL: 1 episode per training step
        max_steps_per_episode=10,         # MINIMAL: 10 steps per episode
        checkpoint_dir=CKPT_DIR,
        device="cuda",
    )
    league.initialize(env)
    print(f"[smoke] FluxLeague initialized: pool={len(league.pool.policies)} "
          f"policies, trainers={len(league.trainers)}")

    print("[smoke] Running 1 PSRO iteration (this may take a few minutes)...")
    t1 = time.time()
    league.psro_iteration(env)
    elapsed = time.time() - t1
    print(f"[smoke] [PASS] PSRO iter completed in {elapsed:.1f}s")

    # ----- Check 2: fingerprints accumulated -----
    n_fp = len(league.payoff.fingerprints)
    assert n_fp >= 1, f"expected fingerprints for >=1 policy, got {n_fp}"
    print(f"[smoke] [PASS] PayoffMatrix.fingerprints has {n_fp} entries; "
          f"sample = {next(iter(league.payoff.fingerprints.values())).round(3)}")

    # ----- Check 4: diag_history JSON round-trip -----
    league.save()
    diag_path = os.path.join(CKPT_DIR, "diag_history.json")
    assert os.path.exists(diag_path)
    with open(diag_path) as f:
        diag = json.load(f)
    assert len(diag) == 1
    iter0 = diag[0]
    assert "teams" in iter0
    for team, td in iter0["teams"].items():
        for key in ("sigma", "nash_conv", "task_entropy", "effective_K"):
            assert key in td, f"diag missing {key} in team {team}"
    print(f"[smoke] [PASS] diag_history.json round-trip OK; "
          f"team0 task_entropy = {iter0['teams']['0']['task_entropy']:.3f}")

    if league.elo_sampler is not None:
        elo_path = os.path.join(CKPT_DIR, "elo.json")
        assert os.path.exists(elo_path)
        with open(elo_path) as f:
            elo = json.load(f)
        print(f"[smoke] [PASS] Elo ratings persisted: {len(elo)} policies, "
              f"sample = {list(elo.items())[:3]}")

    print("\n[smoke] All 5x5 TC-DAMS smoke checks PASSED")


if __name__ == "__main__":
    main()
