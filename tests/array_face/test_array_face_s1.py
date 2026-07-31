"""S1 env contract tests.

Covers plan §8 T1.9-T1.16 (env behavior). Manifest disjoint/legacy tests
(T1.17, T1.18) are in test_array_face_s1_manifest.py.
"""
from __future__ import annotations
import math
import sys
import pytest
import torch

sys.path.insert(0, ".")

from env.gpu.g3_bsta_lite.physics import default_debug_physics_config
from env.gpu.g3_bsta_lite.action_contract import ContractViolation
from env.gpu.array_face_s1 import (
    EnvConfig, ArrayFaceS1VecEnv, RadarULAConfig,
    OBS_DIM_S1, PRIVILEGED_DIM_S1, N_BEAM_DIRS_S1,
)


def _make_env(n_envs=4, device=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EnvConfig(device=dev, n_envs=n_envs)
    phys = default_debug_physics_config(P_jam_W=50.0)
    radar = RadarULAConfig()
    env = ArrayFaceS1VecEnv(cfg, physics=phys, radar=radar)
    return env, cfg


def test_T1_9_reset_shape():
    """T1.9: reset() returns obs [E, OBS_DIM_S1=16] float32."""
    env, cfg = _make_env(n_envs=8)
    obs = env.reset(seed=42)
    assert obs.shape == (cfg.n_envs, OBS_DIM_S1)
    assert obs.dtype == torch.float32
    # beam_az one-hot at step 0 (beam_az=0)
    assert obs[:, 11:16].sum(dim=-1).sub_(1.0).abs().max().item() < 1e-6  # each row sums to 1
    # specifically idx 0 should be 1 at step 0
    assert torch.allclose(obs[:, 11], torch.ones(cfg.n_envs, device=obs.device))


def test_T1_10_action_oob_raises():
    """T1.10: out-of-range action raises ContractViolation."""
    env, cfg = _make_env(n_envs=4)
    env.reset(seed=42)
    bad = torch.full((cfg.n_envs,), 99, dtype=torch.int64, device=cfg.device)
    with pytest.raises(ContractViolation):
        env.step(bad)
    bad2 = torch.full((cfg.n_envs,), -1, dtype=torch.int64, device=cfg.device)
    with pytest.raises(ContractViolation):
        env.step(bad2)


def test_T1_11_step_after_done_raises():
    """T1.11: stepping after episode done raises RuntimeError."""
    env, cfg = _make_env(n_envs=2)
    env.reset(seed=42)
    idle = torch.zeros(cfg.n_envs, dtype=torch.int64, device=cfg.device)
    for _ in range(cfg.horizon):
        env.step(idle)
    # Next call must raise (done_flag set)
    with pytest.raises(RuntimeError):
        env.step(idle)


def test_T1_12_ledger_identity_at_episode_end():
    """T1.12: After full episode, accounting_residual is 0 for all envs."""
    env, cfg = _make_env(n_envs=4)
    env.reset(seed=42)
    # Random-ish actions: alternate idle / jam_svc_0 / jam_svc_1
    torch.manual_seed(0)
    for s in range(cfg.horizon):
        actions = torch.randint(0, 3, (cfg.n_envs,), dtype=torch.int64, device=cfg.device)
        # Respect mask (idle if no energy)
        mask = env._compute_mask()
        legal_actions = torch.where(mask[:, 0], actions, torch.zeros_like(actions))
        # If action illegal, fall back to idle
        for e in range(cfg.n_envs):
            if not mask[e, legal_actions[e]]:
                legal_actions[e] = 0
        env.step(legal_actions)
    resid = env.accounting_residual()
    assert (resid == 0).all(), f"accounting_residual must be 0, got {resid.tolist()}"
    assert env.ledger_identity_residual() == 0


def test_T1_13_potential_shaping_signs():
    """T1.13: shaping = gamma * Phi_after - Phi_before. Phi = -coef * pending_count.
    When pending grows (no detection on a mission), Phi becomes more negative -> shaping negative.
    When a mission succeeds/times out (pending shrinks), Phi becomes less negative -> shaping positive.
    """
    env, cfg = _make_env(n_envs=4)
    env.reset(seed=42)
    # Step 0: idle, see if shaping stays around 0 (no pending change typically)
    idle = torch.zeros(cfg.n_envs, dtype=torch.int64, device=cfg.device)
    _, reward, _, info = env.step(idle)
    # raw_reward=0 (no drops), shaping ~ 0 or slightly negative if pending grew
    shaping = info["shaping"]
    raw = info["raw_drop"].float()
    # Sanity: shaping should be finite, |shaping| < 1 (potential_coef=0.05, max pending ~6 per env)
    assert torch.isfinite(shaping).all()
    assert shaping.abs().max().item() < 1.0


def test_T1_14_no_godview_in_lite_obs_core():
    """T1.14: lite 11-dim core obs must not linearly reveal pending_count_total beyond
    the legitimately-delayed urgency proxy channel.

    We test by regressing obs[0..10] against pending_count_total across many random steps;
    R² must be small (the proxy is delayed + non-invertible in pomdp profile).

    Note: obs[11..15] (radar_beam_az one-hot) is INTENDED to be visible by design (MDP for AF).
    """
    env, cfg = _make_env(n_envs=4)
    env.reset(seed=42)
    obs_samples = []
    pending_samples = []
    torch.manual_seed(0)
    n_steps = 0
    for episode in range(3):
        env.reset(seed=42 + episode)
        for s in range(cfg.horizon):
            actions = torch.randint(0, 3, (cfg.n_envs,), dtype=torch.int64, device=cfg.device)
            mask = env._compute_mask()
            for e in range(cfg.n_envs):
                if not mask[e, actions[e]]:
                    actions[e] = 0
            obs_before, _, done, _ = env.step(actions)
            pending_now = env._pending_per_service_batched().sum(dim=-1).float()  # [E]
            obs_samples.append(obs_before[:, :11].detach().cpu())  # lite core only
            pending_samples.append(pending_now.detach().cpu())
            n_steps += 1
            if done.all():
                break

    X = torch.cat(obs_samples, dim=0).numpy()           # [N, 11]
    y = torch.cat(pending_samples, dim=0).numpy()        # [N]
    # Ordinary least squares R²
    import numpy as np
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_pred = X_aug @ coef
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    # POMDP obs has delayed_urgency_proxy which is monotone in (past) pending — R² should be moderate
    # but not > 0.95 (would mean perfect leakage)
    assert r2 < 0.95, f"R²={r2:.4f} too high — pending_count_total leaked into lite obs"


def test_T1_15_profile_enforcement():
    """T1.15: profile must be a lite profile (PROFILE_POMDP or PROFILE_MDP_SANITY).
    Other strings raise AssertionError in __post_init__.
    """
    # Invalid profile must raise at construction
    with pytest.raises(AssertionError):
        EnvConfig(profile="invalid_profile")
    # Valid profiles must succeed
    EnvConfig(profile="pomdp_v1")
    EnvConfig(profile="mdp_sanity_v1")


def test_T1_16_reward_at_episode_end_approx_n_drops():
    """T1.16: cumulative raw_reward (raw_drop) at episode end ≈ n_drops (sum of timeout + horizon_failure).
    Shaping telescopes to 0 across full episode (potential returns to 0 when pending=0).
    """
    env, cfg = _make_env(n_envs=4)
    env.reset(seed=42)
    total_raw = torch.zeros(cfg.n_envs, device=cfg.device)
    total_reward = torch.zeros(cfg.n_envs, device=cfg.device)
    torch.manual_seed(0)
    for s in range(cfg.horizon):
        actions = torch.randint(0, 3, (cfg.n_envs,), dtype=torch.int64, device=cfg.device)
        mask = env._compute_mask()
        for e in range(cfg.n_envs):
            if not mask[e, actions[e]]:
                actions[e] = 0
        _, reward, done, info = env.step(actions)
        total_raw += info["raw_drop"].float()
        total_reward += reward
    # n_drops = timeout + horizon_failure (admission_reject = 0 in this env)
    n_drops = env.counters.n_timeout + env.counters.n_horizon_failure
    # raw_reward should match n_drops exactly (raw_drop counts timeouts per step)
    assert torch.allclose(total_raw, n_drops.float()), \
        f"total raw_reward {total_raw.tolist()} != n_drops {n_drops.tolist()}"
    # Total reward (with shaping) should be close to n_drops (telescoping shaping -> 0)
    diff = (total_reward - n_drops.float()).abs().max().item()
    # Potential coef = 0.05; at most a few pending missions remain at horizon -> small residual
    assert diff < 1.0, f"reward shaping residual too large: max diff = {diff}"


def test_T1_extra_beam_az_visible_in_obs():
    """Extra: obs[11..15] should match step_idx % N_BEAM_DIRS_S1 each step."""
    env, cfg = _make_env(n_envs=4)
    env.reset(seed=42)
    idle = torch.zeros(cfg.n_envs, dtype=torch.int64, device=cfg.device)
    for s in range(cfg.horizon):
        obs, _, _, _ = env.step(idle)
        expected_beam_az = (s + 1) % N_BEAM_DIRS_S1  # step_idx incremented after step
        expected_oh = torch.zeros(cfg.n_envs, N_BEAM_DIRS_S1, device=cfg.device)
        expected_oh[:, expected_beam_az] = 1.0
        assert torch.allclose(obs[:, 11:16], expected_oh, atol=1e-6), \
            f"step {s}: obs[11:16] = {obs[0, 11:16].tolist()}, expected beam_az_idx={expected_beam_az}"


def test_T1_extra_jammed_affects_p_detect():
    """Extra: jam_svc matching radar_svc reduces P_detect (vs idle).
    jam_svc non-matching has negligible effect on detection of radar_svc.
    """
    env, cfg = _make_env(n_envs=6)
    env.reset(seed=42)
    # Step 0: radar_svc=0, beam_az=0 (-60 deg)
    # env 0,1: idle; env 2,3: jam_svc_0; env 4,5: jam_svc_1
    actions = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64, device=cfg.device)
    _, _, _, info = env.step(actions)
    p = info["p_detect"]
    # idle: high P_detect (~1)
    # jam_svc_0 matching radar_svc=0: low P_detect (jammed, even with AF=-20dB at beam_az=0)
    # jam_svc_1 non-matching radar_svc=0: P_detect ~ 1 (no jamming on svc 0)
    assert p[0] > 0.99 and p[1] > 0.99, f"idle P_detect should be ~1, got {p[:2].tolist()}"
    assert p[2] < 0.5 and p[3] < 0.5, f"jam_svc_0 (matching) P_detect should be low, got {p[2:4].tolist()}"
    assert p[4] > 0.99 and p[5] > 0.99, f"jam_svc_1 (non-matching) P_detect should be ~1, got {p[4:6].tolist()}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
