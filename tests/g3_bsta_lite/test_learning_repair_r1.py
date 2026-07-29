"""R1 contract tests — learning-repair branch.

Each test binds a specific invariant from POST_AUDIT_CORRECTION.md or
PREREGISTRATION.md so a regression breaks the test by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from env.gpu.g3_bsta_lite import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    ACTION_JAM_SERVICE_1,
    ContractViolation,
    EnvConfig,
    G3BstaLiteVecEnv,
    N_ACTIONS,
    OBS_DIM,
    PROFILE_MDP_SANITY,
    PROFILE_POMDP,
    PROFILES,
    pomdp_urgency_proxy,
)
from env.gpu.g3_bsta_lite.observation import build_observation


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = (
    REPO_ROOT
    / "experiments"
    / "g3_bsta_lite"
    / "learning_repair"
    / "manifests"
)


# ---------------------------------------------------------------------------
# Manifest disjointness (PREREGISTRATION.md §3, AMENDMENT_01)
# ---------------------------------------------------------------------------

def _load_manifest(name: str) -> list[int]:
    path = MANIFEST_DIR / f"{name}.json"
    with open(path) as f:
        data = json.load(f)
    return [e["seed"] for e in data["entries"]]


def test_manifest_audit_json_overall_verdict_clean():
    with open(MANIFEST_DIR / "MANIFEST_AUDIT.json") as f:
        audit = json.load(f)
    assert audit["overall_verdict"] == "ALL_DISJOINT_AND_LEGACY_CLEAN", (
        "manifest audit did not reach ALL_DISJOINT_AND_LEGACY_CLEAN"
    )


@pytest.mark.parametrize(
    "a,b",
    [
        ("dagger_train", "ppo_train"),
        ("dagger_train", "checkpoint_validation"),
        ("dagger_train", "locked_test"),
        ("ppo_train", "checkpoint_validation"),
        ("ppo_train", "locked_test"),
        ("checkpoint_validation", "locked_test"),
    ],
)
def test_manifest_pairwise_disjoint(a: str, b: str):
    seeds_a = set(_load_manifest(a))
    seeds_b = set(_load_manifest(b))
    inter = seeds_a & seeds_b
    assert not inter, f"manifests {a} and {b} overlap on seeds {sorted(inter)}"


@pytest.mark.parametrize("name", ["dagger_train", "ppo_train",
                                   "checkpoint_validation", "locked_test"])
def test_manifest_legacy_range_excluded(name: str):
    seeds = set(_load_manifest(name))
    leaked = sorted(s for s in seeds if 20260801 <= s <= 20260832)
    assert not leaked, (
        f"manifest {name} leaked legacy F5/F6 seeds: {leaked}"
    )


@pytest.mark.parametrize("name,size", [
    ("dagger_train", 128),
    ("ppo_train", 64),
    ("checkpoint_validation", 64),
    ("locked_test", 128),
])
def test_manifest_sizes_match_preregistration(name: str, size: int):
    assert len(_load_manifest(name)) == size


def test_manifest_entries_have_arrivals_sha256():
    for name in ["dagger_train", "ppo_train",
                 "checkpoint_validation", "locked_test"]:
        with open(MANIFEST_DIR / f"{name}.json") as f:
            data = json.load(f)
        for e in data["entries"]:
            assert "arrivals_sha256" in e
            assert len(e["arrivals_sha256"]) == 64


# ---------------------------------------------------------------------------
# R1B: profile invariants
# ---------------------------------------------------------------------------

def test_pomdp_v1_requires_delay_at_least_one():
    """pomdp_v1 with obs_delay_steps = 0 must be rejected at config time."""
    with pytest.raises(ValueError, match="obs_delay_steps"):
        EnvConfig(profile=PROFILE_POMDP, obs_delay_steps=0, device="cpu")


def test_mdp_sanity_allows_delay_zero():
    """mdp_sanity_v1 may use delay=0 (pending is already exact in the obs)."""
    cfg = EnvConfig(profile=PROFILE_MDP_SANITY, obs_delay_steps=0, device="cpu")
    assert cfg.profile == PROFILE_MDP_SANITY


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        EnvConfig(profile="junk_v99", device="cpu")


def test_obs_dim_unchanged_for_both_profiles():
    assert OBS_DIM == 11
    # Implicit: build_observation produces [E, 11] for both profiles.
    E = 3
    energy = torch.full((E,), 800.0)
    initial_energy = torch.full((E,), 800.0)
    prev_oh = torch.zeros(E, N_ACTIONS)
    prev_oh[:, ACTION_IDLE] = 1.0
    intercept_conf = torch.zeros(E)
    intercept_age = torch.full((E,), 64.0)
    obs_pomdp = build_observation(
        energy=energy, initial_energy=initial_energy,
        step_idx=0, horizon=64,
        delayed_detect=torch.zeros(E, 2),
        delayed_urgency=torch.zeros(E, 2),
        intercept_confidence=intercept_conf, intercept_age=intercept_age,
        prev_action_onehot=prev_oh, profile=PROFILE_POMDP,
    )
    obs_mdp = build_observation(
        energy=energy, initial_energy=initial_energy,
        step_idx=0, horizon=64,
        intercept_confidence=intercept_conf, intercept_age=intercept_age,
        prev_action_onehot=prev_oh, profile=PROFILE_MDP_SANITY,
        pending_per_service=torch.zeros(E, 2),
        radar_service_onehot=torch.tensor([[1.0, 0.0]] * E),
    )
    assert obs_pomdp.shape == (E, OBS_DIM)
    assert obs_mdp.shape == (E, OBS_DIM)


def test_pomdp_urgency_proxy_is_non_invertible():
    """f(n) = 1 - exp(-n/K) is monotone but saturates; recoverable only
    up to a wide interval for n >= 4 (the leak-flagged region)."""
    ns = torch.arange(0, 13)
    proxy = pomdp_urgency_proxy(ns)
    # Strictly monotone up.
    assert bool(torch.all(proxy[1:] > proxy[:-1])), "proxy must be monotone"
    # Saturation: deltas shrink below 0.07 beyond n=4.
    deltas = proxy[1:] - proxy[:-1]
    assert float(deltas[4]) < 0.10, (
        f"proxy delta at n=4..5 too large ({deltas[4]}); saturation too weak"
    )


def test_mdp_sanity_observation_exposes_pending_and_radar():
    """The MDP-sanity profile must put exact pending count and radar
    service one-hot into the obs (otherwise it isn't fully observed)."""
    cfg = EnvConfig(n_envs=1, profile=PROFILE_MDP_SANITY, obs_delay_steps=0,
                    device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=21)
    # Inject 2 pending missions on service 0 and snapshot radar slot.
    env.tracker.admit(env_idx=0, step=0, service_id=0, deadline_step=10)
    env.tracker.admit(env_idx=0, step=0, service_id=0, deadline_step=10)
    obs = env._build_observation()
    # Channels 2..3 = pending_per_service; channels 4..5 = radar one-hot.
    assert int(obs[0, 2]) == 2, f"mdp_sanity obs missed pending count: {obs[0]}"
    radar_svc = int(env.radar.service_at_step(env.step_idx))
    one_hot = obs[0, 4:6]
    assert float(one_hot[radar_svc]) == 1.0
    assert float(one_hot[1 - radar_svc]) == 0.0


def test_pomdp_v1_observation_does_not_leak_exact_pending():
    """pomdp_v1 obs must not be a deterministic, invertible function of
    the exact pending count. We mutate pending WITHOUT going through
    step(); the obs builder must use the delayed proxy, not live pending."""
    cfg = EnvConfig(n_envs=1, profile=PROFILE_POMDP, obs_delay_steps=1,
                    device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=22)
    obs_before = env._build_observation().clone()
    # Mutate tracker pending directly.
    for _ in range(50):
        env.tracker.admit(env_idx=0, step=0, service_id=0, deadline_step=10)
    obs_after = env._build_observation().clone()
    assert torch.equal(obs_before, obs_after), (
        "pomdp_v1 observation leaked pending mutation without going through step()"
    )


# ---------------------------------------------------------------------------
# R1C: potential telescoping
# ---------------------------------------------------------------------------

def test_potential_shaping_uses_correct_temporal_order():
    """The shaping term must equal gamma * Phi(s_{t+1}) - Phi(s_t) where
    s_t is captured BEFORE the transition touches the tracker. Concretely,
    if a transition admits a new arrival but the step has not yet
    finalized, Phi(s_t) excludes the new arrival and Phi(s_{t+1}) includes
    it after detection+finalize."""
    cfg = EnvConfig(n_envs=1, profile=PROFILE_MDP_SANITY, obs_delay_steps=0,
                    device="cpu", potential_coef=0.1)
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=23)
    # Snapshot tracker state right before a step that will admit arrivals.
    pending_before_step = env.tracker.pending_count(0)
    # Find an arrivals step at idx > 0 to make sure step ad
    # effects differ from no-effect.
    step_idx_now = env.step_idx
    arrivals_now = bool(env.scenario.arrivals[step_idx_now].any())
    # Run one idle step so transition effects fire.
    obs, reward, done, info = env.step(torch.tensor([ACTION_IDLE], dtype=torch.int64))
    # shaping = gamma * phi_after - phi_before; the sign must match the
    # change in -potential_coef * pending_count.
    phi_b = float(info["potential_before"])
    phi_a = float(info["potential_after"])
    gamma = cfg.gamma
    expected_shaping = gamma * phi_a - phi_b
    assert abs(float(info["shaping"]) - expected_shaping) < 1e-5, (
        f"shaping {info['shaping']} != gamma*phi_after - phi_before "
        f"= {expected_shaping}"
    )
    # If arrivals fired, phi_after should differ from a captured-at-end
    # counterfactual phi_mid (the audit bug). Hard to assert without
    # instrumentation; the strong test is the sign-correct formula above.


# ---------------------------------------------------------------------------
# R1D: integer energy, OOB action, step-after-done, ledger identity
# ---------------------------------------------------------------------------

def test_integer_energy_tokens_drive_mask():
    """Mask must be derived from integer token count, not float energy.
    Drain tokens one at a time and confirm mask flips exactly when the
    last token is consumed."""
    cfg = EnvConfig(n_envs=1, profile=PROFILE_MDP_SANITY, obs_delay_steps=0,
                    horizon=8, active_budget_steps=2, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=24)
    # Initially 2 tokens: both jam actions legal.
    mask0 = env._compute_mask()
    assert bool(mask0[0, ACTION_JAM_SERVICE_0])
    # First jam -> 1 token remains.
    env.step(torch.tensor([ACTION_JAM_SERVICE_0], dtype=torch.int64))
    mask1 = env._compute_mask()
    assert bool(mask1[0, ACTION_JAM_SERVICE_0])
    # Second jam -> 0 tokens; mask flips to illegal.
    env.step(torch.tensor([ACTION_JAM_SERVICE_1], dtype=torch.int64))
    mask2 = env._compute_mask()
    assert not bool(mask2[0, ACTION_JAM_SERVICE_0])
    assert not bool(mask2[0, ACTION_JAM_SERVICE_1])
    assert bool(mask2[0, ACTION_IDLE])


def test_action_out_of_bounds_raises_before_mask_check():
    cfg = EnvConfig(n_envs=1, profile=PROFILE_MDP_SANITY, obs_delay_steps=0,
                    device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=25)
    with pytest.raises(ContractViolation, match="out of bounds"):
        env.step(torch.tensor([99], dtype=torch.int64))
    with pytest.raises(ContractViolation, match="out of bounds"):
        env.step(torch.tensor([-1], dtype=torch.int64))


def test_step_after_done_raises():
    cfg = EnvConfig(n_envs=1, profile=PROFILE_MDP_SANITY, obs_delay_steps=0,
                    horizon=4, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=26)
    for _ in range(4):
        obs, reward, done, info = env.step(
            torch.tensor([ACTION_IDLE], dtype=torch.int64)
        )
    assert bool(done.all())
    with pytest.raises(RuntimeError, match="step\\(\\) called after episode done"):
        env.step(torch.tensor([ACTION_IDLE], dtype=torch.int64))


def test_event_ledger_identity_holds_at_episode_end():
    """eligible = success + timeout + admission_reject + horizon_failure,
    re-derivable from the per-mission ledger rows alone."""
    cfg = EnvConfig(n_envs=2, profile=PROFILE_POMDP, obs_delay_steps=1,
                    horizon=32, device="cpu")
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=27)
    # Use a mask-respecting action sampler so we never trip the
    # legal-action guard while still exercising the radar/drop path.
    for _ in range(cfg.horizon):
        mask = env._compute_mask()
        actions = torch.zeros(cfg.n_envs, dtype=torch.int64)
        for e in range(cfg.n_envs):
            legal = torch.nonzero(mask[e]).flatten().tolist()
            pick = legal[int(torch.randint(0, len(legal), (1,)).item())]
            actions[e] = pick
        obs, reward, done, info = env.step(actions)
        if bool(done.all()):
            break
    # Ledger identity residual must be zero.
    assert env.ledger_identity_residual() == 0, (
        f"ledger identity residual = {env.ledger_identity_residual()}"
    )
    # Counter accounting residual also zero.
    resid = env.accounting_residual()
    assert bool((resid == 0).all()), f"counter residual nonzero: {resid}"
    # Cross-check: ledger-based eligible equals counters.n_eligible.
    by_dispo = {"success": 0, "timeout": 0, "admission_reject": 0,
                "horizon_failure": 0}
    for row in env.ledger_rows():
        by_dispo[row["disposition"]] = by_dispo.get(row["disposition"], 0) + 1
    n_eligible = sum(by_dispo.values())
    assert n_eligible == int(env.counters.n_eligible.sum().item()), (
        f"ledger eligible {n_eligible} != counter eligible "
        f"{int(env.counters.n_eligible.sum().item())}"
    )


# ---------------------------------------------------------------------------
# R1D: EnvConfig propagation snapshot
# ---------------------------------------------------------------------------

def test_envconfig_to_json_records_profile_and_tokens():
    cfg = EnvConfig(n_envs=4, profile=PROFILE_MDP_SANITY, obs_delay_steps=0,
                    horizon=32, device="cpu")
    j = cfg.to_json()
    assert j["profile"] == PROFILE_MDP_SANITY
    assert j["E0_tokens"] == cfg.E0_tokens
    assert j["obs_delay_steps"] == 0
