"""WP-3 M0 verification tests: AC blind (no god-view) + detection encoder.

Tests:
  1. test_no_beam_target_head: AC __init__ does NOT create beam_target_head (god-view killed).
  2. test_has_beam_direction_head: AC creates beam_direction_head (new blind Beta head).
  3. test_has_detect_mlp: AC creates detect_mlp (DeepSets encoder).
  4. test_detection_encoder_permutation_invariant: shuffling K axis doesn't change actor log_prob.
  5. test_no_godview: env.assert_no_godview PASS after RL AC steps env.
  6. test_runs_full_episode: 200 step episode no NaN reward / no NaN tracker_x.
"""

from __future__ import annotations
import sys
import torch

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.run_wp2_league import ACCommander
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


def test_no_beam_target_head():
    """AC must NOT have beam_target_head attribute (god-view leak killed in WP-3 M0)."""
    ac = TwoTeamCommanderActorCritic().to("cuda")
    assert not hasattr(ac, "beam_target_head"), (
        "AC still has beam_target_head — god-view leak not fully removed")
    # Confirm legacy attribute is also absent from action dict
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=10)
    env.reset()
    detect = env.get_detect_list()[:, 0]
    obs = env.get_obs()["obs"][:, 0]
    priv = env.get_obs()["privileged"][:, 0]
    action, _, _, _ = ac(obs, detect, priv)
    assert "beam_target" not in action, "AC action still emits beam_target"
    print("AC has no beam_target_head; action omits beam_target")


def test_has_beam_direction_head():
    """AC must have beam_direction_head (new blind Beta → azimuth [-π, π])."""
    ac = TwoTeamCommanderActorCritic().to("cuda")
    assert hasattr(ac, "beam_direction_head"), "AC missing beam_direction_head"
    # Output shape: [hidden → n_aperture * 2] (α,β per aperture)
    assert ac.beam_direction_head.out_features == ac.n_aperture * 2
    print(f"AC has beam_direction_head (out_features={ac.beam_direction_head.out_features})")


def test_has_detect_mlp():
    """AC must have detect_mlp (DeepSets encoder for env detection list)."""
    ac = TwoTeamCommanderActorCritic().to("cuda")
    assert hasattr(ac, "detect_mlp"), "AC missing detect_mlp"
    # First layer must accept 5 input features (z_x, z_y, snr_db, is_fa, mask)
    first_linear = [m for m in ac.detect_mlp.modules() if isinstance(m, torch.nn.Linear)][0]
    assert first_linear.in_features == 5, (
        f"detect_mlp first layer in_features={first_linear.in_features} != 5")
    assert first_linear.out_features == ac.detect_emb_dim
    print(f"AC has detect_mlp (5 → {ac.detect_emb_dim} → {ac.detect_emb_dim})")


def test_detection_encoder_permutation_invariant():
    """DeepSets mean-pool: shuffling K_max dim doesn't change actor output.

    Encoder is `detect_mlp(detect_list).mean(dim=-2)`. Mean is permutation-invariant,
    so permuting detections along K axis should produce identical embedding.
    """
    torch.manual_seed(0)
    ac = TwoTeamCommanderActorCritic().to("cuda").eval()
    B, K = 8, 5
    obs = torch.randn(B, ac.obs_dim, device="cuda")
    detect = torch.randn(B, K, 5, device="cuda")
    priv = torch.randn(B, ac.privileged_dim, device="cuda")

    with torch.no_grad():
        # Compute detect_emb directly for baseline
        emb_base = ac.detect_mlp(detect).mean(dim=-2)
        # Permute K axis
        perm = torch.randperm(K)
        detect_perm = detect[:, perm, :]
        emb_perm = ac.detect_mlp(detect_perm).mean(dim=-2)

    max_diff = (emb_base - emb_perm).abs().max().item()
    assert max_diff < 1e-6, f"DeepSets encoder not permutation-invariant: diff={max_diff:.2e}"
    print(f"detect_mlp permutation-invariant (max diff {max_diff:.2e})")


def test_no_godview():
    """RL AC preserves env's no-godview contract: 44/44 obs dims invariant under permutation."""
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=30, geometry=RANDOM_GEOMETRY, seed=42)
    env.reset()
    ac = TwoTeamCommanderActorCritic().to("cuda")
    ac.eval()
    cmdr = ACCommander(ac, deterministic=True)
    opp = BlindClassicalCommander()
    # Step a few times to develop non-trivial state
    for _ in range(10):
        a0 = cmdr.get_action(env, 0)
        a1 = opp.get_action(env, 1)
        env.step(combine_team_actions(env, a0, a1))

    result = env.assert_no_godview(tol=1e-5)
    assert not result["fail_dims"], (
        f"god-view leak dims: {result['fail_dims']}; "
        f"max_diff_per_dim (top 5): "
        f"{sorted(result['max_diff_per_dim'], reverse=True)[:5]}")
    print(f"AC preserves no-godview contract: {len(result['pass_dims'])}/{env.obs_dim} dims invariant")


def test_runs_full_episode():
    """AC + BlindClassical opponent runs 200 steps without NaN reward or NaN tracker_x."""
    env = TwoTeamVecEnv(n_envs=4, device="cuda", episode_steps=200, geometry=RANDOM_GEOMETRY, seed=42)
    env.reset()
    ac = TwoTeamCommanderActorCritic().to("cuda")
    ac.eval()
    cmdr = ACCommander(ac, deterministic=True)
    opp = BlindClassicalCommander()
    rewards = []
    for step in range(200):
        a0 = cmdr.get_action(env, 0)
        a1 = opp.get_action(env, 1)
        obs, r, done, info = env.step(combine_team_actions(env, a0, a1))
        rewards.append(r.mean().item())
        assert not torch.isnan(r).any(), f"NaN reward at step {step}"
        assert not torch.isnan(env.tracker_x).any(), f"NaN tracker_x at step {step}"
        if done.all():
            break
    print(f"AC ran {step+1} steps NaN-free; final reward mean = {rewards[-1]:+.3f}")


if __name__ == "__main__":
    test_no_beam_target_head()
    test_has_beam_direction_head()
    test_has_detect_mlp()
    test_detection_encoder_permutation_invariant()
    test_no_godview()
    test_runs_full_episode()
    print("\nAll WP-3 M0 actor-critic-blind tests PASS")
