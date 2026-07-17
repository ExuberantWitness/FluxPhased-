"""WP-3 M2 verification tests: BC pretrain with BlindClassical teacher.

Tests:
  1. test_bc_uses_blind_classical_teacher: BC pretrain accepts BlindClassicalCommander.
  2. test_bc_buffer_has_beam_direction_no_beam_target: buffer has beam_direction, no beam_target.
  3. test_bc_buffer_has_detect_list: buffer has detect_list.
  4. test_bc_loss_decreases: BC 5 epochs, train_loss decreases.
  5. test_bc_no_beam_target_in_buffer: same as #2 but explicit (caught by test name).
"""

from __future__ import annotations
import sys
import torch

ROOT = "/home/ubuntu/CODE/FluxPhased-"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.pilot.twoteam.commander_actor_critic import TwoTeamCommanderActorCritic
from algo._shared.pilot.twoteam.bc_pretrain import TwoTeamBCPretrainer


def _make_setup(n_envs=4, episode_steps=30):
    env = TwoTeamVecEnv(n_envs=n_envs, device="cuda", episode_steps=episode_steps,
                        geometry=RANDOM_GEOMETRY, seed=42)
    env.reset()
    ac = TwoTeamCommanderActorCritic().to("cuda")
    rule = BlindClassicalCommander()
    bc = TwoTeamBCPretrainer(ac, lr=1e-3, batch_size=64)
    return env, ac, rule, bc


def test_bc_uses_blind_classical_teacher():
    """BC pretrain runs end-to-end with BlindClassicalCommander as teacher."""
    env, ac, rule, bc = _make_setup()
    samples = bc.collect_samples(env, rule, n_samples=200, episode_steps=30, verbose=False)
    assert samples["obs"].shape[0] >= 200
    history = bc.train(samples, n_epochs=2, log_every=1, )
    assert len(history) == 2
    print(f"BC pretrain ran with BlindClassical teacher; final val_loss={history[-1]['val_loss']:.3f}")


def test_bc_buffer_has_beam_direction_no_beam_target():
    """BC buffer must contain beam_direction, NOT legacy beam_target (god-view killed)."""
    env, ac, rule, bc = _make_setup()
    samples = bc.collect_samples(env, rule, n_samples=100, episode_steps=20, verbose=False)
    assert "beam_direction" in samples, "BC buffer missing beam_direction"
    assert "beam_target" not in samples, (
        "BC buffer still has beam_target — god-view leak not fully removed")
    assert samples["beam_direction"].shape[1:] == (env.n_radars_per_team,)
    print("BC buffer: has beam_direction, no beam_target")


def test_bc_buffer_has_detect_list():
    """BC buffer must contain detect_list [N, K_max, 5] (DeepSets encoder input)."""
    env, ac, rule, bc = _make_setup()
    samples = bc.collect_samples(env, rule, n_samples=100, episode_steps=20, verbose=False)
    assert "detect_list" in samples, "BC buffer missing detect_list"
    assert samples["detect_list"].shape[1:] == (env.k_max, 5)
    print(f"BC buffer.detect_list shape[1:] = {tuple(samples['detect_list'].shape[1:])}")


def test_bc_loss_decreases():
    """BC 5 epochs, train_loss at end < train_loss at start."""
    env, ac, rule, bc = _make_setup()
    samples = bc.collect_samples(env, rule, n_samples=400, episode_steps=30, verbose=False)
    history = bc.train(samples, n_epochs=5, log_every=1)
    assert history[-1]["train_loss"] < history[0]["train_loss"], (
        f"BC train_loss didn't decrease: {history[0]['train_loss']:.3f} -> "
        f"{history[-1]['train_loss']:.3f}"
    )
    print(f"BC train_loss decreased: {history[0]['train_loss']:.3f} -> "
          f"{history[-1]['train_loss']:.3f}")


def test_bc_no_beam_target_in_buffer():
    """Explicit alias for test_bc_buffer_has_beam_direction_no_beam_target."""
    env, ac, rule, bc = _make_setup()
    samples = bc.collect_samples(env, rule, n_samples=100, episode_steps=20, verbose=False)
    assert "beam_target" not in samples, "buffer still has beam_target"
    print("Confirmed: no beam_target in BC buffer")


if __name__ == "__main__":
    test_bc_uses_blind_classical_teacher()
    test_bc_buffer_has_beam_direction_no_beam_target()
    test_bc_buffer_has_detect_list()
    test_bc_loss_decreases()
    test_bc_no_beam_target_in_buffer()
    print("\nAll WP-3 M2 bc-pretrain-blind tests PASS")
