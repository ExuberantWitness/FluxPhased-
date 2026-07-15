"""WP-C D1 unit tests: AdaptiveSpectrumJammer + TrueFixedJammer behavior.

D1 requires a truly harder dynamic enemy than fixed. These tests verify:
  1. Both jammers output the 6-key action dict
  2. jam_fraction apple-to-apple in task_alloc
  3. TrueFixed stays on one constant channel regardless of victim
  4. AdaptiveSpectrum splits enemy radars across victim channels
  5. AdaptiveSpectrum follows victim when victim changes channels
"""

import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")

import torch
from env.gpu.twoteam import TwoTeamVecEnv, MIRROR_GEOMETRY
from algo._shared.baselines.adaptive_spectrum_jammer import (
    AdaptiveSpectrumJammer, TrueFixedJammer,
)
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions


def _make_env(n_envs=4, episode_steps=50):
    return TwoTeamVecEnv(
        n_envs=n_envs, device="cuda",
        episode_steps=episode_steps,
        geometry=MIRROR_GEOMETRY,
        team_offset_m=2500.0,
    )


def test_true_fixed_constant_channel():
    """TrueFixed outputs same channel regardless of victim freq."""
    env = _make_env()
    env.reset()
    # Set victim team 1 to weird channels
    E, R = env.E, env.n_radars_per_team
    dev = env.device
    weird_freq = env.fc_hz + torch.tensor([3, 5], device=dev).float() * env.channel_spacing_hz
    freqs = env.radar_freq_hz.clone()
    freqs[:, 1, :] = weird_freq.unsqueeze(0).expand(E, R)
    env.set_radar_freqs(freqs)

    jammer = TrueFixedJammer(jam_fraction=1e-4, fixed_channel=0)
    a = jammer.get_action(env, team=0)
    cs = a["channel_select"]
    # All enemy radars should be on ch0, ignoring victim's [3,5]
    assert (cs == 0).all(), f"TrueFixed leaked victim channel: cs={cs.unique()}"


def test_adaptive_split_follows_victim_channels():
    """AdaptiveSpectrum: enemy radar i → victim radar i's channel."""
    env = _make_env()
    env.reset()
    E, R = env.E, env.n_radars_per_team
    dev = env.device
    # Victim team 1 on channels [2, 6]
    ch_freqs = env.fc_hz + torch.tensor([2, 6], device=dev).float() * env.channel_spacing_hz
    freqs = env.radar_freq_hz.clone()
    freqs[:, 1, :] = ch_freqs.unsqueeze(0).expand(E, R)
    env.set_radar_freqs(freqs)

    jammer = AdaptiveSpectrumJammer(jam_fraction=1e-4)
    a = jammer.get_action(env, team=0)
    cs = a["channel_select"]   # [E, R]
    # Enemy radar 0 should be on ch2, radar 1 on ch6
    assert (cs[:, 0] == 2).all(), f"enemy radar 0 should be ch2, got {cs[:, 0].unique()}"
    assert (cs[:, 1] == 6).all(), f"enemy radar 1 should be ch6, got {cs[:, 1].unique()}"


def test_adaptive_beam_targets_per_pair():
    """AdaptiveSpectrum: enemy radar i beams at victim radar i."""
    env = _make_env()
    env.reset()
    jammer = AdaptiveSpectrumJammer(jam_fraction=1e-4)
    a = jammer.get_action(env, team=0)
    bt = a["beam_target"]
    R = env.n_radars_per_team
    # beam_target should be [0, 1, 2, ..., R-1] per env
    expected = torch.arange(R, device=env.device).unsqueeze(0).expand(env.E, R)
    assert torch.equal(bt, expected), f"beam_target should be diag, got {bt[0]}"


def test_adaptive_jam_fraction_apple_to_apple():
    """AdaptiveSpectrum f_emit == jam_fraction (apple-to-apple with fixed)."""
    env = _make_env()
    env.reset()
    jam_frac = 1e-4
    jammer = AdaptiveSpectrumJammer(jam_fraction=jam_frac)
    a = jammer.get_action(env, team=0)
    f_emit = a["task_alloc"][..., :3].sum(dim=-1).max().item()
    assert abs(f_emit - jam_frac) < 1e-6, f"f_emit {f_emit} != jam_fraction {jam_frac}"


def test_adaptive_follows_victim_change():
    """When victim changes channel, AdaptiveSpectrum follows next step."""
    env = _make_env()
    env.reset()
    jammer = AdaptiveSpectrumJammer(jam_fraction=1e-4)
    dev = env.device

    # Step 1: victim on ch0
    a1 = jammer.get_action(env, team=0)
    assert (a1["channel_select"][:, 0] == 0).all()

    # Move victim team 1 to ch5
    E, R = env.E, env.n_radars_per_team
    freqs = env.radar_freq_hz.clone()
    freqs[:, 1, :] = (env.fc_hz + 5 * env.channel_spacing_hz)
    env.set_radar_freqs(freqs)

    a2 = jammer.get_action(env, team=0)
    assert (a2["channel_select"][:, 0] == 5).all(), \
        f"adaptive should follow to ch5, got {a2['channel_select'][:, 0].unique()}"


def test_adaptive_runs_full_episode():
    """AdaptiveSpectrum runs cleanly with env.step for a full episode."""
    env = _make_env(n_envs=4, episode_steps=30)
    jammer = AdaptiveSpectrumJammer(jam_fraction=1e-3)

    env.reset()
    for step in range(30):
        a_jam = jammer.get_action(env, team=0)
        E, R = env.E, env.n_radars_per_team
        a_opp = {
            "task_alloc": torch.ones(E, R, 4, device=env.device) / 4,
            "beam_target": torch.zeros(E, R, dtype=torch.long, device=env.device),
            "laser_target": torch.zeros(E, dtype=torch.long, device=env.device),
            "emission_on": torch.ones(E, R, device=env.device),
            "freq_hop_rate": torch.ones(E, R, device=env.device),
            "channel_select": torch.zeros(E, R, dtype=torch.long, device=env.device),
        }
        action = combine_team_actions(env, a_jam, a_opp)
        obs, reward, done, info = env.step(action)
        if done.all():
            break

    assert not torch.isnan(env.tracker_P).any(), "tracker_P NaN after episode"
    assert not torch.isnan(env.radar_freq_hz).any(), "radar_freq_hz NaN"


if __name__ == "__main__":
    test_true_fixed_constant_channel()
    test_adaptive_split_follows_victim_channels()
    test_adaptive_beam_targets_per_pair()
    test_adaptive_jam_fraction_apple_to_apple()
    test_adaptive_follows_victim_change()
    test_adaptive_runs_full_episode()
    print("All AdaptiveSpectrumJammer + TrueFixedJammer tests PASS")
