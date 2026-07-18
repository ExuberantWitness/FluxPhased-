"""Debug: inspect BC actions and tracker state during rollout."""
import sys
sys.path.insert(0, "/home/ubuntu/CODE/FluxPhased-")
import torch
from env.gpu.twoteam.twoteam_env import TwoTeamVecEnv, RANDOM_GEOMETRY
from algo._shared.baselines.twoteam_blind_classical import BlindClassicalCommander
from algo._shared.baselines.twoteam_strong_rule_commander import TwoTeamStrongRuleCommander
from algo._shared.pilot.twoteam.extreme_commanders import combine_team_actions, STRATEGIES

dev = "cuda"
env = TwoTeamVecEnv(n_envs=4, device=dev, episode_steps=200,
                    geometry=RANDOM_GEOMETRY, seed=42)
env.reset()
print(f"e_kill={env.e_kill} dwell_rate={env.dwell_rate} "
      f"laser_hit_radius_m={env.laser_hit_radius_m} tau_track={env.tau_track}")

bc = BlindClassicalCommander()
sr = TwoTeamStrongRuleCommander()
opp = STRATEGIES["pure_track"]

print("\n=== BC (lt=0) vs pure_track (lt=1) ===")
for step in range(40):
    a0 = bc.get_action(env, 0)
    a1 = opp.get_action(env, 1)
    action = combine_team_actions(env, a0, a1)

    # Pre-step tracker state for team 0
    trace_P_t = env.tracker_P[:, 0, :, 0, 0] + env.tracker_P[:, 0, :, 2, 2]
    init_0 = env.tracker_initialized[:, 0]
    belief_pos_t0 = env.tracker_x[:, 0, :, [0, 2]]   # [E, R, 2] (x, y) per slot

    # Fire-time checks (env line 694-719)
    lt = a0["laser_target"]
    lsr_trace_P = torch.gather(trace_P_t, 1, lt.unsqueeze(1)).squeeze(1)
    lsr_init = init_0.gather(1, lt.unsqueeze(1)).squeeze(1)
    lsr_track_ok = (lsr_trace_P < env.tau_track) & lsr_init
    lsr_belief = torch.gather(env.tracker_x[:, 0], 1,
                              lt.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)
    lsr_belief_pos = lsr_belief[:, [0, 2]]
    enemy_pos = env.radar_pos[:, 1]
    d = (enemy_pos - lsr_belief_pos.unsqueeze(1)).norm(dim=-1)
    enemy_alive = env.radar_alive[:, 1]
    d_masked = torch.where(enemy_alive, d, torch.full_like(d, 1e9))
    nearest_d, nearest_e = d_masked.min(dim=-1)
    hit_mask = nearest_d < env.laser_hit_radius_m
    emit = a0["emission_on"][:, 0] > 0.5

    obs, reward, done, info = env.step(action)

    radar_E_et = env.radar_E[:, 1].sum(-1)   # enemy of team 0 = team 1
    kills_t0 = info["team_kills"][:, 0]

    if step % 5 == 0 or step < 5 or radar_E_et.max() > 0.01:
        e = 0
        print(f"  s={step:3d} lt={lt[e].item()} emit={emit[e].item()} "
              f"track_ok={lsr_track_ok[e].item()} init={lsr_init[e].item()} "
              f"traceP={lsr_trace_P[e]:.3f} "
              f"belief=({lsr_belief_pos[e,0]:.1f},{lsr_belief_pos[e,1]:.1f}) "
              f"nearest_d={nearest_d[e]:.1f} hit={hit_mask[e].item()} "
              f"| radar_E_et={radar_E_et[e]:.3f} kills={kills_t0[e].item()}")

print(f"\nFinal radar_E[:,1] per env: {env.radar_E[:, 1].sum(-1).tolist()}")
print(f"Final team_kills[0]: {info['team_kills'][:, 0].tolist()}")
print(f"Final team_kills[1]: {info['team_kills'][:, 1].tolist()}")

print("\n=== StrongRule (lt=0) vs pure_track (lt=1) — sanity check SR does fire+kill ===")
env.reset()
for step in range(60):
    a0 = sr.get_action(env, 0)
    a1 = opp.get_action(env, 1)
    action = combine_team_actions(env, a0, a1)
    obs, reward, done, info = env.step(action)
print(f"SR vs pure_track final team_kills[0]: {info['team_kills'][:, 0].tolist()}")
print(f"SR vs pure_track final radar_E[:,1]:  {env.radar_E[:, 1].sum(-1).tolist()}")
