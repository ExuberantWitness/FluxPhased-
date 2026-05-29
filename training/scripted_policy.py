"""Scripted (heuristic) radar + commander policies for demonstration data generation.

These policies use known ground-truth state (target positions, radar positions,
etc.) to produce reasonable actions. They are NOT optimal — they provide
"good enough" behavior to bootstrap critic and actor pre-training.

All computation is fully vectorized on GPU — no Python loops, no .item() calls.
"""

import torch
import math


def _compute_target_az_el(radar_pos, target_pos):
    """Compute azimuth and elevation from each radar to its target.

    Args:
        radar_pos: [E, R, 3] — (x, y, z) in meters
        target_pos: [E, n_targets, 3]

    Returns:
        az: [E, R] radians in [-pi, pi]
        el: [E, R] radians in [-pi/2, pi/2]
    """
    tgt = target_pos[:, 0:1, :]  # [E, 1, 3]
    diff = tgt - radar_pos  # [E, R, 3]
    dx, dy, dz = diff[..., 0], diff[..., 1], diff[..., 2]

    dist_xy = torch.sqrt(dx ** 2 + dy ** 2).clamp(min=1.0)
    az = torch.atan2(dy, dx)
    el = torch.atan2(dz, dist_xy)

    return az, el


def _az_el_to_action(az, el, max_az_deg=60.0, max_el_deg=45.0):
    """Convert azimuth/elevation (radians) to 0-1 action range.

    The env decodes: beam_az_rad = (action - 0.5) * 2 * max_az_deg_rad
    So: action = (beam_az_rad / (2 * max_az_deg_rad)) + 0.5
    """
    max_az_rad = math.radians(max_az_deg)
    max_el_rad = math.radians(max_el_deg)
    az_action = (az / (2 * max_az_rad) + 0.5).clamp(0.01, 0.99)
    el_action = (el / (2 * max_el_rad) + 0.5).clamp(0.01, 0.99)
    return az_action, el_action


def scripted_radar_policy(env, team: int):
    """Heuristic radar policy — fully GPU-vectorized.

    Task allocation: 40% detect, 20% jam, 20% comm, 20% recon (cycled across elements).

    Args:
        env: MFARVecEnv instance
        team: 0 (red) or 1 (blue)

    Returns:
        actions: [E, n_radars, N*22 + 3] flat
    """
    E = env.num_envs
    R = env.n_radars
    N = env.n_elem
    dev = env.device

    n_teams = 2
    r_per_team = R // n_teams
    r_start = team * r_per_team
    r_end = r_start + r_per_team

    # Ground truth state
    radar_pos = env.radar_pos  # [E, R, 3]
    target_pos = env.target_pos  # [E, n_targets, 3]

    # Compute target directions per radar — [E, R]
    target_az, target_el = _compute_target_az_el(radar_pos, target_pos)
    az_act, el_act = _az_el_to_action(target_az, target_el)  # [E, R]

    # Enemy direction for jamming — [E, R]
    enemy_team = 1 - team
    enemy_r0 = enemy_team * r_per_team
    enemy_r1 = enemy_r0 + r_per_team
    enemy_pos = radar_pos[:, enemy_r0:enemy_r1, :].mean(dim=1)  # [E, 3]
    enemy_diff = enemy_pos.unsqueeze(1) - radar_pos  # [E, R, 3]
    edx, edy, edz = enemy_diff[..., 0], enemy_diff[..., 1], enemy_diff[..., 2]
    enemy_dist_xy = torch.sqrt(edx ** 2 + edy ** 2).clamp(min=1.0)
    enemy_az = torch.atan2(edy, edx)
    enemy_el = torch.atan2(edz, enemy_dist_xy)
    jam_az_act, jam_el_act = _az_el_to_action(enemy_az, enemy_el)  # [E, R]

    # Task allocation: cycle [detect, detect, jam, comm, recon] across elements
    # task_ids in {0=recon, 1=detect, 2=jam, 3=comm} matching env action layout
    idx = (torch.arange(N, device=dev).unsqueeze(0) % 5)  # [1, N]
    # idx: 0→detect, 1→detect, 2→jam, 3→comm, 4→recon
    task_id = torch.where(idx <= 1, torch.tensor(1, device=dev),  # detect
                 torch.where(idx == 2, torch.tensor(2, device=dev),  # jam
                 torch.where(idx == 3, torch.tensor(3, device=dev),  # comm
                             torch.tensor(0, device=dev))))  # recon
    task_id = task_id.expand(E, -1)  # [E, N]

    # Task one-hot — [E, N, 4]
    task_onehot = torch.zeros(E, N, 4, device=dev)
    task_onehot.scatter_(-1, task_id.unsqueeze(-1), 1.0)

    # Repeat for each radar — [E, R, N, 4]
    task_block = task_onehot.unsqueeze(1).expand(-1, R, -1, -1)

    # Beam steering: 4 beams per element (dims 4-11 in element action)
    # Each beam gets az/el based on task
    # [E, R] → [E, R, 1, 1]
    az_a = az_act[:, :, None, None]   # [E, R, 1, 1]
    el_a = el_act[:, :, None, None]   # [E, R, 1, 1]
    jam_az = jam_az_act[:, :, None, None]  # [E, R, 1, 1]
    jam_el = jam_el_act[:, :, None, None]  # [E, R, 1, 1]

    # Task masks per element — [E, N] → [E, 1, N, 1] → [E, R, N, 1]
    is_detect = (task_id == 1).float().unsqueeze(1).unsqueeze(-1).expand(-1, R, -1, -1)  # [E, R, N, 1]
    is_jam    = (task_id == 2).float().unsqueeze(1).unsqueeze(-1).expand(-1, R, -1, -1)
    is_comm   = (task_id == 3).float().unsqueeze(1).unsqueeze(-1).expand(-1, R, -1, -1)
    is_recon  = (task_id == 0).float().unsqueeze(1).unsqueeze(-1).expand(-1, R, -1, -1)

    # Comm direction: team 0 → negative el offset, team 1 → positive
    comm_el_offset = 0.3 * (-1.0 if team == 0 else 1.0)

    # Scan phase per element — [N] → [1, 1, N, 1]
    scan_phase = (torch.arange(N, device=dev).float() / N) * 2 * math.pi
    scan_az_vals = 0.5 + 0.4 * torch.sin(scan_phase)  # [N]
    scan_el_vals = 0.5 + 0.2 * torch.cos(scan_phase)  # [N]
    scan_az = scan_az_vals.view(1, 1, N, 1).expand(E, R, -1, -1)
    scan_el = scan_el_vals.view(1, 1, N, 1).expand(E, R, -1, -1)

    # Pre-expand steering targets to [E, R, N, 1]
    az_a_exp = az_a.expand(-1, -1, N, -1)
    el_a_exp = el_a.expand(-1, -1, N, -1)
    jam_az_exp = jam_az.expand(-1, -1, N, -1)
    jam_el_exp = jam_el.expand(-1, -1, N, -1)

    # Build beam az/el for 4 beams (dims 4,5 | 6,7 | 8,9 | 10,11)
    # Detect → target; Jam → enemy; Comm → own side; Recon → scan
    beam_az = (is_detect * az_a_exp + is_jam * jam_az_exp +
               is_comm * 0.5 + is_recon * scan_az)  # [E, R, N, 1]
    beam_el = (is_detect * el_a_exp + is_jam * jam_el_exp +
               is_comm * (0.5 + comm_el_offset) + is_recon * scan_el)  # [E, R, N, 1]

    # 4 beams: each gets beam_az, beam_el
    beam_az4 = beam_az.expand(-1, -1, -1, 4)  # [E, R, N, 4]
    beam_el4 = beam_el.expand(-1, -1, -1, 4)  # [E, R, N, 4]
    # Interleave: az, el, az, el, az, el, az, el → 8 values
    beams_8 = torch.stack([beam_az4[..., 0], beam_el4[..., 0],
                           beam_az4[..., 1], beam_el4[..., 1],
                           beam_az4[..., 2], beam_el4[..., 2],
                           beam_az4[..., 3], beam_el4[..., 3]], dim=-1)  # [E, R, N, 8]

    # Detection params: [center_freq, bw, pulse_width] — dims 12,13,14
    detect_params = torch.tensor([0.5, 0.3, 0.5], device=dev).view(1, 1, 1, 3).expand(E, R, N, -1)

    # Jam params: [center_freq, bw, power] — dims 15,16,17
    jam_params = torch.tensor([0.7, 0.8, 0.9], device=dev).view(1, 1, 1, 3).expand(E, R, N, -1)

    # Comm params: [center_freq, bw, mod_index, tx_power] — dims 18,19,20,21
    comm_params = torch.tensor([0.3, 0.2, 0.5, 0.6], device=dev).view(1, 1, 1, 4).expand(E, R, N, -1)

    # Concatenate element actions: task(4) + beams(8) + detect(3) + jam(3) + comm(4) = 22
    elem_actions = torch.cat([
        task_block,        # [E, R, N, 4]
        beams_8,           # [E, R, N, 8]
        detect_params,     # [E, R, N, 3]
        jam_params,        # [E, R, N, 3]
        comm_params,       # [E, R, N, 4]
    ], dim=-1)  # [E, R, N, 22]

    # Flatten to [E, R, N*22]
    flat = elem_actions.reshape(E, R, N * 22)

    # Vehicle control: [speed, heading, _] — neutral
    vehicle = torch.zeros(E, R, 3, device=dev)
    vehicle[..., 0] = 0.5  # neutral speed
    vehicle[..., 1] = 0.5  # neutral heading

    # Full flat action: [E, R, N*22 + 3]
    actions = torch.cat([flat, vehicle], dim=-1)  # [E, R, N*22 + 3]

    return actions


def scripted_commander_policy(env, team: int, crc_ok=None):
    """Heuristic commander policy — fully GPU-vectorized.

    Launches missile when comm link is established.

    Args:
        env: MFARVecEnv instance
        team: 0 (red) or 1 (blue)
        crc_ok: [E, n_teams] bool tensor (optional, from previous step)

    Returns:
        commander_actions: [E, commander_action_dim]
    """
    E = env.num_envs
    dev = env.device
    act_dim = env.battlefield.commander_action_dim  # 35

    actions = torch.zeros(E, act_dim, device=dev)

    if crc_ok is not None and crc_ok[:, team].any():
        actions[:, 0] = 1.0  # launch_flag
        actions[:, 1] = 0.0  # target_x (center)
        target_y_val = 1.0 if team == 0 else -1.0
        actions[:, 2] = target_y_val

    # Radar instructions: neutral 0.5 for all 32 instruction dims (2 radars × 16)
    actions[:, 3:35] = 0.5

    return actions
