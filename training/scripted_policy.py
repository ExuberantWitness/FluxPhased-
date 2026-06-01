"""Scripted (heuristic) radar + commander policies for demonstration data generation.

Two implementations:
  - `scripted_radar_policy`  / `scripted_commander_policy`  — simple round-robin (legacy)
  - `hpedf_radar_policy`    / `hpedf_commander_policy`      — HPEDF per-element scheduler

HPEDF treats each of the 625 elements as an independent mini-radar.  Each
element is scored across all 4 tasks (recon/detect/jam/comm) by a synthetic
priority function, then elements are assigned greedily within dynamic quotas
determined by the current tactical scene (Recon/Detect/Jam/Comm-dominant).

Scene classification uses intercept SNR thresholds from EW literature:
  >= 18 dB  → Jam-dominant  (reliable parameter measurement possible)
  13-18 dB  → elevated Jam
  < 13 dB   → stealthy (no Jam needed)

All computation is fully vectorised on GPU.
"""

import torch
import math


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

def _compute_az_el(radar_pos, target_pos):
    """Azimuth / elevation from radar_pos [E,R,3] to target_pos [E,T,3]."""
    tgt = target_pos[:, 0:1, :]  # [E, 1, 3] — first target
    diff = tgt - radar_pos  # [E, R, 3]
    dx, dy, dz = diff[..., 0], diff[..., 1], diff[..., 2]
    dist_xy = torch.sqrt(dx ** 2 + dy ** 2).clamp(min=1.0)
    az = torch.atan2(dy, dx)
    el = torch.atan2(dz, dist_xy)
    return az, el


def _az_el_to_action(az, el, max_az_deg=60.0, max_el_deg=45.0):
    """Convert radians to [0.01, 0.99] action range."""
    max_az_rad = math.radians(max_az_deg)
    max_el_rad = math.radians(max_el_deg)
    az_a = (az / (2 * max_az_rad) + 0.5).clamp(0.01, 0.99)
    el_a = (el / (2 * max_el_rad) + 0.5).clamp(0.01, 0.99)
    return az_a, el_a


# ---------------------------------------------------------------------------
# Scene constants (EW literature — SNR thresholds)
# ---------------------------------------------------------------------------

SNR_JAM_DOMINANT = 18.0   # dB — reliable parameter measurement + DOA
SNR_DETECTED     = 13.0   # dB — basic detection (PD=90%, PFA=1e-6)

# Scene quotas: [detect, jam, comm, recon]  (normalised, sum=1.0)
SCENE_QUOTAS = {
    "recon":   torch.tensor([0.25, 0.15, 0.10, 0.50]),
    "detect":  torch.tensor([0.45, 0.20, 0.15, 0.20]),
    "jam":     torch.tensor([0.20, 0.40, 0.20, 0.20]),
    "comm":    torch.tensor([0.30, 0.15, 0.35, 0.20]),
}

# Task priority weights (for synthetic score)
W_TASK = torch.tensor([0.1, 0.4, 0.3, 0.2])  # recon, detect, jam, comm


# ---------------------------------------------------------------------------
# HPEDF Scheduler
# ---------------------------------------------------------------------------

class HPEDFScheduler:
    """Per-element HPEDF task scheduler for one team.

    Each call to __call__() returns the per-element action block for the
    team's radars — [E, n_own_radars, N, 22] — ready to be assembled into
    the full flat action tensor.
    """

    def __init__(self):
        self.augment_noise = 0.0  # 0.0=no noise; 1.0=full augmentation

    def __call__(self, env, team: int):
        """Run one scheduling cycle.

        Returns
        -------
        elem_actions : [E, R_own, N, 22]  (task + beams + params)
        """
        E = env.num_envs
        R = env.n_radars
        N = env.n_elem
        dev = env.device

        n_teams = 2
        r_per_team = R // n_teams
        r_start = team * r_per_team
        r_end = r_start + r_per_team
        n_own = r_end - r_start

        radar_pos = env.radar_pos                          # [E, R, 3]
        own_pos = radar_pos[:, r_start:r_end, :]           # [E, R_own, 3]

        # ── 1. Tactical scene assessment ──
        scene, intercept_snr = self._classify_scene(env, team, r_start, r_end)
        quotas = SCENE_QUOTAS[scene].to(dev).clone()       # [4]

        # ── Augmentation: SCENE_QUOTAS ±5% Gaussian noise ──
        if self.augment_noise > 0:
            noise = torch.randn(4, device=dev) * 0.05 * self.augment_noise
            quotas = (quotas + noise).clamp(0.05, 0.55)  # keep within valid range
            quotas = quotas / quotas.sum()  # re-normalise

        # ── 2. Per-element geometry scores ──
        # Target direction (detect)
        target_az, target_el = _compute_az_el(own_pos, env.target_pos)
        ta, te = _az_el_to_action(target_az, target_el)    # [E, R_own]

        # Enemy direction (jam)
        enemy_team = 1 - team
        enemy_r0 = enemy_team * r_per_team
        enemy_r1 = enemy_r0 + r_per_team
        enemy_pos = radar_pos[:, enemy_r0:enemy_r1, :].mean(dim=1, keepdim=True)  # [E, 1, 3]
        enemy_az, enemy_el = _compute_az_el(own_pos, enemy_pos)
        ja, je = _az_el_to_action(enemy_az, enemy_el)     # [E, R_own]

        # Missile direction (comm) — team's own missile
        missile = env.battlefield.missile
        m_pos = missile.missile_pos[:, team:team+1, :]     # [E, 1, 3]
        m_az, m_el = _compute_az_el(own_pos, m_pos)
        ma, me = _az_el_to_action(m_az, m_el)              # [E, R_own]

        # Recon scan pattern — per-element phase
        scan_phase = (torch.arange(N, device=dev).float() / N) * 2 * math.pi
        sa = (0.5 + 0.4 * torch.sin(scan_phase)).view(1, 1, N)  # [1, 1, N]
        se = (0.5 + 0.2 * torch.cos(scan_phase)).view(1, 1, N)

        # ── 3. Build per-element geometry alignment score ──
        geom_detect = 1.0  # all elements can detect; direction via ta/te
        geom_jam    = 1.0  # same — use ja/je
        geom_comm   = 1.0
        geom_recon  = 1.0  # scan already diversified

        # ── 4. Threat score (intercept SNR) ──
        threat = torch.zeros(E, n_own, N, device=dev)
        if intercept_snr is not None:
            # Normalise: 18 dB → 1.0, 0 dB → 0.0
            snr_clipped = intercept_snr.clamp(0.0, SNR_JAM_DOMINANT)
            threat = snr_clipped / SNR_JAM_DOMINANT  # [E, R_own] → [E, R_own, 1]
            threat = threat.unsqueeze(-1).expand(-1, -1, N)

        # ── 5. Synthetic priority per element × task ──
        # S(e, t) = w_task[t] + w_geom + w_threat
        S = torch.zeros(E, n_own, N, 4, device=dev)

        # w_task (same for all elements) — with optional augmentation
        w_task = W_TASK.to(dev).clone()
        if self.augment_noise > 0:
            task_noise = torch.randn(4, device=dev) * 0.10 * self.augment_noise
            w_task = (w_task + task_noise).clamp(0.02, 0.6)  # keep valid
        S[..., :] = w_task.view(1, 1, 1, 4)  # [1,1,1,4]

        # w_threat — jam and detect benefit from high threat
        S[..., 1] += threat.squeeze(-1) * 0.5  # detect
        S[..., 2] += threat.squeeze(-1) * 1.0  # jam  (stronger incentive)

        # ── 6. Greedy assignment ──
        task_ids = torch.full((E, n_own, N), 0, dtype=torch.long, device=dev)  # default recon
        assigned = torch.zeros(E, n_own, N, dtype=torch.bool, device=dev)

        # For each task type (comm first — highest scene priority override)
        task_order = [3, 2, 1, 0]  # comm, jam, detect, recon  (recon last = fill)
        for t in task_order:
            target_n = int(quotas[t].item() * N)
            if target_n <= 0:
                continue
            # Scores for this task, masked by already-assigned
            S_t = S[..., t].clone()  # [E, R_own, N]
            S_t[assigned] = -float('inf')
            # Pick top target_n elements per (env, radar)
            _, top_idx = S_t.reshape(E * n_own, N).topk(target_n, dim=-1)  # [E*n_own, target_n]
            # Build flat indices
            e_r_idx = torch.arange(E * n_own, device=dev).unsqueeze(-1).expand(-1, target_n)
            task_ids_flat = task_ids.reshape(E * n_own, N)
            task_ids_flat[e_r_idx, top_idx] = t
            assigned_flat = assigned.reshape(E * n_own, N)
            assigned_flat[e_r_idx, top_idx] = True

        # ── 7. Build per-element action block ──
        # task one-hot [E, R_own, N, 4]
        task_onehot = torch.zeros(E, n_own, N, 4, device=dev)
        task_onehot.scatter_(-1, task_ids.unsqueeze(-1), 1.0)

        # Beam steering per element
        # Selectors: [E, R_own, N]
        is_detect = (task_ids == 1).float()
        is_jam    = (task_ids == 2).float()
        is_comm   = (task_ids == 3).float()
        is_recon  = (task_ids == 0).float()

        az_a = ta.unsqueeze(-1).expand(-1, -1, N)  # [E, R_own, N]
        el_a = te.unsqueeze(-1).expand(-1, -1, N)
        j_az = ja.unsqueeze(-1).expand(-1, -1, N)
        j_el = je.unsqueeze(-1).expand(-1, -1, N)
        m_az = ma.unsqueeze(-1).expand(-1, -1, N)
        m_el = me.unsqueeze(-1).expand(-1, -1, N)

        beam_az = (is_detect * az_a + is_jam * j_az +
                   is_comm * m_az + is_recon * sa.expand(E, n_own, N))  # [E, R_own, N, 1]
        beam_el = (is_detect * el_a + is_jam * j_el +
                   is_comm * m_el + is_recon * se.expand(E, n_own, N))

        beam_az = beam_az.squeeze(-1).unsqueeze(-1)  # [E, R_own, N, 1]
        beam_el = beam_el.squeeze(-1).unsqueeze(-1)

        # 4 beams each
        beam_az4 = beam_az.expand(-1, -1, -1, 4)
        beam_el4 = beam_el.expand(-1, -1, -1, 4)
        beams_8 = torch.stack([beam_az4[..., 0], beam_el4[..., 0],
                               beam_az4[..., 1], beam_el4[..., 1],
                               beam_az4[..., 2], beam_el4[..., 2],
                               beam_az4[..., 3], beam_el4[..., 3]], dim=-1)

        # Task-specific params (per-element)
        detect_params = torch.tensor([0.5, 0.3, 0.5], device=dev).view(1, 1, 1, 3).expand(E, n_own, N, -1)
        jam_params    = torch.tensor([0.7, 0.8, 0.9], device=dev).view(1, 1, 1, 3).expand(E, n_own, N, -1)
        comm_params   = torch.tensor([0.3, 0.2, 0.5, 0.6], device=dev).view(1, 1, 1, 4).expand(E, n_own, N, -1)

        elem_actions = torch.cat([
            task_onehot,    # [E, R_own, N, 4]
            beams_8,        # [E, R_own, N, 8]
            detect_params,  # [E, R_own, N, 3]
            jam_params,     # [E, R_own, N, 3]
            comm_params,    # [E, R_own, N, 4]
        ], dim=-1)

        return elem_actions  # [E, R_own, N, 22]

    # ------------------------------------------------------------------
    # Scene classification
    # ------------------------------------------------------------------

    def _classify_scene(self, env, team: int, r_start: int, r_end: int):
        """Return (scene_name, intercept_snr_tensor_or_None).

        scene_name in {"recon", "detect", "jam", "comm"}
        intercept_snr: [E, R_own] or None
        """
        missile = env.battlefield.missile
        in_flight = missile.in_flight[:, team]  # [E]

        # If own missile is flying, Comm-dominant
        if in_flight.any():
            return "comm", None

        # Check intercept SNR from env result cache (set by store_transition
        # after previous step).  On step 0 this is None.
        cti = getattr(env, "_cached_cross_team_intercept", None)
        if cti is not None:
            detail_key = f"team{team}_intercept_detail"
            detail = cti.get(detail_key)  # [E, 3]
            if detail is not None:
                # detail[:, 0] = detect_intercept, detail[:, 1] = jam_intercept
                max_snr = detail[:, :2].max(dim=-1).values  # [E] — max of detect/jam intercept
                # Broadcast to per-radar
                snr_per_radar = max_snr.unsqueeze(-1).expand(-1, r_end - r_start)  # [E, R_own]

                if max_snr.max() >= SNR_JAM_DOMINANT:
                    return "jam", snr_per_radar
                elif max_snr.max() >= SNR_DETECTED:
                    return "jam", snr_per_radar  # elevated jam
                else:
                    return "recon", snr_per_radar

        # Default: check if targets exist
        target_pos = env.target_pos
        if target_pos.abs().sum() > 0:
            return "detect", None
        return "recon", None


# ---------------------------------------------------------------------------
# Public API — HPEDF
# ---------------------------------------------------------------------------

_hpedf_scheduler = HPEDFScheduler()


def hpedf_radar_policy(env, team: int):
    """HPEDF per-element radar policy.

    Returns
    -------
    actions : [E, n_radars, N*22 + 3]  flat action tensor
    """
    E = env.num_envs
    R = env.n_radars
    N = env.n_elem
    dev = env.device
    r_per_team = R // 2
    r_start = team * r_per_team
    r_end = r_start + r_per_team
    n_own = r_end - r_start

    # Get per-element actions for own team [E, R_own, N, 22]
    own_elem = _hpedf_scheduler(env, team)

    # Build full per-radar flat action
    # Own radars: HPEDF
    # Opponent radars: neutral (recon-only, zero beam)
    flat_own = own_elem.reshape(E, n_own, N * 22)

    # Recon-only for opponent team (scripted for data generation only)
    opp_start = (1 - team) * r_per_team
    opp_end = opp_start + r_per_team
    n_opp = opp_end - opp_start

    opp_task = torch.zeros(E, n_opp, N, 4, device=dev)
    opp_task[..., 0] = 1.0  # all recon
    opp_rest = torch.zeros(E, n_opp, N, 22 - 4, device=dev)
    opp_elem = torch.cat([opp_task, opp_rest], dim=-1)
    flat_opp = opp_elem.reshape(E, n_opp, N * 22)

    # Assemble full action
    full_flat = torch.zeros(E, R, N * 22, device=dev)
    full_flat[:, r_start:r_end, :] = flat_own
    if n_opp > 0:
        full_flat[:, opp_start:opp_end, :] = flat_opp

    # Vehicle control — random tactical movement
    # Dim 0: speed [0, 1] → 0–60 km/h  (average ~30 km/h)
    # Dim 1: heading change (±60°/step) — small random drift
    # Dim 2: array rotation (±60°/step) — random scan pattern
    vehicle = torch.zeros(E, R, 3, device=dev)
    vehicle[..., 0] = 0.4 + torch.rand(E, R, device=dev) * 0.3   # 0.4–0.7 → 24–42 km/h
    vehicle[..., 1] = 0.4 + torch.rand(E, R, device=dev) * 0.2   # mild heading drift
    vehicle[..., 2] = 0.3 + torch.rand(E, R, device=dev) * 0.4   # array scan variation

    return torch.cat([full_flat, vehicle], dim=-1)


def hpedf_commander_policy(env, team: int, crc_ok=None):
    """HPEDF commander — tactical launch with threat-aware targeting.

    Architecture (from the user's design):
      Phase 1 (Launch): Commander predicts enemy position from 68-dim
          battlefield observation (radar positions + latents from both radars),
          decides launch timing and initial direction.
      Phase 2 (Terminal): After launch, radars track the missile + detect
          enemy radar's real-time position, and communicate updates to the
          missile via BPSK.  CRC reliability is a reward signal, not a launch gate.

    Launch criteria:
      - No missile already in flight for this team
      - Enemy radar is detected (intercept SNR above threshold)
      → Launch toward enemy radar centroid.
    """
    E = env.num_envs
    dev = env.device
    act_dim = env.battlefield.commander_action_dim  # 35

    actions = torch.zeros(E, act_dim, device=dev)

    enemy_team = 1 - team
    r_per_team = env.n_radars // 2
    enemy_r0 = enemy_team * r_per_team
    enemy_r1 = enemy_r0 + r_per_team

    # ── Tactical launch: no CRC gate ──
    # Commander decides launch based on enemy detection, not comm link.
    # CRC is used for mid-flight target updates + comm_reliability reward.
    missile = env.battlefield.missile
    not_flying = ~missile.in_flight[:, team]  # [E]

    # Check if enemy is detected (intercept SNR >= detection threshold)
    cti = getattr(env, "_cached_cross_team_intercept", None)
    if cti is not None:
        detail_key = f"team{enemy_team}_intercept_detail"
        enemy_detail = cti.get(detail_key)  # [E, 3]
        if enemy_detail is not None:
            # detect_intercept (dim 0) or jam_intercept (dim 1) above 13 dB
            enemy_detected = (enemy_detail[:, 0] > 13.0) | (enemy_detail[:, 1] > 13.0)
        else:
            enemy_detected = torch.ones(E, dtype=torch.bool, device=dev)
    else:
        # No intercept data yet — default to detected (will improve with steps)
        enemy_detected = torch.ones(E, dtype=torch.bool, device=dev)

    want_launch = not_flying & enemy_detected & ~env.battlefield.dones

    if want_launch.any():
        actions[want_launch, 0] = 1.0  # launch_flag

        # ── Target: enemy radar centroid ──
        # The Commander's 68-dim obs encodes both own radar positions and
        # radar latents (which contain detection info).  The BC-pretrained
        # Commander learns to predict launch direction from this observation.
        enemy_pos = env.radar_pos[:, enemy_r0:enemy_r1, :]
        enemy_center = enemy_pos.mean(dim=1)  # [E, 3]
        half_x = env.battlefield.map_size[0] / 2.0
        half_y = env.battlefield.map_size[1] / 2.0
        actions[want_launch, 1] = (enemy_center[want_launch, 0] / half_x).clamp(-1, 1)
        actions[want_launch, 2] = (enemy_center[want_launch, 1] / half_y).clamp(-1, 1)

    # Radar instructions: neutral (no commander guidance during pretraining)
    actions[:, 3:35] = 0.5

    return actions


# ---------------------------------------------------------------------------
# Legacy round-robin policy (kept for comparison / fallback)
# ---------------------------------------------------------------------------

def scripted_radar_policy(env, team: int):
    """Heuristic radar policy — fixed round-robin task allocation.

    Task allocation: 40% detect, 20% jam, 20% comm, 20% recon (cycled across elements).
    Kept as baseline for comparison with HPEDF.

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
    target_az, target_el = _compute_az_el(radar_pos, target_pos)
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
    idx = (torch.arange(N, device=dev).unsqueeze(0) % 5)  # [1, N]
    task_id = torch.where(idx <= 1, torch.tensor(1, device=dev),
                 torch.where(idx == 2, torch.tensor(2, device=dev),
                 torch.where(idx == 3, torch.tensor(3, device=dev),
                             torch.tensor(0, device=dev))))
    task_id = task_id.expand(E, -1)  # [E, N]

    # Task one-hot — [E, N, 4]
    task_onehot = torch.zeros(E, N, 4, device=dev)
    task_onehot.scatter_(-1, task_id.unsqueeze(-1), 1.0)

    # Repeat for each radar — [E, R, N, 4]
    task_block = task_onehot.unsqueeze(1).expand(-1, R, -1, -1)

    # Beam steering
    az_a = az_act[:, :, None, None]
    el_a = el_act[:, :, None, None]
    jam_az = jam_az_act[:, :, None, None]
    jam_el = jam_el_act[:, :, None, None]

    is_detect = (task_id == 1).float().unsqueeze(1).unsqueeze(-1).expand(-1, R, -1, -1)
    is_jam    = (task_id == 2).float().unsqueeze(1).unsqueeze(-1).expand(-1, R, -1, -1)
    is_comm   = (task_id == 3).float().unsqueeze(1).unsqueeze(-1).expand(-1, R, -1, -1)
    is_recon  = (task_id == 0).float().unsqueeze(1).unsqueeze(-1).expand(-1, R, -1, -1)

    comm_el_offset = 0.3 * (-1.0 if team == 0 else 1.0)

    scan_phase = (torch.arange(N, device=dev).float() / N) * 2 * math.pi
    scan_az = (0.5 + 0.4 * torch.sin(scan_phase)).view(1, 1, N, 1).expand(E, R, -1, -1)
    scan_el = (0.5 + 0.2 * torch.cos(scan_phase)).view(1, 1, N, 1).expand(E, R, -1, -1)

    az_a_exp = az_a.expand(-1, -1, N, -1)
    el_a_exp = el_a.expand(-1, -1, N, -1)
    jam_az_exp = jam_az.expand(-1, -1, N, -1)
    jam_el_exp = jam_el.expand(-1, -1, N, -1)

    beam_az = (is_detect * az_a_exp + is_jam * jam_az_exp +
               is_comm * 0.5 + is_recon * scan_az)
    beam_el = (is_detect * el_a_exp + is_jam * jam_el_exp +
               is_comm * (0.5 + comm_el_offset) + is_recon * scan_el)

    beam_az4 = beam_az.expand(-1, -1, -1, 4)
    beam_el4 = beam_el.expand(-1, -1, -1, 4)
    beams_8 = torch.stack([beam_az4[..., 0], beam_el4[..., 0],
                           beam_az4[..., 1], beam_el4[..., 1],
                           beam_az4[..., 2], beam_el4[..., 2],
                           beam_az4[..., 3], beam_el4[..., 3]], dim=-1)

    detect_params = torch.tensor([0.5, 0.3, 0.5], device=dev).view(1, 1, 1, 3).expand(E, R, N, -1)
    jam_params = torch.tensor([0.7, 0.8, 0.9], device=dev).view(1, 1, 1, 3).expand(E, R, N, -1)
    comm_params = torch.tensor([0.3, 0.2, 0.5, 0.6], device=dev).view(1, 1, 1, 4).expand(E, R, N, -1)

    elem_actions = torch.cat([
        task_block, beams_8, detect_params, jam_params, comm_params,
    ], dim=-1)

    flat = elem_actions.reshape(E, R, N * 22)
    vehicle = torch.zeros(E, R, 3, device=dev)
    vehicle[..., 0] = 0.5
    vehicle[..., 1] = 0.5

    return torch.cat([flat, vehicle], dim=-1)


def scripted_commander_policy(env, team: int, crc_ok=None):
    """Simple CRC-based commander — launches when comm link established.

    Kept as baseline; HPEDF commander adds CRC accumulation + threat-aware targeting.
    """
    E = env.num_envs
    dev = env.device
    act_dim = env.battlefield.commander_action_dim

    actions = torch.zeros(E, act_dim, device=dev)

    if crc_ok is not None and crc_ok[:, team].any():
        actions[:, 0] = 1.0
        actions[:, 1] = 0.0
        target_y_val = 1.0 if team == 0 else -1.0
        actions[:, 2] = target_y_val

    actions[:, 3:35] = 0.5
    return actions
