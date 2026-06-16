"""Laser drone weapon training: pulse-level PPO self-play.

Pulse-level loop: env.step() runs every 100μs (1 pulse).
CPI accumulation + FFT runs every 4 pulses.
NN decision + PPO transition storage runs every 5 pulses (2kHz).
PPO update runs when buffer fills.

Usage:
    python -m training.train_laser --config configs/laser_25x25_config.yaml
"""

import argparse
import time
import yaml
import torch
import torch.nn as nn
import numpy as np

from radar_sim.gpu.vec_mfar_env import MFARVecEnv
from training.radar_policy import CPIAccumulator
from training.ppo.actor_critic import (
    SubArrayRadarActorCritic,
    CommanderActorCritic,
    TeamCritic,
    build_team_state,
)
from training.ppo.buffer import RolloutBuffer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(cfg: dict) -> MFARVecEnv:
    env_cfg = cfg.get("env", {})
    return MFARVecEnv(
        num_envs=env_cfg.get("num_envs", 4),
        n_radars=env_cfg.get("n_radars", 4),
        rows=env_cfg.get("rows", 25),
        cols=env_cfg.get("cols", 25),
        fc=10e9,
        bandwidth=env_cfg.get("bandwidth", 200e6),
        prf=env_cfg.get("prf", 10e3),
        pulses_per_cpi=env_cfg.get("pulses_per_cpi", 4),
        fft_size=env_cfg.get("fft_size", 64),
        tx_power_w=env_cfg.get("tx_power_w", 1.0),
        n_teams=env_cfg.get("n_teams", 2),
        device=env_cfg.get("device", "cuda"),
        kill_radius_m=env_cfg.get("kill_radius_m", 0.2),
        illumination_time_s=env_cfg.get("illumination_time_s", 0.002),
        drone_altitude_m=env_cfg.get("drone_altitude_m", 3000.0),
        map_size=env_cfg.get("map_size", [20000.0, 20000.0]),
        vehicle_speed_ms=env_cfg.get("vehicle_speed_ms", 20.0),
        reward_config=cfg.get("reward_shaping", {}),
    )


def build_actors(cfg: dict, n_elem: int, n_pulses: int, n_bins: int, device: str):
    sub_size = cfg.get("sub_array_size", 5)
    num_output_length = 16
    # CTDE privileged critic dim for commander — 0 disables (uses deployment value head only)
    commander_priv_dim = cfg.get("training", {}).get("commander_privileged_dim", 0)

    radar_ac = SubArrayRadarActorCritic(
        n_elem=n_elem,
        n_pulses=n_pulses,
        n_bins=n_bins,
        sub_array_size=sub_size,
        commander_instr_dim=num_output_length,
    ).to(device)

    commander_ac = CommanderActorCritic(
        obs_dim=76,
        act_dim=5,
        hidden_dim=256,
        privileged_dim=commander_priv_dim,
        # action[0] = fire is a discrete trigger → model it with a Bernoulli head so
        # PPO can assign advantage to the fire decision (learned, never forced).
        hybrid_fire=cfg.get("training", {}).get("hybrid_fire", True),
        # decouple the value trunk so value-loss gradient stops churning the policy trunk
        # → the residual aim head can converge its mean to ~0 (sub-meter), not ~1-2m.
        decouple_value=cfg.get("training", {}).get("decouple_value", True),
    ).to(device)

    return radar_ac, commander_ac


class LaserTrainer:
    """Pulse-level PPO trainer for laser drone weapon system.

    Self-play: one shared (radar + commander) policy plays both teams.
    Transitions collected from both teams, PPO updates shared networks.
    """

    def __init__(
        self,
        env: MFARVecEnv,
        radar_ac: nn.Module,
        commander_ac: nn.Module,
        cfg: dict,
    ):
        self.env = env
        self.radar_ac = radar_ac
        self.commander_ac = commander_ac
        self.cfg = cfg
        self.device = cfg.get("env", {}).get("device", "cuda")
        dev = torch.device(self.device)

        # --- PSRO-lite league ---------------------------------------------------------
        # When enabled, team 0 (red) is the TRAINING policy (transitions collected, PPO'd);
        # team 1 (blue) is a FROZEN opponent sampled from a pool of past snapshots. This
        # tests adversarial robustness against a population instead of plain self-play.
        import copy as _copy
        self.league = bool(cfg.get("training", {}).get("league", False))
        if self.league:
            self.radar_opp = _copy.deepcopy(radar_ac).to(dev)
            self.commander_opp = _copy.deepcopy(commander_ac).to(dev)
            for p in self.radar_opp.parameters(): p.requires_grad_(False)
            for p in self.commander_opp.parameters(): p.requires_grad_(False)
            self.pool = []          # list of (radar_state, commander_state) snapshots
            self.pool_winrate = []  # PFSP: opponent i's win-rate vs current policy (lower = harder)
        else:
            self.radar_opp = self.commander_opp = None
            self.pool = []

        E = env.num_envs
        R = env.n_radars
        N = env.n_elem
        P = env.n_pulses
        S = env.n_samples
        n_bins = env.n_bins

        self.E, self.R, self.N, self.P, self.S, self.n_bins = E, R, N, P, S, n_bins

        ppo_cfg = cfg.get("ppo", {})
        shared = ppo_cfg.get("shared", {})
        cmd_cfg = ppo_cfg.get("commander", {})
        rad_cfg = ppo_cfg.get("radar", {})

        self.gamma = shared.get("gamma", 0.999)
        self.gae_lambda = shared.get("gae_lambda", 0.99)
        self.n_epochs = shared.get("n_epochs", 3)
        self.batch_size = shared.get("batch_size", 128)
        self.max_grad_norm = shared.get("max_grad_norm", 0.5)
        self.value_coef = shared.get("value_coef", 0.5)
        # Scalar clamp on value predictions to prevent value_loss spikes from
        # destabilizing the shared actor/critic trunk (PPO 37-details pattern).
        # None disables clipping.
        self.vf_clip_range = shared.get("vf_clip_range", None)

        # PPO optimizers
        self.radar_optimizer = torch.optim.Adam(
            radar_ac.parameters(),
            lr=rad_cfg.get("lr", 1e-4),
        )
        self.commander_optimizer = torch.optim.Adam(
            commander_ac.parameters(),
            lr=cmd_cfg.get("lr", 3e-4),
        )

        self.radar_clip = rad_cfg.get("clip_range", 0.1)
        self.commander_clip = cmd_cfg.get("clip_range", 0.2)
        self.radar_entropy = rad_cfg.get("entropy_coef", 0.02)
        self.commander_entropy = cmd_cfg.get("entropy_coef", 0.01)

        # MAPPO: centralized team critic (optional). When enabled, replaces
        # commander's per-agent value_head for advantage computation. Sees
        # global team state (commander_obs + alive + beam_hit_time + ...).
        # Decentralized actor (commander_ac) still uses local 76-dim obs.
        self.use_mappo = cfg.get("training", {}).get("use_mappo", False)
        if self.use_mappo:
            self.team_critic = TeamCritic(input_dim=104, hidden_dim=256).to(self.device)
            # Init last layer bias to expected return (~2.8) like commander value_head
            last = self.team_critic.net[-1]
            torch.nn.init.orthogonal_(last.weight, gain=1.0)
            torch.nn.init.constant_(last.bias, 2.8)
            self.team_critic_optimizer = torch.optim.Adam(
                self.team_critic.parameters(),
                lr=cmd_cfg.get("team_critic_lr", 1e-3),
            )
        else:
            self.team_critic = None
            self.team_critic_optimizer = None

        # CPI accumulator
        self.cpi_buffer = CPIAccumulator(E, R, N, P, S, device=self.device)
        self._spectrum = None

        # Cached actions (reused between NN control steps)
        self._cached_radar_action = None
        self._cached_cmd_action = torch.zeros(E, env.n_teams, 5, device=dev)
        self._cached_tx = torch.zeros(E, R, N, S, dtype=torch.complex64, device=dev)
        self._cached_veh = torch.zeros(E, R, 3, device=dev)

        # Pulse counter
        self._pulse_count = 0
        self.pulses_per_control = cfg.get("env", {}).get("pulses_per_control", 5)

        # Element positions for TX assembly
        dx_m = 0.5 * env.array.wavelength
        dy_m = 0.5 * env.array.wavelength
        x_pos = (np.arange(env.cols) - (env.cols - 1) / 2.0) * dx_m
        y_pos = (np.arange(env.rows) - (env.rows - 1) / 2.0) * dy_m
        X, Y = np.meshgrid(x_pos, y_pos)
        self.elem_x = torch.tensor(X.ravel().astype(np.float32), device=dev)
        self.elem_y = torch.tensor(Y.ravel().astype(np.float32), device=dev)
        self.wavelength = env.array.wavelength

        # Transition buffer for PPO
        buf_size = shared.get("buffer_size", 2048)
        radar_state_dim = radar_ac.spectrum_flat_dim + radar_ac.comm_flat_dim + radar_ac.recon_flat_dim + radar_ac.other_dim
        radar_action_dim = N * 22 + 3
        cmd_action_dim = 5

        self.radar_buf = RolloutBuffer(
            buf_size, obs_dim=radar_state_dim, act_dim=radar_action_dim,
            gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device,
        )
        self.cmd_buf = RolloutBuffer(
            buf_size, obs_dim=76, act_dim=cmd_action_dim,
            gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device,
            privileged_dim=getattr(commander_ac, "privileged_dim", 0),
        )

        # Reward config
        rc = cfg.get("reward_shaping", {})
        self.kill_bonus = rc.get("kill_bonus", 100.0)
        self.death_penalty = rc.get("death_penalty", -10.0)
        self.illumination_weight = rc.get("illumination_progress_weight", 1.0)
        self.emission_cost = rc.get("emission_cost", -0.001)

        # --- Integrated-EW race: kill the enemy FAST while surviving, via jamming -------
        # The 0.2m kill_radius is a forcing constraint (precise targeting needs good sensing);
        # the GAME is to destroy an enemy radar before they destroy yours, using jamming to
        # degrade their localisation. Commander action[4] (was reserved) = JAM level ∈ [0,1].
        self.race_time_cost = rc.get("race_time_cost", 0.0)       # per-step penalty → 尽快
        self.race_death_penalty = rc.get("race_death_penalty", 0.0)  # extra penalty if own radar dies → 保存自己
        self.jam_gain = rc.get("jam_gain", 0.0)   # enemy jam ×(1+gain·jam) on MY range+crossrange σ (SNR↓)
        self.jam_cost = rc.get("jam_cost", 0.0)   # per-step cost of jamming (emission) → not free
        self._jam_level = torch.zeros(self.E, env.n_teams, device=torch.device(self.device))

        # Dense (1/r²)×t⁴ reward: guides commander toward sustained close illumination
        # r = distance from laser aim to nearest enemy (meters)
        # t = continuous beam-hit time (seconds), accumulates within beam_hit_radius
        # spatial = (r_ref / max(r, r_floor))², clamped to [0, spatial_cap]
        # temporal = ε + (t/t_max)⁴  — ε gives tiny reward even at t=0 (nearby)
        # reward = spatial × temporal × weight
        self.beam_reward_weight = rc.get("beam_reward_weight", 5.0)
        self.beam_hit_radius_m = rc.get("beam_hit_radius_m", 200.0)
        self.beam_r_ref_m = rc.get("beam_r_ref_m", 100.0)
        self.beam_r_floor_m = rc.get("beam_r_floor_m", 1.0)
        self.beam_spatial_cap = rc.get("beam_spatial_cap", 100.0)
        self.beam_temporal_epsilon = rc.get("beam_temporal_epsilon", 0.01)
        self.t_max = env.battlefield.laser.illumination_time_s

        # --- Reshaped reward (removes the spatial-cap dead zone + couples reward to fire) ---
        # Guidance: log-distance potential on the commander's INTENDED aim. Monotone and
        # non-saturating from r_ref → r_floor, so the policy keeps refining aim all the way
        # into kill range — no 95m plateau where (1/r²) used to flatten under the cap.
        self.beam_guidance_weight = rc.get("beam_guidance_weight", self.beam_reward_weight)
        # Illumination: fire-gated dwell within the CURRENT kill_radius (shrunk by curriculum).
        # Paid only when the commander actually fires AND is locked on, so the fire trigger
        # becomes instrumentally valuable and the Bernoulli fire head learns it from reward.
        self.illum_reward_weight = rc.get("illum_reward_weight", 50.0)
        # Dense fire signal: the dwell illumination reward only pays after ~20 continuous
        # locked pulses, too sparse for the Bernoulli fire head to learn from — so it drifts
        # to "never fire" even with perfect aim (eval aim 2m but 0 kills). These give the
        # fire decision an IMMEDIATE per-step signal: + for firing while locked, − for
        # firing while not locked. Makes "fire iff aimed within kill_radius" learnable.
        self.fire_lock_bonus = rc.get("fire_lock_bonus", 5.0)
        self.misfire_penalty = rc.get("misfire_penalty", 0.5)

        # Residual aiming: aim = observed-enemy anchor (obs[68:70]) + policy residual × scale.
        # Absolute tanh-Gaussian aim can't resolve 0.2m on a ±10km range (~700m floor, confirmed
        # by pure-BC plateau). Anchoring at the obs enemy truth and outputting a small ±scale
        # correction makes sub-meter aim reachable; the policy only learns the last-meters refine.
        tcfg = cfg.get("training", {})
        self.residual_aim = bool(tcfg.get("residual_aim", False))
        self.residual_scale_m = float(tcfg.get("residual_scale_m", 100.0))

        # §5 Stage-0: anisotropic sensing noise on the enemy position the commander observes.
        # Models the radar error ellipse — precise along range (bandwidth-limited, cm), poor
        # across range (diffraction-limited, σ_cross = R × crossrange_factor). Weans the
        # commander off perfect ground truth before the real radar→BPSK pipeline (S1–S3).
        # Disabled (both 0) → exact truth (back-compat).
        scfg = cfg.get("sensing_noise", {})
        self.sensing_range_sigma_m = float(scfg.get("range_sigma_m", 0.0))
        self.sensing_crossrange_factor = float(scfg.get("crossrange_factor", 0.0))
        # mode: "single" = one radar (S0 baseline, cross-range wall);
        #       "fused"  = multi-static range triangulation across own radars (S2). Each
        #                  radar is range-precise/cross-range-poor from its own angle;
        #                  information-filter fusion makes BOTH axes range-precise where
        #                  the angular baseline is good — the integrated-sensing advantage.
        #       "tracked"= "fused" + a Kalman filter over time. The radars MOVE (20 m/s),
        #                  so the geometry diversifies frame-to-frame; the KF integrates the
        #                  time-varying (geometry-dependent) measurement covariance, rotating
        #                  & intersecting the error ellipses to collapse the bad-collinear
        #                  (GDOP-limited) directions that floor pure spatial fusion at ~10m.
        #                  [Nardone&Aidala 1981 observer-motion observability; Bar-Shalom KF.]
        self.sensing_mode = scfg.get("mode", "single")
        self.track_q_m = float(scfg.get("track_q_m", 0.05))  # per-step process-noise σ (slow target)
        self.track_burnin = int(scfg.get("track_burnin", 30))  # per-episode warm-start updates
        # Acquisition maneuver: during warm-start the radars sweep perpendicular to their LOS
        # (opposite senses) to actively widen the angular baseline → geometry diversity that
        # collapses bad-collinear GDOP (Nardone&Aidala observer-motion observability). Models
        # the pre-engagement track-while-maneuver phase. 0 → static acquisition.
        self.acq_baseline_m = float(scfg.get("acq_baseline_m", 0.0))
        # 方案1: enforce a minimum deployment baseline between each team's two radars at reset.
        # Two widely-separated radars triangulate the target's precise (cm) RANGE from
        # different angles → sub-0.2m fused position with NO tracking, even for a static
        # target. Random placement sometimes lands the two radars near-collinear (tiny
        # crossing angle → poor fusion); guaranteeing ≥ baseline removes those bad geometries,
        # mirroring how a real SAM battery deliberately spreads its radar vehicles.
        self.min_radar_baseline_m = float(cfg.get("env", {}).get("min_radar_baseline_m", 0.0))
        self._trk_init = False   # lazy per-episode track init
        self._trk_x = None       # [E, T, 2 enemies, 2] track position estimate
        self._trk_P = None       # [E, T, 2 enemies, 2, 2] track covariance

        # Per-team continuous beam-hit time [E, n_teams]
        self._beam_hit_time = torch.zeros(
            self.E, env.n_teams, device=torch.device(self.device),
        )

    def _process_cpi(self):
        """FFT on accumulated CPI data."""
        cpi_data = self.cpi_buffer.data()  # [E, R, N, P, S]
        spectrum = torch.fft.fft(cpi_data, n=self.n_bins, dim=-1)  # [E, R, N, P, n_bins]
        self._spectrum = spectrum.abs().float()
        self.cpi_buffer.reset()

    def _build_radar_obs(self, events: dict) -> torch.Tensor:
        """Build radar observation from spectrum + events."""
        dev = torch.device(self.device)
        E, R, N = self.E, self.R, self.N

        if self._spectrum is not None:
            spec_flat = self._spectrum.reshape(E, R, -1)
        else:
            spec_flat = torch.zeros(E, R, N * self.P * self.n_bins, device=dev)

        comm_flat = torch.zeros(E, R, N * 2, device=dev)
        recon_flat = torch.zeros(E, R, N * 4, device=dev)
        vehicle = torch.zeros(E, R, 5, device=dev)
        laser_state = torch.zeros(E, R, 12, device=dev)
        cmd_instr = torch.zeros(E, R, 16, device=dev)

        if "radar_pos" in events:
            vehicle[:, :, 0] = events["radar_pos"][:, :, 0]
            vehicle[:, :, 1] = events["radar_pos"][:, :, 1]

        obs = torch.cat([spec_flat, comm_flat, recon_flat, vehicle, laser_state, cmd_instr], dim=-1)
        return obs

    def _build_commander_obs(self, events: dict) -> torch.Tensor:
        """Build commander observation from env battlefield."""
        dev = torch.device(self.device)
        radar_latents = torch.zeros(self.E, self.R, 32, device=dev)
        obs = self.env.battlefield.get_commander_observation(
            self.env.radar_pos, radar_latents,
        )
        if self.sensing_range_sigma_m <= 0.0 and self.sensing_crossrange_factor <= 0.0:
            return obs  # exact truth (back-compat)
        if self.sensing_mode in ("fused", "tracked"):
            obs = self._fused_sensing(obs, track=(self.sensing_mode == "tracked"))
            return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)  # belt-and-suspenders
        return self._add_sensing_noise(obs)  # "single" (S0 baseline)

    def _reset_tracks(self):
        """Clear the Kalman track state at the start of each episode."""
        self._trk_init = False
        self._trk_x = None
        self._trk_P = None

    def _enforce_radar_baseline(self):
        """方案1: push each team's two radars apart to at least min_radar_baseline_m,
        keeping their midpoint. Guarantees a good triangulation crossing angle so the
        cm-range fusion localises the target to sub-0.2m without tracking."""
        if self.min_radar_baseline_m <= 0.0:
            return
        rp = self.env.radar_pos  # [E, R, 3]
        for t in range(self.env.n_teams):
            idx = self.env.battlefield.team_radar_indices[t]
            if len(idx) < 2:
                continue
            a, b = int(idx[0]), int(idx[1])
            pa = rp[:, a, :2]
            pb = rp[:, b, :2]
            mid = 0.5 * (pa + pb)
            d = pb - pa
            dist = d.norm(dim=-1, keepdim=True).clamp(min=1.0)
            unit = d / dist
            half = 0.5 * dist.clamp(min=self.min_radar_baseline_m)  # ≥ baseline
            rp[:, a, :2] = mid - unit * half
            rp[:, b, :2] = mid + unit * half

    # --- PSRO-lite league helpers ----------------------------------------------------
    def _snapshot_to_pool(self):
        """Freeze the current training policy as a new opponent in the pool."""
        import copy
        self.pool.append((copy.deepcopy(self.radar_ac.state_dict()),
                          copy.deepcopy(self.commander_ac.state_dict())))
        self.pool_winrate.append(0.5)  # opp i's win-rate vs us (PFSP weight)

    def _sample_opponent(self):
        """PFSP: load an opponent from the pool, weighted toward HARD ones (those that
        beat the current policy), so training prioritises its weaknesses."""
        if not self.pool:
            return
        import numpy as np
        w = np.array(self.pool_winrate) + 0.1  # opp win-rate vs us; +0.1 keeps all sampleable
        w = w / w.sum()
        self._opp_idx = int(np.random.choice(len(self.pool), p=w))
        rs, cs = self.pool[self._opp_idx]
        self.radar_opp.load_state_dict(rs)
        self.commander_opp.load_state_dict(cs)

    def _add_sensing_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """§5-S0: replace the exact enemy positions in obs[68:72] with a radar-realistic
        anisotropic-noisy estimate (range σ small, cross-range σ = R × factor), measured
        from the team's own radar-0. Both the policy AND the residual anchor then use this
        noisy estimate, so the commander aims at what a real radar could know — while the
        kill check still uses ground truth. obs: [E, T, 76].
        """
        if self.sensing_range_sigma_m <= 0.0 and self.sensing_crossrange_factor <= 0.0:
            return obs
        half_x = self.env.map_size[0] / 2.0
        half_y = self.env.map_size[1] / 2.0
        ox = obs[..., 0] * half_x  # own radar-0 (the sensor) position, metres
        oy = obs[..., 1] * half_y
        for k in range(2):  # two enemy radars at obs[68:70] and [70:72]
            off = 68 + 2 * k
            ex = obs[..., off] * half_x
            ey = obs[..., off + 1] * half_y
            dx, dy = ex - ox, ey - oy
            R = torch.sqrt(dx * dx + dy * dy).clamp(min=1.0)
            rx, ry = dx / R, dy / R          # unit vector along range
            cx, cy = -ry, rx                 # unit vector across range
            nr = torch.randn_like(R) * self.sensing_range_sigma_m
            nc = torch.randn_like(R) * (R * self.sensing_crossrange_factor)
            obs[..., off] = (ex + nr * rx + nc * cx) / half_x
            obs[..., off + 1] = (ey + nr * ry + nc * cy) / half_y
        return obs

    def _fuse_one(self, ex, ey, own, sr2, jam_mul=1.0):
        """One information-filter-fused measurement of (ex,ey) from the own radars, with fresh
        anisotropic noise. jam_mul (scalar or [E,T]) is the noise-floor multiplier from enemy
        jamming — it raises BOTH range and cross-range σ (a noise jammer lowers effective SNR),
        so even good-baseline range-triangulation degrades. Returns mean + 2×2 covariance."""
        sr = self.sensing_range_sigma_m * jam_mul   # jamming raises range noise too (SNR↓)
        cf = self.sensing_crossrange_factor * jam_mul
        L00 = torch.zeros_like(ex); L01 = torch.zeros_like(ex); L11 = torch.zeros_like(ex)
        e0 = torch.zeros_like(ex);  e1 = torch.zeros_like(ex)
        for (ox, oy) in own:
            dx, dy = ex - ox, ey - oy
            R = torch.sqrt(dx * dx + dy * dy).clamp(min=1.0)
            rx, ry = dx / R, dy / R          # along-range unit
            cx, cy = -ry, rx                 # cross-range unit
            sc2 = (R * cf) ** 2 + 1e-6
            sr2_eff = sr ** 2 + 1e-9   # jammed range variance (matches the noise added below)
            nr = torch.randn_like(R) * sr
            nc = torch.randn_like(R) * (R * cf)
            mx = ex + nr * rx + nc * cx
            my = ey + nr * ry + nc * cy
            a, b = 1.0 / sr2_eff, 1.0 / sc2
            i00 = a * rx * rx + b * cx * cx
            i01 = a * rx * ry + b * cx * cy
            i11 = a * ry * ry + b * cy * cy
            L00 += i00; L01 += i01; L11 += i11
            e0 += i00 * mx + i01 * my
            e1 += i01 * mx + i11 * my
        det = (L00 * L11 - L01 * L01).clamp(min=1e-9)
        zx = (L11 * e0 - L01 * e1) / det
        zy = (-L01 * e0 + L00 * e1) / det
        return zx, zy, L11 / det, -L01 / det, L00 / det

    @staticmethod
    def _kalman_step(x0, x1, P00, P01, P11, zx, zy, R00, R01, R11, q):
        """One Kalman predict (random walk, +q) + update with measurement (z, R). 2×2 closed form."""
        P00 = P00 + q; P11 = P11 + q
        S00 = P00 + R00; S01 = P01 + R01; S11 = P11 + R11
        sdet = (S00 * S11 - S01 * S01).clamp(min=1e-9)
        Si00 = S11 / sdet; Si01 = -S01 / sdet; Si11 = S00 / sdet
        K00 = P00 * Si00 + P01 * Si01; K01 = P00 * Si01 + P01 * Si11
        K10 = P01 * Si00 + P11 * Si01; K11 = P01 * Si01 + P11 * Si11
        yx = zx - x0; yy = zy - x1
        nx0 = x0 + K00 * yx + K01 * yy
        nx1 = x1 + K10 * yx + K11 * yy
        nP00 = (1 - K00) * P00 - K01 * P01
        nP01 = (1 - K00) * P01 - K01 * P11
        nP11 = -K10 * P01 + (1 - K11) * P11
        return nx0, nx1, nP00, nP01, nP11

    def _fused_sensing(self, obs: torch.Tensor, track: bool = False) -> torch.Tensor:
        """§5-S2: multi-static range triangulation (+ optional Kalman tracking).

        Information-filter fusion of the own radars' anisotropic measurements (range-precise,
        cross-range-poor) makes both axes ~range-precise where the angular baseline is good.
        track=True adds a Kalman filter over time; the moving radars diversify the geometry so
        the KF collapses the bad-collinear (GDOP) directions that floor pure fusion at ~10m.
        Each episode the track is WARM-STARTED with track_burnin pre-convergence updates so the
        anchor is already tight at step 0 — otherwise the within-episode convergence makes the
        commander's observation non-stationary and destabilises PPO. obs: [E, T, 76].
        """
        half_x = self.env.map_size[0] / 2.0
        half_y = self.env.map_size[1] / 2.0
        sr2 = self.sensing_range_sigma_m ** 2
        q = self.track_q_m ** 2
        own = [(obs[..., 0] * half_x, obs[..., 1] * half_y),    # own radar-0 (sensor A)
               (obs[..., 2] * half_x, obs[..., 3] * half_y)]    # own radar-1 (sensor B)
        # Jamming: the ENEMY team's jam level multiplies MY sensing noise floor (range AND
        # cross-range) → degrades my localisation → I can't reach 0.2m → the jammer wins the
        # kill-race. jam_mul [E,T] = 1 + gain · enemy_jam (flip swaps team↔enemy).
        if self.jam_gain > 0.0:
            jam_mul = 1.0 + self.jam_gain * self._jam_level.flip(-1)
        else:
            jam_mul = 1.0
        if track and not self._trk_init:
            E, T = obs.shape[0], obs.shape[1]
            self._trk_x = torch.zeros(E, T, 2, 2, device=obs.device)
            self._trk_P = torch.zeros(E, T, 2, 2, 2, device=obs.device)
        for e in range(2):  # two enemy radars
            off = 68 + 2 * e
            ex = obs[..., off] * half_x      # true enemy position (m) [E, T]
            ey = obs[..., off + 1] * half_y
            zx, zy, R00, R01, R11 = self._fuse_one(ex, ey, own, sr2, jam_mul=jam_mul)
            # Near-collinear geometry makes the fused info matrix near-singular; clamp the
            # estimate to the map so a degenerate frame yields a bounded (wrong) anchor
            # rather than a huge/Inf value that would NaN the network.
            zx = zx.clamp(-half_x, half_x); zy = zy.clamp(-half_y, half_y)
            if not track:
                obs[..., off] = zx / half_x
                obs[..., off + 1] = zy / half_y
                continue
            if not self._trk_init:
                # warm-start: pre-converge with track_burnin fused measurements. If
                # acq_baseline_m>0, the radars sweep perpendicular to their LOS (opposite
                # senses) to actively widen the angular baseline — geometry diversity that
                # collapses bad-collinear GDOP, not just √K noise averaging.
                x0, x1, P00, P01, P11 = zx, zy, R00, R01, R11
                K = max(self.track_burnin, 1)
                for k in range(self.track_burnin):
                    own_k = own
                    if self.acq_baseline_m > 0.0:
                        d = self.acq_baseline_m * ((k + 1) / K - 0.5)  # sweep ±½ baseline
                        own_k = []
                        for ri, (ox, oy) in enumerate(own):
                            dxr, dyr = ex - ox, ey - oy
                            Rr = torch.sqrt(dxr * dxr + dyr * dyr).clamp(min=1.0)
                            sgn = 1.0 if ri == 0 else -1.0   # opposite senses → widen baseline
                            own_k.append((ox - sgn * d * dyr / Rr, oy + sgn * d * dxr / Rr))
                    bzx, bzy, BR00, BR01, BR11 = self._fuse_one(ex, ey, own_k, sr2, jam_mul=jam_mul)
                    bzx = bzx.clamp(-half_x, half_x); bzy = bzy.clamp(-half_y, half_y)
                    x0, x1, P00, P01, P11 = self._kalman_step(
                        x0, x1, P00, P01, P11, bzx, bzy, BR00, BR01, BR11, q)
            else:
                x0 = self._trk_x[..., e, 0]; x1 = self._trk_x[..., e, 1]
                P00 = self._trk_P[..., e, 0, 0]; P01 = self._trk_P[..., e, 0, 1]
                P11 = self._trk_P[..., e, 1, 1]
                x0, x1, P00, P01, P11 = self._kalman_step(
                    x0, x1, P00, P01, P11, zx, zy, R00, R01, R11, q)
            x0 = x0.clamp(-half_x, half_x); x1 = x1.clamp(-half_y, half_y)  # keep track in-map
            self._trk_x[..., e, 0] = x0; self._trk_x[..., e, 1] = x1
            self._trk_P[..., e, 0, 0] = P00; self._trk_P[..., e, 0, 1] = P01
            self._trk_P[..., e, 1, 0] = P01; self._trk_P[..., e, 1, 1] = P11
            obs[..., off] = x0 / half_x
            obs[..., off + 1] = x1 / half_y
        if track:
            self._trk_init = True
        return obs

    def _assemble_tx(self, radar_actions: torch.Tensor) -> torch.Tensor:
        """Convert flat radar actions [E*R, action_dim] → TX signal [E, R, N, S]."""
        E, R, N = self.E, self.R, self.N
        S = self.S
        dev = torch.device(self.device)
        ACTION_PER_ELEM = 22

        actions = radar_actions.reshape(E, R, -1)

        # Decode per-element actions
        elem_actions = actions[:, :, :N * ACTION_PER_ELEM].reshape(E, R, N, ACTION_PER_ELEM)
        task_ids = elem_actions[..., 0:4].argmax(dim=-1)  # [E, R, N]
        beam_az = elem_actions[..., 4] * 60.0  # scale from [0,1] → [-60,60] approx
        beam_el = elem_actions[..., 5] * 45.0
        wf_types = elem_actions[..., 4:6].argmax(dim=-1).long()
        detect_params = elem_actions[..., 12:15]
        jam_params = elem_actions[..., 15:18]
        comm_params = elem_actions[..., 18:22]

        # Beam steering weights
        k = 2.0 * np.pi / self.wavelength
        DEG2RAD = np.pi / 180.0
        az_rad = beam_az.clamp(-90, 90) * DEG2RAD
        el_rad = beam_el.clamp(-90, 90) * DEG2RAD
        u = torch.sin(az_rad) * torch.cos(el_rad)
        v = torch.sin(el_rad)
        phase = -k * (self.elem_x.view(1, 1, N) * u + self.elem_y.view(1, 1, N) * v)
        weights = torch.exp(1j * phase)  # [E, R, N]

        # Simple LFM chirp waveform
        t = torch.linspace(0, S / 200e6, S, device=dev)
        bw = 200e6 * 0.5
        chirp = torch.exp(1j * 2 * np.pi * bw * t**2 / (S / 200e6))  # [S]

        # TX = weights * chirp
        tx = weights.unsqueeze(-1) * chirp.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return tx.to(torch.complex64)

    def _get_rewards(self, result: dict) -> dict:
        """Compute rewards: (1/r²)×t⁴ dense reward + kill bonus + emission cost.

        r = distance from laser aim to nearest alive enemy
        t = continuous beam-hit time (accumulates within beam_hit_radius)

        spatial = (r_ref / max(r, r_floor))², clamped to [0, spatial_cap]
        temporal = ε + (t/t_max)⁴
        beam_reward = spatial × temporal × weight
        """
        dev = torch.device(self.device)
        dt = self.env.pri

        radar_rewards = result.get("radar_rewards", torch.zeros(self.E, self.R, device=dev)).clone()
        cmd_rewards = result.get("commander_rewards", torch.zeros(self.E, self.env.n_teams, device=dev)).clone()

        drone = self.env.battlefield.drone
        radar_pos = self.env.radar_pos

        for t in range(self.env.n_teams):
            enemy_t = 1 - t
            enemy_idx = self.env.battlefield.team_radar_indices[enemy_t]
            enemy_alive = self.env.battlefield.alive[:, enemy_idx]  # [E, R/2]

            if not enemy_alive.any():
                self._beam_hit_time[:, t] = 0.0
                continue

            enemy_pos = radar_pos[:, enemy_idx, :]  # [E, R/2, 3]
            # Use the commander's INTENDED aim (set every step regardless of fire) so the
            # guidance gradient exists even before the policy learns to fire. laser_aim only
            # updates on fire (vec_drone.update_aim), which would hide the aim signal pre-fire.
            aim = drone._commander_aim[:, t, :].unsqueeze(1)  # [E, 1, 3]
            dist_all = (aim - enemy_pos).norm(dim=-1)  # [E, R/2]

            # Mask dead enemies with large distance
            dist_all = dist_all + (~enemy_alive).float() * 1e6
            min_dist = dist_all.min(dim=-1).values  # [E]

            # (A) Guidance: log-distance potential, monotone & non-saturating to r_floor.
            r_eff = min_dist.clamp(min=self.beam_r_floor_m)
            guidance = torch.log(self.beam_r_ref_m / r_eff).clamp(min=0.0)
            beam_reward = guidance * self.beam_guidance_weight

            # (C) Fire-gated illumination: continuous dwell within the CURRENT kill_radius,
            # only credited when the commander fires. Resets if it drifts out or stops firing.
            kill_radius = float(self.env.battlefield.laser.kill_radius_m)  # live (curriculum)
            fire_on = drone._commander_fire[:, t]  # [E] bool — commander's own decision
            locked = (min_dist < kill_radius) & fire_on
            # (C1) Immediate dense reward for firing while locked (teaches the fire trigger).
            beam_reward = beam_reward + locked.float() * self.fire_lock_bonus
            # (C2) Small penalty for firing while NOT locked (discourage spray-firing).
            misfire = fire_on & (min_dist >= kill_radius)
            beam_reward = beam_reward - misfire.float() * self.misfire_penalty
            # (C3) Sustained-dwell illumination — the actual kill mechanic (20 continuous pulses).
            self._beam_hit_time[:, t] = torch.where(
                locked,
                self._beam_hit_time[:, t] + dt,
                torch.zeros_like(self._beam_hit_time[:, t]),
            )
            t_norm = (self._beam_hit_time[:, t] / self.t_max).clamp(0.0, 1.0)
            beam_reward = beam_reward + locked.float() * (t_norm ** 2) * self.illum_reward_weight

            cmd_rewards[:, t] += beam_reward

            # Share a fraction of beam reward with own radars (team reward)
            own_idx = self.env.battlefield.team_radar_indices[t]
            for ri in own_idx:
                radar_rewards[:, ri] += beam_reward * 0.1

        # Kill bonus / death penalty (from env rewards, already included)
        # Emission cost
        radar_rewards += self.emission_cost

        # --- Integrated-EW race terms (kill fast, survive, jamming isn't free) ----------
        bf = self.env.battlefield
        for t in range(self.env.n_teams):
            # 尽快: per-step time cost while the game is undecided → reward fast kills.
            cmd_rewards[:, t] -= self.race_time_cost * (~bf.dones).float()
            # 保存自己: extra penalty the step MY radar is destroyed.
            own_idx = bf.team_radar_indices[t]
            own_dead = (~bf.alive[:, own_idx]).any(dim=-1).float()
            cmd_rewards[:, t] -= self.race_death_penalty * own_dead
            # jamming costs emission (and exposes you) → not free.
            cmd_rewards[:, t] -= self.jam_cost * self._jam_level[:, t]

        return {"radar_rewards": radar_rewards, "commander_rewards": cmd_rewards}

    def _ppo_update(self, ac, optimizer, buffer, clip_range, entropy_coef,
                    bc_weight=0.0, team_critic=None, team_critic_optimizer=None,
                    bc_only=False):
        """Run one PPO update on the buffer.

        Args:
            bc_weight: if > 0, add BC auxiliary loss that supervises commander
                       to copy enemy position from obs[68:70] to action[1:3].
            team_critic: optional MAPPO centralized critic. When provided, uses
                         team_state from buffer and team_critic for value
                         predictions (instead of ac.value_head).
            team_critic_optimizer: required when team_critic is provided.
            bc_only: if True, optimize ONLY the BC loss (skip policy/value/entropy
                     and the team critic). Used for a supervised pre-training phase
                     that locks the commander's pointing BEFORE PPO can perturb it —
                     PPO drift was bouncing the mean aim (eval 70m↔1046m). After the
                     pretrain phase, normal PPO (with a strong BC anchor) refines.
        """
        if buffer.ptr < self.batch_size:
            return {}

        # Compute returns
        with torch.no_grad():
            last_val = torch.zeros(1)
        buffer.compute_returns(last_val)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_bc_loss = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in buffer.get_minibatches(self.batch_size):
                obs = batch["obs"]
                old_actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                priv_info = batch.get("privileged_info")  # None if buffer has no privileged_infos
                team_state = batch.get("team_states") if team_critic is not None else None

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                log_prob, entropy, value, priv_value = ac.evaluate_actions(
                    obs, old_actions, privileged_info=priv_info,
                )

                ratio = torch.exp(log_prob - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Choose critic for value loss:
                # - MAPPO: use team_critic(team_state)
                # - CTDE: use privileged_value (commander's priv head)
                # - Default: use ac.value_head output
                if team_critic is not None and team_state is not None:
                    critic_value = team_critic(team_state)
                elif priv_info is not None:
                    critic_value = priv_value
                else:
                    critic_value = value

                value_pred = critic_value.squeeze(-1)
                if self.vf_clip_range is not None:
                    # Clamp value predictions to return range ± vf_clip_range so
                    # large value_loss spikes (e.g. early-training value=0 vs
                    # returns=2.8) don't dominate the shared-trunk gradient.
                    value_pred = value_pred.clamp(
                        returns.min() - self.vf_clip_range,
                        returns.max() + self.vf_clip_range,
                    )
                value_loss = ((value_pred - returns) ** 2).mean()
                entropy_loss = -entropy.mean()

                # When MAPPO is active, the actor's value_head isn't used.
                # Build loss accordingly: actor takes policy+BC+entropy,
                # team_critic takes value loss separately.
                if team_critic is not None:
                    loss = policy_loss + entropy_coef * entropy_loss
                else:
                    loss = policy_loss + self.value_coef * value_loss + entropy_coef * entropy_loss

                # BC auxiliary loss for commander: aim at enemy position in obs
                if bc_weight > 0 and obs.shape[1] >= 70:
                    with torch.no_grad():
                        enemy_xy = obs[:, 68:70]  # [B, 2] enemy 0 normalized position
                    features = ac.shared(obs)
                    mean_raw = ac.action_head(features)
                    action_mean = torch.tanh(mean_raw)
                    if self.residual_aim:
                        # Residual semantics: pull the correction (dims 1:4) toward 0 so the
                        # aim sits on the anchor; the reward then refines the last meters.
                        bc_loss = (action_mean[:, 1:4] ** 2).mean()
                    else:
                        bc_loss = ((action_mean[:, 1:3] - enemy_xy) ** 2).mean()
                    loss = loss + bc_weight * bc_loss
                    total_bc_loss += bc_loss.item()

                    # Supervised pre-training: optimize ONLY the BC loss this phase
                    # so PPO/entropy cannot perturb the pointing before it converges.
                    if bc_only:
                        loss = bc_weight * bc_loss

                optimizer.zero_grad()
                if team_critic_optimizer is not None:
                    team_critic_optimizer.zero_grad()
                loss.backward()
                if team_critic is not None and team_critic_optimizer is not None and not bc_only:
                    # Add value loss for team_critic and backprop separately
                    team_critic_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(team_critic.parameters(), self.max_grad_norm)
                    team_critic_optimizer.step()
                nn.utils.clip_grad_norm_(ac.parameters(), self.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        buffer.reset()
        if n_updates == 0:
            return {}
        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
        if total_bc_loss > 0:
            metrics["bc_loss"] = total_bc_loss / n_updates
        return metrics

    def train_episode(self) -> dict:
        """Run one training episode (pulse-level loop)."""
        env = self.env
        E, R, N = self.E, self.R, self.N
        dev = torch.device(self.device)
        max_pulses = self.cfg.get("league", {}).get("max_steps_per_episode", 600000)

        env.reset()
        self.cpi_buffer.reset()
        self._reset_tracks()
        self._enforce_radar_baseline()
        self._spectrum = None
        self._pulse_count = 0
        self._beam_hit_time.zero_()

        # Cache initial TX (zeros)
        self._cached_tx = torch.zeros(E, R, N, self.S, dtype=torch.complex64, device=dev)
        self._cached_cmd_action = torch.zeros(E, env.n_teams, 5, device=dev)
        self._cached_veh = torch.zeros(E, R, 3, device=dev)

        # Training metadata for last NN step
        self._last_radar_obs = None
        self._last_radar_action = None
        self._last_radar_logp = None
        self._last_radar_val = None
        self._last_cmd_obs = None
        self._last_cmd_action = None
        self._last_cmd_logp = None
        self._last_cmd_val = None
        self._last_cmd_priv_val = None
        self._last_cmd_priv_info = None
        self._last_team_state = None

        episode_stats = {
            "kills": 0, "total_pulses": 0, "radar_updates": 0, "cmd_updates": 0,
            "total_beam_reward": 0.0, "total_cmd_reward": 0.0,
            "min_aim_dist": 1e9, "beam_reward_samples": 0,
        }
        events = {"cpi_complete": False, "nn_control_step": False}

        t0 = time.perf_counter()

        for pulse in range(max_pulses):
            # 1. CPI accumulation check
            cpi_complete = events.get("cpi_complete", False)
            if isinstance(cpi_complete, torch.Tensor):
                cpi_complete = cpi_complete.any()
            if cpi_complete:
                self._process_cpi()

            # 2. NN control step
            nn_control = events.get("nn_control_step", False)
            if isinstance(nn_control, torch.Tensor):
                nn_control = nn_control.any()
            if nn_control:
                self._nn_step(events)

            # 3. Step env
            result = env.step(self._cached_tx, self._cached_cmd_action, self._cached_veh)

            # 4. Accumulate RX
            self.cpi_buffer.append(result["rx_iq"])

            # 5. Store transitions if NN step happened this pulse
            if nn_control and self._last_radar_obs is not None:
                rewards = self._get_rewards(result)
                done = result["dones"].float()

                # Diagnostics: track beam reward and aim distance
                drone = env.battlefield.drone
                for t in range(env.n_teams):
                    episode_stats["total_cmd_reward"] += rewards["commander_rewards"][:, t].sum().item()
                # Min aim distance across all teams/envs
                for t in range(env.n_teams):
                    enemy_idx = env.battlefield.team_radar_indices[1 - t]
                    enemy_pos = env.radar_pos[:, enemy_idx, :]
                    aim = drone.laser_aim[:, t, :].unsqueeze(1)
                    dist = (aim - enemy_pos).norm(dim=-1).min(dim=-1).values
                    episode_stats["min_aim_dist"] = min(episode_stats["min_aim_dist"], dist.min().item())
                episode_stats["beam_reward_samples"] += 1

                # Radar transitions: only team-0 (training) radars in league mode.
                radar_idxs = ([int(i) for i in env.battlefield.team_radar_indices[0]]
                              if self.league else range(R))
                for r in radar_idxs:
                    for e in range(E):
                        if self.radar_buf.ptr >= self.radar_buf.buffer_size:
                            break
                        self.radar_buf.add(
                            obs=self._last_radar_obs[e, r].cpu(),
                            action=self._last_radar_action.reshape(E, R, -1)[e, r].cpu(),
                            reward=rewards["radar_rewards"][e, r].item(),
                            done=done[e].item(),
                            value=self._last_radar_val.reshape(E, R)[e, r].item(),
                            log_prob=self._last_radar_logp.reshape(E, R)[e, r].item(),
                        )

                # Commander transitions: only team-0 (training) in league mode.
                team_idxs = [0] if self.league else range(env.n_teams)
                for t in team_idxs:
                    for e in range(E):
                        if self.cmd_buf.ptr >= self.cmd_buf.buffer_size:
                            break
                        priv_val = (
                            self._last_cmd_priv_val[e, t].item()
                            if self._last_cmd_priv_val is not None else None
                        )
                        priv_info = (
                            self._last_cmd_priv_info[e, t].cpu()
                            if self._last_cmd_priv_info is not None else None
                        )
                        team_st = (
                            self._last_team_state[e, t].cpu()
                            if self._last_team_state is not None else None
                        )
                        self.cmd_buf.add(
                            obs=self._last_cmd_obs[e, t].cpu(),
                            action=self._last_cmd_action[e, t].cpu(),
                            reward=rewards["commander_rewards"][e, t].item(),
                            done=done[e].item(),
                            value=self._last_cmd_val.reshape(E, env.n_teams)[e, t].item(),
                            log_prob=self._last_cmd_logp.reshape(E, env.n_teams)[e, t].item(),
                            privileged_value=priv_val,
                            privileged_info=priv_info,
                            team_state=team_st,
                        )

            # 6. Check kills/dones
            if result["kills"].any():
                episode_stats["kills"] += result["kills"].sum().item()

            self._pulse_count += 1
            events = result

            if result["dones"].all():
                break

        elapsed = time.perf_counter() - t0
        episode_stats["total_pulses"] = self._pulse_count
        episode_stats["time_s"] = elapsed
        episode_stats["pulses_per_s"] = self._pulse_count / max(elapsed, 1e-6)

        return episode_stats

    def eval_episode(self) -> dict:
        """Deterministic eval rollout. No noise, no buffer writes, no reward.

        Measures true policy quality by taking action_mean directly (no sampling).
        Reports kills + min_aim_dist so we can compare against the noisy train
        rollouts — eval_min_aim_dist < train_min_aim_dist is expected since the
        train metric is dominated by log_std noise.
        """
        env = self.env
        E, R, N = self.E, self.R, self.N
        dev = torch.device(self.device)
        max_pulses = self.cfg.get("league", {}).get("max_steps_per_episode", 600000)

        env.reset()
        self.cpi_buffer.reset()
        self._reset_tracks()
        self._enforce_radar_baseline()
        self._spectrum = None
        self._pulse_count = 0
        self._beam_hit_time.zero_()

        self._cached_tx = torch.zeros(E, R, N, self.S, dtype=torch.complex64, device=dev)
        self._cached_cmd_action = torch.zeros(E, env.n_teams, 5, device=dev)
        self._cached_veh = torch.zeros(E, R, 3, device=dev)

        stats = {"kills": 0, "total_pulses": 0, "min_aim_dist": 1e9}
        events = {"cpi_complete": False, "nn_control_step": False}
        jam_sum = 0.0; jam_n = 0   # track mean jam level to see if jamming emerges

        for pulse in range(max_pulses):
            cpi_complete = events.get("cpi_complete", False)
            if isinstance(cpi_complete, torch.Tensor):
                cpi_complete = cpi_complete.any()
            if cpi_complete:
                self._process_cpi()

            nn_control = events.get("nn_control_step", False)
            if isinstance(nn_control, torch.Tensor):
                nn_control = nn_control.any()
            if nn_control:
                # Deterministic NN step (no noise, no buffer writes)
                self._eval_nn_step(events)

            result = env.step(self._cached_tx, self._cached_cmd_action, self._cached_veh)
            self.cpi_buffer.append(result["rx_iq"])

            # Track aim distance on every NN step (cheap)
            if nn_control:
                drone = env.battlefield.drone
                for t in range(env.n_teams):
                    enemy_idx = env.battlefield.team_radar_indices[1 - t]
                    enemy_pos = env.radar_pos[:, enemy_idx, :]
                    # Measure the commander's intended aim (valid even when not firing),
                    # so the eval aim metric is honest without forcing fire.
                    aim = drone._commander_aim[:, t, :].unsqueeze(1)
                    dist = (aim - enemy_pos).norm(dim=-1).min(dim=-1).values
                    stats["min_aim_dist"] = min(stats["min_aim_dist"], dist.min().item())
                if self.jam_gain > 0.0:
                    jam_sum += self._jam_level.mean().item(); jam_n += 1

            if result["kills"].any():
                stats["kills"] += result["kills"].sum().item()

            self._pulse_count += 1
            events = result

            if result["dones"].all():
                break

        # True per-team outcome: winners[e] = 0 (red/team0), 1 (blue/team1), -1 (draw/timeout)
        w = env.battlefield.winners
        stats["red_wins"] = int((w == 0).sum().item())
        stats["blue_wins"] = int((w == 1).sum().item())
        stats["draws"] = int((w == -1).sum().item())
        stats["n_games"] = int(env.num_envs)
        stats["total_pulses"] = self._pulse_count
        stats["jam_mean"] = jam_sum / max(jam_n, 1)
        return stats

    def _eval_nn_step(self, events: dict):
        """Deterministic NN decision — no noise, no buffer writes.

        Forces commander fire=True during eval so laser_aim reflects the policy's
        true aim direction. Otherwise eval_min_aim_dist measures distance from
        origin when the policy hasn't learned to fire yet.
        """
        E, R, N = self.E, self.R, self.N
        dev = torch.device(self.device)

        radar_obs = self._build_radar_obs(events)
        cmd_obs = self._build_commander_obs(events)

        T = self.env.n_teams
        with torch.no_grad():
            radar_flat = radar_obs.reshape(E * R, -1)
            r_action, _, _, _ = self.radar_ac.get_action(radar_flat, deterministic=True)
            cmd_flat = cmd_obs.reshape(E * T, -1)
            c_action, _, _, _ = self.commander_ac.get_action(cmd_flat, deterministic=True)

        r_exec = r_action.reshape(E, R, -1)
        c_exec = c_action.reshape(E, T, 5)
        if self.league and self.commander_opp is not None:
            # eval = training team 0 (deterministic) vs the sampled frozen opponent (team 1)
            with torch.no_grad():
                r_o, _, _, _ = self.radar_opp.get_action(radar_flat, deterministic=True)
                c_o, _, _, _ = self.commander_opp.get_action(cmd_flat, deterministic=True)
            r_exec = r_exec.clone(); c_exec = c_exec.clone()
            r_o = r_o.reshape(E, R, -1); c_o = c_o.reshape(E, T, 5)
            for ri in self.env.battlefield.team_radar_indices[1]:
                r_exec[:, int(ri)] = r_o[:, int(ri)]
            c_exec[:, 1] = c_o[:, 1]

        self._cached_tx = self._assemble_tx(r_exec.reshape(E * R, -1))
        self._cached_cmd_action = self._to_env_cmd_action(c_exec, cmd_obs)
        self._cached_veh = r_exec[:, :, -3:]
        self._jam_level = ((c_exec[..., 4] + 1.0) * 0.5).clamp(0.0, 1.0)  # [E, T] jam per team

    def _to_env_cmd_action(self, raw, cmd_obs):
        """Map the raw commander action to the env action.

        With residual_aim, the aim dims become anchor(enemy obs) + residual × scale, so the
        policy outputs only a small correction around the known enemy position. The buffer
        keeps the RAW action (for PPO); the env receives the transformed absolute aim.
        raw: [E, T, 5], cmd_obs: [E, T, 76].
        """
        if not self.residual_aim:
            return raw
        env_a = raw.clone()
        half_x = self.env.map_size[0] / 2.0
        half_y = self.env.map_size[1] / 2.0
        anchor = cmd_obs[..., 68:70]  # normalized enemy radar-0 position
        env_a[..., 1] = anchor[..., 0] + raw[..., 1] * (self.residual_scale_m / half_x)
        env_a[..., 2] = anchor[..., 1] + raw[..., 2] * (self.residual_scale_m / half_y)
        env_a[..., 3] = raw[..., 3] * (self.residual_scale_m / 1000.0)
        return env_a

    def _nn_step(self, events: dict):
        """Run NN decision and cache actions."""
        E, R, N = self.E, self.R, self.N
        dev = torch.device(self.device)

        # Build observations
        radar_obs = self._build_radar_obs(events)  # [E, R, state_dim]
        cmd_obs = self._build_commander_obs(events)  # [E, n_teams, 76]

        # Build privileged info for commander (CTDE): [E, T, priv_dim]
        # Contains state that's hard to derive from local obs but useful for
        # predicting cumulative reward (beam_hit_time, distance to enemies).
        cmd_priv_info = self._build_commander_privileged_info()  # None or [E, T, priv_dim]

        with torch.no_grad():
            # Radar: flatten to [E*R, state_dim] for the actor-critic
            radar_flat = radar_obs.reshape(E * R, -1)
            r_action, r_logp, r_val, _ = self.radar_ac.get_action(radar_flat)
            self._last_radar_action = r_action  # [E*R, action_dim]
            self._last_radar_logp = r_logp.reshape(E, R)
            self._last_radar_val = r_val.reshape(E, R)
            self._last_radar_obs = radar_obs

            # Commander: flatten to [E*T, 76]
            T = self.env.n_teams
            cmd_flat = cmd_obs.reshape(E * T, -1)
            if cmd_priv_info is not None:
                priv_flat = cmd_priv_info.reshape(E * T, -1)
                c_action, c_logp, c_val, c_priv_val = self.commander_ac.get_action(
                    cmd_flat, privileged_info=priv_flat,
                )
                self._last_cmd_priv_val = c_priv_val.reshape(E, T)
                self._last_cmd_priv_info = cmd_priv_info
            else:
                c_action, c_logp, c_val, _ = self.commander_ac.get_action(cmd_flat)
                self._last_cmd_priv_val = None
                self._last_cmd_priv_info = None
            self._last_cmd_action = c_action.reshape(E, T, 5)
            self._last_cmd_logp = c_logp.reshape(E, T)
            self._last_cmd_val = c_val.reshape(E, T)
            self._last_cmd_obs = cmd_obs

            # MAPPO: build team_state and compute V_team (centralized critic)
            if self.use_mappo:
                team_state = self._build_team_state_mappo(cmd_obs)  # [E, T, 104]
                team_state_flat = team_state.reshape(E * T, 104)
                v_team = self.team_critic(team_state_flat).squeeze(-1)  # [E*T]
                self._last_cmd_val = v_team.reshape(E, T)  # override per-agent value
                self._last_team_state = team_state
            else:
                self._last_team_state = None

        # Cache actions for env step (buffer keeps raw _last_cmd_action for team 0).
        if self.league and self.commander_opp is not None:
            # League: team 1 (blue) acts with the frozen opponent; team 0 (red) is training.
            with torch.no_grad():
                r_opp, _, _, _ = self.radar_opp.get_action(radar_flat)
                c_opp, _, _, _ = self.commander_opp.get_action(cmd_flat)
            r_exec = r_action.reshape(E, R, -1).clone()
            c_exec = self._last_cmd_action.clone()
            r_opp = r_opp.reshape(E, R, -1)
            c_opp = c_opp.reshape(E, self.env.n_teams, 5)
            for ri in self.env.battlefield.team_radar_indices[1]:
                r_exec[:, int(ri)] = r_opp[:, int(ri)]
            c_exec[:, 1] = c_opp[:, 1]
            self._cached_tx = self._assemble_tx(r_exec.reshape(E * R, -1))
            self._cached_cmd_action = self._to_env_cmd_action(c_exec, cmd_obs)
            self._cached_veh = r_exec[:, :, -3:]
            exec_cmd = c_exec
        else:
            self._cached_tx = self._assemble_tx(r_action)
            self._cached_cmd_action = self._to_env_cmd_action(self._last_cmd_action, cmd_obs)
            self._cached_veh = r_action.reshape(E, R, -1)[:, :, -3:]
            exec_cmd = self._last_cmd_action
        # Jam level per team from the executed commander action[4] ∈ [-1,1] → [0,1].
        self._jam_level = ((exec_cmd[..., 4] + 1.0) * 0.5).clamp(0.0, 1.0)  # [E, T]

    def _build_commander_privileged_info(self):
        """Build privileged_info for commander CTDE critic. None if disabled.

        Returns: [E, T, priv_dim] or None
        Layout: [beam_hit_time_norm, dist_to_enemy0_norm, dist_to_enemy1_norm,
                 enemy0_alive, enemy1_alive]
        """
        priv_dim = getattr(self.commander_ac, "privileged_dim", 0)
        if priv_dim == 0:
            return None

        env = self.env
        E = env.num_envs
        T = env.n_teams
        dev = torch.device(self.device)

        # beam_hit_time [E, T] → normalize by t_max (illumination_time * prf)
        t_max = max(self.t_max, 1.0)
        bht = (self._beam_hit_time / t_max).clamp(0.0, 1.0)  # [E, T]

        # Distances from laser_aim to each enemy radar
        drone = env.battlefield.drone
        radar_alive = env.radar_alive if hasattr(env, "radar_alive") else None

        # Layout: [bht, dist0_norm, dist1_norm, alive0, alive1] = 5 dims
        half_diag = float((env.map_size[0] ** 2 + env.map_size[1] ** 2) ** 0.5) / 2.0
        priv = torch.zeros(E, T, 5, device=dev)
        for t in range(T):
            enemy_idx = env.battlefield.team_radar_indices[1 - t]
            enemy_pos = env.radar_pos[:, enemy_idx, :]  # [E, n_enemy, 3]
            aim = drone.laser_aim[:, t, :].unsqueeze(1)  # [E, 1, 3]
            dists = (aim - enemy_pos).norm(dim=-1)  # [E, n_enemy]
            priv[:, t, 0] = bht[:, t]
            n_enemy = min(dists.shape[1], 2)
            for i in range(n_enemy):
                priv[:, t, 1 + i] = dists[:, i] / half_diag
            # Alive flags (default 1.0 if not tracked)
            if radar_alive is not None:
                for i in range(n_enemy):
                    priv[:, t, 3 + i] = radar_alive[:, enemy_idx[i]].float()
            else:
                priv[:, t, 3:5] = 1.0

        return priv

    def _build_team_state_mappo(self, cmd_obs):
        """Build team_state [E, T, 104] for MAPPO centralized critic.

        Combines commander_obs (76) with task_fingerprint, alive, missile state,
        using the existing build_team_state helper. Missile fields are zeros
        (no missile weapon in this laser-only mode).
        """
        env = self.env
        E = env.num_envs
        T = env.n_teams
        dev = torch.device(self.device)

        # Per-team team_state via build_team_state (handles batching)
        # Reshape cmd_obs to [E*T, 76] for batched call
        cmd_flat = cmd_obs.reshape(E * T, -1)  # [E*T, 76]

        # Alive flags per radar [E, R]
        radar_alive = env.radar_alive if hasattr(env, "radar_alive") else None
        if radar_alive is None:
            alive = torch.ones(E, env.n_radars, dtype=torch.bool, device=dev)
        else:
            alive = radar_alive.bool()

        # Expand alive to [E*T, R] (each team sees same alive array)
        alive_expanded = alive.unsqueeze(1).expand(E, T, env.n_radars).reshape(E * T, env.n_radars)

        # task_fingerprint: zeros (no task extraction in laser mode)
        task_fp = torch.zeros(E * T, T, 4, device=dev)

        # avg_snr: zeros (would need radar spectrum processing)
        avg_snr = torch.zeros(E * T, env.n_radars, device=dev)

        # Missile: zeros (no missile in laser mode)
        missile_pos = torch.zeros(E * T, T, 3, device=dev)
        missile_in_flight = torch.zeros(E * T, T, dtype=torch.bool, device=dev)
        missile_target = torch.zeros(E * T, T, 3, device=dev)

        team_state_flat = build_team_state(
            commander_obs=cmd_flat,
            task_fingerprint=task_fp,
            avg_snr=avg_snr,
            alive=alive_expanded,
            missile_pos=missile_pos,
            missile_in_flight=missile_in_flight,
            missile_target=missile_target,
        )  # [E*T, 104]
        return team_state_flat.reshape(E, T, 104)

    def update(self, cmd_bc_weight: float = 0.0, cmd_bc_only: bool = False) -> dict:
        """Run PPO update on both radar and commander buffers.

        Args:
            cmd_bc_weight: if > 0, add BC auxiliary loss on commander only.
            cmd_bc_only: if True, commander update optimizes ONLY the BC loss
                         (supervised pre-training phase that locks the pointing
                         before PPO engages). Radar still trains normally.
        """
        radar_metrics = self._ppo_update(
            self.radar_ac, self.radar_optimizer, self.radar_buf,
            self.radar_clip, self.radar_entropy,
        )
        cmd_metrics = self._ppo_update(
            self.commander_ac, self.commander_optimizer, self.cmd_buf,
            self.commander_clip, self.commander_entropy,
            bc_weight=cmd_bc_weight,
            team_critic=self.team_critic,
            team_critic_optimizer=self.team_critic_optimizer,
            bc_only=cmd_bc_only,
        )
        return {"radar": radar_metrics, "commander": cmd_metrics}


def main():
    parser = argparse.ArgumentParser(description="Laser drone weapon training")
    parser.add_argument("--config", type=str, default="configs/laser_25x25_config.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override for quick test
    if args.rows and args.cols:
        cfg["env"]["rows"] = args.rows
        cfg["env"]["cols"] = args.rows  # square
        cfg["sub_array_size"] = max(1, args.rows // 5)

    league_cfg = cfg.get("league", {})
    n_episodes = args.episodes or league_cfg.get("episodes_per_training", 100)
    psro_iters = cfg.get("training", {}).get("psro_iterations", 30)

    print(f"Config: {args.config}")
    print(f"Rows×Cols: {cfg['env'].get('rows', 25)}×{cfg['env'].get('cols', 25)}")
    print(f"Episodes per PSRO iter: {n_episodes}")
    print(f"PSRO iterations: {psro_iters}")

    # Build env
    env = build_env(cfg)
    E, R, N = env.num_envs, env.n_radars, env.n_elem
    n_bins = env.n_bins
    n_pulses = env.n_pulses
    print(f"Env: E={E}, R={R}, N={N}, S={env.n_samples}, P={n_pulses}, bins={n_bins}")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Build actors
    radar_ac, commander_ac = build_actors(cfg, N, n_pulses, n_bins, args.device)
    print(f"Radar AC params: {sum(p.numel() for p in radar_ac.parameters()):,}")
    print(f"Commander AC params: {sum(p.numel() for p in commander_ac.parameters()):,}")
    print(f"VRAM after models: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Build trainer
    trainer = LaserTrainer(env, radar_ac, commander_ac, cfg)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: {psro_iters} PSRO iters × {n_episodes} episodes")
    print(f"Max pulses/episode: {league_cfg.get('max_steps_per_episode', 600000):,}")
    print(f"{'='*60}\n")

    # BC auxiliary loss weight (supervises commander to aim at enemy in obs).
    # Schedule: high early, decay over iters so PPO can take over later.
    cmd_bc_weight_init = cfg.get("training", {}).get("cmd_bc_weight_init", 1.0)
    cmd_bc_weight_final = cfg.get("training", {}).get("cmd_bc_weight_final", 0.0)
    cmd_bc_decay_iters = cfg.get("training", {}).get("cmd_bc_decay_iters", psro_iters)
    print(f"BC weight: {cmd_bc_weight_init} → {cmd_bc_weight_final} over {cmd_bc_decay_iters} iters")

    # kill_radius curriculum: start wide so illumination/kills are reachable, then ratchet
    # toward the env's true kill_radius, gated on recent aim accuracy (never shrink below
    # what the policy can currently hit). See LASER_ROOT_CAUSE_ANALYSIS.md §7 P0-3.
    kr_init = float(cfg.get("training", {}).get("kill_radius_init", 50.0))
    kr_final = float(cfg.get("env", {}).get("kill_radius_m", 0.2))
    env.battlefield.laser.kill_radius_m = kr_init
    print(f"kill_radius curriculum: {kr_init}m → {kr_final}m (success-gated)")

    # Supervised BC pre-training: lock the commander's pointing before PPO engages,
    # so PPO drift can't bounce the mean aim (eval was 70m↔1046m without this).
    bc_pretrain_iters = int(cfg.get("training", {}).get("bc_pretrain_iters", 0))
    if bc_pretrain_iters > 0:
        print(f"BC pre-training: first {bc_pretrain_iters} iters optimize commander BC only")

    # PSRO-lite league: seed the pool with the initial policy, then snapshot every N iters.
    league_snapshot_every = int(cfg.get("training", {}).get("league_snapshot_every", 3))
    cum_red = cum_blue = cum_draw = 0  # cumulative league outcomes over all eval games
    if trainer.league:
        trainer._snapshot_to_pool()
        print(f"League ON: red=training vs blue=opponent pool (snapshot every {league_snapshot_every} iters)")

    for psro_iter in range(psro_iters):
        # League: sample a (PFSP-weighted) opponent for this iteration's red-vs-blue games.
        if trainer.league:
            trainer._sample_opponent()
        iter_t0 = time.perf_counter()
        iter_kills = 0
        iter_pulses = 0
        iter_time = 0
        n_updates = 0
        iter_min_dist = 1e9
        iter_total_cmd_r = 0.0
        iter_reward_samples = 0

        # BC weight schedule: linear decay
        if cmd_bc_decay_iters > 0:
            bc_frac = max(0.0, 1.0 - psro_iter / cmd_bc_decay_iters)
        else:
            bc_frac = 0.0
        cur_bc_weight = cmd_bc_weight_final + (cmd_bc_weight_init - cmd_bc_weight_final) * bc_frac
        # During the pretrain phase, force full BC weight and BC-only commander updates.
        in_pretrain = psro_iter < bc_pretrain_iters
        if in_pretrain:
            cur_bc_weight = cmd_bc_weight_init

        for ep in range(n_episodes):
            stats = trainer.train_episode()
            iter_kills += stats["kills"]
            iter_pulses += stats["total_pulses"]
            iter_time += stats["time_s"]
            iter_min_dist = min(iter_min_dist, stats.get("min_aim_dist", 1e9))
            iter_total_cmd_r += stats.get("total_cmd_reward", 0.0)
            iter_reward_samples += stats.get("beam_reward_samples", 0)

            # PPO update when buffers fill
            radar_ptr = trainer.radar_buf.ptr
            cmd_ptr = trainer.cmd_buf.ptr
            buf_size = trainer.radar_buf.buffer_size

            if radar_ptr >= buf_size * 0.8 or cmd_ptr >= buf_size * 0.8:
                metrics = trainer.update(cmd_bc_weight=cur_bc_weight, cmd_bc_only=in_pretrain)
                n_updates += 1

                if n_updates == 1:  # log first update
                    rm = metrics.get("radar", {})
                    cm = metrics.get("commander", {})
                    radar_loss = rm.get("policy_loss", 0)
                    cmd_loss = cm.get("policy_loss", 0)
                    bc_loss = cm.get("bc_loss", 0)
                    print(f"  [Update] radar_loss={radar_loss:.4f} cmd_loss={cmd_loss:.4f} bc_loss={bc_loss:.4f}")

        # Force update at end of iteration
        if trainer.radar_buf.ptr > 0 or trainer.cmd_buf.ptr > 0:
            metrics = trainer.update(cmd_bc_weight=cur_bc_weight, cmd_bc_only=in_pretrain)
            n_updates += 1

        # (kill_radius curriculum moved below — it is gated on the EVAL kill rate, not the
        #  optimistic noisy train min, so kr only tightens when the deterministic policy is
        #  actually killing at the current tolerance.)

        # Anneal commander AIM exploration: decrease log_std each iteration.
        # Defaults: -1.0 → -6.0 over iters with 0.20/iter decay (configurable).
        # At log_std=-6, std≈2.5e-3 on [-1,1] → ~25m physical noise at 3km (half_map=10km).
        # Only affects training-time sampling of the continuous aim dims (eval is
        # deterministic; the fire bit is Bernoulli with its own entropy).
        log_std_init = cfg.get("training", {}).get("log_std_init", -1.0)
        log_std_floor = cfg.get("training", {}).get("log_std_floor", -6.0)
        log_std_decay = cfg.get("training", {}).get("log_std_decay", 0.20)
        target_log_std = max(log_std_floor, log_std_init - psro_iter * log_std_decay)
        with torch.no_grad():
            commander_ac.log_std.data.fill_(target_log_std)
            # Keep the JAM dim (action[4]) exploring throughout. Jamming only pays off once
            # the kill_radius curriculum tightens to ≈ the jammed localisation error (<0.5m,
            # late iters); if jam shared the aim anneal it would freeze at its 0.50 baseline
            # before it ever mattered → never discovered. A sustained jam_log_std lets the
            # policy keep sampling jam=0 vs jam=1 so the gradient can find it when it counts.
            if trainer.jam_gain > 0.0:
                jam_log_std = cfg.get("training", {}).get("jam_log_std", -1.0)
                commander_ac.log_std.data[4] = jam_log_std

        iter_elapsed = time.perf_counter() - iter_t0
        avg_pulses = iter_pulses / max(n_episodes, 1)
        pulse_rate = iter_pulses / max(iter_elapsed, 1e-6)
        kill_rate = iter_kills / max(n_episodes * env.n_teams, 1)
        avg_cmd_r = iter_total_cmd_r / max(iter_reward_samples * env.n_teams, 1)

        cur_log_std = commander_ac.log_std.data[1:4].mean().item()  # aim dims only (jam held high)

        print(
            f"[PSRO {psro_iter+1}/{psro_iters}] "
            f"kills={iter_kills} kill_rate={kill_rate:.3f} "
            f"min_aim_dist={iter_min_dist:.0f}m "
            f"kr={env.battlefield.laser.kill_radius_m:.2f}m "
            f"avg_cmd_r={avg_cmd_r:.4f} "
            f"log_std={cur_log_std:.2f} "
            f"bc_w={cur_bc_weight:.2f} "
            f"rate={pulse_rate:.0f}p/s "
            f"time={iter_elapsed:.0f}s "
            f"upd={n_updates}"
        )

        # Deterministic eval rollout EVERY iter — measures true policy quality (no log_std
        # noise) and drives the curriculum.
        eval_stats = trainer.eval_episode()
        eval_kill_rate = eval_stats["kills"] / max(env.n_teams * env.num_envs, 1)
        eval_min = eval_stats["min_aim_dist"]
        eval_min_str = f"{eval_min:.0f}m" if eval_min < 1e8 else "inf"

        # kill_radius curriculum, gated on the EVAL kill rate (success-gated, GOID-style):
        # only tighten when the deterministic policy actually kills at the current tolerance;
        # relax if it can't (anti-collapse). Frozen during BC pretrain.
        if not in_pretrain:
            kr = env.battlefield.laser.kill_radius_m
            if eval_kill_rate >= 0.5:
                kr = max(kr_final, kr * 0.7)      # killing reliably → tighten toward 0.2m
            elif eval_kill_rate <= 1e-6:
                kr = min(kr_init, kr * 1.2)       # can't kill at this kr → give room back
            env.battlefield.laser.kill_radius_m = kr

        if trainer.league:
            cum_red += eval_stats.get("red_wins", 0)
            cum_blue += eval_stats.get("blue_wins", 0)
            cum_draw += eval_stats.get("draws", 0)
            tot = max(cum_red + cum_blue + cum_draw, 1)
            league_str = (f" pool={len(trainer.pool)} opp={getattr(trainer,'_opp_idx',-1)}"
                          f" | this:R{eval_stats.get('red_wins',0)}/B{eval_stats.get('blue_wins',0)}/D{eval_stats.get('draws',0)}"
                          f" | cum red={cum_red/tot:.2f} blue={cum_blue/tot:.2f} draw={cum_draw/tot:.2f} (n={tot})"
                          f" | jam={eval_stats.get('jam_mean',0.0):.2f}")
        else:
            league_str = ""
        print(
            f"[Eval @ iter {psro_iter+1}] "
            f"eval_kills={eval_stats['kills']} "
            f"eval_min_aim_dist={eval_min_str} "
            f"eval_kill_rate={eval_kill_rate:.3f} "
            f"kr_next={env.battlefield.laser.kill_radius_m:.2f}m{league_str}"
        )

        # League: freeze the improved policy into the opponent pool every N iters.
        if trainer.league and (psro_iter + 1) % league_snapshot_every == 0:
            trainer._snapshot_to_pool()

        # Save checkpoint
        ckpt_dir = league_cfg.get("checkpoint_dir", "checkpoints/laser_train")
        import os
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            "psro_iter": psro_iter,
            "radar_ac": radar_ac.state_dict(),
            "commander_ac": commander_ac.state_dict(),
            "radar_optimizer": trainer.radar_optimizer.state_dict(),
            "commander_optimizer": trainer.commander_optimizer.state_dict(),
        }, f"{ckpt_dir}/iter_{psro_iter:03d}.pt")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
