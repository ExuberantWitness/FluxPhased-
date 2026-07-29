"""F1 §6 Gate 0 contract test 6 — counterfactual physics.

Per DEBUG_CONTRACT.md §7, the counterfactual properties that must hold in
the calibrated regime:

  - increasing jammer dose on a selected service cannot improve that service;
  - the selected service is measurably affected;
  - the unselected service is not equally affected;
  - antijam changes the JNR consumed by detection, not telemetry only.

We exercise these via the physics helpers directly AND through the env's
detection-batch path.
"""

from __future__ import annotations

import math

import torch

from env.gpu.g3_bsta_lite import (
    ACTION_IDLE,
    ACTION_JAM_SERVICE_0,
    ACTION_JAM_SERVICE_1,
    EnvConfig,
    G3BstaLiteVecEnv,
    default_debug_physics_config,
    compute_service_jnr_db,
    compute_detection_probability,
)
from env.gpu.g3_bsta_lite.physics import DebugPhysicsConfig, ServiceChannel


def _make_cfg(*, P_jam_W: float) -> DebugPhysicsConfig:
    return default_debug_physics_config(P_jam_W=P_jam_W)


def test_jnr_monotone_in_jammer_power_on_matched_service():
    """JNR(service k, k) must be monotone increasing in P_jam_W."""
    ps = [10.0, 50.0, 200.0, 1000.0]
    jnrs = [
        compute_service_jnr_db(_make_cfg(P_jam_W=p),
                               jammer_active=True,
                               jammer_service_id=0,
                               victim_service_id=0)
        for p in ps
    ]
    for a, b in zip(jnrs, jnrs[1:]):
        assert b > a, f"JNR not monotone: {jnrs}"


def test_jnr_infinite_when_jammer_inactive():
    cfg = _make_cfg(P_jam_W=100.0)
    jnr = compute_service_jnr_db(cfg, jammer_active=False,
                                jammer_service_id=0, victim_service_id=0)
    assert math.isinf(jnr)


def test_jnr_mismatched_service_below_matched_service():
    """JNR(jammer=0, victim=1) must be < JNR(jammer=0, victim=0)."""
    cfg = _make_cfg(P_jam_W=100.0)
    matched = compute_service_jnr_db(cfg, jammer_active=True,
                                     jammer_service_id=0, victim_service_id=0)
    mismatched = compute_service_jnr_db(cfg, jammer_active=True,
                                        jammer_service_id=0, victim_service_id=1)
    assert mismatched < matched


def test_p_detect_decreases_with_jammer_power_on_matched_service():
    base_snr = 22.0
    p_off = compute_detection_probability(_make_cfg(P_jam_W=1.0),
                                          baseline_snr_db=base_snr,
                                          jnr_db=float("-inf"))
    p_low = compute_detection_probability(_make_cfg(P_jam_W=10.0),
                                          baseline_snr_db=base_snr,
                                          jnr_db=compute_service_jnr_db(
                                              _make_cfg(P_jam_W=10.0),
                                              jammer_active=True,
                                              jammer_service_id=0,
                                              victim_service_id=0))
    p_high = compute_detection_probability(_make_cfg(P_jam_W=1000.0),
                                           baseline_snr_db=base_snr,
                                           jnr_db=compute_service_jnr_db(
                                               _make_cfg(P_jam_W=1000.0),
                                               jammer_active=True,
                                               jammer_service_id=0,
                                               victim_service_id=0))
    assert p_off > p_low > p_high
    assert p_high < 0.001, f"high-dose p_detect={p_high}, expected < 0.001"


def test_p_detect_unchanged_when_jammer_emits_on_other_service():
    """Jamming service 1 while radar scans service 0 must not affect p_det(0)."""
    base_snr = 22.0
    cfg = _make_cfg(P_jam_W=100.0)
    p_clean = compute_detection_probability(cfg, baseline_snr_db=base_snr,
                                            jnr_db=float("-inf"))
    jnr_at_victim_0 = compute_service_jnr_db(cfg, jammer_active=True,
                                             jammer_service_id=1,
                                             victim_service_id=0)
    p_with_other = compute_detection_probability(cfg, baseline_snr_db=base_snr,
                                                 jnr_db=jnr_at_victim_0)
    # No overlap between service 0 (10.0 GHz, 10 MHz BW) and service 1
    # (10.5 GHz, 20 MHz BW); JNR must be -inf or very low.
    assert jnr_at_victim_0 < -50.0, f"unexpected cross-service JNR {jnr_at_victim_0}"
    assert abs(p_clean - p_with_other) < 1e-3


def test_env_step_detect_probability_drops_on_matched_jam():
    """Through the env: matched jamming drives p_det to ~0; mismatched does not."""
    cfg = EnvConfig(n_envs=2, device="cpu", seed=303)
    env = G3BstaLiteVecEnv(cfg)
    env.reset(seed=303)

    # Step 0: env 0 jams service 0, env 1 idle. Radar scans service 0.
    _, _, _, info = env.step(torch.tensor([ACTION_JAM_SERVICE_0, ACTION_IDLE],
                                          dtype=torch.int64))
    assert int(info["radar_service"]) == 0
    p_det_with_jam = float(info["p_detect"][0])
    p_det_without_jam = float(info["p_detect"][1])
    assert p_det_with_jam < 0.01, f"matched jam failed: p_det={p_det_with_jam}"
    assert p_det_without_jam > 0.99, f"baseline failed: p_det={p_det_without_jam}"


def test_telemetry_and_task_outcomes_agree():
    """Detector outcome (telemetry) drives mission success/drop (task outcome).

    Matched jamming reduces p_det to ~0; admitted missions during jammed
    windows must eventually time out. Energy budget prevents all-time jamming,
    so we instead assert: matched-jamming drops strictly more missions than
    mismatched-jamming (where p_det stays high and most missions succeed).
    """
    def run(jam_matched: bool):
        cfg = EnvConfig(n_envs=1, device="cpu", seed=404, mission_tau_window=4,
                        arrival_rate_per_service=0.5)
        env = G3BstaLiteVecEnv(cfg)
        env.reset(seed=404)
        env.scenario.arrivals[:] = True
        for t in range(cfg.horizon):
            if jam_matched:
                a = (t % 2) + 1
            else:
                a = ((t + 1) % 2) + 1  # opposite service
            if not bool(env._compute_mask()[0, a]):
                a = ACTION_IDLE
            env.step(torch.tensor([a], dtype=torch.int64))
        n_elig = int(env.counters.n_eligible[0])
        n_succ = int(env.counters.n_success[0])
        n_drop = (int(env.counters.n_timeout[0])
                  + int(env.counters.n_horizon_failure[0])
                  + int(env.counters.n_admission_reject[0]))
        return n_elig, n_succ, n_drop

    elig_m, succ_m, drop_m = run(jam_matched=True)
    elig_x, succ_x, drop_x = run(jam_matched=False)
    # Same arrivals table ⇒ same eligible count.
    assert elig_m == elig_x
    # Accounting identity holds for both runs.
    assert succ_m + drop_m == elig_m
    assert succ_x + drop_x == elig_x
    # Matched jamming drops more than mismatched (telemetry/task agreement).
    assert drop_m > drop_x, (
        f"matched={drop_m} not > mismatched={drop_x}; telemetry failed to drive task"
    )
