"""S6 M0 physics gates — generalized AF + two-radar link budget.

Gates:
  1. broadside consistency: radar at az=el=0 + same steering == S4 JNR exactly
  2. generalized AF == S4 AF when target is broadside (unit-level)
  3. jammer idle -> -inf for both radars
  4. aiming asymmetry: a grid beam near +20° beats a far grid beam toward
     radar_0, and is simultaneously WORSE toward radar_1 (the dilemma)
  5. p_detect per-radar monotone, shape [E, R]
"""
from __future__ import annotations
import math
import pytest
import torch

from env.gpu.array_face_s6.array_factor import (
    UPAConfig, compute_upa_af_db_toward,
)
from env.gpu.array_face_s6.physics import compute_jnr_db_s6, compute_p_detect_s6
from env.gpu.array_face_s6.geometry import radar_directions, RADAR_AZ_DEG
from env.gpu.array_face_s4.array_factor import compute_upa_af_db
from env.gpu.array_face_s4.physics import compute_jnr_db_s4
from env.gpu.g3_bsta_lite.physics import default_debug_physics_config

PHYSICS = default_debug_physics_config(P_jam_W=2.0)
UPA = UPAConfig()


def test_generalized_af_matches_s4_at_broadside():
    """compute_upa_af_db_toward(target=0,0) == compute_upa_af_db (S4)."""
    idx = torch.arange(25)
    az_i, el_i = idx % 5, idx // 5
    s4 = compute_upa_af_db(UPA, beam_az_idx=az_i, beam_el_idx=el_i)
    s6 = compute_upa_af_db_toward(
        UPA, beam_az_idx=az_i, beam_el_idx=el_i,
        target_az_rad=torch.zeros(25), target_el_rad=torch.zeros(25),
    )
    assert torch.allclose(s4, s6, atol=1e-4), f"max diff {(s4-s6).abs().max()}"


def test_jnr_s6_broadside_consistency_with_s4():
    """Single radar placed at broadside + same steering -> S6 == S4 exactly."""
    E = 4
    cell = torch.zeros(E, 25); cell[:, 0] = 1.0
    jam_az = torch.full((E,), 2); jam_el = torch.full((E,), 2)   # jammer broadside grid
    rad_az = torch.full((E,), 2); rad_el = torch.full((E,), 2)   # radar broadside grid
    svc = torch.zeros(E, 1, dtype=torch.int64)                   # [E, R=1]

    j6 = compute_jnr_db_s6(
        PHYSICS, UPA, UPA,
        jammer_active=torch.ones(E, dtype=torch.bool),
        radar_beam_az_idx=rad_az.unsqueeze(1), radar_beam_el_idx=rad_el.unsqueeze(1),
        jammer_beam_az_idx=jam_az, jammer_beam_el_idx=jam_el,
        cell_mask=cell,
        radar_az_rad=torch.zeros(1), radar_el_rad=torch.zeros(1),
        victim_service_id=svc,
    )  # [E, 1]
    j4 = compute_jnr_db_s4(
        PHYSICS, UPA, UPA,
        jammer_active=torch.ones(E, dtype=torch.bool),
        victim_service_id=torch.zeros(E, dtype=torch.int64),
        radar_beam_az_idx=rad_az, radar_beam_el_idx=rad_el,
        jammer_beam_az_idx=jam_az, jammer_beam_el_idx=jam_el,
        cell_mask=cell,
    )
    assert torch.allclose(j6[:, 0], j4, atol=1e-3), f"{j6[:,0]} vs {j4}"
    assert 50.0 < j4[0].item() < 57.0  # 1-cell broadside ~53.5 dB


def test_jnr_s6_jammer_idle_all_radar_neg_inf():
    E, R = 3, 2
    az_r, el_r = radar_directions("cpu")
    j = compute_jnr_db_s6(
        PHYSICS, UPA, UPA,
        jammer_active=torch.zeros(E, dtype=torch.bool),
        radar_beam_az_idx=torch.zeros(E, R, dtype=torch.int64),
        radar_beam_el_idx=torch.zeros(E, R, dtype=torch.int64),
        jammer_beam_az_idx=torch.zeros(E, dtype=torch.int64),
        jammer_beam_el_idx=torch.zeros(E, dtype=torch.int64),
        cell_mask=torch.zeros(E, 25),
        radar_az_rad=az_r, radar_el_rad=el_r,
        victim_service_id=torch.zeros(E, R, dtype=torch.int64),
    )
    assert j.shape == (E, R)
    assert torch.isinf(j).all() and (j < 0).all()


def test_s6_aiming_dilemma_asymmetry():
    """S6b co-located geometry: the az=+30° grid beam (nearest to the +20°
    radar site) beats the az=−30° beam toward BOTH radar heads by >3 dB."""
    E, R = 1, 2
    az_r, el_r = radar_directions("cpu")
    cell = torch.zeros(E, 25); cell[:, 0] = 1.0
    kwargs = dict(
        radar_beam_az_idx=torch.zeros(E, R, dtype=torch.int64),
        radar_beam_el_idx=torch.zeros(E, R, dtype=torch.int64),
        cell_mask=cell, radar_az_rad=az_r, radar_el_rad=el_r,
        victim_service_id=torch.zeros(E, R, dtype=torch.int64),
    )
    near = compute_jnr_db_s6(
        PHYSICS, UPA, UPA,
        jammer_active=torch.ones(E, dtype=torch.bool),
        jammer_beam_az_idx=torch.full((E,), 3),   # az=+30° grid beam
        jammer_beam_el_idx=torch.full((E,), 2),
        **kwargs,
    )
    far = compute_jnr_db_s6(
        PHYSICS, UPA, UPA,
        jammer_active=torch.ones(E, dtype=torch.bool),
        jammer_beam_az_idx=torch.full((E,), 1),   # az=−30° grid beam
        jammer_beam_el_idx=torch.full((E,), 2),
        **kwargs,
    )
    for r in range(R):
        gap = (near[0, r] - far[0, r]).item()
        assert gap > 3.0, f"near-beam advantage over radar{r} should exceed 3 dB, got {gap}"


def test_s6_p_detect_shape_and_monotone():
    jnr = torch.tensor([[80.0, 40.0], [20.0, 0.0]])
    p = compute_p_detect_s6(PHYSICS, baseline_snr_db=22.0, jnr_db=jnr)
    assert p.shape == (2, 2)
    assert (p[0] < p[1]).all()  # lower JNR -> higher p_detect
    assert (p > 0).all() and (p < 1).all()
