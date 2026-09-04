"""Focused trainer-contract tests that do not run PPO or a full episode."""
from __future__ import annotations

import torch

from env.gpu.array_face_s7 import OBS_DIM_RADAR
from experiments.array_face_s7.learning_repair.trainer_s7 import S7SelfPlayTrainer


def make_obs(pending_by_lane):
    """Make [E=2,R=2,D=60] observations with distinct pending maps."""
    obs = torch.zeros(2, 2, OBS_DIM_RADAR)
    for e, (svc, az) in enumerate(pending_by_lane):
        obs[e, :, 1 + svc * 5 + az] = 1.0
    return obs


def test_greedy_radar_actions_are_lane_isolated():
    obs = make_obs([(0, 0), (1, 4)])
    beam, svc = S7SelfPlayTrainer._greedy_radar_actions(obs)
    assert beam.shape == (2, 2) and svc.shape == (2, 2)
    assert beam.tolist() == [[10, 10], [14, 14]]
    assert svc.tolist() == [[0, 0], [1, 1]]


def test_greedy_radar_actions_preserve_equal_lanes_and_heads():
    obs = make_obs([(1, 2), (1, 2)])
    beam, svc = S7SelfPlayTrainer._greedy_radar_actions(obs)
    assert beam.tolist() == [[12, 12], [12, 12]]
    assert svc.tolist() == [[1, 1], [1, 1]]
    assert (beam >= 0).all() and (beam < 25).all()
    assert (svc >= 0).all() and (svc < 2).all()
