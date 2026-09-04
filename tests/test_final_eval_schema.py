"""Schema v2 tests for final-evaluation evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from paper.figures.final_eval_schema import (
    ACTION_SEEDS,
    FinalEvalSchemaError,
    build_metadata,
    validate_final_eval,
    wrap_final_eval,
)


def _legacy_views():
    row = {
        "h2h_drop": 0.1,
        "h2h_success": 0.9,
        "jam_vs_sweep_drop": 0.2,
        "rad_vs_idle_success": 0.98,
        "j1_only_drop": 0.1,
        "elapsed_s": 1.0,
    }
    return {
        "aseed_4242": copy.deepcopy(row),
        "aseed_777": copy.deepcopy(row),
        "aseed_31337": copy.deepcopy(row),
        "sweep_vs_idle_floor": {"drop": 0.05, "elapsed_s": 1.0},
    }


def _valid():
    meta = build_metadata(
        train_seed=20261011,
        algorithm="mappo",
        checkpoint_iteration=1999,
        n_jammers=3,
        jammer_az_deg=(60.0, 0.0, -60.0),
        radar_az_deg=(20.0, -20.0),
        validation_manifest=None,
        code_rev="test",
    )
    meta["validation_seed_count"] = 64
    return wrap_final_eval(_legacy_views(), meta)


def test_valid_schema_and_expected_metadata():
    data = _valid()
    got = validate_final_eval(
        data,
        expected={
            "n_jammers": 3,
            "jammer_az_deg": [60.0, 0.0, -60.0],
            "radar_az_deg": [20.0, -20.0],
        },
        require_terminal=True,
    )
    assert got["schema_version"] == 2
    assert got["metadata"]["action_seeds"] == list(ACTION_SEEDS)


@pytest.mark.parametrize("mutator", [
    lambda d: d.pop("metadata"),
    lambda d: d.update({"schema_version": 1}),
    lambda d: d.update({"skipped": True}),
])
def test_invalid_schema_rejected(mutator):
    data = _valid()
    mutator(data)
    with pytest.raises(FinalEvalSchemaError):
        validate_final_eval(data)


def test_geometry_mismatch_rejected():
    with pytest.raises(FinalEvalSchemaError, match="metadata mismatch"):
        validate_final_eval(
            _valid(),
            expected={"jammer_az_deg": [60.0, 60.0]},
        )


def test_n_geometry_length_rejected():
    data = _valid()
    data["metadata"]["jammer_az_deg"] = [60.0, 0.0]
    with pytest.raises(FinalEvalSchemaError, match="length disagree"):
        validate_final_eval(data)


def test_nonterminal_rejected_when_required():
    data = _valid()
    data["metadata"]["checkpoint_iteration"] = 1949
    with pytest.raises(FinalEvalSchemaError, match="non-terminal"):
        validate_final_eval(data, require_terminal=True)


def test_schema_roundtrip_file(tmp_path: Path):
    path = tmp_path / "final_eval.json"
    path.write_text(json.dumps(_valid()))
    assert validate_final_eval(path)["artifact_kind"] == "final_eval"
