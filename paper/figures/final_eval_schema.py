"""Canonical schema and validation for FluxPhased final-evaluation artifacts.

A final-evaluation file is evidence, not just a bag of numbers. Schema v2
records the resolved runtime configuration alongside the four evaluation views
so a wrong geometry, stale checkpoint, or skipped marker cannot be silently
consumed by the results table.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
ARTIFACT_KIND = "final_eval"
ACTION_SEEDS = (4242, 777, 31337)
DEFAULT_RADAR_AZ = (20.0, -20.0)
DEFAULT_S6_RADAR_AZ = (20.0, 20.0)
DEFAULT_JAMMER_AZ_N2 = (60.0, -60.0)


class FinalEvalSchemaError(ValueError):
    """Raised when a result artifact is missing or violates its contract."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_revision(repo: str | Path | None = None) -> str:
    """Return the current git revision, or ``unknown`` outside a checkout."""
    try:
        cwd = str(repo or Path(__file__).resolve().parents[2])
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _float_list(values) -> list[float]:
    return [float(x) for x in values]


def build_metadata(*, train_seed: int, algorithm: str, checkpoint_iteration: int,
                   n_jammers: int, n_radars: int = 2,
                   jammer_az_deg=None, radar_az_deg=None,
                   baseline_snr_db: float = 12.0, P_jam_W: float = 0.1,
                   active_budget_steps: int = 63, horizon: int = 64,
                   validation_manifest: str | Path | None = None,
                   action_seeds=ACTION_SEEDS, n_action_reps: int = 1,
                   code_rev: str | None = None, env_profile: str = "array_face_s7_v1",
                   device: str | None = None) -> dict[str, Any]:
    """Build metadata from resolved runtime values, never raw CLI strings."""
    if jammer_az_deg is None:
        if env_profile == "array_face_s6_v1":
            jammer_az_deg = (0.0,)
        else:
            jammer_az_deg = DEFAULT_JAMMER_AZ_N2 if n_jammers == 2 else tuple(
                -60.0 + 120.0 * i / max(n_jammers - 1, 1) for i in range(n_jammers)
            )
    if radar_az_deg is None:
        radar_az_deg = DEFAULT_S6_RADAR_AZ if env_profile == "array_face_s6_v1" else DEFAULT_RADAR_AZ
    manifest = Path(validation_manifest) if validation_manifest else None
    metadata = {
        "train_seed": int(train_seed),
        "algorithm": str(algorithm),
        "checkpoint_iteration": int(checkpoint_iteration),
        "n_jammers": int(n_jammers),
        "n_radars": int(n_radars),
        "jammer_az_deg": _float_list(jammer_az_deg),
        "radar_az_deg": _float_list(radar_az_deg),
        "baseline_snr_db": float(baseline_snr_db),
        "P_jam_W": float(P_jam_W),
        "active_budget_steps": int(active_budget_steps),
        "horizon": int(horizon),
        "validation_manifest": str(manifest.name if manifest else "unknown"),
        "validation_manifest_sha256": sha256_file(manifest) if manifest and manifest.exists() else "unknown",
        "validation_seed_count": 0,
        "action_seeds": [int(x) for x in action_seeds],
        "n_action_reps": int(n_action_reps),
        "code_revision": code_rev or code_revision(),
    }
    if manifest and manifest.exists():
        try:
            payload = json.loads(manifest.read_text())
            metadata["validation_seed_count"] = len(payload.get("entries", payload))
        except Exception:
            metadata["validation_seed_count"] = 0
    if device is not None:
        metadata["device"] = str(device)
    return metadata


def wrap_final_eval(results: dict, metadata: dict) -> dict:
    """Wrap legacy view results in schema v2."""
    out = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "metadata": dict(metadata),
    }
    out.update(results)
    return out


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a) == list(b)
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= 1e-6
    return a == b


def validate_final_eval(data_or_path: dict | str | Path, *, expected: dict | None = None,
                        require_terminal: bool = False) -> dict:
    """Validate and return a v2 artifact.

    ``expected`` contains metadata keys that must match exactly. It is intended
    for canonical allowlists (e.g. n=3/+60,0,-60), not for arbitrary discovery.
    """
    path = Path(data_or_path) if isinstance(data_or_path, (str, Path)) else None
    try:
        data = json.loads(path.read_text()) if path else data_or_path
    except Exception as exc:
        raise FinalEvalSchemaError(f"cannot read final_eval: {path}: {exc}") from exc
    if data.get("skipped") is True:
        raise FinalEvalSchemaError("skipped artifact is not canonical evidence")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise FinalEvalSchemaError(
            f"legacy/unknown schema {data.get('schema_version')!r}; explicit upgrade required")
    if data.get("artifact_kind") != ARTIFACT_KIND:
        raise FinalEvalSchemaError(f"artifact_kind must be {ARTIFACT_KIND!r}")
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        raise FinalEvalSchemaError("metadata object is required")
    required = (
        "train_seed", "algorithm", "checkpoint_iteration", "n_jammers", "n_radars",
        "jammer_az_deg", "radar_az_deg", "baseline_snr_db", "P_jam_W",
        "active_budget_steps", "horizon", "validation_manifest",
        "validation_manifest_sha256", "validation_seed_count", "action_seeds", "n_action_reps",
        "code_revision",
    )
    missing = [k for k in required if k not in meta]
    if missing:
        raise FinalEvalSchemaError(f"missing metadata fields: {missing}")
    for key in ("aseed_4242", "aseed_777", "aseed_31337", "sweep_vs_idle_floor"):
        if key not in data:
            raise FinalEvalSchemaError(f"missing evaluation view: {key}")
    if list(meta["action_seeds"]) != list(ACTION_SEEDS):
        raise FinalEvalSchemaError(f"action seeds must be {list(ACTION_SEEDS)}")
    if int(meta["n_action_reps"]) != 1:
        raise FinalEvalSchemaError("canonical final evaluation requires n_action_reps=1")
    if int(meta["validation_seed_count"]) != 64:
        raise FinalEvalSchemaError("canonical final evaluation requires 64 validation scenarios")
    if int(meta["n_jammers"]) < 1 or len(meta["jammer_az_deg"]) != int(meta["n_jammers"]):
        raise FinalEvalSchemaError("n_jammers and jammer_az_deg length disagree")
    if int(meta["n_radars"]) != 2 or len(meta["radar_az_deg"]) != 2:
        raise FinalEvalSchemaError("canonical protocol requires two radars")
    if require_terminal and int(meta["checkpoint_iteration"]) < 1999:
        raise FinalEvalSchemaError("non-terminal checkpoint is not canonical")
    if expected:
        for key, value in expected.items():
            if key not in meta or not _same(meta[key], value):
                raise FinalEvalSchemaError(
                    f"metadata mismatch for {key}: expected {value!r}, got {meta.get(key)!r}")
    return data


def load_final_eval(path: str | Path, *, expected: dict | None = None,
                    require_terminal: bool = False) -> dict:
    return validate_final_eval(path, expected=expected, require_terminal=require_terminal)


def legacy_upgrade(data: dict, metadata: dict) -> dict:
    """Explicitly wrap a legacy result; caller must record the upgrade source."""
    if data.get("schema_version") is not None:
        raise FinalEvalSchemaError("input is not a legacy artifact")
    return wrap_final_eval(data, metadata)
