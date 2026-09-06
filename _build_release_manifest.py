"""Build a machine-readable release manifest for canonical evidence.

The manifest is generated only from schema-v2 final_eval artifacts. It records
relative path, SHA-256, resolved metadata, and status. Missing/legacy/invalid
files are never silently promoted to canonical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paper" / "figures"))
from final_eval_schema import validate_final_eval, FinalEvalSchemaError, sha256_file  # noqa: E402

BASE = ROOT / "experiments/array_face_s7/learning_repair"
S6BASE = ROOT / "experiments/array_face_s6/learning_repair"

CASES = []
for sd in (20260730, 20260731, 20260732):
    CASES.append((S6BASE / f"s6_selfplay_output_seed{sd}/final_eval_v2.json", "s6"))
CASES.append((BASE / "s7_strict_n2_output_seed20260801/final_eval_v2.json", "n2"))
for sd in (20260802, 20260803):
    CASES.append((BASE / f"s7_strict_n2_output_seed{sd}/final_eval_v2.json", "n2"))
for sd in (20261011, 20261012, 20261013):
    CASES.append((BASE / f"s9_n3_output_seed{sd}/final_eval_v2.json", "n3"))
CASES.append((BASE / "s9_n4_output_seed20261021/final_eval_v2.json", "n4"))
for sd in (20261022, 20261023):
    CASES.append((BASE / f"s9_strict_n4_output_seed{sd}/final_eval_v2.json", "n4"))

entries = []
for path, group in CASES:
    row = {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
           "group": group, "status": "missing"}
    if path.exists():
        try:
            data = validate_final_eval(path, require_terminal=True)
            meta = data["metadata"]
            row.update({
                "status": "canonical",
                "sha256": sha256_file(path),
                "train_seed": meta["train_seed"],
                "algorithm": meta["algorithm"],
                "checkpoint_iteration": meta["checkpoint_iteration"],
                "n_jammers": meta["n_jammers"],
                "n_radars": meta["n_radars"],
                "jammer_az_deg": meta["jammer_az_deg"],
                "radar_az_deg": meta["radar_az_deg"],
                "baseline_snr_db": meta["baseline_snr_db"],
                "P_jam_W": meta["P_jam_W"],
                "active_budget_steps": meta["active_budget_steps"],
                "horizon": meta["horizon"],
                "validation_seed_count": meta["validation_seed_count"],
                "action_seeds": meta["action_seeds"],
                "n_action_reps": meta["n_action_reps"],
                "code_revision": meta["code_revision"],
            })
        except FinalEvalSchemaError as exc:
            row.update({"status": "invalid", "error": str(exc)})
    entries.append(row)

invalid = BASE / "s7_ablation_output_seed20260811/final_eval_wrong_default_geometry.json"
if invalid.exists():
    entries.append({
        "path": str(invalid.relative_to(ROOT)).replace("\\", "/"),
        "group": "audit", "status": "invalid_wrong_geometry",
        "sha256": sha256_file(invalid),
    })

out = {"schema_version": 1, "generated_by": "_build_release_manifest.py",
       "canonical_count": sum(x["status"] == "canonical" for x in entries),
       "entries": entries}
path = ROOT / "paper" / "RELEASE_MANIFEST.json"
path.write_text(json.dumps(out, indent=2))
print(f"wrote {path}; canonical={out['canonical_count']}/{len(CASES)}")
for row in entries:
    if row["status"] != "canonical":
        print(row["status"], row["path"])
