"""Explicitly upgrade legacy final_eval JSONs to schema v2.

This script is intentionally allowlist-based. It never discovers files by glob,
and it refuses skipped/wrong-geometry inputs. Stale n=4 checkpoints are not
upgraded as terminal evidence; they are upgraded only when explicitly marked
non-terminal and remain excluded by strict results-table expectations.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "paper" / "figures"))
from final_eval_schema import build_metadata, legacy_upgrade, validate_final_eval  # noqa: E402

VAL = ROOT / "experiments/array_face_s1/manifests/checkpoint_validation.json"
S6 = ROOT / "experiments/array_face_s6/learning_repair"
S7 = ROOT / "experiments/array_face_s7/learning_repair"

# (input path, output path, seed, algorithm, checkpoint iter, K, jammer az,
#  radar az, SNR, profile). Outputs are explicit and auditable.
CASES = []
for sd, it in ((20260730, 999), (20260731, 999), (20260732, 999)):
    CASES.append((S6 / f"s6_selfplay_output_seed{sd}/final_eval.json",
                  S6 / f"s6_selfplay_output_seed{sd}/final_eval_v2.json",
                  sd, "mappo", it, 1, (0.0,), (20.0, 20.0), 12.0, "array_face_s6_v1"))
for name, sd, it in (("s7_continue2_output_seed20260801", 20260801, 2999),
                     ("s7_seed02_cont_output_seed20260802", 20260802, 2999),
                     ("s7_seed03_cont_output_seed20260803", 20260803, 2999)):
    CASES.append((S7 / name / "final_eval.json", S7 / name / "final_eval_v2.json",
                  sd, "mappo", it, 2, (60.0, -60.0), (20.0, -20.0), 12.0,
                  "array_face_s7_v1"))
for sd in (20260811, 20260812, 20260813):
    CASES.append((S7 / f"s7_ablation_output_seed{sd}/final_eval.json",
                  S7 / f"s7_ablation_output_seed{sd}/final_eval_v2.json",
                  sd, "mappo", 1999, 2, (60.0, 60.0), (20.0, -20.0), 12.0,
                  "array_face_s7_v1"))
for sd in (20261011, 20261012, 20261013):
    CASES.append((S7 / f"s9_n3_output_seed{sd}/final_eval.json",
                  S7 / f"s9_n3_output_seed{sd}/final_eval_v2.json",
                  sd, "mappo", 1999, 3, (60.0, 0.0, -60.0), (20.0, -20.0), 12.0,
                  "array_face_s7_v1"))
# n=4 seed 21 is terminal; 22/23 are deliberately handled by the strict
# continuation script after 50 additional iterations and are not upgraded here.
CASES.append((S7 / "s9_n4_output_seed20261021/final_eval.json",
              S7 / "s9_n4_output_seed20261021/final_eval_v2.json",
              20261021, "mappo", 1999, 4, (60.0, 20.0, -20.0, -60.0),
              (20.0, -20.0), 12.0, "array_face_s7_v1"))

for src, dst, seed, algo, it, k, jaz, raz, snr, profile in CASES:
    if not src.exists():
        print(f"SKIP missing {src}")
        continue
    if not dst.exists():
        data = json.loads(src.read_text())
        if data.get("skipped") is True:
            raise SystemExit(f"refusing skipped artifact: {src}")
        metadata = build_metadata(
            train_seed=seed, algorithm=algo, checkpoint_iteration=it,
            n_jammers=k, n_radars=2, jammer_az_deg=jaz, radar_az_deg=raz,
            baseline_snr_db=snr, P_jam_W=0.1, active_budget_steps=63,
            horizon=64, validation_manifest=VAL,
            action_seeds=[4242, 777, 31337], n_action_reps=1,
            env_profile=profile, code_rev="legacy-upgrade-source",
        )
        upgraded = legacy_upgrade(data, metadata)
        dst.write_text(json.dumps(upgraded, indent=2))
    validate_final_eval(dst)
    print(f"OK {dst}")
