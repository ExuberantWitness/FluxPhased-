"""Generate four disjoint scenario manifests for the array_face_s1 branch.

Mirrors experiments/g3_bsta_lite/learning_repair/manifests/generate_manifests.py
but uses a new seed block `2101xxxx` (vs lite's `2100xxxx`) to guarantee
disjointness from lite manifests and from the F5/F6 legacy range.

Preregistered in experiments/array_face_s1/PREREGISTRATION.md §3.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
import torch

from env.gpu.g3_bsta_lite.scenario import generate_paired_manifest


MANIFEST_SPECS = {
    "dagger_train":           {"base_seed": 21010101, "size": 128},
    "ppo_train":              {"base_seed": 21011101, "size":  64},
    "checkpoint_validation":  {"base_seed": 21012101, "size":  64},
    "locked_test":            {"base_seed": 21013101, "size": 128},
}

PROFILE = {
    "horizon": 64,
    "n_services": 2,
    "arrival_rate_per_service": 0.15,
    "baseline_snr_db": 22.0,
    "device": "cpu",
}

# Legacy ranges to exclude (must be disjoint from)
# 1. F5/F6 fast-work legacy range
# 2. lite's 2100xxxx manifest block (21000101..21004099, conservative bound)
LEGACY_EXCLUDED_RANGES = [
    (20260801, 20260832),
    (21000101, 21004099),
]


def _git_commit() -> str:
    here = Path(__file__).resolve()
    repo = here
    while repo != repo.parent:
        if (repo / ".git").exists():
            break
        repo = repo.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _arrivals_sha256(arrivals: torch.Tensor) -> str:
    arr = arrivals.bool().cpu().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


def _build_one(name: str, spec: dict, generator_commit: str) -> dict:
    scenarios = generate_paired_manifest(
        base_seed=spec["base_seed"],
        n_scenarios=spec["size"],
        horizon=PROFILE["horizon"],
        n_services=PROFILE["n_services"],
        arrival_rate_per_service=PROFILE["arrival_rate_per_service"],
        baseline_snr_db=PROFILE["baseline_snr_db"],
        device=PROFILE["device"],
    )
    entries = []
    for s in scenarios:
        entries.append({
            "seed": int(s.seed),
            "eligible": bool(s.eligible()),
            "baseline_snr_db": float(PROFILE["baseline_snr_db"]),
            "arrivals_shape": list(s.arrivals.shape),
            "arrivals_sha256": _arrivals_sha256(s.arrivals),
        })
    return {
        "manifest": name,
        "base_seed": int(spec["base_seed"]),
        "size": int(spec["size"]),
        "profile": dict(PROFILE),
        "generator_commit": generator_commit,
        "entries": entries,
    }


def _pairwise_intersection(manifests: dict[str, dict]) -> dict[str, dict]:
    audit: dict[str, dict] = {}
    names = list(manifests.keys())
    for i, a in enumerate(names):
        seeds_a = {e["seed"] for e in manifests[a]["entries"]}
        for b in names[i + 1:]:
            seeds_b = {e["seed"] for e in manifests[b]["entries"]}
            inter = sorted(seeds_a & seeds_b)
            audit[f"{a}__{b}"] = {
                "intersection_size": len(inter),
                "intersection_seeds": inter,
                "verdict": "DISJOINT" if len(inter) == 0 else "OVERLAP",
            }
    return audit


def _legacy_range_check(manifests: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, m in manifests.items():
        seeds = {e["seed"] for e in m["entries"]}
        leaked_all = []
        for lo, hi in LEGACY_EXCLUDED_RANGES:
            leaked = sorted(s for s in seeds if lo <= s <= hi)
            leaked_all.extend([(lo, hi, s) for s in leaked])
        out[name] = {
            "legacy_ranges": [list(r) for r in LEGACY_EXCLUDED_RANGES],
            "leaked_count": len(leaked_all),
            "leaked": leaked_all,
            "verdict": "CLEAN" if len(leaked_all) == 0 else "LEGACY_OVERLAP",
        }
    return out


def _sha256sums_text(manifests: dict[str, dict]) -> str:
    lines = []
    for name, m in manifests.items():
        for e in m["entries"]:
            lines.append(f"{e['arrivals_sha256']}  {name}/seed={e['seed']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    generator_commit = _git_commit()
    out_dir = Path(__file__).resolve().parent

    manifests: dict[str, dict] = {}
    for name, spec in MANIFEST_SPECS.items():
        manifests[name] = _build_one(name, spec, generator_commit)

    pairwise = _pairwise_intersection(manifests)
    legacy = _legacy_range_check(manifests)

    audit = {
        "document": "MANIFEST_AUDIT.json",
        "generated_by_commit": generator_commit,
        "preregistration": "experiments/array_face_s1/PREREGISTRATION.md",
        "profile": dict(PROFILE),
        "manifest_sizes": {n: len(m["entries"]) for n, m in manifests.items()},
        "pairwise_intersections": pairwise,
        "legacy_range_check": legacy,
        "overall_verdict": (
            "ALL_DISJOINT_AND_LEGACY_CLEAN"
            if all(v["verdict"] == "DISJOINT" for v in pairwise.values())
            and all(v["verdict"] == "CLEAN" for v in legacy.values())
            else "AUDIT_FAILURE"
        ),
    }

    for name, m in manifests.items():
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(m, f, indent=2, sort_keys=True)
    with open(out_dir / "MANIFEST_AUDIT.json", "w") as f:
        json.dump(audit, f, indent=2, sort_keys=True)
    with open(out_dir / "SHA256SUMS.txt", "w") as f:
        f.write(_sha256sums_text(manifests))

    print(f"manifests written to {out_dir}")
    print(f"  sizes: {audit['manifest_sizes']}")
    print(f"  overall_verdict: {audit['overall_verdict']}")
    if audit["overall_verdict"] != "ALL_DISJOINT_AND_LEGACY_CLEAN":
        print("AUDIT FAILURE — pairwise disjointness or legacy exclusion failed")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
