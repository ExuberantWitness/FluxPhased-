"""Strict n-scale supplementation driver.

The canonical curve uses n=2/n=3/n=4 with the same 2000-iteration terminal
policy and the same 64x3 final-evaluation protocol. Existing legacy outputs
are copied into separate directories; no historical final_eval.json is
overwritten.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
BASE = ROOT / "experiments/array_face_s7/learning_repair"


def run(cmd: list[str], log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        print("$", " ".join(cmd), file=f, flush=True)
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def max_iter(path: Path) -> int:
    if not path.exists():
        return -1
    for line in reversed(path.read_text(errors="replace").splitlines()):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "iteration" in obj:
            return int(obj["iteration"])
    return -1


def ensure_copy(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "selfplay_latest.pt").exists():
        shutil.copy2(src / "selfplay_latest.pt", dst / "selfplay_latest.pt")


def continue_s7(*, src: Path, dst: Path, seed: int, n: int, jammer_az: str,
                target: int = 2000):
    ensure_copy(src, dst)
    # Checkpoint is the authoritative resume point; train_metrics may contain
    # rows beyond the last atomic save after an interrupted process.
    probe = subprocess.check_output(
        [PY, "-c", "import torch,sys; print(torch.load(sys.argv[1],map_location='cpu')['iteration'])",
         str(dst / "selfplay_latest.pt")], cwd=ROOT, text=True).strip()
    stored = int(probe)
    if stored < target - 1:
        run([
            PY, "-u", "experiments/array_face_s7/learning_repair/run_s7_selfplay.py",
            "--seed", str(seed), "--resume", "--iterations", str(target),
            "--anneal-done", "--val-every", "50", "--n-jammers", str(n),
            "--jammer-az", jammer_az, "--out-dir", str(dst),
        ], dst / "strict_train.log")
    # Evaluate into a new filename with schema-v2 metadata.
    run([
        PY, "-u", "_s7_final_eval.py", "--seed", str(seed),
        "--n-jammers", str(n), "--jammer-az", jammer_az,
        "--out-dir", str(dst), "--output-name", "final_eval_v2.json",
        "--device", "cpu",
    ], dst / "strict_eval.log")


def main():
    # n=2 seed 1: existing 2000 continuation checkpoint; only clean eval.
    n2a = BASE / "s7_strict_n2_output_seed20260801"
    ensure_copy(BASE / "s7_continue_output_seed20260801", n2a)
    if not (n2a / "final_eval_v2.json").exists():
        run([PY, "-u", "_s7_final_eval.py", "--seed", "20260801",
             "--n-jammers", "2", "--jammer-az", "+60,-60",
             "--out-dir", str(n2a), "--output-name", "final_eval_v2.json",
             "--device", "cpu"], n2a / "strict_eval.log")

    # n=2 seeds 2/3: original 999 checkpoint -> 1999 terminal.
    for seed in (20260802, 20260803):
        continue_s7(
            src=BASE / f"s7_selfplay_output_seed{seed}",
            dst=BASE / f"s7_strict_n2_output_seed{seed}",
            seed=seed, n=2, jammer_az="+60,-60",
        )

    # n=3: clean eval from existing terminal checkpoints.
    for seed in (20261011, 20261012, 20261013):
        dst = BASE / f"s9_n3_output_seed{seed}"
        if not (dst / "final_eval_v2.json").exists():
            run([PY, "-u", "_s7_final_eval.py", "--seed", str(seed),
                 "--n-jammers", "3", "--jammer-az", "+60,0,-60",
                 "--out-dir", str(dst), "--output-name", "final_eval_v2.json",
                 "--device", "cpu"], dst / "strict_eval.log")

    # n=4 seed 1 terminal; 2/3 require the saved checkpoint 1949 -> 1999.
    dst = BASE / "s9_n4_output_seed20261021"
    if not (dst / "final_eval_v2.json").exists():
        run([PY, "-u", "_s7_final_eval.py", "--seed", "20261021",
             "--n-jammers", "4", "--jammer-az", "+60,+20,-20,-60",
             "--out-dir", str(dst), "--output-name", "final_eval_v2.json",
             "--device", "cpu"], dst / "strict_eval.log")
    for seed in (20261022, 20261023):
        continue_s7(
            src=BASE / f"s9_n4_output_seed{seed}",
            dst=BASE / f"s9_strict_n4_output_seed{seed}",
            seed=seed, n=4, jammer_az="+60,+20,-20,-60",
        )
    print("STRICT NSCALE SUPPLEMENT COMPLETE")


if __name__ == "__main__":
    main()
