#!/usr/bin/env python3
"""main.py — Phase 1.5 三方对比统一纯 Python 入口。

Usage:
    python main.py --config algo/pspfix/code/config.yaml
    python main.py --config algo/mappo/code/config.yaml
    python main.py --config algo/ippo/code/config.yaml

For background running:
    nohup python main.py --config ... > log.txt 2>&1 &

Self-contained: activates conda env via subprocess if not already active,
enables faulthandler, writes PID file, then forwards to
algo._shared.train_laser:main.
"""
import os
import shlex
import subprocess
import sys

EXPECTED_ENV = "fluxphased"
CONDA_SH = "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
PID_FILE = "/tmp/train_laser.pid"


def _in_fluxphased_env() -> bool:
    """True if sys.prefix points at the fluxphased conda env."""
    return os.path.basename(sys.prefix.rstrip("/")) == EXPECTED_ENV


def _reexec_in_fluxphased() -> None:
    """Re-launch main.py under `conda activate fluxphased` via bash subprocess."""
    if not os.path.isfile(CONDA_SH):
        sys.exit(
            f"[main.py] ERROR: conda.sh not found at {CONDA_SH}\n"
            f"  → activate fluxphased manually and re-run:\n"
            f"      conda activate {EXPECTED_ENV}\n"
            f"      python {' '.join(shlex.quote(a) for a in sys.argv)}"
        )
    # After `conda activate`, "python" resolves to the fluxphased env's interpreter.
    # Use sys.executable's basename as fallback if python isn't on PATH post-activate.
    script = os.path.abspath(sys.argv[0])
    rest = " ".join(shlex.quote(a) for a in sys.argv[1:])
    outer = (
        f"source {shlex.quote(CONDA_SH)} && "
        f"conda activate {shlex.quote(EXPECTED_ENV)} && "
        f"exec python {shlex.quote(script)} {rest}"
    )
    result = subprocess.run(["bash", "-c", outer])
    sys.exit(result.returncode)


def main() -> None:
    if not _in_fluxphased_env():
        _reexec_in_fluxphased()
        return  # unreachable; _reexec_in_fluxphased exits

    import faulthandler
    faulthandler.enable()

    try:
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    except OSError as e:
        print(f"[main.py] WARN: cannot write PID file {PID_FILE}: {e}", file=sys.stderr)

    sys.argv[0] = "main.py"
    from algo._shared.train_laser import main as train_main
    train_main()


if __name__ == "__main__":
    main()
