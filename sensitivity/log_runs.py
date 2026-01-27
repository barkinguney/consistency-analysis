from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import platform
import subprocess


def make_run_dir(root: str | Path = "runs", prefix: str = "") -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{prefix}{ts}" if prefix else ts
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=False)  # fail if collision
    return run_dir


def save_config(run_dir: Path, config: dict) -> None:
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))


def snapshot_script(run_dir: Path) -> None:
    # Best-effort: copy the file that launched the process
    src = Path(sys.argv[0]).resolve()
    if src.exists() and src.is_file():
        shutil.copy2(src, run_dir / "script.py")


def save_env(run_dir: Path) -> None:
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
    }
    (run_dir / "env.json").write_text(json.dumps(meta, indent=2))

    # Optional: capture pip freeze if available
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (run_dir / "pip_freeze.txt").write_text(freeze)
    except Exception:
        pass


def tee_stdout_stderr(run_dir: Path) -> None:
    """
    Redirect stdout/stderr to files while still printing to console.
    """
    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    out_f = open(run_dir / "stdout.log", "a", buffering=1)
    err_f = open(run_dir / "stderr.log", "a", buffering=1)

    atexit.register(out_f.close)
    atexit.register(err_f.close)

    sys.stdout = Tee(sys.__stdout__, out_f)
    sys.stderr = Tee(sys.__stderr__, err_f)


def setup_run(config: dict, root: str | Path = "runs", prefix: str = "") -> Path:
    run_dir = make_run_dir(root=root, prefix=prefix)
    snapshot_script(run_dir)

    tee_stdout_stderr(run_dir)
    print(f"[run_dir] {run_dir}")
    return run_dir


# ---- example usage ----
if __name__ == "__main__":
    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "seed": 42,
        "note": "baseline",
    }

    run_dir = setup_run(config, prefix="sensitivity_results/exp1_")

    # Write outputs wherever you like under run_dir
    results_path = run_dir / "metrics.json"
    results_path.write_text(json.dumps({"loss": 0.123}, indent=2))
    print("done")
