from __future__ import annotations

"""Compatibility entry point forwarding to the repository-root one-click runner."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "run_experiments.py"

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, str(ENTRY), *sys.argv[1:]], cwd=ROOT))
