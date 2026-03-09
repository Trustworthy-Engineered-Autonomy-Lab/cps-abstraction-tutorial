"""Self-loop-removal pipeline runner (docker-friendly).

The self-loop-removal pipelines live under src/selfloop-pipelines/*.py and are
intended to be run as scripts.

This dispatcher lets users run them without referencing src/ paths.

Examples:
  python -u scripts/run_selfloop.py --case synthetic
  python -u scripts/run_selfloop.py --case mountain_car
  python -u scripts/run_selfloop.py --case unicycle

Docker:
  docker run --rm -v ${PWD}/runs:/app/runs <image> python -u scripts/run_selfloop.py --case synthetic

Notes:
- Output paths are controlled by the underlying pipeline scripts and default to:
  runs/selfloop/<case>/...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]

_CASE_TO_SCRIPT = {
	"synthetic": _REPO_ROOT / "src" / "selfloop-pipelines" / "abstract_synthetic.py",
	"mountain_car": _REPO_ROOT / "src" / "selfloop-pipelines" / "abstract_mountain_car.py",
	"unicycle": _REPO_ROOT / "src" / "selfloop-pipelines" / "abstract_unicycle.py",
}


def parse_args(argv=None):
	p = argparse.ArgumentParser()
	p.add_argument("--case", choices=sorted(_CASE_TO_SCRIPT.keys()), required=True)
	p.add_argument(
		"--dry-run",
		action="store_true",
		help="Print the underlying command without running it.",
	)
	return p.parse_args(argv)


def main(argv=None) -> int:
	args = parse_args(argv)

	script_path = _CASE_TO_SCRIPT[args.case]
	if not script_path.exists():
		raise FileNotFoundError(f"Pipeline script not found: {script_path}")

	cmd = [sys.executable, "-u", str(script_path)]

	if args.dry_run:
		print(" ".join(cmd))
		return 0

	# Run from repo root so relative output paths land in ./runs.
	subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

