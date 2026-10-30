#!/usr/bin/env python3
"""Run the existing project checks in a consistent order; no pytest required."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also run simulator subsystem tests")
    args = parser.parse_args()
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2",
           "MINEMIND_OBS": "v7"}
    checks = [["tools/registry/validate.py"], ["tasklib/test_tasklib.py"],
              ["tests/test_sim.py"]]
    if args.full:
        checks += [[f"tests/test_{name}.py"] for name in ("mobs", "survival", "nether", "build")]
        checks += [["-m", "experts.test_core"], ["-m", "trainer.test_bc_v7"]]
    failed = []
    for check in checks:
        print(f"\n>>> {' '.join(check)}", flush=True)
        result = subprocess.run([sys.executable, *check], cwd=ROOT, env=env)
        if result.returncode:
            failed.append(" ".join(check))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} check programs passed.")
    if failed:
        print("Failed: " + ", ".join(failed))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
