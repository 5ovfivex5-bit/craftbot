"""MineMind PPO trainer and telemetry server (see README.md).

The observation version is fixed per process: "v7" (default, the v7 sim's 295-dim obs, 21 primary
actions + recipe / item / msg heads) or "v6" (the 102-dim obs, 28 actions). It is read from the
MINEMIND_OBS environment variable, or from a `--obs v6|v7` argument on the command line, before
any module reads the sim constants (see envs.py).
"""
import os as _os
import sys as _sys


def _obs_from_argv(argv):
    for i, a in enumerate(argv):
        if a == "--obs" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--obs="):
            return a.split("=", 1)[1]
    return None


if "MINEMIND_OBS" not in _os.environ:
    _o = _obs_from_argv(_sys.argv)
    _os.environ["MINEMIND_OBS"] = _o if _o in ("v6", "v7") else "v7"
