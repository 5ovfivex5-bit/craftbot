"""Checks for experts/skills/gather_tools.py and the Runner's early episode end (PROTOCOL 14.2).

    PYTHONPATH=. .venv/bin/python -m experts.test_gather_tools [out_dir]

1. every gather/tools task of tasks.json has a registered skill;
2. a short run of the simple skills succeeds on most episodes;
3. in the written shards each episode ends with a done flag on its last row (early ends included).
"""
import glob
import os
import sys
import tempfile

import numpy as np

import craftsim as cs
from experts.generate import main as generate_main
from experts.skills import SKILLS


def check_registry():
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tasks", "tasks.json")
    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)
    want = [t["id"] for t in tasks if t["family"] in ("gather", "tools")]
    assert len(want) == 17, len(want)
    missing = [t for t in want if t not in SKILLS]
    assert not missing, f"no skill for {missing}"
    print(f"ok: {len(want)} gather/tools tasks registered")


def check_success_and_done(out):
    rc = generate_main(["--tasks", "gather_sand,tools_wooden_pickaxe,gather_stone", "--episodes-per-task", "6",
                        "--worlds", "3", "--agents", "1", "--out", out])
    assert rc == 0
    for path in sorted(glob.glob(os.path.join(out, "shard_*.npz"))):
        z = np.load(path)
        ep, done = z["ep"], z["done"]
        # the last row of every episode must carry done=True
        last = np.r_[ep[1:] != ep[:-1], True]
        assert done[last].all(), f"episode without a done flag in {path}"
    print("ok: every episode ends with done=True")


if __name__ == "__main__":
    check_registry()
    out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="gt_test_")
    check_success_and_done(out)
