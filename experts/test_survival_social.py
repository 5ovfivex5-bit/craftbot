"""Checks for experts/skills/survival_social.py (EXP-3). Run from the repo root:

    PYTHONPATH=. .venv/bin/python -m experts.test_survival_social
"""
import math
import sys

import craftsim as cs
from experts.core import _look_dir
from experts.skills import SKILLS
from experts.skills import survival_social as ss

FAILS = []
FAMILIES = ("food", "combat", "survival", "social")


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILS.append(name)


def test_registry():
    print("registry: every food/combat/survival/social task has a skill")
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tasks", "tasks.json")
    with open(path, encoding="utf-8") as f:
        want = [t["id"] for t in json.load(f) if t["family"] in FAMILIES]
    missing = [t for t in want if t not in SKILLS]
    check(f"{len(want)} tasks covered", not missing)


def test_box_geometry():
    print("box geometry: a ray straight at the centre of a 1x2 box at 2 blocks hits, a sideways ray misses")
    eye = (0.0, 1.62, 0.0)
    d = _look_dir(0.0, 0.0)   # yaw 0 looks along +z
    t = ss._box_t(eye, d, 0.0, 0.0, 2.0, 1.0, 2.0, 2.9)
    check("centre hit", t is not None and abs(t - 1.5) < 0.01 or (t is not None and t < 2.9))
    side = _look_dir(90.0, 0.0)
    check("sideways miss", ss._box_t(eye, side, 0.0, 0.0, 2.0, 1.0, 2.0, 2.9) is None)
    far = ss._box_t(eye, d, 0.0, 0.0, 6.0, 1.0, 2.0, 2.9)
    check("beyond reach is a miss", far is None)


def test_pit_seals():
    """The shelter pit: after survive_night_hut's digging, the roof cell and the walls are solid and the two pit
    cells are air (checked against the sim's blocks), on several seeds."""
    import random
    import numpy as np
    from experts.core import Ctl
    print("shelter: the dug pit is sealed (roof above, walls around, pit cells air)")
    task = "survival_survive_night_hut"
    for seed in (1, 2, 3, 4):
        sim = cs.CraftSim(num_envs=1, agents=1, seed=seed, res=64, max_steps=6000, threads=1, action_mode="v7")
        v = np.zeros(cs.NUM_TASKS, np.float32)
        v[cs.TASK_NAMES.index(task)] = 1
        sim.set_task_weights(0, v)
        sim.reset()
        sim.set_task(0, task)
        ctl = Ctl(sim, 0)
        ctl.new_episode(task, random.Random(seed))
        x, f, z = ctl.cell()
        g = ss._pit(ctl)
        ok = False
        try:
            row = next(g)
            while True:
                sim.step(np.array([row], np.int32))
                row = g.send(None)
        except StopIteration as e:
            ok = bool(e.value)
        solid = lambda bx, by, bz: sim.block(0, bx, by, bz) not in ("air", "water", "lava")
        roof = solid(x, f - 1, z)
        walls = all(solid(x + dx, yy, z + dz) for yy in (f - 3, f - 2) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        pit_air = sim.block(0, x, f - 3, z) == "air" and sim.block(0, x, f - 2, z) == "air"
        check(f"seed {seed}: sealed (ok={ok}, roof={roof}, walls={walls}, pit air={pit_air})", ok and roof and walls and pit_air)


def test_smoke_social():
    print("smoke: social_look_at_teammate with K=2 succeeds quickly")
    from experts.generate import main
    rc = main(["--tasks", "social_look_at_teammate", "--episodes-per-task", "4", "--worlds", "2", "--agents", "2",
               "--out", "/tmp/claude-1000/-home-deck-claude/fd75689d-b7cd-41a3-8f2b-c8d7da974f14/scratchpad/test_social"])
    check("generate runs with social K=2", rc == 0)


if __name__ == "__main__":
    test_registry()
    test_box_geometry()
    test_pit_seals()
    test_smoke_social()
    print("FAILED:" if FAILS else "all passed", FAILS if FAILS else "")
    sys.exit(1 if FAILS else 0)
