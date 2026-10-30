"""Primitive tests for experts/core.py on fixed seeds (plain Python, no pytest needed).

    PYTHONPATH=. .venv/bin/python -m experts.test_core

Each test builds a one-world, one-agent CraftSim, drives a primitive generator decision by decision, and checks
the sim state afterwards (the same loop the Runner uses).
"""
import math
import random
import sys
import time

import numpy as np

import craftsim as cs
from experts.core import EYE, Ctl, _look_dir, astar, ray_hit, standable

FAILS = []


def make_sim(seed, task="gather_wood"):
    sim = cs.CraftSim(num_envs=1, agents=1, seed=seed, res=64, max_steps=3000, threads=1, action_mode="v7")
    v = np.zeros(cs.NUM_TASKS, np.float32)
    v[cs.TASK_NAMES.index(task)] = 1
    sim.set_task_weights(0, v)
    sim.reset()
    ctl = Ctl(sim, 0)
    ctl.new_episode(task, random.Random(seed))
    return sim, ctl


def drive(sim, ctl, gen, limit=4000):
    """Run a primitive to completion. Returns (its return value, decisions used)."""
    n = 0
    try:
        row = next(gen)
    except StopIteration as e:
        return e.value, n
    while True:
        img, vec, rew, done, ach = sim.step(np.array([row], np.int32))
        n += 1
        if done[0]:
            raise RuntimeError("episode ended during a primitive test")
        if n > limit:
            raise RuntimeError("primitive did not finish")
        try:
            row = gen.send(None)
        except StopIteration as e:
            return e.value, n


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def inv(sim, item):
    return sim.inventory(0).get(item, 0)


def angle_to(sim, ctl, cell):
    """Degrees between the crosshair and the centre of block `cell`."""
    s = sim.state(0)
    ex, ey, ez = s["x"], s["y"] + EYE, s["z"]
    dx, dy, dz = cell[0] + 0.5 - ex, cell[1] + 0.5 - ey, cell[2] + 0.5 - ez
    d = _look_dir(s["yaw"], s["pitch"])
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    dot = (d[0] * dx + d[1] * dy + d[2] * dz) / n
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def test_face():
    print("face: turn and look at a block within 15 deg")
    for seed in (1, 2, 3):
        sim, ctl = make_sim(seed)
        cands = [c for c, name in ctl.find_blocks(["oak_log", "oak_leaves", "stone", "dirt", "grass"], radius=5)
                 if 2 <= math.dist(c, ctl.cell()) <= 5]
        if not cands:
            continue
        target = cands[0]
        ok, n = drive(sim, ctl, ctl.face(*target))
        err = angle_to(sim, ctl, target)
        check(f"seed {seed}: face {target} (error {err:.1f} deg)", err <= 15.0, f"{n} decisions, hit={ok}")


def test_walk_20():
    print("walk_to: 20 blocks over terrain (goals confirmed reachable by A*)")
    passed = 0
    tried = 0
    for seed in (1, 2, 3, 4, 5, 6):
        sim, ctl = make_sim(seed)
        ctl.region_radius = 24
        reg = ctl.region()
        start = ctl.cell()
        goal = None
        for y_off in (0, 1, -1):
            for dx in range(-21, 22):
                for dz in range(-21, 22):
                    d = max(abs(dx), abs(dz))
                    if d < 18 or d > 21:
                        continue
                    c = (start[0] + dx, start[1] + y_off, start[2] + dz)
                    if standable(reg, *c):
                        path = astar(reg, start, lambda n, r, c=c: n == c, c[0], c[2], False)
                        if path is not None and len(path) >= 18:
                            goal = c
                            break
                if goal:
                    break
            if goal:
                break
        if goal is None:
            continue
        tried += 1
        ok, n = drive(sim, ctl, ctl.walk_to(*goal, max_decisions=800))
        end = ctl.cell()
        arrived = ok and end == goal
        if arrived:
            passed += 1
        check(f"seed {seed}: walk {max(abs(goal[0] - start[0]), abs(goal[2] - start[2]))} blocks to {goal}",
              arrived, f"{n} decisions, ended {end}")
    check("walk_to reaches reachable 20-block goals", tried > 0 and passed == tried, f"{passed}/{tried}")


def find_tree_near(sim, ctl, max_d=5):
    sites = [c for c, name in ctl.find_blocks(["oak_log"], radius=max_d) if name == "oak_log"]
    return sites[0] if sites else None


def test_mine_log():
    print("mine_block: mine a log, get oak_log")
    for seed in range(1, 12):
        sim, ctl = make_sim(seed)
        tree = find_tree_near(sim, ctl, 6)
        if tree is None:
            continue
        before = inv(sim, "oak_log")
        ok, n = drive(sim, ctl, ctl.mine_block(*tree))
        after = inv(sim, "oak_log")
        check(f"seed {seed}: mine log at {tree}", ok and after == before + 1 and ctl.block_at(*tree) == "air",
              f"oak_log {before}->{after}, {n} decisions")
        return
    check("found a tree within 6 blocks on seeds 1..11", False)


def craft_planks_from_log(sim, ctl):
    ok, _ = drive(sim, ctl, ctl.craft("oak_planks"))
    return ok


def test_craft_planks():
    print("craft: planks from a log, then a crafting table")
    for seed in range(1, 12):
        sim, ctl = make_sim(seed)
        tree = find_tree_near(sim, ctl, 6)
        if tree is None:
            continue
        ok, n = drive(sim, ctl, ctl.mine_block(*tree))
        if not ok:
            continue
        before = inv(sim, "oak_planks")
        ok = craft_planks_from_log(sim, ctl)
        check(f"seed {seed}: craft oak_planks from a log", ok and inv(sim, "oak_planks") == before + 4,
              f"planks {before}->{inv(sim, 'oak_planks')}")
        ok2, _ = drive(sim, ctl, ctl.craft("crafting_table"))
        check("craft crafting_table from 4 planks (no table needed)", ok2 and inv(sim, "crafting_table") == 1,
              f"table={inv(sim, 'crafting_table')} planks={inv(sim, 'oak_planks')}")
        return
    check("found a tree within 6 blocks on seeds 1..11", False)


def test_place_block():
    print("place_block: planks onto the intended cell")
    for seed in range(1, 12):
        sim, ctl = make_sim(seed)
        tree = find_tree_near(sim, ctl, 6)
        if tree is None:
            continue
        ok, _ = drive(sim, ctl, ctl.mine_block(*tree))
        if not ok or not craft_planks_from_log(sim, ctl):
            continue
        cx, cy, cz = ctl.cell()
        reg = ctl.region()
        # an empty cell at feet level, 2 blocks away on the ground, with solid ground under it
        target = None
        for dx, dz in ((2, 0), (-2, 0), (0, 2), (0, -2), (2, 1), (-2, 1), (1, 2), (1, -2)):
            c = (cx + dx, cy, cz + dz)
            if reg.get(*c) == 0 and reg.get(c[0], c[1] - 1, c[2]) > 0 and reg.get(c[0], c[1] + 1, c[2]) == 0:
                target = c
                break
        if target is None:
            continue
        before = sim.region(0, 16)[1].copy()
        ok, n = drive(sim, ctl, ctl.place_block("oak_planks", *target))
        after = sim.region(0, 16)[1]
        changed = np.argwhere(before != after)
        check(f"seed {seed}: place oak_planks at {target}",
              ok and ctl.block_at(*target) == "oak_planks" and len(changed) == 1,
              f"{n} decisions, changed cells {len(changed)}")
        return
    check("placed a block on seeds 1..11", False)


def test_task_obs_consistency():
    print("runner contract: the obs task one-hot matches the task of the world")
    sim = cs.CraftSim(num_envs=2, agents=1, seed=7, res=64, max_steps=3000, threads=1, action_mode="v7")
    names = ["gather_wood", "tools_crafting_table"]
    for w, t in enumerate(names):
        v = np.zeros(cs.NUM_TASKS, np.float32)
        v[cs.TASK_NAMES.index(t)] = 1
        sim.set_task_weights(w, v)
    img, vec = sim.reset()
    base = cs.VEC_NAMES.index(f"task_{names[0]}")
    for w, t in enumerate(names):
        hot = [cs.VEC_NAMES[i] for i in range(base, base + cs.NUM_TASKS) if vec[w, i] > 0.5]
        check(f"world {w} obs one-hot is task_{t}", hot == [f"task_{t}"], str(hot))


def test_region_and_ray_consistency():
    print("ray: the primitive ray agrees with the sim crosshair target")
    agree = 0
    total = 0
    for seed in (1, 2, 3):
        sim, ctl = make_sim(seed)
        for pitch_yaw in range(8):
            s = sim.state(0)
            hit = None
            d = _look_dir(s["yaw"], s["pitch"])
            hit = ray_hit(ctl.region(), ctl.eye(), d)
            sim_t = sim.state(0)["target"]
            total += 1
            if (hit is None and sim_t is None) or (hit and sim_t and tuple(hit[0]) == (sim_t["x"], sim_t["y"], sim_t["z"])):
                agree += 1
            # turn to a new orientation (any decision is fine: the sim state is re-read each time)
            drive(sim, ctl, ctl.turn_to_yaw((s["yaw"] + 45 * (pitch_yaw + 1)) % 360))
    check("primitive ray == sim target on sampled orientations", agree == total, f"{agree}/{total}")


def main():
    t0 = time.time()
    test_task_obs_consistency()
    test_region_and_ray_consistency()
    test_face()
    test_walk_20()
    test_mine_log()
    test_craft_planks()
    test_place_block()
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}  ({time.time() - t0:.1f}s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
