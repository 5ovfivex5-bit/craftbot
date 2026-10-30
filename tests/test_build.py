"""Build targets and worldgen variety (PROTOCOL 12.3, 12.4, 12.6; SIM-4).

Run: .venv/bin/python tests/test_build.py   (after ./build.sh)

Target tests: the build task's blueprint is placed lazily on the first obs of the episode; progress counts matched
solid cells (minus violated required-air cells) / solid cells; build_complete fires at full progress. The placement
uses the sim's own use action (test_real_placement) and, for the full build, a save_state/load_state round trip
that writes the wanted blocks into the blob (there is no set-block binding; the blob's block array starts at byte 16).
Worldgen tests: determinism, spawn rule, biome variety, exposed coal in cave walls, villages/chests, boulders, time.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from craftsim import (  # noqa: E402
    BLOCKS, ITEMS, PRIMARY_NAMES, SEA_LEVEL, TASK_NAMES, V7_BUILD, VEC_DIM, CraftSim,
)

WX, WY, WZ = 128, 64, 128
BLOB_HEADER = 16                       # SNAP magic (u32) + version (u32) + block-array length (u64)
PRIM = {n: i for i, n in enumerate(PRIMARY_NAMES)}
AIR, WATER, GRASS, SAND, LOG, LEAVES = (BLOCKS.index(n) for n in ("air", "water", "grass", "sand", "oak_log", "oak_leaves"))
COAL = BLOCKS.index("coal_ore")
DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
PORTAL_CELLS = 14                      # nether_portal: 14 obsidian cells, 6 required-air interior cells
FARM_CELLS = 25                        # farm_plot: 25 solid cells (dirt + planks)


def sim(n=1, seed=1, agents=1, res=8, max_steps=3000):
    return CraftSim(action_mode="v7", num_envs=n, agents=agents, seed=seed, res=res, max_steps=max_steps, ticks=1,
                    threads=1)


def world_array(s, w=0):
    """Block ids of world w as uint8[y, z, x] (from save_state; read-only use)."""
    blob = s.save_state(w)
    return np.frombuffer(blob, dtype=np.uint8, count=WX * WY * WZ, offset=BLOB_HEADER).reshape(WY, WZ, WX)


def row(vec):
    """One agent's obs row (accepts a row or the [N, V] array returned by step)."""
    v = np.asarray(vec)
    return v[0] if v.ndim == 2 else v


def decode_slots(vec, st):
    """The 8 build slots of one obs row -> [((x, y, z), item index)] in slot order (nearest first, empties dropped)."""
    vec = row(vec)
    yr = math.radians(st["yaw"])
    fx, fz, rx, rz = -math.sin(yr), math.cos(yr), -math.cos(yr), -math.sin(yr)
    out = []
    for j in range(8):
        b = V7_BUILD + 4 * j
        xl, yl, zl = vec[b] * 8, vec[b + 1] * 8, vec[b + 2] * 8
        if xl == 0 and yl == 0 and zl == 0:
            continue
        dx = xl * rx + zl * fx
        dz = xl * rz + zl * fz
        cell = tuple(math.floor(v) for v in (st["x"] + dx, st["y"] + yl, st["z"] + dz))
        out.append((cell, int(round(float(vec[b + 3]) * len(ITEMS)))))
    return out


def progress_of(vec):
    return float(row(vec)[V7_BUILD + 32])


def set_target_task(s, name):
    s.set_inventory(0, "obsidian", 64)
    s.set_inventory(0, "oak_planks", 64)
    s.set_inventory(0, "dirt", 64)
    s.set_task(0, name, None, None)
    img, vec, rew, done, ach = s.step(np.zeros((1, 4), np.int32))
    return row(vec), bool(done[0]), int(ach[0])


# ---------------------------------------------------------------- target


def test_target_placed_for_build_task():
    s = sim(seed=1)
    s.reset_env(0, 1)
    vec, done, _ = set_target_task(s, "build_farm_plot")
    assert TASK_NAMES[s.task(0)] == "build_farm_plot"
    slots = decode_slots(vec, s.state(0))
    assert len(slots) == 8, f"expected 8 nearest unfilled cells, got {len(slots)}"
    assert progress_of(vec) == 0.0
    spawn = s.state(0)
    for cell, item in slots:
        d = math.hypot(cell[0] + 0.5 - spawn["x"], cell[2] + 0.5 - spawn["z"])
        assert d <= 16.0, (cell, d)
        assert ITEMS[item] in ("oak_planks", "dirt"), ITEMS[item]
        assert s.block(0, *cell) in ("grass", "air", "dirt", "sand"), s.block(0, *cell)
    print(f"  farm plot target: nearest cell {slots[0][0]} wants {ITEMS[slots[0][1]]}, {len(slots)} slots")


def test_obs_slots_non_build_task_zero():
    s = sim(seed=2)
    s.reset_env(0, 2)
    s.set_task(0, "build_farm_plot", None, None)
    img, vec, rew, done, ach = s.step(np.zeros((1, 4), np.int32))
    assert decode_slots(vec[0], s.state(0))
    s.set_task(0, "tools_crafting_table", None, None)
    img, vec, rew, done, ach = s.step(np.zeros((1, 4), np.int32))
    assert all(float(v) == 0.0 for v in vec[0][V7_BUILD:V7_BUILD + 33]), "non-build task must have zero build slots"
    assert VEC_DIM == V7_BUILD + 33


def test_save_load_keeps_target():
    s = sim(seed=3)
    s.reset_env(0, 3)
    vec, _, _ = set_target_task(s, "build_farm_plot")
    before = decode_slots(vec, s.state(0))
    blob = s.save_state(0)
    s.load_state(0, blob)
    img, vec3, rew, done, ach = s.step(np.zeros((1, 4), np.int32))
    after = decode_slots(vec3[0], s.state(0))
    assert before and after == before, (before, after)


def block_cell_index(x, y, z):
    return (y * WZ + z) * WX + x


def patch_blob(blob, cells_items):
    """Write the wanted block of each (cell, item) into a save_state blob."""
    blob = bytearray(blob)
    for (x, y, z), item in cells_items:
        blob[BLOB_HEADER + block_cell_index(x, y, z)] = BLOCKS.index(ITEMS[item])
    return bytes(blob)


def test_build_complete_fires_with_full_build():
    """nether_build_portal: fill the 14 solid cells through save/load (the target survives the load via the placement
    memo), progress goes 0 -> 8/14 -> 1.0, and the build_complete success ends the episode (ach bit 31)."""
    s = sim(seed=2)
    s.reset_env(0, 2)
    vec, done, ach = set_target_task(s, "nether_build_portal")
    assert not done and progress_of(vec) == 0.0
    assert len(decode_slots(vec, s.state(0))) == 8
    filled = 0
    rounds = 0
    while True:
        rounds += 1
        slots = decode_slots(vec, s.state(0))
        if not slots:
            break
        blob = s.save_state(0)
        out = s.load_state(0, patch_blob(blob, slots))
        img, vec, rew, done, ach = s.step(np.zeros((1, 4), np.int32))
        filled += len(slots)
        if done[0]:
            assert (int(ach[0]) >> 31) & 1, "task success bit (build_complete) not set"
            break
        expect = filled / PORTAL_CELLS
        assert abs(progress_of(vec) - expect) < 1e-6, (progress_of(vec), expect, filled)
        assert rounds < 6
    assert done[0], "build_complete did not end the episode"
    print(f"  portal built in {rounds} patch rounds; success bit set, episode ended")


def test_progress_exact_with_placed_blocks():
    """Real placement: walk to a target cell, aim at a face of a neighbouring solid block, use the held block.
    Progress must rise by exactly 1/solid for each correct placement (several seeds tried)."""
    total = 0
    for seed in range(1, 9):
        s = sim(seed=seed)
        s.reset_env(0, seed)
        vec, done, ach = set_target_task(s, "nether_build_portal")
        placed, last = walk_and_place(s, vec, want=3)
        total += placed
        if placed >= 2:
            assert abs(progress_of(last) - placed / PORTAL_CELLS) < 1e-6, (progress_of(last), placed)
            print(f"  seed {seed}: {placed} placements, progress {progress_of(last):.4f} = {placed}/{PORTAL_CELLS}")
            return
    assert total >= 2, "no seed allowed two placements with the use action"


def walk_and_place(s, vec, want):
    placed = 0
    bad = {}
    for it in range(600):
        st = s.state(0)
        slots = decode_slots(vec, st)
        if not slots or placed >= want:
            break
        pick = None
        for cell, item in slots:
            if bad.get(cell, 0) >= 2:
                continue
            faces = [d for d in DIRS if s.block(0, cell[0] + d[0], cell[1] + d[1], cell[2] + d[2]) not in ("air", "water")]
            if faces:
                pick = (cell, item, faces)
                break
        if pick is None:
            cell = slots[0][0]
            vec, done = walk_step(s, cell, vec)
            if done:
                break
            continue
        cell, item, faces = pick
        if math.hypot(cell[0] + 0.5 - st["x"], cell[2] + 0.5 - st["z"]) > 3.0:
            vec, done = walk_step(s, cell, vec)
            if done:
                break
            continue
        ok = False
        for d in faces:
            n = (cell[0] + d[0], cell[1] + d[1], cell[2] + d[2])
            aim_at(s, cell, d)
            tg = s.state(0)["target"]
            if tg and (tg["x"], tg["y"], tg["z"]) == n:
                img, vec, rew, done, ach = s.step(np.array([[PRIM["use"], 0, item + 1, 0]], np.int32))
                if done[0]:
                    return placed, vec[0]
                if s.block(0, *cell) == ITEMS[item]:
                    placed += 1
                    ok = True
                break
        if not ok:
            bad[cell] = bad.get(cell, 0) + 1
            st = s.state(0)
            s.set_pose(0, st["yaw"] + 37, st["pitch"])
            img, vec, rew, done, ach = s.step(np.zeros((1, 4), np.int32))
    return placed, vec[0]


def aim_at(s, cell, d):
    """Set the pose so the crosshair centre hits the face of the block at cell + d that touches cell."""
    st = s.state(0)
    n = (cell[0] + d[0], cell[1] + d[1], cell[2] + d[2])
    ax, ay, az = n[0] + 0.5 - 0.5 * d[0], n[1] + 0.5 - 0.5 * d[1], n[2] + 0.5 - 0.5 * d[2]
    dx, dz = ax - st["x"], az - st["z"]
    dy = ay - (st["y"] + 1.62)
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = -math.degrees(math.atan2(dy, math.hypot(dx, dz)))
    s.set_pose(0, yaw, pitch)


def walk_step(s, cell, vec):
    st = s.state(0)
    base = math.degrees(math.atan2(-(cell[0] + 0.5 - st["x"]), cell[2] + 0.5 - st["z"]))
    fy = math.floor(st["y"])

    def solid(x, y, z):
        return s.block(0, x, y, z) not in ("air", "water")

    for off in (0, 45, 90, 135, 180, 225, 270, 315):
        yaw = base + off
        fx, fz = -math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        ax, az = math.floor(st["x"] + 0.8 * fx), math.floor(st["z"] + 0.8 * fz)
        if not solid(ax, fy, az) and not solid(ax, fy + 1, az) and solid(ax, fy - 1, az):
            prim = "forward"
        elif solid(ax, fy, az) and not solid(ax, fy + 1, az) and not solid(ax, fy + 2, az):
            prim = "forward_jump"
        else:
            continue
        s.set_pose(0, yaw, 0.0)
        img, vec, rew, done, ach = s.step(np.array([[PRIM[prim], 0, 0, 0]], np.int32))
        return vec[0], bool(done[0])
    s.set_pose(0, base, 0.0)
    img, vec, rew, done, ach = s.step(np.zeros((1, 4), np.int32))
    return vec[0], bool(done[0])


def test_obs_slots_order_and_padding():
    s = sim(seed=2)
    s.reset_env(0, 2)
    vec, _, _ = set_target_task(s, "nether_build_portal")
    st = s.state(0)
    slots = decode_slots(vec, st)
    dists = [math.dist((c[0] + 0.5, c[1] + 0.5, c[2] + 0.5), (st["x"], st["y"], st["z"])) for c, _ in slots]
    assert dists == sorted(dists), "slots must be nearest first"
    # After 8 cells are filled the next 6 fill the slots and the rest are zero padding.
    blob = s.save_state(0)
    s.load_state(0, patch_blob(blob, slots))
    img, vec2, rew, done, ach = s.step(np.zeros((1, 4), np.int32))
    left = decode_slots(vec2[0], s.state(0))
    assert len(left) == PORTAL_CELLS - 8 == 6, len(left)
    assert all(v == 0.0 for v in vec2[0][V7_BUILD + 4 * 6:V7_BUILD + 32]), "padding must be zero"
    assert all(ITEMS[item] == "obsidian" for _, item in left)


# ---------------------------------------------------------------- worldgen


def test_worldgen_deterministic():
    a, b = sim(seed=5), sim(seed=5)
    for seed in (11, 12, 13):
        a.reset_env(0, seed)
        b.reset_env(0, seed)
        assert a.save_state(0) == b.save_state(0), f"seed {seed}: worldgen not deterministic"
    a.reset_env(0, 11)
    b.reset_env(0, 12)
    assert a.save_state(0) != b.save_state(0)


def test_spawn_on_dry_grass_and_no_desert_in_spawn_box():
    """Spawn stands on dry grass above sea level, most of the 5x5 ring is grass; no desert (high sand) inside the
    box around the centre (worldgen keeps the spawn area plains)."""
    s = sim(seed=0)
    for seed in range(1, 31):
        s.reset_env(0, seed)
        st = s.state(0)
        sx, sz = int(st["x"]), int(st["z"])
        below = s.block(0, sx, int(st["y"]) - 1, sz)
        assert below == "grass", (seed, below)
        assert st["y"] > SEA_LEVEL, (seed, st["y"])
        w = world_array(s)
        for dz in range(-2, 3):
            for dx in range(-2, 3):
                col = [y for y in np.nonzero(w[:, sz + dz, sx + dx])[0] if w[y, sz + dz, sx + dx] not in (LOG, LEAVES)]
                top, ty = w[col[-1], sz + dz, sx + dx], col[-1]
                # the core may put a beach (sand at sea level) or a lake next to spawn; no raised sand
                assert top in (GRASS, SAND, WATER, BLOCKS.index("dirt"), BLOCKS.index("stone")), (seed, top)
                assert top != SAND or ty <= SEA_LEVEL + 3, (seed, dx, dz, ty)
        box = w[:, 64 - 26:64 + 27, 64 - 26:64 + 27]
        high_sand = 0
        for z in range(box.shape[1]):
            for x in range(box.shape[2]):
                col = np.nonzero(box[:, z, x])[0]
                if len(col) and box[col[-1], z, x] == SAND and col[-1] > SEA_LEVEL + 3:
                    high_sand += 1
        assert high_sand == 0, (seed, high_sand)


def surface_stats(w):
    solid = (w != AIR) & (w != LOG) & (w != LEAVES)
    top = np.where(solid.any(axis=0), WY - 1 - np.argmax(solid[::-1], axis=0), -1)
    tb = np.take_along_axis(w, np.maximum(top, 0)[None], axis=0)[0]
    desert = int(((top > SEA_LEVEL + 3) & (tb == SAND)).sum())
    logs = int((w == LOG).sum())
    return desert, logs


def test_biome_variety_across_seeds():
    s = sim(seed=0)
    deserts, logs = [], []
    for seed in range(1, 41):
        s.reset_env(0, seed)
        d, l = surface_stats(world_array(s))
        deserts.append(d)
        logs.append(l)
    n_desert = sum(1 for d in deserts if d > 300)
    n_plain = sum(1 for d in deserts if d < 300)
    print(f"  desert-like worlds {n_desert}/40 (sand tops above sea+3 >300), logs min {min(logs)} max {max(logs)}")
    assert n_desert >= 10, n_desert
    assert n_plain >= 3, n_plain
    assert max(logs) - min(logs) > 800, (min(logs), max(logs))


def test_exposed_coal_in_cave_walls_and_ore_bottleneck():
    s = sim(seed=0)
    exposed, coal = [], []
    for seed in range(1, 21):
        s.reset_env(0, seed)
        w = world_array(s)
        air_n = np.zeros(w.shape, dtype=bool)
        air_n[1:] |= w[:-1] == AIR
        air_n[:-1] |= w[1:] == AIR
        air_n[:, 1:] |= w[:, :-1] == AIR
        air_n[:, :-1] |= w[:, 1:] == AIR
        air_n[:, :, 1:] |= w[:, :, :-1] == AIR
        air_n[:, :, :-1] |= w[:, :, 1:] == AIR
        exposed.append(int(((w == COAL) & air_n)[20:].sum()))
        coal.append(int((w == COAL).sum()))
    print(f"  coal total avg {np.mean(coal):.0f} (baseline core ~946), exposed coal y>=20 avg {np.mean(exposed):.1f}")
    assert np.mean(exposed) >= 30, np.mean(exposed)
    assert np.mean(coal) >= 800, np.mean(coal)


def test_villages_chests_and_boulders():
    s = sim(seed=0)
    worlds_with_chest = 0
    worlds_with_boulders = 0
    for seed in range(1, 41):
        s.reset_env(0, seed)
        chests = s.chests(0)
        if any(c["items"] for c in chests):
            worlds_with_chest += 1
        w = world_array(s)
        if int((w == BLOCKS.index("cobblestone")).sum()) > 20:
            worlds_with_boulders += 1
    print(f"  worlds with a chest {worlds_with_chest}/40, with cobblestone boulders/foundations {worlds_with_boulders}/40")
    assert worlds_with_chest >= 8, worlds_with_chest
    assert worlds_with_boulders >= 30, worlds_with_boulders


def test_worldgen_time():
    n = 64
    s = CraftSim(action_mode="v7", num_envs=n, agents=1, seed=7, res=8, max_steps=3000, ticks=1, threads=1)
    s.reset()
    t0 = time.perf_counter()
    for w in range(n):
        s.reset_env(w, 1000 + w)
    per = (time.perf_counter() - t0) / n * 1000
    print(f"  reset_env incl. core terrain + python overhead: {per:.1f} ms per world")
    assert per < 60.0, per


TESTS = [
    test_target_placed_for_build_task,
    test_obs_slots_non_build_task_zero,
    test_save_load_keeps_target,
    test_build_complete_fires_with_full_build,
    test_progress_exact_with_placed_blocks,
    test_obs_slots_order_and_padding,
    test_worldgen_deterministic,
    test_spawn_on_dry_grass_and_no_desert_in_spawn_box,
    test_biome_variety_across_seeds,
    test_exposed_coal_in_cave_walls_and_ore_bottleneck,
    test_villages_chests_and_boulders,
    test_worldgen_time,
]

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        print(fn.__name__)
        try:
            fn()
            print("  PASS")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e!r}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
