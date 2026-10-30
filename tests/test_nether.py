"""Nether tests (SIM-3): portal lighting, travel after standing in a lit portal, team moves, dimension swap of
blocks and chests, in_dimension predicate, return trip, nether block content, generation time.

Run: .venv/bin/python tests/test_nether.py   (after ./build.sh)

Scenes are written through save_state/load_state blobs (the Python API has no set-block or set-position call).
The blob layout is parsed here (see saveState in src/sim.cpp). load_state replaces the whole world, including
nether::State, so every scene is patched first and the portal is lit afterwards.
"""
import math
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from craftsim import BLOCKS, ITEMS, NUM_ITEMS, PRIMARY_NAMES, CraftSim  # noqa: E402

WX, WY, WZ = 128, 64, 128
HDR = 16          # magic (4) + version (4) + block count (8)
K = 2             # agents per world
PRIM = {n: i for i, n in enumerate(PRIMARY_NAMES)}
N_WORLDS = 4      # 0: valid frame, 1: invalid frame, 2: travel + return + chests, 3: in_dimension task


def cell(x, y, z):
    return (y * WZ + z) * WX + x


def layout(blob):
    """Offsets in a save_state blob: chest section (nc_off, chests) and the agent section."""
    off = HDR + WX * WY * WZ + 41          # seed, goal, got, t, episode, restart, task, taskP0, taskP1, maxT
    for _ in range(2):                     # rng streams: uint32 length + text
        (n,) = struct.unpack_from("<I", blob, off)
        off += 4 + n
    nc_off = off
    (nc,) = struct.unpack_from("<I", blob, off)
    off += 4
    chests = []
    for _ in range(nc):
        (key,) = struct.unpack_from("<i", blob, off)
        counts = struct.unpack_from("<%di" % NUM_ITEMS, blob, off + 4)
        chests.append((key, list(counts)))
        off += 4 + 4 * NUM_ITEMS + 1       # key, counts, deposit mask
    k_off = off
    (k,) = struct.unpack_from("<I", blob, k_off)
    assert k == K, k
    agents = k_off + 4
    return {"nc_off": nc_off, "k_off": k_off, "agents": agents, "size": (len(blob) - agents) // K, "chests": chests}


def patch(blob, blocks=None, pos=None, chests=None):
    """blocks: {(x, y, z): block name}; pos: {agent index in world: (x, y, z) feet block}; chests: {(x, y, z):
    {item name: count}} replaces the chest list of the world."""
    b = bytes(blob)
    if chests is not None:
        lay = layout(b)
        body = struct.pack("<I", len(chests))
        for (x, y, z), items in chests.items():
            counts = [0] * NUM_ITEMS
            for name, n in items.items():
                counts[ITEMS.index(name)] = n
            body += struct.pack("<i", cell(x, y, z)) + struct.pack("<%di" % NUM_ITEMS, *counts) + bytes([0])
        b = b[: lay["nc_off"]] + body + b[lay["k_off"]:]
    lay = layout(b)
    ba = bytearray(b)
    for (x, y, z), name in (blocks or {}).items():
        ba[HDR + cell(x, y, z)] = BLOCKS.index(name)
    for k, (x, y, z) in (pos or {}).items():
        struct.pack_into("<ddd", ba, lay["agents"] + k * lay["size"], x + 0.5, y, z + 0.5)
    return bytes(ba)


def env():
    s = CraftSim(action_mode="v7", num_envs=N_WORLDS, agents=K, seed=11, res=8, max_steps=3000, ticks=1, threads=1)
    s.reset()
    return s


def step(s, spec):
    """spec: {agent index: (primary name, item name or None)}; other agents noop. Returns done per agent."""
    a = np.zeros((s.n, 4), np.int32)
    for i, (prim, item) in spec.items():
        a[i, 0] = PRIM[prim]
        if item:
            a[i, 2] = ITEMS.index(item) + 1
    return s.step(a)[3]


def dim(s, i):
    return s.state(i)["dim"]


def inside_portal(st):
    """Scene portal (see build_scene): feet in the frame plane z = 50 for the overworld, z = 6 for the nether."""
    return math.floor(st["x"]) in (50, 51) and math.floor(st["z"]) == 50 and math.floor(st["y"]) == 20


def build_scene(s, w, valid=True, chest=False):
    """Fresh world w with a flat stone floor, a clear box and an obsidian 2x3 frame in the plane z = 50
    (interior x 50..51, y 20..22). Agent 0 stands in the interior facing +x (it can light the frame), agent 1
    stands outside. valid=False removes one frame cell."""
    s.reset_env(w * K, 1000 + w)
    blob = s.save_state(w)
    blocks = {}
    for x in range(40, 71):
        for z in range(40, 61):
            blocks[(x, 19, z)] = "stone"
            for y in range(20, 31):
                blocks[(x, y, z)] = "air"
    for dx in range(-1, 3):
        blocks[(50 + dx, 19, 50)] = "obsidian"
        blocks[(50 + dx, 23, 50)] = "obsidian"
    for dy in range(3):
        blocks[(49, 20 + dy, 50)] = "obsidian"
        blocks[(52, 20 + dy, 50)] = "obsidian"
    if not valid:
        blocks[(50, 23, 50)] = "air"
    chests = None
    if chest:
        blocks[(44, 20, 44)] = "chest"
        chests = {(44, 20, 44): {"cobblestone": 7}}
    blob = patch(blob, blocks=blocks, pos={0: (50, 20, 50), 1: (46, 20, 46)}, chests=chests)
    s.load_state(w, blob)
    s.set_inventory(w * K, "flint_and_steel", 1)
    s.set_pose(w * K, -90, 0)        # yaw -90: looks along +x at the frame's right column
    s.set_pose(w * K + 1, 0, 0)


def travel_to_nether(s, w, task, check_each=True):
    """Lights the frame (decision 1) and stands in it. Returns the step times of decisions 2..20."""
    s.set_task(w, task)
    step(s, {w * K: ("use", "flint_and_steel")})
    times = []
    for n in range(2, 21):
        t0 = time.perf_counter()
        done = step(s, {})
        times.append(time.perf_counter() - t0)
        if n < 20 and check_each:
            assert dim(s, w * K) == 0 and dim(s, w * K + 1) == 0, f"team moved too early (decision {n})"
        if n == 20:
            return times, done
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- tests

def test_valid_frame_lights():
    s = env()
    build_scene(s, 0, valid=True)
    s.set_task(0, "nether_light_portal")
    done = step(s, {0: ("use", "flint_and_steel")})
    assert done[0], "a valid obsidian frame with flint and steel did not light (portal_lit not reached)"


def test_invalid_frame_does_not_light():
    s = env()
    w = 1
    build_scene(s, w, valid=False)
    s.set_task(w, "nether_light_portal")
    for _ in range(3):
        done = step(s, {w * K: ("use", "flint_and_steel")})
        assert not done[w * K], "an incomplete frame lit a portal"
    assert dim(s, w * K) == 0


def test_travel_after_20_decisions_moves_team():
    s = env()
    w = 2
    build_scene(s, w, valid=True, chest=True)
    times, done = travel_to_nether(s, w, "nether_kill_piglin")
    assert dim(s, w * K) == 1 and dim(s, w * K + 1) == 1, "team did not move together to the nether"
    assert not done[w * K], "nether_kill_piglin should not succeed on arrival"
    # Arrival cells: feet on the nether portal's front pad (created at 50 // 8 = 6, 6 with 1:8 scaling).
    st0, st1 = s.state(w * K), s.state(w * K + 1)
    assert math.floor(st0["x"]) == 6 and math.floor(st0["z"]) == 7, (st0["x"], st0["z"])
    assert math.floor(st1["x"]) == 7 and math.floor(st1["z"]) == 7, (st1["x"], st1["z"])
    assert s.block(w * K, math.floor(st0["x"]), math.floor(st0["y"]) - 1, math.floor(st0["z"])) == "netherrack"
    print(f"    travel decision time {times[-1] * 1000:.1f} ms, median other decision {np.median(times[:-1]) * 1000:.2f} ms")


def test_in_dimension_predicate_true_on_arrival():
    s = env()
    w = 3
    build_scene(s, w, valid=True)
    s.set_task(w, "nether_enter")
    step(s, {w * K: ("use", "flint_and_steel")})
    for n in range(2, 21):
        done = step(s, {})
        if n < 20:
            assert not done[w * K], f"in_dimension succeeded before the team travelled (decision {n})"
    # Success ends the episode (auto-reset), so the predicate result is read from done.
    assert done[w * K] and done[w * K + 1], "in_dimension {nether} was not true for the team after travel"


def test_nether_blocks_present_and_no_water():
    s = env()
    w = 2
    build_scene(s, w, valid=True, chest=True)
    travel_to_nether(s, w, "nether_kill_piglin", check_each=False)
    origin, arr = s.region(w * K, radius=64)
    names = {BLOCKS[int(b)] for b in np.unique(arr)}
    for want in ("netherrack", "lava", "bedrock", "glowstone", "nether_quartz_ore", "soul_sand", "obsidian"):
        assert want in names, f"nether region has no {want}: {sorted(names)}"
    assert "water" not in names and "grass" not in names and "stone" not in names, sorted(names)
    x, z = 6, 7
    assert s.block(w * K, 40, 0, 40) == "bedrock" and s.block(w * K, 40, 63, 40) == "bedrock"
    assert s.block(w * K, 40, 5, 40) == "lava", "no lava sea at y = 5"


def test_chests_do_not_leak_between_dimensions():
    s = env()
    w = 2
    build_scene(s, w, valid=True, chest=True)
    before = s.chests(w)
    assert len(before) == 1 and before[0]["items"] == {"cobblestone": 7}, before
    travel_to_nether(s, w, "nether_kill_piglin", check_each=False)
    assert s.chests(w) == [], f"overworld chest visible in the nether: {s.chests(w)}"
    # Return trip (below) brings the overworld chest back with its contents.
    walk_back(s, w)
    after = s.chests(w)
    assert len(after) == 1 and after[0]["items"] == {"cobblestone": 7}, after


def walk_back(s, w):
    """Walks the team's first agent from the nether front pad into the nether portal (plane z = 6), then stands
    still. The team must travel on the 20th consecutive decision inside the portal and land at the overworld
    portal (interior x 50..51, z 50)."""
    i0 = w * K
    st = s.state(i0)
    assert math.floor(st["x"]) == 6 and math.floor(st["z"]) == 7, (st["x"], st["z"])
    s.set_pose(i0, 180, 0)           # yaw 180 looks along -z, into the frame plane
    count, walking = 0, True
    for _ in range(80):
        step(s, {i0: ("forward" if walking else "noop", None)})
        st = s.state(i0)
        if st["dim"] == 0:
            assert count == 19, f"team left the nether after {count + 1} portal decisions, expected 20"
            break
        inside = math.floor(st["z"]) == 6 and math.floor(st["x"]) in (6, 7)
        count = count + 1 if inside else 0
        if inside:
            walking = False
    else:
        raise AssertionError("walk into the nether portal did not travel")
    assert dim(s, w * K) == 0 and dim(s, w * K + 1) == 0, "team did not return together"
    st0 = s.state(i0)
    assert 49 <= math.floor(st0["x"]) <= 52 and 49 <= math.floor(st0["z"]) <= 53 and math.floor(st0["y"]) == 20, \
        f"return trip did not land at the overworld portal: {st0['x']}, {st0['y']}, {st0['z']}"


def test_return_trip_works():
    s = env()
    w = 2
    build_scene(s, w, valid=True, chest=True)
    travel_to_nether(s, w, "nether_kill_piglin", check_each=False)
    walk_back(s, w)


TESTS = [
    test_valid_frame_lights,
    test_invalid_frame_does_not_light,
    test_travel_after_20_decisions_moves_team,
    test_in_dimension_predicate_true_on_arrival,
    test_nether_blocks_present_and_no_water,
    test_chests_do_not_leak_between_dimensions,
    test_return_trip_works,
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
