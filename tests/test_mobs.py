"""Mob tests (SIM-1): passive spawning, obs slots, pseudo ids in the image, attacks, drops, hostile AI, determinism.

Run: .venv/bin/python tests/test_mobs.py   (after ./build.sh)
Needs the core hook `mobs::tick(W, ag, K)` in sim.cpp stepWorld (see the report). Hostile tests (zombie,
creeper) need night or low light from survival (SIM-2); while survival reports constant daylight they are
skipped and printed as SKIP.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from craftsim import (  # noqa: E402
    ITEMS, MOB_NAMES, MOBS, NUM_MOBS, PRIMARY_NAMES, V7_MOBS, VEC_DIM, VEC_NAMES, CraftSim,
)

P = {n: i for i, n in enumerate(PRIMARY_NAMES)}
PSEUDO = {m["name"]: m["block"] for m in MOBS}
MOB_IDS = set(PSEUDO.values())
COW, COW_ID = "cow", PSEUDO["cow"]
BEEF = ITEMS.index("beef")
NOOP = np.array([P["noop"], 0, 0, 0], np.int32)
FWD = np.array([P["forward"], 0, 0, 0], np.int32)
ATTACK = np.array([P["attack"], 0, 0, 0], np.int32)
CROSS_REACH = 2.5          # aim-and-attack distance (blocks; the sim's attack reach is 3.0)


def env(n=1, k=1, seed=3, res=8, threads=1):
    return CraftSim(num_envs=n, agents=k, seed=seed, res=res, max_steps=3000, ticks=4, threads=threads)


def noop_for(s):
    return np.tile(NOOP, (s.n, 1))


def slots(vec, i=0):
    """The 3 nearest mobs of agent i: dicts with present, xl/zl (yaw frame, blocks), dy (world, blocks),
    hostile, hp fraction."""
    out = []
    for j in range(3):
        b = V7_MOBS + 6 * j
        v = vec[i, b:b + 6]
        out.append({"present": v[0] > 0.5, "xl": float(v[1] * 16), "dy": float(v[2] * 16),
                    "zl": float(v[3] * 16), "hostile": v[4] > 0.5, "hp": float(v[5])})
    return out


def to_world(st, sl):
    """Slot offset (yaw frame) -> world offset (dx, dy, dz) from the agent's feet."""
    yr = math.radians(st["yaw"])
    rx, rz = -math.cos(yr), -math.sin(yr)
    fx, fz = -math.sin(yr), math.cos(yr)
    dx = sl["xl"] * rx + sl["zl"] * fx
    dz = sl["xl"] * rz + sl["zl"] * fz
    return dx, sl["dy"], dz


def aim_pose(st, sl):
    """Yaw/pitch that look at the mob's body centre (slot dy is relative to the body centre at feet+0.9)."""
    dx, dy, dz = to_world(st, sl)
    yaw = math.degrees(math.atan2(-dx, dz))
    dy_eye = dy - (1.62 - 0.9)          # centre relative to the eye
    pitch = math.degrees(math.atan2(-dy_eye, math.hypot(dx, dz)))
    return yaw, pitch


def dist_slot(sl):
    return math.sqrt(sl["xl"] ** 2 + sl["dy"] ** 2 + sl["zl"] ** 2)


def image_has(img, i, block_id):
    return bool(np.any(img[i, :, :, 0] == block_id))


def mob_ids_in(img, i=0):
    ids = np.unique(img[i, :, :, 0])
    return [int(x) for x in ids if int(x) in MOB_IDS]


def test_shapes_and_obs_layout():
    assert NUM_MOBS == 9 and len(MOB_NAMES) == 9
    names = VEC_NAMES[V7_MOBS:V7_MOBS + 18]
    assert len(names) == 18 and len(VEC_NAMES) == VEC_DIM
    s = env()
    _, vec = s.reset()
    assert vec.shape == (1, VEC_DIM)
    assert np.all(vec[0, V7_MOBS:V7_MOBS + 18] == 0.0), "no mob slots before the first step"
    print("  shapes ok:", VEC_DIM, "vec dim, mob slots at", V7_MOBS)


def test_passive_spawn_by_day_near_agent():
    s = env(seed=3)
    _, vec = s.reset()
    for _ in range(5):
        _, vec, *_ = s.step(noop_for(s))
    sl = slots(vec)
    assert sl[0]["present"], "passive animals spawn at the first step near the agent"
    assert not sl[0]["hostile"], "daylight spawns are passive"
    assert dist_slot(sl[0]) <= 20.5, "spawned within the 6..14 block ring (plus wander)"
    hostile_seen = False
    for _ in range(300):
        _, vec, *_ = s.step(noop_for(s))
        hostile_seen = hostile_seen or any(x["present"] and x["hostile"] for x in slots(vec))
    if hostile_seen:
        print("  note: hostile mobs spawned in daylight (survival light/time now live?)")
    print("  passive spawn near agent ok; nearest at %.1f blocks" % dist_slot(sl[0]))


def find_cow_world(max_seeds=40):
    """A seed whose nearest mob is a cow (its pseudo id 24 is in the image when aimed at it)."""
    for seed in range(max_seeds):
        s = env(seed=seed)
        s.reset()
        img, vec = s.step(noop_for(s))[:2]
        sl = slots(vec)
        if not sl[0]["present"] or dist_slot(sl[0]) > 10:
            continue
        st = s.state(0)
        s.set_pose(0, *aim_pose(st, sl[0]))
        img, vec = s.step(noop_for(s))[:2]
        if image_has(img, 0, COW_ID):
            return seed
    raise AssertionError("no cow in view in %d seeds" % max_seeds)


def test_mob_pseudo_id_visible_in_image():
    seed = find_cow_world()
    s = env(seed=seed)
    s.reset()
    img, vec = s.step(noop_for(s))[:2]
    st = s.state(0)
    sl = slots(vec)[0]
    s.set_pose(0, *aim_pose(st, sl))
    img, vec = s.step(noop_for(s))[:2]
    ids = mob_ids_in(img)
    assert COW_ID in ids, "aimed at a cow: the image shows pseudo id %d (got %s)" % (COW_ID, ids)
    print("  cow pseudo id %d visible in the image (seed %d)" % (COW_ID, seed))


def kill_cow(weapon, max_decisions=400):
    """Walk up to the cow (nearest mob, aimed from the image), attack it until it dies. Returns (hits, beef)."""
    seed = find_cow_world()
    s = env(seed=seed)
    s.reset()
    if weapon:
        s.set_inventory(0, weapon, 1)
    beef0 = s.inventory(0).get("beef", 0)
    img, vec = s.step(noop_for(s))[:2]
    hits = 0
    for _ in range(max_decisions):
        sl = slots(vec)[0]
        if not sl["present"]:
            break
        st = s.state(0)
        s.set_pose(0, *aim_pose(st, sl))
        in_view = image_has(img, 0, COW_ID)
        if in_view and dist_slot(sl) <= CROSS_REACH:
            hits += 1
            img, vec, *_ = s.step(np.tile(ATTACK, (s.n, 1)))
        else:
            img, vec, *_ = s.step(np.tile(FWD, (s.n, 1)))
        beef = s.inventory(0).get("beef", 0)
        if beef > beef0:
            return hits, beef - beef0
    return hits, 0


def test_attack_kills_cow_in_n_hits_and_gives_beef():
    hits, beef = kill_cow(None)
    assert beef >= 1, "the cow was killed and dropped beef (hits %d)" % hits
    assert hits == 10, "cow has 10 health, fists do 1 damage: expected 10 hits, got %d" % hits
    hits_s, beef_s = kill_cow("wooden_sword")
    assert beef_s >= 1 and hits_s == 3, "wooden sword (4 damage): 3 hits, got %d" % hits_s
    print("  cow: %d fist hits -> %d beef; wooden sword: %d hits -> %d beef" % (hits, beef, hits_s, beef_s))


def test_obs_three_nearest_sorted():
    s = env(n=2, k=2, seed=11)
    _, vec = s.reset()
    for _ in range(40):
        _, vec, *_ = s.step(np.tile(NOOP, (s.n, 1)))
    checked = 0
    for i in range(s.n):
        d = [dist_slot(x) for x in slots(vec, i) if x["present"]]
        assert d == sorted(d), "slots are sorted nearest first: %s" % d
        checked += len(d)
    assert checked > 0
    print("  obs slots sorted by distance (%d present slots checked)" % checked)


def test_determinism_same_seed_same_obs():
    def run():
        s = CraftSim(num_envs=4, agents=2, seed=21, res=8, max_steps=3000, ticks=4, threads=1)
        img, vec = s.reset()
        rng = np.random.default_rng(7)
        imgs, vecs = [img.copy()], [vec.copy()]
        for _ in range(300):
            a = np.zeros((s.n, 4), np.int32)
            a[:, 0] = rng.integers(0, 21, size=s.n)
            img, vec, *_ = s.step(a)
            imgs.append(img.copy())
            vecs.append(vec.copy())
        return np.stack(imgs), np.stack(vecs)
    i1, v1 = run()
    i2, v2 = run()
    assert np.array_equal(i1, i2) and np.array_equal(v1, v2), "same seed + actions -> identical obs"
    assert np.any(v1[:, :, V7_MOBS] > 0.5), "mobs were present during the run"
    print("  determinism ok (300 steps, 4 worlds x 2 agents, mobs present)")


def hostile_available(seeds=8, steps=300):
    for seed in range(seeds):
        s = env(seed=seed)
        s.reset()
        for _ in range(steps):
            _, vec, *_ = s.step(noop_for(s))
            if any(x["present"] and x["hostile"] for x in slots(vec)):
                return seed
    return None


def test_zombie_hits_idle_agent():
    # Mobs walk straight at the agent (no path finding), so some terrain traps them: try several worlds.
    if hostile_available() is None:
        print("  SKIP zombie damage: no hostile mobs spawn (survival light/time still neutral)")
        return
    for seed in range(12):
        s = env(seed=seed)
        s.reset()
        h0 = s.state(0)["health"]
        for _ in range(800):
            _, vec, *_ = s.step(noop_for(s))
            if s.state(0)["health"] < h0:
                print("  seed %d: idle agent took %d damage from a hostile mob" % (seed, h0 - s.state(0)["health"]))
                return
    raise AssertionError("no hostile mob damaged the idle agent in 12 worlds x 800 steps")


def solid_in_box(s, c, r=4):
    """Non-air blocks in the cube of half-size r around the fixed cell c (absolute coordinates)."""
    origin, arr = s.region(0, 6)
    ox, oy, oz = origin
    x0, y0, z0 = c
    sub = arr[max(0, y0 - r - oy):y0 + r + 1 - oy, max(0, z0 - r - oz):z0 + r + 1 - oz,
              max(0, x0 - r - ox):x0 + r + 1 - ox]
    return int(np.count_nonzero(sub))


def test_creeper_explosion_removes_blocks():
    # An idle agent loses blocks around it only to a creeper explosion (noop never mines).
    if hostile_available() is None:
        print("  SKIP creeper explosion: no hostile mobs spawn (survival light/time still neutral)")
        return
    for seed in range(40):
        s = env(seed=seed)
        s.reset()
        st = s.state(0)
        c = (math.floor(st["x"]), math.floor(st["y"]), math.floor(st["z"]))
        before = solid_in_box(s, c)
        h0 = st["health"]
        ep0 = st["episode"]
        for _ in range(800):
            s.step(noop_for(s))
            if s.state(0)["episode"] != ep0:   # the world auto-reset (new terrain): not an explosion
                break
            now = solid_in_box(s, c)
            if now < before:
                print("  seed %d: solid blocks within 4 of the agent %d -> %d, health %s -> %s" % (
                    seed, before, now, h0, s.state(0)["health"]))
                return
    raise AssertionError("no block loss near an idle agent in 40 worlds x 800 steps")


def test_throughput_with_mobs():
    W, K = 32, 4
    s = CraftSim(num_envs=W, agents=K, seed=5, res=64, max_steps=3000, ticks=4, threads=0)
    img, vec = s.reset()
    act = np.zeros((s.n, 4), np.int32)
    act[:, 0] = P["forward"]
    s.step(act)
    t0 = time.perf_counter()
    steps = 60
    mobs_seen = []
    for _ in range(steps):
        img, vec, *_ = s.step(act)
        mobs_seen.append(int(np.sum(vec[:, V7_MOBS] > 0.5)))
    dt = time.perf_counter() - t0
    rate = steps * s.n / dt
    print("  throughput W=%d K=%d res=64: %.0f agent-steps/s (mobs in nearest slots: %.1f per step)" % (
        W, K, rate, float(np.mean(mobs_seen))))


def core_hook_wired():
    """mobs::tick needs the agents (core hook). Without it no mob ever spawns."""
    s = env(seed=3)
    s.reset()
    for _ in range(5):
        _, vec, *_ = s.step(noop_for(s))
    return bool(np.any(vec[:, V7_MOBS] > 0.5))


def main():
    if not core_hook_wired():
        print("FAIL: no mobs spawn. sim.cpp stepWorld must call `mobs::tick(W, ag, K);` (currently `mobs::tick(W);`) and "
              "writeObs must call `mobs::beginView(W, a);` before its pixel loop. See the mobs report.")
        return 1
    tests = [
        test_shapes_and_obs_layout,
        test_passive_spawn_by_day_near_agent,
        test_mob_pseudo_id_visible_in_image,
        test_attack_kills_cow_in_n_hits_and_gives_beef,
        test_obs_three_nearest_sorted,
        test_determinism_same_seed_same_obs,
        test_zombie_hits_idle_agent,
        test_creeper_explosion_removes_blocks,
        test_throughput_with_mobs,
    ]
    failed = 0
    for t in tests:
        try:
            print(t.__name__)
            t()
            print("  PASS")
        except AssertionError as e:
            failed += 1
            print("  FAIL:", e)
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
