"""Sim tests: physics, mining, crafting, regions, determinism, throughput.

Run: .venv/bin/python tests/test_sim.py   (after ./build.sh)
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from craftsim import (  # noqa: E402
    FAMILIES, ITEMS_V6, BLOCKS_V6, MOB_NAMES, NUM_BLOCKS, NUM_FAMILIES, NUM_MOBS, NUM_PRIMARY, NUM_RECIPES,
    NUM_TASKS, PRIMARY_NAMES, RECIPE_NAMES, RECIPES, TASK_NAMES, TASKS, VEC_DIM_V6, NUM_ITEMS, NUM_MSGS,
    V7_BUILD, V7_TASK, V7_TASK_PARAM, V7_MOBS, _TASK_TYPE_NAMES,
    ACTIONS, AGENT_BLOCK, BLOCKS, BLOCKS_V6, CHEST_OFFSET, GOAL_OFFSET, ITEMS, ITEMS_V6, NUM_ACTIONS, SEA_LEVEL,
    TEAM_OFFSET, VEC_DIM, VEC_DIM_V6, VEC_NAMES, VEC_NAMES_V6, WORLD_SIZE, XHAIR_OFFSET, CraftSim,
)

A = {name: i for i, name in enumerate(ACTIONS)}
SEED = 0


def sim(n=1, seed=SEED, max_steps=3000, ticks=1):
    return CraftSim(action_mode="v6", num_envs=n, seed=seed, res=8, max_steps=max_steps, ticks=ticks, threads=1)


def reset_env(s, i, seed):
    """reset_env + pin the goal to diamond (never obtained by these tests, so goal-achievement
    cannot end an episode mid-test). Goal behaviour is tested separately."""
    s.reset_env(i, seed)
    s.set_goal(i, "diamond")


def feet(st):
    return math.floor(st["x"]), math.floor(st["y"]), math.floor(st["z"])


def under(s, i=0):
    fx, fy, fz = feet(s.state(i))
    return s.block(i, fx, fy - 1, fz)


def act(s, name, i=0):
    """Step env i with `name`; other envs noop. Returns done[i]."""
    a = np.zeros(s.n, np.int32)
    a[i] = A[name]
    return bool(s.step(a)[3][i])


def settle(s, i=0, n=40):
    for _ in range(n):
        act(s, "noop", i)
    assert s.state(i)["on_ground"] == 1.0, "player did not settle on ground"


def land(s, i=0, n=60):
    """Noop until the player is on the ground (mining starts on the ground, as in the on-ground numbers).
    At least one tick is always simulated: on_ground is only refreshed by a physics tick, so right
    after breaking the block under us it still reads True for one tick."""
    for _ in range(n):
        act(s, "noop", i)
        if s.state(i)["on_ground"] == 1.0:
            return
    raise AssertionError("player never landed")


WATER = BLOCKS.index("water")
GRASS_NAME = BLOCKS[1]   # registry name of grass (DATA-1 may rename it)


def test_shapes_and_constants():
    assert WORLD_SIZE == (128, 64, 128)
    assert SEA_LEVEL == 30
    assert NUM_ACTIONS == len(ACTIONS) == 28 and len(ITEMS_V6) == 15 and NUM_MSGS == 8
    assert VEC_DIM_V6 == 102 and GOAL_OFFSET == 26 and TEAM_OFFSET == 41 and CHEST_OFFSET == 89
    assert XHAIR_OFFSET == 93 and XHAIR_OFFSET + 9 == VEC_DIM_V6
    assert len(BLOCKS_V6) == 20 and BLOCKS_V6[14:] == ["sand", "gravel", "water", "deepslate", "chest", "agent"]
    assert ACTIONS[23:] == ["craft_chest", "place_chest", "deposit", "withdraw", "give"]
    assert ITEMS[14] == "chest" and AGENT_BLOCK == 19
    s = CraftSim(action_mode="v6", num_envs=3, seed=1, res=8, threads=1)
    assert s.num_agents == 3 and s.agents == 1
    img, vec = s.reset()
    assert img.shape == (3, 8, 8, 2) and img.dtype == np.uint8
    assert vec.shape == (3, VEC_DIM_V6) and vec.dtype == np.float32
    img, vec, rew, done, ach = s.step(np.zeros(3, np.int32))
    assert img.shape == (3, 8, 8, 2) and rew.shape == (3,) and done.dtype == bool and ach.dtype == np.uint32
    img, vec, rew, done, ach = s.step(np.zeros(3, np.int32), np.array([1, 2, 3], np.int32))
    assert list(s.last_msgs()) == [1, 2, 3]


def test_spawn_on_grass_seeds_0_19():
    s = sim(1)
    for seed in range(20):
        reset_env(s, 0, seed)
        st = s.state(0)
        assert under(s) == GRASS_NAME, f"seed {seed}: feet on {under(s)}"
        fx, fy, fz = feet(st)
        assert all(s.block(0, fx, y, fz) == "air" for y in range(fy, WORLD_SIZE[1])), \
            f"seed {seed}: canopy overhead"
        assert abs(st["y"] - (fy)) < 1e-9, f"seed {seed}: feet not at h+1"


def test_jump_apex():
    s = sim(1)
    reset_env(s, 0, SEED)
    settle(s)
    y0 = s.state(0)["y"]
    top = y0
    act(s, "jump")
    top = max(top, s.state(0)["y"])
    for _ in range(20):
        act(s, "noop")
        top = max(top, s.state(0)["y"])
    apex = top - y0
    print(f"  jump apex (ticks=1): {apex:.4f} blocks")
    assert abs(apex - 1.25) < 0.02, apex


def test_walk_speed():
    # Best unobstructed walk over yaws, on a few spawns (the spawn terrain differs per world seed).
    s = sim(1)
    best = (0.0, None)
    for seed in range(SEED, SEED + 8):
        for yaw in range(0, 360, 15):
            reset_env(s, 0, seed)
            s.set_pose(0, float(yaw), 0.0)
            settle(s, n=10)
            for _ in range(40):
                act(s, "forward")
            a = s.state(0)
            for _ in range(20):
                act(s, "forward")
            b = s.state(0)
            speed = math.hypot(b["x"] - a["x"], b["z"] - a["z"]) * 20.0 / 20.0  # blocks/20 ticks = m/s
            if speed > best[0]:
                best = (speed, yaw)
    print(f"  walking speed, best unobstructed yaw {best[1]}: {best[0]:.4f} m/s (target 4.317)")
    assert abs(best[0] - 4.317) < 0.05, best


def test_break_grass_by_hand():
    s = sim(1)
    reset_env(s, 0, SEED)
    settle(s)
    s.set_pose(0, 0.0, 90.0)  # look straight down
    assert under(s) == GRASS_NAME
    t0 = s.state(0)["ticks"]
    act(s, "attack")  # one attack decision holds the button until the block breaks
    ticks = s.state(0)["ticks"] - t0
    print(f"  grass by hand: {ticks} ticks in one attack decision, inventory {s.inventory(0)}")
    assert ticks == 18, ticks
    assert s.inventory(0) == {"dirt": 1}, s.inventory(0)


def test_stone_by_hand_gives_nothing():
    s = sim(1)
    reset_env(s, 0, SEED)
    settle(s)
    s.set_pose(0, 0.0, 90.0)
    for _ in range(2000):
        t = s.state(0)["target"]
        if t and t["block"] == "stone":
            break
        land(s)
        act(s, "attack")
    t = s.state(0)["target"]
    assert t and t["block"] == "stone", "no stone reached below spawn"
    before = s.inventory(0)
    tx, ty, tz = t["x"], t["y"], t["z"]
    land(s)   # breaking the block under us drops us; the timed mining starts on the ground
    t0, n = s.state(0)["ticks"], 0
    while s.block(0, tx, ty, tz) != "air":
        act(s, "attack")
        n += 1
        assert n < 20, "stone never broke"
    ticks = s.state(0)["ticks"] - t0
    after = s.inventory(0)
    print(f"  stone by hand: broke in {ticks} ticks, inventory before {before} after {after}")
    assert after == before, (before, after)
    assert "cobblestone" not in after
    assert ticks == 150, ticks  # 1.5 hardness, wrong tool: 1/1.5/100 per tick


def test_crafting_chain_and_table_proximity():
    s = sim(2)
    reset_env(s, 0, SEED)
    reset_env(s, 1, SEED)
    settle(s, 0)
    s.set_inventory(0, "oak_log", 3)
    for _ in range(3):
        act(s, "craft_planks")
    assert s.inventory(0) == {"oak_planks": 12}, s.inventory(0)
    act(s, "craft_sticks")
    assert s.inventory(0) == {"oak_planks": 10, "stick": 4}, s.inventory(0)
    act(s, "craft_table")
    assert s.inventory(0) == {"oak_planks": 6, "stick": 4, "crafting_table": 1}, s.inventory(0)

    # Place the table by looking ~45 degrees down, trying yaws until one placement succeeds.
    s.set_pose(0, 0.0, 45.0)
    placed = False
    for yaw in range(0, 360, 15):
        s.set_pose(0, float(yaw), 45.0)
        act(s, "place_table")
        if "crafting_table" not in s.inventory(0):
            placed = True
            break
    assert placed, "could not place table"
    t = s.state(0)["target"]
    assert t and t["block"] == "crafting_table", t
    tx, ty, tz = t["x"], t["y"], t["z"]
    print(f"  table placed at {(tx, ty, tz)} with yaw {yaw}")

    # Near the table: craft works.
    s.set_pose(0, float(yaw), 0.0)
    act(s, "craft_wooden_pickaxe")
    inv = s.inventory(0)
    print(f"  near table, inventory after craft: {inv}")
    assert inv.get("wooden_pickaxe") == 1 and inv.get("oak_planks") == 3 and inv.get("stick") == 2, inv

    # Without a table (env 1, same seed, nothing placed): craft fails.
    s.set_inventory(1, "oak_planks", 3)
    s.set_inventory(1, "stick", 2)
    act(s, "craft_wooden_pickaxe", i=1)
    assert "wooden_pickaxe" not in s.inventory(1), s.inventory(1)

    # Walk away from the table in env 0 until out of crafting range (>3 blocks), then craft.
    s.set_inventory(0, "oak_planks", 3)
    s.set_inventory(0, "stick", 2)
    s.set_inventory(0, "wooden_pickaxe", 0)
    away = math.degrees(math.atan2(-(feet(s.state(0))[0] - tx), feet(s.state(0))[2] - tz))
    far = False
    for dyaw in (0, 15, -15, 30, -30, 45, -45, 60, -60):
        s.set_pose(0, away + dyaw, 0.0)
        for _ in range(120):
            fx, _, fz = feet(s.state(0))
            if abs(fx - tx) > 3 or abs(fz - tz) > 3:
                far = True
                break
            act(s, "forward")
        if far:
            break
    assert far, "could not walk out of table range"
    fx, _, fz = feet(s.state(0))
    print(f"  walked to {(fx, fz)}, table at {(tx, tz)}")
    act(s, "craft_wooden_pickaxe")
    assert "wooden_pickaxe" not in s.inventory(0), s.inventory(0)


def test_region_shape_and_bounds():
    s = sim(1)
    reset_env(s, 0, SEED)
    cx, cy, cz = feet(s.state(0))
    origin, arr = s.region(0, 16)
    ox, oy, oz = origin
    assert (ox, oy, oz) == (max(0, cx - 16), 0, max(0, cz - 16)), origin
    sx = min(WORLD_SIZE[0] - 1, cx + 16) - ox + 1
    sz = min(WORLD_SIZE[2] - 1, cz + 16) - oz + 1
    assert arr.shape == (WORLD_SIZE[1], sz, sx) and arr.dtype == np.uint8, arr.shape
    rng = np.random.default_rng(0)
    for _ in range(200):
        y = int(rng.integers(0, WORLD_SIZE[1]))
        z = int(rng.integers(0, sz))
        x = int(rng.integers(0, sx))
        assert BLOCKS[arr[y, z, x]] == s.block(0, ox + x, oy + y, oz + z)
    _, full = s.region(0, 100)
    assert full.shape == (WORLD_SIZE[1], WORLD_SIZE[2], WORLD_SIZE[0]), full.shape
    o2, small = s.region(0, 0)
    assert small.shape == (WORLD_SIZE[1], 1, 1) and o2 == (cx, 0, cz)
    print(f"  region(16) origin {origin} shape {arr.shape}; region(100) shape {full.shape}")


def test_reset_env_determinism():
    s = sim(2)
    reset_env(s, 0, 1234)
    reset_env(s, 1, 1234)
    o0, r0 = s.region(0, 16)
    o1, r1 = s.region(1, 16)
    assert o0 == o1 and np.array_equal(r0, r1)
    assert s.state(0)["seed"] == 1234 and s.state(0)["episode"] == 0
    assert s.state(0)["x"] == s.state(1)["x"] and s.state(0)["z"] == s.state(1)["z"]
    reset_env(s, 1, 999)
    _, r2 = s.region(1, 16)
    assert not np.array_equal(r0, r2)
    assert s.state(1)["seed"] == 999
    print("  reset_env(seed) gives identical regions for identical seeds")


def test_episode_counter_and_last_action():
    s = sim(1, max_steps=5)
    reset_env(s, 0, SEED)
    act(s, "attack")
    assert s.state(0)["last_action"] == A["attack"]
    assert s.state(0)["episode"] == 0 and s.state(0)["seed"] == SEED
    seed0 = s.state(0)["seed"]
    done = [act(s, "noop") for _ in range(4)]
    assert done == [False, False, False, True], done
    st = s.state(0)
    assert st["episode"] == 1 and st["seed"] != seed0 and st["t"] == 0 and st["last_action"] == 0
    print(f"  auto-reset: episode 1, new seed {st['seed']}")


def test_airborne_mining_5x_slower():
    # Separate worlds (act() steps every env of a sim). Env a: jump, then hold attack on the grass
    # under us. Env b: jump, then count the airborne ticks n. Mining is /5 while airborne, so the
    # break takes 18 ground-ticks worth of progress: T = n + (18 - n/5) = 18 + 0.8 n.
    a, b = sim(1), sim(1)
    for s in (a, b):
        reset_env(s, 0, SEED)
        settle(s)
        s.set_pose(0, 0.0, 90.0)
    assert under(a) == GRASS_NAME
    act(a, "jump")
    act(b, "jump")
    n_air = 0
    while True:
        act(b, "noop")
        if b.state(0)["on_ground"] == 1.0:
            break
        n_air += 1
        assert n_air < 40, "never landed"
    t0 = a.state(0)["ticks"]
    act(a, "attack")
    t_air = a.state(0)["ticks"] - t0
    print(f"  airborne mining: {n_air} airborne ticks, grass broke in {t_air} ticks (ground: 18)")
    assert a.inventory(0) == {"dirt": 1}, a.inventory(0)
    assert abs(t_air - (18 + 0.8 * n_air)) <= 2, (t_air, n_air)
    assert t_air > 18 + 4, t_air


def find_water_walk(s, i=0, radius=12):
    """Walk toward the nearest water cell until the feet are in water. Returns True on success."""
    fx, fy, fz = feet(s.state(i))
    (ox, oy, oz), arr = s.region(i, radius)
    ys, zs, xs = np.nonzero(arr == WATER)
    if len(xs) == 0:
        return False
    d2 = (xs + ox - fx) ** 2 + (zs + oz - fz) ** 2
    k = int(np.argmin(d2))
    tx, tz = xs[k] + ox + 0.5, zs[k] + oz + 0.5
    px, pz = s.state(i)["x"], s.state(i)["z"]
    yaw = math.degrees(math.atan2(-(tx - px), tz - pz))
    s.set_pose(i, yaw, 0.0)
    for _ in range(300):
        fx, fy, fz = feet(s.state(i))
        if s.block(i, fx, fy, fz) == "water":
            return True
        act(s, "forward", i)
    return False


def test_water_exists_most_seeds():
    have = 0
    for seed in range(20):
        s = sim(1)
        reset_env(s, 0, seed)
        _, arr = s.region(0, 100)
        if (arr == WATER).any():
            have += 1
    print(f"  seeds with water: {have}/20")
    assert have >= 16, have


def test_water_slows_and_buoys():
    walk_speed = 4.317 / 20.0   # blocks per tick on land
    for seed in range(20):
        s = sim(1)
        reset_env(s, 0, seed)
        settle(s)
        if not find_water_walk(s):
            continue
        # Still in the lake: settle the swim speed, then measure it over 20 ticks.
        for _ in range(30):
            act(s, "forward")
        a = s.state(0)
        for _ in range(20):
            act(s, "forward")
        b = s.state(0)
        if s.block(0, *feet(b)) != "water":
            continue
        swim = math.hypot(b["x"] - a["x"], b["z"] - a["z"]) / 20.0
        # Buoyancy: sinking in water tends to -0.02 blocks/tick, not free fall.
        for _ in range(40):
            act(s, "noop")
        st = s.state(0)
        if s.block(0, *feet(st)) != "water" or st["on_ground"] == 1.0:
            continue   # reached the lake floor: the sinking rate is not measurable here
        print(f"  seed {seed}: swim {swim:.4f} blocks/tick (land {walk_speed:.4f}), "
              f"sinking vy {st['vy']:.4f} after 40 ticks")
        assert swim < 0.75 * walk_speed, (swim, walk_speed)
        assert swim > 0.05, swim
        assert -0.03 < st["vy"] < -0.01, st["vy"]
        return
    raise AssertionError("could not walk into a deep enough lake on seeds 0..19")


def test_ore_height_distribution():
    # Vanilla ore ranges (OrePlacements.java) scaled to our 64-high world. Diamonds sit low
    # (deepslate), coal is higher, iron in between.
    ys = {"coal_ore": [], "iron_ore": [], "diamond_ore": []}
    for seed in range(20):
        s = sim(1)
        reset_env(s, 0, seed)
        (_, oy, _), arr = s.region(0, 100)
        for name in ys:
            yy, _, _ = np.nonzero(arr == BLOCKS.index(name))
            ys[name].extend((yy + oy).tolist())
    med = {k: float(np.median(v)) for k, v in ys.items()}
    hi = {k: float(np.mean(np.array(v) >= 30)) for k, v in ys.items()}
    for k, v in ys.items():
        print(f"  {k}: n={len(v)} median y={med[k]:.1f} max y={max(v)} frac y>=30 {hi[k]:.2f}")
    assert len(ys["diamond_ore"]) > 20 and len(ys["coal_ore"]) > 200
    assert max(ys["diamond_ore"]) <= 26, max(ys["diamond_ore"])
    assert med["diamond_ore"] < 12, med["diamond_ore"]
    assert med["coal_ore"] > med["iron_ore"] > med["diamond_ore"], med
    # Coal sits in the upper part of the stone column. The vanilla upper band (y 136+) scales to
    # 47..63 here, which only mountain peaks reach, so the median is ~mid-height, not the top.
    assert med["coal_ore"] >= 25, med["coal_ore"]


def test_goal_sampling_weights():
    s = CraftSim(action_mode="v6", num_envs=250, seed=3, res=8, max_steps=3000, ticks=1, threads=0)
    s.reset()
    default = [s.goal(i) for i in range(250)]
    assert all(0 <= g <= 12 for g in default), "default goals must be items 0..12 (no dirt)"
    w = np.zeros(len(ITEMS), np.float32)
    w[ITEMS.index("wooden_pickaxe")] = 3.0
    w[ITEMS.index("coal")] = 1.0
    s.set_goal_weights(w)
    goals = []
    for _ in range(4):
        s.reset()
        goals += [s.goal(i) for i in range(250)]
    frac = goals.count(ITEMS.index("wooden_pickaxe")) / len(goals)
    print(f"  weighted goals over {len(goals)} samples: wooden_pickaxe {frac:.3f} (expect 0.75)")
    assert set(goals) == {ITEMS.index("wooden_pickaxe"), ITEMS.index("coal")}
    assert abs(frac - 0.75) < 0.06, frac


def test_goal_achieved_reward():
    s = sim(1)
    reset_env(s, 0, SEED)
    s.set_goal(0, "oak_log")
    settle(s)
    s.set_inventory(0, "oak_log", 1)
    img, vec, rew, done, ach = s.step(np.array([A["noop"]], np.int32))
    print(f"  goal reached: reward {rew[0]:.3f}, done {done[0]}, ach bit31 {(int(ach[0]) >> 31) & 1}")
    assert abs(rew[0] - 10.0) < 1e-5, rew[0]
    assert bool(done[0])
    assert (int(ach[0]) >> 31) & 1 == 1
    st = s.state(0)
    assert st["episode"] == 1 and st["t"] == 0, st
    assert 0 <= st["goal"] < len(ITEMS)


def test_set_goal_rewrites_obs():
    s = sim(2)
    reset_env(s, 0, SEED)
    reset_env(s, 1, SEED)
    _, vec, _, _, _ = s.step(np.zeros(2, np.int32))
    s.set_goal(0, "iron_ingot")
    assert int(np.argmax(vec[0, GOAL_OFFSET:])) == ITEMS.index("iron_ingot")
    assert vec[0, GOAL_OFFSET:].sum() == 1.0
    _, vec2, _, _, _ = s.step(np.zeros(2, np.int32))
    assert int(np.argmax(vec2[0, GOAL_OFFSET:])) == ITEMS.index("iron_ingot")
    assert vec2[0, GOAL_OFFSET:].sum() == 1.0
    assert s.state(0)["goal"] == ITEMS.index("iron_ingot")
    assert int(np.argmax(vec2[1, GOAL_OFFSET:])) == s.state(1)["goal"]


def test_spawn_never_underwater_50_seeds():
    s = sim(1)
    for seed in range(50):
        reset_env(s, 0, seed)
        st = s.state(0)
        fx, fy, fz = feet(st)
        assert under(s) == GRASS_NAME, f"seed {seed}: feet on {under(s)}"
        assert s.block(0, fx, fy, fz) != "water", f"seed {seed}: spawned in water"
        assert st["y"] > SEA_LEVEL, f"seed {seed}: spawn y {st['y']} not above sea {SEA_LEVEL}"
    print("  50 seeds: all spawn on dry grass above sea level")


def test_benchmark():
    rng = np.random.default_rng(0)
    print("  agent-steps/sec (K=1, res=32, ticks=4, random actions):")
    for n in (1, 16, 64, 256):
        s = CraftSim(action_mode="v6", num_envs=n, seed=0, res=32, max_steps=3000, ticks=4, threads=0)
        s.reset()
        for _ in range(3):
            s.step(rng.integers(0, NUM_ACTIONS, n).astype(np.int32))
        # Best of 3 one-second windows: the box is shared (trainer, other agents), so a single
        # window can be pessimistic.
        rates = []
        for _ in range(3 if n == 64 else 1):
            steps = 0
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 1.0:
                s.step(rng.integers(0, NUM_ACTIONS, n).astype(np.int32))
                steps += 1
            rates.append(n * steps / (time.perf_counter() - t0))
        rate = max(rates)
        print(f"    N={n:4d}: {rate:12.0f} env-steps/s  (best of {len(rates)})  ({rate / n:8.1f} batch/s)")
        if n == 64:
            # Regression floor only: the Steam Deck is shared (other processes, 8 OpenMP threads).
            assert rate >= 3000, f"N=64 throughput {rate:.0f} < 3k env-steps/s"


def test_worldgen_blocks_present():
    names = ["oak_log", "oak_leaves", "sand", "gravel", "deepslate", "water", "coal_ore", "iron_ore", "diamond_ore"]
    seeds_with = {n: 0 for n in names}
    logs = []
    for seed in range(20):
        s = sim(1)
        reset_env(s, 0, seed)
        _, arr = s.region(0, 100)
        for n in names:
            c = int((arr == BLOCKS.index(n)).sum())
            if c:
                seeds_with[n] += 1
            if n == "oak_log":
                logs.append(c)
    print(f"  seeds containing: {seeds_with}; logs per seed mean {np.mean(logs):.0f} min {min(logs)}")
    assert seeds_with["oak_log"] >= 19 and seeds_with["oak_leaves"] >= 19, seeds_with
    assert seeds_with["deepslate"] == 20 and seeds_with["sand"] >= 18, seeds_with
    assert seeds_with["gravel"] >= 1 and seeds_with["coal_ore"] == 20, seeds_with


# ---------------------------------------------------------------- v3: start kits, novelty

DIAMOND = ITEMS.index("diamond")


def goal_only_sim(n=1, seed=SEED, threads=1, max_steps=3000, ticks=1):
    """Every world samples the diamond goal (weights), so the goal is known before reset."""
    s = CraftSim(action_mode="v6", num_envs=n, seed=seed, res=8, max_steps=max_steps, ticks=ticks, threads=threads)
    w = np.zeros(len(ITEMS), np.float32)
    w[DIAMOND] = 1.0
    s.set_goal_weights(w)
    return s


def kit_table(rows):
    """rows: {goal_name: {item_name: count}} -> float32[14, 14] table (other entries 0)."""
    t = np.zeros((len(ITEMS), len(ITEMS)), np.float32)
    for g, items in rows.items():
        for name, c in items.items():
            t[ITEMS.index(g), ITEMS.index(name)] = c
    return t


def kit_prob(goal_name, p):
    v = np.zeros(len(ITEMS), np.float32)
    v[ITEMS.index(goal_name)] = p
    return v


def step1(s, name, i=0):
    """Step env i with `name` (others noop). Returns (reward, done, novelty) of env i."""
    a = np.zeros(s.n, np.int32)
    a[i] = A[name]
    _, _, rew, done, _ = s.step(a)
    return float(rew[i]), bool(done[i]), float(s.last_novelty()[i])


def test_start_kit_prob_one_and_zero():
    s = goal_only_sim()
    s.set_start_kits(kit_table({"diamond": {"oak_planks": 3, "stick": 2}}), kit_prob("diamond", 1.0))
    img, vec = s.reset()
    st = s.state(0)
    assert st["kit"] and st["goal"] == DIAMOND, st
    assert s.inventory(0) == {"oak_planks": 3, "stick": 2}, s.inventory(0)

    s.set_start_kits(kit_table({"diamond": {"oak_planks": 3, "stick": 2}}), kit_prob("diamond", 0.0))
    s.reset()
    assert not s.state(0)["kit"] and s.inventory(0) == {}, s.inventory(0)

    # Mid probability: both outcomes occur across many resets.
    s.set_start_kits(kit_table({"diamond": {"oak_planks": 3}}), kit_prob("diamond", 0.5))
    kits = []
    for _ in range(200):
        s.reset()
        kits.append(s.state(0)["kit"])
    frac = sum(kits) / len(kits)
    print(f"  kit fraction at prob 0.5 over 200 resets: {frac:.2f}")
    assert 0.35 < frac < 0.65, frac
    print("  prob 1 gives the kit, prob 0 gives none, prob 0.5 gives both")


def test_kit_items_no_shaping_no_ach_but_crafting_pays():
    s = goal_only_sim()
    s.set_start_kits(kit_table({"diamond": {"oak_planks": 2}}), kit_prob("diamond", 1.0))
    s.reset()
    assert s.inventory(0) == {"oak_planks": 2}
    rew, done, _ = step1(s, "noop")
    assert rew == 0.0 and not done
    ach = int(s.step(np.zeros(1, np.int32))[4][0])
    assert ach == 0, ach                        # kit planks set no ach bit
    # Crafting sticks from kit planks: the stick is obtained by the agent, so stick shaping pays.
    a = np.array([A["craft_sticks"]], np.int32)
    _, _, rew, done, ach = s.step(a)
    expect = 0.1 * (1 + math.log2(4))           # stick value 4 (MineRL)
    print(f"  craft sticks from kit planks: reward {rew[0]:.4f} (expect {expect:.4f}), ach {int(ach[0]):#x}")
    assert abs(float(rew[0]) - expect) < 1e-5, rew[0]
    assert int(ach[0]) == 1 << ITEMS.index("stick"), ach[0]
    assert s.inventory(0) == {"stick": 4}, s.inventory(0)
    # Planks from logs: planks were given by the kit, so this episode they still pay no shaping.
    s.set_inventory(0, "oak_log", 1)
    _, _, rew, done, ach = s.step(np.array([A["craft_planks"]], np.int32))
    assert abs(float(rew[0])) < 1e-6 and not (int(ach[0]) >> ITEMS.index("oak_planks") & 1), (rew, ach)


def test_goal_never_in_kit():
    s = goal_only_sim()
    s.set_start_kits(kit_table({"diamond": {"diamond": 5, "oak_planks": 1}}), kit_prob("diamond", 1.0))
    s.reset()
    assert s.inventory(0) == {"oak_planks": 1}, s.inventory(0)
    rew, done, _ = step1(s, "noop")
    assert not done and abs(rew) < 1e-6, (rew, done)


def test_first_obs_reflects_kit():
    s = goal_only_sim(n=2)
    s.set_start_kits(kit_table({"diamond": {"oak_planks": 3}}), kit_prob("diamond", 1.0))
    img, vec = s.reset()
    planks = ITEMS.index("oak_planks")
    expect = np.log1p(3.0) / np.log1p(64.0)
    assert abs(float(vec[0, planks]) - expect) < 1e-6, (vec[0, planks], expect)
    assert abs(float(vec[1, planks]) - expect) < 1e-6
    # reset_env goes through the same path and refreshes the cached obs row.
    s.set_start_kits(kit_table({"diamond": {"oak_planks": 3}}), kit_prob("diamond", 0.0))
    s.reset_env(0, 77)
    assert abs(float(vec[0, planks])) < 1e-6, vec[0, planks]
    s.set_start_kits(kit_table({"diamond": {"oak_planks": 3}}), kit_prob("diamond", 1.0))
    s.reset_env(0, 78)
    assert abs(float(vec[0, planks]) - expect) < 1e-6, vec[0, planks]
    assert int(np.argmax(vec[0, GOAL_OFFSET:])) == DIAMOND
    print("  first obs and reset_env obs carry the kit inventory")


def test_novelty_zero_when_scale_zero():
    s = CraftSim(action_mode="v6", num_envs=2, seed=4, res=8, max_steps=3000, ticks=2, threads=1)
    s.reset()
    s.set_novelty(0.0)
    rng = np.random.default_rng(1)
    for _ in range(100):
        s.step(rng.integers(0, NUM_ACTIONS, 2).astype(np.int32))
        assert not s.last_novelty().any(), s.last_novelty()
    assert s.last_novelty().dtype == np.float32 and s.last_novelty().shape == (2,)


def test_novelty_matches_reward_difference():
    # Same seeds and actions with scale 0 and scale 1: physics is identical, so the reward
    # difference must be exactly the novelty part.
    rng = np.random.default_rng(9)
    acts = [rng.integers(0, NUM_ACTIONS, 2).astype(np.int32) for _ in range(300)]
    runs = []
    for scale in (0.0, 1.0):
        s = CraftSim(action_mode="v6", num_envs=2, seed=21, res=8, max_steps=400, ticks=2, threads=1)
        s.set_novelty(scale)
        s.reset()
        rec = []
        for a in acts:
            _, vec, rew, done, ach = s.step(a)
            rec.append((vec.copy(), rew.copy(), done.copy(), ach.copy(), s.last_novelty()))
        runs.append(rec)
    total = 0.0
    for (v0, r0, d0, a0, n0), (v1, r1, d1, a1, n1) in zip(*runs):
        assert np.array_equal(v0, v1) and np.array_equal(d0, d1) and np.array_equal(a0, a1)
        assert not n0.any()
        assert np.allclose(r1 - r0, n1, atol=1e-5), (r1 - r0, n1)
        total += float(n1.sum())
    print(f"  novelty total over 300 steps x 2 envs: {total:.2f}; reward difference matches last_novelty")
    assert total > 0


def test_novelty_new_cell_pays_revisit_does_not():
    s = goal_only_sim()
    s.set_novelty(1.0)
    s.reset_env(0, SEED)
    settle(s)

    def cell(st):
        return (math.floor(st["x"]) >> 2, math.floor(st["y"]) >> 2, math.floor(st["z"]) >> 2)

    start = cell(s.state(0))
    out = []
    for yaw in (0.0, 90.0, 180.0, 270.0):     # first direction that is not blocked by terrain
        s.reset_env(0, SEED)
        s.set_goal(0, "diamond")
        settle(s)
        s.set_pose(0, yaw, 0.0)
        start = cell(s.state(0))
        for _ in range(200):
            out.append(step1(s, "forward")[2])
            if cell(s.state(0)) != start:
                break
        if cell(s.state(0)) != start:
            break
    assert cell(s.state(0)) != start, "never left the start cell"
    assert out[-1] >= 1.0 - 1e-6, out[-1]       # the step entering the new cell pays >= 1
    print(f"  entering a new 4x4x4 cell paid {out[-1]:.1f}")

    s.set_pose(0, (yaw + 180.0) % 360.0, 0.0)   # walk back over visited cells only
    back = []
    for _ in range(200):
        back.append(step1(s, "forward")[2])
        if cell(s.state(0)) == start:
            break
    assert cell(s.state(0)) == start, "did not return to the start cell"
    # Cells pay 1, new block types 2, depth 3 (none here). Visited cells give nothing, so every
    # return step is an even sum (only block-type bonuses).
    for nv in back:
        assert abs(nv - 2 * round(nv / 2)) < 1e-6, (nv, back)
    print(f"  walking back over visited cells: per-step novelty {back}")


def test_novelty_new_block_type_pays_once():
    s = goal_only_sim()
    s.set_novelty(1.0)
    s.reset_env(0, SEED)
    settle(s)
    first = sum(step1(s, "turn_right")[2] for _ in range(24))   # full turn: new blocks in view
    second = sum(step1(s, "turn_right")[2] for _ in range(24))  # same views again
    print(f"  full turn novelty first {first:.1f}, second {second:.1f}")
    assert first > 0, first
    assert abs(second) < 1e-6, second


def test_novelty_depth_bonus_when_digging_down():
    s = goal_only_sim()
    s.set_novelty(1.0)
    s.reset_env(0, SEED)
    s.set_inventory(0, "wooden_pickaxe", 1)     # tier 1 can mine stone
    settle(s)
    total = 0.0
    depth_y = SEA_LEVEL - 4                      # 2 depth steps below sea level
    for _ in range(60):
        fy = math.floor(s.state(0)["y"])
        if fy <= depth_y:
            break
        s.set_pose(0, s.state(0)["yaw"], 90.0)   # look straight down
        for _ in range(120):                     # land (novelty of every step counts)
            total += step1(s, "noop")[2]
            if s.state(0)["on_ground"] == 1.0:
                break
        total += step1(s, "attack")[2]
        for _ in range(120):                     # drop into the hole, count every step
            total += step1(s, "noop")[2]
            if s.state(0)["on_ground"] == 1.0:
                break
    fy = math.floor(s.state(0)["y"])
    print(f"  dug down to y={fy} (sea {SEA_LEVEL}), novelty collected {total:.1f}")
    assert fy <= depth_y, fy
    assert total >= 3.0 * ((SEA_LEVEL - fy) // 2), (total, fy)


def test_kits_novelty_deterministic_same_seed():
    table = np.zeros((len(ITEMS), len(ITEMS)), np.float32)
    table[:, ITEMS.index("oak_log")] = 2
    table[:, ITEMS.index("oak_planks")] = 1
    prob = np.full(len(ITEMS), 0.5, np.float32)
    rng = np.random.default_rng(3)
    acts = [rng.integers(0, NUM_ACTIONS, 4).astype(np.int32) for _ in range(150)]
    runs = []
    for threads in (1, 0):
        s = CraftSim(action_mode="v6", num_envs=4, seed=11, res=8, max_steps=120, ticks=2, threads=threads)
        s.set_start_kits(table, prob)
        s.set_novelty(0.7)
        img, vec = s.reset()
        rec = [(img.copy(), vec.copy())]
        for a in acts:
            img, vec, rew, done, ach = s.step(a)
            rec.append((img.copy(), vec.copy(), rew.copy(), done.copy(), ach.copy(), s.last_novelty()))
        runs.append(rec)
    for x, y in zip(*runs):
        for u, v in zip(x, y):
            assert np.array_equal(u, v)
    print(f"  {len(acts)} steps x 4 envs with kits + novelty: identical across thread counts")


def test_state_reports_kit_flag():
    s = goal_only_sim()
    s.set_start_kits(kit_table({"diamond": {"oak_planks": 1}}), kit_prob("diamond", 1.0))
    s.reset()
    assert s.state(0)["kit"] is True
    s.set_start_kits(kit_table({}), kit_prob("diamond", 0.0))
    s.reset()
    assert s.state(0)["kit"] is False


def test_start_kit_and_novelty_validation():
    s = CraftSim(action_mode="v6", num_envs=1, seed=0, res=8, threads=1)
    for bad in (np.zeros((98, 14), np.float32), np.zeros((14, 14, 1), np.float32)):
        try:
            s.set_start_kits(bad, np.zeros(14, np.float32))
            raise AssertionError("bad table shape accepted")
        except ValueError:
            pass
    try:
        s.set_start_kits(np.zeros((14, 14), np.float32), np.zeros(98, np.float32))
        raise AssertionError("bad prob shape accepted")
    except ValueError:
        pass
    try:
        s.set_novelty(-1.0)
        raise AssertionError("negative novelty accepted")
    except (ValueError, RuntimeError):
        pass
    print("  bad shapes and negative scale rejected")


# ---------------------------------------------------------------- v4: env snapshots

def test_save_load_round_trip():
    s = CraftSim(action_mode="v6", num_envs=2, seed=7, res=8, max_steps=3000, ticks=2, threads=1)
    s.set_novelty(1.0)
    s.reset()
    s.set_goal(0, "diamond")
    s.set_goal(1, "diamond")
    rng = np.random.default_rng(5)
    for _ in range(60):   # some progress: visited cells, seen blocks, inventory
        s.step(np.full(2, rng.integers(0, NUM_ACTIONS), np.int32))
    blob = s.save_state(0)
    print(f"  snapshot size {len(blob)} bytes")
    assert 1_000_000 < len(blob) < 2 * 1024 * 1024, len(blob)
    # Both envs get the same action each step, so env 0 and env 1 must stay identical.
    acts = [np.full(2, rng.integers(0, NUM_ACTIONS), np.int32) for _ in range(80)]

    def trajectory(i):
        out = []
        for a in acts:
            _, vec, rew, done, ach = s.step(a)
            out.append((vec[i].copy(), float(rew[i]), bool(done[i]), int(ach[i])))
        return out

    s.load_state(0, blob)
    t0 = trajectory(0)
    s.load_state(0, blob)
    t0b = trajectory(0)
    s.load_state(1, blob)                # loaded into another env index
    t1 = trajectory(1)
    for a, b in zip(t0, t0b):
        assert np.array_equal(a[0], b[0]) and a[1:] == b[1:], "same load, different trajectory"
    for a, b in zip(t0, t1):
        assert np.array_equal(a[0], b[0]) and a[1:] == b[1:], "other env index differs"
    print("  load -> same actions gives identical obs/rewards, also in another env index")


def test_load_state_goal_override_and_restart_flag():
    s = CraftSim(action_mode="v6", num_envs=1, seed=2, res=8, max_steps=3000, ticks=1, threads=1)
    s.reset()
    s.set_goal(0, "diamond")
    blob = s.save_state(0)
    assert s.state(0)["restart"] is False
    img, vec = s.load_state(0, blob, goal="iron_ingot")
    iron = ITEMS.index("iron_ingot")
    assert s.state(0)["goal"] == iron and s.state(0)["restart"] is True
    assert s.state(0)["t"] == 0
    assert int(np.argmax(vec[0, GOAL_OFFSET:])) == iron and vec[0, GOAL_OFFSET:].sum() == 1.0
    assert img.shape == (1, 8, 8, 2) and vec.shape == (1, VEC_DIM_V6), (img.shape, vec.shape)
    s.step(np.zeros(1, np.int32))
    assert int(np.argmax(s._vec[0, GOAL_OFFSET:])) == iron
    s.load_state(0, blob)       # goal None keeps the saved goal
    assert s.state(0)["goal"] == DIAMOND
    print("  goal override rewrites the one-hot; restart flag and t = 0 after load")


def test_load_then_auto_reset_clears_restart():
    s = CraftSim(action_mode="v6", num_envs=1, seed=5, res=8, max_steps=5, ticks=1, threads=1)
    s.reset()
    s.set_goal(0, "diamond")
    s.step(np.zeros(1, np.int32))
    blob = s.save_state(0)
    s.load_state(0, blob)
    assert s.state(0)["restart"] is True
    done = [bool(s.step(np.zeros(1, np.int32))[3][0]) for _ in range(5)]
    assert done == [False, False, False, False, True], done
    st = s.state(0)
    assert st["restart"] is False and st["episode"] == 1 and st["t"] == 0, st


def test_load_state_rejects_garbage():
    s = CraftSim(action_mode="v6", num_envs=1, seed=0, res=8, threads=1)
    s.reset()
    for bad in (b"", b"nope" * 10):
        try:
            s.load_state(0, bad, None)
            raise AssertionError("garbage blob accepted")
        except (ValueError, RuntimeError):
            pass


def test_obs_resolution_64_and_32():
    # v6 default vision is 64x64 rays (craftsim.py default res=64); res=32 must still work.
    assert CraftSim.__init__.__defaults__[2] == 64
    for res in (64, 32):
        s = CraftSim(action_mode="v6", num_envs=2, agents=2, seed=3, res=res, max_steps=3000, ticks=1, threads=1)
        img, vec = s.reset()
        assert img.shape == (4, res, res, 2) and img.dtype == np.uint8, img.shape
        img, vec, rew, done, ach = s.step(np.zeros(4, np.int32))
        assert img.shape == (4, res, res, 2) and vec.shape == (4, VEC_DIM_V6)
    print("  obs shapes OK at res 64 and res 32")


def test_obs_resolution_benchmark():
    rng = np.random.default_rng(1)
    for res in (32, 64):
        s = CraftSim(action_mode="v6", num_envs=32, agents=4, seed=0, res=res, max_steps=3000, ticks=4, threads=0)
        s.reset()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            s.step(rng.integers(0, NUM_ACTIONS, s.n).astype(np.int32))
            n += 1
        rate = s.n * n / (time.perf_counter() - t0)
        print(f"  W=32,K=4 res={res}: {rate:.0f} agent-steps/s")
        assert rate > 1000, rate



# ---------------------------------------------------------------- v6: worlds, teams, chests, messages

PLANKS = ITEMS.index("oak_planks")
STICK = ITEMS.index("stick")
TABLE_ITEM = ITEMS.index("crafting_table")
CHEST_ITEM = ITEMS.index("chest")


def team_sim(w=1, k=4, seed=SEED, max_steps=3000, ticks=1, res=8):
    return CraftSim(action_mode="v6", num_envs=w, agents=k, seed=seed, res=res, max_steps=max_steps, ticks=ticks, threads=1)


def zeros(n):
    return np.zeros(n, np.int32)


def test_k1_reward_formulas_match_v5():
    # v5 shaping: first time each item is obtained, 0.1 * (1 + log2(MineRL value)); repeats pay nothing.
    s = sim(1)
    reset_env(s, 0, SEED)
    settle(s)
    s.set_inventory(0, "oak_log", 1)
    _, _, rew, _, ach = s.step(np.array([A["craft_planks"]], np.int32))
    assert abs(float(rew[0]) - 0.1 * (1 + math.log2(2))) < 1e-6, rew[0]          # planks value 2
    assert int(ach[0]) == 1 << PLANKS
    s.set_inventory(0, "oak_planks", 8)
    _, _, rew, _, _ = s.step(np.array([A["craft_sticks"]], np.int32))
    assert abs(float(rew[0]) - 0.1 * (1 + math.log2(4))) < 1e-6, rew[0]          # stick value 4
    _, _, rew, _, _ = s.step(np.array([A["craft_sticks"]], np.int32))
    assert abs(float(rew[0])) < 1e-6, rew[0]                                       # repeat pays nothing
    s.set_inventory(0, "oak_planks", 4)
    _, _, rew, _, _ = s.step(np.array([A["craft_table"]], np.int32))
    assert abs(float(rew[0]) - 0.1 * (1 + math.log2(4))) < 1e-6, rew[0]          # table value 4 (v5 ITEM_VALUE)
    print("  K=1 shaping equals 0.1*(1+log2(value)) once per item")


def test_distinct_seeds_across_worlds_and_resets():
    s = CraftSim(action_mode="v6", num_envs=16, agents=1, seed=123, res=8, max_steps=2, ticks=1, threads=1)
    s.reset()
    per_world = {i: [s.state(i)["seed"]] for i in range(16)}
    for _ in range(10):
        s.step(zeros(16)); s.step(zeros(16))            # max_steps 2: every world resets
        for i in range(16):
            per_world[i].append(s.state(i)["seed"])
    for i in range(16):
        assert len(set(per_world[i])) == len(per_world[i]), f"world {i} reused a seed"
    allseeds = [x for v in per_world.values() for x in v]
    assert len(set(allseeds)) == len(allseeds), "seeds repeat across worlds"
    assert s.state(3)["episode"] == 10
    print(f"  {len(allseeds)} world seeds, all distinct")


def test_team_done_same_step():
    # Movement/turn/look actions only: nothing is obtained, so episodes end at max_steps only.
    s = team_sim(w=4, k=3, max_steps=7)
    s.reset()
    rng = np.random.default_rng(5)
    for t in range(30):
        _, _, _, done, _ = s.step(rng.integers(0, 11, s.n).astype(np.int32))
        d = done.reshape(4, 3)
        assert (d == d[:, :1]).all(), f"step {t}: agents of one world finished on different steps"
        if (t + 1) % 7 == 0:
            assert d.all(), f"step {t}: max_steps reached but world not done"
        else:
            assert not d.any(), f"step {t}: world ended early"


def test_shared_achievement_mask():
    s = team_sim(w=1, k=2)
    s.reset_env(0, SEED)
    s.set_goal(0, "diamond")
    settle(s, 0)
    s.set_inventory(0, "oak_log", 1)
    _, _, rew, _, ach = s.step(np.array([A["craft_planks"], A["noop"]], np.int32))
    assert int(ach[0]) == int(ach[1]) == 1 << PLANKS, (ach[0], ach[1])
    s.set_inventory(1, "oak_log", 1)
    _, _, rew, _, ach = s.step(np.array([A["noop"], A["craft_planks"]], np.int32))
    assert abs(float(rew[1])) < 1e-6 and abs(float(rew[0])) < 1e-6, rew   # teammate already has planks
    print("  team mask shared; second agent pays nothing for planks the team already has")


def test_team_share_reward_split():
    for ts, expect_a, expect_b in [(0.5, 0.2, 0.1), (0.2, 0.2, 0.04)]:
        s = team_sim(w=1, k=2)
        s.set_team_share(ts)
        s.reset_env(0, SEED)
        s.set_goal(0, "diamond")
        s.set_inventory(0, "oak_log", 1)
        _, _, rew, _, _ = s.step(np.array([A["craft_planks"], A["noop"]], np.int32))
        # planks value 2 -> r = 0.2: obtainer gets ts*r + (1-ts)*r, teammate ts*r.
        assert abs(float(rew[0]) - expect_a) < 1e-6, (ts, rew[0])
        assert abs(float(rew[1]) - expect_b) < 1e-6, (ts, rew[1])
    print("  team_share 0.5 -> (0.2, 0.1); 0.2 -> (0.2, 0.04)")


def chest_setup(s):
    """Place a crafting table and a chest in front of agent 0 (K=1). Returns the chest's block pose."""
    settle(s)
    s.set_inventory(0, "oak_planks", 12)
    act(s, "craft_table")                     # 4 planks -> table
    assert s.inventory(0)["crafting_table"] == 1
    placed = False
    for yaw in range(0, 360, 15):
        s.set_pose(0, float(yaw), 45.0)
        act(s, "place_table")
        if "crafting_table" not in s.inventory(0):
            placed = True
            break
    assert placed, "table not placed"
    act(s, "craft_chest")                     # 8 planks + table in range -> chest
    assert s.inventory(0).get("chest") == 1, s.inventory(0)
    for yaw in range(0, 360, 15):
        s.set_pose(0, float(yaw), 45.0)
        act(s, "place_chest")
        if "chest" not in s.inventory(0):
            break
    assert "chest" not in s.inventory(0), "chest not placed"
    assert s.state(0)["target"]["block"] == "chest"
    return s.state(0)["target"]


def test_chest_craft_place_deposit_withdraw_spill():
    s = sim(1)
    reset_env(s, 0, SEED)
    s.set_goal(0, "diamond")
    t = chest_setup(s)
    chests = s.chests(0)
    assert len(chests) == 1 and chests[0]["items"] == {}, chests
    assert (chests[0]["x"], chests[0]["y"], chests[0]["z"]) == (t["x"], t["y"], t["z"])

    # Deposit: every non-tool item goes in, the wooden pickaxe stays.
    s.set_inventory(0, "oak_planks", 5)
    s.set_inventory(0, "stick", 3)
    s.set_inventory(0, "wooden_pickaxe", 1)
    act(s, "deposit")
    assert s.inventory(0) == {"wooden_pickaxe": 1}, s.inventory(0)
    assert s.chests(0)[0]["items"] == {"oak_planks": 5, "stick": 3}, s.chests(0)

    # Withdraw: all items come out, no achievements, no reward.
    a = np.array([A["withdraw"]], np.int32)
    _, _, rew, _, ach = s.step(a)
    assert s.inventory(0) == {"wooden_pickaxe": 1, "oak_planks": 5, "stick": 3}, s.inventory(0)
    assert s.chests(0)[0]["items"] == {}
    assert abs(float(rew[0])) < 1e-6 and int(ach[0]) & ((1 << PLANKS) | (1 << STICK)) == 0, (rew, ach)

    # Mining the chest drops the chest item and spills its contents, paying nothing for the contents.
    s.set_inventory(0, "oak_planks", 0)
    s.set_inventory(0, "stick", 0)
    s.set_inventory(0, "oak_planks", 2)
    act(s, "deposit")
    assert s.chests(0)[0]["items"] == {"oak_planks": 2}, s.chests(0)
    land(s)
    total = 0.0
    for _ in range(20):
        if s.block(0, t["x"], t["y"], t["z"]) == "air":
            break
        _, _, rew, _, ach = s.step(np.array([A["attack"]], np.int32))
        total += float(rew[0])
    assert s.block(0, t["x"], t["y"], t["z"]) == "air", "chest never broke"
    assert s.chests(0) == [], s.chests(0)
    inv = s.inventory(0)
    assert inv.get("chest") == 1 and inv.get("oak_planks") == 2, inv
    assert abs(total) < 1e-6, total       # chest item already an achievement (crafted); spill pays nothing
    assert int(ach[0]) & (1 << PLANKS) == 0
    print(f"  chest: craft, place, deposit, withdraw, mine-spill OK; inventory after break {inv}")


def feet_dist(s, a, b):
    p, q = s.state(a), s.state(b)
    return math.dist((p["x"], p["y"], p["z"]), (q["x"], q["y"], q["z"]))


def test_give_within_4_blocks_and_noop_beyond():
    near = None
    for seed in range(40):
        s = team_sim(w=1, k=2, seed=seed)
        s.reset_env(0, seed)
        if feet_dist(s, 0, 1) <= 4:
            near = seed
            break
    assert near is not None, "no spawn pair within 4 blocks"

    # Near: half (rounded up) of the planks go to the teammate.
    s = team_sim(w=1, k=2, seed=near)
    s.reset_env(0, near)
    s.set_goal(0, "diamond")
    s.set_inventory(0, "oak_planks", 5)
    s.set_inventory(0, "wooden_pickaxe", 1)
    _, _, rew, _, ach = s.step(np.array([A["give"], A["noop"]], np.int32))
    assert s.inventory(0) == {"oak_planks": 2, "wooden_pickaxe": 1}, s.inventory(0)
    assert s.inventory(1) == {"oak_planks": 3}, s.inventory(1)      # half, rounded up
    assert abs(float(rew[1])) < 1e-6 and int(ach[1]) == 0       # transfer pays nothing

    # Far: walk the teammate away (> 4 blocks), then give is a no-op.
    s = team_sim(w=1, k=2, seed=near)
    s.reset_env(0, near)
    s.set_goal(0, "diamond")
    for yaw in range(0, 360, 45):
        s.set_pose(1, float(yaw), 0.0)
        for _ in range(80):
            if feet_dist(s, 0, 1) > 4.5:
                break
            act(s, "forward", 1)
        if feet_dist(s, 0, 1) > 4.5:
            break
    assert feet_dist(s, 0, 1) > 4, "could not walk the teammate beyond 4 blocks"
    s.set_inventory(0, "oak_planks", 5)
    s.set_inventory(0, "wooden_pickaxe", 1)
    _, _, rew, _, ach = s.step(np.array([A["give"], A["noop"]], np.int32))
    assert s.inventory(0) == {"oak_planks": 5, "wooden_pickaxe": 1}, s.inventory(0)
    assert s.inventory(1) == {}, s.inventory(1)
    print(f"  give: teammate at {feet_dist(s, 0, 1):.2f} blocks is a no-op; near spawn seed {near} transfers")


def test_msgs_appear_in_teammate_slots():
    s = team_sim(w=1, k=3)
    s.reset()
    img, vec, rew, done, ach = s.step(zeros(3), np.array([5, 0, 0], np.int32))
    assert list(s.last_msgs()) == [5, 0, 0]
    # Agent 1's teammates ascending: [0, 2] -> slot 0 is agent 0; agent 2's slot 0 is agent 0 too.
    assert vec[1, TEAM_OFFSET] == 1.0 and vec[1, TEAM_OFFSET + 4 + 5] == 1.0
    assert vec[2, TEAM_OFFSET] == 1.0 and vec[2, TEAM_OFFSET + 4 + 5] == 1.0
    # Agent 0 sees agent 1 (silent) then agent 2 (silent) in slots 0 and 1; slots 2 and 3 are empty.
    assert vec[0, TEAM_OFFSET + 4] == 1.0 and vec[0, TEAM_OFFSET + 12 + 4] == 1.0
    assert (vec[0, TEAM_OFFSET + 24:TEAM_OFFSET + 48] == 0).all()
    img, vec, rew, done, ach = s.step(zeros(3), zeros(3))
    assert vec[1, TEAM_OFFSET + 4 + 5] == 0.0 and vec[1, TEAM_OFFSET + 4] == 1.0
    print("  message tokens appear one-hot in teammate slots (next obs)")


def test_agent_pseudo_block_visible_when_looking_at_teammate():
    found = None
    for seed in range(10):
        s = team_sim(w=1, k=2, seed=seed, res=32)
        s.reset()
        st0, st1 = s.state(0), s.state(1)
        dx, dz = st1["x"] - st0["x"], st1["z"] - st0["z"]
        dy = (st1["y"] + 1.0) - (st0["y"] + 1.62)
        s.set_pose(0, math.degrees(math.atan2(-dx, dz)), -math.degrees(math.atan2(dy, math.hypot(dx, dz))))
        img, vec, rew, done, ach = s.step(zeros(2))
        if (img[0, ..., 0] == AGENT_BLOCK).any():
            found = (seed, s, img)
            break
    assert found is not None, "teammate never visible in seeds 0..9"
    seed, s, img = found
    tgt = s.state(0)["target"]
    assert tgt is None or tgt["block"] != "agent", tgt   # targeting rays ignore agents
    rgb = s.render(0, 64, 48)
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    assert ((r > 2 * g) & (r > 2 * b) & (r > 60)).any(), "teammate box not drawn in render"
    print(f"  seed {seed}: AGENT pseudo-block in obs ({int((img[0, ..., 0] == AGENT_BLOCK).sum())} px), red box rendered")


def test_yaw_frame_teammate_offsets():
    s = team_sim(w=1, k=2, seed=SEED)
    s.reset()
    for yaw in (0.0, 37.0, 90.0, 200.0, 315.0):
        s.set_pose(0, yaw, 0.0)
        img, vec, rew, done, ach = s.step(zeros(2))
        st0, st1 = s.state(0), s.state(1)
        y = math.radians(st0["yaw"])
        fwd = (-math.sin(y), math.cos(y))
        right = (-math.cos(y), -math.sin(y))
        dx, dz = st1["x"] - st0["x"], st1["z"] - st0["z"]
        xl, zl = dx * right[0] + dz * right[1], dx * fwd[0] + dz * fwd[1]
        assert abs(vec[0, TEAM_OFFSET + 1] - xl / 32) < 1e-5, (yaw, vec[0, TEAM_OFFSET + 1], xl / 32)
        assert abs(vec[0, TEAM_OFFSET + 3] - zl / 32) < 1e-5, (yaw, vec[0, TEAM_OFFSET + 3], zl / 32)
        assert abs(vec[0, TEAM_OFFSET + 2] - (st1["y"] - st0["y"]) / 32) < 1e-5
    # Aim straight at the teammate: forward offset positive, sideways offset zero.
    st0, st1 = s.state(0), s.state(1)
    s.set_pose(0, math.degrees(math.atan2(-(st1["x"] - st0["x"]), st1["z"] - st0["z"])), 0.0)
    img, vec, rew, done, ach = s.step(zeros(2))
    assert vec[0, TEAM_OFFSET + 3] > 0 and abs(vec[0, TEAM_OFFSET + 1]) < 1e-3, vec[0, TEAM_OFFSET:TEAM_OFFSET + 4]
    print("  teammate offsets are in the yaw frame (x right, z forward)")


def test_chest_obs_slot_and_vec_names():
    assert len(VEC_NAMES_V6) == VEC_DIM_V6 == 102 and len(set(VEC_NAMES_V6)) == 102
    assert VEC_NAMES_V6[GOAL_OFFSET] == "goal_oak_log" and VEC_NAMES_V6[TEAM_OFFSET] == "tm0_present"
    assert VEC_NAMES_V6[CHEST_OFFSET] == "chest_present" and VEC_NAMES_V6[XHAIR_OFFSET] == "xhair_chest_oak_log"
    s = sim(1)
    reset_env(s, 0, SEED)
    s.set_goal(0, "diamond")
    t = chest_setup(s)
    img, vec, rew, done, ach = s.step(zeros(1))
    assert vec[0, CHEST_OFFSET] == 1.0 and abs(vec[0, CHEST_OFFSET + 1]) <= 1.0
    print("  VEC_NAMES_V6 unique, length 102; nearest chest slot present")


def test_save_load_world_determinism_k4():
    s = CraftSim(action_mode="v6", num_envs=2, agents=4, seed=9, res=8, max_steps=3000, ticks=2, threads=1)
    s.set_novelty(1.0)
    s.reset()
    rng = np.random.default_rng(4)
    for _ in range(40):
        a = rng.integers(0, NUM_ACTIONS, 4).astype(np.int32)
        s.step(np.concatenate([a, a]))
    blob = s.save_state(0)
    assert len(blob) < 2 * 1024 * 1024, len(blob)
    acts = []
    for _ in range(60):
        a = rng.integers(0, NUM_ACTIONS, 4).astype(np.int32)
        m = rng.integers(0, NUM_MSGS, 4).astype(np.int32)
        acts.append((np.concatenate([a, a]), np.concatenate([m, m])))

    def run():
        s.load_state(0, blob)
        s.load_state(1, blob)
        out = []
        for a, m in acts:
            _, vec, rew, done, ach = s.step(a, m)
            out.append((vec.copy(), rew.copy(), done.copy(), ach.copy()))
        return out

    r1, r2 = run(), run()
    for (v1, w1, d1, a1), (v2, w2, d2, a2) in zip(r1, r2):
        assert np.array_equal(v1, v2) and np.array_equal(w1, w2) and np.array_equal(d1, d2) and np.array_equal(a1, a2)
        # Same actions in worlds 0 and 1: their agents stay identical.
        assert np.array_equal(v1[:4], v1[4:]) and np.array_equal(w1[:4], w1[4:])
    print(f"  K=4 world snapshot {len(blob)} bytes: identical trajectories after reload")


def test_benchmark_v6_teams():
    rng = np.random.default_rng(0)
    for w, k in ((32, 4), (128, 1)):
        s = CraftSim(action_mode="v6", num_envs=w, agents=k, seed=0, res=64, max_steps=3000, ticks=4, threads=0)
        s.reset()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            s.step(rng.integers(0, NUM_ACTIONS, s.n).astype(np.int32))
            n += 1
        rate = s.n * n / (time.perf_counter() - t0)
        print(f"  W={w},K={k} res=64: {rate:.0f} agent-steps/s ({n / (time.perf_counter() - t0):.0f} steps/s)")
        assert rate > 1000, rate


# ---------------------------------------------------------------- v7 (PROTOCOL 12.3, 12.8)

P = {n: i for i, n in enumerate(PRIMARY_NAMES)}


def v7(n=1, agents=1, seed=SEED, res=8, max_steps=3000, ticks=1):
    return CraftSim(action_mode="v7", num_envs=n, agents=agents, seed=seed, res=res, max_steps=max_steps,
                    ticks=ticks, threads=1)


def step7(s, spec=None, msgs=None):
    """spec: {agent: (primary[, recipe_col, item_col, msg])}; other agents noop."""
    a = np.zeros((s.n, 4), np.int32)
    for i, v in (spec or {}).items():
        a[i, :len(v)] = v
    return s.step(a, msgs)


def recipe_col(name):
    return RECIPE_NAMES.index(name) + 1


def item_col(name):
    return ITEMS.index(name) + 1


def task_index_with(pred):
    return next(i for i, t in enumerate(TASKS) if _TASK_TYPE_NAMES[t["success_type"]] == pred)


def face(s, looker, target):
    """Turn agent `looker` (pose only) to look at agent `target`'s body centre."""
    a, b = s.state(looker), s.state(target)
    dx, dz = b["x"] - a["x"], b["z"] - a["z"]
    dy = (b["y"] + 0.9) - (a["y"] + 1.62)
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = -math.degrees(math.atan2(dy, math.hypot(dx, dz)))
    s.set_pose(looker, yaw, pitch)


def test_v7_registry_counts_and_names():
    print(f"  NUM_ITEMS={NUM_ITEMS} NUM_BLOCKS={NUM_BLOCKS} NUM_RECIPES={NUM_RECIPES} NUM_MOBS={NUM_MOBS} "
          f"NUM_TASKS={NUM_TASKS} NUM_FAMILIES={NUM_FAMILIES} NUM_PRIMARY={NUM_PRIMARY} VEC_DIM={VEC_DIM}")
    assert len(ITEMS) == NUM_ITEMS and len(BLOCKS) == NUM_BLOCKS and len(RECIPES) == NUM_RECIPES
    assert len(MOB_NAMES) == NUM_MOBS and len(TASK_NAMES) == NUM_TASKS == len(set(TASK_NAMES))
    assert len(FAMILIES) == NUM_FAMILIES and len(PRIMARY_NAMES) == NUM_PRIMARY == 21
    assert ITEMS[:15] == ITEMS_V6 and BLOCKS[:20] == BLOCKS_V6
    assert len(VEC_NAMES) == VEC_DIM == V7_BUILD + 33 and len(set(VEC_NAMES)) == VEC_DIM
    assert VEC_NAMES[V7_TASK] == "task_" + TASK_NAMES[0] and VEC_NAMES[V7_TASK_PARAM] == "task_target_item"
    assert VEC_NAMES[V7_MOBS] == "mob0_present" and VEC_NAMES[V7_BUILD + 32] == "build_progress"
    assert all(0 <= t["family"] < NUM_FAMILIES for t in TASKS) and "social" in FAMILIES
    assert sum(1 for t in TASKS if FAMILIES[t["family"]] == "social") >= 6
    s = v7()
    img, vec = s.reset()
    assert vec.shape == (1, VEC_DIM) and s.vec_dim == VEC_DIM and s.action_mode == "v7"
    print("  registry consistent, VEC_NAMES unique and aligned")


def test_v7_recipes_come_from_data():
    s = v7()
    s.reset_env(0, 1)
    r = RECIPES[RECIPE_NAMES.index("oak_planks")]
    assert r["needs"] == 0 and r["ingredients"] == {ITEMS.index("oak_log"): 1}, r
    s.set_inventory(0, "oak_log", 1)
    step7(s, {0: (P["craft"], recipe_col("oak_planks"))})
    assert s.inventory(0).get("oak_planks") == r["count"], s.inventory(0)
    assert "oak_log" not in s.inventory(0)
    # a table recipe fails with no table in reach
    s.set_inventory(0, "oak_planks", 3)
    s.set_inventory(0, "stick", 2)
    step7(s, {0: (P["craft"], recipe_col("wooden_pickaxe"))})
    assert "wooden_pickaxe" not in s.inventory(0), s.inventory(0)
    # craft with recipe column 0 (none) is a no-op
    before = s.inventory(0)
    step7(s, {0: (P["craft"], 0)})
    assert s.inventory(0) == before
    print("  oak_planks from 1 log via PR_CRAFT; table recipe refused without a table")


def test_v7_factored_actions_select_use_msg():
    s = v7()
    s.reset_env(0, 1)
    try:
        s.step(np.zeros(s.n, np.int32))
        raise AssertionError("v7 accepted a 1-column action array")
    except ValueError:
        pass
    s.set_inventory(0, "dirt", 2)
    step7(s, {0: (P["noop"], 0, item_col("dirt"), 5)})
    assert s.state(0)["held"] == "dirt" and s.state(0)["msg"] == 5
    s.set_pose(0, 0, 60)          # look down at the ground
    before = s.inventory(0).get("dirt", 0)
    step7(s, {0: (P["use"],)})
    after = s.inventory(0).get("dirt", 0)
    assert after == before - 1, (before, after)
    print("  item select -> use places the held block; msg column reaches state()")


def test_v7_task_success_has_item_and_reward():
    ti = task_index_with("has_item")
    item = ITEMS[TASKS[ti]["success_a"]]
    s = v7()
    s.reset_env(0, 2)
    s.set_task(0, ti, target=item, count=1)
    assert s.task(0) == ti and s.state(0)["task"] == TASK_NAMES[ti]
    _, _, rew, done, _ = step7(s)
    assert not done[0]
    s.set_inventory(0, item, 1)
    _, _, rew, done, _ = step7(s)
    assert done[0] and rew[0] >= 9.9, (done, rew)
    print(f"  has_item success ({item}) ends the episode with team reward {rew[0]:.2f}")


def test_v7_task_weights_select_task_on_reset():
    s = v7(n=2, seed=5)
    w = np.zeros(NUM_TASKS, np.float32)
    w[7] = 1.0
    s.set_task_weights(0, w)
    for seed in (11, 12, 13):
        s.reset_env(0, seed)
        assert s.task(0) == 7, s.task(0)
    print("  one-hot task weights give that task on every reset")


def test_v7_task_milestone_is_paid_once():
    s = v7()
    s.reset_env(0, 4)
    ti = next(i for i, t in enumerate(TASKS) if t["milestones"] and
              _TASK_TYPE_NAMES[t["success_type"]] != "has_item")
    s.set_task(0, ti)
    s.set_inventory(0, "oak_planks", 0)
    rewards = []
    for _ in range(3):
        rewards.append(float(step7(s)[2][0]))
    assert all(r >= -1e-6 for r in rewards)
    print(f"  task {TASK_NAMES[ti]}: rewards over 3 steps {['%.3f' % r for r in rewards]}")


def test_v7_social_near_teammate():
    ti = task_index_with("near_teammate")
    s = v7(agents=2, seed=3)
    s.reset_env(0, 3)
    s.set_task(0, ti, count=3)           # steps within the distance bucket
    done = False
    for _ in range(6):
        done = bool(step7(s)[3][0])
        if done:
            break
    assert done, "near_teammate did not succeed within 3 decisions of teammates being near"
    print(f"  near_teammate ({TASK_NAMES[ti]}) succeeds after 3 close decisions")


def test_v7_social_look_at_teammate():
    ti = task_index_with("look_at_teammate")
    s = v7(agents=2, seed=3)
    s.reset_env(0, 3)
    s.set_task(0, ti, count=1)
    face(s, 0, 1)
    done = bool(step7(s)[3][0])
    assert done, "crosshair on teammate did not count look_at_teammate"
    print("  crosshair ray hits the teammate's box -> look_at_teammate counted")


def test_v7_social_crouch_greet():
    ti = task_index_with("crouch_greet")
    s = v7(agents=2, seed=3)
    s.reset_env(0, 3)
    s.set_task(0, ti, count=1)
    face(s, 0, 1)
    assert not step7(s, {0: (P["sneak"],)})[3][0], "one toggle must not greet"
    assert step7(s, {0: (P["sneak"],)})[3][0], "two toggles near a teammate in view must greet"
    print("  two sneak toggles while facing a teammate -> crouch_greet")


def test_v7_social_msg_near_teammate():
    ti = task_index_with("msg_near_teammate")
    s = v7(agents=2, seed=3)
    s.reset_env(0, 3)
    s.set_task(0, ti, count=1)
    assert step7(s, {0: (P["noop"], 0, 0, 3)})[3][0], "non-silent token near a teammate must count"
    print("  non-silent message while a teammate is within 8 blocks -> msg_near_teammate")


def test_v7_social_gave_item():
    # gave_item is a milestone in the task data (social_share_chest): it pays the giver once.
    import json
    tasks_json = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tasks",
                                             "tasks.json")))
    src = next(t for t in tasks_json for m in t["milestones"] if m["predicate"]["type"] == "gave_item")
    reward = next(m["reward"] for m in src["milestones"] if m["predicate"]["type"] == "gave_item")
    ti = TASK_NAMES.index(src["id"])
    s = v7(agents=2, seed=3)
    s.reset_env(0, 3)
    s.set_task(0, ti)
    s.set_inventory(0, "dirt", 2)
    rew = step7(s, {0: (P["give"],)})[2]
    assert s.inventory(1).get("dirt") == 1 and s.inventory(0).get("dirt") == 1
    assert rew[0] >= reward - 1e-6, (rew, reward)
    print(f"  give half of the dirt stack -> gave_item milestone of {src['id']} pays {reward}")


def test_v7_knowledge_shaping_pays_once():
    ti = task_index_with("has_item")
    s = v7()
    s.reset_env(0, 6)
    s.set_task(0, ti, target="coal", count=99)     # success out of reach: only shaping matters
    s.set_shaping_table(ti, ["oak_planks"])
    s.set_inventory(0, "oak_log", 1)
    first = float(step7(s, {0: (P["craft"], recipe_col("oak_planks"))})[2][0])
    second = float(step7(s, {0: (P["craft"], recipe_col("oak_planks"))})[2][0])
    assert first >= 0.1 - 1e-6, first          # 0.05 * value 2 for planks
    assert second < 0.05, second                # second time: no shaping
    print(f"  knowledge shaping: first planks {first:.3f}, second {second:.3f}")


def test_v7_dimensions_overworld_only_for_now():
    ti = task_index_with("in_dimension")
    s = v7()
    s.reset_env(0, 1)
    s.set_task(0, ti)
    assert s.state(0)["dim"] == 0
    assert not step7(s)[3][0], "nether is not generated yet: in_dimension(nether) must not succeed"
    print("  dimension 0 (overworld); nether hook present but inert until SIM-3")


def test_v7_seeds_distinct_across_worlds_and_resets():
    s = v7(n=3, seed=9)
    s.reset()
    seeds = [s.world_seed(w) for w in range(3)]
    assert len(set(seeds)) == 3, seeds
    regions = [s.region(w, 12)[1] for w in range(3)]
    assert not np.array_equal(regions[0], regions[1]) and not np.array_equal(regions[0], regions[2])
    # auto-reset draws a new seed every episode, never reused
    s2 = v7(n=2, seed=9, max_steps=3)
    s2.reset()
    seen = [s2.world_seed(0)]
    for _ in range(10):
        s2.step(np.zeros((s2.n, 4), np.int32))
        seen.append(s2.world_seed(0))
    changes = [seen[i] for i in range(1, len(seen)) if seen[i] != seen[i - 1]]
    assert len(changes) >= 3 and len(set(seen)) == len(set(seen)), seen
    assert len(set(seen)) == len(seen) - sum(1 for i in range(1, len(seen)) if seen[i] == seen[i - 1]), seen
    print(f"  {len(set(seen))} distinct world seeds over {len(changes)} auto-resets; worlds differ")


def test_v7_obs_layout_task_onehot():
    s = v7(agents=2, seed=2)
    img, vec = s.reset()
    t = s.task(0)
    assert vec.shape == (2, VEC_DIM)
    onehot = vec[0, V7_TASK:V7_TASK + NUM_TASKS]
    assert onehot.sum() == 1.0 and onehot[t] == 1.0
    assert vec[0, V7_TASK_PARAM + 1] >= 0.0
    # inventory block uses log2(1+n)/7
    s.set_inventory(0, "oak_log", 3)
    _, vec, _, _, _ = step7(s)
    assert abs(vec[0, ITEMS.index("oak_log")] - math.log2(4) / 7) < 1e-6
    print("  task one-hot and inventory scaling match the v7 layout")


def test_v6_mode_and_v7_default():
    assert CraftSim(num_envs=1, seed=0, res=8, threads=1).action_mode == "v7"
    v = CraftSim(action_mode="v6", num_envs=1, seed=0, res=8, threads=1)
    assert v.vec_dim == VEC_DIM_V6 == 102 and v.cols == 1
    try:
        CraftSim(action_mode="v5", num_envs=1, seed=0, res=8, threads=1)
        raise AssertionError("bad action_mode accepted")
    except ValueError:
        pass
    print("  default action_mode is v7 (295-float obs); v6 is 102 floats, 1 action column")


def test_benchmark_v7_vs_v6():
    rng = np.random.default_rng(0)
    rates = {}
    for mode in ("v6", "v7"):
        s = CraftSim(action_mode=mode, num_envs=32, agents=4, seed=0, res=64, max_steps=3000, ticks=4, threads=0)
        s.reset()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            if mode == "v6":
                acts = rng.integers(0, NUM_ACTIONS, s.n).astype(np.int32)
            else:
                acts = np.zeros((s.n, 4), np.int32)
                acts[:, 0] = rng.integers(0, NUM_PRIMARY, s.n)
                acts[:, 3] = rng.integers(0, NUM_MSGS, s.n)
            s.step(acts)
            n += 1
        rates[mode] = s.n * n / (time.perf_counter() - t0)
        print(f"  {mode} W=32,K=4 res=64: {rates[mode]:.0f} agent-steps/s")
    ratio = rates["v7"] / rates["v6"]
    print(f"  v7/v6 = {ratio:.2f} (target >= 0.85)")
    assert ratio >= 0.85, ratio


TESTS = [
    test_shapes_and_constants,
    test_spawn_on_grass_seeds_0_19,
    test_jump_apex,
    test_walk_speed,
    test_break_grass_by_hand,
    test_stone_by_hand_gives_nothing,
    test_crafting_chain_and_table_proximity,
    test_region_shape_and_bounds,
    test_reset_env_determinism,
    test_episode_counter_and_last_action,
    test_airborne_mining_5x_slower,
    test_water_exists_most_seeds,
    test_water_slows_and_buoys,
    test_ore_height_distribution,
    test_goal_sampling_weights,
    test_goal_achieved_reward,
    test_set_goal_rewrites_obs,
    test_spawn_never_underwater_50_seeds,
    test_worldgen_blocks_present,
    test_start_kit_prob_one_and_zero,
    test_kit_items_no_shaping_no_ach_but_crafting_pays,
    test_goal_never_in_kit,
    test_first_obs_reflects_kit,
    test_novelty_zero_when_scale_zero,
    test_novelty_matches_reward_difference,
    test_novelty_new_cell_pays_revisit_does_not,
    test_novelty_new_block_type_pays_once,
    test_novelty_depth_bonus_when_digging_down,
    test_kits_novelty_deterministic_same_seed,
    test_state_reports_kit_flag,
    test_start_kit_and_novelty_validation,
    test_save_load_round_trip,
    test_load_state_goal_override_and_restart_flag,
    test_load_then_auto_reset_clears_restart,
    test_load_state_rejects_garbage,
    test_benchmark,
    test_obs_resolution_64_and_32,
    test_obs_resolution_benchmark,
    test_k1_reward_formulas_match_v5,
    test_distinct_seeds_across_worlds_and_resets,
    test_team_done_same_step,
    test_shared_achievement_mask,
    test_team_share_reward_split,
    test_chest_craft_place_deposit_withdraw_spill,
    test_give_within_4_blocks_and_noop_beyond,
    test_msgs_appear_in_teammate_slots,
    test_agent_pseudo_block_visible_when_looking_at_teammate,
    test_yaw_frame_teammate_offsets,
    test_chest_obs_slot_and_vec_names,
    test_save_load_world_determinism_k4,
    test_benchmark_v6_teams,
    test_v7_registry_counts_and_names,
    test_v7_recipes_come_from_data,
    test_v7_factored_actions_select_use_msg,
    test_v7_task_success_has_item_and_reward,
    test_v7_task_weights_select_task_on_reset,
    test_v7_task_milestone_is_paid_once,
    test_v7_social_near_teammate,
    test_v7_social_look_at_teammate,
    test_v7_social_crouch_greet,
    test_v7_social_msg_near_teammate,
    test_v7_social_gave_item,
    test_v7_dimensions_overworld_only_for_now,
    test_v7_knowledge_shaping_pays_once,
    test_v7_seeds_distinct_across_worlds_and_resets,
    test_v7_obs_layout_task_onehot,
    test_v6_mode_and_v7_default,
    test_benchmark_v7_vs_v6,
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
