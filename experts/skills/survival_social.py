"""Expert skills for the food, combat, survival and social families (PROTOCOL 14, EXP-3).

Mob combat uses the normal attack column: the sim hits the nearest mob within 3 blocks along the view ray (before any
block), so a skill sends `attack` only when its crosshair ray hits the mob box. Mobs come from sim.entities(w)
(type, x, y, z = feet centre, health). Night is read from sim.time_of_day(w) (night from 14000 of 24000 ticks), which
is privileged state the expert may read. Hunger is not in state(): eating tries the food and, when the sim refuses
(hunger already 20), sprints to spend exhaustion and tries again.

Social skills run in every agent of a K >= 2 world; each agent knows its slot (ctl.agent % K) and reads its teammates
through sim.state(j). Slot 0 leads (explores or holds still), the others follow.
"""
import math

import numpy as np

import craftsim as cs
from experts.core import (BID, EYE, TURN, _SOLID, _diggable, _load_json, _look_angles, _look_dir, _wrap180,
                          ray_hit)
from experts.skills import gather_logs, skill

ATTACK_REACH = 2.9          # sim mobs::attack reach is 3.0 (margin kept)
SEE_REACH = 4.2             # crosshair reach used for look_at_teammate (sim REACH 4.5)
NIGHT = 14000 / 24000.0     # survival.h NIGHT_START / DAY_TICKS
HOSTILE = ("zombie", "skeleton", "spider", "creeper")
PASSIVE_FOOD = ("cow", "pig", "sheep", "chicken")
FOOD = ("cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken", "bread", "apple",
        "beef", "porkchop", "mutton", "chicken")
IRON_ORE = ("iron_ore", "deepslate_iron_ore")
SWORDS = ("wooden_sword", "stone_sword", "iron_sword")
PICKS_STONE = ("stone_pickaxe", "iron_pickaxe")
PICKS_WOOD = ("wooden_pickaxe",) + PICKS_STONE
ROOF_ITEMS = ("dirt", "cobblestone", "oak_planks", "stone")

# Weapon policy before mob fights: "wood" crafts a wooden sword when wood is near (see experiments in
# test_survival_social), "none" fights bare-handed.
ARM = "stone"
RETREAT_AFTER = 2           # creeper: decisions spent within 3.2 blocks before backing off (hit, then leave)
HUNGER_IDX = cs.VEC_NAMES.index("hunger")   # obs value = hunger / 20

MOB_SIZE = {m["name"]: (float(m["size"][0]), float(m["size"][1])) for m in _load_json("mobs.json")}
MOB_SIZE["agent"] = (0.6, 1.8)


# ---------------------------------------------------------------- geometry and aiming

def _box_t(eye, d, x, y, z, w, h, reach):
    """Entry distance of the ray (eye, unit d) into the box with feet (x, y, z), size w x h, within reach; else None."""
    hw = w * 0.5
    lo = (x - hw, y, z - hw)
    hi = (x + hw, y + h, z + hw)
    t0, t1 = 0.0, reach
    for a in range(3):
        if abs(d[a]) < 1e-12:
            if eye[a] < lo[a] or eye[a] > hi[a]:
                return None
            continue
        ta = (lo[a] - eye[a]) / d[a]
        tb = (hi[a] - eye[a]) / d[a]
        if ta > tb:
            ta, tb = tb, ta
        t0 = max(t0, ta)
        t1 = min(t1, tb)
        if t0 > t1:
            return None
    return t0


def _current_hits(ctl, e, size, reach):
    """Does the crosshair as it is now hit box e within reach with no block nearer (attack would land)?"""
    st = ctl.state()
    eye = (st["x"], st["y"] + EYE, st["z"])
    d = _look_dir(st["yaw"], st["pitch"])
    t = _box_t(eye, d, e["x"], e["y"], e["z"], size[0], size[1], reach)
    return t is not None and ray_hit(ctl.region(), eye, d, maxd=t) is None


def _aim_box(ctl, x, y, z, size, reach):
    """Quantized (yaw, pitch, cost) whose crosshair ray hits the box at feet (x, y, z) within `reach` with no block
    nearer. Candidates are the 9 orientations around the exact centre aim; cost = turns from the current look.
    None when no candidate hits."""
    st = ctl.state()
    eye = (st["x"], st["y"] + EYE, st["z"])
    w, h = size
    yaw_t, pitch_t = _look_angles(x - eye[0], y + h * 0.5 - eye[1], z - eye[2])
    yq = int(round(yaw_t / TURN)) * 15 % 360
    pq = max(-90, min(90, int(round(pitch_t / TURN)) * 15))
    reg = ctl.region()
    best = None
    for dyaw in (0, -15, 15):
        for dp in (0, -15, 15):
            y2 = (yq + dyaw) % 360
            p2 = max(-90, min(90, pq + dp))
            d = _look_dir(y2, p2)
            t = _box_t(eye, d, x, y, z, w, h, reach)
            if t is None or ray_hit(reg, eye, d, maxd=t) is not None:
                continue
            cost = abs(_wrap180(y2 - st["yaw"])) / TURN + abs(p2 - st["pitch"]) / TURN
            if best is None or cost < best[2]:
                best = (y2, p2, cost)
    return best


def _center_orient(ctl, x, y, z, size):
    """Quantized (yaw, pitch) that looks at the centre of the box (feet x, y, z)."""
    st = ctl.state()
    yaw_t, pitch_t = _look_angles(x - st["x"], y + size[1] * 0.5 - (st["y"] + EYE), z - st["z"])
    return int(round(yaw_t / TURN)) * 15 % 360, max(-90, min(90, int(round(pitch_t / TURN)) * 15))


def _face(ctl, yaw, pitch):
    """One discrete turn or look toward (yaw, pitch) (one decision). Returns True, with no decision, when aligned."""
    st = ctl.state()
    dy = _wrap180(yaw - st["yaw"])
    dp = pitch - st["pitch"]
    if abs(dy) > TURN / 2:
        yield from ctl._act("turn_right" if dy > 0 else "turn_left")
        return False
    if abs(dp) > TURN / 2:
        yield from ctl._act("look_down" if dp > 0 else "look_up")
        return False
    return True


def _approach(ctl, tx, tz, budget=8):
    """Walk toward the window's standable cell nearest to (tx, tz). Returns True when it moved (or arrived)."""
    reg = ctl.region()
    pts = np.argwhere(reg.stand_mask())
    if pts.size == 0:
        return False
    xs = pts[:, 2] + reg.ox
    ys = pts[:, 0] + reg.oy
    zs = pts[:, 1] + reg.oz
    j = int(np.argmin((xs - tx) ** 2 + (zs - tz) ** 2))
    cell = (int(xs[j]), int(ys[j]), int(zs[j]))
    if cell == ctl.cell():
        return False
    return (yield from ctl.walk_to(*cell, max_decisions=budget))


def _night(ctl):
    return ctl.sim.time_of_day(ctl.world) >= NIGHT


def _ents(ctl, kinds):
    return [e for e in ctl.sim.entities(ctl.world) if kinds is None or e["type"] in kinds]


def _eye_dist(ctl, e):
    h = MOB_SIZE[e["type"]][1]
    return math.dist(ctl.eye(), (e["x"], e["y"] + h * 0.5, e["z"]))


def _hostile_near(ctl, r):
    return any(_eye_dist(ctl, e) <= r for e in _ents(ctl, HOSTILE))


# ---------------------------------------------------------------- generic loops

def _fight_step(ctl, kinds):
    """One combat decision against the nearest mob of `kinds`: attack when the ray hits it, else turn toward it,
    else walk toward it. Returns False when no such mob exists."""
    ents = _ents(ctl, kinds)
    if not ents:
        return False
    e = min(ents, key=lambda m: _eye_dist(ctl, m))
    size = MOB_SIZE[e["type"]]
    if _current_hits(ctl, e, size, ATTACK_REACH):
        yield from ctl._act("attack")
        return True
    aim = _aim_box(ctl, e["x"], e["y"], e["z"], size, ATTACK_REACH)
    if aim is not None:
        yield from _face(ctl, aim[0], aim[1])
        return True
    if _eye_dist(ctl, e) > 3.6:
        yield from _chase_once(ctl, e)
        return True
    # in striking range but no clear ray: face the box centre (pitch included), then step closer
    yq, pq = _center_orient(ctl, e["x"], e["y"], e["z"], size)
    if (yield from _face(ctl, yq, pq)):
        yield from _chase_once(ctl, e)
    return True


def _chase_once(ctl, e):
    """One chase decision: turn toward the mob until it is roughly ahead, then move forward (jump when the way is
    blocked or we stopped). Forward-only steering is much faster than strafing toward the mob. When forward
    movement stays blocked, the path planner (_go) takes over for a few decisions."""
    st = ctl.state()
    dx, dz = e["x"] - st["x"], e["z"] - st["z"]
    yaw_t = math.degrees(math.atan2(-dx, dz)) % 360.0
    err = _wrap180(yaw_t - st["yaw"])
    if abs(err) > 25.0:
        yield from ctl._act("turn_right" if err > 0 else "turn_left")
        return
    fx, fz = -math.sin(math.radians(st["yaw"])), math.cos(math.radians(st["yaw"]))
    reg = ctl.region()
    fxc, fyc, fzc = math.floor(st["x"] + 0.8 * fx), math.floor(st["y"] + 0.5), math.floor(st["z"] + 0.8 * fz)
    blocked = _solid_at(reg, fxc, fyc, fzc) or _solid_at(reg, fxc, fyc + 1, fzc)
    last = getattr(ctl, "_chase_last", None)
    moved = last is None or math.dist((st["x"], st["z"]), last) > 0.02
    ctl._chase_last = (st["x"], st["z"])
    ctl._chase_stuck = 0 if moved else getattr(ctl, "_chase_stuck", 0) + 1
    if ctl._chase_stuck >= 4 or (blocked and _solid_at(reg, fxc, fyc + 2, fzc)):
        ctl._chase_stuck = 0
        yield from _go(ctl, e, budget=3)
        return
    if blocked or ctl._chase_stuck >= 2 or e["y"] > st["y"] + 0.6:
        yield from ctl._act("forward_jump")
    else:
        yield from ctl._act("forward")


def _go(ctl, e, budget):
    """Walk to a standable cell next to entity e: open path first, then a path that digs through blocks, then the
    window cell nearest to it."""
    cell = (math.floor(e["x"]), math.floor(e["y"] + 1e-3), math.floor(e["z"]))
    if (yield from ctl.walk_to(*cell, within=1.5, max_decisions=budget)):
        return True
    if (yield from ctl.walk_to(*cell, within=1.5, dig=True, max_decisions=budget)):
        return True
    return (yield from _approach(ctl, e["x"], e["z"], budget=budget))


def _fight_until(ctl, kinds, done=None, idle="explore", retreat=False):
    """Fight mobs of `kinds` until done() holds (the sim ends the episode on success anyway). idle: what to do
    when none is in sight ('explore' or 'wait'). retreat: back off after RETREAT_AFTER decisions within 3.2 blocks
    (creepers: hit, then leave before the fuse ends)."""
    near_run = 0
    while True:
        if done is not None and done():
            return True
        d0 = ctl.decisions
        ents = _ents(ctl, kinds)
        if retreat and ents:
            e = min(ents, key=lambda m: _eye_dist(ctl, m))
            near_run = near_run + 1 if _eye_dist(ctl, e) <= 3.2 else 0
            if near_run >= RETREAT_AFTER:
                near_run = 0
                yield from ctl.flee(e["x"], e["y"], e["z"], distance=7)
                continue
        if not any(k in HOSTILE for k in kinds) and _hostile_near(ctl, 10.0):
            yield from _fight_step(ctl, HOSTILE)   # defend while searching for passive mobs
            if ctl.decisions == d0:
                yield from ctl.wait(1)
            continue
        if not (yield from _fight_step(ctl, kinds)):
            if idle == "wait":
                yield from ctl.wait(1)
            else:
                yield from _wander(ctl, min_d=8, max_d=16)
        if ctl.decisions == d0:
            yield from ctl.wait(1)


def _wander(ctl, min_d=6, max_d=12):
    """Move somewhere new: explore, else a digging walk toward a random window cell, else a jump-turn to unstick.
    Always spends at least one decision."""
    d0 = ctl.decisions
    if (yield from ctl.explore(min_d=min_d, max_d=max_d)):
        return True
    reg = ctl.region()
    cx, cy, cz = ctl.cell()
    tx = cx + ctl.rng.randint(-max_d, max_d)
    tz = cz + ctl.rng.randint(-max_d, max_d)
    pts = np.argwhere(reg.stand_mask())
    if pts.size:
        xs = pts[:, 2] + reg.ox
        zs = pts[:, 1] + reg.oz
        ys = pts[:, 0] + reg.oy
        j = int(np.argmin((xs - tx) ** 2 + (zs - tz) ** 2))
        if (int(xs[j]), int(ys[j]), int(zs[j])) != (cx, cy, cz):
            if (yield from ctl.walk_to(int(xs[j]), int(ys[j]), int(zs[j]), dig=True, max_decisions=40)):
                return True
    # boxed in (leaves, walls): mine the solid block straight ahead, else turn and jump
    st = ctl.state()
    fx, fz = -math.sin(math.radians(st["yaw"])), math.cos(math.radians(st["yaw"]))
    fcell = (math.floor(st["x"] + 0.9 * fx), cy, math.floor(st["z"] + 0.9 * fz))
    if _solid_at(reg, *fcell) and _diggable(reg.get(*fcell)):
        if (yield from ctl.mine_block(*fcell, walk=False)):
            return False
    if ctl.decisions == d0:
        yield from ctl._act("turn_left" if ctl.rng.random() < 0.5 else "turn_right")
    yield from ctl._act("forward_jump")
    return False


def _gather(ctl, blocks, item, n, max_fails=30):
    """Mine nearest blocks of `blocks` until the inventory holds `n` of `item` (generalises gather_logs)."""
    blacklist = set()
    fails = 0
    while ctl.count(item) < n:
        if fails > max_fails:
            return False
        before = ctl.count(item)
        found = yield from ctl.mine_nearest(list(blocks), radius=12, skip=blacklist)
        if found is None:
            fails += 1
            yield from _wander(ctl)
            continue
        cell, ok = found
        if not ok or ctl.count(item) <= before:
            blacklist.add(cell)
            fails += 1
            yield from _wander(ctl)
    return True


# ---------------------------------------------------------------- crafting chain

def _has(ctl, names):
    return any(ctl.count(n) > 0 for n in names)


def _planks(ctl, n, fails_max=6):
    """At least n oak_planks: crafts from logs, gathering a log when the inventory has none."""
    fails = 0
    while ctl.count("oak_planks") < n:
        if ctl.count("oak_log") == 0 and not (yield from gather_logs(ctl, 1)):
            return False
        if (yield from ctl.craft("oak_planks")):
            fails = 0
        else:
            fails += 1
            if fails > fails_max:
                return False
    return True


def _table(ctl):
    """A crafting table is in the inventory or in the window, else one is crafted (placed by the first craft)."""
    if ctl.count("crafting_table") > 0 or ctl.find_blocks(["crafting_table"], radius=12):
        return True
    if not (yield from _planks(ctl, 4)):
        return False
    return (yield from ctl.craft("crafting_table"))


def _wood_sword(ctl):
    if _has(ctl, SWORDS):
        return True
    if not (yield from _planks(ctl, 8)) or not (yield from _table(ctl)):
        return False
    if ctl.count("stick") < 1 and not (yield from ctl.craft("stick")):
        return False
    return (yield from ctl.craft("wooden_sword"))


def _stone_sword(ctl):
    if _has(ctl, ("stone_sword", "iron_sword")):
        return True
    if not (yield from _stone_pick(ctl)):
        return False
    if not (yield from _gather(ctl, ["stone"], "cobblestone", 2)):
        return False
    if ctl.count("stick") < 1 and not (yield from ctl.craft("stick")):
        return False
    return (yield from ctl.craft("stone_sword"))


def _arm(ctl, stone=True):
    """Before a mob fight: a stone sword when time allows (ARM == 'stone'), else a wooden sword, when wood is near."""
    if _has(ctl, SWORDS):
        return
    if ctl.count("oak_log") == 0 and not ctl.find_blocks(["oak_log"], radius=10):
        return
    if ARM == "stone" and stone and (yield from _stone_sword(ctl)):
        return
    yield from _wood_sword(ctl)


def _wood_pick(ctl):
    if _has(ctl, PICKS_WOOD):
        return True
    if not (yield from _planks(ctl, 9)) or not (yield from _table(ctl)):
        return False
    if ctl.count("stick") < 2 and not (yield from ctl.craft("stick")):
        return False
    return (yield from ctl.craft("wooden_pickaxe"))


def _stone_pick(ctl):
    if _has(ctl, PICKS_STONE):
        return True
    if not (yield from _wood_pick(ctl)):
        return False
    if not (yield from _gather(ctl, ["stone"], "cobblestone", 3)):
        return False
    if ctl.count("stick") < 2 and not (yield from ctl.craft("stick")):
        return False
    return (yield from ctl.craft("stone_pickaxe"))


def _furnace(ctl):
    """A furnace is placed and reachable (crafted from 8 cobblestone at a table)."""
    if ctl.count("furnace") == 0 and not ctl.find_blocks(["furnace"], radius=12):
        if not (yield from _wood_pick(ctl)):
            return False
        if ctl.count("cobblestone") < 8 and not (yield from _gather(ctl, ["stone"], "cobblestone", 8)):
            return False
        if not (yield from _table(ctl)) or not (yield from ctl.craft("furnace")):
            return False
    return (yield from ctl.ensure_station("furnace"))


def _smelt(ctl, raw):
    """One smelt of `raw` in a furnace; fuel (a log or planks) is gathered when the inventory has none."""
    if not (yield from _furnace(ctl)):
        return False
    if not _has(ctl, ("oak_log", "oak_planks", "coal")):
        yield from _gather(ctl, ["oak_log"], "oak_log", 1)
    return (yield from ctl.smelt(raw))


# ---------------------------------------------------------------- food

def _hunger(ctl):
    """Hunger 0..20 from the observation the sim already computes (obs index 'hunger', value / 20)."""
    return float(ctl.sim._vec[ctl.agent][HUNGER_IDX]) * 20.0


def _eat(ctl, item):
    """Eat one `item` when hunger is below 20 (the sim refuses at 20). Otherwise sprint to spend exhaustion."""
    if ctl.count(item) == 0:
        return False
    if _hunger(ctl) < 19.5 and (yield from ctl.eat(item)):
        return True
    for _ in range(6):
        yield from ctl._act("sprint_forward")
    yield from ctl._act("turn_left")
    return False


def _cook(ctl, raw, cooked, n, mob):
    """Hunt `mob` for raw food and smelt it until `n` cooked items are held."""
    while ctl.count(cooked) < n:
        d0 = ctl.decisions
        if ctl.count(raw) == 0:
            yield from _fight_until(ctl, [mob], done=lambda: ctl.count(raw) > 0)
        elif not (yield from _smelt(ctl, raw)):
            yield from _wander(ctl)
        if ctl.decisions == d0:
            yield from ctl.wait(1)


def _hunt(ctl, mob):
    yield from _arm(ctl, stone=False)
    yield from _fight_until(ctl, [mob])


@skill("food_hunt_cow")
def food_hunt_cow(ctl, task):
    yield from _hunt(ctl, "cow")


@skill("food_hunt_pig")
def food_hunt_pig(ctl, task):
    yield from _hunt(ctl, "pig")


@skill("food_hunt_sheep")
def food_hunt_sheep(ctl, task):
    yield from _hunt(ctl, "sheep")


@skill("food_hunt_chicken")
def food_hunt_chicken(ctl, task):
    yield from _hunt(ctl, "chicken")


@skill("food_cook_beef")
def food_cook_beef(ctl, task):
    yield from _cook(ctl, "beef", "cooked_beef", int(task["params"].get("count", 2)), "cow")


@skill("food_cook_porkchop")
def food_cook_porkchop(ctl, task):
    yield from _cook(ctl, "porkchop", "cooked_porkchop", int(task["params"].get("count", 2)), "pig")


@skill("food_eat_cooked_beef")
def food_eat_cooked_beef(ctl, task):
    yield from _cook(ctl, "beef", "cooked_beef", 1, "cow")
    while True:
        yield from _eat(ctl, "cooked_beef")


# ---------------------------------------------------------------- combat

def _zombie_like(ctl, mob, retreat=False):
    yield from _arm(ctl)
    yield from _fight_until(ctl, [mob], idle="wait", retreat=retreat)


@skill("combat_kill_zombie")
def combat_kill_zombie(ctl, task):
    yield from _zombie_like(ctl, "zombie")


@skill("combat_zombie_horde")
def combat_zombie_horde(ctl, task):
    yield from _zombie_like(ctl, "zombie")


@skill("combat_kill_skeleton")
def combat_kill_skeleton(ctl, task):
    yield from _zombie_like(ctl, "skeleton")


@skill("combat_kill_spider")
def combat_kill_spider(ctl, task):
    yield from _zombie_like(ctl, "spider")


@skill("combat_kill_creeper")
def combat_kill_creeper(ctl, task):
    yield from _zombie_like(ctl, "creeper", retreat=True)


def _iron_gear(ctl, recipe, n_ingots):
    """Craft `recipe` (iron sword / chestplate): iron ore (stone pickaxe), smelt, sticks, table."""
    while ctl.count(recipe) == 0:
        d0 = ctl.decisions
        if not (yield from _stone_pick(ctl)):
            yield from _wander(ctl)
        elif ctl.count("raw_iron") > 0:
            if not (yield from _smelt(ctl, "raw_iron")):
                yield from _wander(ctl)
        elif ctl.count("iron_ingot") < n_ingots:
            if not (yield from _gather(ctl, IRON_ORE, "raw_iron", 1, max_fails=60)):
                yield from _wander(ctl)
        else:
            if ctl.count("stick") < 1:
                if not (yield from _planks(ctl, 2)):
                    yield from _wander(ctl)
                    continue
                yield from ctl.craft("stick")
            if not (yield from _table(ctl)):
                yield from _wander(ctl)
            else:
                yield from ctl.craft(recipe)
        if ctl.decisions == d0:
            yield from ctl.wait(1)


@skill("combat_iron_sword")
def combat_iron_sword(ctl, task):
    yield from _iron_gear(ctl, "iron_sword", 2)


@skill("combat_iron_chestplate")
def combat_iron_chestplate(ctl, task):
    yield from _iron_gear(ctl, "iron_chestplate", 8)


# ---------------------------------------------------------------- survival

@skill("survival_eat_any")
def survival_eat_any(ctl, task):
    while True:
        food = next((f for f in FOOD if ctl.count(f) > 0), None)
        if food is None:
            yield from _fight_until(ctl, list(PASSIVE_FOOD),
                                    done=lambda: any(ctl.count(f) > 0 for f in FOOD))
        else:
            yield from _eat(ctl, food)


@skill("survival_recover_health")
def survival_recover_health(ctl, task):
    """Eat when hunger allows (the sim refuses at full hunger); natural regeneration heals while fed."""
    while True:
        d0 = ctl.decisions
        food = next((f for f in FOOD if ctl.count(f) > 0), None)
        if food is None:
            yield from _fight_until(ctl, list(PASSIVE_FOOD),
                                    done=lambda: any(ctl.count(f) > 0 for f in FOOD))
        elif not (yield from ctl.eat(food)):
            yield from ctl.wait(2)
        if ctl.decisions == d0:
            yield from ctl.wait(1)


def _bed_items(ctl):
    """A white_bed: 3 white_wool (sheep drop one each) + 3 planks at a crafting table."""
    while ctl.count("white_bed") == 0:
        d0 = ctl.decisions
        if ctl.count("white_wool") < 3:
            if ctl.count("string") >= 4:
                yield from ctl.craft("white_wool_from_string")
            else:
                yield from _fight_until(ctl, ["sheep"], done=lambda: ctl.count("white_wool") >= 3)
        elif not (yield from _planks(ctl, 7)) or not (yield from _table(ctl)):
            yield from _wander(ctl)
        else:
            yield from ctl.craft("white_bed")
        if ctl.decisions == d0:
            yield from ctl.wait(1)


def _bed_run(ctl):
    """Use the bed at night once no hostile is within 8 blocks (survival.cpp sleep); fight hostiles before that."""
    while True:
        d0 = ctl.decisions
        if _night(ctl) and ctl.count("white_bed") > 0 and not _hostile_near(ctl, 8.6):
            yield from ctl.equip("white_bed")
            yield from ctl._act("use")
        elif _hostile_near(ctl, 12.0):
            yield from _fight_step(ctl, HOSTILE)
        else:
            yield from ctl.wait(1)
        if ctl.decisions == d0:
            yield from ctl.wait(1)


@skill("survival_sleep_in_bed")
def survival_sleep_in_bed(ctl, task):
    yield from _bed_items(ctl)
    yield from _bed_run(ctl)


@skill("survival_survive_night_bed")
def survival_survive_night_bed(ctl, task):
    yield from _bed_run(ctl)


def _solid_at(reg, x, y, z):
    b = reg.get(x, y, z)
    return b > 0 and _SOLID[b]


def _pit_site_ok(ctl, x, f, z):
    """Pit under the feet cell (x, f, z): solid ground below and beside the three pit cells, no liquid nearby,
    and a solid block beside the roof cell so the roof can be placed."""
    reg = ctl.region()
    for dy in range(-4, 2):
        for dx, dz in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            b = reg.get(x + dx, f + dy, z + dz)
            if b < 0 or b in (BID["water"], BID["lava"]):
                return False
    for dy in (-1, -2, -3):
        if not _diggable(reg.get(x, f + dy, z)):
            return False
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if not _solid_at(reg, x + dx, f + dy, z + dz):
                return False
    return _solid_at(reg, x, f - 4, z) and (_solid_at(reg, x + 1, f - 1, z) or _solid_at(reg, x - 1, f - 1, z))


def _pit(ctl):
    """Dig three cells straight down (the body falls into each hole), then seal the top of the pit with a block
    on the ground layer. Zombies cannot reach it, skeletons see nothing, and a creeper on the surface is more than
    3 blocks away, so it never fuses. Returns True when sealed."""
    x, f, z = ctl.cell()
    if not _pit_site_ok(ctl, x, f, z):
        return False
    for depth in (1, 2, 3):
        if not (yield from ctl.mine_block(x, f - depth, z, walk=False)):
            return False
        for _ in range(4):
            if ctl.cell()[1] <= f - depth:
                break
            yield from ctl.wait(1)
    # settle on the floor of the pit (the body is still falling right after the last fall)
    for _ in range(10):
        if abs(ctl.state()["y"] - round(ctl.state()["y"])) < 0.02:
            break
        yield from ctl.wait(1)
    roof = next((i for i in ROOF_ITEMS if ctl.count(i) > 0), None)
    if roof is None:
        return False
    return (yield from ctl.place_block(roof, x, f - 1, z, walk=False))


@skill("survival_survive_night_hut")
def survival_survive_night_hut(ctl, task):
    while True:
        d0 = ctl.decisions
        if (yield from _pit(ctl)):
            break
        yield from _wander(ctl, min_d=3, max_d=6)
        if ctl.decisions == d0:
            yield from ctl.wait(1)
    while True:   # sealed: wait for the night to pass (survive_night pays at dawn)
        yield from ctl.wait(1)


# ---------------------------------------------------------------- social (K >= 2)

def _mates(ctl):
    out = []
    for j in ctl.state()["team"]:
        s = ctl.sim.state(j)
        out.append({"type": "agent", "x": s["x"], "y": s["y"], "z": s["z"]})
    return out


def _mate_dist(ctl, m):
    return math.dist(ctl.pos(), (m["x"], m["y"], m["z"]))


def _nearest_mate(ctl, mates):
    m = min(mates, key=lambda m: _mate_dist(ctl, m))
    return m, _mate_dist(ctl, m)


def _slot(ctl):
    return ctl.agent % ctl.K


@skill("social_stay_together")
def social_stay_together(ctl, task):
    """Slot 0 holds still (or wanders a few blocks when close); the others close in to 3.5 blocks."""
    while True:
        d0 = ctl.decisions
        mates = _mates(ctl)
        if not mates:
            yield from _wander(ctl)
        else:
            m, dist = _nearest_mate(ctl, mates)
            if _slot(ctl) == 0:
                if dist > 5:
                    yield from ctl.wait(1)
                else:
                    yield from _wander(ctl, min_d=2, max_d=4)
            elif dist > 3.5:
                if not (yield from _approach(ctl, m["x"], m["z"], budget=4)):
                    yield from _wander(ctl, min_d=2, max_d=4)
            else:
                yield from ctl.wait(1)
        if ctl.decisions == d0:
            yield from ctl.wait(1)


@skill("social_follow_teammate")
def social_follow_teammate(ctl, task):
    """Slot 0 explores; the others walk after it, keeping it 3..8 blocks away and in view."""
    if _slot(ctl) == 0:
        while True:
            d0 = ctl.decisions
            yield from _wander(ctl, min_d=4, max_d=8)
            if ctl.decisions == d0:
                yield from ctl.wait(1)
    lead = ctl.world * ctl.K
    last, stuck = None, 0
    while True:
        d0 = ctl.decisions
        ld = ctl.sim.state(lead)
        dist = _mate_dist(ctl, {"x": ld["x"], "y": ld["y"], "z": ld["z"]})
        if dist > 8.5:
            if not (yield from _approach(ctl, ld["x"], ld["z"], budget=4)):
                yield from _wander(ctl, min_d=2, max_d=5)
        elif dist < 3.0:
            yield from ctl.wait(1)
        else:
            yq, pq = _center_orient(ctl, ld["x"], ld["y"], ld["z"], MOB_SIZE["agent"])
            if (yield from _face(ctl, yq, pq)):
                pos = ctl.pos()
                stuck = stuck + 1 if last is not None and math.dist(pos, last) < 0.01 else 0
                last = pos
                yield from ctl._act("forward_jump" if stuck >= 3 else "forward")
        if ctl.decisions == d0:
            yield from ctl.wait(1)


@skill("social_crouch_greet")
def social_crouch_greet(ctl, task):
    """Walk up to a teammate, face its body and toggle sneak (two toggles within 10 decisions greet)."""
    while True:
        d0 = ctl.decisions
        mates = _mates(ctl)
        if not mates:
            yield from _wander(ctl)
        else:
            m, dist = _nearest_mate(ctl, mates)
            if dist > 3.2:
                if not (yield from _approach(ctl, m["x"], m["z"], budget=4)):
                    yield from _wander(ctl, min_d=2, max_d=4)
            else:
                yq, pq = _center_orient(ctl, m["x"], m["y"], m["z"], MOB_SIZE["agent"])
                if (yield from _face(ctl, yq, pq)):
                    yield from ctl._act("sneak")
        if ctl.decisions == d0:
            yield from ctl.wait(1)


@skill("social_look_at_teammate")
def social_look_at_teammate(ctl, task):
    """Keep a teammate under the crosshair: aim at its box (the crosshair ray hits the AGENT pseudo block)."""
    while True:
        d0 = ctl.decisions
        mates = _mates(ctl)
        if not mates:
            yield from _wander(ctl)
        else:
            m, dist = _nearest_mate(ctl, mates)
            if dist > SEE_REACH - 1.0:
                if not (yield from _approach(ctl, m["x"], m["z"], budget=4)):
                    yield from _wander(ctl, min_d=2, max_d=4)
            else:
                aim = _aim_box(ctl, m["x"], m["y"], m["z"], MOB_SIZE["agent"], SEE_REACH)
                if aim is None:
                    yq, pq = _center_orient(ctl, m["x"], m["y"], m["z"], MOB_SIZE["agent"])
                    if (yield from _face(ctl, yq, pq)):
                        yield from ctl.wait(1)
                elif aim[2] == 0:
                    yield from ctl.wait(1)
                else:
                    yield from _face(ctl, aim[0], aim[1])
        if ctl.decisions == d0:
            yield from ctl.wait(1)


@skill("social_chat_nearby")
def social_chat_nearby(ctl, task):
    """Send non-silent chat tokens (msg column) while a teammate is within 8 blocks."""
    while True:
        d0 = ctl.decisions
        mates = _mates(ctl)
        if not mates:
            yield from _wander(ctl)
        else:
            m, dist = _nearest_mate(ctl, mates)
            if dist <= 6.5:
                yield from ctl._act("noop", msg=1 + ctl.decisions % 7)
            elif not (yield from _approach(ctl, m["x"], m["z"], budget=4)):
                yield from _wander(ctl, min_d=2, max_d=4)
        if ctl.decisions == d0:
            yield from ctl.wait(1)


def _chest_site(ctl):
    """An empty cell 2 blocks away on a solid floor (for the chest)."""
    reg = ctl.region()
    x, f, z = ctl.cell()
    for dx, dz in ((2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2)):
        cx, cz = x + dx, z + dz
        if reg.get(cx, f, cz) == BID["air"] and _solid_at(reg, cx, f - 1, cz):
            return cx, f, cz
    return None


@skill("social_share_chest")
def social_share_chest(ctl, task):
    """Slot 0 places a chest, deposits its non-tool items there; the other slots withdraw from it (shared_chest)."""
    if _slot(ctl) != 0:
        lead = ctl.world * ctl.K
        while True:
            d0 = ctl.decisions
            full = [c for c in ctl.sim.chests(ctl.world) if sum(c["items"].values()) > 0]
            if not full:
                s = ctl.sim.state(lead)
                if math.dist(ctl.pos(), (s["x"], s["y"], s["z"])) > 4.0:
                    yield from _approach(ctl, s["x"], s["z"], budget=4)
                else:
                    yield from ctl.wait(1)
            else:
                c = min(full, key=lambda c: math.dist(ctl.pos(), (c["x"], c["y"], c["z"])))
                if math.dist(ctl.pos(), (c["x"], c["y"], c["z"])) > 3.0:
                    yield from _approach(ctl, c["x"], c["z"], budget=4)
                elif (yield from ctl.face(c["x"], c["y"], c["z"])):
                    yield from ctl._act("withdraw")
                else:
                    yield from ctl.wait(1)
            if ctl.decisions == d0:
                yield from ctl.wait(1)
    while ctl.count("oak_log") == 0 and ctl.count("oak_planks") == 0:
        if not (yield from gather_logs(ctl, 1)):
            yield from _wander(ctl)
    while True:
        site = _chest_site(ctl)
        if site is not None:
            break
        yield from _wander(ctl, min_d=2, max_d=4)
    if not (yield from ctl.place_block("chest", *site)):
        return
    if not (yield from ctl.face(*site)):
        return
    yield from ctl._act("deposit")
    while True:
        yield from ctl.wait(1)
