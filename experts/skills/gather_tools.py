"""Gather and tools expert skills (PROTOCOL 14.4): raw materials, the tool tree, furnace and iron smelting.

Every skill is built from one recursive maker, `make(ctl, item, n)`:
  * a raw item (SOURCE) is mined: the nearest visible block of its kind first; when the block is out of reach it
    is approached by a staircase (down), a tunnel or a dig path; the needed pickaxe tier is made first;
  * a craftable item is made from its recipe (registry recipes.json, "inventory" or "crafting_table"): the crafting
    table is placed first, then the ingredients are made until all are in the inventory, then the craft runs;
  * iron_ingot is smelted from raw_iron in a furnace (fuel: coal, else planks, one unit per smelt).

Skills only pursue their goal; the sim decides success (tasks.json). A skill that gives up returns, and the Runner
closes the episode early (runner._end_early). Hostile mobs within 4 blocks are fought back (`_guard`).
"""
import math

from experts.core import BLOCK_INFO, ITEM_INFO, RECIPES, BID, standable, _station_near
from experts.skills import skill

PICK = {1: "wooden_pickaxe", 2: "stone_pickaxe", 3: "iron_pickaxe", 4: "diamond_pickaxe"}
HOSTILE = ("zombie", "skeleton", "spider", "creeper", "zombified_piglin")
WEAPONS = ("diamond_sword", "iron_sword", "stone_sword", "wooden_sword")
GUARD_R = 4.0
MAX_FAILS = 25
MAX_DEPTH = 8

# item -> block names that yield it when mined (the first name decides the pickaxe tier)
SOURCE = {
    "oak_log": ("oak_log",),
    "cobblestone": ("stone",),
    "sand": ("sand",),
    "gravel": ("gravel",),
    "flint": ("gravel",),          # gravel drops flint 10% of the time
    "coal": ("coal_ore", "deepslate_coal_ore"),
    "raw_iron": ("iron_ore", "deepslate_iron_ore"),
    "diamond": ("diamond_ore", "deepslate_diamond_ore"),
    "obsidian": ("obsidian",),
}
SURFACE = ("oak_log", "sand", "gravel", "flint")       # search by walking the surface
FAR = ("diamond", "obsidian")                           # search the whole window (16 blocks up/down too)
# depth (feet y) a search descends to before tunnelling, for the ores that sit low in the world
DEPTH = {"diamond": 8, "raw_iron": 18, "coal": 30}
BAD = ("water", "lava")


# ---------------------------------------------------------------------- helpers

def _guard(ctl):
    """A hostile mob within GUARD_R blocks is attacked with the best weapon in the inventory (no fleeing here)."""
    x, y, z = ctl.pos()
    near = [e for e in ctl.sim.entities(ctl.world)
            if e["type"] in HOSTILE and math.dist((e["x"], e["y"], e["z"]), (x, y, z)) <= GUARD_R]
    if not near:
        return False
    for w in WEAPONS:
        if ctl.count(w) > 0:
            if ctl.held() != w:
                yield from ctl.equip(w)
            break
    target = min(near, key=lambda e: math.dist((e["x"], e["y"], e["z"]), (x, y, z)))
    yield from ctl.attack_entity(target["type"], max_decisions=60)
    return True


def _have_pick(ctl, tier):
    for name, cnt in ctl.inventory().items():
        if cnt > 0:
            t = ITEM_INFO.get(name, {}).get("tool")
            if t and t["type"] == "pickaxe" and t["tier"] >= tier:
                return True
    return False


def _recipe_for(item):
    for r in RECIPES:
        if r["result"] == item and r["needs"] in ("inventory", "crafting_table"):
            return r
    return None


def _bad_near(ctl, cells):
    """True if any of the cells is water/lava or unknown (outside the window)."""
    for c in cells:
        b = ctl.block_at(*c)
        if b is None or b in BAD:
            return True
    return False


def _near_water(ctl, cell):
    x, y, z = cell
    return any(ctl.block_at(x + a, y + b, z + c) == "water"
               for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1))


def _walk_to_station(ctl, name):
    sites = ctl.find_blocks([name], radius=16)
    if not sites:
        return False
    (sx, _, sz), _ = sites[0]
    bid = BID[name]
    return (yield from ctl._walk(lambda n, r: standable(r, *n) and _station_near(r, n, bid), sx, sz,
                                 max_decisions=300))


def _place_station(ctl, name):
    """Place `name` from the inventory into a flat neighbour cell (dug out first when it is solid rock)."""
    cx, cy, cz = ctl.cell()
    for dx, dz in _dir_order(ctl, cx, cz, None):
        c = (cx + dx, cy, cz + dz)
        b = ctl.block_at(*c)
        if b in (None,) or b in BAD or ctl.block_at(c[0], cy - 1, c[2]) in (None, "air"):
            continue
        if b != "air" and not (yield from ctl.mine_block(*c, walk=False)):
            continue
        if ctl.block_at(*c) == "air" and (yield from ctl.place_block(name, *c, walk=False, max_attempts=2)):
            return True
    return False


def _ensure_station(ctl, name, depth=0):
    """A crafting table / furnace in reach: already near, placed from the inventory, walked to (window), or made."""
    if ctl.near_block(name):
        return True
    for _ in range(3):
        if ctl.count(name) > 0 and ((yield from ctl.ensure_station(name, walk=False)) or
                                    (yield from _place_station(ctl, name))):
            return True
        if ctl.count(name) == 0:
            break
        yield from ctl.explore(min_d=2, max_d=4)     # the spot was blocked: step aside and try again
    if (yield from _walk_to_station(ctl, name)):
        return True
    if depth > MAX_DEPTH:
        return False
    if ctl.count(name) == 0 and not (yield from make(ctl, name, 1, depth + 1)):
        return False
    return (yield from ctl.ensure_station(name, walk=False)) or (yield from _place_station(ctl, name))


# ---------------------------------------------------------------------- movement into the ground

def _dir_order(ctl, cx, cz, toward):
    """The four flat directions, best first: nearest to `toward` (x, z) if given, else the one the body faces;
    a turn costs decisions (15 deg each), so the facing direction is preferred on ties."""
    yaw = math.radians(ctl.state()["yaw"])
    fx, fz = -math.sin(yaw), math.cos(yaw)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def score(d):
        face = d[0] * fx + d[1] * fz              # 1 = facing it, -1 = behind
        turn = (1.0 - face) * 1.5                 # 0 facing .. 3 behind
        if toward is None:
            return turn
        tx, tz = toward
        return ((cx + d[0] - tx) ** 2 + (cz + d[1] - tz) ** 2) ** 0.5 + turn
    return sorted(dirs, key=score)


def _walk_dir(ctl, dx, dz, tx, ty, tz, tries=6):
    """Face (dx, dz) and walk straight into the target cell (falls one level when it is a staircase step).
    Returns True when the feet are on (tx, ty, tz)."""
    yaw = (math.degrees(math.atan2(-dx, dz)) % 360.0)
    if (yield from ctl.turn_to_yaw(yaw)) is None:
        return False
    for _ in range(tries):
        if ctl.cell() == (tx, ty, tz):
            return True
        yield from ctl._act("forward")
    return ctl.cell() == (tx, ty, tz)


def _center(ctl, cx, cz):
    """Move (a few decisions) so the feet sit over the middle of column (cx, cz): a 1-block shaft must be fallen
    straight down."""
    for _ in range(4):
        x, _, z = ctl.pos()
        dx, dz = cx + 0.5 - x, cz + 0.5 - z
        if abs(dx) <= 0.2 and abs(dz) <= 0.2:
            return
        yield from ctl._act(ctl._dir_prim(dx, dz))


def _shaft_down(ctl):
    """Dig straight down one block and fall into the hole (about 4 decisions a level). Refused when the block below
    the hole is not solid (a cave or a fluid would be entered) or fluids touch the shaft. Returns True when lower."""
    cx, cy, cz = ctl.cell()
    if ctl.block_at(cx, cy - 1, cz) in (None, "air") or ctl.block_at(cx, cy - 2, cz) in (None, "air"):
        return False
    if ctl.block_at(cx, cy - 2, cz) in BAD:
        return False
    if _bad_near(ctl, [(cx + a, cy - 1 + k, cz + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
                       for k in (0, -1)]):
        return False
    yield from _center(ctl, cx, cz)
    if ctl.state()["pitch"] < 80:
        yield from ctl.look_to_pitch(90)
    if not (yield from ctl.mine_block(cx, cy - 1, cz, walk=False)):
        return False
    for _ in range(6):
        yield from ctl._act("noop")
        if ctl.cell()[1] < cy:
            return True
    return ctl.cell()[1] < cy


def _stair_down(ctl, toward):
    """One staircase step down: clear the cells of the next feet/head (the floor below stays solid), then step.
    The direction nearest to `toward` (x, z) is tried first, else the facing one. Returns True when one level lower."""
    cx, cy, cz = ctl.cell()
    for dx, dz in _dir_order(ctl, cx, cz, toward):
        nx, nz = cx + dx, cz + dz
        floor = ctl.block_at(nx, cy - 2, nz)
        if floor is None or floor in BAD or floor == "air":
            continue
        # the new cells and their lateral neighbours must not touch fluids
        side = [(nx + a, cy - 1 + k, nz + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)) for k in (0, 1)]
        if _bad_near(ctl, [(nx, cy, nz), (nx, cy - 1, nz), (nx, cy - 2, nz)] + side):
            continue
        ok = True
        for c in ((nx, cy, nz), (nx, cy - 1, nz)):
            if ctl.block_at(*c) != "air":
                if not (yield from ctl.mine_block(*c, walk=False)):
                    ok = False
                    break
        if not ok:
            continue
        if (yield from _walk_dir(ctl, dx, dz, nx, cy - 1, nz)):
            return True
    return False


def _descend(ctl, target_y, toward=None, budget=400):
    """Staircase down until the feet are at target_y (or below the target is not needed). Returns bool."""
    start = ctl.decisions
    while ctl.cell()[1] > target_y:
        if ctl.decisions - start > budget:
            return False
        yield from _guard(ctl)
        if (yield from _shaft_down(ctl)):
            continue
        if not (yield from _stair_down(ctl, toward)):
            return False
    return True


def _step_flat(ctl, dx, dz):
    """Dig one block forward at feet level and step into it (the floor there must be solid). Returns bool."""
    cx, cy, cz = ctl.cell()
    nx, nz = cx + dx, cz + dz
    if ctl.block_at(nx, cy - 1, nz) in (None, "air") or _bad_near(ctl, [(nx, cy, nz), (nx, cy + 1, nz)]):
        return False
    for c in ((nx, cy, nz), (nx, cy + 1, nz)):
        if ctl.block_at(*c) != "air" and not (yield from ctl.mine_block(*c, walk=False)):
            return False
    return (yield from _walk_dir(ctl, dx, dz, nx, cy, nz, tries=4))


def _tunnel(ctl, dx, dz, length):
    """Dig a flat tunnel of `length` blocks at feet level (exposes the ores of its walls). Returns bool."""
    for _ in range(length):
        yield from _guard(ctl)
        if not (yield from _step_flat(ctl, dx, dz)):
            return False
    return True


def _stair_up(ctl, toward):
    """One staircase step up toward `toward` (x, z): clear the head room above, mine the next feet/head cells,
    then walk up onto them. Returns True when one level higher."""
    cx, cy, cz = ctl.cell()
    for dx, dz in _dir_order(ctl, cx, cz, toward):
        nx, nz = cx + dx, cz + dz
        floor = ctl.block_at(nx, cy, nz)
        if floor in (None, "air") or floor in BAD:
            continue
        side = [(nx + a, cy + 1 + k, nz + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)) for k in (0, 1)]
        if _bad_near(ctl, [(nx, cy + 1, nz), (nx, cy + 2, nz), (cx, cy + 2, cz)] + side):
            continue
        ok = True
        for c in ((cx, cy + 2, cz), (nx, cy + 1, nz), (nx, cy + 2, nz)):
            if ctl.block_at(*c) != "air" and not (yield from ctl.mine_block(*c, walk=False)):
                ok = False
                break
        if not ok:
            continue
        if (yield from _walk_dir(ctl, dx, dz, nx, cy + 1, nz, tries=8)):
            return True
    return False


def _approach(ctl, cell, budget=300):
    """Get within reach of `cell` by digging greedily: vertical staircases first when the cell is more than one
    level away, then flat steps toward it (a block in the way is mined). Works in solid rock where A* digging
    gives up. Returns True when the feet are within 2 blocks horizontally and 1 level of the cell."""
    x, y, z = cell
    start = ctl.decisions
    while ctl.decisions - start < budget:
        cx, cy, cz = ctl.cell()
        if math.hypot(x - cx, z - cz) <= 2.0 and abs(y - cy) <= 1:
            return True
        yield from _guard(ctl)
        if y < cy - 1:
            if (yield from _shaft_down(ctl)):
                continue
            if not (yield from _stair_down(ctl, (x, z))):
                return False
            continue
        if y > cy + 1:
            if not (yield from _stair_up(ctl, (x, z))):
                return False
            continue
        moved = False
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        dirs.sort(key=lambda d: (cx + d[0] - x) ** 2 + (cz + d[1] - z) ** 2)
        for dx, dz in dirs:
            if (cx + dx - x) ** 2 + (cz + dz - z) ** 2 >= (cx - x) ** 2 + (cz - z) ** 2:
                continue    # only steps that bring us closer
            if (yield from _step_flat(ctl, dx, dz)):
                moved = True
                break
        if not moved:
            return False
    return False


def _mine_at(ctl, cell):
    """Break the block at `cell`: from here when visible, else approach it (descend, dig, walk). Returns bool."""
    x, y, z = cell
    if (yield from ctl.mine_block(x, y, z, walk=False)):
        return True
    cy = ctl.cell()[1]
    if cy - y >= 2:
        if not (yield from _descend(ctl, y + 1, toward=(x, z))):
            return False
    if not (yield from _approach(ctl, cell)):
        if not (yield from ctl.walk_to(x, y, z, within=2.5, dig=True, max_decisions=300)):
            return False
    return (yield from ctl.mine_block(x, y, z, walk=False))


def _underground(ctl):
    """Solid rock 2..6 blocks above the head (no sky over the body within the window)."""
    cx, cy, cz = ctl.cell()
    return any(ctl.block_at(cx, cy + k, cz) not in (None, "air") for k in range(2, 7))


def _search(ctl, item):
    """No wanted block in the window: go where it is likely (down to its depth, else tunnel / explore)."""
    depth = DEPTH.get(item)
    if depth is not None and ctl.cell()[1] > depth + 4:
        if (yield from _descend(ctl, depth, budget=200)):
            return True
    if item in SURFACE:
        if _underground(ctl):
            return False     # no trees underground: give up at once (the caller counts the failure)
        return (yield from ctl.explore(min_d=6, max_d=14))
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    ctl.rng.shuffle(dirs)
    for dx, dz in dirs[:2]:
        if (yield from _tunnel(ctl, dx, dz, 8)):
            return True
    return (yield from ctl.explore(min_d=6, max_d=14))


# ---------------------------------------------------------------------- mining and making

def _collect(ctl, item, n, budget=None):
    """Mine until the inventory holds `n` of `item` (SOURCE); `budget` caps the decisions spent. Returns bool."""
    names = SOURCE[item]
    tier = max(BLOCK_INFO[b]["min_tier"] for b in names)
    if tier > 0 and not _have_pick(ctl, tier):
        if not (yield from make(ctl, PICK[tier], 1)):
            return False
    radius = 40 if item in FAR else 16
    blacklist = set()
    fails = 0
    start = ctl.decisions
    while ctl.count(item) < n:
        if fails > MAX_FAILS or (budget is not None and ctl.decisions - start > budget):
            return False
        yield from _guard(ctl)
        cands = [c for c, _ in ctl.find_blocks(list(names), radius=radius) if c not in blacklist]
        # ores touching water (a lake or a pond) are dry-land unreachable: take them last
        dry = [c for c in cands[:60] if not _near_water(ctl, c)]
        cands = dry or cands
        if not cands:
            fails += 1
            yield from _search(ctl, item)
            continue
        cell = cands[0]
        if (yield from _mine_at(ctl, cell)):
            continue
        # a failed target spoils its whole vein: skip the neighbouring cells too
        for c in cands:
            if max(abs(c[0] - cell[0]), abs(c[1] - cell[1]), abs(c[2] - cell[2])) <= 2:
                blacklist.add(c)
        blacklist.add(cell)
        fails += 1
    return True


def _smelt_iron(ctl, need, depth):
    if need <= 0:
        return True
    if not (yield from make(ctl, "raw_iron", need, depth + 1)):
        return False
    if ctl.count("coal") == 0:
        # fuel: coal (8 smelts each) if it is close, else the wood held (one smelt per plank or log)
        wood = ctl.count("oak_planks") + ctl.count("oak_log")
        if wood < need:
            if not (yield from _collect(ctl, "coal", 1, budget=150)) and wood < need:
                return False
    for _ in range(need):
        if not (yield from _ensure_station(ctl, "furnace", depth)):
            return False
        if not (yield from ctl.smelt("raw_iron")):
            return False
    return True


def make(ctl, item, n=1, depth=0):
    """Hold at least `n` of `item`: mine it (SOURCE), smelt it (iron_ingot), or craft it from its recipe.
    Returns bool."""
    have = ctl.count(item)
    if have >= n:
        return True
    if depth > MAX_DEPTH:
        return False
    if item in SOURCE:
        return (yield from _collect(ctl, item, n))
    if item == "iron_ingot":
        return (yield from _smelt_iron(ctl, n - have, depth))
    R = _recipe_for(item)
    if R is None:
        return False
    times = -(-(n - have) // R["count"])
    table = R["needs"] == "crafting_table"
    if table and not (yield from _ensure_station(ctl, "crafting_table", depth)):
        return False
    # ingredients: make each short one; a make can consume another ingredient (sticks eat planks), so repeat
    done = False
    for _ in range(4):
        short = [(ing, c * times) for ing, c in R["ingredients"].items() if ctl.count(ing) < c * times]
        if not short:
            done = True
            break
        for ing, need in short:
            if ctl.count(ing) < need and not (yield from make(ctl, ing, need, depth + 1)):
                return False
    if not done:
        return False
    for _ in range(times):
        if table and not (yield from _ensure_station(ctl, "crafting_table", depth)):
            return False
        if not (yield from ctl.craft(R["name"])):
            return False
    return ctl.count(item) >= n


# ---------------------------------------------------------------------- gather skills

def _stock_wood(ctl, units=16):
    """Hold about `units` planks' worth of wood (logs x4 + planks) while still on the surface with trees near."""
    while ctl.count("oak_log") * 4 + ctl.count("oak_planks") < units:
        if not (yield from _collect(ctl, "oak_log", ctl.count("oak_log") + 1)):
            return False
    return True


def _count(task, default):
    return int(task["params"].get("count", default))


@skill("gather_sand")
def gather_sand(ctl, task):
    """Dig up sand (hand is enough; no tool needed)."""
    yield from _collect(ctl, "sand", _count(task, 8))


@skill("gather_stone")
def gather_stone(ctl, task):
    """Mine stone with a wooden pickaxe (stone drops cobblestone; the mined count is the sim's)."""
    yield from _collect(ctl, "cobblestone", _count(task, 8))


@skill("gather_gravel")
def gather_gravel(ctl, task):
    """Dig gravel (flint drops 10%); the wooden pickaxe is made first, the sim checks the 8 mined gravel blocks."""
    yield from make(ctl, "wooden_pickaxe", 1)
    yield from _collect(ctl, "gravel", _count(task, 8))


@skill("gather_coal")
def gather_coal(ctl, task):
    """Collect coal (wooden pickaxe)."""
    yield from _collect(ctl, "coal", _count(task, 8))


@skill("gather_iron")
def gather_iron(ctl, task):
    """Mine iron ore with a stone pickaxe (the chain is made first)."""
    yield from _stock_wood(ctl, 20)
    yield from make(ctl, "stone_pickaxe", 1)
    yield from _collect(ctl, "raw_iron", _count(task, 6))


@skill("gather_diamond")
def gather_diamond(ctl, task):
    """Find a diamond: iron pickaxe (iron ingots from the furnace), then staircase down to the diamond band."""
    yield from _stock_wood(ctl, 28)
    yield from make(ctl, "diamond", _count(task, 1))


@skill("gather_obsidian")
def gather_obsidian(ctl, task):
    """Best effort. Obsidian needs a diamond pickaxe and a source: the overworld has no lava and no natural
    obsidian (only the nether portal blueprint), so this searches the window for obsidian after the diamond
    pickaxe is made and most likely gives up."""
    yield from _stock_wood(ctl, 28)
    if not (yield from make(ctl, "diamond_pickaxe", 1)):
        return
    yield from _collect(ctl, "obsidian", _count(task, 10))


# ---------------------------------------------------------------------- tools skills

@skill("tools_wooden_pickaxe")
def tools_wooden_pickaxe(ctl, task):
    yield from make(ctl, "wooden_pickaxe", 1)


@skill("tools_wooden_axe")
def tools_wooden_axe(ctl, task):
    yield from make(ctl, "wooden_axe", 1)


@skill("tools_stone_pickaxe")
def tools_stone_pickaxe(ctl, task):
    yield from make(ctl, "stone_pickaxe", 1)


@skill("tools_stone_axe")
def tools_stone_axe(ctl, task):
    yield from make(ctl, "stone_axe", 1)


@skill("tools_furnace")
def tools_furnace(ctl, task):
    yield from make(ctl, "furnace", 1)


@skill("tools_iron_pickaxe")
def tools_iron_pickaxe(ctl, task):
    yield from _stock_wood(ctl, 20)
    yield from make(ctl, "iron_pickaxe", 1)


@skill("tools_iron_axe")
def tools_iron_axe(ctl, task):
    yield from _stock_wood(ctl, 20)
    yield from make(ctl, "iron_axe", 1)


@skill("tools_diamond_pickaxe")
def tools_diamond_pickaxe(ctl, task):
    yield from _stock_wood(ctl, 28)
    yield from make(ctl, "diamond_pickaxe", 1)
