"""EXP-4 skills: build, explore and nether families (PROTOCOL 14.1 / 14.4).

Builds: the target comes from `CraftSim.build_target(w)` (read after one noop, when the sim has placed it).
Materials: dirt is dug here; every other item goes through gather_tools.make (mine / smelt / craft).
Placement: a cell is placed when it has a solid face neighbour; stand spots are searched and walked to.
Climbing: when the walk-reachable stands cannot reach the upper cells, a two-column dirt ladder is built
beside the footprint (outside the blueprint box, so it never occupies blueprint air cells).

Portals: the nether_portal frame is built from the target (or on a flat site chosen by the expert when the task
has no target), lit with flint and steel from a frame face, and the agent stands in the interior for 25
decisions (nether.cpp PORTAL_DECISIONS = 20).
"""
import json
import math
import os

import numpy as np

import craftsim as cs
from experts.core import AIR, BID, IID, ITEM_INFO, _SOLID, standable
from experts.skills import skill
from experts.skills import gather_tools as gt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BP_PATH = os.path.join(ROOT, "data", "builds", "blueprints.json")
RADIUS = 14                       # Chebyshev search radius for dug blocks
MARGIN = 4                        # spare blocks per build item (stray placements), except obsidian
NEIGH = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
DIG_ONLY = {"dirt": ("dirt", "grass"), "netherrack": ("netherrack",), "quartz": ("nether_quartz_ore",),
            "glowstone_dust": ("glowstone",), "soul_sand": ("soul_sand",)}


# ---------------------------------------------------------------- blueprints (portal site only)

def _load_blueprint(name):
    with open(BP_PATH, encoding="utf-8") as f:
        bp = next(b for b in json.load(f) if b["name"] == name)
    sx, sy, sz = bp["size"]
    solids = [(x, y, z, bp["palette"][ch]) for y, layer in enumerate(bp["layers"])
              for z, row in enumerate(layer) for x, ch in enumerate(row) if ch != "."]
    grid = {(x, y, z) for x, y, z, _ in solids}

    def enclosed(x, y, z):
        for ax, ay, az in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            lo = hi = False
            for sgn in (-1, 1):
                k = 1
                while True:
                    p = (x + sgn * ax * k, y + sgn * ay * k, z + sgn * az * k)
                    if not (0 <= p[0] < sx and 0 <= p[1] < sy and 0 <= p[2] < sz):
                        break
                    if p in grid:
                        if sgn < 0:
                            lo = True
                        else:
                            hi = True
                        break
                    k += 1
            if lo and hi:
                return True
        return False

    airs = [(x, y, z) for y in range(sy) for z in range(sz) for x in range(sx)
            if (x, y, z) not in grid and enclosed(x, y, z)]
    return {"sx": sx, "sy": sy, "sz": sz, "solids": solids, "airs": airs}


def _place_site(bpd, ox, oy, oz):
    """Target dict for the blueprint at footprint corner (ox, oy, oz), rotation 0."""
    solids = [(ox + x, oy + y, oz + z, item) for x, y, z, item in bpd["solids"]]
    airs = [(ox + x, oy + y, oz + z) for x, y, z in bpd["airs"]]
    return {"solids": solids, "airs": airs, "bbox": (ox, oz, ox + bpd["sx"] - 1, oz + bpd["sz"] - 1),
            "sy": bpd["sy"]}


# ---------------------------------------------------------------- helpers

def _excl_box(box):
    """Mining exclusion: the footprint box plus a 1-cell margin stays untouched."""
    if box is None:
        return lambda c: False
    x0, z0, x1, z1 = box
    return lambda c: x0 - 1 <= c[0] <= x1 + 1 and z0 - 1 <= c[2] <= z1 + 1


def _ensure_progress(ctl, d0):
    if ctl.decisions == d0:
        yield from ctl.wait(1)


def _is_solid(reg, x, y, z):
    b = reg.get(x, y, z)
    return b > 0 and _SOLID[b]


def _pit(c, fx, fy, fz):
    """Digging here would drop the agent: the cell under its feet or anything more than one block below."""
    return c[1] < fy - 1 or (c[1] == fy - 1 and c[0] == fx and c[2] == fz)


def _mine_for(ctl, item, n, excl, max_mines=600):
    names = DIG_ONLY[item]
    blacklist, fails, mines = set(), 0, 0
    while ctl.count(item) < n:
        if fails > 60 or mines > max_mines:
            return False
        d0 = ctl.decisions
        fx, fy, fz = ctl.cell()
        sites = [c for c, _ in ctl.find_blocks(names, radius=RADIUS)
                 if c not in blacklist and not excl(c) and not _pit(c, fx, fy, fz)]
        if not sites:
            fails += 1
            yield from ctl.explore()
            yield from _ensure_progress(ctl, d0)
            continue
        ok = yield from ctl.mine_block(*sites[0], walk=True)
        mines += 1
        if not ok:
            blacklist.add(sites[0])
            fails += 1
        yield from _ensure_progress(ctl, d0)
    return True


def obtain(ctl, item, n, excl=None):
    """Hold at least n of item: dig (dirt, netherrack, ...) or make it (gather_tools.make)."""
    if ctl.count(item) >= n:
        return True
    if item in DIG_ONLY:
        return (yield from _mine_for(ctl, item, n, excl or (lambda c: False)))
    return (yield from gt.make(ctl, item, n))


def _place(ctl, item, cell, solid_cells):
    """place_block from where the agent is; else walk to a nearby stand (never a target cell, its own cell
    excluded) from which the placement face is visible, then place from there."""
    if (yield from ctl.place_block(item, *cell, walk=False)):
        return True
    reg = ctl.region(24)
    cx, cy, cz = cell
    fx, fy, fz = ctl.cell()
    best = None
    for dx in range(-4, 5):
        for dz in range(-4, 5):
            for dy in range(-3, 2):
                x, y, z = cx + dx, cy + dy, cz + dz
                if (x, y, z) in solid_cells or (x, y, z) == (fx, fy, fz) or not reg.standable(x, y, z):
                    continue
                if (x - cx) ** 2 + (z - cz) ** 2 > 12.0:
                    continue
                if ctl._place_plan(reg, cell, eye=(x + 0.5, y + 1.62, z + 0.5), feet=(x + 0.5, y, z + 0.5)) is None:
                    continue
                dd = (x - fx) ** 2 + (y - fy) ** 2 + (z - fz) ** 2
                if best is None or dd < best[0]:
                    best = (dd, (x, y, z))
    if best is not None:
        yield from ctl.walk_to(*best[1], max_decisions=120)
    return (yield from ctl.place_block(item, *cell, walk=True))


def _clear(ctl, cells):
    """Mine obstacles in the volume: solid cells holding another block, and anything in required-air cells."""
    for x, y, z, item in cells:
        d0 = ctl.decisions
        reg = ctl.region(24)
        b = reg.get(x, y, z)
        if b in (AIR, -1):
            continue
        if item is not None and cs.BLOCKS[b] == item:
            continue
        yield from ctl.mine_block(x, y, z, walk=True)
        yield from _ensure_progress(ctl, d0)


# ---------------------------------------------------------------- target

def _target(ctl):
    """Read the sim's build target after one noop (the target is placed on the first obs of the episode)."""
    yield from ctl.wait(1)
    t = ctl.sim.build_target(ctl.world)
    cells = t["cells"]
    if not cells:
        return None
    solids = [(x, y, z, cs.BLOCKS[b]) for x, y, z, b, _ in cells if b]
    airs = [(x, y, z) for x, y, z, b, _ in cells if not b]
    xs = [c[0] for c in cells]
    zs = [c[2] for c in cells]
    return {"solids": solids, "airs": airs, "bbox": (min(xs), min(zs), max(xs), max(zs)),
            "blueprint": t["blueprint"], "sy": max(y for _, y, _, _ in solids) - min(y for _, y, _, _ in solids) + 1}


def _required(tgt):
    req = {}
    for _, _, _, item in tgt["solids"]:
        req[item] = req.get(item, 0) + 1
    return req


# ---------------------------------------------------------------- building

def _step_off(ctl, cells):
    """If the agent's feet are inside the target volume, walk to the nearest standable cell outside it."""
    fx, fy, fz = ctl.cell()
    if (fx, fy, fz) not in cells and (fx, fy + 1, fz) not in cells:
        return
    reg = ctl.region(24)

    def goal(node, r):
        return standable(r, *node) and node not in cells and (node[0], node[1] + 1, node[2]) not in cells
    yield from ctl._walk(goal, fx, fz, max_decisions=60)


def build_passes(ctl, tgt):
    """Place unfilled solid cells bottom-up (nearest first in a layer), each one from a face-visible stand.
    A pass that places nothing ends the build. Returns True when every solid cell holds its block."""
    solids, airs = tgt["solids"], tgt["airs"]
    solid_cells = {(x, y, z) for x, y, z, _ in solids}
    yield from _clear(ctl, [(x, y, z, None) for x, y, z in airs] + [(x, y, z, it) for x, y, z, it in solids])
    tries = {}
    dry = 0
    while dry < 3:
        yield from _clear(ctl, [(x, y, z, None) for x, y, z in airs])   # strays in required-air cells
        yield from _step_off(ctl, solid_cells | set(airs))
        reg = ctl.region(24)
        px, py, pz = ctl.pos()
        todo = []
        for x, y, z, item in solids:
            b = reg.get(x, y, z)
            if b == -1 or b == BID[item]:
                continue
            todo.append((y, (x + 0.5 - px) ** 2 + (z + 0.5 - pz) ** 2, x, y, z, item))
        if not todo:
            return True
        todo.sort()
        placed = False
        for _y, _d, x, y, z, item in todo:
            reg = ctl.region(24)
            if reg.get(x, y, z) != AIR or tries.get((x, y, z), 0) >= 8:
                continue
            if not any(_is_solid(reg, x + dx, y + dy, z + dz) for dx, dy, dz in NEIGH):
                continue
            if ctl.count(item) <= 0 and not (yield from obtain(ctl, item, 1)):
                return False
            tries[(x, y, z)] = tries.get((x, y, z), 0) + 1
            d0 = ctl.decisions
            ok = yield from _place(ctl, item, (x, y, z), solid_cells)
            yield from _ensure_progress(ctl, d0)
            if ok:
                placed = True
                break
        dry = 0 if placed else dry + 1
    return False


def _walk(ctl, x, y, z, budget=120):
    return (yield from ctl.walk_to(x, y, z, max_decisions=budget))


def _ladder_sides(ctl, tgt, top_feet):
    """Feasible ladder placements around the footprint, cheapest first: [(A, C, F0)] where A is the column
    touching the footprint, C its neighbour along the wall and F0 the feet level on the ground."""
    x0, z0, x1, z1 = tgt["bbox"]
    reg = ctl.region(24)
    options = [
        ((x1 + 1, z0), (x1 + 1, z0 + 1)),
        ((x1 + 1, z1 - 1), (x1 + 1, z1)),
        ((x0 - 1, z0), (x0 - 1, z0 + 1)),
        ((x0 - 1, z1 - 1), (x0 - 1, z1)),
        ((x0, z1 + 1), (x0 + 1, z1 + 1)),
        ((x1 - 1, z1 + 1), (x1, z1 + 1)),
        ((x0, z0 - 1), (x0 + 1, z0 - 1)),
        ((x1 - 1, z0 - 1), (x1, z0 - 1)),
    ]
    out = []
    for A, C in options:
        ga = [y for y in range(top_feet + 4, top_feet - 40, -1) if _is_solid(reg, A[0], y - 1, A[1]) and
              reg.get(A[0], y, A[1]) == AIR]
        gc = [y for y in range(top_feet + 4, top_feet - 40, -1) if _is_solid(reg, C[0], y - 1, C[1]) and
              reg.get(C[0], y, C[1]) == AIR]
        if not ga or not gc or ga[0] != gc[0]:
            continue
        f0 = ga[0]
        if f0 >= top_feet or f0 < top_feet - 30:
            continue
        out.append((top_feet - f0, A, C, f0))
    out.sort()
    return [(A, C, f0) for _, A, C, f0 in out]


def _approach(ctl, ax, y, az):
    """Walk to a standable cell within 3 blocks of the ladder start (the window may not reach it from far)."""
    def goal(node, reg):
        return (standable(reg, *node) and abs(node[1] - y) <= 1 and (node[0] - ax) ** 2 + (node[2] - az) ** 2 <= 9)
    if goal(ctl.cell(), ctl.region()):
        return True
    return (yield from ctl._walk(goal, ax, az, max_decisions=300))


def _ladder(ctl, A, C, f0, top_feet):
    """Two columns grow together: stand on A at feet h, place C_h beside you, step on it, place A_h from C,
    step back onto A (feet h+1). Each level costs two placements. Returns bool."""
    ax, az = A
    cx, cz = C
    if not (yield from _approach(ctl, ax, f0, az)):
        return False
    if not (yield from _walk(ctl, ax, f0, 200)):
        return False
    h = f0
    while h < top_feet:
        if not (yield from _walk(ctl, ax, h)):
            return False
        d0 = ctl.decisions
        if not (yield from _place(ctl, "dirt", (cx, h, cz), set())):
            return False
        yield from _ensure_progress(ctl, d0)
        if not (yield from _walk(ctl, cx, h + 1)):
            return False
        d0 = ctl.decisions
        if not (yield from _place(ctl, "dirt", (ax, h, az), set())):
            return False
        yield from _ensure_progress(ctl, d0)
        if not (yield from _walk(ctl, ax, h + 1)):
            return False
        h += 1
    return True


def build_skill(ctl, tgt):
    """Gather the target's materials, place it from the ground, climb a ladder and finish the upper cells."""
    excl = _excl_box(tgt["bbox"])
    ladder = 2 * (tgt["sy"] + 6) + 4 if "sy" in tgt else 2 * 16 + 4
    for item, n in _required(tgt).items():
        m = 0 if item == "obsidian" else MARGIN
        extra = ladder if item == "dirt" else 0      # ladder blocks are taken now: later digs stay near the site
        if not (yield from obtain(ctl, item, n + m + extra, excl)):
            if m == 0 and extra == 0 or not (yield from obtain(ctl, item, n, excl)):
                return False
    top_feet = max(y for _, y, _, _ in tgt["solids"]) + 1
    if "dirt" not in _required(tgt) and tgt.get("sy", 1) >= 4:
        # tall builds of other materials still climb a dirt ladder: take its dirt before the passes
        yield from obtain(ctl, "dirt", ladder, excl)
    if (yield from build_passes(ctl, tgt)):
        return True
    tried = 0
    for A, C, f0 in _ladder_sides(ctl, tgt, top_feet):
        if tried >= 4:
            break
        tried += 1
        if not (yield from _ladder(ctl, A, C, f0, top_feet)):
            continue
        if (yield from build_passes(ctl, tgt)):
            return True
    return False


def _build_task(ctl, bp):
    tgt = yield from _target(ctl)
    if tgt is None:
        return False
    return (yield from build_skill(ctl, tgt))


# ---------------------------------------------------------------- skills: build

@skill("build_dirt_hut")
def build_dirt_hut(ctl, task):
    yield from _build_task(ctl, "dirt_hut")


@skill("build_wall_segment")
def build_wall_segment(ctl, task):
    yield from _build_task(ctl, "wall_segment")


@skill("build_farm_plot")
def build_farm_plot(ctl, task):
    yield from _build_task(ctl, "farm_plot")


@skill("build_bridge")
def build_bridge(ctl, task):
    yield from _build_task(ctl, "bridge")


@skill("build_watchtower")
def build_watchtower(ctl, task):
    yield from _build_task(ctl, "watchtower")


@skill("build_cobble_house")
def build_cobble_house(ctl, task):
    yield from _build_task(ctl, "cobble_house")


@skill("build_wooden_house")
def build_wooden_house(ctl, task):
    yield from _build_task(ctl, "wooden_house")


@skill("build_storage_chest")
def build_storage_chest(ctl, task):
    """Craft a chest and place it on a free cell with a floor (placing any chest succeeds the task)."""
    if not (yield from obtain(ctl, "chest", 1)):
        return
    for _ in range(12):
        reg = ctl.region()
        cx, cy, cz = ctl.cell()
        cells = [(cx + dx, cy, cz + dz) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2))
                 if reg.get(cx + dx, cy, cz + dz) == AIR and _is_solid(reg, cx + dx, cy - 1, cz + dz)]
        if cells:
            d0 = ctl.decisions
            if (yield from ctl.place_block("chest", *cells[0], walk=False)):
                return
            yield from _ensure_progress(ctl, d0)
        yield from ctl.explore(min_d=2, max_d=4)


# ---------------------------------------------------------------- skills: explore

def _walk_far(ctl):
    """Keep walking over open ground; the sim ends the episode on the distance predicate."""
    while True:
        d0 = ctl.decisions
        yield from ctl.explore(min_d=8, max_d=16)
        yield from _ensure_progress(ctl, d0)


@skill("explore_walk_100")
def explore_walk_100(ctl, task):
    yield from _walk_far(ctl)


@skill("explore_walk_500")
def explore_walk_500(ctl, task):
    yield from _walk_far(ctl)


@skill("explore_deep_mine")
def explore_deep_mine(ctl, task):
    """Stair down a shaft: mine the block below, step into it, repeat until y <= max_y (the sim ends it)."""
    max_y = int(task["params"].get("max_y", 16))
    if not (yield from gt.make(ctl, "stone_pickaxe", 1)):
        return
    fails = 0
    while fails < 30:
        x, y, z = ctl.cell()
        if y <= max_y:
            return
        d0 = ctl.decisions
        ok = yield from ctl.mine_block(x, y - 1, z, walk=False)
        if ok:
            yield from ctl.walk_to(x, y - 1, z, max_decisions=20)
        else:
            fails += 1
        yield from _ensure_progress(ctl, d0)


# ---------------------------------------------------------------- skills: nether

def _flat_site(ctl, bpd):
    """A flat site for the portal frame near the agent: 4 ground columns at one height, air above.
    Returns (ox, oy, oz) or None."""
    reg = ctl.region(16)
    cx, cy, cz = ctl.cell()
    best = None
    for dz in range(-8, 9):
        for dx in range(-8, 9):
            ox, oz = cx + dx, cz + dz
            if ox < -16 or oz < -16:
                continue
            ground = []
            for i in range(bpd["sx"]):
                col = [y for y in range(cy + 4, cy - 6, -1) if _is_solid(reg, ox + i, y, oz)]
                if not col:
                    break
                ground.append(col[0])
            if len(ground) != bpd["sx"] or len(set(ground)) != 1:
                continue
            oy = ground[0] + 1
            if any(reg.get(ox + i, oy + y, oz) not in (AIR,) for i in range(bpd["sx"]) for y in range(bpd["sy"])):
                continue
            d = dx * dx + dz * dz
            if best is None or d < best[0]:
                best = (d, (ox, oy, oz))
    return None if best is None else best[1]


def _portal_target(ctl):
    """(target, lit-frame info). Uses the sim's nether_portal target when there is one, else a chosen site."""
    tgt = yield from _target(ctl)
    if tgt is not None and tgt.get("blueprint") == -1 or tgt is None:
        return None
    if tgt.get("blueprint") is not None and any(it == "obsidian" for _, _, _, it in tgt["solids"]):
        return tgt
    return None


def _light(ctl, tgt):
    """Light the frame with flint and steel: aim at a frame cell whose neighbour is interior air."""
    frame = [(x, y, z) for x, y, z, _ in tgt["solids"]]
    air = set(tgt["airs"])
    for fx, fy, fz in sorted(frame):
        for dx, dy, dz in NEIGH:
            I = (fx + dx, fy + dy, fz + dz)
            if I not in air:
                continue
            yield from ctl.equip("flint_and_steel")
            accept = (lambda h, F=(fx, fy, fz), J=I: h is not None and h[0] == F and h[1] == J)
            ok = yield from ctl.aim_at((fx + 0.5, fy + 0.5, fz + 0.5), accept=accept)
            if ok:
                yield from ctl._act("use", item=IID["flint_and_steel"] + 1)
                yield from ctl.wait(1)
                return True
    return False


def _enter_portal(ctl, tgt):
    """Light the portal, then stand in its interior until the team travels. Returns bool (dim check left to
    the sim)."""
    if not (yield from _light(ctl, tgt)):
        return False
    interior = sorted(tgt["airs"])
    for _ in range(4):
        for ix, iy, iz in interior:
            yield from ctl.walk_to(ix, iy, iz, max_decisions=80)
            break
        yield from ctl.wait(25)
        if ctl.state().get("dim") == 1:
            return True
    return ctl.state().get("dim") == 1


def _portal_frame(ctl, need_obsidian=True):
    """Build the nether_portal frame: the sim target when present, else a flat site beside the agent."""
    bpd = _load_blueprint("nether_portal")
    tgt = yield from _target(ctl)
    if tgt is None or tgt.get("blueprint", -1) == -1 or not tgt["solids"]:
        site = _flat_site(ctl, bpd)
        if site is None:
            return None
        tgt = _place_site(bpd, *site)
    tgt["solids"] = [(x, y, z, "obsidian") for x, y, z, _ in tgt["solids"]]
    if not (yield from obtain(ctl, "obsidian", 14)):
        return None
    if not (yield from build_skill(ctl, tgt)):
        return None
    return tgt


@skill("nether_build_portal")
def nether_build_portal(ctl, task):
    yield from _portal_frame(ctl)


@skill("nether_light_portal")
def nether_light_portal(ctl, task):
    tgt = yield from _portal_frame(ctl)
    if tgt is None:
        return
    if not (yield from obtain(ctl, "flint_and_steel", 1)):
        return
    yield from _light(ctl, tgt)


@skill("nether_enter")
def nether_enter(ctl, task):
    tgt = yield from _portal_frame(ctl)
    if tgt is None:
        return
    if not (yield from obtain(ctl, "flint_and_steel", 1)):
        return
    yield from _enter_portal(ctl, tgt)


def _in_nether(ctl):
    """Enter the nether through a portal built from the kit, unless already there. Returns bool."""
    if ctl.state().get("dim") == 1:
        return True
    tgt = yield from _portal_frame(ctl)
    if tgt is None:
        return False
    if not (yield from obtain(ctl, "flint_and_steel", 1)):
        return False
    return (yield from _enter_portal(ctl, tgt))


def _nether_mine(ctl, item, count):
    if not (yield from _in_nether(ctl)):
        return
    yield from obtain(ctl, item, count)


@skill("nether_gather_netherrack")
def nether_gather_netherrack(ctl, task):
    yield from _nether_mine(ctl, "netherrack", int(task["params"].get("count", 16)))


@skill("nether_gather_quartz")
def nether_gather_quartz(ctl, task):
    yield from _nether_mine(ctl, "quartz", int(task["params"].get("count", 4)))


@skill("nether_glowstone")
def nether_glowstone(ctl, task):
    yield from _nether_mine(ctl, "glowstone_dust", int(task["params"].get("count", 4)))


@skill("nether_soul_sand")
def nether_soul_sand(ctl, task):
    yield from _nether_mine(ctl, "soul_sand", int(task["params"].get("count", 4)))


@skill("nether_resources")
def nether_resources(ctl, task):
    if not (yield from _in_nether(ctl)):
        return
    yield from obtain(ctl, "quartz", 2)
    yield from obtain(ctl, "glowstone_dust", 2)


@skill("nether_kill_piglin")
def nether_kill_piglin(ctl, task):
    if not (yield from _in_nether(ctl)):
        return
    yield from ctl.attack_entity("zombified_piglin")


@skill("nether_flint_and_steel")
def nether_flint_and_steel(ctl, task):
    yield from gt.make(ctl, "flint_and_steel", 1)


@skill("explore_nether_walk")
def explore_nether_walk(ctl, task):
    if not (yield from _in_nether(ctl)):
        return
    yield from _walk_far(ctl)
