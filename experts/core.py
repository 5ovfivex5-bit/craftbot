"""Expert primitives for the v7 CraftSim (PROTOCOL 14.1 / 14.2). API reference: experts/README.md.

Every primitive is a generator. Each decision it yields one action row (primary, recipe column,
item column, msg column). The caller steps the sim with that row, then resumes the generator: the
world is then fresh (state, region and inventory are re-read). Call primitives with `yield from`
and read their return value:

    ok = yield from ctl.walk_to(x, y, z)

Only the v7 action columns are used; no pose, inventory or block edits (PROTOCOL 14.1).
"""
import heapq
import json
import math
import os
import random

import numpy as np

import craftsim as cs

EYE = 1.62            # eye height above the feet (sim.cpp EYE)
HALF_W = 0.3          # half body width (sim.cpp HALF_W)
HEIGHT = 1.8          # body height (sim.cpp HEIGHT)
REACH = 4.5           # block reach of the crosshair ray (sim.cpp REACH)
TURN = 15.0           # degrees per turn/look decision (sim.cpp TURN_DEG)
AIM_REACH = 4.2       # planned aims and stand spots keep this margin under REACH
WALK_STUCK = 25       # decisions without changing cell before a walk gives up

_REG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "registry")


def _load_json_path(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_json(name):
    return _load_json_path(os.path.join(_REG_DIR, name))


BLOCK_INFO = {b["name"]: b for b in _load_json("blocks.json")}
ITEM_INFO = {it["name"]: it for it in _load_json("items.json")}
RECIPES = _load_json("recipes.json")          # dicts: id, name, result, count, needs, ingredients {name: n}
RECIPE_BY_NAME = {r["name"]: r for r in RECIPES}
TASK_INFO = {t["id"]: t for t in _load_json_path(os.path.join(os.path.dirname(_REG_DIR), "tasks", "tasks.json"))}

PRIM = {n: i for i, n in enumerate(cs.PRIMARY_NAMES)}
BID = {n: i for i, n in enumerate(cs.BLOCKS)}
IID = {n: i for i, n in enumerate(cs.ITEMS)}
AIR = BID["air"]
WATER = BID["water"]
LAVA = BID["lava"]
BEDROCK = BID["bedrock"]
# sim.cpp solid(): every block except air, water and lava blocks the body and the crosshair ray.
_SOLID = [b not in (AIR, WATER, LAVA) for b in range(cs.NUM_BLOCKS)]
_SOLID_NP = np.array(_SOLID, dtype=bool)
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIRS6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _diggable(b):
    return b > 0 and b not in (WATER, LAVA, BEDROCK)


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def _look_dir(yaw, pitch):
    """Crosshair direction, identical to sim.cpp lookDir."""
    y = math.radians(yaw)
    p = math.radians(pitch)
    return (-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p))


def _look_angles(dx, dy, dz):
    """Inverse of _look_dir: (yaw, pitch) in degrees, yaw in [0, 360), pitch positive = looking down."""
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-9:
        return 0.0, 0.0
    yaw = math.degrees(math.atan2(-dx, dz)) % 360.0
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -dy / n))))
    return yaw, pitch


def _yaw_of(dx, dz):
    return math.degrees(math.atan2(-dx, dz)) % 360.0


class Region:
    """Block window around an agent: arr[y - oy][z - oz][x - ox] (block ids). Outside: get() == -1.
    `stand` (feet cell over a solid floor with two air cells) is computed once, vectorised."""
    __slots__ = ("ox", "oy", "oz", "arr", "sx", "sy", "sz", "_stand")

    def __init__(self, origin, arr):
        self.ox, self.oy, self.oz = (int(v) for v in origin)
        self.arr = arr
        self.sy, self.sz, self.sx = arr.shape
        self._stand = None

    def get(self, x, y, z):
        ix = x - self.ox
        iy = y - self.oy
        iz = z - self.oz
        if 0 <= ix < self.sx and 0 <= iy < self.sy and 0 <= iz < self.sz:
            return int(self.arr[iy, iz, ix])
        return -1

    def stand_mask(self):
        if self._stand is None:
            a = self.arr
            air = a == AIR
            solid = _SOLID_NP[a]
            st = np.zeros(a.shape, bool)
            if self.sy > 2:
                st[1:self.sy - 1] = air[1:self.sy - 1] & air[2:self.sy] & solid[0:self.sy - 2]
            self._stand = st
        return self._stand

    def standable(self, x, y, z):
        ix = x - self.ox
        iy = y - self.oy
        iz = z - self.oz
        if 0 <= ix < self.sx and 0 <= iy < self.sy and 0 <= iz < self.sz:
            return bool(self.stand_mask()[iy, iz, ix])
        return False


def standable(reg, x, y, z):
    """Feet at (x, y, z): two air cells (feet, head) over a solid floor."""
    return reg.standable(x, y, z)


def _body_overlaps(px, py, pz, cell):
    """Does a body with feet at (px, py, pz) overlap the block cell? Same test as sim.cpp placeBlock."""
    x, y, z = cell
    return (x < px + HALF_W and x + 1 > px - HALF_W and y < py + HEIGHT and y + 1 > py and
            z < pz + HALF_W and z + 1 > pz - HALF_W)


def _station_near(reg, cell, bid):
    """sim.cpp nearBlock: a block `bid` within dx,dz in [-3, 3] and dy in [-3, 4] of the feet cell."""
    cx, cy, cz = cell
    x0, x1 = max(cx - 3 - reg.ox, 0), min(cx + 4 - reg.ox, reg.sx)
    y0, y1 = max(cy - 3 - reg.oy, 0), min(cy + 5 - reg.oy, reg.sy)
    z0, z1 = max(cz - 3 - reg.oz, 0), min(cz + 4 - reg.oz, reg.sz)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return False
    return bool(np.any(reg.arr[y0:y1, z0:z1, x0:x1] == bid))


def ray_hit(reg, o, d, maxd=REACH):
    """First non-air block on the ray from o along d (same DDA as sim.cpp raycast).
    Returns (cell, previous_cell) or None. previous_cell is the air cell the ray entered the hit from
    (where place_block puts a block). Rays leaving the window return None."""
    ox, oy, oz = o
    dx, dy, dz = d
    x, y, z = math.floor(ox), math.floor(oy), math.floor(oz)
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    sz = 1 if dz > 0 else -1
    inf = float("inf")
    tdx = abs(1.0 / dx) if dx != 0 else inf
    tdy = abs(1.0 / dy) if dy != 0 else inf
    tdz = abs(1.0 / dz) if dz != 0 else inf
    tmx = ((x + 1 - ox) if dx > 0 else (ox - x)) * tdx if dx != 0 else inf
    tmy = ((y + 1 - oy) if dy > 0 else (oy - y)) * tdy if dy != 0 else inf
    tmz = ((z + 1 - oz) if dz > 0 else (oz - z)) * tdz if dz != 0 else inf
    lx, ly, lz = x, y, z
    # sim.cpp target(): water stops the ray unless the eye is submerged (then water is see-through)
    pass_water = reg.get(x, y, z) == WATER
    while True:
        b = reg.get(x, y, z)
        if b < 0:
            return None
        if b != AIR and not (pass_water and b == WATER):
            return (x, y, z), (lx, ly, lz)
        lx, ly, lz = x, y, z
        if tmx < tmy and tmx < tmz:
            t = tmx
            x += sx
            tmx += tdx
        elif tmy < tmz:
            t = tmy
            y += sy
            tmy += tdy
        else:
            t = tmz
            z += sz
            tmz += tdz
        if t > maxd:
            return None


def _moves(reg, x, y, z, nx, nz, dig):
    """Ways to enter column (nx, nz) from (x, y, z): list of (new feet y, cost)."""
    out = []
    flat = standable(reg, nx, y, nz)
    if flat:
        out.append((y, 1.0))
    # step up one block (jump): head room above the origin and a standable cell one higher
    if reg.get(x, y + 2, z) == AIR and standable(reg, nx, y + 1, nz):
        out.append((y + 1, 1.6))
    # walk off a ledge of 1..3 blocks
    if reg.get(nx, y, nz) == AIR:
        for k in (1, 2, 3):
            if reg.get(nx, y - k + 1, nz) != AIR:
                break
            if standable(reg, nx, y - k, nz):
                out.append((y - k, 1.0 + 0.5 * k))
                break
    if dig and not flat:
        # dig through the feet/head cells of a flat neighbour
        f = reg.get(nx, y - 1, nz)
        feet = reg.get(nx, y, nz)
        head = reg.get(nx, y + 1, nz)
        if (f > 0 and _SOLID[f] and f not in (WATER, LAVA) and feet >= 0 and head >= 0 and
                (feet == AIR or _diggable(feet)) and (head == AIR or _diggable(head))):
            out.append((y, 1.0 + 4.0 * (feet != AIR) + 4.0 * (head != AIR)))
        # dig a staircase: up one block into the neighbour column (feet and head cells mined)
        if reg.get(x, y + 2, z) >= 0 and (reg.get(x, y + 2, z) == AIR or _diggable(reg.get(x, y + 2, z))):
            f1 = reg.get(nx, y, nz)
            feet1 = reg.get(nx, y + 1, nz)
            head1 = reg.get(nx, y + 2, nz)
            if (f1 > 0 and _SOLID[f1] and f1 not in (WATER, LAVA) and feet1 >= 0 and head1 >= 0 and
                    (feet1 == AIR or _diggable(feet1)) and (head1 == AIR or _diggable(head1))):
                out.append((y + 1, 1.6 + 4.0 * (feet1 != AIR) + 4.0 * (head1 != AIR)
                            + 4.0 * (reg.get(x, y + 2, z) != AIR)))
    return out


def astar(reg, start, goal_fn, hx, hz, dig=False, max_nodes=1200):
    """Weighted A* over standable feet cells of `reg`. goal_fn(node, reg) -> bool. Heuristic: Manhattan to (hx, hz).
    Moves: 4 horizontal neighbours; flat, up 1 (jump), down 1..3 (ledge); dig=True also allows clearing
    the feet/head cells of a flat neighbour (cost 4 per block). Returns [start, ..., goal] or None."""
    if goal_fn(start, reg):
        return [start]
    w = 1.6   # weighted A*: a little longer paths for far fewer expansions
    heap = [(w * (abs(start[0] - hx) + abs(start[2] - hz)), 0.0, 0, start)]
    g = {start: 0.0}
    parent = {start: None}
    counter = 0
    expanded = 0
    while heap:
        _, gc, _, node = heapq.heappop(heap)
        if gc > g[node] + 1e-9:
            continue
        if goal_fn(node, reg):
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]
        expanded += 1
        if expanded > max_nodes:
            return None
        x, y, z = node
        for dx, dz in _DIRS4:
            nx, nz = x + dx, z + dz
            for ny, cost in _moves(reg, x, y, z, nx, nz, dig):
                nn = (nx, ny, nz)
                ng = gc + cost
                if ng < g.get(nn, float("inf")):
                    g[nn] = ng
                    parent[nn] = node
                    counter += 1
                    heapq.heappush(heap, (ng + w * (abs(nx - hx) + abs(nz - hz)), ng, counter, nn))
    return None




class Ctl:
    """Controller of one agent row (agent index i = world * K + k) of a v7 CraftSim.

    Read-only views (state, region, inventory) are cached until the next decision. The caller must not
    keep them across a yield. Results are plain Python values."""

    region_radius = 16     # half-size of the block window read for planning

    def __init__(self, sim, agent):
        self.sim = sim
        self.agent = int(agent)
        self.K = sim.agents
        self.world = self.agent // self.K
        self.task = None
        self.task_info = None
        self.rng = random.Random(self.agent)
        self.decisions = 0
        self._cache = {}

    # ------------------------------------------------------------------ episode bookkeeping

    def new_episode(self, task_name, rng=None):
        """Called by the Runner at the start of every episode of this agent's world."""
        self.task = task_name
        self.task_info = TASK_INFO[task_name]
        self.rng = rng if rng is not None else random.Random(self.agent)
        self.decisions = 0
        self._cache.clear()

    def _act(self, prim, recipe=0, item=0, msg=0):
        """Yield one decision. recipe / item are the raw columns (recipe r -> r + 1, item i -> i + 1)."""
        yield (PRIM[prim], int(recipe), int(item), int(msg))
        self._cache.clear()
        self.decisions += 1

    # ------------------------------------------------------------------ observation

    def state(self):
        s = self._cache.get("state")
        if s is None:
            s = self._cache["state"] = self.sim.state(self.agent)
        return s

    def pos(self):
        s = self.state()
        return s["x"], s["y"], s["z"]

    def cell(self):
        """Feet cell (integer block coordinates)."""
        x, y, z = self.pos()
        return (math.floor(x), math.floor(y + 1e-3), math.floor(z))

    def eye(self):
        x, y, z = self.pos()
        return (x, y + EYE, z)

    def held(self):
        return self.state()["held"]

    def region(self, radius=None):
        r = self.region_radius if radius is None else radius
        key = ("region", r)
        reg = self._cache.get(key)
        if reg is None:
            origin, arr = self.sim.region(self.agent, r)
            reg = self._cache[key] = Region(origin, arr)
        return reg

    def block_at(self, x, y, z):
        """Block name at a world cell (None outside the cached window)."""
        b = self.region().get(x, y, z)
        return cs.BLOCKS[b] if b >= 0 else None

    def inventory(self):
        inv = self._cache.get("inv")
        if inv is None:
            inv = self._cache["inv"] = self.sim.inventory(self.agent)
        return inv

    def count(self, item):
        return self.inventory().get(item, 0)

    def find_blocks(self, names, radius=12):
        """Cells of the given block names within `radius` (Chebyshev) of the feet, nearest first.
        Returns [((x, y, z), name), ...]."""
        reg = self.region()
        ids = [BID[n] for n in names]
        pts = np.argwhere(np.isin(reg.arr, ids))
        if pts.size == 0:
            return []
        xs = pts[:, 2] + reg.ox
        ys = pts[:, 0] + reg.oy
        zs = pts[:, 1] + reg.oz
        fx, fy, fz = self.cell()
        keep = (np.abs(xs - fx) <= radius) & (np.abs(zs - fz) <= radius) & (np.abs(ys - fy) <= radius)
        xs, ys, zs = xs[keep], ys[keep], zs[keep]
        d2 = (xs - fx) ** 2 + (ys - fy) ** 2 + (zs - fz) ** 2
        order = np.argsort(d2, kind="stable")
        out = []
        for j in order[:400]:
            cell = (int(xs[j]), int(ys[j]), int(zs[j]))
            out.append((cell, cs.BLOCKS[reg.get(*cell)]))
        return out

    def near_block(self, name):
        """A crafting table / furnace is within sim reach (craft/smelt need it)."""
        return _station_near(self.region(), self.cell(), BID[name])

    # ------------------------------------------------------------------ aiming

    def _aim_plan(self, point, accept=None, eye=None):
        """Quantized orientation that looks at `point`. With `accept`, the first candidate whose crosshair ray
        satisfies accept(hit) wins; candidates are the 9 quantized combinations around the exact angle,
        cheapest turn first. Returns (yaw, pitch, ok); ok False means no candidate passed accept."""
        st = self.state()
        ex, ey, ez = eye if eye is not None else (st["x"], st["y"] + EYE, st["z"])
        dx, dy, dz = point[0] - ex, point[1] - ey, point[2] - ez
        yaw_t, pitch_t = _look_angles(dx, dy, dz)
        yq = int(round(yaw_t / TURN)) * 15 % 360
        pq = max(-90, min(90, int(round(pitch_t / TURN)) * 15))
        if accept is None:
            return yq, pq, True
        cur_y, cur_p = st["yaw"], st["pitch"]
        combos = []
        for dyaw in (0, -15, 15):
            for dp in (0, -15, 15):
                y2 = (yq + dyaw) % 360
                p2 = max(-90, min(90, pq + dp))
                cost = abs(_wrap180(y2 - cur_y)) / TURN + abs(p2 - cur_p) / TURN
                err = abs(_wrap180(y2 - yaw_t)) + abs(p2 - pitch_t)
                combos.append((cost, err, y2, p2))
        combos.sort()
        reg = self.region()
        for _, _, y2, p2 in combos:
            hit = ray_hit(reg, (ex, ey, ez), _look_dir(y2, p2))
            if hit is not None and accept(hit):
                return y2, p2, True
        return yq, pq, False

    def turn_to_yaw(self, yaw):
        cur = self.state()["yaw"]
        steps = int(round(_wrap180(yaw - cur) / TURN))
        for _ in range(abs(steps)):
            yield from self._act("turn_right" if steps > 0 else "turn_left")
        return True

    def look_to_pitch(self, pitch):
        cur = self.state()["pitch"]
        steps = int(round((pitch - cur) / TURN))
        for _ in range(abs(steps)):
            yield from self._act("look_down" if steps > 0 else "look_up")
        return True

    def aim_at(self, point, accept=None):
        """Turn and look toward a world point (see _aim_plan). Returns the accept result (True without accept)."""
        y2, p2, ok = self._aim_plan(point, accept)
        yield from self.turn_to_yaw(y2)
        yield from self.look_to_pitch(p2)
        return ok

    def face(self, x, y, z):
        """Turn and look at block (x, y, z) so the crosshair hits it. Returns True when the ray hits it;
        otherwise the agent is still turned to the closest quantized direction (<= ~10 deg off)."""
        cell = (x, y, z)
        return (yield from self.aim_at((x + 0.5, y + 0.5, z + 0.5),
                                       accept=lambda h: h is not None and h[0] == cell))

    def look_at_block(self, x, y, z):
        """Alias of face(x, y, z) (PROTOCOL 14.2 name)."""
        return (yield from self.face(x, y, z))

    def face_point(self, px, py, pz):
        """Turn and look toward a point (no ray check). Returns True."""
        return (yield from self.aim_at((px, py, pz)))

    # ------------------------------------------------------------------ movement

    def _dir_prim(self, dx, dz):
        yaw = math.radians(self.state()["yaw"])
        fx, fz = -math.sin(yaw), math.cos(yaw)
        rx, rz = -math.cos(yaw), -math.sin(yaw)
        df = dx * fx + dz * fz
        dr = dx * rx + dz * rz
        if abs(df) >= abs(dr):
            return "forward" if df > 0 else "back"
        return "right" if dr > 0 else "left"

    def _facing(self, dx, dz):
        yaw = math.radians(self.state()["yaw"])
        return -math.sin(yaw) * dx + math.cos(yaw) * dz > 0.5

    def _step(self, cur, nxt):
        """One move toward the adjacent node nxt, steering from the actual position toward the node's centre
        (the feet cell can lag the body at block edges). Flat or down: strafe/back/forward; up: turn, jump."""
        st = self.state()
        dx, dz = nxt[0] + 0.5 - st["x"], nxt[2] + 0.5 - st["z"]
        if nxt[1] > cur[1]:
            if not self._facing(dx, dz):
                yield from self.turn_to_yaw(_yaw_of(dx, dz))
            yield from self._act("forward_jump")
        elif math.hypot(dx, dz) < 0.12:
            yield from self._act("noop")
        else:
            # centre the body on the lane first when it is off by more than a quarter block: a body corner
            # overlapping the neighbouring column (solid there) blocks sideways moves
            if abs(dz) > 0.25 and abs(dx) >= abs(dz):
                yield from self._act(self._dir_prim(0.0, dz))
            elif abs(dx) > 0.25 and abs(dz) > abs(dx):
                yield from self._act(self._dir_prim(dx, 0.0))
            else:
                yield from self._act(self._dir_prim(dx, dz))

    def _walk(self, goal_fn, hx, hz, dig=False, max_decisions=600, allow_dig=True):
        """Walk along A* paths until goal_fn(cell, reg) holds. Replans when the path is left. When no open path
        exists (or dig=True) the paths may dig through blocks, and blocking cells are mined. Returns True on
        arrival, False on failure or budget exhaustion."""
        start = self.decisions
        path, idx, replans, stuck, last = None, 0, 0, 0, None
        use_dig = dig
        while True:
            reg = self.region()
            cur = self.cell()
            if goal_fn(cur, reg):
                return True
            if self.decisions - start >= max_decisions:
                return False
            if path is None:
                replans += 1
                if replans > 60:
                    return False
                path = astar(reg, cur, goal_fn, hx, hz, use_dig)
                if path is None and not use_dig and allow_dig:
                    path = astar(reg, cur, goal_fn, hx, hz, True)
                    use_dig = path is not None
                if path is None:
                    return False
                idx = 0
            j = None
            for k in range(idx, min(len(path), idx + 6)):
                if path[k] == cur:
                    j = k
                    break
            if j is None:
                # momentum can carry the agent off the path: rejoin at a nearby flat node of the path
                for k in range(idx, min(len(path) - 1, idx + 6)):
                    p_ = path[k]
                    if p_[1] == cur[1] and abs(p_[0] - cur[0]) + abs(p_[2] - cur[2]) <= 2:
                        j = k
                        break
            if j is None or j + 1 >= len(path):
                path = None
                continue
            idx = j
            nxt = path[j + 1]
            need = [nxt, (nxt[0], nxt[1] + 1, nxt[2])]
            if nxt[1] == cur[1] + 1:
                need.append((cur[0], cur[1] + 2, cur[2]))
            blocked = [c for c in need if reg.get(*c) != AIR]
            if blocked:
                # a dig path: clear the cells it needs and keep following it; otherwise replan
                cleared = False
                if use_dig and all(_diggable(reg.get(*c)) for c in blocked):
                    cleared = True
                    for c in blocked:
                        if not (yield from self.mine_block(c[0], c[1], c[2], walk=False)):
                            cleared = False
                            break
                if not cleared:
                    path = None
                continue
            if last == cur:
                stuck += 1
                if stuck > WALK_STUCK:
                    return False
            else:
                stuck = 0
                last = cur
            yield from self._step(cur, nxt)

    def walk_to(self, x, y, z, within=0.5, dig=False, max_decisions=600):
        """Walk to the feet cell (x, y, z) (or a standable cell within `within` blocks of its column, dy <= 1).
        Returns True on arrival. Without an open path it digs through blocks (dig=True, or after the open
        search failed)."""
        def goal(node, reg):
            return (abs(node[1] - y) <= 1 and (node[0] - x) ** 2 + (node[2] - z) ** 2 <= within ** 2 + 1e-9
                    and standable(reg, *node))
        return (yield from self._walk(goal, x, z, dig=dig, max_decisions=max_decisions))

    def explore(self, min_d=6, max_d=12, tries=60):
        """Walk to a random standable cell 6..12 blocks away (from the window). Returns bool."""
        reg = self.region()
        cx, cy, cz = self.cell()
        cands = []
        for _ in range(tries):
            x = cx + self.rng.randint(-max_d, max_d)
            z = cz + self.rng.randint(-max_d, max_d)
            d = max(abs(x - cx), abs(z - cz))
            if d < min_d:
                continue
            for y in range(cy + 3, cy - 4, -1):
                if standable(reg, x, y, z):
                    cands.append((x, y, z))
                    break
            if len(cands) >= 8:
                break
        if not cands:
            yield from self._act("noop")
            return False
        x, y, z = self.rng.choice(cands)
        return (yield from self._walk(lambda n, r: n == (x, y, z), x, z, max_decisions=200, allow_dig=False))

    # ------------------------------------------------------------------ mining and placing

    def _best_tool(self, block):
        info = BLOCK_INFO[block]
        tt = info.get("best_tool")
        if not tt:
            return None
        best, tier_best = None, -1
        for item, n in self.inventory().items():
            tool = ITEM_INFO[item].get("tool")
            if n > 0 and tool and tool["type"] == tt and tool["tier"] >= info["min_tier"] and tool["tier"] > tier_best:
                best, tier_best = item, tool["tier"]
        return best

    def _visible_from(self, reg, node, target):
        """Some quantized orientation from feet `node` hits block `target` within AIM_REACH."""
        ex, ey, ez = node[0] + 0.5, node[1] + EYE, node[2] + 0.5
        cx, cy, cz = target[0] + 0.5, target[1] + 0.5, target[2] + 0.5
        dx, dy, dz = cx - ex, cy - ey, cz - ez
        if math.sqrt(dx * dx + dy * dy + dz * dz) > AIM_REACH:
            return False
        yaw_t, pitch_t = _look_angles(dx, dy, dz)
        yq = int(round(yaw_t / TURN)) * 15 % 360
        pq = max(-90, min(90, int(round(pitch_t / TURN)) * 15))
        for dyaw in (0, -15, 15):
            for dp in (0, -15, 15):
                hit = ray_hit(reg, (ex, ey, ez), _look_dir((yq + dyaw) % 360, max(-90, min(90, pq + dp))))
                if hit is not None and hit[0] == target:
                    return True
        return False

    def _blocker(self, tgt):
        """The mineable block that stops the exact view ray from the eye to block `tgt` (within reach), or None."""
        reg = self.region()
        ex, ey, ez = self.eye()
        dx, dy, dz = tgt[0] + 0.5 - ex, tgt[1] + 0.5 - ey, tgt[2] + 0.5 - ez
        yaw, pitch = _look_angles(dx, dy, dz)
        hit = ray_hit(reg, (ex, ey, ez), _look_dir(yaw, pitch))
        if hit is None or hit[0] == tgt:
            return None
        cell = hit[0]
        if math.dist((cell[0] + 0.5, cell[1] + 0.5, cell[2] + 0.5), (ex, ey, ez)) > AIM_REACH:
            return None
        return cell if _diggable(reg.get(*cell)) else None

    def mine_block(self, x, y, z, walk=True, max_attempts=10, _depth=0):
        """Break block (x, y, z) with the best tool in inventory (attack holds until it breaks). When the block
        is hidden, first clears the blocking block in reach (leaves in trees), else walks to a stand spot with a
        clear view (walk=True). Returns True when the cell is air afterwards."""
        tgt = (x, y, z)
        for _ in range(max_attempts):
            reg = self.region()
            b = reg.get(x, y, z)
            if b == AIR:
                return True
            if b < 0 or not _diggable(b) or BLOCK_INFO[cs.BLOCKS[b]]["hardness"] < 0:
                return False
            name = cs.BLOCKS[b]
            ok = yield from self.aim_at((x + 0.5, y + 0.5, z + 0.5),
                                        accept=lambda h: h is not None and h[0] == tgt)
            if not ok:
                blocker = self._blocker(tgt)
                if blocker is not None and _depth < 3:
                    if not (yield from self.mine_block(*blocker, walk=False, _depth=_depth + 1)):
                        return False
                    continue
                if not walk:
                    return False
                # the aim failed from here: walk to the nearest visible stand spot around the block
                here = self.cell()
                cands = []
                for dx in range(-4, 5):
                    for dz in range(-4, 5):
                        for dy in range(-3, 4):
                            c = (x + dx, y + dy, z + dz)
                            if c != here and standable(reg, *c) and self._visible_from(reg, c, tgt):
                                cands.append(c)
                if not cands:
                    # nothing visible near the block yet: approach its nearest standable cell (the view is
                    # re-checked there, and blockers in reach are cleared)
                    for dx in range(-4, 5):
                        for dz in range(-4, 5):
                            for dy in range(-3, 4):
                                c = (x + dx, y + dy, z + dz)
                                if c != here and standable(reg, *c):
                                    cands.append(c)
                cands.sort(key=lambda c: abs(c[0] - here[0]) + abs(c[2] - here[2]) + abs(c[1] - here[1]))
                moved = False
                for c in cands[:3]:
                    if (yield from self._walk(lambda n, r, c=c: n == c, c[0], c[2], max_decisions=200)):
                        moved = True
                        break
                if not moved:
                    return False
                continue
            tool = self._best_tool(name)
            yield from self._act("attack", item=IID[tool] + 1 if tool else 0)
        return self.region().get(x, y, z) == AIR

    def _place_plan(self, reg, cell, eye=None, feet=None):
        """(yaw, pitch) that places onto `cell` from a visible face, or None."""
        if feet is not None and _body_overlaps(feet[0], feet[1], feet[2], cell):
            return None
        if reg.get(*cell) != AIR:
            return None
        if eye is None:
            st = self.state()
            feet = (st["x"], st["y"], st["z"])
            if _body_overlaps(feet[0], feet[1], feet[2], cell):
                return None
            eye = (feet[0], feet[1] + EYE, feet[2])
        for dx, dy, dz in _DIRS6:
            n = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            b = reg.get(*n)
            if b <= 0 or not _SOLID[b]:
                continue
            fc = (cell[0] + 0.5 + 0.5 * dx, cell[1] + 0.5 + 0.5 * dy, cell[2] + 0.5 + 0.5 * dz)
            if math.dist(fc, eye) > AIM_REACH:
                continue
            y2, p2, ok = self._aim_plan(
                fc, accept=lambda h, n=n, cell=cell: h is not None and h[0] == n and h[1] == cell, eye=eye)
            if ok:
                return y2, p2
        return None

    def place_block(self, item, x, y, z, walk=True, max_attempts=6):
        """Place `item` (its block) into the empty cell (x, y, z) through a visible face. Walks to a spot within
        reach when needed. Returns True when the cell holds the block afterwards."""
        cell = (x, y, z)
        placed = ITEM_INFO[item].get("places_block")
        if not placed:
            return False
        pid = BID[placed]
        for _ in range(max_attempts):
            reg = self.region()
            if reg.get(*cell) == pid:
                return True
            if reg.get(*cell) != AIR or self.count(item) <= 0:
                return False
            plan = self._place_plan(reg, cell)
            if plan is None:
                if not walk:
                    return False

                def goal(node, r):
                    return (standable(r, *node) and not _body_overlaps(node[0] + 0.5, node[1], node[2] + 0.5, cell)
                            and abs(node[0] - x) + abs(node[2] - z) <= 3 and
                            self._place_plan(r, cell, eye=(node[0] + 0.5, node[1] + EYE, node[2] + 0.5),
                                             feet=(node[0] + 0.5, node[1], node[2] + 0.5)) is not None)
                if not (yield from self._walk(goal, x, z, max_decisions=200)):
                    return False
                continue
            y2, p2 = plan
            yield from self.turn_to_yaw(y2)
            yield from self.look_to_pitch(p2)
            yield from self._act("use", item=IID[item] + 1)
        return self.region().get(*cell) == pid

    # ------------------------------------------------------------------ crafting

    def ensure_station(self, name, walk=True):
        """Make a crafting table / furnace (name) reachable: already near, place one from inventory beside the
        agent, or walk to the nearest one in the window. Returns bool."""
        if self.near_block(name):
            return True
        if self.count(name) > 0:
            cx, cy, cz = self.cell()
            for dx, dz in _DIRS4:
                if (yield from self.place_block(name, cx + dx, cy, cz + dz, walk=False)):
                    return True
        if not walk:
            return False
        sites = self.find_blocks([name], radius=12)
        if not sites:
            return False
        (sx, _, sz), _ = sites[0]
        bid = BID[name]
        return (yield from self._walk(lambda n, r: standable(r, *n) and _station_near(r, n, bid), sx, sz,
                                      max_decisions=300))

    def craft(self, recipe_name, walk=True):
        """Craft one recipe by name (craft column). Table recipes first make sure a crafting table is in reach.
        Returns True if the result count increased."""
        R = RECIPE_BY_NAME[recipe_name]
        if R["needs"] == "furnace":
            return False       # use smelt()
        if any(self.count(i) < n for i, n in R["ingredients"].items()):
            return False
        if R["needs"] == "crafting_table":
            if not (yield from self.ensure_station("crafting_table", walk=walk)):
                return False
        before = self.count(R["result"])
        yield from self._act("craft", recipe=R["id"] + 1)
        return self.count(R["result"]) > before

    def smelt(self, input_item, walk=True):
        """Smelt one unit of `input_item` in a furnace (fuel is taken from inventory). Returns bool."""
        rec = next((r for r in RECIPES if r["needs"] == "furnace" and input_item in r["ingredients"]), None)
        if rec is None or self.count(input_item) < rec["ingredients"][input_item]:
            return False
        if not (yield from self.ensure_station("furnace", walk=walk)):
            return False
        before = self.count(rec["result"])
        yield from self._act("smelt", recipe=rec["id"] + 1)
        return self.count(rec["result"]) > before

    # ------------------------------------------------------------------ items, food, combat

    def equip(self, item):
        """Select `item` as held (item column; the sim applies it before the primary). Returns True if held."""
        if self.held() == item:
            return True
        yield from self._act("noop", item=IID[item] + 1)
        return self.held() == item

    def eat(self, item):
        """Use (eat) one `item` from inventory. Returns True if one was consumed."""
        if self.count(item) <= 0:
            return False
        before = self.count(item)
        yield from self._act("use", item=IID[item] + 1)
        return self.count(item) < before

    def attack_entity(self, mob=None, max_decisions=400):
        """Chase and hit the nearest entity named `mob` (None: any). Returns True once no such entity is left,
        False on timeout or when none was present at the start. Mob combat needs the mobs module."""
        start = self.decisions
        ents = [e for e in self.sim.entities(self.world) if mob is None or e["type"] == mob]
        if not ents:
            return False
        while self.decisions - start < max_decisions:
            ents = [e for e in self.sim.entities(self.world) if mob is None or e["type"] == mob]
            if not ents:
                return True
            fx, fy, fz = self.pos()
            e = min(ents, key=lambda e: (e["x"] - fx) ** 2 + (e["y"] - fy) ** 2 + (e["z"] - fz) ** 2)
            ex, ey, ez = e["x"], e["y"], e["z"]
            dh = math.hypot(ex - fx, ez - fz)
            if dh <= 2.5 and abs(ey - fy) <= 2.0:
                yield from self.aim_at((ex, ey + 0.9, ez))
                yield from self._act("attack")
            else:
                tgt = (math.floor(ex), math.floor(ey + 1e-3), math.floor(ez))
                before = self.decisions
                yield from self._walk(lambda n, r: standable(r, *n) and
                                      (n[0] - tgt[0]) ** 2 + (n[2] - tgt[2]) ** 2 <= 4 and abs(n[1] - tgt[1]) <= 1,
                                      tgt[0], tgt[2], max_decisions=8)
                if self.decisions == before:
                    yield from self._act("noop")
        return False

    def flee(self, x, y, z, distance=10, max_decisions=120):
        """Walk to a standable cell away from (x, y, z), about `distance` blocks from here. Returns bool."""
        reg = self.region()
        cx, cy, cz = self.cell()
        best, best_score = None, None
        for dx in range(-12, 13):
            for dz in range(-12, 13):
                for dy in range(-2, 3):
                    c = (cx + dx, cy + dy, cz + dz)
                    if not standable(reg, *c):
                        continue
                    threat = math.hypot(c[0] - x, c[2] - z)
                    mine = math.hypot(dx, dz)
                    score = min(threat, 16.0) - 0.5 * abs(mine - distance)
                    if best_score is None or score > best_score:
                        best, best_score = c, score
        if best is None:
            yield from self._act("noop")
            return False
        return (yield from self.walk_to(*best, max_decisions=max_decisions))

    def wait(self, n=1):
        """n noop decisions."""
        for _ in range(int(n)):
            yield from self._act("noop")
        return True

    def mine_nearest(self, names, radius=12, skip=()):
        """Mine the nearest block of `names` not in `skip`. Returns (cell, ok) or None when none is in range."""
        sites = [c for c, _ in self.find_blocks(names, radius) if c not in skip]
        if not sites:
            return None
        cell = sites[0]
        ok = yield from self.mine_block(*cell)
        return cell, ok
