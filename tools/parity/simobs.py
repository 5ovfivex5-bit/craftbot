"""Pure-Python re-implementation of src/sim.cpp writeObs() and raycast() for one dumped agent.

The sim cannot be told to build a dumped scene (CraftSim has no set_block / set_position binding),
so the "rebuilt scene" is this module reading the dumped blocks directly. Every formula, constant
and operation order follows sim.cpp at the time of writing:
  constants        sim.cpp top + sim.h (EYE, FOV_DEG, VIEW_DIST, REACH, boxHit sizes, SEA_LEVEL)
  lookDir/camera   lookDir(), camera(), castPixel()
  raycast          raycast() DDA incl. teammate boxes (AGENT) and the y >= WY sky exit
  obs              writeObs() scalars, teammate slots, nearest chest, crosshair chest
If sim.cpp changes, update this file; compare.py --selftest will flag any drift against the live sim.

Unknown data: a ray that reaches a cell that is not in the dump (or a formula whose inputs are not
in the dump) is reported as unknown and excluded from agreement, never counted as a match.
"""
import math

import numpy as np

EYE = 1.62
HALF_W = 0.3
HEIGHT = 1.8
REACH = 4.5
VIEW_DIST = 24.0
FOV_DEG = 70.0
SEA_LEVEL = 30
NUM_MSGS = 8
PI = 3.14159265358979323846
TEAM_SLOTS = 4
TEAM_STRIDE = 12
CHEST_RANGE = 16.0

AIR, BEDROCK, WATER, TABLE, FURNACE, CHEST, AGENT = 0, 13, 16, 8, 9, 18, 19
UNK = -1

from dumpio import ITEMS  # noqa: E402  (sim item order)


class Scene:
    """Dumped blocks + build limits. get() follows the sim's get() / Obs.cast() rules."""

    def __init__(self, origin, grid, build_limit):
        self.ox, self.oy, self.oz = origin
        self.grid = grid.tolist()          # grid[y][z][x], -1 = unknown
        self.ny, self.nz, self.nx = grid.shape
        self.min_y = build_limit[0]
        self.max_y = build_limit[1]

    def get(self, x, y, z):
        if y > self.max_y:
            return AIR
        if y < self.min_y:
            return BEDROCK
        xx = x - self.ox
        yy = y - self.oy
        zz = z - self.oz
        if 0 <= xx < self.nx and 0 <= yy < self.ny and 0 <= zz < self.nz:
            return self.grid[yy][zz][xx]
        return UNK


def _box_hit(m, ox, oy, oz, dx, dy, dz, maxd):
    """sim.cpp boxHit(): teammate box at feet m (0.6 x 1.8 x 0.6). Returns entry t or None."""
    mn = (m[0] - HALF_W, m[1], m[2] - HALF_W)
    mx = (m[0] + HALF_W, m[1] + HEIGHT, m[2] + HALF_W)
    org = (ox, oy, oz)
    dr = (dx, dy, dz)
    t0 = -1e30
    t1 = 1e30
    for a in range(3):
        if abs(dr[a]) < 1e-12:
            if org[a] < mn[a] or org[a] > mx[a]:
                return None
            continue
        inv = 1.0 / dr[a]
        ta = (mn[a] - org[a]) * inv
        tb = (mx[a] - org[a]) * inv
        if ta > tb:
            ta, tb = tb, ta
        if ta > t0:
            t0 = ta
        if tb < t1:
            t1 = tb
        if t0 > t1:
            return None
    if t1 < 0 or t0 > maxd:
        return None
    return max(t0, 0.0)


def raycast(scene, mates, ox, oy, oz, dx, dy, dz, maxd, pass_water):
    """sim.cpp raycast(). mates = [(x, y, z)] teammate feet in k order (empty = blocks only).

    Returns (kind, block_id, dist, cell): kind is 'hit', 'agent', 'miss' or 'unk'.
    """
    aT = 1e30
    aK = -1
    for k, m in enumerate(mates):
        te = _box_hit(m, ox, oy, oz, dx, dy, dz, maxd)
        if te is not None and te < aT:
            aT = te
            aK = k

    x = math.floor(ox)
    y = math.floor(oy)
    z = math.floor(oz)
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    sz = 1 if dz > 0 else -1
    tdx = abs(1.0 / dx) if dx != 0 else 1e30
    tdy = abs(1.0 / dy) if dy != 0 else 1e30
    tdz = abs(1.0 / dz) if dz != 0 else 1e30
    tmx = ((x + 1 - ox) if dx > 0 else (ox - x)) * tdx if dx != 0 else 1e30
    tmy = ((y + 1 - oy) if dy > 0 else (oy - y)) * tdy if dy != 0 else 1e30
    tmz = ((z + 1 - oz) if dz > 0 else (oz - z)) * tdz if dz != 0 else 1e30
    lx, ly, lz = x, y, z
    t = 0.0
    while True:
        if aK >= 0 and aT <= t:
            m = mates[aK]
            return ("agent", AGENT, aT, (math.floor(m[0]), math.floor(m[1]), math.floor(m[2])))
        b = scene.get(x, y, z)
        if b == UNK:
            return ("unk", None, None, None)
        if b != AIR and not (pass_water and b == WATER):
            return ("hit", b, t, (x, y, z))
        if y > scene.max_y and dy >= 0:
            if aK >= 0 and aT <= maxd:
                m = mates[aK]
                return ("agent", AGENT, aT, (math.floor(m[0]), math.floor(m[1]), math.floor(m[2])))
            return ("miss", None, None, None)
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
            if aK >= 0 and aT <= maxd:
                m = mates[aK]
                return ("agent", AGENT, aT, (math.floor(m[0]), math.floor(m[1]), math.floor(m[2])))
            return ("miss", None, None, None)


def look_dir(yaw, pitch):
    """sim.cpp lookDir(float yaw, float pitch)."""
    y = yaw * PI / 180.0
    p = pitch * PI / 180.0
    return (-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p))


def _camera(px, py, pz, yaw, pitch):
    f = look_dir(yaw, pitch)
    yr = yaw * PI / 180.0
    r = (-math.cos(yr), 0.0, -math.sin(yr))
    u = (r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0])
    return (px, py + EYE, pz), f, r, u


def _pass_water(scene, px, py, pz):
    return scene.get(math.floor(px), math.floor(py + EYE), math.floor(pz)) == WATER


def render(scene, px, py, pz, yaw, pitch, res, mates):
    """writeObs() image part. Returns (img uint8 [res,res,2], known bool [res,res], hit_cells dict)."""
    (ox, oy, oz), f, r, up = _camera(px, py, pz, yaw, pitch)
    tanH = math.tan(FOV_DEG * 0.5 * PI / 180.0)
    pass_w = _pass_water(scene, px, py, pz)
    img = np.zeros((res, res, 2), dtype=np.uint8)
    known = np.ones((res, res), dtype=bool)
    for j in range(res):
        for i in range(res):
            u = ((i + 0.5) / res * 2 - 1) * tanH * 1.0
            v = (1 - (j + 0.5) / res * 2) * tanH
            d = [f[k] + r[k] * u + up[k] * v for k in range(3)]
            n = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            d = [d[k] / n for k in range(3)]
            kind, b, dist, _cell = raycast(scene, mates, ox, oy, oz, d[0], d[1], d[2], VIEW_DIST, pass_w)
            if kind == "unk":
                known[j, i] = False
                img[j, i, 1] = 255
                continue
            if kind == "miss":
                img[j, i, 0] = 0
                img[j, i, 1] = 255
                continue
            img[j, i, 0] = b
            img[j, i, 1] = int(min(255.0, dist / VIEW_DIST * 255.0))
    return img, known


def _yaw_frame(yaw, dx, dz):
    yr = yaw * PI / 180.0
    fx, fz = -math.sin(yr), math.cos(yr)
    rx, rz = -math.cos(yr), -math.sin(yr)
    return dx * rx + dz * rz, dx * fx + dz * fz


def _near(scene, px, py, pz, b):
    """nearBlock(): box dy -3..4, dz -3..3, dx -3..3 around the feet block. True / False / None."""
    cx, cy, cz = math.floor(px), math.floor(py), math.floor(pz)
    saw_unk = False
    for dy in range(-3, 5):
        for dz in range(-3, 4):
            for dx in range(-3, 4):
                v = scene.get(cx + dx, cy + dy, cz + dz)
                if v == b:
                    return True
                if v == UNK:
                    saw_unk = True
    return None if saw_unk else False


def vec_values(scene, dump):
    """Every sim vec value we can compute from the dump, keyed by VEC name. Unknown inputs -> NaN."""
    nan = float("nan")
    vals = {}
    px, py, pz = dump["pose"]["x"], dump["pose"]["y"], dump["pose"]["z"]
    yaw = float(np.float32(dump["pose"]["yaw"]))
    pitch = float(np.float32(dump["pose"]["pitch"]))
    inv = dump["inventory"]
    l64 = math.log1p(64.0)
    for it in ITEMS:
        vals[f"inv_{it}"] = math.log1p(float(inv.get(it, 0))) / l64
    p = dump["player"]
    vals["health"] = float(np.float32(p["health"])) / 20.0
    vals["on_ground"] = 1.0 if dump["pose"]["on_ground"] else 0.0
    vals["pitch"] = pitch / 90.0
    yr = yaw * PI / 180.0
    vals["sin_yaw"] = math.sin(yr)
    vals["cos_yaw"] = math.cos(yr)
    vals["mining_progress"] = float(p["mining_progress"])
    vals["height"] = (py - SEA_LEVEL) / 32.0
    nt = _near(scene, px, py, pz, TABLE)
    vals["near_table"] = nan if nt is None else (1.0 if nt else 0.0)
    nf = _near(scene, px, py, pz, FURNACE)
    vals["near_furnace"] = nan if nf is None else (1.0 if nf else 0.0)
    vals["fuel"] = min(int(p["fuel"]), 8) / 8.0
    vals["time_frac"] = float(p["step"]) / float(p["max_steps"])
    goal = dump["goal"]["index"]
    for i, it in enumerate(ITEMS):
        vals[f"goal_{it}"] = 1.0 if i == goal else 0.0

    # Teammate slots: k' != k ascending, compact, 12 floats each (sim writeObs).
    team = sorted(dump["teammates"], key=lambda t: t["index"])
    for j in range(TEAM_SLOTS):
        vals[f"tm{j}_present"] = 0.0
        for m in range(NUM_MSGS):
            vals[f"tm{j}_msg{m}"] = 0.0
        vals[f"tm{j}_dx"] = 0.0
        vals[f"tm{j}_dy"] = 0.0
        vals[f"tm{j}_dz"] = 0.0
    for j, t in enumerate(team[:TEAM_SLOTS]):
        xl, zl = _yaw_frame(yaw, t["x"] - px, t["z"] - pz)
        vals[f"tm{j}_present"] = 1.0
        vals[f"tm{j}_dx"] = xl / 32.0
        vals[f"tm{j}_dy"] = (t["y"] - py) / 32.0
        vals[f"tm{j}_dz"] = zl / 32.0
        for m in range(NUM_MSGS):
            vals[f"tm{j}_msg{m}"] = 1.0 if m == min(max(int(t["msg"]), 0), NUM_MSGS - 1) else 0.0

    # Nearest chest within 16 (sim iterates std::map keys ascending = (y, z, x); ties -> later).
    chests = sorted(dump["chests"], key=lambda c: (c["y"], c["z"], c["x"]))
    best = 256.0
    found = None
    for c in chests:
        dx = c["x"] + 0.5 - px
        dy = c["y"] + 0.5 - py
        dz = c["z"] + 0.5 - pz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 <= best:
            best = d2
            found = c
    vals["chest_present"] = 0.0
    vals["chest_dx"] = vals["chest_dy"] = vals["chest_dz"] = 0.0
    if found is not None:
        xl, zl = _yaw_frame(yaw, found["x"] + 0.5 - px, found["z"] + 0.5 - pz)
        vals["chest_present"] = 1.0
        vals["chest_dx"] = xl / CHEST_RANGE
        vals["chest_dy"] = (found["y"] + 0.5 - py) / CHEST_RANGE
        vals["chest_dz"] = zl / CHEST_RANGE

    # Crosshair chest contents (blocks only, reach 4.5, water stops unless eye is in water).
    (ox, oy, oz), f, _r, _u = _camera(px, py, pz, yaw, pitch)
    kind, b, _dist, cell = raycast(scene, [], ox, oy, oz, f[0], f[1], f[2], REACH, _pass_water(scene, px, py, pz))
    looking_at_chest = kind == "hit" and b == CHEST
    contents = None
    if looking_at_chest:
        for c in dump["chests"]:
            if (c["x"], c["y"], c["z"]) == tuple(cell):
                contents = c["items"]
                break
    for it in ITEMS[:9]:
        if kind == "unk" or (looking_at_chest and contents is None):
            vals[f"xhair_chest_{it}"] = nan       # chest under the crosshair not in the dump
        elif not looking_at_chest:
            vals[f"xhair_chest_{it}"] = 0.0
        else:
            n = float(np.float32(contents.get(it, 0)))
            vals[f"xhair_chest_{it}"] = float(np.float32(math.log2(1.0 + n) / 6.0))
    return vals


def center_cell(scene, dump):
    """Block cell hit by the centre pixel (used by the negative control in compare.py --selftest)."""
    px, py, pz = dump["pose"]["x"], dump["pose"]["y"], dump["pose"]["z"]
    (ox, oy, oz), f, _r, _u = _camera(px, py, pz, float(np.float32(dump["pose"]["yaw"])),
                                       float(np.float32(dump["pose"]["pitch"])))
    kind, b, _d, cell = raycast(scene, [], ox, oy, oz, f[0], f[1], f[2], VIEW_DIST,
                                _pass_water(scene, px, py, pz))
    return cell if kind == "hit" else None
