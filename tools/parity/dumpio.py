"""Read/write the `minemind-parity-1` dump format (see DUMP_FORMAT.md).

Also builds a dump from a live craftsim.CraftSim (used by compare.py --selftest).
"""
import base64
import datetime
import json
import math

import numpy as np

FORMAT = "minemind-parity-1"

# Sim block / item names (craftsim.BLOCKS / craftsim.ITEMS, index == id). Checked against craftsim
# at selftest time; keep in sync with src/sim.h.
BLOCKS = [
    "air", "grass", "dirt", "stone", "cobblestone", "log", "leaves", "planks",
    "crafting_table", "furnace", "coal_ore", "iron_ore", "diamond_ore", "bedrock",
    "sand", "gravel", "water", "deepslate", "chest", "agent",
]
ITEMS = [
    "log", "planks", "stick", "crafting_table", "wooden_pickaxe", "cobblestone",
    "stone_pickaxe", "furnace", "coal", "iron_ore", "iron_ingot", "iron_pickaxe",
    "diamond", "dirt", "chest",
]
AGENT_BLOCK = 19
UNKNOWN = 255  # ours value for an unloaded cell in the dump

# Vanilla id written for each sim block when a dump is built from the sim.
VANILLA_OF_SIM = {
    "air": "minecraft:air", "grass": "minecraft:grass_block", "dirt": "minecraft:dirt",
    "stone": "minecraft:stone", "cobblestone": "minecraft:cobblestone", "log": "minecraft:oak_log",
    "leaves": "minecraft:oak_leaves", "planks": "minecraft:oak_planks",
    "crafting_table": "minecraft:crafting_table", "furnace": "minecraft:furnace",
    "coal_ore": "minecraft:coal_ore", "iron_ore": "minecraft:iron_ore",
    "diamond_ore": "minecraft:diamond_ore", "bedrock": "minecraft:bedrock", "sand": "minecraft:sand",
    "gravel": "minecraft:gravel", "water": "minecraft:water", "deepslate": "minecraft:deepslate",
    "chest": "minecraft:chest", "agent": "craftbot:agent",
}


def b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(path):
    """Load and decode a dump. Returns the raw dict plus decoded arrays under d['_arr']."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if d.get("format") != FORMAT:
        raise ValueError(f"{path}: format {d.get('format')!r}, expected {FORMAT!r}")
    b = d["blocks"]
    ny, nz, nx = b["shape"]
    ours = np.frombuffer(b64d(b["ours"]), dtype=np.uint8)
    if ours.size != ny * nz * nx:
        raise ValueError(f"{path}: blocks.ours has {ours.size} cells, shape says {ny * nz * nx}")
    o = d["obs"]
    res = o["img"]["shape"][0]
    img = np.frombuffer(b64d(o["img"]["data"]), dtype=np.uint8).reshape(res, res, 2)
    vec_names = o["vec"]["names"]
    vec = np.asarray(o["vec"]["values"], dtype=np.float64)
    if len(vec_names) != vec.size:
        raise ValueError(f"{path}: {len(vec_names)} vec names, {vec.size} values")
    d["_arr"] = {
        "ours": ours.reshape(ny, nz, nx).astype(np.int16),
        "img": img,
        "vec": vec,
        "vec_names": vec_names,
    }
    return d


def block_grid_from_ours(ours_u8):
    """uint8 grid with 255 = unknown -> int16 grid with -1 = unknown."""
    g = ours_u8.astype(np.int16)
    g[g == UNKNOWN] = -1
    return g


def decode_vanilla(d):
    """Vanilla id grid (object array [ny,nz,nx]) of a loaded dump; 'unloaded' for unknown cells."""
    b = d["blocks"]
    palette = b["palette"]
    idx = np.frombuffer(b64d(b["vanilla"]), dtype="<u2")
    names = np.array(palette, dtype=object)[idx]
    return names.reshape(b["shape"])


def encode_blocks(origin, ours, vanilla_names, radius, build_limit):
    """ours: int array [ny,nz,nx] (255 unknown). vanilla_names: nested order same shape, str."""
    flat_names = [n for n in np.asarray(vanilla_names, dtype=object).reshape(-1)]
    palette = []
    index = {}
    idx = np.empty(len(flat_names), dtype="<u2")
    for i, name in enumerate(flat_names):
        if name not in index:
            index[name] = len(palette)
            palette.append(name)
        idx[i] = index[name]
    ny, nz, nx = ours.shape
    return {
        "radius": int(radius),
        "origin": [int(v) for v in origin],
        "shape": [int(ny), int(nz), int(nx)],
        "order": "y,z,x (x fastest)",
        "build_limit": {"min_y": int(build_limit[0]), "max_y": int(build_limit[1])},
        "palette": palette,
        "vanilla": b64e(idx.tobytes()),
        "ours": b64e(np.ascontiguousarray(ours, dtype=np.uint8).tobytes()),
    }


def dump_from_sim(sim, i, radius, max_steps, img, vec, vec_names, goal_names=ITEMS):
    """Build a parity dump for sim agent i from the sim's own API.

    Must be called right after the step that produced img[i] / vec[i]: the pose, inventory,
    chests and blocks are read now, so they match the obs. Uses only public CraftSim calls.
    sim.block(i, x, y, z) gives the exact get() semantics (out of world = bedrock, above = air).
    """
    st = sim.state(i)
    res = int(img.shape[1])            # img is [N, res, res, 2]
    w = i // sim.agents
    k = i % sim.agents
    px, py, pz = st["x"], st["y"], st["z"]
    cx, cy, cz = math.floor(px), math.floor(py), math.floor(pz)
    # Whole cube, with the sim's own block semantics (world bounds applied by sim.block).
    x0, x1 = cx - radius, cx + radius
    z0, z1 = cz - radius, cz + radius
    y0, y1 = cy - radius, cy + radius
    lo_y, hi_y = 0, 63  # sim build limits: y<0 bedrock, y>=64 air (the cube stores only these)
    sy0, sy1 = max(y0, lo_y), min(y1, hi_y)
    ny, nz, nx = sy1 - sy0 + 1, z1 - z0 + 1, x1 - x0 + 1
    ours = np.empty((ny, nz, nx), dtype=np.uint8)
    names = np.empty((ny, nz, nx), dtype=object)
    name_to_id = {n: j for j, n in enumerate(BLOCKS)}
    for yy in range(ny):
        y = sy0 + yy
        for zz in range(nz):
            z = z0 + zz
            for xx in range(nx):
                bname = sim.block(i, x0 + xx, y, z)
                ours[yy, zz, xx] = name_to_id[bname]
                names[yy, zz, xx] = VANILLA_OF_SIM[bname]
    # Cells above/below the stored y-range are implied by build_limit (air above 63, bedrock below 0):
    # the sim does exactly that, so the rule applies to rays too.
    blocks = encode_blocks((x0, sy0, z0), ours, names, radius, (lo_y, hi_y))

    inv = sim.inventory(i)
    inv_mapped = {n: int(c) for n, c in inv.items() if n in ITEMS}
    teammates = []
    for j in range(w * sim.agents, (w + 1) * sim.agents):
        if j == i:
            continue
        sj = sim.state(j)
        teammates.append({
            "index": j - w * sim.agents, "x": sj["x"], "y": sj["y"], "z": sj["z"],
            "yaw": float(sj["yaw"]), "pitch": float(sj["pitch"]),
            "msg": int(sj["msg"]), "health": float(sj["health"]),
        })
    chests = []
    for c in sim.chests(w):
        if abs(c["x"] + 0.5 - px) <= 16 + 1 and abs(c["y"] + 0.5 - py) <= 16 + 1 and abs(c["z"] + 0.5 - pz) <= 16 + 1:
            chests.append({"x": c["x"], "y": c["y"], "z": c["z"], "items": c["items"]})
    target = st["target"]
    goal = int(st["goal"])
    return {
        "format": FORMAT,
        "source": "craftsim-selftest",
        "created_utc": _now(),
        "game": {"mc_version": None, "mod": "craftsim", "level": None, "game_tick": None, "seed": int(st["seed"])},
        "bot": {"name": f"sim{i}", "team": w, "index": k},
        "pose": {"x": px, "y": py, "z": pz, "yaw": float(st["yaw"]), "pitch": float(st["pitch"]),
                 "on_ground": bool(st["on_ground"]), "eye_height": 1.62, "sneaking": False},
        "player": {"health": float(st["health"]),
                   # The sim does not expose fuel; v6 random policies never smelt, so it stays 0.
                   "fuel": 0,
                   "mining_progress": float(st["mining_progress"]),
                   "mining_target": None if target is None else [target["x"], target["y"], target["z"]],
                   "step": int(st["t"]), "max_steps": int(max_steps), "hunger": None, "saturation": None},
        "goal": {"name": goal_names[goal], "index": goal},
        "inventory": inv_mapped,
        "inventory_other": {},
        "teammates": teammates,
        "bodies": [],
        "chests": chests,
        "blocks": blocks,
        "vision": {"res": res, "fov_deg": 70.0, "view_dist": 24.0, "reach": 4.5,
                   "ray_origin": "feet + (0, 1.62, 0)", "bodies_in_rays": True,
                   "depth_encoding": "min(255, floor(dist / view_dist * 255)); sky = 255"},
        "obs": {
            "img": {"shape": [res, res, 2],
                    "order": "row-major, row 0 = top; channels = (block_id, depth)",
                    "encoding": "base64-u8", "data": b64e(np.ascontiguousarray(img[i], dtype=np.uint8).tobytes())},
            "vec": {"dim": int(vec.shape[1]), "names": list(vec_names),
                    "values": [float(v) for v in vec[i]]},
            "source": "craftsim (selftest)",
        },
    }


def write(path, d):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
        f.write("\n")
