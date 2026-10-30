"""Builds data/builds/blueprints.json (schema 12.6).

Sources, in id order:
  1. handmade bases (this file, `handmade()`): nether_portal, dirt_hut, wall_segment, farm_plot, ...
  2. vanilla village houses converted by nbt2blueprint.py (data/builds/vanilla_converted.json)
  3. procedural variants (this file, `variants()`): planks<->cobblestone material swap, x2 scale of simple
     shapes, mirror-X of asymmetric shapes. Variants that are identical to an existing blueprint, or that
     fall outside the 5..400 block range, are dropped.

Internal form: numpy array `arr[y, z, x]` of item-name strings, "." = air.
On disk, `layers[y][z]` is a row string over x (bottom-up), `palette` maps chars to item names.

Usage:  .venv/bin/python tools/builds/variants.py   (run nbt2blueprint.py first)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nbt2blueprint as nb  # noqa: E402

ROOT = nb.ROOT
OUT = os.path.join(ROOT, "data", "builds", "blueprints.json")
VANILLA = nb.OUT_VANILLA
AIR = "."
TIER_RANGES = nb.TIER_RANGES
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def new(y, z, x):
    return np.full((y, z, x), AIR, dtype="<U24")


def ring(arr, y, item, skip=()):
    """Fill the perimeter of level y with item (skip: set of (z, x) to leave as air)."""
    _, zn, xn = arr.shape
    for z in range(zn):
        for x in range(xn):
            if z in (0, zn - 1) or x in (0, xn - 1):
                if (z, x) not in skip:
                    arr[y, z, x] = item


def count_nonair(arr):
    return int(np.sum(arr != AIR))


# ---------------------------------------------------------------- handmade bases
def handmade():
    out = []

    # Nether portal: 4 wide x 5 tall obsidian frame, 2x3 air interior, 1 deep.
    a = new(5, 1, 4)
    a[0, 0, :] = "obsidian"
    a[4, 0, :] = "obsidian"
    a[1:4, 0, 0] = "obsidian"
    a[1:4, 0, 3] = "obsidian"
    out.append(("nether_portal", "portal", a, "Nether portal frame (4x5 obsidian, air inside)."))

    # Dirt hut, 4x4 footprint, 3 walls + flat roof, door on the front (z=0) at x=1.
    a = new(4, 4, 4)
    for y in range(3):
        ring(a, y, "dirt", skip={(0, 1)} if y < 2 else set())
    a[3, :, :] = "dirt"
    out.append(("dirt_hut", "shelter", a, "Dirt hut 4x4 with 2-high door and dirt roof."))

    # Small dirt hut 3x3, 2 walls + roof.
    a = new(3, 3, 3)
    ring(a, 0, "dirt", skip={(0, 1)})
    ring(a, 1, "dirt", skip={(0, 1)})
    a[2, :, :] = "dirt"
    out.append(("dirt_hut_small", "shelter", a, "Minimal 3x3 dirt hut, one-block doorway."))

    # Cobble cottage 5x5: planks floor, cobble walls, planks roof, door + two windows.
    a = new(4, 5, 5)
    a[0, :, :] = "oak_planks"
    for y in (1, 2):
        ring(a, y, "cobblestone", skip={(0, 2)})
    a[1:3, 0, 2] = AIR
    a[2, 2, 0] = "glass"
    a[2, 2, 4] = "glass"
    a[3, :, :] = "oak_planks"
    out.append(("cobble_cottage", "house", a, "5x5 cottage: cobblestone walls, planks floor and roof."))

    # Cobble house 7x7, 3 storeys of wall height.
    a = new(5, 7, 7)
    a[0, :, :] = "oak_planks"
    for y in (1, 2, 3):
        ring(a, y, "cobblestone", skip={(0, 3)})
    a[1:3, 0, 3] = AIR
    for z, x in ((3, 0), (3, 6), (0, 3), (6, 3)):
        a[2, z, x] = "glass"
    a[4, :, :] = "oak_planks"
    out.append(("cobble_house", "house", a, "7x7 two-room cobblestone house with glass windows and planks roof."))

    # Wooden house 7x7: planks walls with log corners, cobble floor.
    a = new(5, 7, 7)
    a[0, :, :] = "cobblestone"
    for y in (1, 2, 3):
        ring(a, y, "oak_planks", skip={(0, 3)})
        for z, x in ((0, 0), (0, 6), (6, 0), (6, 6)):
            a[y, z, x] = "oak_log"
    a[1:3, 0, 3] = AIR
    for z, x in ((3, 0), (3, 6), (0, 3), (6, 3)):
        a[2, z, x] = "glass"
    a[4, :, :] = "oak_planks"
    out.append(("wooden_house", "house", a, "7x7 planks house with log corners, cobble floor, glass windows."))

    # Watchtower 3x3 base, 9 high, cobble shaft, planks floor, battlement corners.
    a = new(9, 3, 3)
    a[0, :, :] = "oak_planks"
    for y in range(1, 8):
        ring(a, y, "cobblestone")
    a[1:3, 0, 1] = AIR
    a[4, 1, 0] = "glass"
    a[6, 1, 2] = "glass"
    for z, x in ((0, 0), (0, 2), (2, 0), (2, 2)):
        a[8, z, x] = "cobblestone"
    out.append(("watchtower", "tower", a, "3x3 cobblestone watchtower, 9 high, glass slits, battlement corners."))

    # Bridge 15 long, 3 wide: cobble pillars, planks deck, cobble rails.
    a = new(3, 3, 15)
    for x in (0, 7, 14):
        for z in (0, 2):
            a[0, z, x] = "cobblestone"
    a[1, :, :] = "oak_planks"
    a[2, 0, :] = "cobblestone"
    a[2, 2, :] = "cobblestone"
    out.append(("bridge", "bridge", a, "15-long planks bridge, cobble pillars at x=0,7,14 and cobble rails."))

    # Wall segment: 2-high cobble wall 9 long with crenellation.
    a = new(3, 1, 9)
    a[0, :, :] = "cobblestone"
    a[1, :, :] = "cobblestone"
    for x in range(0, 9, 2):
        a[2, 0, x] = "cobblestone"
    out.append(("wall_segment", "wall", a, "9-long cobblestone wall, 2 high, crenellated top."))

    # Farm plot 5x5: planks border around dirt beds.
    a = new(1, 5, 5)
    a[0, :, :] = "dirt"
    ring(a, 0, "oak_planks")
    out.append(("farm_plot", "farm", a, "5x5 farm plot: dirt bed with a planks border."))

    # Larger farm plot 9x9.
    a = new(1, 9, 9)
    a[0, :, :] = "dirt"
    ring(a, 0, "oak_planks")
    out.append(("farm_plot_large", "farm", a, "9x9 farm plot: dirt bed with a planks border."))

    # Well: cobble ring, log posts, planks cap ring.
    a = new(3, 3, 3)
    ring(a, 0, "cobblestone")
    for z, x in ((0, 0), (0, 2), (2, 0), (2, 2)):
        a[1, z, x] = "oak_log"
    ring(a, 2, "oak_planks")
    out.append(("well", "decor", a, "3x3 village well: cobble ring, four log posts, planks cap."))
    return out


# ---------------------------------------------------------------- vanilla decode
def decode_vanilla(entry):
    pal = entry["palette"]
    layers = entry["layers"]
    sy = len(layers)
    sz = len(layers[0])
    sx = len(layers[0][0])
    arr = new(sy, sz, sx)
    for y in range(sy):
        for z in range(sz):
            for x, c in enumerate(layers[y][z]):
                if c != AIR:
                    arr[y, z, x] = pal[c]
    return arr


# ---------------------------------------------------------------- encoding
def encode(arr):
    items = sorted({str(v) for v in np.unique(arr) if str(v) != AIR})
    if len(items) > len(ALPHABET):
        raise ValueError("palette too large")
    palette = {ALPHABET[i]: it for i, it in enumerate(items)}
    inv = {it: c for c, it in palette.items()}
    layers = []
    for y in range(arr.shape[0]):
        rows = []
        for z in range(arr.shape[1]):
            rows.append("".join(AIR if arr[y, z, x] == AIR else inv[str(arr[y, z, x])]
                                for x in range(arr.shape[2])))
        layers.append(rows)
    counts = {}
    for it in arr.ravel():
        if it != AIR:
            counts[str(it)] = counts.get(str(it), 0) + 1
    return palette, layers, dict(sorted(counts.items())), int(sum(counts.values()))


def tier_for(n):
    for t, (lo, hi) in TIER_RANGES.items():
        if lo <= n <= hi:
            return t
    return None


def record(name, family, arr, source, description=None, extra=None):
    palette, layers, counts, n = encode(arr)
    t = tier_for(n)
    if t is None:
        return None
    sy, sz, sx = arr.shape
    rec = {
        "id": None,
        "name": name,
        "family": family,
        "tier": t,
        "size": [sx, sy, sz],
        "palette": palette,
        "layers": layers,
        "anchor": "bottom_center",
        "required_items": counts,
        "block_count": n,
        "source": source,
        "description": description or "",
    }
    if extra:
        rec.update(extra)
    return rec


def same(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


def swap_material(arr):
    out = arr.copy()
    out[arr == "oak_planks"] = "cobblestone"
    out[arr == "cobblestone"] = "oak_planks"
    return out


def scale2(arr):
    return np.repeat(np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1), 2, axis=2)


def mirror_x(arr):
    return arr[:, :, ::-1].copy()


SWAP_BASES = ["cobble_cottage", "cobble_house", "wooden_house", "watchtower", "well", "farm_plot", "bridge"]
SCALE_BASES = ["wall_segment", "farm_plot", "well", "dirt_hut_small", "watchtower"]
MIRROR_BASES = ["cobble_cottage", "cobble_house", "wooden_house", "watchtower", "farm_plot", "well"]


def build():
    recs = []
    for name, fam, arr, desc in handmade():
        r = record(name, fam, arr, "handmade:" + name, desc)
        assert r is not None, name
        recs.append((r, arr))
    base_by_name = {r["name"]: (r, a) for r, a in recs}

    if os.path.exists(VANILLA):
        with open(VANILLA) as f:
            for c in json.load(f):
                arr = decode_vanilla(c)
                nm = "vanilla_" + c["source_name"]
                fam = "farm" if ("farm" in nm or "animal_pen" in nm) else ("decor" if "accessory" in nm else "house")
                r = record(nm, fam, arr, "vanilla:" + c["source"],
                           "Converted from vanilla village structure; see mapping_losses in README.",
                           {"vanilla_mapping_losses": c["mapping_losses"]})
                if r is not None:
                    recs.append((r, arr))
    vanilla_recs = [(r, a) for r, a in recs if r["source"].startswith("vanilla:")]
    seen = [a for _, a in recs]

    for base in SWAP_BASES:
        r0, a0 = base_by_name[base]
        a = swap_material(a0)
        if not same(a, a0):
            r = record(base + "__swap_material", r0["family"], a, "variant:%s:swap_planks_cobble" % base,
                       "Material swap planks<->cobblestone of " + base + ".")
            if r and not any(same(a, s) for s in seen):
                recs.append((r, a)); seen.append(a)
    for base in SCALE_BASES:
        r0, a0 = base_by_name[base]
        a = scale2(a0)
        r = record(base + "__x2", r0["family"], a, "variant:%s:scale_x2" % base,
                   "Uniform 2x scale of " + base + ".")
        if r and not any(same(a, s) for s in seen):
            recs.append((r, a)); seen.append(a)
    for base in MIRROR_BASES:
        r0, a0 = base_by_name[base]
        a = mirror_x(a0)
        if same(a, a0):
            continue
        r = record(base + "__mirror_x", r0["family"], a, "variant:%s:mirror_x" % base,
                   "Mirror-X of " + base + ".")
        if r and not any(same(a, s) for s in seen):
            recs.append((r, a)); seen.append(a)

    for r0, a0 in vanilla_recs:
        a = mirror_x(a0)
        if same(a, a0):
            continue
        r = record(r0["name"] + "__mirror_x", r0["family"], a, "variant:%s:mirror_x" % r0["name"],
                   "Mirror-X of converted vanilla house " + r0["name"] + ".")
        if r and not any(same(a, s) for s in seen):
            recs.append((r, a)); seen.append(a)

    out = []
    for i, (r, _) in enumerate(recs):
        r["id"] = i
        out.append(r)
    return out


def main():
    bps = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(bps, f, indent=1)
        f.write("\n")
    tiers = {1: 0, 2: 0, 3: 0}
    for b in bps:
        tiers[b["tier"]] += 1
    print("wrote %d blueprints to %s" % (len(bps), OUT))
    print("tiers:", tiers)
    fams = {}
    for b in bps:
        fams[b["family"]] = fams.get(b["family"], 0) + 1
    print("families:", fams)
    return 0


if __name__ == "__main__":
    sys.exit(main())
