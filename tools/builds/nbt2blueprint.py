"""Minimal NBT reader + vanilla structure (.nbt) -> MineMind blueprint converter.

Reads structure files straight out of the Minecraft jar (zipfile, no extraction).
Only stdlib + numpy are used. Run with the craftbot venv:
    .venv/bin/python tools/builds/nbt2blueprint.py --dump          # palette survey
    .venv/bin/python tools/builds/nbt2blueprint.py                 # convert SELECTION -> data/builds/vanilla_converted.json
"""
import gzip
import json
import os
import struct
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
JAR = os.path.expanduser("~/.gradle/caches/fabric-loom/26.2/minecraft-common.jar")
REGISTRY_ITEMS = os.path.join(ROOT, "data", "registry", "items.json")
OUT_VANILLA = os.path.join(ROOT, "data", "builds", "vanilla_converted.json")

# ---------------------------------------------------------------- NBT reader
TAG_END, TAG_BYTE, TAG_SHORT, TAG_INT, TAG_LONG, TAG_FLOAT, TAG_DOUBLE = 0, 1, 2, 3, 4, 5, 6
TAG_BYTE_ARRAY, TAG_STRING, TAG_LIST, TAG_COMPOUND, TAG_INT_ARRAY, TAG_LONG_ARRAY = 7, 8, 9, 10, 11, 12

_FMT = {TAG_BYTE: ">b", TAG_SHORT: ">h", TAG_INT: ">i", TAG_LONG: ">q", TAG_FLOAT: ">f", TAG_DOUBLE: ">d"}


class _Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def take(self, n):
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def scalar(self, t):
        f = _FMT[t]
        return struct.unpack(f, self.take(struct.calcsize(f)))[0]

    def string(self):
        n = struct.unpack(">H", self.take(2))[0]
        return self.take(n).decode("utf-8", "replace")

    def payload(self, t):
        if t in _FMT:
            return self.scalar(t)
        if t == TAG_STRING:
            return self.string()
        if t == TAG_BYTE_ARRAY:
            n = struct.unpack(">i", self.take(4))[0]
            return list(self.take(n))
        if t == TAG_INT_ARRAY:
            n = struct.unpack(">i", self.take(4))[0]
            return list(struct.unpack(">%di" % n, self.take(4 * n)))
        if t == TAG_LONG_ARRAY:
            n = struct.unpack(">i", self.take(4))[0]
            return list(struct.unpack(">%dq" % n, self.take(8 * n)))
        if t == TAG_LIST:
            et = self.scalar(TAG_BYTE)
            n = self.scalar(TAG_INT)
            return [self.payload(et) for _ in range(n)]
        if t == TAG_COMPOUND:
            out = {}
            while True:
                tt = self.scalar(TAG_BYTE)
                if tt == TAG_END:
                    return out
                name = self.string()
                out[name] = self.payload(tt)
        raise ValueError("bad NBT tag %d at %d" % (t, self.p))


def read_nbt(raw_gzipped_bytes):
    """Return the root compound (name discarded) of a gzipped NBT blob."""
    data = gzip.decompress(raw_gzipped_bytes)
    r = _Reader(data)
    t = r.scalar(TAG_BYTE)
    if t != TAG_COMPOUND:
        raise ValueError("root is not a compound")
    r.string()
    return r.payload(TAG_COMPOUND)


def read_structure(jar_path, member):
    with zipfile.ZipFile(jar_path) as z:
        return read_nbt(z.read(member))


def structure_to_grid(nbt):
    """Return (grid int array [x,y,z] of palette indices, -1 = empty, palette list of block names)."""
    sx, sy, sz = nbt["size"]
    palette = [p["Name"] for p in nbt["palette"]]
    grid = np.full((sx, sy, sz), -1, dtype=np.int32)
    for b in nbt["blocks"]:
        x, y, z = b["pos"]
        grid[x, y, z] = b["state"]
    return grid, palette


# ---------------------------------------------------------------- mapping
# Vanilla block name -> our item name (closest structural block), or None = air (decor / non-structural).
# Wood species collapse to oak_*. Stairs/slabs/doors/fences/etc. become the closest full block of the same material.
WOOD = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry", "pale_oak", "bamboo"]


def _map_block(name):
    n = name.replace("minecraft:", "")
    if n in ("air", "cave_air", "void_air"):
        return None
    # wood species collapse
    for w in WOOD:
        if n == w + "_planks" or n == w + "_slab" or n == w + "_stairs" or n == w + "_fence" \
                or n == w + "_fence_gate" or n == w + "_door" or n == w + "_trapdoor" or n == w + "_pressure_plate" \
                or n == w + "_button" or n == w + "_wall_sign" or n == w + "_sign":
            return "oak_planks"
        if n == w + "_log" or n == w + "_wood" or n == "stripped_" + w + "_log" or n == "stripped_" + w + "_wood":
            return "oak_log"
    if n.endswith("_wall_sign") or n.endswith("_sign") or n.endswith("_hanging_sign"):
        return None
    stone_like = ("cobblestone", "stone", "mossy_cobblestone", "stone_brick", "cobblestone_stairs", "cobblestone_slab",
                  "cobblestone_wall", "stone_bricks", "stone_brick_stairs", "stone_brick_slab", "stone_brick_wall",
                  "mossy_cobblestone_stairs", "mossy_cobblestone_slab", "mossy_cobblestone_wall", "mossy_stone_bricks",
                  "andesite", "diorite", "granite", "smooth_stone", "smooth_stone_slab", "tuff", "polished_andesite")
    if n in stone_like or n.startswith("stone_brick") or n.startswith("cobblestone") or n.startswith("mossy_cobblestone"):
        return "cobblestone"
    if n in ("stone", "smooth_stone", "stone_slab", "andesite", "diorite", "granite", "polished_andesite"):
        return "cobblestone"  # no stone-brick item in the palette; cobble is the closest structural stone
    if n in ("dirt", "coarse_dirt", "rooted_dirt", "podzol", "mycelium", "farmland", "dirt_path", "clay", "clay_block"):
        return "dirt"
    if n == "grass_block":
        return "dirt"
    if n.startswith("potted_") or n in ("large_fern", "brewing_stand", "sea_pickle", "tall_grass", "short_grass",
                                       "fern", "grass", "dead_bush"):
        return None
    if n in ("sand", "red_sand", "sandstone", "cut_sandstone", "smooth_sandstone", "chiseled_sandstone",
             "red_sandstone", "terracotta", "sandstone_stairs", "sandstone_slab", "sandstone_wall"):
        return "sand"
    if n in ("gravel",):
        return "gravel"
    if n in ("glass", "glass_pane", "white_stained_glass_pane", "tinted_glass"):
        return "glass"
    if n in ("obsidian", "crying_obsidian"):
        return "obsidian"
    if n in ("crafting_table",):
        return "crafting_table"
    if n in ("furnace", "smoker", "blast_furnace"):
        return "furnace"
    if n in ("chest", "trapped_chest", "barrel"):
        return "chest"
    if n in ("bricks", "brick_stairs", "brick_slab", "brick_wall"):
        return "cobblestone"
    if n in ("coal_ore", "deepslate_coal_ore"):
        return "cobblestone"
    if n in ("cobweb",):
        return None
    if n in ("bed", "white_bed", "red_bed") or n.endswith("_bed"):
        return None
    if n in ("torch", "wall_torch", "lantern", "campfire", "soul_campfire", "candle") or n.endswith("_torch") \
            or n.endswith("_lantern") or n.endswith("_candle"):
        return None
    if n.endswith("_pane") or n.endswith("_stained_glass") or n.endswith("_stained_glass_pane"):
        return "glass"
    if n.endswith("_wool") or n.endswith("_carpet") or n.endswith("_glazed_terracotta") or n.endswith("_concrete") \
            or n.endswith("_bed"):
        return None  # decorative: wool, carpet, glazed terracotta, concrete, beds -> air
    if n.endswith("_terracotta") or n == "terracotta":
        return "sand"
    if n.endswith("_flower") or n.endswith("_tulip") or n in ("dandelion", "poppy", "azure_bluet", "oxeye_daisy",
                                                             "sunflower", "lilac", "rose_bush", "peony", "lily_of_the_valley",
                                                             "cornflower", "wither_rose", "flower_pot", "potted_fern",
                                                             "dead_bush", "vine", "wheat", "carrots", "potatoes",
                                                             "beetroots", "melon_stem", "pumpkin_stem", "sweet_berry_bush",
                                                             "hay_block", "beehive", "bell", "composter", "water_cauldron",
                                                             "cauldron", "anvil", "grindstone", "lectern", "loom",
                                                             "smithing_table", "fletching_table", "cartography_table",
                                                             "stonecutter", "barrel", "bookshelf", "ladder", "iron_bars",
                                                             "chain", "iron_door", "iron_trapdoor", "rail", "water",
                                                             "lava", "snow", "snow_block", "ice", "packed_ice", "pumpkin",
                                                             "carved_pumpkin", "jack_o_lantern", "melon", "moss_block",
                                                             "bookshelf", "chiseled_bookshelf", "red_bed", "white_banner",
                                                             "cactus", "sugar_cane", "bamboo_sapling", "kelp", "seagrass",
                                                             "tnt", "end_rod", "lever", "redstone_torch", "redstone_wire",
                                                             "tripwire", "tripwire_hook", "string", "cobweb", "fire",
                                                             "soul_fire", "hanging_roots", "spore_blossom", "mushroom",
                                                             "brown_mushroom", "red_mushroom", "crimson_fungus",
                                                             "warped_fungus", "nether_sprouts", "dripstone_block"):
        return None
    if n.endswith("_banner") or n.endswith("_head") or n.endswith("_skull") or n.endswith("_pot"):
        return None
    if n.endswith("_leaves") or n in ("leaves", "azalea_leaves", "flowering_azalea_leaves"):
        return None
    if n.endswith("_sapling") or n.endswith("_mushroom_block") or n.endswith("_roots"):
        return None
    if n.endswith("_fence_gate") or n.endswith("_fence") or n.endswith("_wall") or n.endswith("_stairs") \
            or n.endswith("_slab") or n.endswith("_pressure_plate") or n.endswith("_button"):
        return "cobblestone"  # non-wood fences/walls/stairs/slabs: closest stone block
    if n in ("water", "lava", "bubble_column", "flowing_water", "flowing_lava"):
        return None
    if n.endswith("_ore"):
        return "cobblestone"
    if n in ("netherrack", "nether_bricks", "nether_brick_fence", "deepslate", "cobbled_deepslate", "blackstone"):
        return "cobblestone"
    if n in ("light", "structure_block", "jigsaw", "structure_void", "barrier", "spawner"):
        return None
    if n == "ice" or n == "blue_ice" or n == "packed_ice" or n == "frosted_ice":
        return "glass"
    if n.endswith("_planks"):
        return "oak_planks"
    if n.endswith("_log") or n.endswith("_wood") or n.endswith("_stem") or n.endswith("_hyphae"):
        return "oak_log"
    return "UNMAPPED"


def map_palette(palette):
    """Returns (list parallel to palette: our item name or None, list of unmapped vanilla names)."""
    mapped, unmapped = [], []
    for p in palette:
        m = _map_block(p)
        if m == "UNMAPPED":
            unmapped.append(p)
            m = None
        mapped.append(m)
    return mapped, unmapped


# ---------------------------------------------------------------- conversion
# Selection: (member path inside the jar). Chosen for variety of biome and size.
SELECTION = [
    "data/minecraft/structure/village/plains/houses/plains_small_house_5.nbt",
    "data/minecraft/structure/village/plains/houses/plains_accessory_1.nbt",
    "data/minecraft/structure/village/plains/houses/plains_animal_pen_1.nbt",
    "data/minecraft/structure/village/plains/houses/plains_armorer_house_1.nbt",
    "data/minecraft/structure/village/plains/houses/plains_butcher_shop_1.nbt",
    "data/minecraft/structure/village/plains/houses/plains_masons_house_1.nbt",
    "data/minecraft/structure/village/desert/houses/desert_small_house_5.nbt",
    "data/minecraft/structure/village/desert/houses/desert_fletcher_house_1.nbt",
    "data/minecraft/structure/village/savanna/houses/savanna_small_house_5.nbt",
    "data/minecraft/structure/village/savanna/houses/savanna_medium_house_1.nbt",
    "data/minecraft/structure/village/taiga/houses/taiga_small_house_5.nbt",
    "data/minecraft/structure/village/taiga/houses/taiga_small_house_4.nbt",
    "data/minecraft/structure/village/desert/houses/desert_farm_1.nbt",
    "data/minecraft/structure/village/taiga/houses/taiga_medium_house_1.nbt",
    "data/minecraft/structure/village/snowy/houses/snowy_small_house_7.nbt",
    "data/minecraft/structure/village/snowy/houses/snowy_armorer_house_1.nbt",
]

# Concrete tier thresholds (block count of non-air cells)
TIER_RANGES = {1: (5, 30), 2: (30, 120), 3: (120, 400)}


def tier_for(count):
    for t, (lo, hi) in TIER_RANGES.items():
        if lo <= count <= hi:
            return t
    return None


def convert_structure(jar_path, member, name=None):
    nbt = read_structure(jar_path, member)
    grid, palette = structure_to_grid(nbt)
    mapped, unmapped = map_palette(palette)
    sx, sy, sz = grid.shape

    # Palette indices -> item names (None -> air)
    lost = {}  # vanilla name -> count of cells dropped (air-mapped non-air)
    char_for = {}
    item_chars = {}
    palette_out = {}
    alphabet = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

    def char_of(item):
        if item not in item_chars:
            c = next(alphabet)
            item_chars[item] = c
            palette_out[c] = item
        return item_chars[item]

    # Cells: compute item per cell (-1 empty)
    layers = []
    total_non_air = 0
    counts = {}
    for y in range(sy):
        rows = []
        for z in range(sz):
            row = []
            for x in range(sx):
                idx = int(grid[x, y, z])
                if idx < 0:
                    row.append(".")
                    continue
                vname = palette[idx].replace("minecraft:", "")
                item = mapped[idx]
                if item is None:
                    if vname not in ("air", "cave_air", "void_air"):
                        lost[vname] = lost.get(vname, 0) + 1
                    row.append(".")
                    continue
                total_non_air += 1
                counts[item] = counts.get(item, 0) + 1
                row.append(char_of(item))
            rows.append("".join(row))
        layers.append(rows)
    # Trim empty ground layers at the bottom and top so the blueprint is tight.
    def is_empty(layer):
        return all(set(r) == {"."} for r in layer)
    while layers and is_empty(layers[0]):
        layers.pop(0)
    while layers and is_empty(layers[-1]):
        layers.pop()
    if not layers:
        return None
    height = len(layers)
    width = max(len(r) for r in layers[0]) if layers else 0
    # Tighten X and Z
    all_rows = [r for layer in layers for r in layer]
    xs = [i for r in all_rows for i, c in enumerate(r) if c != "."]
    xmin, xmax = min(xs), max(xs)
    zs = [zi for layer in layers for zi, r in enumerate(layer) if any(c != "." for c in r)]
    zmin, zmax = min(zs), max(zs)
    layers = [[r[xmin:xmax + 1] for r in layer[zmin:zmax + 1]] for layer in layers]
    size = [xmax - xmin + 1, height, zmax - zmin + 1]

    used = {}
    for layer in layers:
        for r in layer:
            for c in r:
                if c != ".":
                    used[c] = palette_out[c]
    # Stable palette names
    cnt = sum(counts.values())
    base = name or os.path.basename(member)[:-4]
    return {
        "source": member,
        "source_name": base,
        "size": size,
        "palette": used,
        "layers": layers,
        "block_count": cnt,
        "tier": tier_for(cnt),
        "mapping_losses": {
            "unmapped_vanilla_blocks": sorted(unmapped),
            "dropped_non_structural": dict(sorted(lost.items(), key=lambda kv: -kv[1])),
        },
        "structure_original_size": list(nbt["size"]),
        "structure_palette_size": len(palette),
    }


def convert_all(jar_path=JAR, selection=SELECTION):
    out = []
    for m in selection:
        c = convert_structure(jar_path, m)
        if c is None:
            print("skip empty", m, file=sys.stderr)
            continue
        out.append(c)
    return out


def dump(jar_path=JAR):
    with zipfile.ZipFile(jar_path) as z:
        members = [m for m in z.namelist() if m.startswith("data/minecraft/structure/village/") and m.endswith(".nbt")]
    names = {}
    for m in members:
        nbt = read_structure(jar_path, m)
        grid, palette = structure_to_grid(nbt)
        for p in palette:
            names[p] = names.get(p, 0) + 1
    for p, c in sorted(names.items(), key=lambda kv: -kv[1]):
        print("%4d  %s  -> %s" % (c, p, _map_block(p)))


def main(argv):
    if "--dump" in argv:
        dump()
        return 0
    if "--mapping-only" in argv:
        return 0
    conv = convert_all()
    os.makedirs(os.path.dirname(OUT_VANILLA), exist_ok=True)
    with open(OUT_VANILLA, "w") as f:
        json.dump(conv, f, indent=1)
    for c in conv:
        print("%-60s size=%s blocks=%d tier=%s unmapped=%s" % (
            c["source"], c["size"], c["block_count"], c["tier"], c["mapping_losses"]["unmapped_vanilla_blocks"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
