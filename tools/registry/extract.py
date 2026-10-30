#!/usr/bin/env python3
"""DATA-1: build data/registry/{blocks,items,recipes,mobs}.json for MineMind v7 (PROTOCOL 12.2).

Sources (read-only):
  - jar  ~/.gradle/caches/fabric-loom/26.2/minecraft-common.jar
         data/minecraft/recipe/*.json, loot_table/blocks|entities/*.json, tags/{block,item}/**.json
  - mcsrc/net/minecraft/...  Blocks.java (hardness, requiresCorrectToolForDrops), Items.java
         (tool attack stats, stack sizes), ToolMaterial.java, food/Foods.java, monster/animal
         createAttributes() (health, attack, speed).
Hand-set values (not in those files) are listed in README.md as "assumed" and marked in the
ASSUMED_* tables below.

Run: .venv/bin/python tools/registry/extract.py   (then tools/registry/validate.py)
"""
import json
import os
import re
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JAR = os.path.expanduser("~/.gradle/caches/fabric-loom/26.2/minecraft-common.jar")
MC = os.path.join(ROOT, "mcsrc", "net", "minecraft")
OUT = os.path.join(ROOT, "data", "registry")

_zip = zipfile.ZipFile(JAR)
_names = set(_zip.namelist())


def jar_json(path):
    return json.loads(_zip.read(path).decode("utf-8"))


def src(rel):
    with open(os.path.join(MC, rel), encoding="utf-8") as f:
        return f.read()


def strip_ns(s):
    return s[len("minecraft:"):] if s.startswith("minecraft:") else s


# ---------------------------------------------------------------- scope (v6 ids fixed)
# Blocks: (name, id). v6 ids 0..19 are fixed by PROTOCOL 12.2; the rest are appended.
BLOCK_ORDER = [
    "air", "grass", "dirt", "stone", "cobblestone", "oak_log", "oak_leaves", "oak_planks",
    "crafting_table", "furnace", "coal_ore", "iron_ore", "diamond_ore", "bedrock", "sand", "gravel",
    "water", "deepslate", "chest", "agent",
    # mob pseudo-blocks (one id per mob type, PROTOCOL 12.4 / 12.2); ids 20..28 as in the CORE seed
    "zombie", "skeleton", "spider", "creeper", "cow", "pig", "sheep", "chicken", "zombified_piglin",
    "lava", "glass", "obsidian", "netherrack", "nether_quartz_ore", "soul_sand", "glowstone",
    "magma_block", "gold_ore", "deepslate_coal_ore", "deepslate_iron_ore", "deepslate_diamond_ore",
    "deepslate_gold_ore", "cobbled_deepslate", "granite", "diorite", "andesite", "stone_bricks",
    "cobblestone_stairs", "cobblestone_slab", "oak_stairs", "oak_slab", "oak_fence", "oak_door",
    "oak_trapdoor", "ladder", "torch", "white_wool", "white_bed", "wheat",
]
# vanilla registry name for our (v6-named) block ids
VANILLA_NAME = {"grass": "grass_block"}
PSEUDO = {"agent", "zombie", "skeleton", "spider", "creeper", "cow", "pig", "sheep", "chicken",
          "zombified_piglin"}

# Block physical properties: not in the data files (assumed, hand-set).
# (solid, transparent, liquid, light_emission, color, dimension)
BLOCK_PROPS = {
    "air": (False, True, False, 0, [0, 0, 0], "both"),
    "grass": (True, False, False, 0, [106, 170, 64], "overworld"),
    "dirt": (True, False, False, 0, [134, 96, 67], "overworld"),
    "stone": (True, False, False, 0, [125, 125, 125], "overworld"),
    "cobblestone": (True, False, False, 0, [110, 110, 110], "both"),
    "oak_log": (True, False, False, 0, [102, 81, 49], "both"),
    "oak_leaves": (True, True, False, 0, [56, 110, 40], "overworld"),
    "oak_planks": (True, False, False, 0, [162, 130, 78], "both"),
    "crafting_table": (True, False, False, 0, [150, 110, 60], "both"),
    "furnace": (True, False, False, 0, [115, 115, 115], "both"),
    "coal_ore": (True, False, False, 0, [100, 100, 100], "overworld"),
    "iron_ore": (True, False, False, 0, [136, 120, 110], "overworld"),
    "diamond_ore": (True, False, False, 0, [100, 160, 160], "overworld"),
    "bedrock": (True, False, False, 0, [60, 60, 60], "both"),
    "sand": (True, False, False, 0, [219, 211, 160], "overworld"),
    "gravel": (True, False, False, 0, [136, 130, 128], "overworld"),
    "water": (False, True, True, 0, [60, 90, 220], "both"),
    "deepslate": (True, False, False, 0, [80, 80, 84], "overworld"),
    "chest": (True, False, False, 0, [160, 115, 50], "both"),
    "agent": (False, True, False, 0, [255, 255, 255], "both"),
    "lava": (False, True, True, 15, [230, 100, 20], "both"),
    "glass": (True, True, False, 0, [200, 230, 240], "both"),
    "obsidian": (True, False, False, 0, [20, 15, 30], "both"),
    "netherrack": (True, False, False, 0, [110, 40, 40], "nether"),
    "nether_quartz_ore": (True, False, False, 0, [150, 100, 90], "nether"),
    "soul_sand": (True, False, False, 0, [85, 65, 50], "nether"),
    "glowstone": (True, False, False, 15, [230, 190, 100], "nether"),
    "magma_block": (True, False, False, 3, [170, 80, 30], "nether"),
    "gold_ore": (True, False, False, 0, [150, 140, 70], "overworld"),
    "deepslate_coal_ore": (True, False, False, 0, [70, 70, 74], "overworld"),
    "deepslate_iron_ore": (True, False, False, 0, [90, 80, 80], "overworld"),
    "deepslate_diamond_ore": (True, False, False, 0, [70, 110, 110], "overworld"),
    "deepslate_gold_ore": (True, False, False, 0, [100, 90, 50], "overworld"),
    "cobbled_deepslate": (True, False, False, 0, [70, 70, 74], "overworld"),
    "granite": (True, False, False, 0, [150, 100, 80], "overworld"),
    "diorite": (True, False, False, 0, [190, 190, 190], "overworld"),
    "andesite": (True, False, False, 0, [130, 130, 130], "overworld"),
    "stone_bricks": (True, False, False, 0, [120, 120, 120], "both"),
    "cobblestone_stairs": (True, True, False, 0, [110, 110, 110], "both"),
    "cobblestone_slab": (True, True, False, 0, [110, 110, 110], "both"),
    "oak_stairs": (True, True, False, 0, [162, 130, 78], "both"),
    "oak_slab": (True, True, False, 0, [162, 130, 78], "both"),
    "oak_fence": (True, True, False, 0, [162, 130, 78], "both"),
    "oak_door": (True, True, False, 0, [160, 125, 70], "both"),
    "oak_trapdoor": (True, True, False, 0, [160, 125, 70], "both"),
    "ladder": (False, True, False, 0, [150, 120, 70], "both"),
    "torch": (False, True, False, 14, [250, 200, 80], "both"),
    "white_wool": (True, False, False, 0, [230, 230, 230], "both"),
    "white_bed": (True, True, False, 0, [235, 235, 235], "both"),
    "wheat": (False, True, False, 0, [200, 170, 70], "overworld"),
    "zombie": (False, True, False, 0, [60, 120, 60], "overworld"),
    "skeleton": (False, True, False, 0, [220, 220, 210], "overworld"),
    "spider": (False, True, False, 0, [60, 50, 50], "overworld"),
    "creeper": (False, True, False, 0, [60, 160, 60], "overworld"),
    "cow": (False, True, False, 0, [120, 90, 60], "overworld"),
    "pig": (False, True, False, 0, [240, 160, 160], "overworld"),
    "sheep": (False, True, False, 0, [240, 240, 240], "overworld"),
    "chicken": (False, True, False, 0, [250, 250, 240], "overworld"),
    "zombified_piglin": (False, True, False, 0, [150, 190, 120], "nether"),
}

# Hardness fallbacks for blocks whose declaration could not be parsed (assumed vanilla values).
ASSUMED_HARDNESS = {"water": -1.0, "lava": -1.0, "wheat": 0.0, "air": 0.0, "agent": 0.0,
                    "white_wool": 0.8, "white_bed": 0.2}  # wool/bed declared via ColorCollection (not parsed)
ASSUMED_HARDNESS.update({m: 0.0 for m in PSEUDO})

# Item order: v6 ids 0..14 fixed (PROTOCOL 12.2), then appended.
ITEM_ORDER = [
    "oak_log", "oak_planks", "stick", "crafting_table", "wooden_pickaxe", "cobblestone",
    "stone_pickaxe", "furnace", "coal", "raw_iron", "iron_ingot", "iron_pickaxe", "diamond",
    "dirt", "chest",
    "stone", "sand", "gravel", "flint", "glass", "obsidian", "netherrack", "soul_sand", "glowstone",
    "glowstone_dust", "cobbled_deepslate", "granite", "diorite", "andesite", "stone_bricks",
    "magma_block", "raw_gold", "gold_ingot", "quartz", "cobblestone_stairs", "cobblestone_slab",
    "oak_stairs", "oak_slab", "oak_fence", "oak_door", "oak_trapdoor", "ladder", "torch",
    "white_wool", "white_bed", "wheat", "string", "bone", "arrow", "leather", "feather", "gunpowder",
    "ender_pearl", "bucket", "water_bucket", "lava_bucket", "flint_and_steel",
    "wooden_axe", "wooden_shovel", "wooden_sword", "wooden_hoe",
    "stone_axe", "stone_shovel", "stone_sword", "stone_hoe",
    "iron_axe", "iron_shovel", "iron_sword", "iron_hoe",
    "diamond_pickaxe", "diamond_axe", "diamond_shovel", "diamond_sword", "diamond_hoe",
    "iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots",
    "diamond_helmet", "diamond_chestplate", "diamond_leggings", "diamond_boots",
    "leather_helmet", "leather_chestplate", "leather_leggings", "leather_boots",
    "apple", "bread", "beef", "cooked_beef", "porkchop", "cooked_porkchop", "mutton",
    "cooked_mutton", "chicken", "cooked_chicken", "rotten_flesh",
]
# Items that are block items (kind "block", places_block = same name); the rest use tool/food/etc.
ITEM_PLACES = {
    "oak_log": "oak_log", "oak_planks": "oak_planks", "crafting_table": "crafting_table",
    "furnace": "furnace", "cobblestone": "cobblestone", "dirt": "dirt", "chest": "chest",
    "stone": "stone", "sand": "sand", "gravel": "gravel", "glass": "glass", "obsidian": "obsidian",
    "netherrack": "netherrack", "soul_sand": "soul_sand", "glowstone": "glowstone",
    "magma_block": "magma_block", "cobbled_deepslate": "cobbled_deepslate", "granite": "granite",
    "diorite": "diorite", "andesite": "andesite", "stone_bricks": "stone_bricks",
    "cobblestone_stairs": "cobblestone_stairs", "cobblestone_slab": "cobblestone_slab",
    "oak_stairs": "oak_stairs", "oak_slab": "oak_slab", "oak_fence": "oak_fence",
    "oak_door": "oak_door", "oak_trapdoor": "oak_trapdoor", "ladder": "ladder", "torch": "torch",
    "white_wool": "white_wool", "white_bed": "white_bed", "water_bucket": "water",
    "lava_bucket": "lava",
}
# Fuel burn ticks (furnace). ASSUMED vanilla values, not in the files.
ASSUMED_FUEL = {
    "coal": 1600, "oak_log": 300, "oak_planks": 300, "stick": 100, "crafting_table": 300,
    "chest": 300, "oak_fence": 300, "oak_stairs": 300, "oak_slab": 150, "oak_door": 200,
    "oak_trapdoor": 300, "ladder": 300, "white_wool": 100, "white_bed": 300, "torch": 200,
    "wooden_axe": 200, "wooden_shovel": 200, "wooden_sword": 200, "wooden_hoe": 200,
    "wooden_pickaxe": 200,
}
# Shaping value = 2**depth (MineRL-style); depth = tech depth, ASSUMED hand-set.
ITEM_DEPTH = {
    0: ["oak_log", "cobblestone", "dirt", "stone", "sand", "gravel", "coal", "raw_iron", "raw_gold",
        "diamond", "string", "bone", "feather", "flint", "leather", "gunpowder", "wheat",
        "ender_pearl", "arrow", "glowstone_dust", "quartz", "netherrack", "soul_sand", "glass",
        "obsidian", "granite", "diorite", "andesite", "cobbled_deepslate", "magma_block",
        "glowstone", "apple", "beef", "porkchop", "mutton", "chicken", "rotten_flesh",
        "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken", "bread",
        "white_wool", "bucket", "water_bucket", "lava_bucket"],
    1: ["oak_planks", "stick", "crafting_table", "chest", "furnace", "stone_bricks", "ladder",
        "torch", "oak_fence", "oak_door", "oak_trapdoor", "oak_stairs", "oak_slab",
        "cobblestone_stairs", "cobblestone_slab", "white_bed", "flint_and_steel", "gold_ingot",
        "iron_ingot"],
    2: ["wooden_pickaxe", "wooden_axe", "wooden_shovel", "wooden_sword", "wooden_hoe",
        "stone_pickaxe", "stone_axe", "stone_shovel", "stone_sword", "stone_hoe",
        "iron_helmet", "iron_chestplate", "iron_leggings", "iron_boots", "leather_helmet",
        "leather_chestplate", "leather_leggings", "leather_boots"],
    3: ["iron_pickaxe", "iron_axe", "iron_shovel", "iron_sword", "iron_hoe"],
    4: ["diamond", "diamond_pickaxe", "diamond_axe", "diamond_shovel", "diamond_sword",
        "diamond_hoe", "diamond_helmet", "diamond_chestplate", "diamond_leggings",
        "diamond_boots"],
}
TOOL_TIER = {"WOOD": 1, "STONE": 2, "IRON": 3, "DIAMOND": 4, "GOLD": 1, "NETHERITE": 5, "COPPER": 2}
# ASSUMED armor stats (vanilla). Stored as the extension key "armor".
ASSUMED_ARMOR = {
    "leather": {"helmet": (55, 1), "chestplate": (80, 3), "leggings": (75, 2), "boots": (65, 1)},
    "iron": {"helmet": (165, 2), "chestplate": (240, 6), "leggings": (225, 5), "boots": (195, 2)},
    "diamond": {"helmet": (363, 3), "chestplate": (528, 8), "leggings": (495, 6), "boots": (429, 3)},
}

# Mob specs: spawn / size / attack are ASSUMED (vanilla); health/damage/speed parsed from mcsrc
# where the class declares them (see parse_attrs). drops parsed from loot_table/entities.
MOBS = {
    "zombie": dict(src="monster/zombie/Zombie.java", hostile=True, size=[0.6, 1.95], attack="melee",
                   burns=True, dimension="overworld", light_max=7, day="night", group=(1, 4)),
    "skeleton": dict(src="monster/skeleton/AbstractSkeleton.java", hostile=True, size=[0.6, 1.99],
                     attack="ranged", burns=True, dimension="overworld", light_max=7, day="night",
                     group=(1, 3)),
    "spider": dict(src="monster/spider/Spider.java", hostile=True, size=[1.4, 0.9], attack="melee",
                   burns=False, dimension="overworld", light_max=7, day="night", group=(1, 1)),
    "creeper": dict(src="monster/Creeper.java", hostile=True, size=[0.6, 1.7], attack="explode",
                    burns=False, dimension="overworld", light_max=7, day="night", group=(1, 1)),
    "cow": dict(src="animal/cow/Cow.java", hostile=False, size=[0.9, 1.4], attack="none",
                burns=False, dimension="overworld", light_max=15, day="day", group=(2, 4)),
    "pig": dict(src="animal/pig/Pig.java", hostile=False, size=[0.9, 0.9], attack="none",
                burns=False, dimension="overworld", light_max=15, day="day", group=(2, 4)),
    "sheep": dict(src="animal/sheep/Sheep.java", hostile=False, size=[0.9, 1.3], attack="none",
                  burns=False, dimension="overworld", light_max=15, day="day", group=(2, 4)),
    "chicken": dict(src="animal/chicken/Chicken.java", hostile=False, size=[0.4, 0.7],
                    attack="none", burns=False, dimension="overworld", light_max=15, day="day",
                    group=(2, 4)),
    "zombified_piglin": dict(src="monster/zombie/ZombifiedPiglin.java", hostile=True,
                             size=[0.6, 1.95], attack="melee", burns=False, dimension="nether",
                             light_max=7, day="any", group=(1, 3)),
}
# Fallback attribute defaults (vanilla Attributes defaults; used when a class does not override).
DEFAULT_HEALTH = 20.0
DEFAULT_ATTACK = 2.0
DEFAULT_SPEED = 0.25  # ASSUMED for mobs whose class does not declare MOVEMENT_SPEED
# ASSUMED per-mob values for classes whose createAttributes() is not in mcsrc (Cow.java).
MOB_FALLBACK = {"cow": {"MAX_HEALTH": 10.0, "MOVEMENT_SPEED": 0.2}}


# ---------------------------------------------------------------- tags
def tag_values(kind, name, depth=0):
    """Flatten a minecraft tag (kind 'block' or 'item') into ordered item/block names."""
    path = f"data/minecraft/tags/{kind}/{name}.json"
    if path not in _names or depth > 8:
        return []
    out = []
    for v in jar_json(path).get("values", []):
        if isinstance(v, dict):
            v = v.get("id", "")
        if v.startswith("#"):
            out += tag_values(kind, strip_ns(v[1:]), depth + 1)
        else:
            out.append(strip_ns(v))
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


# ---------------------------------------------------------------- blocks (Blocks.java)
def java_block_decls(text):
    """Map lowercase block name -> declaration body text (balanced parentheses)."""
    decls = {}
    for m in re.finditer(r"public static final Block (\w+) = (?:register|register\w*)\(", text):
        i = m.end() - 1
        depth, j = 0, i
        while j < len(text):
            c = text[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        decls[m.group(1).lower()] = m.group(0) + text[m.end():j]  # keep call head (registerLegacyStair)
    return decls


def helper_body(text, name):
    m = re.search(r"static (?:BlockBehaviour\.)?Properties " + name + r"\(", text)
    if not m:
        return ""
    end = text.find("\n\t}", m.end())
    return text[m.end(): end if end > 0 else m.end() + 800]


def parse_block_strength(text, body, decls=None, depth=0):
    """Return (hardness, requires_tool) from a declaration body, following property helpers and
    legacy copies (ofLegacyCopy(X) / registerLegacyStair(name, X)) to the parent block."""
    parent = None
    pm = re.search(r"ofLegacyCopy\((\w+)\)", body) or re.search(r"registerLegacyStair\(\s*[\w.]+,\s*(\w+)\)", body)
    if pm:
        parent = pm.group(1)
    def find(b):
        s = re.search(r"strength\(\s*(-?[\d.]+)F?(?:\s*,\s*-?[\d.]+F?)?\s*\)", b)
        if s:
            return float(s.group(1))
        if ".instabreak()" in b:
            return 0.0
        return None
    h = find(body)
    if h is None:
        for helper in re.findall(r"(\w+Properties)\(", body):
            h = find(helper_body(text, helper))
            if h is not None:
                break
    req = "requiresCorrectToolForDrops" in body
    if parent and decls is not None and depth < 5 and parent.lower() in decls:
        ph, preq = parse_block_strength(text, decls[parent.lower()], decls, depth + 1)
        if h is None:
            h = ph
        req = req or preq
    return h, req


def load_blocks_from_source():
    text = src("world/level/block/Blocks.java")
    decls = java_block_decls(text)
    info = {}
    for name in BLOCK_ORDER:
        if name in PSEUDO or name in ("air", "agent"):
            continue
        body = decls.get(VANILLA_NAME.get(name, name))
        if body is None:
            info[name] = (None, False, "missing")
            continue
        h, req = parse_block_strength(text, body, decls)
        info[name] = (h, req, "src" if h is not None else "missing")
    return info


# ---------------------------------------------------------------- loot tables (drops)
def _chance(conds):
    for c in conds:
        k = c.get("condition", "")
        if k == "minecraft:random_chance":
            return float(c.get("chance", 1.0))
        if k == "minecraft:random_chance_with_enchanted_bonus":
            return float(c.get("unenchanted_chance", 1.0))
        if k == "minecraft:random_chance_with_looting":
            return float(c.get("chance", 1.0))
        if k == "minecraft:table_bonus":
            ch = c.get("chances", [1.0])
            return float(ch[0])
    return 1.0


def _excluded(conds):
    """Silk-touch / shears-only leaves are not the default drop."""
    for c in conds:
        k = c.get("condition", "")
        if k in ("minecraft:match_tool", "minecraft:any_of"):
            s = json.dumps(c)
            if "silk_touch" in s or "shears" in s:
                return True
    return False


def _count(funcs):
    lo, hi = 1, 1
    for f in funcs or []:
        if f.get("function") == "minecraft:set_count" and not f.get("add", False):
            c = f.get("count", 1)
            if isinstance(c, (int, float)):
                lo = hi = int(c)
            elif c.get("type") == "minecraft:uniform":
                lo, hi = int(c["min"]), int(c["max"])
            elif c.get("type") == "minecraft:constant":
                lo = hi = int(c["value"])
            elif c.get("type") == "minecraft:binomial":
                lo, hi = 0, int(c.get("n", 1))
    return lo, hi


def _leaves(entry, conds_acc):
    """Return list of (item, lo, hi, chance) for an entry (ignores silk/shears-only branches)."""
    t = entry.get("type", "")
    conds = list(conds_acc) + list(entry.get("conditions", []))
    if t == "minecraft:item":
        if _excluded(conds):
            return []
        lo, hi = _count(entry.get("functions", []))
        return [(strip_ns(entry["name"]), lo, hi, _chance(conds))]
    if t == "minecraft:loot_table":
        # a reference to another table (sheep -> entities/sheep/<color>); inline its drops under our conditions
        ref = entry.get("value")
        if not isinstance(ref, str) or _excluded(conds):
            return []
        sub = loot_drops(f"data/minecraft/loot_table/{strip_ns(ref)}.json") or []
        c = _chance(conds)
        return [(it, lo, hi, ch * c) for it, lo, hi, ch in sub]
    if t == "minecraft:alternatives":
        # first matching child wins; a child that only fires by chance (gravel: flint 10%) leaves the
        # remaining probability to the next children (gravel 90%)
        out, remaining = [], 1.0
        for ch in entry.get("children", []):
            got = _leaves(ch, conds)
            if not got:
                continue
            p = max(c for _, _, _, c in got)
            out += [(it, lo, hi, c * remaining) for it, lo, hi, c in got]
            remaining *= 1.0 - p
            if remaining <= 1e-6:
                break
        return out
    out = []
    for ch in entry.get("children", []) + entry.get("entries", []):
        out += _leaves(ch, conds)
    return out


def loot_drops(path):
    """Simplified drops from a loot table: [(item, lo, hi, chance)]."""
    if path not in _names:
        return None
    table = jar_json(path)
    out = []
    for pool in table.get("pools", []):
        pconds = [c for c in pool.get("conditions", [])]
        for e in pool.get("entries", []):
            out += _leaves(e, pconds)
    return out


# ---------------------------------------------------------------- items (Items.java, ToolMaterial, Foods)
def load_tool_materials():
    text = src("world/item/ToolMaterial.java")
    mats = {}
    for m in re.finditer(r"public static final ToolMaterial (\w+) = new ToolMaterial\(BlockTags\.\w+, (\d+), ([\d.]+)F, ([\d.]+)F, \d+", text):
        mats[m.group(1)] = dict(durability=int(m.group(2)), speed=float(m.group(3)),
                                bonus=float(m.group(4)))
    return mats


def load_foods():
    text = src("world/food/Foods.java")
    foods = {}
    for m in re.finditer(r"public static final FoodProperties (\w+) = new FoodProperties\.Builder\(\)\.nutrition\((\d+)\)\.saturationModifier\(([\d.]+)F?\)", text):
        foods[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    return foods


def load_item_tools(mats):
    """item name -> tool dict from Items.java declarations."""
    text = src("world/item/Items.java")
    out = {}
    for m in re.finditer(r"public static final Item (\w+) = (?:registerItem)\(", text):
        name = m.group(1).lower()
        end = text.find(";\n", m.end())
        body = text[m.end(): end]
        t = re.search(r"ToolMaterial\.(\w+),\s*(-?[\d.]+)F,\s*(-?[\d.]+)F", body)
        if not t:
            continue
        mat, base = t.group(1), float(t.group(2))
        kind = name.split("_")[-1]  # pickaxe/axe/shovel/hoe/sword
        if kind not in ("pickaxe", "axe", "shovel", "hoe", "sword"):
            continue
        mm = mats[mat]
        # Melee damage as the player deals it: 1 (player base) + item base + material bonus.
        damage = 1.0 + base + mm["bonus"]
        stack_m = re.search(r"stacksTo\((\d+)\)", body)
        out[name] = dict(type=kind, tier=TOOL_TIER[mat], speed=mm["speed"],
                         damage=damage, durability=mm["durability"],
                         attack_speed=float(t.group(3)))
    return out


def load_item_stacks():
    text = src("world/item/Items.java")
    st = {}
    for m in re.finditer(r"public static final Item (\w+) = (?:registerItem)\(", text):
        end = text.find(";\n", m.end())
        body = text[m.end(): end]
        s = re.search(r"stacksTo\((\d+)\)", body)
        if s:
            st[m.group(1).lower()] = int(s.group(1))
    return st


def load_java_fallback_tools():
    return {}


# ---------------------------------------------------------------- mob attributes
def parse_attrs(rel, depth=0):
    """Return dict(MAX_HEALTH, ATTACK_DAMAGE, MOVEMENT_SPEED) declared in createAttributes chain."""
    got = {}
    try:
        text = src("world/entity/" + rel)
    except FileNotFoundError:
        return got
    m = re.search(r"static AttributeSupplier\.Builder createAttributes\(\)\s*\{(.*?)\n\t\}", text, re.S)
    if not m:
        return got
    body = m.group(1)
    # inherit from parent class createAttributes (e.g. Zombie.createAttributes())
    parent = re.search(r"(\w+)\.createAttributes\(\)", body)
    if parent and parent.group(1) not in ("Monster", "Animal", "Mob"):
        pmap = {"Zombie": "monster/zombie/Zombie.java"}  # relative to world/entity/
        if parent.group(1) in pmap and depth < 3:
            got.update(parse_attrs(pmap[parent.group(1)], depth + 1))
    for a in re.finditer(r"Attributes\.(\w+),?\s*([\d.]+)F?\)", body):
        got[a.group(1)] = float(a.group(2))
    # "add(Attributes.X)" without value means default; keep parsed values only
    return got


# ---------------------------------------------------------------- build
def build():
    mats = load_tool_materials()
    foods = load_foods()
    tools = load_item_tools(mats)
    stacks = load_item_stacks()
    block_src = load_blocks_from_source()

    # ---- blocks
    tag_pick = {k: set(tag_values("block", "mineable/" + k))
                for k in ("pickaxe", "axe", "shovel", "hoe", "sword")}
    need = {4: set(tag_values("block", "needs_diamond_tool")),
            3: set(tag_values("block", "needs_iron_tool")),
            2: set(tag_values("block", "needs_stone_tool"))}
    blocks = []
    for bid, name in enumerate(BLOCK_ORDER):
        solid, transp, liquid, light, color, dim = BLOCK_PROPS[name]
        pseudo = name in PSEUDO
        if pseudo or name == "air":
            hard, req, hsrc = ASSUMED_HARDNESS.get(name, 0.0), False, "pseudo" if pseudo else "air"
        else:
            hard, req, hsrc = block_src.get(name, (None, False, "missing"))
            if hard is None:
                hard, hsrc = ASSUMED_HARDNESS.get(name, 0.0), "assumed"
            if name in ASSUMED_HARDNESS and hsrc == "missing":
                hard, hsrc = ASSUMED_HARDNESS[name], "assumed"
        best = None
        for k in ("pickaxe", "shovel", "axe", "hoe"):
            if name in tag_pick[k]:
                best = k
                break
        min_tier = 0
        for tier in (4, 3, 2):
            if name in need[tier]:
                min_tier = tier
                break
        if min_tier == 0 and req and not pseudo:
            min_tier = 1
        blocks.append(dict(name=name, id=bid, solid=solid, transparent=transp, liquid=liquid,
                           hardness=hard, best_tool=best, min_tier=min_tier, drops=[],
                           light_emission=light, color=color, dimension=dim, pseudo=pseudo,
                           _hardness_source=hsrc, _req_tool=req))

    # drops (filled after items are known)
    # ---- items
    items = []
    depth_of = {}
    for d, names in ITEM_DEPTH.items():
        for n in names:
            depth_of[n] = d
    for iid, name in enumerate(ITEM_ORDER):
        kind, tool, food, places = "material", None, None, ITEM_PLACES.get(name)
        stack = stacks.get(name, 64)
        if name in tools:
            t = tools[name]
            kind = "weapon" if t["type"] == "sword" else "tool"
            tool = dict(type=t["type"], tier=t["tier"], speed=t["speed"], damage=t["damage"],
                        durability=t["durability"])
            stack = 1
        armor = None
        for mat in ("leather", "iron", "diamond"):
            for slot in ("helmet", "chestplate", "leggings", "boots"):
                if name == f"{mat}_{slot}":
                    dur, defense = ASSUMED_ARMOR[mat][slot]
                    kind, stack = "armor", 1
                    armor = dict(slot=slot, defense=defense, durability=dur, _source="assumed")
        if name in foods or name in ("apple", "bread", "beef", "cooked_beef", "porkchop",
                                     "cooked_porkchop", "mutton", "cooked_mutton", "chicken",
                                     "cooked_chicken", "rotten_flesh"):
            fkey = {"apple": "APPLE", "bread": "BREAD", "beef": "BEEF", "cooked_beef": "COOKED_BEEF",
                    "porkchop": "PORKCHOP", "cooked_porkchop": "COOKED_PORKCHOP",
                    "mutton": "MUTTON", "cooked_mutton": "COOKED_MUTTON", "chicken": "CHICKEN",
                    "cooked_chicken": "COOKED_CHICKEN", "rotten_flesh": "ROTTEN_FLESH"}[name]
            nut, mod = foods[fkey]
            food = dict(nutrition=nut, saturation=round(nut * mod * 2, 2))
            kind = "food"
        if places is not None and kind == "material":
            kind = "block"
        if name in ("stick", "coal", "raw_iron", "iron_ingot", "diamond", "raw_gold", "gold_ingot",
                    "quartz", "string", "bone", "arrow", "leather", "feather", "gunpowder", "flint",
                    "glowstone_dust", "wheat", "ender_pearl", "bucket", "flint_and_steel"):
            kind = "material" if name not in ("bucket", "flint_and_steel", "ender_pearl") else "misc"
            if name == "ender_pearl":
                stack = 16
        if name in ("water_bucket", "lava_bucket", "bucket"):
            kind = "misc"
            stack = 16 if name == "bucket" else 1
        if places is not None and name not in ("water_bucket", "lava_bucket"):
            kind = "block" if kind in ("material", "misc") else kind
        if name == "wheat":
            kind = "material"
        fuel = ASSUMED_FUEL.get(name, 0)
        items.append(dict(name=name, id=iid, stack=stack, kind=kind, places_block=places,
                          tool=tool, food=food, fuel_ticks=fuel,
                          value=2 ** depth_of.get(name, 0), armor=armor))

    # ---- drops
    item_names = {it["name"] for it in items}
    skipped_drops = []
    for b in blocks:
        if b["pseudo"] or b["name"] == "air":
            continue
        path = f"data/minecraft/loot_table/blocks/{VANILLA_NAME.get(b['name'], b['name'])}.json"
        raw = loot_drops(path)
        if raw is None:
            continue
        for item, lo, hi, ch in raw:
            if item not in item_names:
                skipped_drops.append((b["name"], item))
                continue
            req = b["min_tier"] if b["_req_tool"] else 0
            b["drops"].append(dict(item=item, count_min=lo, count_max=hi, chance=round(ch, 4),
                                   requires_tool_tier=req))
    # water/lava/bedrock: no item drop
    # ---- mobs
    mobs = []
    skipped_mob_drops = []
    for mid, (name, spec) in enumerate(MOBS.items()):
        attrs = dict(MOB_FALLBACK.get(name, {}))
        attrs.update(parse_attrs(spec["src"]))  # parsed values win
        drops = []
        for item, lo, hi, ch in loot_drops(f"data/minecraft/loot_table/entities/{name}.json") or []:
            if item not in item_names:
                skipped_mob_drops.append((name, item))
                continue
            drops.append(dict(item=item, count_min=lo, count_max=hi, chance=round(ch, 4)))
        dmg = attrs.get("ATTACK_DAMAGE", DEFAULT_ATTACK) if spec["hostile"] else 0.0  # passive: none
        if spec["attack"] == "explode":
            dmg = 0.0  # creeper: explosion, no melee damage attribute
        mobs.append(dict(
            name=name, pseudo_block=name,
            pseudo_block_id=next(b["id"] for b in blocks if b["name"] == name),
            hostile=spec["hostile"],
            health=attrs.get("MAX_HEALTH", DEFAULT_HEALTH),
            damage=dmg,
            speed=attrs.get("MOVEMENT_SPEED", DEFAULT_SPEED),
            spawn=dict(light_max=spec["light_max"], dimension=spec["dimension"], biomes=None,
                       day=spec["day"], group_min=spec["group"][0], group_max=spec["group"][1]),
            drops=drops,
            burns_in_sun=spec["burns"],
            attack=spec["attack"],
            size=spec["size"],
            _source=spec["src"],
        ))

    # ---- recipes
    recipes, rec_log = build_recipes(item_names)

    return blocks, items, recipes, mobs, dict(skipped_drops=skipped_drops,
                                              skipped_mob_drops=skipped_mob_drops,
                                              rec_log=rec_log, block_src=block_src)


# ---------------------------------------------------------------- recipes
SMELT_ALIAS = {"iron_ore": "raw_iron", "deepslate_iron_ore": "raw_iron", "gold_ore": "raw_gold",
               "deepslate_gold_ore": "raw_gold", "nether_gold_ore": "raw_gold"}


def _ing_resolve(ing, item_names, prefer=None):
    """Resolve one ingredient (string, {item}/{tag}, or list of alternatives) to one of our items."""
    if isinstance(ing, list):
        for alt in ing:
            r = _ing_resolve(alt, item_names)
            if r:
                return r
        return None
    if isinstance(ing, dict):
        if "item" in ing:
            ing = ing["item"]
        elif "tag" in ing:
            ing = "#" + strip_ns(ing["tag"])
    if not isinstance(ing, str):
        return None
    if ing.startswith("#"):
        for v in tag_values("item", strip_ns(ing[1:])):
            if v in item_names:
                return v
        return None
    n = strip_ns(ing)
    return n if n in item_names else None


def build_recipes(item_names):
    crafting, smelting = {}, {}
    log = []
    for path in sorted(n for n in _names if n.startswith("data/minecraft/recipe/") and n.endswith(".json")):
        fname = os.path.basename(path)[:-5]
        d = jar_json(path)
        typ = strip_ns(d.get("type", ""))
        if typ in ("crafting_shaped", "crafting_shapeless"):
            res = d.get("result", {})
            rid = strip_ns(res["id"] if isinstance(res, dict) else res)
            if rid not in item_names or not (fname == rid or fname.startswith(rid + "_from_")):
                continue  # canonical name, or a *_from_* variant (e.g. white_wool_from_string)
            if fname != rid and rid in crafting and crafting[rid]["_exact"]:
                continue
            cnt = res.get("count", 1) if isinstance(res, dict) else 1
            ings = {}
            ok = True
            if typ == "crafting_shaped":
                pattern = d["pattern"]
                keymap = d["key"]
                for row in pattern:
                    for ch in row:
                        if ch == " ":
                            continue
                        r = _ing_resolve(keymap[ch], item_names)
                        if r is None:
                            ok = False
                            break
                        ings[r] = ings.get(r, 0) + 1
                    if not ok:
                        break
                rows, cols = len(pattern), max(len(r) for r in pattern)
                needs = "inventory" if rows <= 2 and cols <= 2 else "crafting_table"
            else:
                flat = d["ingredients"]
                for ing in flat:
                    r = _ing_resolve(ing, item_names)
                    if r is None:
                        ok = False
                        break
                    ings[r] = ings.get(r, 0) + 1
                needs = "inventory" if len(flat) <= 4 else "crafting_table"
            if not ok or not ings:
                log.append(("craft-unresolved", fname))
                continue
            if rid in crafting and not crafting[rid]["_exact"] and fname == rid:
                pass  # exact name wins over a *_from_* variant (overwrite below)
            elif rid in crafting:
                continue
            crafting[rid] = dict(result=rid, count=cnt, ingredients=ings, needs=needs,
                                 cook_ticks=0, _exact=(fname == rid), _src=fname)
        elif typ in ("smelting", "campfire_cooking", "smoking"):
            res = d.get("result")
            rid = strip_ns(res["id"] if isinstance(res, dict) else res)
            if rid not in item_names:
                continue
            raw_ing = d.get("ingredient")
            raw_ing = strip_ns(raw_ing) if isinstance(raw_ing, str) else raw_ing
            if isinstance(raw_ing, str) and raw_ing in SMELT_ALIAS:
                ing = SMELT_ALIAS[raw_ing]
            else:
                ing = _ing_resolve(d.get("ingredient"), item_names)
            if ing is None:
                log.append(("smelt-unresolved", fname))
                continue
            key = (rid, ing)
            if key in smelting and typ != "smelting":
                continue
            smelting[key] = dict(name=fname, result=rid, count=1, ingredients={ing: 1},
                                 needs="furnace", cook_ticks=int(d.get("cookingtime", 200)))
    for r in crafting.values():
        r.pop("_exact", None)
        r["name"] = r.pop("_src")
    recs = sorted(crafting.values(), key=lambda r: (r["result"], sorted(r["ingredients"].items()))) \
        + sorted(smelting.values(), key=lambda r: (r["result"], sorted(r["ingredients"].items())))
    out = [dict(id=i, **r) for i, r in enumerate(recs)]
    return out, log


def main():
    blocks, items, recipes, mobs, meta = build()
    os.makedirs(OUT, exist_ok=True)
    for b in blocks:
        for k in [k for k in b if k.startswith("_")]:
            b.pop(k)
    for mob in mobs:
        mob.pop("_source", None)
    # SMELT_ALIAS note: smelting ore items map to raw items (ore block items are out of scope)
    for it in items:
        if it.get("armor") is not None:
            it["armor"].pop("_source", None)
        if it.get("armor") is None:
            it.pop("armor", None)
    write = lambda name, obj: open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8").write(
        json.dumps(obj, indent=1) + "\n")
    write("blocks", blocks)
    write("items", items)
    write("recipes", recipes)
    write("mobs", mobs)
    print("blocks", len(blocks), "items", len(items), "recipes", len(recipes),
          "smelting", sum(1 for r in recipes if r.get("cook_ticks")), "mobs", len(mobs))
    print("skipped drops (not in items):", meta["skipped_drops"])
    print("skipped mob drops:", meta["skipped_mob_drops"])
    print("recipe log:", meta["rec_log"])


if __name__ == "__main__":
    main()
