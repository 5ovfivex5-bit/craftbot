#!/usr/bin/env python3
"""Generate src/gen/registry.h (constexpr tables) from the data files (PROTOCOL 12.2, 12.5, 12.8).

Inputs, each NAME.json if present else NAME.seed.json:
  data/registry/blocks.json, items.json, recipes.json, mobs.json
  data/tasks/tasks.json (else tasks.seed.json)
  data/builds/blueprints.json (optional; blueprint names -> index for build_complete)

Output: src/gen/registry.h, rewritten only when the text changes. build.sh runs this before compiling.
Ids are the file order (dense). Blocks 0..19 and items 0..14 are the v6 ids and are checked.
Run: python3 -I tools/gen_registry.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "src", "gen", "registry.h")

# v6 block ids 0..19 (accepted aliases per id: DATA-1 names grass as grass_block)
V6_BLOCKS = [("air",), ("grass", "grass_block"), ("dirt",), ("stone",), ("cobblestone",), ("oak_log",),
             ("oak_leaves",), ("oak_planks",), ("crafting_table",), ("furnace",), ("coal_ore",),
             ("iron_ore",), ("diamond_ore",), ("bedrock",), ("sand",), ("gravel",), ("water",),
             ("deepslate",), ("chest",), ("agent",)]
V6_ITEMS = ["oak_log", "oak_planks", "stick", "crafting_table", "wooden_pickaxe", "cobblestone",
            "stone_pickaxe", "furnace", "coal", "raw_iron", "iron_ingot", "iron_pickaxe", "diamond", "dirt",
            "chest"]
# v6 recipes by (result, ingredients, needs): named constants REC_V6_* in the header.
V6_RECIPES = {
    "CRAFT_PLANKS": ("oak_planks", {"oak_log": 1}, "inventory"),
    "CRAFT_STICKS": ("stick", {"oak_planks": 2}, "inventory"),
    "CRAFT_TABLE": ("crafting_table", {"oak_planks": 4}, "inventory"),
    "CRAFT_WOOD_PICK": ("wooden_pickaxe", {"oak_planks": 3, "stick": 2}, "crafting_table"),
    "CRAFT_STONE_PICK": ("stone_pickaxe", {"cobblestone": 3, "stick": 2}, "crafting_table"),
    "CRAFT_FURNACE": ("furnace", {"cobblestone": 8}, "crafting_table"),
    "CRAFT_IRON_PICK": ("iron_pickaxe", {"iron_ingot": 3, "stick": 2}, "crafting_table"),
    "CRAFT_CHEST": ("chest", {"oak_planks": 8}, "crafting_table"),
    "SMELT_IRON": ("iron_ingot", {"raw_iron": 1}, "furnace"),
}
KINDS = {"block": 0, "tool": 1, "weapon": 2, "armor": 3, "food": 4, "material": 5, "misc": 6}
TOOL_TYPES = {None: 0, "pickaxe": 1, "axe": 2, "shovel": 3, "sword": 4, "hoe": 5}
NEEDS = {"inventory": 0, "crafting_table": 1, "furnace": 2, "smoker": 2, "campfire": 2}
ATTACKS = {"none": 0, "melee": 1, "ranged": 2, "explode": 3}
DAYS = {"any": 0, "day": 1, "night": 2}
DIMS = {"overworld": 0, "nether": 1, "both": 0}
FAMILIES = ["gather", "food", "tools", "build", "combat", "survival", "nether", "explore", "social"]
PRED_ENUM = ["has_item", "crafted", "mined", "killed", "ate", "slept", "survive_night", "placed",
             "build_complete", "portal_lit", "in_dimension", "reached_y", "distance_travelled",
             "health_above", "near_teammate", "look_at_teammate", "crouch_greet", "follow_teammate",
             "gave_item", "shared_chest", "msg_near_teammate"]
MAX_MS = 8


def fail(msg):
    sys.stderr.write(f"gen_registry: {msg}\n")
    sys.exit(1)


def load(name):
    for fn in (f"data/registry/{name}.json", f"data/registry/{name}.seed.json"):
        p = os.path.join(ROOT, fn)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), fn
    fail(f"no data/registry/{name}.json or {name}.seed.json")


def load_tasks():
    for fn in ("data/tasks/tasks.json", "data/tasks/tasks.seed.json"):
        p = os.path.join(ROOT, fn)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), fn
    fail("no data/tasks/tasks.json or tasks.seed.json")


def cid(name):
    out = []
    for ch in name:
        out.append(ch.upper() if ch.isalnum() else "_")
    return "".join(out)


def cfloat(v):
    s = repr(float(v))
    if "." not in s and "e" not in s and "n" not in s:
        s += ".0"
    return s


def main():
    blocks, bsrc = load("blocks")
    items, isrc = load("items")
    recipes, rsrc = load("recipes")
    mobs, msrc = load("mobs")
    tasks, tsrc = load_tasks()
    bp_path = os.path.join(ROOT, "data/builds/blueprints.json")
    bp_names = []
    if os.path.exists(bp_path):
        with open(bp_path) as f:
            bp_names = [b["name"] for b in json.load(f)]

    # ---- blocks
    for i, b in enumerate(blocks):
        if b["id"] != i:
            fail(f"block ids must be dense: {b['name']} has id {b['id']} at index {i}")
    for i, names in enumerate(V6_BLOCKS):
        if blocks[i]["name"] not in names:
            fail(f"block {i} must be {names[0]} (v6 id), got {blocks[i]['name']}")
    bid = {b["name"]: b["id"] for b in blocks}
    if len(bid) != len(blocks):
        fail("duplicate block names")
    for b in blocks:
        if b["best_tool"] not in TOOL_TYPES:
            fail(f"block {b['name']} best_tool {b['best_tool']} unknown")
        if not 0 <= b["min_tier"] <= 4:
            fail(f"block {b['name']} min_tier out of range")

    # ---- items
    for i, it in enumerate(items):
        if it["id"] != i:
            fail(f"item ids must be dense: {it['name']}")
    if [it["name"] for it in items[:15]] != V6_ITEMS:
        fail("items 0..14 must be the v6 items in v6 order")
    iid = {it["name"]: it["id"] for it in items}
    for it in items:
        if it.get("places_block") and it["places_block"] not in bid:
            fail(f"item {it['name']} places unknown block {it['places_block']}")
        if it["kind"] not in KINDS:
            fail(f"item {it['name']} has unknown kind {it['kind']}")
        if it["kind"] == "tool" and not it.get("tool"):
            fail(f"tool item {it['name']} has no tool block")
    if len(items) > 128:
        fail("at most 128 items (agent shaping bitmask)")

    warnings = []

    def item_id(n, soft=False):
        if n not in iid:
            if not soft:
                fail(f"unknown item '{n}'")
            warnings.append(f"item '{n}'")
            return -1
        return iid[n]

    def block_id(n, soft=False):
        if n not in bid:
            if not soft:
                fail(f"unknown block '{n}'")
            warnings.append(f"block '{n}'")
            return -1
        return bid[n]

    # ---- mobs
    mid = {m["name"]: i for i, m in enumerate(mobs)}
    for m in mobs:
        pb = m["pseudo_block"]
        if pb not in bid or not blocks[bid[pb]].get("pseudo"):
            fail(f"mob {m['name']}: pseudo_block {pb} must be a pseudo block")
        if m["spawn"]["day"] not in DAYS:
            fail(f"mob {m['name']}: spawn.day unknown")

    # ---- recipes: dense ids, names, needs
    names_seen = {}
    rec_names = []
    for i, r in enumerate(recipes):
        if r["id"] != i:
            fail(f"recipe ids must be dense: {r}")
        if r["needs"] not in NEEDS:
            fail(f"recipe {r['result']} needs {r['needs']} unknown")
        item_id(r["result"])
        for n in r["ingredients"]:
            item_id(n)
        if len(r["ingredients"]) > 4:
            fail(f"recipe {i} has more than 4 ingredient types")
        base = cid(r["result"]).lower()
        names_seen[base] = names_seen.get(base, 0) + 1
        rec_names.append(base if names_seen[base] == 1 else f"{base}_{i}")
    rec_consts = {}
    for c, (res, ing, needs) in V6_RECIPES.items():
        hits = [i for i, r in enumerate(recipes)
                if r["result"] == res and r["ingredients"] == ing and NEEDS[r["needs"]] == NEEDS[needs]]
        if len(hits) != 1:
            fail(f"v6 recipe {c} ({res} from {ing}) found {len(hits)} times")
        rec_consts[c] = hits[0]
    max_ing = max([len(r["ingredients"]) for r in recipes] + [1])

    # ---- tasks
    fam = {f: i for i, f in enumerate(FAMILIES)}
    seen_ids = set()
    for t in tasks:
        if t["id"] in seen_ids:
            fail(f"duplicate task id {t['id']}")
        seen_ids.add(t["id"])
        if t["family"] not in fam:
            fail(f"task {t['id']} family {t['family']} unknown")
        if len(t["milestones"]) > MAX_MS:
            fail(f"task {t['id']} has more than {MAX_MS} milestones")
    if len(tasks) > 64:
        fail("at most 64 tasks (task one-hot slots)")

    def pred(p, tid):
        ty = p["type"]
        if ty not in PRED_ENUM:
            fail(f"task {tid}: unknown predicate {ty}")
        if ty == "has_item":
            a, b = item_id(p["item"], soft=True), int(p["count"])
        elif ty == "crafted":
            a, b = item_id(p["item"], soft=True), 1
        elif ty in ("mined", "placed"):
            a, b = block_id(p["block"], soft=True), int(p["count"])
        elif ty == "killed":
            if p["mob"] not in mid:
                warnings.append(f"mob '{p['mob']}' (task {tid})")
            a, b = mid.get(p["mob"], -1), int(p["count"])
        elif ty == "ate":
            food = p.get("food", "any")
            a, b = (-1 if food == "any" else item_id(food, soft=True)), 0
        elif ty in ("slept", "survive_night", "portal_lit"):
            a, b = 0, 0
        elif ty == "build_complete":
            if p["blueprint"] not in bp_names:
                fail(f"task {tid}: blueprint {p['blueprint']} not in data/builds/blueprints.json")
            a, b = bp_names.index(p["blueprint"]), int(round(float(p["min_progress"]) * 1000))
        elif ty == "in_dimension":
            if p["dim"] not in ("overworld", "nether"):
                fail(f"task {tid}: dim {p['dim']}")
            a, b = (1 if p["dim"] == "nether" else 0), 0
        elif ty == "reached_y":
            a, b = int(p["max_y"]), 0
        elif ty == "distance_travelled":
            a, b = int(p["n"]), 0
        elif ty == "health_above":
            a, b = int(p["n"]), 0
        elif ty == "near_teammate":
            a, b = int(p["dist"]), int(p["steps"])
        elif ty == "follow_teammate":
            a, b = 0, int(p["steps"])
        else:  # count-only predicates (look_at_teammate, crouch_greet, gave_item, shared_chest, msg_near_teammate)
            a, b = 0, int(p["count"])
        return PRED_ENUM.index(ty), a, b

    for wmsg in warnings:
        sys.stderr.write(f"gen_registry: warning: unknown {wmsg}; predicate is inert\n")

    # ---- emit
    L = []
    w = L.append
    w("// GENERATED by tools/gen_registry.py. Do not edit by hand.")
    w(f"// sources: {bsrc}, {isrc}, {rsrc}, {msrc}, {tsrc}")
    w("#pragma once")
    w("#include <cstdint>")
    w("")
    w("namespace craftsim::reg {")
    w("")
    w(f"constexpr int NUM_BLOCKS = {len(blocks)};")
    w(f"constexpr int NUM_ITEMS = {len(items)};")
    w(f"constexpr int NUM_RECIPES = {len(recipes)};")
    w(f"constexpr int NUM_MOBS = {len(mobs)};")
    w(f"constexpr int NUM_TASKS = {len(tasks)};")
    w(f"constexpr int NUM_FAMILIES = {len(FAMILIES)};")
    w(f"constexpr int MAX_MILESTONES = {MAX_MS};")
    w(f"constexpr int MAX_RECIPE_INGREDIENTS = {max_ing};")
    w("")
    w("enum PredType : uint8_t {")
    for n in PRED_ENUM:
        w(f"    P_{n.upper()},")
    w("    P_NUM_TYPES")
    w("};")
    w("enum Needs : uint8_t { N_INVENTORY, N_TABLE, N_FURNACE };")
    w("enum Kind : uint8_t { K_BLOCK, K_TOOL, K_WEAPON, K_ARMOR, K_FOOD, K_MATERIAL, K_MISC };")
    w("enum ToolType : uint8_t { T_NONE, T_PICKAXE, T_AXE, T_SHOVEL, T_SWORD, T_HOE };")
    w("enum Attack : uint8_t { A_NONE, A_MELEE, A_RANGED, A_EXPLODE };")
    w("enum Family : uint8_t { " + ", ".join(f"F_{f.upper()}" for f in FAMILIES) + " };")
    w("")
    w("// a = primary parameter (item, block, mob, blueprint index, dim 0/1, max_y, dist, n, food or -1 any),")
    w("// b = count / steps / min_progress*1000 (secondary parameter).")
    w("struct Pred { uint8_t type; int32_t a; int32_t b; };")
    w("struct BlockDef { const char* name; float hardness; uint8_t bestTool; uint8_t minTier; int16_t drop;")
    w("                  uint8_t dropCount; uint8_t solid; uint8_t transparent; uint8_t liquid; uint8_t pseudo;")
    w("                  uint8_t light; uint8_t color[3]; uint8_t dimension; };")
    w("struct ItemDef { const char* name; uint8_t stack; uint8_t kind; int8_t places; uint8_t toolType;")
    w("                uint8_t toolTier; float toolSpeed; int16_t durability; int16_t fuelTicks; int32_t value; };")
    w("struct RecipeDef { const char* name; uint8_t result; uint8_t count; uint8_t needs; uint16_t cookTicks;")
    w("                   uint8_t nIng; uint8_t ingItem[MAX_RECIPE_INGREDIENTS]; uint8_t ingCount[MAX_RECIPE_INGREDIENTS]; };")
    w("struct MobDef { const char* name; uint8_t block; uint8_t hostile; uint8_t attack; uint8_t burnsInSun;")
    w("                uint8_t dimension; uint8_t day; uint8_t lightMax; uint8_t groupMin; uint8_t groupMax;")
    w("                float health; float damage; float speed; float width; float height; };")
    w("struct Milestone { Pred pred; float reward; };")
    w("struct TaskDef { const char* name; uint8_t family; Pred success; uint8_t nMilestones;")
    w("                 Milestone ms[MAX_MILESTONES]; int32_t maxSteps; };")
    w("")
    w("inline constexpr const char* FAMILY_NAMES[NUM_FAMILIES] = {" + ", ".join(f'"{f}"' for f in FAMILIES) + "};")
    w("")
    w("// Blocks (id = index). Pseudo blocks (mobs) are never stored in a world.")
    w("inline constexpr BlockDef BLOCKS[NUM_BLOCKS] = {")
    for b in blocks:
        dr = b.get("drops") or []
        drop, dcount = -1, 0
        if dr:
            drop = item_id(dr[0]["item"])
            dcount = int(dr[0]["count_min"])
        col = b.get("color", [128, 128, 128])
        w(f'    {{"{b["name"]}", {cfloat(b["hardness"])}f, {TOOL_TYPES[b["best_tool"]]}, {int(b["min_tier"])}, '
          f'{drop}, {dcount}, {int(bool(b["solid"]))}, {int(bool(b["transparent"]))}, '
          f'{int(bool(b["liquid"]))}, {int(bool(b.get("pseudo")))}, {int(b.get("light_emission", 0))}, '
          f'{{{col[0]}, {col[1]}, {col[2]}}}, {DIMS.get(b.get("dimension", "both"), 0)}}},')
    w("};")
    w("")
    w("// Items (id = index).")
    w("inline constexpr ItemDef ITEMS[NUM_ITEMS] = {")
    for it in items:
        t = it.get("tool") or {}
        places = block_id(it["places_block"]) if it.get("places_block") else -1
        w(f'    {{"{it["name"]}", {int(it["stack"])}, {KINDS[it["kind"]]}, {places}, '
          f'{TOOL_TYPES[t.get("type")]}, {int(t.get("tier", 0))}, {cfloat(float(t.get("speed", 0)))}f, '
          f'{int(t.get("durability", 0))}, {int(it.get("fuel_ticks", 0))}, {int(it.get("value", 0))}}},')
    w("};")
    w("")
    w("// Recipes (id = index). Action column value for recipe r is r + 1 (0 = none).")
    w("inline constexpr RecipeDef RECIPES[NUM_RECIPES] = {")
    for r, nm in zip(recipes, rec_names):
        ings = list(r["ingredients"].items())
        ii = [item_id(n) for n, _ in ings] + [0] * (max_ing - len(ings))
        cc = [int(c) for _, c in ings] + [0] * (max_ing - len(ings))
        w(f'    {{"{nm}", {item_id(r["result"])}, {int(r["count"])}, {NEEDS[r["needs"]]}, '
          f'{int(r.get("cook_ticks", 0))}, {len(ings)}, {{{", ".join(map(str, ii))}}}, '
          f'{{{", ".join(map(str, cc))}}}}},')
    w("};")
    w("")
    w("// Mobs (id = index). `block` is the pseudo block id seen in observations.")
    w("inline constexpr MobDef MOBS[NUM_MOBS] = {")
    for m in mobs:
        sp = m["spawn"]
        w(f'    {{"{m["name"]}", {bid[m["pseudo_block"]]}, {int(bool(m["hostile"]))}, {ATTACKS[m["attack"]]}, '
          f'{int(bool(m["burns_in_sun"]))}, {DIMS.get(sp["dimension"], 0)}, {DAYS[sp["day"]]}, '
          f'{int(sp["light_max"])}, {int(sp["group_min"])}, {int(sp["group_max"])}, '
          f'{cfloat(m["health"])}f, {cfloat(m["damage"])}f, {cfloat(m["speed"])}f, '
          f'{cfloat(m["size"][0])}f, {cfloat(m["size"][1])}f}},')
    w("};")
    w("")
    w("// Tasks (id = index). Predicates use the Pred encoding above.")
    w("inline constexpr TaskDef TASKS[NUM_TASKS] = {")
    for t in tasks:
        ty, a, b = pred(t["success"], t["id"])
        ms = []
        for m in t["milestones"]:
            mt, ma, mb = pred(m["predicate"], t["id"])
            ms.append(f"{{{{{mt}, {ma}, {mb}}}, {cfloat(m['reward'])}f}}")
        ms += ["{{0, 0, 0}, 0.0f}"] * (MAX_MS - len(ms))
        w(f'    {{"{t["id"]}", F_{t["family"].upper()}, {{{ty}, {a}, {b}}}, {len(t["milestones"])}, '
          f'{{{", ".join(ms)}}}, {int(t["max_steps"])}}},')
    w("};")
    w("")
    w("// Named ids.")
    for i, b in enumerate(blocks):
        w(f"constexpr int BLK_{cid(b['name']).upper()} = {i};")
    for i, it in enumerate(items):
        w(f"constexpr int ITM_{cid(it['name']).upper()} = {i};")
    for i, nm in enumerate(rec_names):
        w(f"constexpr int REC_{cid(nm).upper()} = {i};")
    for c, i in rec_consts.items():
        w(f"constexpr int REC_V6_{c} = {i};")
    for i, m in enumerate(mobs):
        w(f"constexpr int MOB_{cid(m['name']).upper()} = {i};")
    for i, t in enumerate(tasks):
        w(f"constexpr int TSK_{cid(t['id']).upper()} = {i};")
    w("")
    # v7 crosshair slots: the 9 most valuable items (ties: lower id first)
    xh = sorted(range(len(items)), key=lambda i: (-items[i].get("value", 0), i))[:9]
    w("// Crosshair chest slots in v7 obs (9 most valuable items).")
    w("inline constexpr uint8_t XHAIR_ITEMS[9] = {" + ", ".join(str(i) for i in xh) + "};")
    w("")
    w("}  // namespace craftsim::reg")
    text = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    old = open(OUT).read() if os.path.exists(OUT) else None
    if old != text:
        with open(OUT, "w") as f:
            f.write(text)
    print(f"gen_registry: blocks={len(blocks)} items={len(items)} recipes={len(recipes)} mobs={len(mobs)} "
          f"tasks={len(tasks)} -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
