#!/usr/bin/env python3
"""Validate data/registry/{blocks,items,recipes,mobs}.json (PROTOCOL 12.2). Exit 1 on any error.

Run: .venv/bin/python tools/registry/validate.py
"""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REG = os.path.join(ROOT, "data", "registry")

V6_BLOCKS = {0: "air", 1: "grass", 2: "dirt", 3: "stone", 4: "cobblestone", 5: "oak_log",
             6: "oak_leaves", 7: "oak_planks", 8: "crafting_table", 9: "furnace", 10: "coal_ore",
             11: "iron_ore", 12: "diamond_ore", 13: "bedrock", 14: "sand", 15: "gravel",
             16: "water", 17: "deepslate", 18: "chest", 19: "agent"}
V6_ITEMS = {0: "oak_log", 1: "oak_planks", 2: "stick", 3: "crafting_table", 4: "wooden_pickaxe",
            5: "cobblestone", 6: "stone_pickaxe", 7: "furnace", 8: "coal", 9: "raw_iron",
            10: "iron_ingot", 11: "iron_pickaxe", 12: "diamond", 13: "dirt", 14: "chest"}
KINDS = {"block", "tool", "weapon", "armor", "food", "material", "misc"}
NEEDS = {"inventory", "crafting_table", "furnace"}
ITEM_STACKS = {1, 16, 64}

errors, warnings = [], []


def err(msg):
    errors.append(msg)


def load(name):
    with open(os.path.join(REG, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def check_dense(rows, label):
    ids = [r["id"] for r in rows]
    if sorted(ids) != list(range(len(rows))):
        err(f"{label}: ids not dense 0..{len(rows) - 1}")
    names = [r["name"] if "name" in r else r["result"] for r in rows]
    if len(set(names)) != len(names) and label != "recipes":
        err(f"{label}: duplicate names")


def main():
    blocks = load("blocks")
    items = load("items")
    recipes = load("recipes")
    mobs = load("mobs")

    # --- ids / uniqueness
    check_dense(blocks, "blocks")
    check_dense(items, "items")
    check_dense(recipes, "recipes")
    bnames = {b["name"]: b for b in blocks}
    inames = {i["name"] for i in items}

    for bid, name in V6_BLOCKS.items():
        if bnames.get(name, {}).get("id") != bid:
            err(f"v6 block id {bid} is not {name}")
    for iid, name in V6_ITEMS.items():
        if next((i["id"] for i in items if i["name"] == name), None) != iid:
            err(f"v6 item id {iid} is not {name}")

    # --- blocks
    if len(blocks) > 120:
        err(f"blocks: {len(blocks)} > 120")
    for b in blocks:
        for k in ("name", "id", "solid", "transparent", "liquid", "hardness", "best_tool",
                  "min_tier", "drops", "light_emission", "color", "dimension", "pseudo"):
            if k not in b:
                err(f"block {b.get('name')}: missing {k}")
        if b["best_tool"] not in (None, "pickaxe", "axe", "shovel", "hoe", "sword"):
            err(f"block {b['name']}: bad best_tool {b['best_tool']}")
        if b["min_tier"] not in range(5):
            err(f"block {b['name']}: min_tier out of range")
        if b["dimension"] not in ("overworld", "nether", "both"):
            err(f"block {b['name']}: bad dimension")
        if len(b["color"]) != 3:
            err(f"block {b['name']}: color must be [r,g,b]")
        if b["pseudo"] and b["drops"]:
            err(f"block {b['name']}: pseudo block with drops")
        for d in b["drops"]:
            if d["item"] not in inames:
                err(f"block {b['name']}: drop item {d['item']} not in items")
            if not (0 <= d["count_min"] <= d["count_max"]):
                err(f"block {b['name']}: bad count range for {d['item']}")
            if not (0.0 <= d["chance"] <= 1.0):
                err(f"block {b['name']}: chance out of range")
            if d["requires_tool_tier"] not in range(5):
                err(f"block {b['name']}: requires_tool_tier out of range")

    # --- items
    if not (80 <= len(items) <= 140):
        err(f"items: {len(items)} not in 80..140")
    for it in items:
        if it["kind"] not in KINDS:
            err(f"item {it['name']}: bad kind {it['kind']}")
        if it["stack"] not in ITEM_STACKS:
            err(f"item {it['name']}: bad stack {it['stack']}")
        if it["places_block"] is not None and it["places_block"] not in bnames:
            err(f"item {it['name']}: places_block {it['places_block']} not in blocks")
        if it["tool"] is not None:
            t = it["tool"]
            if t["type"] not in ("pickaxe", "axe", "shovel", "hoe", "sword"):
                err(f"item {it['name']}: bad tool type")
            if not (1 <= t["tier"] <= 4):
                err(f"item {it['name']}: bad tool tier")
        if it["food"] is not None and it["food"]["nutrition"] <= 0:
            err(f"item {it['name']}: food nutrition must be > 0")
        if it["fuel_ticks"] < 0 or it["value"] < 1:
            err(f"item {it['name']}: bad fuel/value")

    # --- recipes
    seen = set()
    for r in recipes:
        if r["result"] not in inames:
            err(f"recipe {r['id']}: result {r['result']} not in items")
        for ing, n in r["ingredients"].items():
            if ing not in inames:
                err(f"recipe {r['id']} ({r['result']}): ingredient {ing} not in items")
            if n <= 0:
                err(f"recipe {r['id']}: non-positive count for {ing}")
        if r["needs"] not in NEEDS:
            err(f"recipe {r['id']}: bad needs {r['needs']}")
        if r["needs"] == "furnace" and r.get("cook_ticks", 0) <= 0:
            err(f"recipe {r['id']}: smelting recipe without cook_ticks")
        if r["needs"] != "furnace" and "cook_ticks" not in r:
            err(f"recipe {r['id']}: missing cook_ticks key")
        key = (r["result"], tuple(sorted(r["ingredients"].items())))
        if key in seen:
            err(f"recipe {r['id']}: duplicate of an earlier recipe ({r['result']})")
        seen.add(key)
    if not (40 <= len(recipes)):
        warnings.append(f"recipes: only {len(recipes)}")

    # --- mobs
    for m in mobs:
        pb = bnames.get(m["pseudo_block"])
        if pb is None or not pb["pseudo"] or pb["id"] != m["pseudo_block_id"]:
            err(f"mob {m['name']}: pseudo_block not a pseudo block in blocks.json")
        for d in m["drops"]:
            if d["item"] not in inames:
                err(f"mob {m['name']}: drop item {d['item']} not in items")
        if m["attack"] not in ("melee", "ranged", "explode", "none"):
            err(f"mob {m['name']}: bad attack")
    for need in ("zombie", "skeleton", "spider", "creeper", "cow", "pig", "sheep", "chicken"):
        if need not in {m["name"] for m in mobs}:
            err(f"mob {need} missing")

    # --- optional: compare against a CORE seed file if present (v6 ids must match)
    for seed in glob.glob(os.path.join(REG, "*.seed.json")):
        warnings.append(f"seed file present: {os.path.basename(seed)} (not checked by DATA-1)")

    print(f"blocks={len(blocks)} items={len(items)} recipes={len(recipes)} "
          f"smelting={sum(1 for r in recipes if r['needs'] == 'furnace')} mobs={len(mobs)}")
    for w in warnings:
        print("WARN", w)
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"FAILED ({len(errors)} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
