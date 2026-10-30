"""Validates data/builds/blueprints.json against protocol 12.6 and the item registry.

Checks: list of 30..60 entries, unique names/ids, required names present, schema keys, layer/size shape,
palette coverage, required_items and block_count recount, tier ranges (1: 5-30, 2: 30-120, 3: 120-400),
family set, item names in data/registry/items.json (or the fallback list when that file is absent),
nether_portal geometry (4x5 obsidian frame with 2x3 air inside).

Exit code 0 = pass, 1 = fail.   Usage: .venv/bin/python tools/builds/validate.py [--registry PATH]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUILDS = os.path.join(ROOT, "data", "builds", "blueprints.json")
REGISTRY = os.path.join(ROOT, "data", "registry", "items.json")

REQUIRED_NAMES = ["nether_portal", "dirt_hut", "cobble_house", "wooden_house", "watchtower", "bridge",
                  "wall_segment", "farm_plot"]
FAMILIES = {"shelter", "house", "tower", "bridge", "wall", "portal", "farm", "decor"}
TIERS = {1: (5, 30), 2: (30, 120), 3: (120, 400)}
KEYS = ["id", "name", "family", "tier", "size", "palette", "layers", "anchor", "required_items", "block_count",
        "source", "description"]
# Items the current builds use; used only when data/registry/items.json is absent.
FALLBACK_ITEMS = {"oak_log", "oak_planks", "cobblestone", "dirt", "sand", "gravel", "glass", "obsidian",
                  "furnace", "crafting_table", "chest"}


def load_registry(path):
    if not os.path.exists(path):
        return None, "fallback (items.json absent)"
    with open(path) as f:
        data = json.load(f)
    items = data if isinstance(data, list) else data.get("items", data)
    names = set()
    for it in items:
        names.add(it["name"] if isinstance(it, dict) else it)
    return names, "data/registry/items.json (%d items)" % len(names)


def check(bps, registry, errors):
    if not (30 <= len(bps) <= 60):
        errors.append("blueprint count %d not in 30..60" % len(bps))
    names = [b.get("name") for b in bps]
    if len(set(names)) != len(names):
        errors.append("duplicate names")
    for r in REQUIRED_NAMES:
        if r not in names:
            errors.append("missing required blueprint %r" % r)
    ids = [b.get("id") for b in bps]
    if ids != list(range(len(bps))):
        errors.append("ids are not 0..N-1 in order")
    allowed_items = registry if registry is not None else FALLBACK_ITEMS
    unknown_items = set()
    for b in bps:
        tag = b.get("name", "?")
        for k in KEYS:
            if k not in b:
                errors.append("%s: missing key %s" % (tag, k))
        if any(k not in b for k in KEYS):
            continue
        if b["family"] not in FAMILIES:
            errors.append("%s: family %r not allowed" % (tag, b["family"]))
        sx, sy, sz = b["size"]
        layers = b["layers"]
        if len(layers) != sy:
            errors.append("%s: %d layers but size y=%d" % (tag, len(layers), sy))
        for y, layer in enumerate(layers):
            if len(layer) != sz:
                errors.append("%s: layer %d has %d rows, size z=%d" % (tag, y, len(layer), sz))
            for r in layer:
                if len(r) != sx:
                    errors.append("%s: row of length %d, size x=%d" % (tag, len(r), sx))
                    break
        pal = b["palette"]
        if "." in pal:
            errors.append("%s: '.' is reserved for air but is in palette" % tag)
        for c, it in pal.items():
            if len(c) != 1:
                errors.append("%s: palette key %r not one char" % (tag, c))
            if it not in allowed_items:
                unknown_items.add(it)
        counts = {}
        for layer in layers:
            for r in layer:
                for c in r:
                    if c == ".":
                        continue
                    if c not in pal:
                        errors.append("%s: char %r not in palette" % (tag, c))
                        continue
                    counts[pal[c]] = counts.get(pal[c], 0) + 1
        if sum(counts.values()) != b["block_count"]:
            errors.append("%s: block_count %d != recount %d" % (tag, b["block_count"], sum(counts.values())))
        if counts != b["required_items"]:
            errors.append("%s: required_items mismatch" % tag)
        lo, hi = TIERS.get(b["tier"], (None, None))
        if lo is None or not (lo <= b["block_count"] <= hi):
            errors.append("%s: tier %s incompatible with block_count %d" % (tag, b["tier"], b["block_count"]))
        if b["anchor"] != "bottom_center":
            errors.append("%s: unexpected anchor %r" % (tag, b["anchor"]))
        if b["block_count"] == 0:
            errors.append("%s: empty blueprint" % tag)
    tiers = {t: sum(1 for b in bps if b.get("tier") == t) for t in TIERS}
    for t, n in tiers.items():
        if n == 0:
            errors.append("tier %d has no blueprints" % t)
    if unknown_items:
        errors.append("items not in registry: %s" % ", ".join(sorted(unknown_items)))

    by = {b["name"]: b for b in bps if "name" in b}
    if "nether_portal" in by:
        p = by["nether_portal"]
        if p["size"] != [4, 5, 1]:
            errors.append("nether_portal size %r != [4,5,1]" % (p["size"],))
        else:
            L = p["layers"]
            ob = p["palette"].get
            frame_ok = True
            for y in range(5):
                for x in range(4):
                    edge = y in (0, 4) or x in (0, 3)
                    c = L[y][0][x]
                    if edge and ob(c) != "obsidian":
                        frame_ok = False
                    if not edge and c != ".":
                        frame_ok = False
            if not frame_ok:
                errors.append("nether_portal: frame not 4x5 obsidian with air inside")
    return tiers


def main(argv):
    reg_path = REGISTRY
    if "--registry" in argv:
        reg_path = argv[argv.index("--registry") + 1]
    registry, src = load_registry(reg_path)
    with open(BUILDS) as f:
        bps = json.load(f)
    errors = []
    tiers = check(bps, registry, errors)
    print("validated %d blueprints against %s" % (len(bps), src))
    print("tiers:", tiers)
    if errors:
        print("FAIL (%d)" % len(errors))
        for e in errors:
            print(" -", e)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
