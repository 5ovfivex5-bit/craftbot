#!/usr/bin/env python3
"""Build the item acquisition graph and task shaping tables (PROTOCOL 12.7).

Inputs:  data/registry/{items,blocks,recipes,mobs}.json   (DATA-1; loot.json optional)
         data/tasks/tasks.json                            (DATA-2)
         data/builds/blueprints.json                      (DATA-3, optional)
Outputs: data/knowledge/graph.json    {items: {item: {via, from, tool_tier, tool_type, recipe_id,
                                       depth, prereqs, requirements, alternatives, ...}},
                                       unobtainable: [...], counts: {...}}
         data/knowledge/shaping.json  {tasks: {task_id: {target_items, items, values}}}

Source kinds: mine (block drop), craft (2x2 or crafting_table), smelt (furnace-like),
kill (mob drop), loot (optional loot table). Each source has requirement *groups*: a group is a
set of alternatives of which one must be held (a tool of a given type/tier, or a station). Depth
is the fixed point depth = min over sources of 1 + max over groups of min depth(alternative), so
the chosen source graph is a DAG and the closure is well-founded.

Usage: .venv/bin/python tools/knowledge/build_graph.py [--wait SECONDS]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tasklib.loaders import load_blueprints, load_tasks  # noqa: E402
from tasklib.schema import normalize_predicates, task_predicates  # noqa: E402

TIER_NAMES = {"hand": 0, "none": 0, "wood": 1, "gold": 1, "stone": 2, "iron": 3,
              "diamond": 4, "netherite": 5}
FURNACE_LIKE = {"furnace", "smoker", "campfire", "soul_campfire", "blast_furnace"}
VIA_PRIORITY = {"mine": 0, "craft": 1, "smelt": 2, "kill": 3, "loot": 4}
REGISTRY_FILES = {"items": "items.json", "blocks": "blocks.json",
                  "recipes": "recipes.json", "mobs": "mobs.json"}


def tier_of(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    return TIER_NAMES.get(str(v).lower())


def _as_list(data, key_name="name"):
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            out = []
            for k, v in data.items():
                e = dict(v)
                e.setdefault(key_name, k)
                out.append(e)
            return out
        for v in data.values():
            if isinstance(v, list):
                return v
    raise ValueError("unrecognised registry layout")


def load_registry(reg_dir):
    reg_dir = Path(reg_dir)
    reg = {}
    for key, fname in REGISTRY_FILES.items():
        p = reg_dir / fname
        reg[key] = _as_list(json.loads(p.read_text())) if p.exists() else []
    lp = reg_dir / "loot.json"
    reg["loot"] = _as_list(json.loads(lp.read_text()), key_name="table") if lp.exists() else []
    return reg


def _src(via, frm, groups, **kw):
    s = {"via": via, "from": frm, "groups": groups, "tool_tier": None, "tool_type": None,
         "recipe_id": None, "recipe": None, "dimension": "overworld", "count": None,
         "chance": 1.0}
    s.update(kw)
    return s


def build_graph(reg):
    """Return (graph dict, warnings list)."""
    warnings = []
    items = {}
    for e in reg["items"]:
        n = e.get("name")
        if not n:
            warnings.append("item without name skipped")
        elif n in items:
            warnings.append(f"duplicate item {n!r}")
        else:
            items[n] = e

    def known(name, ctx):
        if name in items:
            return True
        warnings.append(f"{ctx}: unknown item {name!r}")
        return False

    tools = []
    for n, e in items.items():
        t = e.get("tool")
        if t:
            tools.append((n, t.get("type"), tier_of(t.get("tier"))))

    def tool_alts(tool_type, min_tier):
        return sorted(n for n, ty, tr in tools
                      if (tool_type is None or ty == tool_type) and tr is not None and tr >= min_tier)

    sources = {n: [] for n in items}

    for b in reg["blocks"]:
        bname = b.get("name")
        best = b.get("best_tool")
        bmin = tier_of(b.get("min_tier")) or 0
        for d in b.get("drops") or []:
            item = d.get("item")
            if not known(item, f"block {bname} drop"):
                continue
            req = tier_of(d.get("requires_tool_tier"))
            if req is None:
                req = bmin
            groups = []
            if req > 0:
                alts = tool_alts(best, req)
                if not alts:
                    warnings.append(f"block {bname}: no tool of type {best!r} tier>={req}")
                    continue
                groups = [alts]
            sources[item].append(_src(
                "mine", bname, groups, tool_tier=req, tool_type=best if req > 0 else None,
                dimension=b.get("dimension", "overworld"),
                count=[d.get("count_min", 1), d.get("count_max", 1)],
                chance=d.get("chance", 1.0)))

    for r in reg["recipes"]:
        rid = r.get("id", None)
        result = r.get("result")
        if not known(result, f"recipe {rid}"):
            continue
        ings = r.get("ingredients") or {}
        if not ings:
            warnings.append(f"recipe {rid}: no ingredients")
            continue
        bad = [k for k in ings if k not in items]
        if bad:
            warnings.append(f"recipe {rid}: unknown ingredients {bad}")
            continue
        needs = r.get("needs", "inventory")
        if needs in FURNACE_LIKE or r.get("type") == "smelting":
            station, via = "furnace", "smelt"
        elif needs == "inventory":
            station, via = None, "craft"
        elif needs == "crafting_table":
            station, via = "crafting_table", "craft"
        else:
            warnings.append(f"recipe {rid}: unknown needs {needs!r}")
            continue
        groups = [[k] for k in sorted(ings)]
        if station:
            if station not in items:
                warnings.append(f"recipe {rid}: station {station!r} missing")
                continue
            groups.append([station])
        rec = {"id": rid, "result": result, "count": r.get("count", 1),
               "ingredients": dict(ings), "needs": needs, "cook_ticks": r.get("cook_ticks")}
        frm = (sorted(ings)[0] if via == "smelt" else (station or "inventory"))
        sources[result].append(_src(via, frm, groups, recipe_id=rid, recipe=rec,
                                   count=r.get("count", 1)))

    for m in reg["mobs"]:
        mname = m.get("name")
        spawn = m.get("spawn") or {}
        dim = spawn.get("dimension", "overworld")
        if isinstance(dim, list):
            dim = "/".join(map(str, dim))
        for d in m.get("drops") or []:
            item = d.get("item")
            if known(item, f"mob {mname} drop"):
                sources[item].append(_src(
                    "kill", mname, [], dimension=dim,
                    count=[d.get("count_min", 1), d.get("count_max", 1)],
                    chance=d.get("chance", 1.0)))

    for t in reg["loot"]:
        tname = t.get("table")
        container = t.get("container")
        groups = []
        if container:
            if known(container, f"loot {tname} container"):
                groups = [[container]]
        for d in t.get("drops") or []:
            item = d.get("item")
            if known(item, f"loot {tname}"):
                sources[item].append(_src(
                    "loot", tname, groups, dimension=t.get("dimension", "overworld"),
                    count=[d.get("count_min", 1), d.get("count_max", 1)],
                    chance=d.get("chance", 1.0)))

    def group_best(alts, depth):
        best = None
        for a in alts:
            if depth[a] is None:
                continue
            if best is None or (depth[a], a) < (depth[best], best):
                best = a
        return best

    def src_depth(s, depth):
        d = -1
        for alts in s["groups"]:
            b = group_best(alts, depth)
            if b is None:
                return None
            d = max(d, depth[b])
        return d + 1

    depth = {n: None for n in items}
    changed = True
    while changed:
        changed = False
        for n in items:
            for s in sources[n]:
                sd = src_depth(s, depth)
                if sd is not None and (depth[n] is None or sd < depth[n]):
                    depth[n] = sd
                    changed = True

    chosen = {}
    for n in items:
        if depth[n] is None:
            continue
        best = None
        for s in sources[n]:
            if src_depth(s, depth) != depth[n]:
                continue
            key = (VIA_PRIORITY.get(s["via"], 9), str(s["from"] or ""), str(s["recipe_id"] or ""))
            if best is None or key < best[0]:
                best = (key, s)
        chosen[n] = best[1]

    prereqs = {}
    for n, s in chosen.items():
        prereqs[n] = sorted({group_best(g, depth) for g in s["groups"]})

    closure = {}
    for n in sorted(chosen, key=lambda x: (depth[x], x)):
        acc = set()
        for p in prereqs[n]:
            acc.add(p)
            acc |= closure[p]
        closure[n] = acc

    out = {}
    for n in sorted(items):
        e = items[n]
        if n not in chosen:
            out[n] = {"via": None, "from": None, "tool_tier": None, "tool_type": None,
                      "recipe_id": None, "depth": None, "dimension": None, "count": None,
                      "prereqs": [], "requirements": [], "alternatives": [],
                      "value": e.get("value")}
            continue
        s = chosen[n]
        alts = []
        for o in sources[n]:
            od = src_depth(o, depth)
            if o is not s and od is not None:
                alts.append({"via": o["via"], "from": o["from"], "depth": od})
        entry = {
            "via": s["via"], "from": s["from"], "tool_tier": s["tool_tier"],
            "tool_type": s["tool_type"], "recipe_id": s["recipe_id"], "depth": depth[n],
            "dimension": s["dimension"], "count": s["count"],
            "prereqs": prereqs[n], "requirements": sorted(closure[n]),
            "alternatives": sorted(alts, key=lambda a: (a["depth"], a["via"], str(a["from"]))),
            "value": e.get("value"),
        }
        if s["recipe"]:
            entry["recipe"] = s["recipe"]
        out[n] = entry

    unob = sorted(n for n in items if n not in chosen)
    graph = {
        "version": 1,
        "items": out,
        "unobtainable": unob,
        "counts": {
            "items": len(items),
            "obtainable": len(chosen),
            "max_depth": max((depth[n] for n in chosen), default=None),
            "by_via": {v: sum(1 for n in chosen if chosen[n]["via"] == v)
                       for v in sorted(VIA_PRIORITY)},
        },
    }
    return graph, warnings


def _predicate_targets(name, params, blocks, mobs, bps):
    if name in ("has_item", "crafted"):
        return {params["item"]} if params.get("item") else set()
    if name == "placed":
        return {params["block"]} if params.get("block") else set()
    if name == "mined":
        b = params.get("block")
        if b in blocks:
            drops = {d["item"] for d in blocks[b].get("drops") or [] if d.get("item")}
            return drops or {b}
        return {b} if b else set()
    if name == "killed":
        m = mobs.get(params.get("mob"))
        return {d["item"] for d in (m or {}).get("drops") or [] if d.get("item")}
    if name == "ate":
        f = params.get("food")
        return {f} if f and f != "any" else set()
    if name == "build_complete":
        bp = bps.get(str(params.get("blueprint")))
        return set((bp or {}).get("required_items", {}).keys())
    return set()


def build_shaping(graph, reg, tasks, blueprints=()):
    """Per-task prerequisite item lists for knowledge-shaping (PROTOCOL 12.7)."""
    items = graph["items"]
    blocks = {b.get("name"): b for b in reg["blocks"]}
    mobs = {m.get("name"): m for m in reg["mobs"]}
    bps = {str(b.get("id")): b for b in blueprints}
    out = {}
    for t in tasks:
        targets = set()
        for name, params in task_predicates(t):
            targets |= _predicate_targets(name, params, blocks, mobs, bps)
        targets = {x for x in targets if x in items}
        prereq = set()
        for x in targets:
            prereq |= set(items[x]["requirements"])
        prereq -= targets
        out[t["id"]] = {
            "target_items": sorted(targets),
            "items": sorted(prereq),
            "values": {p: items[p].get("value") for p in sorted(prereq)},
        }
    return {"version": 1, "tasks": out}


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _wait_for(paths, seconds):
    deadline = time.time() + seconds
    while True:
        missing = [p for p in paths if not Path(p).exists()]
        if not missing:
            return
        if time.time() > deadline:
            raise SystemExit(f"timed out waiting for {missing}")
        time.sleep(15)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(REPO / "data/registry"))
    ap.add_argument("--tasks", default=str(REPO / "data/tasks/tasks.json"))
    ap.add_argument("--blueprints", default=str(REPO / "data/builds/blueprints.json"))
    ap.add_argument("--out-dir", default=str(REPO / "data/knowledge"))
    ap.add_argument("--wait", type=float, default=0.0,
                    help="poll (seconds) until registry items.json and tasks.json exist")
    args = ap.parse_args(argv)
    if args.wait:
        _wait_for([Path(args.registry) / "items.json", args.tasks], args.wait)
    reg = load_registry(args.registry)
    graph, warnings = build_graph(reg)
    tasks = load_tasks(args.tasks) if Path(args.tasks).exists() else []
    bps = load_blueprints(args.blueprints) if Path(args.blueprints).exists() else []
    shaping = build_shaping(graph, reg, tasks, bps)
    out = Path(args.out_dir)
    _write_json(out / "graph.json", graph)
    _write_json(out / "shaping.json", shaping)
    c = graph["counts"]
    print(f"items={c['items']} obtainable={c['obtainable']} max_depth={c['max_depth']} "
          f"by_via={c['by_via']}")
    print(f"unobtainable={graph['unobtainable']}")
    print(f"shaping tasks={len(shaping['tasks'])} (tasks file: {args.tasks})")
    for w in warnings[:40]:
        print("warn:", w)
    if len(warnings) > 40:
        print(f"... {len(warnings) - 40} more warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
