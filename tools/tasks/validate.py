#!/usr/bin/env python3
"""Validate data/tasks/tasks.json against PROTOCOL.md 12.5 + 12.8.

Checks: schema/types, closed predicate set, family set, count 40-64, unique ids,
prerequisites exist, DAG acyclic, milestone rewards sane, and names exist in
data/registry/*.json and data/builds/blueprints.json when those files exist.
Usage: python3 -I tools/tasks/validate.py [--tasks PATH] [--root DIR]
Exit code 0 = no errors (warnings allowed), 1 = errors.
"""
import argparse, json, os, re, sys
from collections import deque

FAMILIES = {"gather", "food", "tools", "build", "combat", "survival", "nether", "explore", "social"}
TOP_KEYS = {"id", "name", "family", "description", "params", "success", "milestones",
            "max_steps", "prerequisites", "start_kit", "source"}
REQUIRED = TOP_KEYS - {"start_kit"}
DIMS = {"overworld", "nether"}

# predicate type -> {key: validator}; keys listed are exactly the allowed (and required) keys
def _str(v): return isinstance(v, str) and v != ""
def _pos_int(v): return isinstance(v, int) and not isinstance(v, bool) and v >= 1
def _nonneg_int(v): return isinstance(v, int) and not isinstance(v, bool) and v >= 0
def _frac(v): return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 1
def _num(v): return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
def _dim(v): return v in DIMS
def _food(v): return _str(v)  # name or "any"; checked separately

PREDICATES = {
    # 12.5
    "has_item":        {"item": _str, "count": _pos_int},
    "crafted":         {"item": _str},
    "mined":           {"block": _str, "count": _pos_int},
    "killed":          {"mob": _str, "count": _pos_int},
    "ate":             {"food": _food},
    "slept":           {},
    "survive_night":   {},
    "placed":          {"block": _str, "count": _pos_int},
    "build_complete":  {"blueprint": _str, "min_progress": _frac},
    "portal_lit":      {},
    "in_dimension":    {"dim": _dim},
    "reached_y":       {"max_y": lambda v: isinstance(v, int) and not isinstance(v, bool)},
    "distance_travelled": {"n": _pos_int},
    "health_above":    {"n": _num},
    # 12.8 social
    "near_teammate":   {"dist": _pos_int, "steps": _pos_int},
    "look_at_teammate": {"count": _pos_int},
    "crouch_greet":    {"count": _pos_int},
    "follow_teammate": {"steps": _pos_int},
    "gave_item":       {"count": _pos_int},
    "shared_chest":    {"count": _pos_int},
    "msg_near_teammate": {"count": _pos_int},
}

def load_names(path, key="name"):
    """Return a set of names from a registry/builds JSON file (list of dicts or dict keyed by name)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ("items", "blocks", "mobs", "blueprints", "entries"):
            if k in data and isinstance(data[k], list):
                data = data[k]; break
        else:
            return set(data.keys())
    names = set()
    for d in data:
        if isinstance(d, dict):
            v = d.get(key, d.get("id"))
            if isinstance(v, str): names.add(v)
        elif isinstance(d, str):
            names.add(d)
    return names

def check_predicate(p, where, errs):
    if not isinstance(p, dict) or "type" not in p:
        errs.append(f"{where}: predicate must be an object with 'type'"); return None
    t = p["type"]
    if t not in PREDICATES:
        errs.append(f"{where}: predicate type '{t}' not in closed set"); return None
    spec = PREDICATES[t]
    extra = set(p) - set(spec) - {"type"}
    missing = set(spec) - set(p)
    if extra: errs.append(f"{where}: '{t}' has unknown keys {sorted(extra)}")
    if missing: errs.append(f"{where}: '{t}' missing keys {sorted(missing)}")
    for k, check in spec.items():
        if k in p and not check(p[k]):
            errs.append(f"{where}: '{t}'.{k} invalid value {p[k]!r}")
    return t

def topo_cycle(ids, prereqs):
    indeg = {i: 0 for i in ids}
    children = {i: [] for i in ids}
    for i in ids:
        for pre in prereqs.get(i, []):
            if pre in ids:
                children[pre].append(i); indeg[i] += 1
    q = deque(i for i in ids if indeg[i] == 0); seen = 0
    while q:
        n = q.popleft(); seen += 1
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0: q.append(c)
    if seen == len(ids): return None
    return sorted(i for i in ids if indeg[i] > 0)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(default_root, "data", "tasks", "tasks.json"))
    ap.add_argument("--root", default=default_root)
    a = ap.parse_args()
    errs, warns, notes = [], [], []

    with open(a.tasks) as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        print("ERROR: top level must be a list"); return 1
    if not (40 <= len(tasks) <= 64):
        errs.append(f"task count {len(tasks)} outside 40..64")

    ids = []
    for i, t in enumerate(tasks):
        w = f"task[{i}]" + (f" ({t.get('id')})" if isinstance(t, dict) and t.get("id") else "")
        if not isinstance(t, dict):
            errs.append(f"{w}: not an object"); continue
        keys = set(t)
        if keys - TOP_KEYS: errs.append(f"{w}: unknown keys {sorted(keys - TOP_KEYS)}")
        if REQUIRED - keys: errs.append(f"{w}: missing keys {sorted(REQUIRED - keys)}")
        tid = t.get("id")
        if not (isinstance(tid, str) and re.fullmatch(r"[a-z][a-z0-9_]*", tid)):
            errs.append(f"{w}: id must be snake_case string")
        else:
            w = f"task {tid}"
            ids.append(tid)
        for k in ("name", "description", "source"):
            if not _str(t.get(k)): errs.append(f"{w}: '{k}' must be a non-empty string")
        if t.get("family") not in FAMILIES: errs.append(f"{w}: family {t.get('family')!r} not in {sorted(FAMILIES)}")
        if not isinstance(t.get("params"), dict): errs.append(f"{w}: params must be an object")
        ms = t.get("max_steps")
        if not (isinstance(ms, int) and not isinstance(ms, bool) and 100 <= ms <= 100000):
            errs.append(f"{w}: max_steps must be int in 100..100000 (got {ms!r})")
        pre = t.get("prerequisites")
        if not isinstance(pre, list) or not all(isinstance(x, str) for x in pre):
            errs.append(f"{w}: prerequisites must be a list of task ids")
        if "start_kit" in t:
            sk = t["start_kit"]
            if not isinstance(sk, dict) or not all(_str(k) and _pos_int(v) for k, v in sk.items()):
                errs.append(f"{w}: start_kit must map item name -> positive int")
        check_predicate(t.get("success"), f"{w}.success", errs)
        ms_list = t.get("milestones")
        if not isinstance(ms_list, list):
            errs.append(f"{w}: milestones must be a list")
        else:
            if not ms_list: warns.append(f"{w}: no milestones (no dense reward)")
            total = 0.0
            for j, m in enumerate(ms_list):
                if not isinstance(m, dict) or set(m) != {"predicate", "reward"}:
                    errs.append(f"{w}.milestones[{j}]: must be {{predicate, reward}}"); continue
                check_predicate(m["predicate"], f"{w}.milestones[{j}]", errs)
                r = m["reward"]
                if not (isinstance(r, (int, float)) and not isinstance(r, bool) and 0 < r <= 0.5):
                    errs.append(f"{w}.milestones[{j}]: reward must be in (0, 0.5]")
                else:
                    total += r
            if total >= 1.0: errs.append(f"{w}: milestone rewards sum to {total:.2f} >= 1.0")

    dup = {x for x in ids if ids.count(x) > 1}
    if dup: errs.append(f"duplicate ids: {sorted(dup)}")
    idset = set(ids)
    prereqs = {}
    for t in tasks:
        if not isinstance(t, dict) or "id" not in t: continue
        pre = t.get("prerequisites", []) if isinstance(t.get("prerequisites"), list) else []
        prereqs[t["id"]] = pre
        for p in pre:
            if p not in idset: errs.append(f"task {t['id']}: prerequisite '{p}' does not exist")
            if p == t["id"]: errs.append(f"task {t['id']}: prerequisite on itself")
    cyc = topo_cycle(idset, prereqs)
    if cyc: errs.append(f"prerequisite cycle among: {cyc}")
    roots = sorted(i for i in idset if not any(p in idset for p in prereqs.get(i, [])))
    notes.append(f"{len(tasks)} tasks; curriculum roots ({len(roots)}): {', '.join(roots)}")

    fams = {}
    for t in tasks:
        if isinstance(t, dict): fams[t.get("family")] = fams.get(t.get("family"), 0) + 1
    missing_f = FAMILIES - set(fams)
    if missing_f: errs.append(f"families with no tasks: {sorted(missing_f)}")
    notes.append("per family: " + ", ".join(f"{k}={fams[k]}" for k in sorted(fams, key=str)))

    # name checks against data files (only where files exist)
    reg = os.path.join(a.root, "data", "registry")
    bld = os.path.join(a.root, "data", "builds", "blueprints.json")
    sets = {}
    for key, path in (("items", os.path.join(reg, "items.json")), ("blocks", os.path.join(reg, "blocks.json")),
                      ("mobs", os.path.join(reg, "mobs.json")), ("blueprints", bld)):
        if os.path.exists(path):
            try:
                sets[key] = load_names(path)
            except Exception as e:
                errs.append(f"could not parse {path}: {e}")
        else:
            notes.append(f"names: {os.path.relpath(path, a.root)} missing, {key} not checked")
    def need(key, name, where):
        if key in sets and name not in sets[key]:
            errs.append(f"{where}: '{name}' not in {key}")
    for t in tasks:
        if not isinstance(t, dict): continue
        w = f"task {t.get('id')}"
        preds = [("success", t.get("success"))] + [(f"milestone[{j}]", m.get("predicate"))
                 for j, m in enumerate(t.get("milestones") or []) if isinstance(m, dict)]
        for where, p in preds:
            if not isinstance(p, dict): continue
            ty = p.get("type")
            if ty == "has_item" or ty == "crafted":
                need("items", p.get("item"), f"{w}.{where}")
            elif ty == "mined" or ty == "placed":
                need("blocks", p.get("block"), f"{w}.{where}")
            elif ty == "killed":
                need("mobs", p.get("mob"), f"{w}.{where}")
            elif ty == "ate" and p.get("food") != "any":
                need("items", p.get("food"), f"{w}.{where}")
            elif ty == "build_complete":
                need("blueprints", p.get("blueprint"), f"{w}.{where}")
        for item in (t.get("start_kit") or {}):
            need("items", item, f"{w}.start_kit")
        if t.get("family") == "build" or t.get("family") == "nether":
            bp = (t.get("params") or {}).get("blueprint")
            if bp: need("blueprints", bp, f"{w}.params")

    for n in notes: print("note:", n)
    for x in warns: print("warn:", x)
    for e in errs: print("ERROR:", e)
    print(f"RESULT: {len(errs)} error(s), {len(warns)} warning(s)")
    return 1 if errs else 0

if __name__ == "__main__":
    sys.exit(main())
