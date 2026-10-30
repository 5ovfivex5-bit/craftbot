"""Closed vocabularies, predicate normalisation and validation for tasks and env mixes.

Pure Python. No simulator imports.

Task dicts (data/tasks/tasks.json, PROTOCOL 12.5):
  {id, name, family, description, params, success, milestones: [{predicate, reward}],
   max_steps, prerequisites: [task ids], start_kit: {item: n}, source}
Predicates may be written as {"type": "has_item", "item": ..., "count": ...},
{"has_item": {"item": ..., "count": ...}}, or a list of either. Milestone entries
({"predicate": ..., "reward": ...}) are accepted too.
"""
from __future__ import annotations

FAMILIES = ("gather", "food", "tools", "build", "combat", "survival", "nether", "explore", "social")

PREDICATES = frozenset({
    "has_item", "crafted", "mined", "killed", "ate", "slept", "survive_night", "placed",
    "build_complete", "portal_lit", "in_dimension", "reached_y", "distance_travelled",
    "health_above",
    # 12.8 social family
    "near_teammate", "look_at_teammate", "crouch_greet", "follow_teammate", "gave_item",
    "shared_chest", "msg_near_teammate",
})

_TYPE_KEYS = ("type", "kind", "name")
_COMPOSITE_KEYS = ("all", "any", "and", "or")


def normalize_predicates(obj):
    """Yield (predicate_name, params_dict) for every predicate inside a success/milestone spec."""
    if obj is None:
        return
    if isinstance(obj, str):
        if obj in PREDICATES:
            yield obj, {}
        return
    if isinstance(obj, list):
        for x in obj:
            yield from normalize_predicates(x)
        return
    if not isinstance(obj, dict):
        return
    if "predicate" in obj:  # milestone entry {predicate, reward}
        yield from normalize_predicates(obj["predicate"])
        return
    for k in _TYPE_KEYS:
        if isinstance(obj.get(k), str) and obj[k] in PREDICATES:
            params = {kk: vv for kk, vv in obj.items() if kk not in _TYPE_KEYS}
            yield obj[k], params
            return
    for k, v in obj.items():
        if k in PREDICATES:
            yield k, (dict(v) if isinstance(v, dict) else {"value": v})
        elif k in _COMPOSITE_KEYS:
            yield from normalize_predicates(v)


def task_predicates(task):
    """All (name, params) predicates of a task: success plus milestones."""
    out = list(normalize_predicates(task.get("success")))
    for m in task.get("milestones") or []:
        out.extend(normalize_predicates(m))
    return out


def _fmt_params(params):
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def describe(task, status=None):
    """One-line human description of a task. `status` is an optional sampler status dict."""
    if isinstance(task, str):
        raise TypeError("describe() takes a task dict; look the id up first")
    head = f"{task.get('id')} [{task.get('family')}]"
    if task.get("name"):
        head += f" {task['name']}"
    bits = [head]
    if task.get("description"):
        bits.append(task["description"])
    succ = [f"{n}({_fmt_params(p)})" for n, p in normalize_predicates(task.get("success"))]
    bits.append("success: " + (", ".join(succ) or "?"))
    if task.get("prerequisites"):
        bits.append("after: " + ", ".join(task["prerequisites"]))
    if task.get("start_kit"):
        bits.append("kit: " + _fmt_params(task["start_kit"]))
    if task.get("max_steps"):
        bits.append(f"max_steps={task['max_steps']}")
    if status:
        bits.append(
            f"unlocked={status['unlocked']} ema={status['ema']:.2f} "
            f"episodes={status['episodes']} mastered={status['mastered']}")
    return " | ".join(bits)


def _has_cycle(graph):
    state = {}

    def visit(n, stack):
        if state.get(n) == 1:
            return stack[stack.index(n):] + [n] if n in stack else [n]
        if state.get(n) == 2:
            return None
        state[n] = 1
        stack.append(n)
        for m in graph.get(n, ()):
            c = visit(m, stack)
            if c:
                return c
        stack.pop()
        state[n] = 2
        return None

    for n in graph:
        c = visit(n, [])
        if c:
            return c
    return None


def validate_tasks(tasks):
    """Return a list of human-readable errors (empty when valid)."""
    errors = []
    ids = set()
    for i, t in enumerate(tasks):
        tid = t.get("id")
        where = f"task[{i}] {tid!r}"
        if not isinstance(tid, str) or not tid:
            errors.append(f"{where}: missing string id")
            continue
        if tid in ids:
            errors.append(f"{where}: duplicate id")
        ids.add(tid)
        if t.get("family") not in FAMILIES:
            errors.append(f"{where}: unknown family {t.get('family')!r}")
        if not list(normalize_predicates(t.get("success"))):
            errors.append(f"{where}: no recognised success predicate")
        for name, _ in task_predicates(t):
            if name not in PREDICATES:  # pragma: no cover - normalize already filters
                errors.append(f"{where}: unknown predicate {name!r}")
        ms = t.get("max_steps")
        if ms is not None and (not isinstance(ms, int) or ms <= 0):
            errors.append(f"{where}: max_steps must be a positive int")
    graph = {}
    for t in tasks:
        tid = t.get("id")
        for p in t.get("prerequisites") or []:
            if p not in ids:
                errors.append(f"task {tid!r}: unknown prerequisite {p!r}")
        graph[tid] = list(t.get("prerequisites") or [])
    cyc = _has_cycle(graph)
    if cyc:
        errors.append("prerequisite cycle: " + " -> ".join(map(str, cyc)))
    return errors


def validate_envmix(envmix):
    """Return a list of errors for an env-mix dict (default.json shape)."""
    errors = []
    mix = envmix.get("mix")
    if not isinstance(mix, dict) or not mix:
        errors.append("mix must be a non-empty dict of family -> weight")
    else:
        for fam, w in mix.items():
            if fam not in FAMILIES:
                errors.append(f"mix: unknown family {fam!r}")
            if not isinstance(w, (int, float)) or w < 0:
                errors.append(f"mix: weight for {fam!r} must be >= 0")
        if sum(float(w) for w in mix.values() if isinstance(w, (int, float))) <= 0:
            errors.append("mix: weights sum to zero")
    cur = envmix.get("curriculum", {})
    for key in ("unlock_ema_threshold", "mastered_ema_threshold"):
        v = cur.get(key)
        if v is not None and not (isinstance(v, (int, float)) and 0.0 <= v <= 1.0):
            errors.append(f"curriculum.{key} must be in [0, 1]")
    a = cur.get("ema_alpha")
    if a is not None and not (isinstance(a, (int, float)) and 0.0 < a <= 1.0):
        errors.append("curriculum.ema_alpha must be in (0, 1]")
    rw = cur.get("review_weight")
    if rw is not None and not (isinstance(rw, (int, float)) and rw >= 0):
        errors.append("curriculum.review_weight must be >= 0")
    kits = envmix.get("kits", {})
    p = kits.get("start_kit_probability")
    if p is not None and not (isinstance(p, (int, float)) and 0.0 <= p <= 1.0):
        errors.append("kits.start_kit_probability must be in [0, 1]")
    for r in kits.get("rules", []):
        if r.get("family") not in FAMILIES:
            errors.append(f"kits.rules: unknown family {r.get('family')!r}")
    return errors
