"""Timeline and summary of a human session (PROTOCOL 9.5).

  .venv/bin/python -m pretrain.timeline <session dir> [--all] [--json]

Default output shows the notable events (grouped item pickups, crafts, smelts, kills, eats, deaths,
advancements) and activity segments of at least 10 s. --all prints every event and every segment.
Summary: items by category, crafts, smelts, mob kills by type and weapon, food eaten, deaths, and
time per activity (mining / building / fighting / exploring / crafting / idle).

Activity is classified per 1 s window (20 ticks), priority: fighting > mining > building > crafting >
exploring > idle. fighting = attack/kill/killed_by/damage_taken events; mining = mine events, or the
attack key held with a pickaxe/axe/shovel/hoe in hand and no fight event; building = place events;
crafting = craft/smelt events or the crafting/furnace GUI open; exploring = movement keys or position
change; everything else idle.
"""
import argparse
import bisect
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain import session as S  # noqa: E402

WINDOW = S.TICK_RATE                  # 1 s
GROUP_GAP_S = 10.0                    # consecutive pickups of one item within this gap are one line
MIN_SEGMENT_S = 10.0                  # default view hides shorter activity segments
ACTIVITIES = ["mining", "building", "fighting", "exploring", "crafting", "idle"]
NOTABLE = {"craft", "smelt", "kill", "eat", "death", "advancement", "level_up", "dimension",
           "sleep", "wake", "respawn", "enchant", "trade"}
TOOL_HANDS = ("pickaxe", "axe", "shovel", "hoe")


def hms(t):
    t = max(0.0, float(t))
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def activity_windows(sess):
    """Per-second activity label list (len = number of 1 s windows) plus the moving/attack flags."""
    nw = max(1, math.ceil(sess.n_ticks / WINDOW))
    moved = np.zeros(nw, bool)
    attack_tool = np.zeros(nw, bool)
    gui_craft = np.zeros(nw, bool)
    mine = np.zeros(nw, bool)
    place = np.zeros(nw, bool)
    fight = np.zeros(nw, bool)
    craft = np.zeros(nw, bool)
    prev = None
    for rec in sess.input_records():
        w = min(nw - 1, int(rec.get("tick", 0)) // WINDOW)
        keys = rec.get("keys") or {}
        if any(keys.get(k) for k in S.MOVE_KEYS):
            moved[w] = True
        x, z = rec.get("x"), rec.get("z")
        if prev is not None and x is not None and z is not None:
            if abs(x - prev[0]) + abs(z - prev[1]) > 0.02:
                moved[w] = True
        if x is not None and z is not None:
            prev = (x, z)
        held = S._id(rec.get("held_item")) or ""
        if keys.get("attack") and held.endswith(TOOL_HANDS):
            attack_tool[w] = True
        if rec.get("gui") in ("crafting", "furnace"):
            gui_craft[w] = True
    for e in sess.events():
        w = min(nw - 1, e["tick"] // WINDOW)
        typ = e["type"]
        if typ == "mine":
            mine[w] = True
        elif typ == "place":
            place[w] = True
        elif typ in S.FIGHT_TYPES:
            fight[w] = True
        elif typ in ("craft", "smelt"):
            craft[w] = True
    labels = []
    for w in range(nw):
        if fight[w]:
            a = "fighting"
        elif mine[w] or (attack_tool[w] and not fight[w]):
            a = "mining"
        elif place[w]:
            a = "building"
        elif craft[w] or gui_craft[w]:
            a = "crafting"
        elif moved[w]:
            a = "exploring"
        else:
            a = "idle"
        labels.append(a)
    return labels


def segments(labels):
    """Runs of identical activity: list of {activity, start_s, end_s} (window = 1 s)."""
    out = []
    for i, a in enumerate(labels):
        if out and out[-1]["activity"] == a:
            out[-1]["end_s"] = i + 1
        else:
            out.append({"activity": a, "start_s": i, "end_s": i + 1})
    return out


def build(sess):
    events = sess.events()
    labels = activity_windows(sess)
    segs = segments(labels)
    activity_s = {a: 0.0 for a in ACTIVITIES}
    for i, a in enumerate(labels):
        span = min(WINDOW, sess.n_ticks - i * WINDOW) / WINDOW
        activity_s[a] += span
    total = sum(activity_s.values()) or 1.0

    got_cat = defaultdict(lambda: {"count": 0, "items": Counter()})
    crafted, smelted, eaten = Counter(), Counter(), Counter()
    kills = defaultdict(Counter)
    deaths = damage = 0
    for e in events:
        for g in e["got"]:
            cat, _ = S.categorize(g["vanilla"] or g["item"])
            key = g["item"] or g["vanilla"]
            got_cat[cat]["count"] += g["count"]
            got_cat[cat]["items"][key] += g["count"]
        if e["type"] == "craft":
            crafted[e["got"][0]["item"] or e["vanilla"] if e["got"] else e["vanilla"]] += e["count"]
        elif e["type"] == "smelt":
            smelted[e["got"][0]["item"] or e["vanilla"] if e["got"] else e["vanilla"]] += e["count"]
        elif e["type"] == "eat":
            eaten[e["vanilla"]] += 1
        elif e["type"] == "kill" and e["entity"]:
            kills[e["entity"]][e["weapon"] or "hand"] += 1
        elif e["type"] == "death":
            deaths += 1
        elif e["type"] == "damage_taken":
            damage += 1
    summary = {
        "items_by_category": {c: {"count": v["count"], "top": v["items"].most_common(8)}
                              for c, v in sorted(got_cat.items())},
        "crafted": dict(crafted.most_common()),
        "smelted": dict(smelted.most_common()),
        "mob_kills": {k: dict(v.most_common()) for k, v in sorted(kills.items())},
        "kills_total": sum(sum(v.values()) for v in kills.values()),
        "eaten": dict(eaten.most_common()),
        "deaths": deaths,
        "damage_events": damage,
    }
    return {"events": events, "segments": segs, "activity_s": activity_s, "total_activity_s": total,
            "summary": summary}


def timeline_items(sess, data, show_all):
    """(t, text) rows sorted by time. Default: notable events, grouped pickups, long activity segments."""
    rows = []
    pending = None  # (t_start, t_last, item, count, cat)
    notable_types = set(S.OBTAIN_TYPES) | NOTABLE | {"mine", "place", "attack", "use", "drop",
                                                    "damage_taken", "killed_by", "chest_put",
                                                    "chest_take", "give", "chat"}

    def flush():
        nonlocal pending
        if pending:
            t0, _t1, item, count, cat = pending
            rows.append((t0, f"got {item} x{count} ({cat})"))
            pending = None

    for e in data["events"]:
        obtains = (e["type"] in S.OBTAIN_TYPES or e["type"] == "mine") and e["got"]
        if not show_all and e["type"] not in NOTABLE and not obtains:
            continue
        if obtains and not show_all:
            g = e["got"][0]
            item = g["item"] or g["vanilla"]
            cat, _ = S.categorize(g["vanilla"] or g["item"])
            if pending and pending[2] == item and e["t"] - pending[1] <= GROUP_GAP_S:
                pending = (pending[0], e["t"], item, pending[3] + g["count"], cat)
            else:
                flush()
                pending = (e["t"], e["t"], item, g["count"], cat)
            continue
        flush()
        rows.append((e["t"], _event_text(e)))
    flush()
    for seg in data["segments"]:
        dur = seg["end_s"] - seg["start_s"]
        if show_all or (dur >= MIN_SEGMENT_S and seg["activity"] != "idle"):
            rows.append((seg["start_s"], f"-- {seg['activity']} for {dur}s"))
    rows.sort(key=lambda r: r[0])
    return rows


def timeline_rows(sess, data, show_all):
    return [f"{hms(t)}  {txt}" for t, txt in timeline_items(sess, data, show_all)]


def context_summary(data, start_tick, end_tick):
    """Summary of one context segment (9.6): events inside it, items, crafts, kills, eats, activity time."""
    evs = [e for e in data["events"] if start_tick <= e["tick"] < end_tick]
    items = Counter()
    crafts = kills = eats = 0
    for e in evs:
        for g in e["got"]:
            items[g["item"] or g["vanilla"]] += g["count"]
        if e["type"] == "craft":
            crafts += 1
        elif e["type"] == "kill":
            kills += 1
        elif e["type"] == "eat":
            eats += 1
    s0, s1 = start_tick / S.TICK_RATE, end_tick / S.TICK_RATE
    act = {a: 0.0 for a in ACTIVITIES}
    for seg in data["segments"]:
        ov = min(seg["end_s"], s1) - max(seg["start_s"], s0)
        if ov > 0:
            act[seg["activity"]] += ov
    return {"events": len(evs), "items": dict(items.most_common()), "crafts": crafts, "kills": kills,
            "eats": eats, "activity_s": act}


def summary_line(cs):
    parts = [f"{cs['events']} events"]
    if cs["items"]:
        parts.append("got " + ", ".join(f"{k} x{n}" for k, n in list(cs["items"].items())[:6]))
    if cs["crafts"]:
        parts.append(f"{cs['crafts']} crafts")
    if cs["kills"]:
        parts.append(f"{cs['kills']} kills")
    if cs["eats"]:
        parts.append(f"{cs['eats']} eats")
    act = [f"{a} {hms(v)}" for a, v in cs["activity_s"].items() if v >= 1]
    parts.append("time: " + (", ".join(act) or "-"))
    return "summary: " + "; ".join(parts)


def context_blocks(sess, data, show_all):
    """[(header or None, [row text], summary or None)] in time order. Without contexts: one flat block.
    With contexts: one block per segment ("== start-end text ==", its events, a summary) and one
    "(no context)" block per gap that has events."""
    ctxs = sess.contexts()
    rows = timeline_items(sess, data, show_all)
    if not ctxs:
        return [(None, [f"{hms(t)}  {txt}" for t, txt in rows], None)]
    starts = [c["start_tick"] / S.TICK_RATE for c in ctxs]
    dur = sess.n_ticks / S.TICK_RATE
    blocks = {}
    for k, c in enumerate(ctxs):
        s0, s1 = c["start_tick"] / S.TICK_RATE, c["end_tick"] / S.TICK_RATE
        blocks[("ctx", k)] = {"start": s0, "header": f"== {hms(s0)}-{hms(s1)} {c['text']} ==",
                              "lines": [], "summary": summary_line(
                                  context_summary(data, c["start_tick"], c["end_tick"]))}
    for t, txt in rows:
        k = bisect.bisect_right(starts, t) - 1
        if k >= 0 and t < ctxs[k]["end_tick"] / S.TICK_RATE:
            key = ("ctx", k)
        else:
            g = k + 1                                   # gap after context k (k = -1: before the first)
            key = ("gap", g)
            if key not in blocks:
                g_start = ctxs[g - 1]["end_tick"] / S.TICK_RATE if g > 0 else 0.0
                g_end = starts[g] if g < len(ctxs) else dur
                blocks[key] = {"start": g_start, "header": f"== {hms(g_start)}-{hms(g_end)} (no context) ==",
                               "lines": [], "summary": None}
        blocks[key]["lines"].append(f"{hms(t)}  {txt}")
    out = sorted(blocks.values(), key=lambda b: b["start"])
    return [(b["header"], b["lines"], b["summary"]) for b in out]


def _event_text(e):
    typ = e["type"]
    if typ == "craft":
        return f"crafted {e['got'][0]['item'] or e['vanilla']}" + (f" x{e['count']}" if e["count"] > 1 else "")
    if typ == "smelt":
        return f"smelted {e['got'][0]['item'] or e['vanilla']}"
    if typ == "kill":
        return f"killed {e['entity']} with {e['weapon'] or 'hand'}"
    if typ == "killed_by":
        return f"killed by {e['entity']}"
    if typ == "eat":
        return f"ate {e['vanilla']}"
    if typ == "death":
        return "died"
    if typ == "damage_taken":
        return f"took damage ({e['count']})"
    if typ == "mine":
        return f"mined {e['block']}"
    if typ == "place":
        return f"placed {e['block'] or e['vanilla']}"
    if typ == "attack":
        return f"attacked {e['entity'] or '?'} with {e['weapon'] or 'hand'}"
    if typ in ("pickup", "drop", "chest_put", "chest_take", "use", "give", "trade"):
        return f"{typ} {e['vanilla'] or e['block'] or ''} x{e['count']}".rstrip()
    if typ == "advancement":
        return f"advancement {e['vanilla']}"
    return f"{typ} {e['vanilla'] or ''}".rstrip()


def print_text(sess, data, show_all):
    print(f"session {sess.path}")
    print(f"format v{sess.version}  goal={sess.goal}  player={sess.meta.get('player')}  "
          f"duration {hms(sess.n_ticks / S.TICK_RATE)} ({sess.n_ticks} ticks, "
          f"{sess.n_ticks // S.TICKS_PER_STEP} sim steps)")
    for w in sess.warnings:
        print(f"warning: {w}")
    print()
    print("timeline" + (" (all events)" if show_all else ""))
    for header, lines, summary in context_blocks(sess, data, show_all):
        indent = "  "
        if header:
            print(f"  {header}")
            indent = "    "
        for row in lines:
            print(indent + row)
        if summary:
            print(f"    {summary}")
    print()
    s = data["summary"]
    print("summary")
    print("  items by category:")
    for c, v in s["items_by_category"].items():
        top = ", ".join(f"{k} x{n}" for k, n in v["top"])
        print(f"    {c:10s} {v['count']:6d}  {top}")
    print("  crafted:  " + (", ".join(f"{k} x{n}" for k, n in s["crafted"].items()) or "-"))
    print("  smelted:  " + (", ".join(f"{k} x{n}" for k, n in s["smelted"].items()) or "-"))
    print(f"  mob kills: {s['kills_total']}")
    for mob, weps in s["mob_kills"].items():
        print(f"    {mob:24s} " + ", ".join(f"{w} x{n}" for w, n in weps.items()))
    print("  eaten:    " + (", ".join(f"{k} x{n}" for k, n in s["eaten"].items()) or "-"))
    print(f"  deaths {s['deaths']}, damage events {s['damage_events']}")
    print("  time per activity:")
    total = data["total_activity_s"]
    for a in ACTIVITIES:
        sec = data["activity_s"][a]
        print(f"    {a:10s} {hms(sec)}  {100 * sec / total:5.1f}%")


def to_json(sess, data, show_all):
    return {
        "path": sess.path, "version": sess.version, "goal": sess.goal,
        "player": sess.meta.get("player"), "ticks": sess.n_ticks,
        "duration_s": sess.n_ticks / S.TICK_RATE, "sim_steps": sess.n_ticks // S.TICKS_PER_STEP,
        "warnings": sess.warnings, "summary": data["summary"],
        "activity_s": data["activity_s"],
        "segments": [{**s, "activity": s["activity"]} for s in data["segments"]],
        "timeline": timeline_rows(sess, data, show_all),
        "contexts": [{**c, "summary": context_summary(data, c["start_tick"], c["end_tick"])}
                     for c in sess.contexts()],
        "events": [{k: v for k, v in e.items() if k != "gui"} for e in data["events"]],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session dir (v2 or v1 recording), or a folder of sessions")
    ap.add_argument("--all", action="store_true", help="every event and every activity segment")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    dirs = S.find_sessions(args.session)
    if not dirs:
        raise SystemExit(f"no sessions under {args.session}")
    out = []
    for d in dirs:
        sess = S.open_session(d)
        data = build(sess)
        if args.json:
            out.append(to_json(sess, data, args.all))
        else:
            print_text(sess, data, args.all)
            print()
    if args.json:
        print(json.dumps(out[0] if len(out) == 1 else out, indent=1, default=str))


if __name__ == "__main__":
    main()
