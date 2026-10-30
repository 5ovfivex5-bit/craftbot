#!/usr/bin/env python3
"""Sim-vs-real parity check for one bot dump (format minemind-parity-1, see DUMP_FORMAT.md).

Usage (run inside the distrobox):
  distrobox enter hi -- bash -c 'cd /home/deck/claude/ai/test/craftbot && .venv/bin/python tools/parity/compare.py DUMP.json'
  ... compare.py DUMP.json --json report.json --check-mapping --include-bodies
  ... compare.py --selftest [--workdir DIR]

What it does for one dump:
  1. Rebuilds the dumped scene for the sim: blocks around the bot, teammates, chests, pose, inventory.
     The sim has no block/position setter, so simobs.py re-runs the sim's own raycast and writeObs
     formulas on the dumped blocks (same constants and operation order as src/sim.cpp).
  2. Computes the sim obs for that pose and compares it with the obs the Minecraft mod computed:
     - vec: index by index, reported by VEC name (formulas for every name the sim defines)
     - img: block id and depth per pixel, over the pixels whose rays stay inside the dumped cube
  3. Reports coverage (rays that needed blocks outside the dump are unknown, not matches).

--selftest builds a craftsim world, dumps every agent from the sim itself and checks that the
comparison reproduces the sim's obs exactly. It also runs two negative controls (one inventory
change, one block change) to show the comparison is sensitive.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # craftbot/: craftsim.py lives here
sys.path.insert(0, HERE)
sys.path.insert(1, ROOT)

import numpy as np  # noqa: E402

import dumpio  # noqa: E402
import simobs  # noqa: E402

VEC_TOL = 1e-4


def compare_dump(d, check_mapping=False, include_bodies=False):
    """d: loaded dump (dumpio.load). Returns a report dict."""
    arr = d["_arr"]
    b = d["blocks"]
    bl = b["build_limit"]
    scene = simobs.Scene(tuple(b["origin"]), dumpio.block_grid_from_ours(arr["ours"]),
                         (bl["min_y"], bl["max_y"]))
    pose = d["pose"]
    px, py, pz = pose["x"], pose["y"], pose["z"]
    yaw = float(np.float32(pose["yaw"]))
    pitch = float(np.float32(pose["pitch"]))
    res = int(arr["img"].shape[0])

    mates = [(t["x"], t["y"], t["z"]) for t in sorted(d["teammates"], key=lambda t: t["index"])]
    n_bodies = len(d.get("bodies", []))
    if include_bodies:
        mates += [(q["x"], q["y"], q["z"]) for q in d.get("bodies", [])]

    report = {"file": None, "bot": d.get("bot"), "source": d.get("source"), "res": res,
              "radius": b["radius"], "dump_bodies": n_bodies, "bodies_used": include_bodies}

    # ---- image
    sim_img, known = simobs.render(scene, px, py, pz, yaw, pitch, res, mates)
    ref = arr["img"].astype(np.int32)
    sim = sim_img.astype(np.int32)
    n_known = int(known.sum())
    n_total = res * res
    id_eq = (sim[..., 0] == ref[..., 0]) & known
    dep_eq = (sim[..., 1] == ref[..., 1]) & known
    dep_pm1 = (np.abs(sim[..., 1] - ref[..., 1]) <= 1) & known
    pct = lambda n, m: (100.0 * n / m) if m else float("nan")  # noqa: E731
    img_rep = {
        "pixels": n_total,
        "known_pixels": n_known,
        "coverage_pct": pct(n_known, n_total),
        "block_id_exact_pct": pct(int(id_eq.sum()), n_known),
        "depth_exact_pct": pct(int(dep_eq.sum()), n_known),
        "depth_within1_pct": pct(int(dep_pm1.sum()), n_known),
        "block_id_mismatch_pixels": int(n_known - id_eq.sum()),
    }
    # Sample of the worst pixels (by ref id) for debugging.
    bad = np.argwhere(known & ~id_eq)
    img_rep["first_mismatches"] = [
        {"row": int(r), "col": int(c), "sim": [int(sim[r, c, 0]), int(sim[r, c, 1])],
         "mc": [int(ref[r, c, 0]), int(ref[r, c, 1])]} for r, c in bad[:8]]
    report["image"] = img_rep

    # ---- vec
    names = arr["vec_names"]
    ref_vec = arr["vec"]
    vals = simobs.vec_values(scene, d)
    rows = []
    counts = {"compared": 0, "mismatch": 0, "no_formula": 0, "unknown_input": 0}
    for idx, name in enumerate(names):
        mc = float(ref_vec[idx])
        if name not in vals:
            counts["no_formula"] += 1
            rows.append({"index": idx, "name": name, "status": "no_formula", "mc": mc})
            continue
        sv = vals[name]
        if math.isnan(sv):
            counts["unknown_input"] += 1
            rows.append({"index": idx, "name": name, "status": "unknown_input", "mc": mc})
            continue
        counts["compared"] += 1
        diff = abs(sv - mc)
        ok = diff <= VEC_TOL
        if not ok:
            counts["mismatch"] += 1
        rows.append({"index": idx, "name": name, "status": "ok" if ok else "MISMATCH",
                     "sim": sv, "mc": mc, "abs_diff": diff})
    report["vec"] = {"counts": counts, "tol": VEC_TOL, "rows": rows}

    # ---- vision / mapping notes
    vis = d.get("vision", {})
    notes = []
    if vis and (abs(float(vis.get("fov_deg", 70.0)) - 70.0) > 1e-9 or abs(float(vis.get("view_dist", 24.0)) - 24.0) > 1e-9):
        notes.append("mod vision params differ from sim (fov/view_dist)")
    if n_bodies and not include_bodies:
        notes.append(f"{n_bodies} non-team bodies in dump were not rendered (sim has no such AGENT); use --include-bodies")
    if b["radius"] < 16:
        notes.append("radius < 16: chest search and crosshair chest may be unknown")
    report["notes"] = notes

    if check_mapping:
        report["mapping"] = check_block_mapping(d)
    return report


def expected_ours(name):
    """Obs.blockId() for vanilla ids whose class is unambiguous. None = no rule (not checked)."""
    if name == "minecraft:air":
        return 0
    n = name.split(":", 1)[-1]
    if n in ("grass_block", "mycelium", "podzol"):
        return 1
    if n in ("dirt", "coarse_dirt", "rooted_dirt", "dirt_path", "farmland", "mud", "clay"):
        return 2
    if n == "stone":
        return 3
    if n in ("cobblestone", "mossy_cobblestone", "cobbled_deepslate"):
        return 4
    if n.endswith("_log") or n.endswith("_wood") or n.endswith("_stem") or n.startswith("stripped_"):
        return 5
    if n.endswith("_leaves"):
        return 6
    if n.endswith("_planks"):
        return 7
    if n == "crafting_table":
        return 8
    if n == "furnace":
        return 9
    if n in ("coal_ore", "deepslate_coal_ore"):
        return 10
    if n in ("iron_ore", "deepslate_iron_ore"):
        return 11
    if n in ("diamond_ore", "deepslate_diamond_ore"):
        return 12
    if n == "bedrock":
        return 13
    if n in ("sand", "red_sand"):
        return 14
    if n == "gravel":
        return 15
    if n == "water":
        return 16
    if n in ("deepslate", "tuff"):
        return 17
    if n in ("chest", "trapped_chest"):
        return 18
    return None


def check_block_mapping(d):
    """Counts cells whose ours id disagrees with expected_ours() for the vanilla id (mapping drift)."""
    vanilla = dumpio.decode_vanilla(d)
    ours = d["_arr"]["ours"]
    per = {}
    for name in np.unique(vanilla):
        if name == "unloaded":
            continue
        exp = expected_ours(name)
        if exp is None:
            continue
        sel = vanilla == name
        got = ours[sel]
        bad = got != exp
        if bad.any():
            hist = {}
            for g in got[bad].tolist():
                hist[int(g)] = hist.get(int(g), 0) + 1
            per[name] = {"expected": exp, "cells": int(sel.sum()), "wrong": int(bad.sum()), "got_hist": hist}
    return {"wrong_by_vanilla_id": per}


def print_report(rep, out=sys.stdout, max_rows=40):
    p = lambda *a: print(*a, file=out)  # noqa: E731
    img = rep["image"]
    vc = rep["vec"]["counts"]
    p(f"bot {rep['bot']}  source={rep['source']}  res={rep['res']}  radius={rep['radius']}")
    p(f"image: coverage {img['coverage_pct']:.1f}% ({img['known_pixels']}/{img['pixels']} px known)")
    p(f"  block id exact {img['block_id_exact_pct']:.2f}%   depth exact {img['depth_exact_pct']:.2f}%"
      f"   depth within 1 {img['depth_within1_pct']:.2f}%   (over known px)")
    p(f"vec: {vc['compared']} compared, {vc['mismatch']} mismatch (tol {rep['vec']['tol']}), "
      f"{vc['no_formula']} no sim formula, {vc['unknown_input']} unknown input")
    mism = [r for r in rep["vec"]["rows"] if r["status"] == "MISMATCH"]
    mism.sort(key=lambda r: -r["abs_diff"])
    if mism:
        p("  mismatches by name (sim vs mc):")
        for r in mism[:max_rows]:
            p(f"    [{r['index']:3d}] {r['name']:<22s} sim {r['sim']:+.6f}  mc {r['mc']:+.6f}  |d| {r['abs_diff']:.6f}")
    nofm = sorted({r["name"] for r in rep["vec"]["rows"] if r["status"] == "no_formula"})
    if nofm:
        p(f"  no sim formula for: {', '.join(nofm)}")
    if img["block_id_mismatch_pixels"]:
        p(f"  first block-id mismatches (row,col: sim[id,depth] vs mc[id,depth]):")
        for m in img["first_mismatches"][:6]:
            p(f"    ({m['row']},{m['col']}): {m['sim']} vs {m['mc']}")
    if "mapping" in rep:
        wrong = rep["mapping"]["wrong_by_vanilla_id"]
        p(f"  block mapping: {len(wrong)} vanilla ids disagree with expected ours")
        for name, info in sorted(wrong.items()):
            p(f"    {name}: expected {info['expected']}, wrong {info['wrong']}/{info['cells']} got {info['got_hist']}")
    for n in rep["notes"]:
        p(f"  note: {n}")


def compare_file(path, check_mapping=False, include_bodies=False):
    d = dumpio.load(path)
    rep = compare_dump(d, check_mapping=check_mapping, include_bodies=include_bodies)
    rep["file"] = path
    return rep, d


# ---------------------------------------------------------------- self-test

def _import_sim():
    try:
        import craftsim  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    return craftsim, None


def selftest(workdir, seed=1234, steps=150, verbose=True):
    craftsim, err = _import_sim()
    if craftsim is None:
        print(f"SELFTEST SKIPPED: cannot import craftsim ({err}). Rebuild the sim, then retry.")
        return 2
    if list(craftsim.BLOCKS) != dumpio.BLOCKS or list(craftsim.ITEMS) != dumpio.ITEMS:
        print("SELFTEST FAILED: craftsim BLOCKS/ITEMS differ from dumpio.py; update the tables.")
        return 1
    if craftsim.VEC_DIM != 102 and verbose:
        print(f"note: VEC_DIM is {craftsim.VEC_DIM}; formulas are checked by name anyway")

    os.makedirs(workdir, exist_ok=True)
    sim = craftsim.CraftSim(num_envs=2, agents=3, seed=seed, res=64, max_steps=3000, ticks=4, threads=1)
    sim.reset()
    for i in range(sim.num_agents):
        for item, n in (("planks", 12), ("cobblestone", 9), ("chest", 2), ("crafting_table", 1),
                        ("coal", 3), ("iron_ore", 2), ("stick", 2)):
            sim.set_inventory(i, item, n)
    rng = np.random.default_rng(seed)
    N = sim.num_agents
    pool = [a for a in range(craftsim.NUM_ACTIONS) if a != 21]   # no smelting: fuel is not exposed
    img = vec = None
    for t in range(steps):
        if t % 10 == 0:
            for i in range(N):
                sim.set_pose(i, float(rng.uniform(-180, 180)), float(rng.uniform(-60, 60)))
        acts = rng.choice(pool, size=N).astype(np.int32)
        msgs = rng.integers(0, craftsim.NUM_MSGS, size=N).astype(np.int32)
        img, vec, _rew, _done, _ach = sim.step(acts, msgs)
    # Scripted phase so that a crafting table, a chest with items, and the crosshair chest all occur:
    # table placed ahead, chest placed 120 deg to the side, planks deposited into it, then the
    # crosshair is held on the chest. (pitch > 0 looks down in this convention.)
    act = {n: craftsim.ACTIONS.index(n) for n in ("place_table", "place_chest", "deposit")}
    base = [float(rng.uniform(-180, 180)) for _ in range(N)]
    for i in range(N):   # random steps may have spent the items: top them up
        sim.set_inventory(i, "crafting_table", 1)
        sim.set_inventory(i, "chest", 1)
        sim.set_inventory(i, "planks", 6)
    for i in range(N):
        sim.set_pose(i, base[i], 40.0)
    sim.step(np.full(N, act["place_table"], np.int32), np.zeros(N, np.int32))
    for i in range(N):
        sim.set_pose(i, base[i] + 120.0, 40.0)
    sim.step(np.full(N, act["place_chest"], np.int32), np.zeros(N, np.int32))
    sim.step(np.full(N, act["deposit"], np.int32), np.zeros(N, np.int32))
    # Final step: the obs and the state are read from this same step (no step after it).
    # Random poses for half the agents keep the sky/ground variety; the others keep the chest view.
    for i in range(N):
        if i % 2:
            sim.set_pose(i, base[i] + 120.0, 40.0)
        else:
            sim.set_pose(i, float(rng.uniform(-180, 180)), float(rng.uniform(-60, 60)))
    img, vec, _rew, _done, _ach = sim.step(np.zeros(N, np.int32), rng.integers(0, 8, size=N).astype(np.int32))

    ok_all = True
    summary = []
    for radius in (28, 16):
        worst_id = 100.0
        worst_dep = 100.0
        worst_vec = 0
        cov = []
        for i in range(N):
            d = dumpio.dump_from_sim(sim, i, radius, 3000, img, vec, craftsim.VEC_NAMES)
            path = os.path.join(workdir, f"selftest_r{radius}_a{i}.json")
            dumpio.write(path, d)
            rep, dl = compare_file(path, check_mapping=True)
            if verbose and i == 0:
                print_report(rep)
            worst_id = min(worst_id, rep["image"]["block_id_exact_pct"]
                           if rep["image"]["known_pixels"] else 100.0)
            worst_dep = min(worst_dep, rep["image"]["depth_exact_pct"] if rep["image"]["known_pixels"] else 100.0)
            worst_vec = max(worst_vec, rep["vec"]["counts"]["mismatch"])
            cov.append(rep["image"]["coverage_pct"])
            if rep["vec"]["counts"]["unknown_input"]:
                worst_vec = max(worst_vec, rep["vec"]["counts"]["unknown_input"] * 1000)
            if rep["mapping"]["wrong_by_vanilla_id"]:
                worst_vec = max(worst_vec, 1)
        passed = worst_id == 100.0 and worst_dep == 100.0 and worst_vec == 0
        ok_all &= passed
        summary.append((radius, passed, worst_id, worst_dep, worst_vec, min(cov)))

    # Negative controls on agent 0, radius 28: a change must be detected.
    d0 = dumpio.dump_from_sim(sim, 0, 28, 3000, img, vec, craftsim.VEC_NAMES)
    d0p = os.path.join(workdir, "selftest_r28_a0.json")
    dumpio.write(d0p, d0)
    base = dumpio.load(d0p)
    inv_case = dict(base)
    inv_case["inventory"] = dict(base["inventory"])
    inv_case["inventory"]["planks"] = inv_case["inventory"].get("planks", 0) + 1
    rep_inv = compare_dump(inv_case)
    inv_bad = sorted(r["name"] for r in rep_inv["vec"]["rows"] if r["status"] == "MISMATCH")
    neg_inv_ok = inv_bad == ["inv_planks"]

    blk = dumpio.load(d0p)
    scene_cell = simobs.center_cell(simobs.Scene(tuple(blk["blocks"]["origin"]),
                                                 dumpio.block_grid_from_ours(blk["_arr"]["ours"]),
                                                 (blk["blocks"]["build_limit"]["min_y"],
                                                  blk["blocks"]["build_limit"]["max_y"])), blk)
    neg_blk_ok = False
    if scene_cell is not None:
        x, y, z = scene_cell
        o = blk["blocks"]["origin"]
        yy, zz, xx = y - o[1], z - o[2], x - o[0]
        arr = blk["_arr"]["ours"]
        arr[yy, zz, xx] = 3 if arr[yy, zz, xx] != 3 else 2      # stone <-> dirt at the crosshair
        rep_blk = compare_dump(blk)
        neg_blk_ok = rep_blk["image"]["block_id_mismatch_pixels"] > 0
    print("selftest summary (sim dumps compared with the Python re-implementation):")
    for radius, passed, wid, wdep, wvec, mincov in summary:
        print(f"  radius {radius:2d}: {'PASS' if passed else 'FAIL'}  worst block-id {wid:.3f}%  "
              f"worst depth {wdep:.3f}%  vec mismatches/unknown {wvec}  min coverage {mincov:.1f}%")
    print(f"  negative control inventory change -> only inv_planks mismatches: {'ok' if neg_inv_ok else 'FAIL'} {inv_bad[:5]}")
    print(f"  negative control block change at crosshair -> image mismatches: {'ok' if neg_blk_ok else 'FAIL'}")
    ok_all &= neg_inv_ok and neg_blk_ok
    print("SELFTEST " + ("PASSED" if ok_all else "FAILED"))
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="?", help="dump JSON (minemind-parity-1)")
    ap.add_argument("--selftest", action="store_true", help="dump the live craftsim and check parity")
    ap.add_argument("--workdir", default=os.path.join("/tmp", "minemind-parity-selftest"),
                    help="where --selftest writes its dumps (default /tmp/minemind-parity-selftest)")
    ap.add_argument("--json", help="also write the full report as JSON here")
    ap.add_argument("--check-mapping", action="store_true", help="check ours vs vanilla ids (Obs.blockId rules)")
    ap.add_argument("--include-bodies", action="store_true", help="render non-team bodies as AGENT (sim has none)")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless image and vec agree exactly")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(args.workdir)
    if not args.dump:
        ap.error("give a dump file or --selftest")
    rep, _d = compare_file(args.dump, check_mapping=args.check_mapping, include_bodies=args.include_bodies)
    print_report(rep)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, default=float)
    if args.strict:
        good = (rep["image"]["block_id_mismatch_pixels"] == 0 and rep["vec"]["counts"]["mismatch"] == 0)
        return 0 if good else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
