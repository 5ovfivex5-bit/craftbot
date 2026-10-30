"""CPU (craftsim / src/sim.cpp) vs GPU (gpusim) parity. Run on a CUDA machine after building both.

    python gpusim/test_parity.py [--quick]

Both sims load the SAME snapshot blobs (save_state/load_state format), then apply identical action
sequences. Per step we compare rewards, done, achievements, inventory, positions/velocities (1e-4),
the full world (exact), and the observation (img exact-match rate >= 99.5%, vec within 1e-4).
A trajectory stops at the first done (auto-reset draws differ by design: CPU rng vs GPU pool).
"""
import os
import struct
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# drop this script's own directory: gpusim/gpusim.py would shadow the gpusim package
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, ROOT)
import craftsim as cs  # noqa: E402
from gpusim.gpusim import GpuCraftSim, parse_blob, WORLD, WY, WZ, WX  # noqa: E402

SNAP_MAGIC, SNAP_VERSION = 0x31565343, 1
A = {name: i for i, name in enumerate(cs.ACTIONS)}
SEA = cs.SEA_LEVEL
RES = 32
DEV = "cuda"
FAILS = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def pack_blob(d, rng_text):
    """Inverse of gpusim.parse_blob (for patching snapshots)."""
    b = bytearray()
    b += struct.pack("<IIQ", SNAP_MAGIC, SNAP_VERSION, WORLD)
    b += np.ascontiguousarray(d["world"], dtype=np.uint8).tobytes()
    b += struct.pack("<6d", d["px"], d["py"], d["pz"], d["vx"], d["vy"], d["vz"])
    b += struct.pack("<2f", d["yaw"], d["pitch"])
    b += struct.pack("<3B", int(d["onGround"]), int(d["kit"]), int(d["restart"]))
    b += struct.pack("<d", d["fallDist"])
    b += struct.pack("<i", d["health"])
    b += struct.pack("<14i", *d["inv"])
    b += struct.pack("<II", d["got"], d["kitGot"])
    b += struct.pack("<4i", d["fuel"], d["mx"], d["my"], d["mz"])
    b += struct.pack("<d", d["prog"])
    b += struct.pack("<4i", d["t"], d["ticks"], d["episode"], d["lastAction"])
    b += struct.pack("<Qi", d["seed"], d["goal"])
    b += struct.pack("<64Q", *d["visited"])
    b += struct.pack("<Ii", d["seenBlocks"], d["bestDepthStep"])
    b += struct.pack("<I", len(rng_text)) + rng_text
    return bytes(b)


class Pair:
    """A CPU sim and a GPU sim, both with one env, sharing a snapshot."""

    def __init__(self, novelty=0.0, ticks=4):
        self.cpu = cs.CraftSim(1, seed=1, res=RES, max_steps=3000, ticks=ticks, threads=1)
        self.gpu = GpuCraftSim(num_envs=1, seed=1, res=RES, max_steps=3000, ticks=ticks, device=DEV,
                               pool_size=256, pool_chunk=256, refresh_every=0)
        self.cpu.set_novelty(novelty)
        self.gpu.set_novelty(novelty)

    def load(self, blob, goal=None):
        ci, cv = self.cpu.load_state(0, blob, goal)
        gi, gv = self.gpu.load_state(0, blob, goal)
        return ci, cv, gi, gv


def make_worlds(n, seed):
    """n fresh CPU worlds (blob per env), as (blob, parsed dict) pairs."""
    sim = cs.CraftSim(n, seed=seed, res=4, max_steps=3000, ticks=4, threads=0)
    sim.reset()
    out = []
    for i in range(n):
        b = sim.save_state(i)
        out.append((b, parse_blob(b)))
    return out


def compare_run(pair, blob, actions, label, goal=None, verbose=False):
    """Replays actions on both sims from blob. Returns number of compared steps."""
    ci, cv, gi, gv = pair.load(blob, goal)
    rng_ok = True
    bad = {"rew": 0.0, "pos": 0.0, "vel": 0.0, "img": 1.0, "vec": 0.0}
    img_total = img_match = 0
    steps = 0
    for k, a in enumerate(actions):
        a = int(a)
        ci, cv, cr, cd, ca = pair.cpu.step(np.array([a], np.int32))
        gi, gv, gr, gd, ga = pair.gpu.step(torch.tensor([a], dtype=torch.int32, device=DEV))
        cr_, cd_, ca_ = float(cr[0]), bool(cd[0]), int(ca[0])
        gr_, gd_, ga_ = float(gr[0].cpu()), bool(gd[0].cpu()), int(ga[0].cpu())
        bad["rew"] = max(bad["rew"], abs(cr_ - gr_))
        if cr_ != 0 or gr_ != 0:
            if abs(cr_ - gr_) > 1e-4 and verbose:
                print(f"    {label} step {k} a={cs.ACTIONS[a]} rew cpu={cr_} gpu={gr_}")
        if cd_ != gd_ or ca_ != ga_:
            FAILS.append(f"{label} step {k}: done/ach mismatch cpu=({cd_},{ca_}) gpu=({gd_},{ga_})")
            print(f"  FAIL {label} step {k}: done/ach mismatch")
            return steps
        if cd_:   # both finished (goal or death): auto-reset worlds differ by design, stop here
            steps += 1
            break
        s_c = pair.cpu.state(0)
        s_g = pair.gpu.state(0)
        for key in ("x", "y", "z"):
            bad["pos"] = max(bad["pos"], abs(s_c[key] - s_g[key]))
        for key in ("vx", "vy", "vz"):
            bad["vel"] = max(bad["vel"], abs(s_c[key] - s_g[key]))
        for key in ("yaw", "pitch"):
            bad["pos"] = max(bad["pos"], abs(s_c[key] - s_g[key]))
        div = max(abs(s_c[key] - s_g[key]) for key in ("x", "y", "z", "yaw", "pitch", "vx", "vy", "vz"))
        if div > 1e-4:
            hist = [cs.ACTIONS[int(x)] for x in actions[max(0, k - 6):k + 1]]
            FAILS.append(f"{label} step {k}: diverged (max diff {div:.3g}) last actions {hist}")
            print(f"  FAIL {label} step {k}: diverged by {div:.3g} last actions {hist}")
            print(f"    cpu pos=({s_c['x']!r},{s_c['y']!r},{s_c['z']!r}) yaw={s_c['yaw']} vel=({s_c['vx']:.4g},{s_c['vy']:.4g},{s_c['vz']:.4g})")
            print(f"    gpu pos=({s_g['x']!r},{s_g['y']!r},{s_g['z']!r}) yaw={s_g['yaw']} vel=({s_g['vx']:.4g},{s_g['vy']:.4g},{s_g['vz']:.4g})")
            return steps
        if s_c["on_ground"] != s_g["on_ground"] or s_c["health"] != s_g["health"] or s_c["ticks"] != s_g["ticks"]:
            FAILS.append(f"{label} step {k}: on_ground/health/ticks mismatch")
            print(f"  FAIL {label} step {k}: state mismatch {s_c['on_ground']},{s_c['health']},{s_c['ticks']} vs {s_g['on_ground']},{s_g['health']},{s_g['ticks']}")
            return steps
        if (s_c["target"] is None) != (s_g["target"] is None) or (s_c["target"] and s_c["target"] != s_g["target"]):
            FAILS.append(f"{label} step {k}: target mismatch {s_c['target']} vs {s_g['target']}")
            print(f"  FAIL {label} step {k}: target mismatch")
            print(f"    yaw={s_c['yaw']!r} pitch={s_c['pitch']!r} pos=({s_c['x']!r},{s_c['y']!r},{s_c['z']!r})")
            print(f"    gpu yaw={s_g['yaw']!r} pitch={s_g['pitch']!r} pos=({s_g['x']!r},{s_g['y']!r},{s_g['z']!r})")
            return steps
        ci_np = ci.reshape(-1)
        gi_np = gi.reshape(-1).cpu().numpy()
        ci_np = ci_np
        img_total += ci_np.size // 2
        img_match += int(((ci_np.reshape(-1, 2) == gi_np.reshape(-1, 2)).all(-1)).sum())
        cvv = cv.reshape(-1)
        gvv = gv.reshape(-1).cpu().numpy()
        bad["vec"] = max(bad["vec"], float(np.abs(cvv - gvv).max()))
        inv_c = {k: v for k, v in pair.cpu.inventory(0).items() if v}     # CPU facade omits zeros
        inv_g = {k: v for k, v in pair.gpu.inventory(0).items() if v}
        if inv_c != inv_g:
            FAILS.append(f"{label} step {k}: inventory cpu={inv_c} gpu={inv_g}")
            print(f"  FAIL {label} step {k}: inventory mismatch")
            return steps
        # full world: blocks changed by mining / placing must match exactly
        org, arr = pair.cpu.region(0, 64)
        wg = pair.gpu.worlds[0].view(WY, WZ, WX).cpu().numpy()
        if not np.array_equal(arr, wg):
            diff = np.argwhere(arr != wg)
            FAILS.append(f"{label} step {k}: world differs at {len(diff)} blocks, first {diff[0].tolist()}")
            print(f"  FAIL {label} step {k}: world differs at {len(diff)} blocks")
            return steps
        steps += 1
    rate = img_match / max(1, img_total)
    ok = bad["rew"] <= 1e-4 and bad["pos"] <= 1e-4 and bad["vel"] <= 1e-4 and rate >= 0.995 and bad["vec"] <= 1e-4
    check(ok, f"{label}: {steps} steps | max|dRew|={bad['rew']:.2e} max|dPos|={bad['pos']:.2e} "
              f"max|dVel|={bad['vel']:.2e} img_match={rate*100:.3f}% max|dVec|={bad['vec']:.2e}")
    if not ok:
        FAILS.append(label)
    return steps


def random_actions(n, rng):
    """Random walk with attack bias (so mining and placement actually happen)."""
    p = np.ones(cs.NUM_ACTIONS)
    p[A["attack"]] = 6.0
    p[A["forward"]] = 3.0
    p[A["forward_jump"]] = 1.5
    p[A["place_block"]] = 0.5
    p /= p.sum()
    return rng.choice(cs.NUM_ACTIONS, size=n, p=p)


def scripted(*names, rep=1):
    out = []
    for nm in names:
        out += [A[nm]] * rep
    return out


def check_resets():
    """Goal sampling, start kits, episode counting on the GPU reset path (CPU rule: sim.cpp initEnv)."""
    N = 64
    g = GpuCraftSim(num_envs=N, seed=7, res=8, max_steps=40, ticks=4, device=DEV,
                    pool_size=256, pool_chunk=256, refresh_every=0)
    g.set_goal_weights([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    g.reset()
    goals = {g.goal(i) for i in range(N)}
    check(goals == {11}, f"goal weights respected: goals={sorted(goals)}")

    g.set_goal_weights([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])   # table or furnace
    table = np.zeros((cs.NUM_ITEMS, cs.NUM_ITEMS))
    table[cs.ITEMS.index("furnace"), cs.ITEMS.index("cobblestone")] = 8   # kit only for goal furnace
    table[cs.ITEMS.index("furnace"), cs.ITEMS.index("planks")] = 3
    table[cs.ITEMS.index("crafting_table"), cs.ITEMS.index("log")] = 2
    prob = np.ones(cs.NUM_ITEMS)
    g.set_start_kits(table, prob)
    g.reset()
    ok = True
    for i in range(N):
        goal = g.goal(i)
        inv = {k: v for k, v in g.inventory(i).items() if v}
        exp = {cs.ITEMS[k]: int(table[goal, k]) for k in range(cs.NUM_ITEMS) if table[goal, k] > 0 and k != goal}
        ok &= inv == exp and g.state(i)["kit"] == bool(exp)
    check(ok, "start kits applied from table[goal] (prob 1)")

    # auto-reset: episodes count up, t restarts, done fires at max_steps (40)
    g.set_start_kits(np.zeros((14, 14)), np.zeros(14))
    g.reset()
    rng = np.random.default_rng(3)
    dones = 0
    for _ in range(90):
        _i, _v, _r, d, _a = g.step(torch.tensor(rng.integers(0, cs.NUM_ACTIONS, N), device=DEV))
        dones += int(d.sum())
    st = [g.state(i) for i in range(4)]
    check(dones >= N, f"max_steps=40 terminates episodes: done events={dones} over 90 steps x {N} envs")
    check(all(s["t"] <= 40 for s in st) and max(s["episode"] for s in st) >= 1, f"episode counter/t after auto-reset: {[(s['t'], s['episode']) for s in st]}")
    g.close()


def main():
    quick = "--quick" in sys.argv
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(2)
    pair = Pair(novelty=0.0)
    rng = np.random.default_rng(0)
    worlds = make_worlds(3 if quick else 6, seed=123)
    print(f"worlds: {len(worlds)}")

    # --- random play, novelty off, then novelty on (novelty path is part of rew)
    for wi, (blob, d) in enumerate(worlds):
        compare_run(pair, blob, random_actions(400 if quick else 1200, rng), f"random_w{wi}")
    pair.cpu.set_novelty(0.5)
    pair.gpu.set_novelty(0.5)
    blob, d = worlds[0]
    compare_run(pair, blob, random_actions(600, rng), "random_novelty")
    pair.cpu.set_novelty(0.0)
    pair.gpu.set_novelty(0.0)

    # --- walk + jump on the spawn (grass) column
    blob, d = worlds[1]
    walk = scripted("noop", "noop", "forward", rep=40) + scripted("turn_left", "turn_left", "turn_left", "forward", rep=20) \
        + scripted("jump", "noop", "noop", "noop") + scripted("forward_jump", rep=3) + scripted("noop", rep=20)
    compare_run(pair, blob, walk, "walk_jump")

    # --- jump apex (vanilla 1.2522): needs per-tick sampling, so a ticks=1 pair on the spawn column
    tp = Pair(novelty=0.0, ticks=1)
    tp.load(blob)
    for _ in range(10):
        tp.cpu.step(np.array([A["noop"]], np.int32))
        tp.gpu.step(torch.tensor([A["noop"]], dtype=torch.int32, device=DEV))
    y0 = tp.cpu.state(0)["y"]
    apex_c = apex_g = 0.0
    tp.cpu.step(np.array([A["jump"]], np.int32))
    tp.gpu.step(torch.tensor([A["jump"]], dtype=torch.int32, device=DEV))
    for _ in range(12):
        tp.cpu.step(np.array([A["noop"]], np.int32))
        tp.gpu.step(torch.tensor([A["noop"]], dtype=torch.int32, device=DEV))
        apex_c = max(apex_c, tp.cpu.state(0)["y"] - y0)
        apex_g = max(apex_g, tp.gpu.state(0)["y"] - y0)
    tp.gpu.close()
    check(abs(apex_c - 1.2522) < 1e-3 and abs(apex_g - apex_c) < 1e-4,
          f"jump apex cpu={apex_c:.4f} gpu={apex_g:.4f} (expect 1.2522)")

    # --- mining timings: hand on grass 18 ticks, hand on stone 150 ticks (patched world)
    def hand_mine_ticks(block_id, label):
        dd = dict(d)
        w = d["world"].copy()
        # column under the player: make the block below the feet the target, looking straight down
        cx, cz = int(np.floor(d["px"])), int(np.floor(d["pz"]))
        h0 = int(np.floor(d["py"])) - 1
        for y in range(h0 - 1, h0 + 1):
            w[(y * WZ + cz) * WX + cx] = block_id
        dd["world"] = w
        dd["pitch"] = 90.0
        dd["inv"] = [0] * 14
        blob2 = pack_blob(dd, d["rng_text"])
        pair.load(blob2)
        for _ in range(5):
            pair.cpu.step(np.array([A["noop"]], np.int32))
            pair.gpu.step(torch.tensor([A["noop"]], dtype=torch.int32, device=DEV))
        t0c = pair.cpu.state(0)["ticks"]; t0g = pair.gpu.state(0)["ticks"]
        broke_c = broke_g = None
        for _ in range(200):
            pair.cpu.step(np.array([A["attack"]], np.int32))
            pair.gpu.step(torch.tensor([A["attack"]], dtype=torch.int32, device=DEV))
            if broke_c is None and pair.cpu.block(0, cx, h0, cz) == "air":
                broke_c = pair.cpu.state(0)["ticks"] - t0c
            if broke_g is None and pair.gpu.block(0, cx, h0, cz) == "air":
                broke_g = pair.gpu.state(0)["ticks"] - t0g
            if broke_c is not None and broke_g is not None:
                break
        return broke_c, broke_g

    gc, gg = hand_mine_ticks(cs.BLOCKS.index("grass"), "grass")
    check(gc == gg and gc is not None and abs(gc - 18) <= 1, f"hand grass break ticks cpu={gc} gpu={gg} (expect ~18)")
    sc, sg = hand_mine_ticks(cs.BLOCKS.index("stone"), "stone")
    check(sc == sg and sc is not None and abs(sc - 150) <= 2, f"hand stone break ticks cpu={sc} gpu={sg} (expect ~150)")

    # --- craft chain: log -> planks -> sticks -> table -> place -> wooden pickaxe
    dd = dict(d)
    dd["inv"] = [0] * 14
    dd["inv"][cs.ITEMS.index("log")] = 5
    dd["inv"][cs.ITEMS.index("cobblestone")] = 8
    compare_run(pair, pack_blob(dd, d["rng_text"]),
                scripted("craft_planks", "craft_planks", "craft_sticks", "craft_sticks", "craft_table",
                         "look_down", "look_down", "look_down", "place_table", "craft_wooden_pickaxe",
                         "craft_furnace", "craft_planks", "craft_stone_pickaxe", "noop", "noop"),
                "craft_chain", goal=cs.ITEMS.index("iron_pickaxe"))

    # --- swim: put the player inside a water column, random moves and jumps
    water_cols = np.argwhere(d["world"].reshape(WY, WZ, WX)[SEA - 1] == cs.BLOCKS.index("water"))
    if len(water_cols):
        wz, wx = water_cols[len(water_cols) // 2]
        dd = dict(d)
        dd["px"], dd["py"], dd["pz"] = wx + 0.5, SEA - 1.0, wz + 0.5
        dd["vx"] = dd["vy"] = dd["vz"] = 0.0
        dd["onGround"] = 0
        dd["fallDist"] = 0.0
        blob_w = pack_blob(dd, d["rng_text"])
        swim = list(rng.choice([A["forward"], A["left"], A["right"], A["jump"], A["back"], A["noop"]], 300))
        compare_run(pair, blob_w, swim, "swim")
    else:
        print("  (no water column in this world; swim test skipped)")

    # --- dig to deepslate: patch the column under the player to stone/deepslate, dig down with attack
    dd = dict(d)
    cx, cz = int(np.floor(d["px"])), int(np.floor(d["pz"]))
    w = d["world"].copy()
    w_v = w.reshape(WY, WZ, WX)
    w_v[:, cz, cx] = cs.BLOCKS.index("stone")
    w_v[:6, cz, cx] = cs.BLOCKS.index("deepslate")
    dd["world"] = w
    dd["pitch"] = 90.0
    dd["inv"] = [0] * 14
    dd["inv"][cs.ITEMS.index("wooden_pickaxe")] = 1
    dig = scripted("look_down", "look_down", "look_down", "look_down", "look_down", "look_down", rep=1)
    dig += [A["attack"]] * 300
    compare_run(pair, pack_blob(dd, d["rng_text"]), dig, "dig_deepslate")

    # --- goal item path: random play with goal = each of a few items
    for goal in (cs.ITEMS.index("planks"), cs.ITEMS.index("iron_pickaxe")):
        compare_run(pair, worlds[2][0], random_actions(400, rng), f"goal_{cs.ITEMS[goal]}", goal=goal)

    # --- reset path (not compared step-by-step): goal weights, start kits, auto-reset counters
    check_resets()

    pair.gpu.close()
    print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    for f in FAILS:
        print("  -", f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
