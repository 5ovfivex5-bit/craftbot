"""Write a synthetic minemind-rec-1 recording (PROTOCOL 8.2) from the C++ sim with a scripted policy.

Each sim step (4 game ticks) is written as 4 frames that share the observation; the action label is
the scripted action on every tick, so the 8.4 aggregation must give the same action back. The
per-tick input.jsonl is synthesized from the action (keys, mouse deltas at 15/4 deg per tick).

  .venv/bin/python -m pretrain.fake_recording --out <scratch dir> --steps 2000 --goal log --seed 1
  .venv/bin/python -m pretrain.fake_recording --out <scratch dir> --steps 3000 --format v2 --seed 1

--format v2 writes a PROTOCOL 9.2 session (chunks/, index.jsonl, input.jsonl.gz, events.jsonl.gz,
meta.json) with synthetic events (pickup/craft/mine/kill/eat) derived from the sim's inventory changes
and the scripted actions. The events are fakes for exercising the loaders, not real game stats.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import craftsim  # noqa: E402
from pretrain import recordings as rec_lib  # noqa: E402

TICKS = rec_lib.TICKS_PER_STEP
TURN_TICK = 15.0 / TICKS     # per-tick mouse delta equal to one sim turn (TURN_DEG = 15 per decision)

# Scripted policy: hold a random action for a few steps, biased toward walking and mining.
WEIGHTS = {"forward": 6, "forward_jump": 1, "jump": 1, "turn_left": 2, "turn_right": 2, "look_up": 1,
           "look_down": 1, "attack": 4, "left": 1, "right": 1, "back": 1, "noop": 2,
           "place_block": 1, "craft_planks": 1, "craft_sticks": 1, "craft_table": 1}


def keys_for(name):
    k = {"forward": False, "back": False, "left": False, "right": False, "jump": False, "sneak": False,
         "sprint": False, "attack": False, "use": False, "drop": False, "inventory": False}
    k["forward"] = name in ("forward", "forward_jump")
    k["back"] = name == "back"
    k["left"] = name == "left"
    k["right"] = name == "right"
    k["jump"] = name in ("jump", "forward_jump")
    k["attack"] = name == "attack"
    k["use"] = name.startswith("place_")
    return {**k, "hotbar": {str(i): False for i in range(1, 10)}}


def mouse_for(name):
    dyaw = dpitch = 0.0
    if name == "turn_right":
        dyaw = TURN_TICK
    elif name == "turn_left":
        dyaw = -TURN_TICK
    elif name == "look_up":
        dpitch = -TURN_TICK
    elif name == "look_down":
        dpitch = TURN_TICK
    return dyaw, dpitch


V2_CHUNK_TICKS = 1200
# ext_action (9.4): key bits 0..12 in this order; camera bins 5 = none
KEY_BITS = ["forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use", "drop",
            "inventory", "swap_offhand", "pick"]
VANILLA_OF = {"log": "oak_log", "planks": "oak_planks", "iron_ore": "raw_iron"}   # MineMind -> vanilla
FOOD_CHOICE = ["cooked_beef", "bread", "apple", "cooked_porkchop"]
ENTITY_CHOICE = ["minecraft:zombie", "minecraft:skeleton", "minecraft:spider", "minecraft:cow"]


def ext_for(name, keys):
    ext = 0
    for i, k in enumerate(KEY_BITS):
        if keys.get(k):
            ext |= 1 << i
    yaw = 7 if name == "turn_right" else 3 if name == "turn_left" else 5
    pitch = 7 if name == "look_down" else 3 if name == "look_up" else 5
    return ext | (yaw << 17) | (pitch << 22)


def write_v2(args, sim, img, vec, names, goal_idx):
    import gzip
    import zlib

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rdir = os.path.join(os.path.abspath(args.out), f"{stamp}-session-fake")
    cdir = os.path.join(rdir, "chunks")
    os.makedirs(cdir, exist_ok=True)
    f_in = gzip.open(os.path.join(rdir, "input.jsonl.gz"), "wt")
    f_ev = gzip.open(os.path.join(rdir, "events.jsonl.gz"), "wt")
    index = open(os.path.join(cdir, "index.jsonl"), "w")
    rng = np.random.default_rng(args.seed + 1000)

    chunk_buf = []          # bytes of frames in the current chunk
    chunk_frames = 0
    chunk_no = 0
    first_tick = 0
    tick = 0

    def flush_chunk():
        nonlocal chunk_buf, chunk_frames, chunk_no, first_tick
        if not chunk_frames:
            return
        raw = b"".join(chunk_buf)
        comp = zlib.compress(raw, 6)
        name = f"{chunk_no:06d}.bin"
        tmp = os.path.join(cdir, name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(comp)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(cdir, name))
        index.write(json.dumps({"chunk": chunk_no, "first_tick": first_tick,
                                "frames": chunk_frames, "bytes": len(comp)}) + "\n")
        index.flush()
        first_tick += chunk_frames
        chunk_no += 1
        chunk_buf, chunk_frames = [], 0

    def event(typ, **kw):
        rec = {"tick": tick, "session_time_s": round(tick / 20.0, 3), "type": typ,
               "x": float(st["x"]), "y": float(st["y"]), "z": float(st["z"]), "count": 1}
        rec.update(kw)
        f_ev.write(json.dumps(rec) + "\n")

    pool = [names.index(n) for n in WEIGHTS]
    w = np.array([WEIGHTS[n] for n in WEIGHTS], float)
    w /= w.sum()
    contexts = []            # all context segments (9.6), for meta.json
    open_ctx = None          # the currently open segment

    def ctx_start(text):
        nonlocal open_ctx
        if open_ctx is not None:
            ctx_end("replaced")
        open_ctx = {"id": len(contexts) + 1, "text": text, "start_tick": tick, "end_tick": None, "reason": None}
        contexts.append(open_ctx)
        event("context_start", id=open_ctx["id"], text=text)

    def ctx_end(reason):
        nonlocal open_ctx
        if open_ctx is None:
            return
        open_ctx["end_tick"] = tick
        open_ctx["reason"] = reason
        event("context_end", id=open_ctx["id"], reason=reason)
        open_ctx = None

    action_hold, cur = 0, names.index("noop")
    t0 = time.time()
    st = sim.state(0)
    for step in range(args.steps):
        if action_hold <= 0:
            cur = int(rng.choice(pool, p=w))
            action_hold = int(rng.integers(2, 9))
        action_hold -= 1
        name = names[cur]
        # 9.6 context notes: scripted like a player typing /aibot context ...
        if step == 0:
            ctx_start("getting wood")
        elif step == 600:
            ctx_end("finish")
        elif step == 800:
            ctx_start("building a house")
        elif step == 2000:
            ctx_start("mining stone")          # replaces "building a house"
        elif step == 2600:
            ctx_end("finish")
        elif step == 2700:
            ctx_start("exploring")            # left open: ended by stop below
        prev_inv = {k: int(v) for k, v in sim.inventory(0).items() if v}
        st = sim.state(0)
        inv = sim.inventory(0)
        inv_named = {k: int(v) for k, v in inv.items() if v}
        dyaw, dpitch = mouse_for(name)
        obs_img = np.asarray(img[0], np.uint8)
        obs_vec = np.asarray(vec[0], np.float32)
        keys = keys_for(name)
        ext = ext_for(name, keys)
        for _ in range(TICKS):
            chunk_buf.append(obs_img.tobytes() + obs_vec.astype("<f4").tobytes()
                             + np.int32(cur).astype("<i4").tobytes() + np.int32(ext).astype("<i4").tobytes())
            chunk_frames += 1
            line = {"tick": tick, "keys": keys,
                    "dyaw": dyaw, "dpitch": dpitch, "yaw": float(st["yaw"]), "pitch": float(st["pitch"]),
                    "x": float(st["x"]), "y": float(st["y"]), "z": float(st["z"]),
                    "on_ground": bool(st["on_ground"]), "health": float(st["health"]), "selected_slot": 0,
                    "held_item": None, "gui": None, "inventory": inv_named,
                    "context": open_ctx["text"] if open_ctx else None}
            f_in.write(json.dumps(line) + "\n")
            tick += 1
            if chunk_frames >= V2_CHUNK_TICKS:
                flush_chunk()
        img, vec, _rew, _done, _ach = sim.step(np.array([cur], np.int32))
        # synthetic events: inventory delta across this step (after the action) plus scripted kills/eats
        after = {k: int(v) for k, v in sim.inventory(0).items() if v}
        delta = {k: after.get(k, 0) - prev_inv.get(k, 0) for k in set(after) | set(prev_inv)}
        gained = {k: v for k, v in delta.items() if v > 0}
        if name.startswith("craft_") and gained:
            for k, v in gained.items():
                event("craft", item=k, count=v, via="crafting_table" if name == "craft_table" else "inventory")
        elif name == "attack" and gained:
            for k, v in gained.items():
                event("mine", block="minecraft:stone", drops=[{"item": VANILLA_OF.get(k, k), "count": v}])
        elif gained:
            for k, v in gained.items():
                event("pickup", item=k, count=v)
        if step and step % 300 == 0:
            event("kill", entity=ENTITY_CHOICE[int(rng.integers(len(ENTITY_CHOICE)))],
                  weapon="wooden_sword" if prev_inv.get("stick") else "hand")
        if step and step % 500 == 0:
            event("eat", item=FOOD_CHOICE[int(rng.integers(len(FOOD_CHOICE)))], count=1)
    ctx_end("stop")                           # pausing or stopping ends an open context
    flush_chunk()
    for f in (f_in, f_ev, index):
        f.close()
    meta = {
        "format": "minemind-rec-2", "mc_version": "fake (craftsim)", "player": "scripted",
        "world_seed": int(args.seed), "goal": args.goal, "start_time": stamp, "session": True,
        "ticks": tick, "frames": tick, "frames_written": tick, "res": args.res, "vec_dim": craftsim.VEC_DIM,
        "vec_names": list(craftsim.VEC_NAMES), "actions": names, "items": list(craftsim.ITEMS),
        "blocks": list(craftsim.BLOCKS), "frame_every_ticks": 1, "chunk_ticks": V2_CHUNK_TICKS,
        "ext_action_bits": {"keys": KEY_BITS, "hotbar": "13..16", "yaw_bin": "17..21", "pitch_bin": "22..26"},
        "contexts": contexts, "synthetic": True, "finalized": True, "elapsed_s": round(time.time() - t0, 2),
    }
    with open(os.path.join(rdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {rdir} (v2): ticks={tick} chunks={chunk_no} sim_steps={args.steps}")
    return rdir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="parent directory; a <timestamp>-<goal> recording is created inside")
    ap.add_argument("--steps", type=int, default=2000, help="sim steps (4 frames each)")
    ap.add_argument("--goal", default="none")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--format", choices=["v1", "v2"], default="v1")
    args = ap.parse_args(argv)

    names = list(craftsim.ACTIONS)
    if args.goal != "none" and args.goal not in craftsim.ITEMS:
        raise SystemExit(f"unknown goal {args.goal}; items: {craftsim.ITEMS}")
    goal_idx = craftsim.ITEMS.index(args.goal) if args.goal != "none" else None

    rng = np.random.default_rng(args.seed)
    pool = [names.index(n) for n in WEIGHTS]
    w = np.array([WEIGHTS[n] for n in WEIGHTS], float)
    w /= w.sum()

    sim = craftsim.CraftSim(num_envs=1, agents=1, res=args.res, seed=args.seed)
    if goal_idx is not None:
        sim.set_goal(0, goal_idx)
    img, vec = sim.reset()
    if goal_idx is not None:
        vec = np.array(vec, np.float32)
        vec[0, craftsim.GOAL_OFFSET:craftsim.GOAL_OFFSET + craftsim.NUM_ITEMS] = 0
        vec[0, craftsim.GOAL_OFFSET + goal_idx] = 1.0
    if args.format == "v2":
        return write_v2(args, sim, img, vec, names, goal_idx)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rdir = os.path.join(os.path.abspath(args.out), f"{stamp}-{args.goal}-fake")
    os.makedirs(rdir, exist_ok=True)
    f_img = open(os.path.join(rdir, "obs_img.u8"), "wb")
    f_vec = open(os.path.join(rdir, "obs_vec.f32"), "wb")
    f_act = open(os.path.join(rdir, "action.i32"), "wb")
    f_in = open(os.path.join(rdir, "input.jsonl"), "w")
    f_ev = open(os.path.join(rdir, "events.jsonl"), "w")

    frames = 0
    action_hold, cur = 0, names.index("noop")
    t0 = time.time()
    for step in range(args.steps):
        if action_hold <= 0:
            cur = int(rng.choice(pool, p=w))
            action_hold = int(rng.integers(2, 9))
        action_hold -= 1
        name = names[cur]
        st = sim.state(0)
        inv = sim.inventory(0)
        inv_named = {k: int(v) for k, v in inv.items() if v}
        dyaw, dpitch = mouse_for(name)
        obs_img = np.asarray(img[0], np.uint8)
        obs_vec = np.asarray(vec[0], np.float32)
        for tick in range(TICKS):
            f_img.write(obs_img.tobytes())
            f_vec.write(obs_vec.astype("<f4").tobytes())
            f_act.write(np.int32(cur).astype("<i4").tobytes())
            line = {"tick": frames, "keys": keys_for(name), "dyaw": dyaw, "dpitch": dpitch,
                    "yaw": float(st["yaw"]), "pitch": float(st["pitch"]), "x": float(st["x"]),
                    "y": float(st["y"]), "z": float(st["z"]), "on_ground": bool(st["on_ground"]),
                    "health": float(st["health"]), "selected_slot": 0, "held_item": None, "gui": None,
                    "inventory": inv_named}
            f_in.write(json.dumps(line) + "\n")
            frames += 1
        img, vec, _rew, _done, _ach = sim.step(np.array([cur], np.int32))
    for f in (f_img, f_vec, f_act, f_in, f_ev):
        f.flush()
        os.fsync(f.fileno())
        f.close()

    meta = {
        "format": rec_lib.FORMAT, "mc_version": "fake (craftsim)", "player": "scripted",
        "world_seed": int(args.seed), "goal": args.goal, "start_time": stamp,
        "ticks": frames, "frames": frames, "res": args.res, "vec_dim": craftsim.VEC_DIM,
        "vec_names": list(craftsim.VEC_NAMES), "actions": names, "items": list(craftsim.ITEMS),
        "blocks": list(craftsim.BLOCKS), "frame_every_ticks": 1, "synthetic": True,
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(os.path.join(rdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {rdir}: frames={frames} sim_steps={args.steps}")
    return rdir


if __name__ == "__main__":
    main()
