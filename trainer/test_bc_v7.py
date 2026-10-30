"""Behaviour-cloning tests (PROTOCOL 14.3): synthetic demos, BC training, trainer resume and eval.

  .venv/bin/python -m trainer.test_bc_v7 [--out DIR]     (pytest is not installed)

The demos are real v7 CraftSim episodes driven by a small scripted policy (craft a recipe that needs no station
when the inventory allows it, mine a log / stone / gravel / sand / leaves block in the crosshair, greet a teammate in view, otherwise walk with a periodic turn, drop an item
when the inventory holds two or more kinds), written in the PROTOCOL 14.2 layout. Everything is a deterministic function of the
observation (plus a per-agent step counter that the GRU can track), so the heads are learnable.

Checks: the shard format round-trips, the loss falls, every head beats chance and the majority baseline on held-out
episodes, the checkpoint resumes in trainer.run for 4k agent steps (population 4, headless, no server) and loads in
trainer.eval. Outputs go under the scratchpad directory only (never runs/, never port 7777).
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = "/tmp/claude-1000/-home-deck-claude/fd75689d-b7cd-41a3-8f2b-c8d7da974f14/scratchpad"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import craftsim  # noqa: E402
from trainer import bc_v7 as bc  # noqa: E402
from trainer import envs as E  # noqa: E402

PRIM = {n: i for i, n in enumerate(craftsim.PRIMARY_NAMES)}
AGENT = craftsim.BLOCKS.index("agent")
MINE = {craftsim.BLOCKS.index(b) for b in ("oak_log", "stone", "gravel", "sand", "oak_leaves")}
NEED0 = []   # recipes that need no station: (recipe index, {item id: count})
for _r, _rec in enumerate(craftsim.RECIPES):
    if _rec["needs"] == 0:
        _ing = {(E.ITEMS.index(k) if isinstance(k, str) else int(k)): int(n) for k, n in _rec["ingredients"].items()}
        NEED0.append((_r, _ing))


def _counts(vec):
    return np.round(np.exp2(7.0 * vec[:E.NUM_ITEMS].astype(np.float64)) - 1.0)


def scripted_actions(img, vec, counter):
    """One decision per row (N rows): returns int64 [N, 4] = (primary, recipe, item_select, msg)."""
    N = vec.shape[0]
    act = np.zeros((N, 4), np.int64)
    for i in range(N):
        cnt = _counts(vec[i])
        recipe = None
        for r, ing in NEED0:
            if all(cnt[it] >= n for it, n in ing.items()):
                recipe = r
                break
        if recipe is not None:
            act[i] = (PRIM["craft"], recipe + 1, 0, 1)          # craft, and say so on the chat channel
            continue
        held = np.flatnonzero(cnt > 0)
        if held.size >= 2:
            act[i] = (PRIM["drop"], 0, int(held[0]) + 1, 0)     # drop the lowest-id stack
            continue
        center = int(img[i, 32, 32, 0])
        if center == AGENT:
            act[i] = (PRIM["noop"], 0, 0, 2)                    # a teammate in view: say hello
            continue
        if center in MINE:
            act[i] = (PRIM["attack"], 0, 0, 0)                  # mine the block in the crosshair
            continue
        counter[i] += 1
        act[i] = (PRIM["turn_right"] if counter[i] % 12 == 11 else PRIM["forward"], 0, 0, 0)
    return act


def generate_demos(out_dir, worlds=8, agents=4, seed=7, decisions=1000, max_steps=60, rows_per_shard=4000):
    """Writes out_dir/shard_*.npz (PROTOCOL 14.2) and meta.json; returns the meta dict."""
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "shard_*.npz")):
        os.remove(old)
    sim = craftsim.CraftSim(num_envs=worlds, seed=seed, res=E.RES, max_steps=max_steps, agents=agents,
                            action_mode="v7")
    img, vec = sim.reset()
    N = worlds * agents
    counter = np.zeros(N, np.int64)
    ep = np.zeros(N, np.int64)
    buf = [[] for _ in range(N)]
    pending = []                  # finished rows, flushed as shards
    shard_id, n_rows, n_eps, n_succ = 0, 0, 0, 0

    def flush(final=False):
        nonlocal pending, shard_id
        if not pending or (len(pending) < rows_per_shard and not final):
            return
        cols = list(zip(*pending))
        path = os.path.join(out_dir, f"shard_{shard_id:05d}.npz")
        np.savez(path, img=np.asarray(cols[0], np.uint8), vec=np.asarray(cols[1], np.float32),
                 act=np.asarray(cols[2], np.int16), done=np.asarray(cols[3], bool),
                 task=np.asarray(cols[4], np.int16), world=np.asarray(cols[5], np.int32),
                 agent=np.asarray(cols[6], np.int16), ep=np.asarray(cols[7], np.int32),
                 success=np.asarray(cols[8], bool), reward=np.asarray(cols[9], np.float32))
        shard_id += 1
        pending = []

    for _t in range(decisions):
        a = scripted_actions(img, vec, counter)
        tasks = np.array([sim.task(i) for i in range(N)], np.int64)
        for i in range(N):
            buf[i].append([img[i].copy(), vec[i].copy(), a[i], False, tasks[i], i // agents, i % agents, ep[i], False,
                            0.0])
        img, vec, rew, done, ach = sim.step(a.astype(np.int32))
        for i in range(N):
            buf[i][-1][9] = float(rew[i])
            if done[i]:
                success = bool((int(ach[i]) >> 31) & 1)
                rows = buf[i]
                rows[-1][3] = True
                for r in rows:
                    r[8] = success
                pending.extend(rows)
                n_rows += len(rows)
                n_eps += 1
                n_succ += int(success)
                buf[i] = []
                ep[i] += 1
                counter[i] = 0
        flush()
    flush(final=True)
    meta = {"format": "PROTOCOL 14.2", "generator": "trainer.test_bc_v7.generate_demos (scripted, synthetic)",
            "seed": seed, "worlds": worlds, "agents": agents, "max_steps": max_steps, "episodes": n_eps,
            "rows": n_rows, "success_rate": n_succ / max(1, n_eps), "shards": shard_id,
            "vec_dim": E.VEC_DIM, "vec_names_hash": hashlib.sha256(",".join(E.VEC_NAMES).encode()).hexdigest()[:16],
            "num_actions": E.NUM_ACTIONS, "num_recipe_classes": E.NUM_RECIPE_CLASSES,
            "num_item_classes": E.NUM_ITEM_CLASSES, "num_msgs": E.NUM_MSGS}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg, flush=True)


def _run(cmd, timeout):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                       env=dict(os.environ, PYTHONPATH=ROOT))
    return p, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SCRATCH, "bc-test"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--resume-steps", type=int, default=4000)
    ap.add_argument("--skip-train", action="store_true", help="reuse the bc run already under --out")
    args = ap.parse_args(argv)
    demos = os.path.join(args.out, "demos")
    runs = os.path.join(args.out, "runs")
    os.makedirs(runs, exist_ok=True)

    print("== synthetic demos", flush=True)
    t0 = time.time()
    meta = generate_demos(demos)
    print(f"  {meta['episodes']} episodes, {meta['rows']:,} rows, {meta['shards']} shards, success "
          f"{meta['success_rate']:.2f} ({time.time() - t0:.1f}s)", flush=True)
    _check(meta["episodes"] >= 40 and meta["rows"] >= 2000, "enough episodes and rows")
    shards = [bc.Shard(p) for p in sorted(glob.glob(os.path.join(demos, "shard_*.npz")))]
    segs = bc.build_segments(shards)
    _check(len(segs) == meta["episodes"], f"segments == episodes ({len(segs)})")
    _check(all(s.n >= 1 and shards[s.shard].done[s.start + s.n - 1] for s in segs),
           "every episode is contiguous and ends on a done row")

    print("== behaviour cloning", flush=True)
    out = os.path.join(args.out, "bc")
    cmd = [sys.executable, "-m", "trainer.bc_v7", "--demos", demos, "--out", out, "--steps", str(args.steps),
           "--batch", "16", "--chunk", "64", "--lr", "1e-3", "--balance", "--val-frac", "0.2", "--seed", "3",
           "--run-id", "bc-test", "--log-every", "100"]
    if not (args.skip_train and os.path.exists(os.path.join(out, "bc-test", "bc_report.json"))):
        p, secs = _run(cmd, timeout=3600)
        print(p.stdout, flush=True)
        if p.returncode != 0:
            print(p.stderr, flush=True)
            raise SystemExit("bc_v7 failed")
    else:
        secs = 0.0
    rep = json.load(open(os.path.join(out, "bc-test", "bc_report.json")))
    v = rep["val"]
    print(f"  wall {secs:.1f}s; loss {rep['loss_start']:.4f} -> {rep['loss_end']:.4f}; "
          f"{rep['train_rows_per_s']:,.0f} rows/s", flush=True)
    _check(rep["loss_end"] < 0.7 * rep["loss_start"], "loss decreases (>30%)")
    base = rep["primary_majority_baseline"]
    _check(v["acc_primary"] > max(1.0 / E.NUM_ACTIONS, base) + 0.05,
           f"primary {v['acc_primary']:.3f} above chance {1 / E.NUM_ACTIONS:.3f} and majority {base:.3f}")
    _check(v["acc_recipe"] is not None and v["acc_recipe"] > 0.5, f"recipe {v['acc_recipe']:.3f} above chance")
    _check(v["acc_item"] is not None and v["acc_item"] > 0.2, f"item {v['acc_item']:.3f} above chance")
    _check(v["acc_msg"] > 0.5, f"msg {v['acc_msg']:.3f} above chance (1/8)")
    _check(v["sequence_agreement"] > 0.5, f"action-sequence agreement {v['sequence_agreement']:.3f}")

    print("== resume in trainer.run (population 4, headless, 4k agent steps)", flush=True)
    ck = glob.glob(os.path.join(out, "bc-test", "checkpoints", "ckpt_*.pt"))
    _check(len(ck) == 1, "one bc checkpoint written")
    rr = os.path.join(runs, "resume")
    # the bc checkpoint's task scheduler covers the trainer's default 64 worlds, so resume with the default --envs
    cmd = [sys.executable, "-m", "trainer.run", "--resume", ck[0], "--no-server", "--population", "4",
           "--agents", "4", "--total-steps", str(args.resume_steps), "--runs-dir", rr, "--mode", "fast", "--threads", "4",
           "--workers", "2", "--seed", "11", "--no-visualize"]
    p, secs = _run(cmd, timeout=1800)
    tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
    print("\n".join("  | " + t for t in tail), flush=True)
    _check(p.returncode == 0, f"trainer.run resume exited 0 ({secs:.0f}s)")
    new = sorted(glob.glob(os.path.join(rr, "*", "checkpoints", "ckpt_*.pt")))
    _check(bool(new), "resumed run wrote a checkpoint")
    import torch
    pl = torch.load(new[-1], map_location="cpu", weights_only=False)
    steps = pl["counters"]["env_steps"]
    _check(steps >= args.resume_steps, f"resumed run reached env_steps={steps} (>= {args.resume_steps})")
    _check(pl.get("population") is not None and len(pl.get("models", [])) == 4, "population checkpoint with 4 members")

    print("== trainer.eval on the bc checkpoint", flush=True)
    cmd = [sys.executable, "-m", "trainer.eval", ck[0], "--task", "gather_wood", "--episodes", "16",
           "--envs", "8", "--greedy", "--threads", "2", "--seed", "5"]
    p, secs = _run(cmd, timeout=1800)
    print("\n".join("  | " + t for t in (p.stdout + p.stderr).strip().splitlines()[-5:]), flush=True)
    _check(p.returncode == 0, f"trainer.eval exited 0 ({secs:.0f}s)")

    print("== done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
