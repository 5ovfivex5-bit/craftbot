"""Behaviour cloning of the v7 MineMind policy from expert demonstration shards (PROTOCOL 14.3).

  python -m trainer.bc_v7 --demos DIR [DIR...] --out DIR --steps N [--batch 16] [--chunk 64] [--lr 3e-4]
                          [--success-only] [--warm-start CKPT] [--val-frac 0.1] [--balance]
                          [--tasks a,b,...] [--seed S] [--run-id ID]

Shard format (PROTOCOL 14.2, written by experts/record.py): DIR/shard_*.npz with
  img uint8 [T,64,64,2], vec float32 [T,295], act int16 [T,4] (primary, recipe, item_select, msg),
  done bool [T], task int16 [T], world int32 [T], agent int16 [T], ep int32 [T], success bool [T], reward f32 [T].
A "segment" is one episode of one (world, agent): consecutive rows with equal (world, agent, ep) that ends
at a done row. Training: lockstep truncated BPTT. B slots each walk one episode in chunks of --chunk rows and
carry the GRU state from chunk to chunk; a slot that finishes its episode takes a fresh one with a zeroed
state (the episode start). Every optimizer step consumes one chunk per slot.

Losses (per head, each a mean over its own active rows):
  primary  CE on every row (class weights with --balance: inverse sqrt frequency, max/min ratio <= 10);
  recipe   CE only on rows whose primary is craft / smelt (heads.PRIM_CRAFT) and whose target recipe is in the
           inventory mask of heads.masks; other craft/smelt rows are skipped and counted;
  item     the same for primary use / drop / give and item_select;
  msg      CE on every row, weight 0.1.

Output: a normal v7 trainer checkpoint at OUT/<run_id>/checkpoints/ckpt_000000000000_u000000.pt (counters zero,
config = trainer.run.DEFAULTS plus the run's seed / lr / obs, an empty-history task scheduler and the fresh
normalizers), loadable by `python -m trainer.run --resume` (population included) and `trainer.eval`.
A report with per-head and per-task accuracy and the action-sequence agreement goes to OUT/<run_id>/bc_report.json.
"""
import argparse
import collections
import difflib
import glob
import json
import os
import sys
import time

import numpy as np
import torch

import craftsim
from . import checkpoint as ckpt
from . import envs as E
from .demos import (NUM_ACTIONS, NUM_ITEM_CLASSES, NUM_MSGS, NUM_RECIPE_CLASSES, RES, VEC_DIM,  # noqa: F401
                    SHARD_KEYS, Seg, Shard, ShardCache, _check_obs, _hash_unit, _npz_member, _npz_member_shape,
                    _shard_paths, build_model, build_segments, chunk_forward, head_losses, make_chunk,
                    split_segments)

MAX_CLASS_RATIO = 10.0
GRAD_CLIP = 1.0
TASK_NAMES = list(craftsim.TASK_NAMES)
PRIM_NAMES = list(craftsim.PRIMARY_NAMES)


class Batcher:
    """Lockstep truncated-BPTT sampler over the training episodes (see the module docstring)."""

    def __init__(self, segs, B, L, rng):
        self.segs, self.B, self.L, self.rng = segs, B, L, rng
        self.order = []
        self.slots = [None] * B          # [seg index, rows already consumed]
        self.epochs = 0

    def _pull(self):
        if not self.order:
            self.order = list(self.rng.permutation(len(self.segs)))
            self.epochs += 1
        return self.order.pop()

    def next(self, cache):
        new = np.zeros(self.B, bool)
        items = []
        for i in range(self.B):
            s = self.slots[i]
            if s is None or s[1] >= self.segs[s[0]].n:
                s = [self._pull(), 0]
                self.slots[i] = s
                new[i] = True
            seg = self.segs[s[0]]
            items.append((seg, s[1]))
            s[1] += self.L
        batch = make_chunk(cache, items, self.L)
        batch["new"] = new
        return batch


def primary_class_weights(segs, shards, balance):
    """(float32 [NUM_ACTIONS] class weights, primary counts over the training rows). Weights are 1 without balance.
    With balance: inverse-sqrt frequency, normalised to mean 1 under the data and capped so that max / min (over
    the present classes) <= MAX_CLASS_RATIO."""
    counts = np.zeros(NUM_ACTIONS, np.int64)
    for s in segs:
        counts += np.bincount(shards[s.shard].act[s.start:s.start + s.n, 0], minlength=NUM_ACTIONS)
    cw = np.ones(NUM_ACTIONS, np.float64)
    if not balance:
        return torch.tensor(cw, dtype=torch.float32), counts
    present = counts > 0
    f = counts / max(1, counts.sum())
    w = np.zeros(NUM_ACTIONS)
    w[present] = 1.0 / np.sqrt(f[present])
    w[present] = np.minimum(w[present], MAX_CLASS_RATIO * w[present].min())
    w[present] /= float((f * w).sum())
    cw[present] = w[present]
    return torch.tensor(cw, dtype=torch.float32), counts


# ----------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, segs, shards, cache, L, B, cw):
    """Greedy full-episode inference (GRU carried across chunks) over the given segments. Head accuracies use the
    teacher-forced masks of heads.py (recipe / item scored only where active and in-mask)."""
    model.eval()
    tot = collections.Counter()
    per_task = collections.defaultdict(lambda: [0, 0])     # task id -> [correct, rows]
    seq = []                                               # (task, target collapsed, predicted collapsed)
    G = model.gru_hidden
    order = sorted(range(len(segs)), key=lambda i: segs[i].n)
    for g0 in range(0, len(order), B):
        group = [segs[i] for i in order[g0:g0 + B]]
        preds = [[] for _ in group]
        h = torch.zeros(B, G)
        maxn = max(s.n for s in group)
        for j in range(0, maxn, L):
            items = [(group[i], j) if i < len(group) else (None, 0) for i in range(B)]
            ch = make_chunk(cache, items, L)
            img = torch.from_numpy(ch["img"])
            vec = torch.from_numpy(ch["vec"])
            act = torch.from_numpy(ch["act"])
            valid = torch.from_numpy(ch["valid"])
            o, h = chunk_forward(model, img, vec, h)
            _, st, rows = head_losses(model, o, vec, act, valid, cw)
            for k in ("n", "n_rec", "n_rec_skip", "n_item", "n_item_skip", "ok_prim", "ok_rec", "ok_item", "ok_msg"):
                tot[k] += st[k]
            for k in ("loss_p", "loss_r", "loss_i", "loss_m"):
                tot[k + "_sum"] += st[k]
            tot["chunks"] += 1
            for i, seg in enumerate(group):
                m = ch["valid"][i]
                if m.any():
                    preds[i].append(rows["pred_prim"][i][m])
                    tk = int(seg.task)
                    per_task[tk][0] += int(rows["ok_prim"][i][m].sum())
                    per_task[tk][1] += int(m.sum())
        for i, seg in enumerate(group):
            tgt = shards[seg.shard].act[seg.start:seg.start + seg.n, 0]
            pred = np.concatenate(preds[i]) if preds[i] else np.zeros(0, np.int64)
            seq.append((seg.task, _collapse(tgt), _collapse(pred)))
    model.train()
    return tot, per_task, seq


def _collapse(x):
    x = np.asarray(x, np.int64)
    if x.size == 0:
        return []
    keep = np.ones(x.size, bool)
    keep[1:] = x[1:] != x[:-1]
    return x[keep].tolist()


def sequence_agreement(seq):
    """Mean over episodes of the SequenceMatcher ratio of the run-collapsed primary sequences (1.0 = identical)."""
    rs = []
    for _task, tgt, pred in seq:
        if not tgt and not pred:
            rs.append(1.0)
            continue
        rs.append(difflib.SequenceMatcher(None, tgt, pred, autojunk=False).ratio())
    return float(np.mean(rs)) if rs else None


def _rate(c, n):
    return (c / n) if n else None


# ----------------------------------------------------------------------------- CLI / main

def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m trainer.bc_v7", description="behaviour cloning of the v7 policy")
    p.add_argument("--demos", nargs="+", required=True, help="shard directories (shard_*.npz, PROTOCOL 14.2)")
    p.add_argument("--out", required=True, help="output directory (OUT/<run_id>/checkpoints/...)")
    p.add_argument("--steps", type=int, required=True, help="optimizer steps")
    p.add_argument("--batch", type=int, default=16, help="episodes per step (chunk columns)")
    p.add_argument("--chunk", type=int, default=64, help="rows per BPTT chunk")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--success-only", action="store_true", help="train on successful episodes only")
    p.add_argument("--warm-start", metavar="CKPT", default=None, help="v7 checkpoint to start from")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of episodes held out (by hash)")
    p.add_argument("--balance", action="store_true", help="inverse-sqrt class weights on the primary head")
    p.add_argument("--tasks", default=None, help="comma-separated task names to keep (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--cache-shards", type=int, default=4, help="shards kept open (LRU)")
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args(argv)
    if args.steps < 1 or args.batch < 1 or args.chunk < 1:
        p.error("--steps, --batch and --chunk must be >= 1")
    if not (0.0 <= args.val_frac < 1.0):
        p.error("--val-frac must be in [0, 1)")
    return args


def main(argv=None):
    _check_obs()
    args = parse_args(argv)
    from .run import DEFAULTS
    from .ppo import ReturnNormalizer, RunningStd
    from .tasks import WorldTasks

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    task_ids = None
    if args.tasks:
        names = [t.strip() for t in args.tasks.split(",") if t.strip()]
        bad = [t for t in names if t not in TASK_NAMES]
        if bad:
            raise SystemExit(f"unknown task names: {bad}")
        task_ids = {TASK_NAMES.index(t) for t in names}

    paths = _shard_paths(args.demos)
    t0 = time.time()
    shards = [Shard(p) for p in paths]
    segs = build_segments(shards, success_only=args.success_only, task_ids=task_ids)
    if not segs:
        raise SystemExit("no episodes left after the filters (--success-only / --tasks)")
    train, val = split_segments(segs, shards, args.val_frac, args.seed)
    if not train:
        raise SystemExit("no training episodes (lower --val-frac)")
    cache = ShardCache(shards, args.cache_shards)
    cw, prim_counts = primary_class_weights(train, shards, args.balance)
    n_rows = sum(s.T for s in shards)
    print(f"demos: {len(paths)} shards, {n_rows:,} rows, {len(segs):,} episodes after filters "
          f"(train {len(train):,} / val {len(val):,}); load {time.time() - t0:.1f}s", flush=True)
    tr_rows = sum(s.n for s in train)
    succ_rate = sum(s.success for s in segs) / len(segs)
    print(f"  success rate {succ_rate:.3f}; train rows {tr_rows:,}; balance={'on' if args.balance else 'off'}", flush=True)

    # model: fresh, or the warm-start checkpoint's policy
    size, resblock = "small", bool(DEFAULTS["resblock"])
    warm = None
    if args.warm_start:
        warm = ckpt.load_for_training(args.warm_start)
        wc = warm.get("config", {})
        size, resblock = wc.get("model_size", "small"), bool(wc.get("resblock", resblock))
    model = build_model(size, resblock)
    if warm is not None:
        res = model.load_state_dict(warm["model"], strict=False)
        if res.unexpected_keys:
            raise SystemExit(f"warm start: unexpected parameters {res.unexpected_keys[:5]}")
        print(f"warm start from {args.warm_start} (missing: {sorted({k.split('.')[0] for k in res.missing_keys})})",
              flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    model.train()

    B, L = args.batch, args.chunk
    batcher = Batcher(train, B, L, rng)
    h = torch.zeros(B, model.gru_hidden)
    run_id = args.run_id or ("bc-v7-" + time.strftime("%Y%m%d-%H%M%S") + "-" + hex(args.seed)[2:])
    run_dir = os.path.join(args.out, run_id)
    os.makedirs(run_dir, exist_ok=True)

    hist = collections.deque(maxlen=max(1, args.log_every))
    rows_seen = 0
    t_train = time.time()
    first_losses = []
    for step in range(1, args.steps + 1):
        b = batcher.next(cache)
        h = h * torch.from_numpy(~b["new"]).float()[:, None]
        img = torch.from_numpy(b["img"])
        vec = torch.from_numpy(b["vec"])
        act = torch.from_numpy(b["act"])
        valid = torch.from_numpy(b["valid"])
        o, hn = chunk_forward(model, img, vec, h)
        loss, st, _ = head_losses(model, o, vec, act, valid, cw)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        h = hn.detach()
        rows_seen += st["n"]
        hist.append(st)
        if len(first_losses) < 20:
            first_losses.append(st["loss"])
        if step % args.log_every == 0 or step == args.steps:
            el = time.time() - t_train
            agg = _agg(hist)
            print(f"step {step:6d}/{args.steps} loss {agg['loss']:.4f} (p {agg['loss_p']:.3f} r {agg['loss_r']:.3f} "
                  f"i {agg['loss_i']:.3f} m {agg['loss_m']:.3f}) acc prim {agg['acc_prim']:.3f} rec {_fmt(agg['acc_rec'])} "
                  f"item {_fmt(agg['acc_item'])} msg {agg['acc_msg']:.3f} | epochs {batcher.epochs} "
                  f"{rows_seen / max(el, 1e-9):,.0f} rows/s", flush=True)
    train_secs = time.time() - t_train
    start_loss = float(np.mean(first_losses)) if first_losses else None
    end_loss = _agg(hist)["loss"] if hist else None
    print(f"training: {args.steps} steps in {train_secs:.1f}s = {rows_seen / max(train_secs, 1e-9):,.0f} rows/s "
          f"({rows_seen:,} valid rows); loss {start_loss:.4f} -> {end_loss:.4f}", flush=True)

    report = {"run_id": run_id, "steps": args.steps, "batch": B, "chunk": L, "lr": args.lr,
              "seed": args.seed, "success_only": bool(args.success_only), "balance": bool(args.balance),
              "tasks": args.tasks, "shards": len(paths), "episodes": len(segs), "train_episodes": len(train),
              "val_episodes": len(val), "train_rows": tr_rows, "warm_start": args.warm_start,
              "loss_start": start_loss, "loss_end": end_loss, "train_rows_per_s": rows_seen / max(train_secs, 1e-9),
              "train_secs": train_secs, "primary_majority_baseline": float(prim_counts.max() / max(1, prim_counts.sum()))}
    if val:
        t_val = time.time()
        tot, per_task, seq = evaluate(model, val, shards, cache, L, B, cw)
        report["val"] = {
            "rows": int(tot["n"]),
            "acc_primary": _rate(tot["ok_prim"], tot["n"]),
            "acc_recipe": _rate(tot["ok_rec"], tot["n_rec"]), "recipe_rows": int(tot["n_rec"]),
            "recipe_skipped": int(tot["n_rec_skip"]),
            "acc_item": _rate(tot["ok_item"], tot["n_item"]), "item_rows": int(tot["n_item"]),
            "item_skipped": int(tot["n_item_skip"]),
            "acc_msg": _rate(tot["ok_msg"], tot["n"]),
            "sequence_agreement": sequence_agreement(seq),
            "per_task_primary": {TASK_NAMES[t]: {"acc": _rate(c, n), "rows": n}
                                 for t, (c, n) in sorted(per_task.items())},
            "eval_secs": time.time() - t_val,
        }
        _print_val(report["val"])
    else:
        print("no validation episodes (val-frac 0): skipping the held-out report", flush=True)

    payload = _payload(model, opt, run_id, args, report, size, resblock, DEFAULTS, ReturnNormalizer, RunningStd,
                       WorldTasks)
    path = ckpt.save(os.path.join(run_dir, "checkpoints"), payload)
    with open(os.path.join(run_dir, "bc_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"checkpoint: {path}", flush=True)
    print(f"report: {os.path.join(run_dir, 'bc_report.json')}", flush=True)
    return 0


def _agg(hist):
    keys = ("loss", "loss_p", "loss_r", "loss_i", "loss_m")
    out = {k: float(np.mean([s[k] for s in hist])) if hist else float("nan") for k in keys}
    def rate(ok, n):
        ns = sum(s[n] for s in hist)
        return sum(s[ok] for s in hist) / ns if ns else None
    n_all = sum(s["n"] for s in hist)
    out["acc_prim"] = sum(s["ok_prim"] for s in hist) / max(1, n_all)
    out["acc_rec"] = rate("ok_rec", "n_rec")
    out["acc_item"] = rate("ok_item", "n_item")
    out["acc_msg"] = sum(s["ok_msg"] for s in hist) / max(1, n_all)
    return out


def _fmt(x):
    return "  n/a" if x is None else f"{x:.3f}"


def _print_val(v):
    print("validation (held-out episodes, greedy, full-episode GRU):", flush=True)
    print(f"  primary  {_fmt(v['acc_primary'])}  over {v['rows']:,} rows", flush=True)
    print(f"  recipe   {_fmt(v['acc_recipe'])}  over {v['recipe_rows']:,} active rows ({v['recipe_skipped']} out-of-mask skipped)",
          flush=True)
    print(f"  item     {_fmt(v['acc_item'])}  over {v['item_rows']:,} active rows ({v['item_skipped']} out-of-mask skipped)",
          flush=True)
    print(f"  msg      {_fmt(v['acc_msg'])}", flush=True)
    print(f"  action-sequence agreement (collapsed primary sequences, SequenceMatcher ratio): "
          f"{_fmt(v['sequence_agreement'])}", flush=True)
    for name, r in v["per_task_primary"].items():
        print(f"    {name:<32} primary {_fmt(r['acc'])}  rows {r['rows']:,}", flush=True)


def _payload(model, opt, run_id, args, report, size, resblock, defaults, ReturnNormalizer, RunningStd, WorldTasks):
    cfg = dict(defaults)
    cfg.update({"seed": int(args.seed), "obs": E.OBS, "res": RES, "model_size": size, "resblock": bool(resblock),
                "lr": float(args.lr), "warm_start": None, "envmix": "default"})
    tasks = WorldTasks(int(cfg["num_envs"]), seed=int(args.seed), envmix="default")
    return {
        "run_id": run_id,
        "obs": E.obs_signature(RES),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": cfg,
        "counters": {"env_steps": 0, "updates": 0, "episodes": 0},
        "curriculum": None,
        "curriculum_stage": tasks.telemetry()["curriculum_stage"],
        "tasks": tasks.state_dict(),
        "reward_norm": ReturnNormalizer(cfg["gamma"]).state_dict(),
        "rnd_norm": RunningStd().state_dict(),
        "rng": ckpt.rng_state(),
        "metadata": {"reason": "bc_v7", "run_id": run_id, "mode": "bc",
                     "bc": {k: report[k] for k in ("steps", "batch", "chunk", "lr", "seed", "success_only",
                                                   "balance", "tasks", "episodes", "train_episodes",
                                                   "val_episodes", "warm_start")},
                     "bc_val": report.get("val")},
    }


if __name__ == "__main__":
    sys.exit(main())
