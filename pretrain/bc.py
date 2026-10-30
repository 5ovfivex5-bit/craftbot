"""Behavior-cloning pretraining of the v6 MineMindPolicy on human sessions (PROTOCOL 8.4, 9.5).

Inputs: v2 sessions (9.2) and v1 recording dirs (8.2), either given directly or as a folder holding
them (pretrain/session.py). Model and checkpoint are the trainer's own (trainer/model.py,
trainer/checkpoint.py), so the output is a normal checkpoint for `trainer.run --resume` and the bridge.

- Actions: one label per sim step (4 ticks), group vote from recordings.step_labels; -1 is masked.
- Goals (9.5, default `--goal-mode hindsight`): the goal one-hot (vec[26:41]) is set per step to the
  next MineMind item the player obtains within 5 min (craft/smelt/pickup, not chest/give), else zeros.
  `--goal-mode recorded` keeps the goal stored in the recording.
- Loss: cross-entropy on the action head with class weights ~ inverse-sqrt frequency, capped so the
  rarest weight is at most `--max-weight-ratio` (10) times the weight of the most frequent class.
- Chat head: untouched unless `--msg-silent` (then it is pulled to the silent token, weight 0.1).
- Split: per session, the last `--val-frac` (10%) of its ticks are validation, the rest training.
- Memory: no image array is held for a whole session; each training segment (`--chunk` sim steps) is
  read through the session's chunk cache (an LRU of decompressed storage chunks).
- Report: validation CE, accuracy, balanced accuracy (mean per-class recall) and per-action accuracy.

Usage (from the project root):
  .venv/bin/python -m pretrain.bc --recordings DIR_OR_SESSION [...] --out DIR --steps 500 \
      [--warm-start runs/team-v6/checkpoints/ckpt_....pt] [--lr 3e-4] [--batch 8] [--chunk 64] \
      [--goal-mode hindsight|recorded] [--msg-silent]
"""
import argparse
import datetime
import os
import sys
import time
import uuid
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import craftsim  # noqa: E402
from pretrain import recordings as rec_lib  # noqa: E402
from pretrain import session as sess_lib  # noqa: E402
from trainer import checkpoint as ckpt  # noqa: E402
from trainer.envs import NUM_ACTIONS, NUM_BLOCKS, NUM_MSGS, VEC_DIM, obs_signature  # noqa: E402
from trainer.model import MineMindPolicy  # noqa: E402

MSG_LOSS_COEF = 0.1


def build(res, warm_start=None):
    """Fresh v6 policy, or the one stored in a v6 checkpoint (same as trainer.eval.load_policy)."""
    if warm_start:
        from trainer.eval import load_policy
        model, payload = load_policy(warm_start)
        model.train()
        return model, payload
    model = MineMindPolicy(res=res, vec_dim=VEC_DIM, num_actions=NUM_ACTIONS, num_blocks=NUM_BLOCKS,
                           size="small", resblock=True, num_msgs=NUM_MSGS)
    return model, None


def forward_chunks(model, img, vec):
    """img uint8 [B,L,H,W,2], vec float [B,L,D] -> (action logits [B,L,A], msg logits [B,L,M])."""
    B, L = img.shape[:2]
    feat = model.encode(img.reshape(B * L, *img.shape[2:]), vec.reshape(B * L, vec.shape[-1]))
    x = feat.view(B, L, -1).transpose(0, 1)                       # [L,B,feat]
    h0 = torch.zeros(1, B, model.gru_hidden, device=img.device)
    out, _ = model.gru(x, h0)                                     # [L,B,gru]
    logits = model.policy_head(out).transpose(0, 1)
    msg = model.msg_head(out).transpose(0, 1)
    return logits, msg


class SessionData:
    """Per-session step table: first tick of each sim step, labels, hindsight goal index."""

    def __init__(self, sess, goal_mode, val_frac):
        self.sess = sess
        self.first, self.labels = sess.sim_steps()
        self.goals = sess_lib.hindsight_goals(sess, self.first) if goal_mode == "hindsight" else None
        self.context = sess.step_context(self.first)     # 9.6 per-step context text (dataset only, not a model input)
        self.cut = int(len(self.labels) * (1.0 - val_frac))

    def segments(self, lo, hi, chunk):
        """(s0, s1) windows of `chunk` steps in [lo, hi) that contain at least one labelled step."""
        out = []
        for s0 in range(lo, hi, chunk):
            s1 = min(hi, s0 + chunk)
            if (self.labels[s0:s1] >= 0).any():
                out.append((s0, s1))
        return out

    def load(self, s0, s1):
        t = self.sess.read(self.first[s0], self.first[s1 - 1] + 1)
        rel = np.minimum(self.first[s0:s1] - t.t0, len(t.img) - 1)
        img = t.img[rel]
        vec = t.vec[rel].copy()
        if self.goals is not None:
            g = self.goals[s0:s1]
            vec[:, craftsim.GOAL_OFFSET:craftsim.GOAL_OFFSET + craftsim.NUM_ITEMS] = 0.0
            ok = g >= 0
            rows = np.nonzero(ok)[0]
            vec[rows, craftsim.GOAL_OFFSET + g[ok]] = 1.0
        return img, vec, self.labels[s0:s1].copy(), self.context[s0:s1]


def collate(items, chunk):
    """List of (img, vec, action, context) -> padded tensors; padding has action -1 (masked).
    The context text is carried in the dataset but is not a model input."""
    B = len(items)
    img = np.zeros((B, chunk, rec_lib.RES, rec_lib.RES, 2), np.uint8)
    vec = np.zeros((B, chunk, VEC_DIM), np.float32)
    act = np.full((B, chunk), -1, np.int64)
    for b, (i, v, a, _ctx) in enumerate(items):
        L = len(a)
        img[b, :L] = i
        vec[b, :L] = v
        act[b, :L] = a
    return torch.from_numpy(img), torch.from_numpy(vec), torch.from_numpy(act)


def class_weights(counts, max_ratio):
    """Inverse-sqrt frequency weights, scaled to mean 1 over the training steps. The rarest class weight is
    capped at max_ratio x the weight of the most frequent class (the minimum weight)."""
    n = np.asarray(counts, np.float64)
    present = n > 0
    w = np.ones(NUM_ACTIONS, np.float64)
    if not present.any():
        return w.astype(np.float32)
    w[present] = n[present] ** -0.5
    wmin = w[present].min()
    w[present] = np.minimum(w[present], max_ratio * wmin)
    mean = (n[present] * w[present]).sum() / n[present].sum()
    w[present] /= mean
    return w.astype(np.float32)


def evaluate(model, items, batch, chunk, weight):
    """items: list of (SessionData, s0, s1). Returns unweighted CE, accuracy, balanced accuracy, per-action."""
    model.eval()
    names = rec_lib.action_names()
    correct = np.zeros(NUM_ACTIONS, np.int64)
    total = np.zeros(NUM_ACTIONS, np.int64)
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(items), batch):
            loaded = [sd.load(s0, s1) for sd, s0, s1 in items[i:i + batch]]
            img, vec, act = collate(loaded, chunk)
            logits, _msg = forward_chunks(model, img, vec)
            loss_sum += F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), act.reshape(-1),
                                        ignore_index=-1, reduction="sum").item()
            valid = act >= 0
            n += int(valid.sum())
            a = act[valid].numpy()
            p = logits.argmax(-1)[valid].numpy()
            np.add.at(total, a, 1)
            np.add.at(correct, a, (a == p).astype(np.int64))
    model.train()
    seen = total > 0
    acc = correct.sum() / max(1, total.sum())
    bal = float((correct[seen] / total[seen]).mean()) if seen.any() else 0.0
    per_action = {names[i]: (int(total[i]), float(correct[i] / total[i]) if total[i] else None)
                  for i in range(NUM_ACTIONS) if total[i]}
    return {"loss": loss_sum / max(1, n), "acc": float(acc), "balanced_acc": bal,
            "per_action": per_action, "n": int(n)}


def describe_sessions(sds):
    names = rec_lib.action_names()
    for sd in sds:
        s = sd.sess
        lab = Counter(names[a] for a in sd.labels.tolist() if a >= 0)
        cover = f", hindsight goals on {100 * (sd.goals >= 0).mean():.0f}% of steps" if sd.goals is not None else ""
        n_ctx = sum(1 for c in sd.context if c is not None)
        cover += f", context text on {n_ctx} steps ({len(s.contexts())} segments)" if n_ctx or s.contexts() else ""
        print(f"session v{s.version} {s.path}: ticks={s.n_ticks} sim_steps={len(sd.labels)} "
              f"goal={s.goal} train_steps={sd.cut} val_steps={len(sd.labels) - sd.cut}{cover}")
        print(f"  labels: {dict(lab.most_common(6))}")
        for w in s.warnings:
            print(f"  warning: {w}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", "--sessions", dest="recordings", nargs="+", required=True,
                    help="session dirs (v2 or v1), or folders containing them")
    ap.add_argument("--out", required=True, help="output directory; the checkpoint goes to <out>/<run_id>/checkpoints/")
    ap.add_argument("--steps", type=int, default=500, help="gradient steps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64, help="sim steps per training sequence")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1, help="last fraction of each session's time")
    ap.add_argument("--goal-mode", choices=["hindsight", "recorded"], default="hindsight")
    ap.add_argument("--max-weight-ratio", type=float, default=10.0, help="class weight cap (rarest/most frequent)")
    ap.add_argument("--msg-silent", action="store_true", help="pull the chat head to the silent token")
    ap.add_argument("--warm-start", default=None, help="v6 checkpoint to start from (weights only)")
    ap.add_argument("--res", type=int, default=rec_lib.RES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    paths = []
    for r in args.recordings:
        paths.extend(sess_lib.find_sessions(r))
    if not paths:
        raise SystemExit("no sessions found")
    sds = [SessionData(sess_lib.open_session(p), args.goal_mode, args.val_frac) for p in paths]
    describe_sessions(sds)

    train, val = [], []
    for sd in sds:
        train.extend((sd, s0, s1) for s0, s1 in sd.segments(0, sd.cut, args.chunk))
        val.extend((sd, s0, s1) for s0, s1 in sd.segments(sd.cut, len(sd.labels), args.chunk))
    if not train:
        raise SystemExit("no labelled training segments")
    print(f"segments: train={len(train)} val={len(val)} (chunk={args.chunk} sim steps)")

    counts = np.zeros(NUM_ACTIONS, np.int64)
    for sd, s0, s1 in train:
        lab = sd.labels[s0:s1]
        counts += np.bincount(lab[lab >= 0], minlength=NUM_ACTIONS)
    weight_np = class_weights(counts, args.max_weight_ratio)
    weight = torch.from_numpy(weight_np)
    names = rec_lib.action_names()
    print("class weights (count, weight): " + ", ".join(
        f"{names[i]}={counts[i]}:{weight_np[i]:.2f}" for i in np.argsort(-counts) if counts[i]))

    model, src_payload = build(args.res, args.warm_start)
    if src_payload is not None:
        print(f"warm start from {args.warm_start} (run {src_payload.get('run_id')})")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    model.train()

    losses = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train), size=min(args.batch, len(train)))
        loaded = [train[i][0].load(train[i][1], train[i][2]) for i in idx]
        img, vec, act = collate(loaded, args.chunk)
        logits, msg = forward_chunks(model, img, vec)
        ce = F.cross_entropy(logits.reshape(-1, NUM_ACTIONS), act.reshape(-1), weight=weight,
                             ignore_index=-1)
        loss = ce
        msg_loss = torch.zeros(())
        if args.msg_silent:
            valid = (act >= 0).float()
            msg_target = torch.zeros_like(act)                               # silent token
            msg_loss = (F.cross_entropy(msg.reshape(-1, NUM_MSGS), msg_target.reshape(-1), reduction="none")
                        .view(act.shape) * valid).sum() / valid.sum().clamp(min=1)
            loss = ce + MSG_LOSS_COEF * msg_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(ce.item()))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step:5d}  wce {np.mean(losses[-args.log_every:]):.4f}  msg {float(msg_loss):.4f}  "
                  f"({time.time() - t_start:.1f}s)", flush=True)

    head = float(np.mean(losses[:20]))
    tail = float(np.mean(losses[-20:]))
    print(f"train weighted CE first-20 mean {head:.4f} -> last-20 mean {tail:.4f}")

    res = evaluate(model, val, args.batch, args.chunk, weight) if val else \
        {"loss": None, "acc": None, "balanced_acc": None, "per_action": {}, "n": 0}
    print(f"validation: CE {res['loss']}  accuracy {res['acc']}  balanced accuracy {res['balanced_acc']}  "
          f"over {res['n']} labelled steps")
    print("per-action validation accuracy (count, acc):")
    for name, (cnt, a) in sorted(res["per_action"].items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:22s} {cnt:6d}  {'-' if a is None else f'{a:.3f}'}")

    run_id = args.run_id or ("bc-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    from trainer.run import DEFAULTS
    config = dict(DEFAULTS)
    config.update({"res": args.res, "model_size": "small", "resblock": True, "warm_start": None,
                   "backend": "cpu", "device": "cpu", "amp": False})
    payload = {
        "run_id": run_id,
        "obs": obs_signature(args.res),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": config,
        "counters": {"env_steps": 0, "updates": int(args.steps), "episodes": 0},
        "metadata": {"reason": "behavior-cloning", "run_id": run_id, "source": "pretrain.bc",
                     "sessions": [os.path.abspath(p) for p in paths], "bc_steps": args.steps,
                     "bc_lr": args.lr, "warm_start": os.path.abspath(args.warm_start) if args.warm_start else None,
                     "goal_mode": args.goal_mode, "class_weights": weight_np.tolist(),
                     "max_weight_ratio": args.max_weight_ratio, "msg_silent": bool(args.msg_silent),
                     "context_texts": sorted({c for sd in sds for c in sd.context if c}),
                     "val_frac": args.val_frac, "val": res},
    }
    ckpt_dir = os.path.join(os.path.abspath(args.out), run_id, "checkpoints")
    path = ckpt.save(ckpt_dir, payload)
    print(f"checkpoint: {path}")
    return path


if __name__ == "__main__":
    main()
