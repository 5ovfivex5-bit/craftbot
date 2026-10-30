"""Load minemind-rec-1 recordings (PROTOCOL 8.2) and turn them into v6 sim-step sequences (8.4).

A recording directory holds per-tick (20 Hz) frames:
  obs_img.u8 [F,64,64,2] uint8, obs_vec.f32 [F,102] float32 (LE), action.i32 [F] int32 (-1 unknown),
  input.jsonl, events.jsonl, meta.json.
One sim decision = `ticks` (4) game ticks. Each sim step takes its observation from the first tick
of its group and its action label by aggregation over the group (rare actions win, see below).
"""
import json
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

RES = 64
VEC_DIM = 102
FORMAT = "minemind-rec-1"
TICKS_PER_STEP = 4

# Names used in 8.3 that differ from craftsim.ACTIONS.
ACTION_ALIASES = {"fwd": "forward", "fwd_jump": "forward_jump"}

# Actions that are rare in a recording and win the group vote (8.3: "priority to rare actions").
RARE_ACTIONS = {"craft_planks", "craft_sticks", "craft_table", "craft_wooden_pickaxe", "craft_stone_pickaxe",
                "craft_furnace", "smelt_iron", "craft_iron_pickaxe", "craft_chest", "place_block",
                "place_table", "place_furnace", "place_chest", "attack", "deposit", "withdraw", "give"}


def _actions():
    # Imported lazily so that `import pretrain.recordings` does not need the native sim.
    import craftsim
    return list(craftsim.ACTIONS)


ACTIONS = None  # filled on first use by action_names()


def action_names():
    global ACTIONS
    if ACTIONS is None:
        ACTIONS = _actions()
    return ACTIONS


def action_index(name):
    return action_names().index(ACTION_ALIASES.get(name, name))


@dataclass
class Recording:
    path: str
    meta: dict
    img: np.ndarray        # uint8 [F, 64, 64, 2] (memmap)
    vec: np.ndarray        # float32 [F, 102] (memmap)
    action: np.ndarray     # int32 [F] (memmap)
    frames: int            # F, the number of complete frames (min over the binary arrays)
    input_lines: int = 0   # lines in input.jsonl (diagnostic)
    event_lines: int = 0   # lines in events.jsonl (diagnostic)
    warnings: list = field(default_factory=list)


def _count_lines(path):
    if not os.path.isfile(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def load(path, strict=True):
    """Open a recording directory. Arrays are memory-mapped. A crashed recording is accepted: F is
    the largest frame count all binary arrays agree on (the shortest one wins)."""
    meta_path = os.path.join(path, "meta.json")
    if not os.path.isfile(meta_path):
        raise ValueError(f"{path}: no meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    warnings = []
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: format {meta.get('format')!r}, expected {FORMAT!r}")
    if int(meta.get("res", RES)) != RES or int(meta.get("vec_dim", VEC_DIM)) != VEC_DIM:
        raise ValueError(f"{path}: res/vec_dim {meta.get('res')}/{meta.get('vec_dim')} "
                         f"(this loader reads {RES}/{VEC_DIM})")
    if int(meta.get("frame_every_ticks", 1)) != 1:
        raise ValueError(f"{path}: frame_every_ticks must be 1")

    img_p = os.path.join(path, "obs_img.u8")
    vec_p = os.path.join(path, "obs_vec.f32")
    act_p = os.path.join(path, "action.i32")
    for p in (img_p, vec_p, act_p):
        if not os.path.isfile(p):
            raise ValueError(f"{path}: missing {os.path.basename(p)}")
    counts = {
        "obs_img.u8": os.path.getsize(img_p) // (RES * RES * 2),
        "obs_vec.f32": os.path.getsize(vec_p) // (VEC_DIM * 4),
        "action.i32": os.path.getsize(act_p) // 4,
    }
    frames = min(counts.values())
    if len(set(counts.values())) != 1:
        warnings.append(f"partial last frame(s) (crash?): arrays hold {counts}; using F={frames}")
        if strict and frames == 0:
            raise ValueError(f"{path}: no complete frames")
    if frames == 0:
        raise ValueError(f"{path}: no frames")
    img = np.memmap(img_p, dtype=np.uint8, mode="r", shape=(frames, RES, RES, 2))
    vec = np.memmap(vec_p, dtype="<f4", mode="r", shape=(frames, VEC_DIM))
    action = np.memmap(act_p, dtype="<i4", mode="r", shape=(frames,))

    rec = Recording(path=path, meta=meta, img=img, vec=vec, action=action, frames=frames,
                    input_lines=_count_lines(os.path.join(path, "input.jsonl")),
                    event_lines=_count_lines(os.path.join(path, "events.jsonl")), warnings=warnings)
    validate(rec, strict=strict)
    return rec


def validate(rec, strict=True):
    """Checks labels are in range and the observation is well formed. Raises ValueError when strict."""
    problems = []
    act = np.asarray(rec.action)
    n_act = len(action_names())
    if act.size and (act.min() < -1 or act.max() >= n_act):
        problems.append(f"action labels out of range [-1, {n_act}): min={act.min()} max={act.max()}")
    if rec.input_lines and rec.input_lines < rec.frames:
        problems.append(f"input.jsonl has {rec.input_lines} lines for {rec.frames} frames")
    vec = np.asarray(rec.vec[:min(rec.frames, 1000)])
    if not np.isfinite(vec).all():
        problems.append("non-finite values in obs_vec")
    unknown = float((act < 0).mean()) if act.size else 1.0
    if unknown > 0.5:
        problems.append(f"{unknown:.0%} of frames have unknown (-1) labels")
    if problems and strict:
        raise ValueError(f"{rec.path}: " + "; ".join(problems))
    rec.warnings.extend(problems)
    return problems


def step_labels(action, ticks=TICKS_PER_STEP):
    """Aggregate per-tick labels (int32 [F]) into one label per sim step (int64 [T], T = F // ticks).
    Group vote: if any rare action (RARE_ACTIONS) occurs in the group, the most frequent rare action
    wins (ties: earliest); otherwise the most frequent non-unknown label; all-unknown -> -1."""
    names = action_names()
    act = np.asarray(action)
    T = act.shape[0] // ticks
    out = np.full(T, -1, dtype=np.int64)
    rare_ids = {names.index(n) for n in RARE_ACTIONS}
    for t in range(T):
        group = [int(a) for a in act[t * ticks:(t + 1) * ticks] if a >= 0]
        if not group:
            continue
        rare = [a for a in group if a in rare_ids]
        pool = rare if rare else group
        counts = Counter(pool)
        best = max(counts.values())
        out[t] = next(a for a in pool if counts[a] == best)
    return out


def sim_steps(rec, ticks=TICKS_PER_STEP):
    """Whole-recording view: (step_index_frame [T] = first tick of each group, labels [T] int64)."""
    T = rec.frames // ticks
    first = np.arange(T, dtype=np.int64) * ticks
    labels = step_labels(np.asarray(rec.action[:T * ticks]), ticks)
    return first, labels


def iter_chunks(rec, chunk=64, ticks=TICKS_PER_STEP, labels=None, first=None):
    """Yield dicts {img uint8 [L,64,64,2], vec float32 [L,102], action int64 [L], start} for consecutive
    chunks of sim steps (L <= chunk). Observations are copied out of the memmap."""
    if first is None or labels is None:
        first, labels = sim_steps(rec, ticks)
    T = len(labels)
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        idx = first[t0:t1]
        yield {"img": np.ascontiguousarray(rec.img[idx]), "vec": np.ascontiguousarray(rec.vec[idx]),
               "action": labels[t0:t1].copy(), "start": t0}


def find_recordings(root):
    """Recording directories under root (root itself if it holds a meta.json), sorted."""
    if os.path.isfile(os.path.join(root, "meta.json")):
        return [root]
    out = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isfile(os.path.join(p, "meta.json")):
                out.append(p)
    return out


def describe(rec):
    act = np.asarray(rec.action)
    _first, labels = sim_steps(rec)
    names = action_names()
    counts = Counter(names[a] for a in labels.tolist() if a >= 0)
    return {"path": rec.path, "frames": rec.frames, "sim_steps": len(labels),
            "goal": rec.meta.get("goal"), "unknown_frames": int((act < 0).sum()),
            "label_counts": dict(counts.most_common()), "warnings": rec.warnings}


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:] or ["."]:
        for p in find_recordings(d):
            print(json.dumps(describe(load(p, strict=False)), indent=2))
