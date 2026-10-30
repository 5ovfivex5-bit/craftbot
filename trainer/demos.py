"""Expert demonstration shards (PROTOCOL 14.2) and the masked multi-head policy loss.

Shared by two users:
  - trainer.bc_v7 (behaviour cloning; the shard / segment / chunk helpers and head_losses moved here unchanged);
  - trainer.run --demos (DAPG-style auxiliary loss inside RecurrentPPO): DemoData loads the shards with the
    idle filter applied, DemoSampler draws one demo batch per PPO minibatch and returns the same masked
    multi-head cross-entropy as bc_v7 (RecurrentPPO adds coef * loss, no value loss).

Shard format: DIR/shard_*.npz with img uint8 [T,64,64,2], vec float32 [T,295], act int16 [T,4] (primary,
recipe, item_select, msg), done bool [T], task int16 [T], world int32 [T], agent int16 [T], ep int32 [T],
success bool [T], reward f32 [T]. A segment is one episode of one (world, agent): consecutive rows with equal
(world, agent, ep) that ends at a done row.
"""
import collections
import glob
import hashlib
import os
import struct
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

import craftsim
from . import envs as E
from . import heads
from .model import MineMindPolicy

RES = E.RES
VEC_DIM = E.VEC_DIM
NUM_ACTIONS = E.NUM_ACTIONS
NUM_RECIPE_CLASSES = E.NUM_RECIPE_CLASSES
NUM_ITEM_CLASSES = E.NUM_ITEM_CLASSES
NUM_MSGS = E.NUM_MSGS

DEMO_CHUNK = 64                       # rows per demo sequence (bc_v7's default --chunk)
IDLE_NAMES = ("noop", "sneak")        # idle primaries: noop (wait / night / mob waits), sneak-toggle spam
IDLE_PRIMS = tuple(list(E.ACTIONS).index(n) for n in IDLE_NAMES if n in list(E.ACTIONS))
NOOP_PRIM = list(E.ACTIONS).index("noop") if "noop" in list(E.ACTIONS) else 0
SNEAK_PRIM = list(E.ACTIONS).index("sneak") if "sneak" in list(E.ACTIONS) else -1



MSG_WEIGHT = 0.1


SHARD_KEYS = ("done", "world", "agent", "ep", "task", "success", "act")


def _check_obs():
    if not E.V7:
        raise SystemExit("bc_v7 needs the v7 observation (run with --obs v7, the default)")


def _npz_member(path, name):
    """Member `name` of an .npz as an array. A stored (uncompressed) member is memory-mapped at its data offset,
    so only the rows a batch touches are read; a deflated member is read into memory."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        with np.load(path) as z:
            return np.asarray(z[name])
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        n_name, n_extra = struct.unpack("<HH", f.read(30)[26:30])
        f.seek(info.header_offset + 30 + n_name + n_extra)
        ver = np.lib.format.read_magic(f)
        if ver == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        offset = f.tell()
    if int(np.prod(shape)) == 0:
        return np.zeros(shape, dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset,
                     order="F" if fortran else "C")


def _npz_member_shape(path, name):
    with zipfile.ZipFile(path) as zf, zf.open(name + ".npy") as f:
        ver = np.lib.format.read_magic(f)
        if ver == (1, 0):
            return np.lib.format.read_array_header_1_0(f)[0]
        return np.lib.format.read_array_header_2_0(f)[0]


class Shard:
    """One shard: the small per-row members are read eagerly; img / vec stay on disk (memmap) behind an LRU."""

    def __init__(self, path):
        self.path = path
        with np.load(path) as z:
            names = set(z.files)
        missing = [k for k in ("img", "vec") + SHARD_KEYS if k not in names]
        if missing:
            raise ValueError(f"{path}: missing members {missing} (PROTOCOL 14.2 format)")
        for k in SHARD_KEYS:
            setattr(self, k, _npz_member(path, k))
        T = len(self.done)
        img_shape = _npz_member_shape(path, "img")
        vec_shape = _npz_member_shape(path, "vec")
        if img_shape[1:] != (RES, RES, 2):
            raise ValueError(f"{path}: image {img_shape[1:]} != model res ({RES}, {RES}, 2)")
        if vec_shape[1:] != (VEC_DIM,) or img_shape[0] != T or vec_shape[0] != T or self.act.shape != (T, 4):
            raise ValueError(f"{path}: row counts / vec width do not match (T={T}, vec {vec_shape}, act {self.act.shape})")
        act = np.asarray(self.act).astype(np.int64)
        if act.min() < 0 or act[:, 0].max() >= NUM_ACTIONS or act[:, 1].max() > NUM_RECIPE_CLASSES - 1 \
                or act[:, 2].max() > NUM_ITEM_CLASSES - 1 or act[:, 3].max() >= NUM_MSGS:
            raise ValueError(f"{path}: action columns out of range")
        self.act = act
        self.done = np.asarray(self.done, bool)
        self.success = np.asarray(self.success, bool)
        self.T = T


class ShardCache:
    """LRU of open shards' img / vec arrays (memmaps for stored members)."""

    def __init__(self, shards, cap):
        self.shards = shards
        self.cap = max(1, int(cap))
        self.lru = collections.OrderedDict()
        self.loads = 0

    def arrays(self, k):
        if k in self.lru:
            self.lru.move_to_end(k)
            return self.lru[k]
        arr = (_npz_member(self.shards[k].path, "img"), _npz_member(self.shards[k].path, "vec"))
        self.loads += 1
        self.lru[k] = arr
        while len(self.lru) > self.cap:
            self.lru.popitem(last=False)
        return arr


class Seg:
    __slots__ = ("shard", "start", "n", "world", "agent", "ep", "task", "success")

    def __init__(self, shard, start, n, world, agent, ep, task, success):
        self.shard, self.start, self.n = shard, start, n
        self.world, self.agent, self.ep, self.task, self.success = world, agent, ep, task, success


def build_segments(shards, success_only=False, task_ids=None):
    segs = []
    for k, sh in enumerate(shards):
        T = sh.T
        if T == 0:
            continue
        brk = np.ones(T, bool)
        brk[1:] = ((sh.world[1:] != sh.world[:-1]) | (sh.agent[1:] != sh.agent[:-1])
                   | (sh.ep[1:] != sh.ep[:-1]) | sh.done[:-1])
        starts = np.flatnonzero(brk)
        ends = np.append(starts[1:], T)
        for s, e in zip(starts.tolist(), ends.tolist()):
            succ = bool(sh.success[s:e].any())
            task = int(sh.task[s])
            if success_only and not succ:
                continue
            if task_ids is not None and task not in task_ids:
                continue
            segs.append(Seg(k, s, e - s, int(sh.world[s]), int(sh.agent[s]), int(sh.ep[s]), task, succ))
    return segs


def _hash_unit(shard_path, seg, seed):
    key = f"{seed}|{os.path.basename(shard_path)}|{seg.world}|{seg.agent}|{seg.ep}|{seg.start}"
    return int(hashlib.md5(key.encode()).hexdigest()[:12], 16) / float(16 ** 12)


def split_segments(segs, shards, val_frac, seed):
    """Episode-level split: an episode is validation when its hash falls below val_frac."""
    if val_frac <= 0:
        return segs, []
    tr, va = [], []
    for s in segs:
        (va if _hash_unit(shards[s.shard].path, s, seed) < val_frac else tr).append(s)
    return tr, va


def make_chunk(cache, items, L):
    """items: list of (Seg or None, pos). Returns numpy arrays for B = len(items) slots of L rows each."""
    B = len(items)
    img = np.zeros((B, L, RES, RES, 2), np.uint8)
    vec = np.zeros((B, L, VEC_DIM), np.float32)
    act = np.zeros((B, L, 4), np.int64)
    valid = np.zeros((B, L), bool)
    done = np.zeros((B, L), bool)
    task = np.full((B, L), -1, np.int64)
    for i, (seg, pos) in enumerate(items):
        if seg is None or pos >= seg.n:
            continue
        take = min(L, seg.n - pos)
        a = seg.start + pos
        im, ve = cache.arrays(seg.shard)
        sh = cache.shards[seg.shard]
        img[i, :take] = im[a:a + take]
        vec[i, :take] = ve[a:a + take]
        act[i, :take] = sh.act[a:a + take]
        done[i, :take] = sh.done[a:a + take]
        valid[i, :take] = True
        task[i, :take] = seg.task
    return {"img": img, "vec": vec, "act": act, "valid": valid, "done": done, "task": task}


def build_model(size="small", resblock=False):
    return MineMindPolicy(res=RES, vec_dim=VEC_DIM, num_actions=NUM_ACTIONS, num_blocks=E.NUM_BLOCKS,
                          num_msgs=NUM_MSGS, size=size, resblock=bool(resblock),
                          num_recipe_classes=NUM_RECIPE_CLASSES, num_item_classes=NUM_ITEM_CLASSES)


def chunk_forward(model, img, vec, h0):
    """img uint8 [B,L,H,W,2], vec [B,L,D], h0 [B,G] -> (out [B,L,G], h_last [B,G])."""
    B, L = img.shape[0], img.shape[1]
    feat = model.encode(img.reshape(B * L, RES, RES, 2), vec.reshape(B * L, VEC_DIM)).view(B, L, -1)
    out, hn = model.gru(feat.transpose(0, 1).contiguous(), h0.unsqueeze(0))
    return out.transpose(0, 1), hn[0]


def head_losses(model, o, vec, act, valid, cw):
    """Masked per-head losses over one chunk. Returns (loss tensor, stats dict of numpy/int values, row arrays)."""
    B, L, _ = o.shape
    N = B * L
    prim = act[..., 0].reshape(N)
    rec = act[..., 1].reshape(N)
    itm = act[..., 2].reshape(N)
    msg = act[..., 3].reshape(N)
    v = valid.reshape(N)
    vflat = vec.reshape(N, VEC_DIM)
    rmask, imask, r_act, i_act = heads.masks(vflat, prim)

    plog = model.policy_head(o).reshape(N, -1)
    ce_p = F.cross_entropy(plog, prim, reduction="none")
    w = cw[prim] * v.float()
    loss_p = (w * ce_p).sum() / w.sum().clamp_min(1e-8)

    r_in = rmask.gather(1, rec[:, None])[:, 0]
    r_tr = v & r_act & r_in
    r_skip = v & r_act & ~r_in
    rlog = heads.masked(model.recipe_head(o).reshape(N, -1), rmask)
    loss_r = F.cross_entropy(rlog[r_tr], rec[r_tr]) if bool(r_tr.any()) else rlog.sum() * 0.0

    i_in = imask.gather(1, itm[:, None])[:, 0]
    i_tr = v & i_act & i_in
    i_skip = v & i_act & ~i_in
    ilog = heads.masked(model.item_head(o).reshape(N, -1), imask)
    loss_i = F.cross_entropy(ilog[i_tr], itm[i_tr]) if bool(i_tr.any()) else ilog.sum() * 0.0

    mlog = model.msg_head(o).reshape(N, -1)
    ce_m = F.cross_entropy(mlog, msg, reduction="none")
    loss_m = (ce_m * v.float()).sum() / v.float().sum().clamp_min(1.0)

    loss = loss_p + loss_r + loss_i + MSG_WEIGHT * loss_m
    with torch.no_grad():
        pp = plog.argmax(-1)
        pr = rlog.argmax(-1)
        pi = ilog.argmax(-1)
        pm = mlog.argmax(-1)
        rows = {
            "pred_prim": pp.view(B, L).cpu().numpy(),
            "ok_prim": (pp == prim).view(B, L).cpu().numpy(),
            "ok_rec": (pr == rec).view(B, L).cpu().numpy(), "m_rec": r_tr.view(B, L).cpu().numpy(),
            "ok_item": (pi == itm).view(B, L).cpu().numpy(), "m_item": i_tr.view(B, L).cpu().numpy(),
            "ok_msg": (pm == msg).view(B, L).cpu().numpy(),
            "valid": valid.cpu().numpy(),
        }
        stats = {
            "loss": float(loss), "loss_p": float(loss_p), "loss_r": float(loss_r), "loss_i": float(loss_i),
            "loss_m": float(loss_m),
            "n": int(v.sum()), "n_rec": int(r_tr.sum()), "n_rec_skip": int(r_skip.sum()),
            "n_item": int(i_tr.sum()), "n_item_skip": int(i_skip.sum()),
            "ok_prim": int((rows["ok_prim"] & rows["valid"]).sum()),
            "ok_rec": int((rows["ok_rec"] & rows["m_rec"]).sum()),
            "ok_item": int((rows["ok_item"] & rows["m_item"]).sum()),
            "ok_msg": int((rows["ok_msg"] & rows["valid"]).sum()),
        }
    return loss, stats, rows


def _shard_paths(dirs):
    paths = []
    for d in dirs:
        if os.path.isfile(d) and d.endswith(".npz"):
            paths.append(d)
            continue
        found = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
        if not found:
            raise SystemExit(f"no shard_*.npz under {d}")
        paths.extend(found)
    return paths


# ----------------------------------------------------------------------------- idle filter

def _episode_starts(sh):
    """bool [T]: True at the first row of each episode (segment break, see the module docstring)."""
    T = sh.T
    brk = np.ones(T, bool)
    brk[1:] = ((sh.world[1:] != sh.world[:-1]) | (sh.agent[1:] != sh.agent[:-1])
               | (sh.ep[1:] != sh.ep[:-1]) | sh.done[:-1])
    return brk


def idle_keep_mask(prim, brk, max_idle):
    """Keep mask over the rows of one shard. An idle run is a maximal run of consecutive rows whose primary is in
    IDLE_PRIMS (noop, sneak) and that stays inside one episode (brk marks episode starts). Runs longer than
    max_idle are removed; shorter runs are kept. Returns (keep bool [T], number of removed runs)."""
    idle = np.isin(prim, IDLE_PRIMS)
    prev = np.r_[False, idle[:-1]]
    start = idle & (~prev | brk)
    rid = np.cumsum(start) * idle                     # run id on idle rows, 0 elsewhere
    counts = np.bincount(rid)
    long_ids = np.flatnonzero(counts > max_idle)
    long_ids = long_ids[long_ids > 0]
    drop = np.isin(rid, long_ids) & idle
    return ~drop, int(long_ids.size)


# ----------------------------------------------------------------------------- demo data (training side)

class DemoData:
    """Shards plus the training segments after the filters. A segment is a contiguous kept run of one episode:
    the idle filter splits an episode at every removed span, so a GRU chunk never bridges a removed span.
    stats[family] holds the counters behind the startup log (family = the shard directory's name)."""

    def __init__(self, dirs, success_only=True, max_idle=8, task_ids=None):
        self.paths = _shard_paths(dirs)
        self.shards = [Shard(p) for p in self.paths]
        self.family = [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in self.paths]
        self.max_idle = int(max_idle)
        self.success_only = bool(success_only)
        self.stats = collections.defaultdict(collections.Counter)
        self.segs = []
        for k, sh in enumerate(self.shards):
            st = self.stats[self.family[k]]
            st["shards"] += 1
            st["rows_all"] += sh.T
            if sh.T == 0:
                continue
            prim = sh.act[:, 0]
            st["noop_rows_all"] += int((prim == NOOP_PRIM).sum())
            if SNEAK_PRIM >= 0:
                st["sneak_rows_all"] += int((prim == SNEAK_PRIM).sum())
            brk = _episode_starts(sh)
            if self.max_idle >= 0:
                keep, n_runs = idle_keep_mask(prim, brk, self.max_idle)
            else:
                keep, n_runs = np.ones(sh.T, bool), 0
            st["idle_runs_removed"] += n_runs
            starts = np.flatnonzero(brk)
            ends = np.append(starts[1:], sh.T)
            for s, e in zip(starts.tolist(), ends.tolist()):
                succ = bool(sh.success[s:e].any())
                task = int(sh.task[s])
                st["episodes"] += 1
                if self.success_only and not succ:
                    continue
                if task_ids is not None and task not in task_ids:
                    continue
                st["episodes_kept"] += 1
                st["rows_before_idle"] += e - s
                ke = keep[s:e]
                st["idle_rows_dropped"] += int((~ke).sum())
                if ke.all():
                    runs = [(0, e - s)]
                else:
                    d = np.diff(np.r_[0, ke.astype(np.int8), 0])
                    runs = list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
                for a, b in runs:
                    if b <= a:
                        continue
                    self.segs.append(Seg(k, s + a, b - a, int(sh.world[s]), int(sh.agent[s]), int(sh.ep[s]),
                                         task, succ))
                    st["rows_kept"] += b - a
                    st["segments"] += 1

    def totals(self):
        t = collections.Counter()
        for c in self.stats.values():
            t.update(c)
        return t

    def describe(self):
        """Startup log lines: per family rows before / after the idle filter and the removed-run counts."""
        out = []
        for fam in sorted(self.stats):
            c = self.stats[fam]
            out.append(f"  {fam}: {c['shards']} shards, rows {c['rows_all']:,}; episodes {c['episodes']:,} "
                       f"(kept {c['episodes_kept']:,}); training rows {c['rows_before_idle']:,} -> "
                       f"{c['rows_kept']:,} after idle filter (dropped {c['idle_rows_dropped']:,} rows in "
                       f"{c['idle_runs_removed']:,} runs); noop rows {c['noop_rows_all']:,}"
                       + (f", sneak rows {c['sneak_rows_all']:,}" if SNEAK_PRIM >= 0 else "")
                       + f"; segments {c['segments']:,}")
        t = self.totals()
        out.append(f"  total: training rows {t['rows_before_idle']:,} -> {t['rows_kept']:,} "
                   f"(max_idle {self.max_idle}{' off' if self.max_idle < 0 else ''}, "
                   f"success_only {'on' if self.success_only else 'off'}), {len(self.segs):,} segments")
        return out


# ----------------------------------------------------------------------------- demo batches (PPO side)

class DemoSampler:
    """One demo batch per call: `batch` sequences of DEMO_CHUNK rows. A sequence starts at an aligned offset of a
    training segment, chosen with probability proportional to its rows (so rows are sampled uniformly). The GRU
    starts from zero for every sequence, as in bc_v7. `coef` is set by the trainer's schedule each update;
    RecurrentPPO adds coef * loss(model) to each minibatch loss."""

    DECISIONS = ("craft", "smelt", "use", "deposit", "withdraw", "give", "drop")

    def __init__(self, data, batch, seed, coef, actions="all"):
        if not data.segs:
            raise ValueError("no demo segments left after the filters")
        self.segs = data.segs
        self.cache = ShardCache(data.shards, cap=8)
        rows = np.array([s.n for s in self.segs], np.float64)
        self.cum = np.cumsum(rows)
        self.nslot = np.ceil(rows / DEMO_CHUNK).astype(np.int64)
        self.batch = int(batch)
        self.rng = np.random.default_rng(int(seed))
        self.coef = float(coef)
        self.cw = torch.ones(NUM_ACTIONS, dtype=torch.float32)
        # primary-head CE weight per expert action: 0 for actions the demos should not teach (movement / aiming
        # rely on the expert's privileged map knowledge); recipe / item heads only train on craft / smelt / use rows
        import craftsim
        names = list(craftsim.PRIMARY_NAMES)
        if actions and actions != "all":
            keep = self.DECISIONS if actions == "decisions" else tuple(a.strip() for a in str(actions).split(","))
            bad = [a for a in keep if a not in names]
            if bad:
                raise ValueError(f"unknown primary actions for --demo-actions: {bad}")
            self.cw.zero_()
            for a in keep:
                self.cw[names.index(a)] = 1.0
        self._names = names

    def describe_actions(self):
        on = [n for n, w in zip(self._names, self.cw.tolist()) if w > 0]
        return "all" if len(on) == len(self._names) else ", ".join(on)

    def sample(self):
        picks = np.searchsorted(self.cum, self.rng.random(self.batch) * self.cum[-1], side="right")
        items = []
        for k in picks.tolist():
            j = int(self.rng.integers(self.nslot[k]))
            items.append((self.segs[k], j * DEMO_CHUNK))
        return make_chunk(self.cache, items, DEMO_CHUNK)

    def loss(self, model):
        """(loss tensor with grad, stats dict from head_losses): masked multi-head CE of one demo batch."""
        b = self.sample()
        img = torch.from_numpy(b["img"])
        vec = torch.from_numpy(b["vec"])
        act = torch.from_numpy(b["act"])
        valid = torch.from_numpy(b["valid"])
        h0 = torch.zeros(self.batch, model.gru_hidden)
        o, _ = chunk_forward(model, img, vec, h0)
        loss, st, _ = head_losses(model, o, vec, act, valid, self.cw)
        return loss, st
