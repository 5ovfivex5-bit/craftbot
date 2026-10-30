"""Demonstration shard writer (PROTOCOL 14.2 format).

Layout of out_dir:
  shard_00000.npz, shard_00001.npz, ...   arrays per decision row (T rows per shard):
      img uint8 [T,64,64,2]   vec float32 [T,VEC_DIM]   act int16 [T,4] (primary, recipe, item, msg)
      done bool [T]           task int16 [T] (index into craftsim.TASK_NAMES)
      world int32 [T]         agent int16 [T]           ep int32 [T] (global episode id)
      success bool [T]        (the episode's sim success, written on every row of the episode)
      reward float32 [T]
  meta.json                   constants, vector-name hash, per-task counts and success rates.

Each episode is contiguous inside a shard (never split across shards), so a (world, agent) sequence can be
read in order: rows of one episode follow each other, and `ep` changes at episode boundaries.
"""
import hashlib
import json
import os
import platform
import time

import numpy as np

import craftsim as cs


class ShardWriter:
    def __init__(self, out_dir, shard_rows=20000, compress=False, info=None):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.shard_rows = int(shard_rows)
        self.compress = compress
        self.info = dict(info or {})
        self._parts = []          # list of dict of arrays, one per episode
        self._rows = 0
        self.shards = []
        self.total_rows = 0
        self.counts = {}          # task name -> {episodes, successes, rows}
        self.t_start = time.time()

    def add_episode(self, task_index, world, agent, ep, rows, success):
        """rows: list of (img uint8[64,64,2], vec float32[V], act int32[4], reward float, done bool)."""
        n = len(rows)
        if n == 0:
            return
        part = {
            "img": np.stack([r[0] for r in rows]).astype(np.uint8),
            "vec": np.stack([r[1] for r in rows]).astype(np.float32),
            "act": np.stack([r[2] for r in rows]).astype(np.int16),
            "reward": np.asarray([r[3] for r in rows], np.float32),
            "done": np.asarray([r[4] for r in rows], bool),
            "task": np.full(n, task_index, np.int16),
            "world": np.full(n, world, np.int32),
            "agent": np.full(n, agent, np.int16),
            "ep": np.full(n, ep, np.int32),
            "success": np.full(n, bool(success), bool),
        }
        self._parts.append(part)
        self._rows += n
        self.total_rows += n
        name = cs.TASK_NAMES[task_index]
        c = self.counts.setdefault(name, {"episodes": 0, "successes": 0, "rows": 0})
        c["episodes"] += 1
        c["successes"] += int(bool(success))
        c["rows"] += n
        if self._rows >= self.shard_rows:
            self.flush()

    def flush(self):
        if not self._parts:
            return
        keys = self._parts[0].keys()
        arrays = {k: np.concatenate([p[k] for p in self._parts]) for k in keys}
        path = os.path.join(self.out_dir, f"shard_{len(self.shards):05d}.npz")
        if self.compress:
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)
        self.shards.append(os.path.basename(path))
        self._parts = []
        self._rows = 0

    def close(self, extra=None):
        self.flush()
        meta = {
            "format": "minemind-demos-v1",
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_s": round(time.time() - self.t_start, 2),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "vec_dim": int(cs.VEC_DIM),
            "vec_names_sha1": hashlib.sha1("|".join(cs.VEC_NAMES).encode()).hexdigest(),
            "constants": {
                "NUM_ITEMS": int(cs.NUM_ITEMS), "NUM_BLOCKS": int(cs.NUM_BLOCKS),
                "NUM_RECIPES": int(cs.NUM_RECIPES), "NUM_MOBS": int(cs.NUM_MOBS),
                "NUM_TASKS": int(cs.NUM_TASKS), "NUM_PRIMARY": int(cs.NUM_PRIMARY),
                "NUM_MSGS": int(cs.NUM_MSGS),
            },
            "primary_names": list(cs.PRIMARY_NAMES),
            "task_names": list(cs.TASK_NAMES),
            "total_rows": int(self.total_rows),
            "shards": self.shards,
            "per_task": {k: dict(v, success_rate=(v["successes"] / v["episodes"] if v["episodes"] else 0.0))
                         for k, v in sorted(self.counts.items())},
            "info": self.info,
        }
        if extra:
            meta.update(extra)
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta
