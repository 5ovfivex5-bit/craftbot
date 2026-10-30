"""GPU (CUDA) batched MineMind simulator: same rules as craftsim.CraftSim, state lives on the GPU.

Usage:
    from gpusim.gpusim import GpuCraftSim
    sim = GpuCraftSim(num_envs=1024, seed=0, res=32, pool_size=1024)
    img, vec = sim.reset()                       # torch CUDA tensors
    img, vec, rew, done, ach = sim.step(actions) # actions: int32/int64 CUDA tensor [N]

Returned tensors are the sim's own buffers (overwritten by the next call): clone to keep them.

Design
- Per env: one POD struct `GEnv` (device memory, ~736 B, AoS) + world uint8[64*64*64] (256 KB).
- step: 1 thread / env (physics, mining, crafting, goal). reset: 1 block / env (copies a pool world).
  obs: 1 thread / (env, pixel) DDA over the 32x32 image rays + 1 thread / env for the 39-d vector.
- Worlds come from the CPU generator (craftsim.CraftSim.reset() + save_state()) and are kept in a
  GPU pool. Auto-reset picks a uniform random pool world (not a fresh generation). A background
  thread generates new worlds on the CPU; the main thread commits one chunk every `refresh_every`
  steps into a pool slot (on the same CUDA stream, so no races).
"""
import os
import queue
import struct
import sys
import threading

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

ACTIONS = [
    "noop", "forward", "back", "left", "right", "jump", "forward_jump",
    "turn_left", "turn_right", "look_up", "look_down",
    "attack", "place_block", "place_table", "place_furnace",
    "craft_planks", "craft_sticks", "craft_table", "craft_wooden_pickaxe",
    "craft_stone_pickaxe", "craft_furnace", "smelt_iron", "craft_iron_pickaxe",
]
ITEMS = [
    "log", "planks", "stick", "crafting_table", "wooden_pickaxe", "cobblestone",
    "stone_pickaxe", "furnace", "coal", "iron_ore", "iron_ingot", "iron_pickaxe",
    "diamond", "dirt",
]
BLOCKS = [
    "air", "grass", "dirt", "stone", "cobblestone", "log", "leaves", "planks",
    "crafting_table", "furnace", "coal_ore", "iron_ore", "diamond_ore", "bedrock",
    "sand", "gravel", "water", "deepslate",
]
NUM_ACTIONS, NUM_ITEMS, NUM_BLOCKS, VEC_DIM, GOAL_OFFSET = 23, 14, 18, 39, 25
WX = WY = WZ = 64
WORLD = WX * WY * WZ
SEA_LEVEL = 30
GOAL_BIT = 1 << 31
_SNAP_MAGIC, _SNAP_VERSION = 0x31565343, 1

_ext = None


def _load_ext():
    """JIT-compiles csrc/gpusim_kernels.cu (torch.utils.cpp_extension, cached per source hash)."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        _ext = load(
            name="gpusim_ext",
            sources=[os.path.join(_HERE, "csrc", "gpusim_kernels.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    return _ext


def parse_blob(blob):
    """Parses a CraftSim.save_state() blob (layout: sim.cpp saveState) into GPU state fields."""
    b = bytes(blob)
    magic, version, wn = struct.unpack_from("<IIQ", b, 0)
    if magic != _SNAP_MAGIC or version != _SNAP_VERSION or wn != WORLD:
        raise ValueError("not a craftsim env snapshot")
    off = 16
    world = np.frombuffer(b, np.uint8, count=wn, offset=off).copy()
    off += wn
    d = {"world": world}
    d["px"], d["py"], d["pz"], d["vx"], d["vy"], d["vz"] = struct.unpack_from("<6d", b, off); off += 48
    d["yaw"], d["pitch"] = struct.unpack_from("<2f", b, off); off += 8
    d["onGround"], d["kit"], d["restart"] = struct.unpack_from("<3B", b, off); off += 3
    d["fallDist"], = struct.unpack_from("<d", b, off); off += 8
    d["health"], = struct.unpack_from("<i", b, off); off += 4
    d["inv"] = list(struct.unpack_from("<14i", b, off)); off += 56
    d["got"], d["kitGot"] = struct.unpack_from("<II", b, off); off += 8
    d["fuel"], d["mx"], d["my"], d["mz"] = struct.unpack_from("<4i", b, off); off += 16
    d["prog"], = struct.unpack_from("<d", b, off); off += 8
    d["t"], d["ticks"], d["episode"], d["lastAction"] = struct.unpack_from("<4i", b, off); off += 16
    d["seed"], d["goal"] = struct.unpack_from("<Qi", b, off); off += 12
    d["visited"] = list(struct.unpack_from("<64Q", b, off)); off += 512
    d["seenBlocks"], d["bestDepthStep"] = struct.unpack_from("<Ii", b, off); off += 8
    rn, = struct.unpack_from("<I", b, off); off += 4
    d["rng_text"] = b[off:off + rn]                         # CPU rng state (kept for repacking)
    off += rn
    if off != len(b):
        raise ValueError("env snapshot has trailing bytes")
    d["seenBlocks"] &= 0xFFFFFFFF
    d["got"] &= 0xFFFFFFFF
    d["kitGot"] &= 0xFFFFFFFF
    d["seenObs"] = 0
    d["lastNov"] = 0.0
    return d


def _i64(u):
    return struct.unpack("<q", struct.pack("<Q", int(u) & 0xFFFFFFFFFFFFFFFF))[0]


def gen_worlds(n, seed):
    """CPU worldgen: n worlds from CraftSim.reset() (OpenMP). Returns (worlds u8[n,WORLD],
    spawn f64[n,4] = px,py,pz,yaw, seeds i64[n])."""
    import craftsim
    cs = craftsim.CraftSim(n, seed=seed, res=4, max_steps=3000, ticks=4, threads=0)
    cs.reset()
    worlds = np.empty((n, WORLD), np.uint8)
    spawn = np.empty((n, 4), np.float64)
    seeds = np.empty(n, np.int64)
    for i in range(n):
        d = parse_blob(cs.save_state(i))
        worlds[i] = d["world"]
        spawn[i] = (d["px"], d["py"], d["pz"], d["yaw"])
        seeds[i] = _i64(d["seed"])
    return worlds, spawn, seeds


class GpuCraftSim:
    """N envs on the GPU. See module docstring for the design and the pool trade-off.

    pool_size: number of pregenerated worlds on the GPU (256 KB each: 1024 -> 256 MB).
    pool_chunk: worlds generated per CPU batch (default min(256, pool_size)); pool is rounded up
        to a multiple of it.
    refresh_every: commit one background-generated chunk into the pool every this many steps
        (0 disables refresh).
    """

    def __init__(self, num_envs=16, seed=0, res=32, max_steps=3000, ticks=4, device="cuda",
                 pool_size=512, pool_chunk=None, refresh_every=200):
        self.n, self.res, self.max_steps, self.ticks = int(num_envs), int(res), int(max_steps), int(ticks)
        self.dev = torch.device(device)
        self.ext = _load_ext()
        self.env_bytes = int(self.ext.ENV_BYTES)
        n, dev = self.n, self.dev

        self.envs = torch.zeros(n * self.env_bytes, dtype=torch.uint8, device=dev)
        self.worlds = torch.zeros(n, WORLD, dtype=torch.uint8, device=dev)
        self.img = torch.zeros(n, self.res, self.res, 2, dtype=torch.uint8, device=dev)
        self.vec = torch.zeros(n, VEC_DIM, dtype=torch.float32, device=dev)
        self.rew = torch.zeros(n, dtype=torch.float32, device=dev)
        self.nov = torch.zeros(n, dtype=torch.float32, device=dev)
        self.done = torch.zeros(n, dtype=torch.uint8, device=dev)     # bool view returned to callers
        self.ach = torch.zeros(n, dtype=torch.int64, device=dev)

        w = torch.ones(NUM_ITEMS, dtype=torch.float32)
        w[13] = 0.0                                                    # default: items 0..12 (no dirt)
        self.goal_w = w.to(dev)
        self.kit_table = torch.zeros(NUM_ITEMS * NUM_ITEMS, dtype=torch.int32, device=dev)
        self.kit_prob = torch.zeros(NUM_ITEMS, dtype=torch.float32, device=dev)
        self.novelty = 0.0

        self.chunk = int(pool_chunk) if pool_chunk else min(256, int(pool_size))
        self.n_chunks = max(1, -(-int(pool_size) // self.chunk))
        self.pool_size = self.chunk * self.n_chunks
        P = self.pool_size
        self.pool_w = torch.zeros(P, WORLD, dtype=torch.uint8, device=dev)
        self.pool_spawn = torch.zeros(P, 4, dtype=torch.float64, device=dev)
        self.pool_seed = torch.zeros(P, dtype=torch.int64, device=dev)
        self._gen_seed = int(seed) * 7919 + 1
        for k in range(self.n_chunks):
            self._commit(k, gen_worlds(self.chunk, self._next_seed()))

        rng = np.random.default_rng(seed)
        self.ext.seed(self.envs, torch.from_numpy(rng.integers(1, 2**62, n, dtype=np.int64)).to(dev))

        # Background refresh of the pool (CPU generation only; GPU writes stay on the main thread).
        self.refresh_every = int(refresh_every)
        self._steps = 0
        self._refresh_idx = 0
        self._queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = None
        if self.refresh_every > 0:
            self._thread = threading.Thread(target=self._refill_loop, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------ pool
    def _next_seed(self):
        self._gen_seed += 1
        return self._gen_seed

    def _commit(self, k, batch):
        worlds, spawn, seeds = batch
        sl = slice(k * self.chunk, (k + 1) * self.chunk)
        self.pool_w[sl].copy_(torch.from_numpy(worlds).to(self.dev))
        self.pool_spawn[sl].copy_(torch.from_numpy(spawn).to(self.dev))
        self.pool_seed[sl].copy_(torch.from_numpy(seeds).to(self.dev))

    def _refill_loop(self):
        while not self._stop.is_set():
            try:
                batch = gen_worlds(self.chunk, self._next_seed())
            except Exception:  # keep the sim alive if CPU generation fails once
                continue
            while not self._stop.is_set():
                try:
                    self._queue.put(batch, timeout=0.5)
                    break
                except queue.Full:
                    pass

    def _maybe_refresh(self):
        if self.refresh_every <= 0:
            return
        self._steps += 1
        if self._steps % self.refresh_every:
            return
        try:
            batch = self._queue.get_nowait()
        except queue.Empty:
            return
        self._commit(self._refresh_idx, batch)
        self._refresh_idx = (self._refresh_idx + 1) % self.n_chunks

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # ------------------------------------------------------------ core API
    def _obs(self):
        self.ext.obs(self.envs, self.worlds, self.img, self.vec, self.max_steps, self.res)

    def reset(self):
        """Fresh episode for every env (pool worlds, sampled goals). Returns (img, vec)."""
        self.ext.reset(self.envs, self.worlds, self.pool_w, self.pool_spawn, self.pool_seed,
                       self.goal_w, self.kit_table, self.kit_prob, self.done, True)
        self._obs()
        self.rew.zero_()
        self.nov.zero_()
        self.done.fill_(1)                 # finalize: "done" path = seenBlocks from the new obs, no pay
        self.ext.finalize(self.envs, self.rew, self.nov, self.done, 0.0)
        self.done.zero_()
        self.ach.zero_()
        return self.img, self.vec

    def step(self, actions):
        """actions: int tensor [N] on the sim device. Returns (img, vec, rew, done, ach)."""
        a = actions.to(device=self.dev, dtype=torch.int32).contiguous()
        if a.numel() != self.n:
            raise ValueError("actions must have length num_envs")
        self._maybe_refresh()
        self.ext.step(self.envs, self.worlds, a, self.rew, self.nov, self.done, self.ach, self.ticks, self.max_steps)
        self.ext.reset(self.envs, self.worlds, self.pool_w, self.pool_spawn, self.pool_seed,
                       self.goal_w, self.kit_table, self.kit_prob, self.done, False)
        self._obs()
        self.ext.finalize(self.envs, self.rew, self.nov, self.done, float(self.novelty))
        return self.img, self.vec, self.rew, self.done.view(torch.bool), self.ach

    # ------------------------------------------------------------ configuration
    def set_goal_weights(self, weights):
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.size != NUM_ITEMS or (w < 0).any() or not w.sum() > 0:
            raise ValueError("goal weights: NUM_ITEMS non-negative values, sum > 0")
        self.goal_w.copy_(torch.from_numpy(w).to(self.dev))

    def set_start_kits(self, table, prob):
        t = np.asarray(table, dtype=np.float64).reshape(NUM_ITEMS, NUM_ITEMS)
        p = np.asarray(prob, dtype=np.float32).reshape(NUM_ITEMS)
        if (t < 0).any() or not np.isfinite(t).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("start kits: counts >= 0, probs in [0, 1]")
        counts = np.floor(t + 0.5).astype(np.int32)          # == lround for non-negative values
        self.kit_table.copy_(torch.from_numpy(counts.reshape(-1)).to(self.dev))
        self.kit_prob.copy_(torch.from_numpy(p).to(self.dev))

    def set_novelty(self, scale):
        if not (scale >= 0) or not np.isfinite(scale):
            raise ValueError("novelty scale must be finite and >= 0")
        self.novelty = float(scale)

    def goal(self, i=0):
        return int(self.ext.get_env(self.envs, int(i))["goal"])

    def set_goal(self, i, item):
        item = int(item)
        if not 0 <= item < NUM_ITEMS:
            raise ValueError("item index out of range")
        self.ext.set_env(self.envs, int(i), {"goal": item})
        self.vec[i, GOAL_OFFSET:] = 0.0
        self.vec[i, GOAL_OFFSET + item] = 1.0

    # ------------------------------------------------------------ debug helpers (copy one env)
    def state(self, i=0):
        i = int(i)
        e = self.ext.get_env(self.envs, i)
        tg = torch.empty(self.n, 5, dtype=torch.int32, device=self.dev)
        self.ext.target(self.envs, self.worlds, tg)
        hit, tx, ty, tz, tb = (int(v) for v in tg[i].tolist())
        return {
            "x": e["px"], "y": e["py"], "z": e["pz"], "vx": e["vx"], "vy": e["vy"], "vz": e["vz"],
            "yaw": e["yaw"], "pitch": e["pitch"], "on_ground": bool(e["onGround"]), "health": e["health"],
            "t": e["t"], "ticks": e["ticks"], "seed": e["seed"] & 0xFFFFFFFFFFFFFFFF,
            "episode": e["episode"], "last_action": e["lastAction"], "mining_progress": e["prog"],
            "goal": e["goal"], "kit": bool(e["kit"]), "restart": bool(e["restart"]),
            "target": ({"x": tx, "y": ty, "z": tz, "block": BLOCKS[tb]} if hit else None),
        }

    def inventory(self, i=0):
        inv = self.ext.get_env(self.envs, int(i))["inv"]
        return {ITEMS[k]: int(inv[k]) for k in range(NUM_ITEMS)}

    def set_inventory(self, i, item, count):
        d = self.ext.get_env(self.envs, int(i))
        inv = list(d["inv"])
        inv[ITEMS.index(item)] = int(count)
        self.ext.set_env(self.envs, int(i), {"inv": inv})

    def block(self, i, x, y, z):
        if y >= WY:
            return "air"
        if not (0 <= x < WX and 0 <= z < WZ and y >= 0):
            return "bedrock"
        return BLOCKS[int(self.worlds[int(i), (y * WZ + z) * WX + x])]

    def region(self, i, radius=16):
        """Like CraftSim.region: ((ox, 0, oz), uint8[sy, sz, sx]) with array index [y][z][x]."""
        d = self.ext.get_env(self.envs, int(i))
        cx, cz = int(np.floor(d["px"])), int(np.floor(d["pz"]))
        radius = max(0, int(radius))
        ox, oz = max(0, cx - radius), max(0, cz - radius)
        x1, z1 = min(WX - 1, cx + radius), min(WZ - 1, cz + radius)
        sx, sz = max(0, x1 - ox + 1), max(0, z1 - oz + 1)
        arr = self.worlds[int(i)].view(WY, WZ, WX)[:, oz:oz + sz, ox:ox + sx]
        return (ox, 0, oz), arr.contiguous().cpu().numpy()

    def target(self, i=0):
        """Block under the crosshair (or None), like Snapshot.hasTarget."""
        return self.state(i)["target"]

    def set_pose(self, i, yaw, pitch):
        self.ext.set_env(self.envs, int(i), {"yaw": float(yaw), "pitch": float(np.clip(pitch, -90, 90))})

    # ------------------------------------------------------------ snapshots (Go-Explore style)
    def load_state(self, i, blob, goal=None):
        """Restores env i from CraftSim.save_state() bytes (same semantics as CraftSim.load_state).
        Returns (img[res,res,2], vec[VEC_DIM]) clones of env i's fresh obs."""
        i = int(i)
        d = parse_blob(blob)
        self.worlds[i].copy_(torch.from_numpy(d.pop("world")).to(self.dev))
        d["restart"] = 1
        d["t"] = 0
        if goal is not None:
            d["goal"] = int(goal)
        d["onGround"] = int(d["onGround"])
        d["kit"] = int(d["kit"])
        self.ext.set_env(self.envs, i, d)
        self._obs()
        return self.img[i].clone(), self.vec[i].clone()
