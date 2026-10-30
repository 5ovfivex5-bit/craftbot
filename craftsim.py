"""Batched Minecraft-like voxel sim. Python facade over the native pybind11 module `_craftsim`.

Build the native module first: ./build.sh (runs tools/gen_registry.py, then builds build/_craftsim*.so).

Indexing: CraftSim(num_envs=W, agents=K) has W worlds with K agents each. Every per-agent argument or
result (step, reset, state, inventory, render, set_pose, region, block, task, ...) takes the agent index
i = w * K + k and has N = W * K rows. World-level calls (save_state, load_state, chests, set_task, ...)
take a world index.

Action modes (PROTOCOL 12.3 / src/README_v7.md):
  action_mode="v7" (default): step(actions int[N, 4]) with columns [primary, craft_recipe, item_select, msg].
      PRIMARY_NAMES, RECIPE names, ITEMS. obs width VEC_DIM with VEC_NAMES.
  action_mode="v6": step(actions int[N]) with the 28 v6 ids in ACTIONS; obs width VEC_DIM_V6 (102) with
      VEC_NAMES_V6; inventory/goal over the first 15 items (ITEMS_V6), the v6 blocks (BLOCKS_V6).
"""
import os
import struct
import sys
import zlib

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_build = os.path.join(_here, "build")
if _build not in sys.path:
    sys.path.insert(0, _build)
try:
    import _craftsim as _native
except ImportError as e:
    raise ImportError(f"native module _craftsim not found in {_build}; run ./build.sh") from e

_REG = _native.registry()
ITEMS = [it["name"] for it in _REG["items"]]
BLOCKS = [b["name"] for b in _REG["blocks"]]
RECIPES = _REG["recipes"]                      # dicts: name, result (item id), count, needs, ingredients {id: n}
MOBS = _REG["mobs"]
TASKS = _REG["tasks"]                          # dicts: name, family, max_steps, success_type, success_a, success_b
TASK_NAMES = [t["name"] for t in TASKS]
FAMILIES = list(_REG["families"])
RECIPE_NAMES = [r["name"] for r in RECIPES]
MOB_NAMES = [m["name"] for m in MOBS]

# v6 names (action_mode "v6"). ITEMS_V6 / BLOCKS_V6 are the first 15 / 20 ids of the registry.
ACTIONS = [
    "noop", "forward", "back", "left", "right", "jump", "forward_jump",
    "turn_left", "turn_right", "look_up", "look_down",
    "attack", "place_block", "place_table", "place_furnace",
    "craft_planks", "craft_sticks", "craft_table", "craft_wooden_pickaxe",
    "craft_stone_pickaxe", "craft_furnace", "smelt_iron", "craft_iron_pickaxe",
    "craft_chest", "place_chest", "deposit", "withdraw", "give",
]
PRIMARY_NAMES = [
    "noop", "forward", "back", "left", "right", "jump", "forward_jump", "sneak", "sprint_forward",
    "turn_left", "turn_right", "look_up", "look_down", "attack", "use", "craft", "smelt",
    "deposit", "withdraw", "give", "drop",
]
ITEMS_V6 = ITEMS[:15]
BLOCKS_V6 = BLOCKS[:20]
MODE_V6, MODE_V7 = 6, 7

NUM_ITEMS = _native.NUM_ITEMS
NUM_BLOCKS = _native.NUM_BLOCKS
NUM_RECIPES = _native.NUM_RECIPES
NUM_MOBS = _native.NUM_MOBS
NUM_TASKS = _native.NUM_TASKS
NUM_FAMILIES = _native.NUM_FAMILIES
NUM_PRIMARY = _native.NUM_PRIMARY
NUM_ACTION_COLS = _native.NUM_ACTION_COLS
NUM_MSGS = _native.NUM_MSGS
MAX_AGENTS = _native.MAX_AGENTS
NUM_ITEMS_V6 = _native.V6_NUM_ITEMS
NUM_ACTIONS = len(ACTIONS)                     # v6 action count (28); v7 uses NUM_PRIMARY x recipes x items x msg
NUM_BLOCKS_V6 = _native.V6_NUM_BLOCKS
VEC_DIM = _native.VEC_DIM                      # v7 obs width (default)
VEC_DIM_V6 = _native.VEC_DIM_V6                # 102
WORLD_SIZE = tuple(_native.WORLD_SIZE)         # (sx, sy, sz)
SEA_LEVEL = _native.SEA_LEVEL
GOAL_OFFSET = _native.GOAL_OFFSET              # v6 obs: goal one-hot
TEAM_OFFSET = GOAL_OFFSET + NUM_ITEMS_V6      # v6 obs: 4 teammate slots x 12
CHEST_OFFSET = TEAM_OFFSET + 4 * 12          # v6 obs: nearest chest [present, dx, dy, dz]
XHAIR_OFFSET = CHEST_OFFSET + 4              # v6 obs: crosshair chest contents, items 0..8
AGENT_BLOCK = BLOCKS.index("agent")            # pseudo block seen when looking at a teammate
CHEST_BLOCK = BLOCKS.index("chest")
assert len(PRIMARY_NAMES) == NUM_PRIMARY and len(ACTIONS) == _native.V6_NUM_ACTIONS
assert NUM_ITEMS == len(ITEMS) and NUM_BLOCKS == len(BLOCKS) and NUM_TASKS == len(TASKS)


def _xhair_items():
    """The 9 most valuable items (registry value, ties: lower id) for the crosshair chest slots."""
    order = sorted(range(NUM_ITEMS), key=lambda i: (-_REG["items"][i]["value"], i))
    return order[:9]


XHAIR_ITEMS = _xhair_items()
V7_INV = _native.V7_INV
V7_SCALARS = _native.V7_SCALARS
V7_TASK = _native.V7_TASK
V7_TASK_PARAM = _native.V7_TASK_PARAM
V7_TEAM = _native.V7_TEAM
V7_CHEST = _native.V7_CHEST
V7_XHAIR = _native.V7_XHAIR
V7_MOBS = _native.V7_MOBS
V7_BUILD = _native.V7_BUILD


def _vec_names_v6():
    names = [f"inv_{it}" for it in ITEMS_V6]
    names += ["health", "on_ground", "pitch", "sin_yaw", "cos_yaw", "mining_progress", "height",
              "near_table", "near_furnace", "fuel", "time_frac"]
    names += [f"goal_{it}" for it in ITEMS_V6]
    for j in range(4):
        names += [f"tm{j}_present", f"tm{j}_dx", f"tm{j}_dy", f"tm{j}_dz"]
        names += [f"tm{j}_msg{m}" for m in range(NUM_MSGS)]
    names += ["chest_present", "chest_dx", "chest_dy", "chest_dz"]
    names += [f"xhair_chest_{it}" for it in ITEMS_V6[:9]]
    return names


def _vec_names_v7():
    names = [f"inv_{it}" for it in ITEMS]
    names += ["health", "hunger", "saturation", "on_ground", "pitch", "sin_yaw", "cos_yaw", "height",
              "in_water", "light_level", "time_sin", "time_cos", "dimension", "armor", "mining_progress",
              "fuel", "steps_left", "sneak", "held_index", "speed"]
    names += [f"task_{t}" for t in TASK_NAMES]
    names += ["task_target_item", "task_target_count"]
    for j in range(4):
        names += [f"tm{j}_present", f"tm{j}_dx", f"tm{j}_dy", f"tm{j}_dz"]
        names += [f"tm{j}_msg{m}" for m in range(NUM_MSGS)]
    names += ["chest_present", "chest_dx", "chest_dy", "chest_dz"]
    names += [f"xhair_chest_{ITEMS[i]}" for i in XHAIR_ITEMS]
    for j in range(3):
        names += [f"mob{j}_present", f"mob{j}_dx", f"mob{j}_dy", f"mob{j}_dz", f"mob{j}_hostile",
                  f"mob{j}_health"]
    for c in range(8):
        names += [f"build{c}_dx", f"build{c}_dy", f"build{c}_dz", f"build{c}_item"]
    names += ["build_progress"]
    return names


VEC_NAMES_V6 = _vec_names_v6()
VEC_NAMES = _vec_names_v7()
assert len(VEC_NAMES_V6) == VEC_DIM_V6 and len(set(VEC_NAMES_V6)) == VEC_DIM_V6
assert len(VEC_NAMES) == VEC_DIM and len(set(VEC_NAMES)) == VEC_DIM, (len(VEC_NAMES), VEC_DIM)


def _item(x):
    return ITEMS.index(x) if isinstance(x, str) else int(x)


def _block(x):
    return BLOCKS.index(x) if isinstance(x, str) else int(x)


class CraftSim:
    """W worlds x K agents stepped in parallel (OpenMP over worlds). Done worlds auto-reset.

    action_mode "v7" (default): step(actions int[N,4], msgs=None). "v6": step(actions int[N]).
    Obs: img uint8 [N, res, res, 2] (block id hit by ray, depth) + vec float32 [N, vec_dim].
    N = num_envs * agents. threads=0 -> OpenMP default thread count.
    """

    def __init__(self, num_envs=16, seed=0, res=64, max_steps=3000, ticks=4, threads=0, agents=1,
                 action_mode="v7"):
        if action_mode not in ("v6", "v7"):
            raise ValueError("action_mode must be 'v6' or 'v7'")
        self.action_mode = action_mode
        self.mode = MODE_V6 if action_mode == "v6" else MODE_V7
        self.num_envs = num_envs          # worlds W
        self.agents = agents              # agents per world K
        self.n = num_envs * agents        # agent rows N (kept for the per-agent API)
        self.res = res
        self.cols = 1 if self.mode == MODE_V6 else NUM_ACTION_COLS
        self.vec_dim = VEC_DIM_V6 if self.mode == MODE_V6 else VEC_DIM
        self.vec_names = VEC_NAMES_V6 if self.mode == MODE_V6 else VEC_NAMES
        self._env = _native.VecEnv(num_envs, agents, seed, res, max_steps, ticks, threads, self.mode)
        self._vec = None   # last obs vector returned to the caller (set_goal rewrites its one-hot)

    @property
    def num_agents(self):
        return self.n

    @property
    def items(self):
        return ITEMS_V6 if self.mode == MODE_V6 else ITEMS

    def _world_rows(self, i):
        """Agent-row slice of the world that owns agent i."""
        w = i // self.agents
        return slice(w * self.agents, (w + 1) * self.agents)

    def reset(self):
        img, vec = self._env.reset()
        self._vec = vec
        return img, vec

    def step(self, actions, msgs=None):
        """v7: actions int[N, 4] (or flat N*4) = [primary, craft_recipe, item_select, msg]; msgs is then
        ignored unless given (overrides column 3). v6: actions int[N] (ids in ACTIONS); msgs int[N].
        Returns (img, vec, rew, done, ach); done is per agent and equal within a world; ach = team mask."""
        a = np.asarray(actions, dtype=np.int32).reshape(self.n, self.cols).reshape(-1)
        m = None if msgs is None else np.asarray(msgs, dtype=np.int32).reshape(-1)
        if m is not None and m.shape[0] != self.n:
            raise ValueError("msgs must have num_agents entries")
        img, vec, rew, done, ach = self._env.step(a, m)
        self._vec = vec
        return img, vec, rew, done, ach

    def last_msgs(self):
        """int32[N]: the message token each agent sent on its last step."""
        return self._env.last_msgs()

    def _sync_rows(self, start, vec):
        """Keep the cached last-obs rows [start, start+len) in sync with fresh obs."""
        if self._vec is not None:
            self._vec[start:start + len(vec)] = vec

    def reset_env(self, i, seed):
        """Fresh world from `seed` for the world of agent i (start kit applied); updates the cached
        obs rows of that world's agents."""
        _img, vec = self._env.reset_env(int(i), int(seed))
        rows = self._world_rows(i)
        self._sync_rows(rows.start, vec)

    def set_start_kits(self, table, prob):
        """table: float [goals, items] counts (NUM_ITEMS x NUM_ITEMS, or the v6 15 x 15 prefix, zero padded);
        prob: float [goals]. At each reset, if the sampled goal g is drawn with prob[g], every agent of the
        world gets table[g, item] of each item (table[g, g] is ignored). Kit items pay no shaping."""
        t = np.zeros((NUM_ITEMS, NUM_ITEMS), np.float32)
        src = np.asarray(table, dtype=np.float32)
        if src.ndim != 2 or src.shape[0] > NUM_ITEMS or src.shape[1] > NUM_ITEMS:
            raise ValueError(f"table must be at most {(NUM_ITEMS, NUM_ITEMS)}, got {src.shape}")
        t[:src.shape[0], :src.shape[1]] = src
        p = np.zeros(NUM_ITEMS, np.float32)
        pp = np.asarray(prob, dtype=np.float32).reshape(-1)
        if pp.shape[0] > NUM_ITEMS:
            raise ValueError(f"prob must have at most {NUM_ITEMS} entries, got {pp.shape[0]}")
        p[:pp.shape[0]] = pp
        self._env.set_start_kits(np.ascontiguousarray(t), np.ascontiguousarray(p))

    def set_novelty(self, scale):
        """Curiosity bonus scale (default 0 = off). Included in rew; see last_novelty()."""
        self._env.set_novelty(float(scale))

    def last_novelty(self):
        """float32[N]: the novelty part of each agent's last step (already scaled)."""
        return self._env.last_novelty()

    def set_team_share(self, share):
        """Fraction of each team achievement / goal reward paid to every agent of the world
        (the rest goes to the agent who obtained it). Default 0.5."""
        self._env.set_team_share(float(share))

    @property
    def team_share(self):
        return self._env.team_share

    # ---- v7 tasks

    def set_task_weights(self, w, weights):
        """Per-world task sampling weights (length NUM_TASKS, non-negative, sum > 0). Used at every later
        episode start of world w (v7 only)."""
        v = np.asarray(weights, dtype=np.float32).reshape(-1)
        if v.shape[0] != NUM_TASKS:
            raise ValueError(f"expected {NUM_TASKS} task weights, got {v.shape[0]}")
        self._env.set_task_weights(int(w), np.ascontiguousarray(v))

    def set_task(self, w, task, target=None, count=None):
        """Set the task of world w now (resets its episode statistics; v7 only). task: TASK_NAMES entry or
        index. target / count override the success predicate's parameters: target is an item name (has_item,
        crafted), block (mined, placed), mob (killed), or a raw int for other predicates; count is the count
        (or steps / distance / n for the other predicates). None keeps the task's own value."""
        idx = TASK_NAMES.index(task) if isinstance(task, str) else int(task)
        t = TASKS[idx]
        p0, p1 = -1, -1
        if target is not None:
            st = t["success_type"]
            if isinstance(target, str):
                ty = _TASK_TYPE_NAMES[st]
                if ty in ("has_item", "crafted"):
                    target = ITEMS.index(target)
                elif ty in ("mined", "placed"):
                    target = BLOCKS.index(target)
                elif ty == "killed":
                    target = MOB_NAMES.index(target)
                else:
                    raise ValueError(f"target name not valid for predicate {ty}")
            p0 = int(target)
        if count is not None:
            p1 = int(count)
        self._env.set_task(int(w), idx, p0, p1)

    def task(self, i=0):
        """Task index of the world of agent i (TASK_NAMES[task])."""
        return int(self._env.task(i))

    def set_shaping_table(self, task, items):
        """Knowledge shaping for a task: while it is active, the first time an agent obtains one of these
        items (names or indices) it gets 0.05 * value. Replaces the previous table of that task."""
        idx = TASK_NAMES.index(task) if isinstance(task, str) else int(task)
        arr = np.asarray([_item(x) for x in items], dtype=np.int32)
        self._env.set_shaping_table(idx, arr)

    def time_of_day(self, w=0):
        """0..1 time of day of world w (0 until the survival module fills it)."""
        return float(self._env.time_of_day(int(w)))

    def world_seed(self, w):
        """Seed of world w's current generation (every reset/auto-reset draws a new one, never reused)."""
        return int(self._env.world_seed(int(w)))

    def build_target(self, w=0):
        """World w's build target: {"blueprint": index or -1, "rot": 0..3, "cells": [(x, y, z, block, item)]}
        (block 0 = cell that must stay air). Placed on the first obs of a build episode."""
        return dict(self._env.build_target(int(w)))

    def entities(self, w=0):
        """Mobs alive in world w (empty until the mobs module fills it)."""
        return list(self._env.entities(int(w)))

    def set_goal_weights(self, weights):
        """v6 goal sampling weights over the items (the first 15 names, or NUM_ITEMS values; zero padded).
        Applies to every later world generation (reset, auto-reset, reset_env)."""
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.shape[0] > NUM_ITEMS:
            raise ValueError(f"expected at most {NUM_ITEMS} weights, got {w.shape[0]}")
        full = np.zeros(NUM_ITEMS, np.float32)
        full[:w.shape[0]] = w
        self._env.set_goal_weights(full)

    def goal(self, i=0):
        """Goal item index (ITEMS index) of the world of agent i (v6)."""
        return int(self._env.goal(i))

    def set_goal(self, i, item):
        """Change the goal of agent i's world (item name or index) and rewrite the obs one-hot of all its
        agents (v6 obs layout only)."""
        idx = _item(item)
        self._env.set_goal(i, idx)
        if self._vec is not None and self.mode == MODE_V6:
            rows = self._vec[self._world_rows(i)]
            rows[:, GOAL_OFFSET:GOAL_OFFSET + NUM_ITEMS_V6] = 0.0
            if idx < NUM_ITEMS_V6:
                rows[:, GOAL_OFFSET + idx] = 1.0

    def render(self, i=0, width=320, height=240):
        return self._env.render(i, width, height)

    def inventory(self, i=0):
        counts = self._env.inventory(i)
        return {name: int(c) for name, c in zip(ITEMS, counts) if c}

    def set_inventory(self, i, item, count):
        self._env.set_inventory(i, _item(item), int(count))

    def chests(self, w):
        """Chests of world w: list of {"x","y","z","items": {name: count}} (non-zero items only)."""
        out = []
        for x, y, z, counts in self._env.chests(int(w)):
            items = {name: int(c) for name, c in zip(ITEMS, counts) if c}
            out.append({"x": x, "y": y, "z": z, "items": items})
        return out

    def state(self, i=0):
        s = self._env.state(i)
        target = None
        if s.has_target:
            target = {"x": s.tx, "y": s.ty, "z": s.tz, "block": BLOCKS[s.tblock]}
        k = i % self.agents
        w = i // self.agents
        team = [w * self.agents + j for j in range(self.agents) if j != k]
        return {
            "x": s.x, "y": s.y, "z": s.z, "vx": s.vx, "vy": s.vy, "vz": s.vz,
            "yaw": s.yaw, "pitch": s.pitch, "on_ground": float(s.on_ground),
            "health": float(s.health), "t": float(s.t), "ticks": s.ticks,
            "seed": s.seed, "episode": s.episode, "last_action": s.last_action,
            "mining_progress": s.mining_progress, "goal": s.goal, "kit": bool(s.kit),
            "restart": bool(s.restart), "target": target,
            "world": w, "agent": k, "msg": s.msg, "team": team,
            "held": None if s.held < 0 else ITEMS[s.held], "dim": s.dim,
            "task": TASK_NAMES[s.task], "task_max_steps": s.task_max_t, "sneak": bool(s.sneak),
        }

    def set_pose(self, i, yaw, pitch):
        self._env.set_pose(i, float(yaw), float(pitch))

    def block(self, i, x, y, z):
        """Block name at world coordinates of agent i's world (agents are never stored)."""
        return BLOCKS[self._env.block(i, x, y, z)]

    def region(self, i, radius=16):
        """Full world height, XZ window of +-radius around agent i, clipped to the world.
        Returns ((ox, oy, oz), uint8[sy, sz, sx]); index [y - oy][z - oz][x - ox] = block id."""
        origin, arr = self._env.region(i, radius)
        return tuple(origin), arr

    def save_state(self, w):
        """Serialize world w completely (blocks, chests, all its agents, task, counters, rng).
        w is a world index. Returns bytes."""
        return self._env.save_state(int(w))

    def load_state(self, w, blob, goal=None):
        """Restore world w from a save_state blob. goal (v6: ITEMS index or name) overrides the saved goal.
        The episode restarts with t = 0 and state(i)['restart'] True until the world auto-resets. Returns
        (img, vec) for the world's K agents and updates the cached obs rows."""
        if isinstance(goal, str):
            goal = _item(goal)
        img, vec = self._env.load_state(int(w), bytes(blob), -1 if goal is None else int(goal))
        self._sync_rows(w * self.agents, vec)
        return img, vec

    def close(self):
        self._env = None


_TASK_TYPE_NAMES = [
    "has_item", "crafted", "mined", "killed", "ate", "slept", "survive_night", "placed",
    "build_complete", "portal_lit", "in_dimension", "reached_y", "distance_travelled", "health_above",
    "near_teammate", "look_at_teammate", "crouch_greet", "follow_teammate", "gave_item", "shared_chest",
    "msg_near_teammate",
]


def achievements(mask):
    return [ITEMS[i] for i in range(min(31, NUM_ITEMS)) if mask >> i & 1]


def save_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
