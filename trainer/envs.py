"""Adapter around craftsim.CraftSim.

Works with both the current ctypes facade and the pybind11 module described in PROTOCOL.md:
optional methods (region, reset_env, state extras, threads=) are feature-detected.

Observation version (process-wide, see trainer/__init__.py): OBS = "v7" (default) or "v6".
  v7: VEC_DIM 295, NUM_ACTIONS = 21 primary actions (craftsim.PRIMARY_NAMES), the sim's action columns
      [primary, recipe, item_select, msg], ITEMS / BLOCKS of the v7 registry, task API (set_task_weights,
      set_shaping_table). The trainer's action arrays are [N, 3] (primary, recipe, item) for v7.
  v6: VEC_DIM 102, NUM_ACTIONS 28 (craftsim.ACTIONS), 15 items, 20 blocks, goal API. Actions are [N].
"""
import inspect
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import craftsim  # noqa: E402

OBS = os.environ.get("MINEMIND_OBS", "v7")
if OBS not in ("v6", "v7"):
    raise ValueError(f"MINEMIND_OBS must be 'v6' or 'v7', got {OBS!r}")
V7 = OBS == "v7"

# v6 defaults (PROTOCOL 6.10): 64x64 rays. Runs record their res in the config; this is the default.
RES = 64
NUM_MSGS = craftsim.NUM_MSGS
MAX_AGENTS = int(getattr(craftsim, "MAX_AGENTS", 5))
HAS_REGION = hasattr(craftsim.CraftSim, "region")
HAS_RESET_ENV = hasattr(craftsim.CraftSim, "reset_env")
# v3 sim API (PROTOCOL section 4): start kits and the novelty (curiosity) bonus. Optional here so
# the trainer still runs against a sim without them (kits off, novelty 0).
HAS_KITS = hasattr(craftsim.CraftSim, "set_start_kits")
HAS_STATE = hasattr(craftsim.CraftSim, "save_state") and hasattr(craftsim.CraftSim, "load_state")
HAS_NOVELTY = hasattr(craftsim.CraftSim, "set_novelty") and hasattr(craftsim.CraftSim, "last_novelty")
TEAM_CAPABLE = hasattr(craftsim.CraftSim, "last_msgs")
_INIT_PARAMS = set(inspect.signature(craftsim.CraftSim.__init__).parameters)

if V7:
    ACTION_MODE = "v7"
    ACTIONS = list(craftsim.PRIMARY_NAMES)          # primary action names (column 0)
    NUM_ACTIONS = craftsim.NUM_PRIMARY              # 21: the primary head
    ITEMS = list(craftsim.ITEMS)
    BLOCKS = list(craftsim.BLOCKS)
    NUM_ITEMS = craftsim.NUM_ITEMS                  # 97
    NUM_BLOCKS = craftsim.NUM_BLOCKS                # 59 (incl. mob pseudo blocks)
    VEC_DIM = craftsim.VEC_DIM                      # 295
    VEC_NAMES = list(craftsim.VEC_NAMES)
    GOAL_OFFSET = None                              # v7 obs has no goal one-hot (tasks instead)
    NUM_RECIPES = craftsim.NUM_RECIPES              # 65 recipes; the recipe column is 0 (none) or r + 1
    NUM_RECIPE_CLASSES = NUM_RECIPES + 1            # 66: recipe head
    NUM_ITEM_CLASSES = NUM_ITEMS + 1                # 98: item head (0 keep, i + 1 select item i)
    NUM_TASKS = craftsim.NUM_TASKS
    OBS_VERSION = 4
else:
    ACTION_MODE = "v6"
    ACTIONS = list(craftsim.ACTIONS)
    NUM_ACTIONS = len(ACTIONS)                      # 28
    # v6 item ids keep their v6 names in the trainer (curriculum, eval); the v7 registry renamed three of them
    _V6_LEGACY = {"oak_log": "log", "oak_planks": "planks", "raw_iron": "iron_ore"}
    ITEMS = [_V6_LEGACY.get(n, n) for n in craftsim.ITEMS_V6]
    BLOCKS = list(craftsim.BLOCKS_V6)
    NUM_ITEMS = craftsim.NUM_ITEMS_V6               # 15
    NUM_BLOCKS = craftsim.NUM_BLOCKS_V6             # 20
    VEC_DIM = craftsim.VEC_DIM_V6                   # 102
    VEC_NAMES = list(craftsim.VEC_NAMES_V6)
    GOAL_OFFSET = craftsim.GOAL_OFFSET              # v6 obs: goal one-hot
    NUM_RECIPES = NUM_RECIPE_CLASSES = NUM_ITEM_CLASSES = 0
    NUM_TASKS = 0
    OBS_VERSION = 3

WORLD_SIZE = tuple(getattr(craftsim, "WORLD_SIZE", (64, 64, 64)))
# v2+ sims expose set_goal_weights / goal; v7 uses tasks, the goal is inert there.
HAS_GOALS = hasattr(craftsim.CraftSim, "set_goal_weights") and hasattr(craftsim.CraftSim, "goal")
HAS_TASKS = V7 and hasattr(craftsim.CraftSim, "set_task_weights")
# Bytes per env in the sim beyond the world grid. Measured by `python -m trainer.bench`:
# sim RSS 197.1 KiB/env vs 192 KiB world grid -> ~8 KiB with margin. Used only for mem_warning.
SIM_OVERHEAD_BYTES = 8 * 1024


def _vec_index(name, default=None):
    """Column of a named vec feature in this obs layout (names are the source of truth)."""
    if name in VEC_NAMES:
        return VEC_NAMES.index(name)
    if default is None:
        raise KeyError(name)
    return default


# Vec columns used by the trainer (v6 height was read at index 20 in older code, which is
# mining_progress in VEC_NAMES_V6; looked up by name now).
HEIGHT_IDX = _vec_index("height")
INV_SLICE = slice(0, NUM_ITEMS) if V7 else None   # v7: log2(1+n)/7 per item


def obs_signature(res=RES):
    """Everything a policy's input/output layout depends on. Stored in checkpoints; a checkpoint
    whose signature differs from the running sim cannot be trained on. `res` is the run's image
    size; it is not compared by obs_mismatch (the CNN weights themselves carry it).
    The v7 signature adds the obs name and the head sizes; the v6 signature is unchanged."""
    sig = {
        "obs_version": OBS_VERSION, "res": int(res), "vec_dim": VEC_DIM, "num_blocks": NUM_BLOCKS,
        "num_actions": NUM_ACTIONS, "num_items": NUM_ITEMS, "world_size": list(WORLD_SIZE),
        "goal_offset": GOAL_OFFSET, "num_msgs": NUM_MSGS,
    }
    if V7:
        sig.update({"obs": "v7", "num_recipe_classes": NUM_RECIPE_CLASSES,
                    "num_item_classes": NUM_ITEM_CLASSES, "num_tasks": NUM_TASKS})
    return sig


def obs_mismatch(sig):
    """None if `sig` (from a checkpoint) matches this sim, else a human-readable reason."""
    cur = obs_signature()
    if not isinstance(sig, dict):
        return ("checkpoint has no observation signature (written by the v1 trainer, "
                "obs v1: VEC_DIM=25, NUM_BLOCKS=14, no goal one-hot)")
    keys = [k for k in cur if k != "res"]
    diffs = [f"{k}: checkpoint={sig.get(k)!r} sim={cur[k]!r}" for k in keys
             if sig.get(k) != cur[k]]
    if not diffs:
        return None
    hint = ""
    if sig.get("obs_version") == 3 and V7:
        hint = " (a v6 checkpoint: run with --obs v6)"
    elif sig.get("obs_version") == 4 and not V7:
        hint = " (a v7 checkpoint: run with --obs v7)"
    return ("checkpoint observation layout differs from the running sim (" + "; ".join(diffs) + ")" + hint)


def world_bytes():
    sx, sy, sz = WORLD_SIZE
    return sx * sy * sz


def per_env_bytes(rollout_len, gru_hidden=256, agents=1, res=RES):
    """Estimated resident bytes for one WORLD: world grid + sim overhead + rollout buffers of its
    `agents` agent rows."""
    obs = res * res * 2
    per_step = obs + VEC_DIM * 4 + gru_hidden * 4 + 4 * 4 + 1 * 8 + 4 * 4 + 8
    return world_bytes() + SIM_OVERHEAD_BYTES + rollout_len * agents * per_step


def primary_of(a):
    """Primary action column of a policy action array: [N] (v6) or [N, 3] (v7) -> [N]."""
    a = np.asarray(a)
    return a if a.ndim == 1 else a[:, 0]


class SimAdapter:
    """num_envs = worlds W, agents = K agents per world; every per-agent call takes the global
    agent index i = w * K + k and step()/reset() rows are the N = W * K agents.

    v7 actions: step(actions int[N, 3] = (primary, recipe, item)  [or [N, 4] with msg], msgs int[N]);
    v6 actions: step(actions int[N], msgs int[N])."""

    def __init__(self, num_envs, seed=0, threads=0, res=RES, max_steps=3000, agents=1):
        self.worlds = int(num_envs)
        self.agents = int(agents)
        self.n = self.worlds * self.agents      # agent rows
        self.seed = int(seed)
        self.res = res
        kw = dict(num_envs=self.worlds, seed=self.seed, res=res, max_steps=max_steps)
        if "threads" in _INIT_PARAMS:
            kw["threads"] = int(threads)
        if "agents" in _INIT_PARAMS:
            kw["agents"] = self.agents
        elif self.agents != 1:
            raise RuntimeError("craftsim has no team API: agents must be 1")
        if "action_mode" in _INIT_PARAMS:
            kw["action_mode"] = ACTION_MODE
        self.sim = craftsim.CraftSim(**kw)
        self.img = None
        self.vec = None

    def reset(self):
        self.img, self.vec = self.sim.reset()
        return self.img, self.vec

    def set_goal_weights(self, weights):
        if HAS_GOALS and not V7:
            self.sim.set_goal_weights(weights)

    def goal(self, i):
        return int(self.sim.goal(i)) if (HAS_GOALS and not V7) else -1

    def set_start_kits(self, table, prob):
        """v6 only: table float32[NUM_ITEMS goals, NUM_ITEMS items], prob float32[NUM_ITEMS] (by goal).
        v7 has no goal kits (task start kits go through set_inventory, see tasks.py)."""
        if HAS_KITS and not V7:
            self.sim.set_start_kits(np.asarray(table, np.float32), np.asarray(prob, np.float32))
        elif HAS_KITS and V7:
            self.sim.set_start_kits(np.zeros((craftsim.NUM_ITEMS, craftsim.NUM_ITEMS), np.float32),
                                    np.zeros(craftsim.NUM_ITEMS, np.float32))

    def kit(self, i):
        """Whether env i's current episode started with a kit (False when the sim lacks kits)."""
        if not HAS_KITS:
            return False
        return bool(self.sim.state(i).get("kit", False))

    def set_novelty(self, scale):
        if HAS_NOVELTY:
            self.sim.set_novelty(float(scale))

    # ----- v7 tasks -----
    def set_task_weights(self, w, weights):
        """Task sampling weights of world w (length NUM_TASKS), used by its next auto-reset."""
        if HAS_TASKS:
            self.sim.set_task_weights(int(w), np.asarray(weights, np.float32))

    def task(self, i):
        """Index (TASK_NAMES) of the task of agent i's world (v7)."""
        return int(self.sim.task(i)) if HAS_TASKS else -1

    def set_shaping_table(self, task, items):
        if HAS_TASKS:
            self.sim.set_shaping_table(task, items)

    def set_inventory(self, i, item, count):
        self.sim.set_inventory(int(i), item, int(count))

    def save_state(self, w):
        """Serialized world w (bytes; all its agents), or None when the sim lacks snapshots."""
        if not HAS_STATE:
            return None
        return bytes(self.sim.save_state(w))

    def load_state(self, w, blob, goal=None):
        """Restore world w from a snapshot; returns (img[K,res,res,2], vec[K,VEC_DIM]), or None."""
        if not HAS_STATE:
            return None
        return self.sim.load_state(w, blob, goal)

    def set_team_share(self, share):
        if hasattr(self.sim, "set_team_share"):
            self.sim.set_team_share(float(share))

    def last_msgs(self):
        """int32[N] message token each agent sent on its last step (zeros without the team API)."""
        if not TEAM_CAPABLE:
            return np.zeros(self.n, np.int32)
        return np.asarray(self.sim.last_msgs(), np.int32)

    def chests(self, w):
        """Chests of world w: [{"x","y","z","items"{}}] ([] without the team API)."""
        return self.sim.chests(w) if TEAM_CAPABLE else []

    def last_novelty(self):
        """float32[N] novelty part of the last step, or None when the sim lacks the bonus."""
        if not HAS_NOVELTY:
            return None
        return np.asarray(self.sim.last_novelty(), np.float32)

    def step(self, actions, msgs=None):
        """v7: actions [N, 3] (primary, recipe, item) and msgs [N]; v6: actions [N], msgs [N]."""
        a = np.asarray(actions, np.int32)
        if V7:
            a = a.reshape(self.n, -1)[:, :3]
            m = np.zeros(self.n, np.int32) if msgs is None else np.asarray(msgs, np.int32).reshape(-1)
            packed = np.ascontiguousarray(np.column_stack([a, m]), dtype=np.int32)
            self.img, self.vec, rew, done, ach = self.sim.step(packed)
        elif msgs is not None and TEAM_CAPABLE:
            self.img, self.vec, rew, done, ach = self.sim.step(a, msgs)
        else:
            self.img, self.vec, rew, done, ach = self.sim.step(a)
        return self.img, self.vec, rew, done, ach

    def time_of_day(self, w):
        """0..1 time of day of world w (0.0 when the sim has no day/night yet)."""
        if not V7 or not hasattr(self.sim, "time_of_day"):
            return 0.0
        return float(self.sim.time_of_day(int(w))) % 1.0

    def entities(self, w):
        """[{"type", "x", "y", "z", "health"}] mobs alive in world w ([] until the mobs module fills it)."""
        if not V7 or not hasattr(self.sim, "entities"):
            return []
        out = []
        for e in self.sim.entities(int(w)):
            d = dict(e) if isinstance(e, dict) else {"type": getattr(e, "type", "?"), "x": getattr(e, "x", 0.0),
                                                      "y": getattr(e, "y", 0.0), "z": getattr(e, "z", 0.0),
                                                      "health": getattr(e, "health", 0.0)}
            out.append({"type": str(d.get("type")), "x": float(d.get("x", 0.0)), "y": float(d.get("y", 0.0)),
                        "z": float(d.get("z", 0.0)), "health": float(d.get("health", 0.0))})
        return out

    def state(self, i):
        st = dict(self.sim.state(i))
        st.setdefault("seed", None)
        st.setdefault("episode", None)
        st.setdefault("target", None)
        return st

    def inventory(self, i):
        return self.sim.inventory(i)

    def region(self, i, radius=16):
        """Returns ((ox, oy, oz), uint8[sy, sz, sx]) or None when the sim lacks region()."""
        if not HAS_REGION:
            return None
        return self.sim.region(i, radius)

    def reset_env(self, i, seed):
        if not HAS_RESET_ENV:
            raise RuntimeError("sim has no reset_env(); deterministic per-env seeding unavailable")
        self.sim.reset_env(i, int(seed))

    def close(self):
        close = getattr(self.sim, "close", None)
        if close is not None:
            close()
        self.sim = None
