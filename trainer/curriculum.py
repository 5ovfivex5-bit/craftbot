"""Goal curriculum over the crafting tech tree (v2 goal-conditioned sim, v3 start kits).

Each episode has one goal item (sampled from `CraftSim.set_goal_weights`). The curriculum tracks a
per-goal success EMA (success = ach bit 31, goal achieved) and reshapes the goal distribution at
every rollout:

- only the first `unlocked` goals of GOAL_ORDER are sampled (locked goals get weight 0);
- an unlocked goal gets `floor + zone(success)`, where zone is 1.0 inside the learning zone
  [lo, hi], `low_weight` below it (too hard so far) and `high_weight` above it (mastered);
- the frontier (highest unlocked goal, i.e. the next step up the tech tree) gets `frontier_bonus`
  extra weight;
- the next goal unlocks when the frontier's success EMA reaches `unlock_at` (after at least
  `min_episodes` episodes of that goal), or when the frontier's KIT success EMA reaches
  `KIT_UNLOCK_AT` (see below).

Start kits (v3): for goals from `cobblestone` on, a fraction of episodes start with a kit of
goal-relevant items (KIT_SPEC; never the goal item). The kit probability per goal is
`kit_prob_for(no-kit success)`: 0.5 while the no-kit success is <= 0.2, annealed linearly to 0.1 at
>= 0.8 (and 0.5 if the goal was never tried). Kit episodes are recorded in separate EMAs
(`ema_kit`, `n_kit`) and never touch the no-kit EMA that drives weights and unlocking.

Unlock rule, documented choice: the frontier unlocks when EITHER its no-kit EMA >= `unlock_at`
(the normal condition, measured without help) OR its kit EMA >= KIT_UNLOCK_AT (0.5). The kit branch
lets the agent move on to the next skill while the long chain (e.g. iron_pickaxe needs iron mining
+ smelting) is still too hard from scratch; the next goal is then learned with its own kit.
"""
import math

import numpy as np

from .envs import ITEMS, NUM_ITEMS, V7

GOAL_ORDER = [
    "log", "planks", "stick", "crafting_table", "wooden_pickaxe", "cobblestone",
    "stone_pickaxe", "furnace", "coal", "iron_ore", "iron_ingot", "iron_pickaxe", "diamond",
]
# episode kinds (what the episode started from)
EP_SCRATCH, EP_KIT, EP_RESTART = 0, 1, 2
EMA_MIN_ALPHA = 0.02  # EMA floor: a goal's success is averaged over ~50 episodes once warm

# Start kits per goal (item -> count). Built from the tech tree in src/craftsim.cpp: planks -> sticks
# (2 planks -> 4 sticks), 4 planks -> table, table + 8 cobblestone -> furnace, table + 3 cobblestone
# + 2 sticks -> stone pickaxe, table + 3 iron ingots + 2 sticks -> iron pickaxe, furnace + ore + fuel
# -> ingot. Goals before cobblestone never get a kit.
KIT_SPEC = {
    "cobblestone": {"wooden_pickaxe": 1},
    "stone_pickaxe": {"wooden_pickaxe": 1, "stick": 2, "planks": 4},
    "furnace": {"wooden_pickaxe": 1, "cobblestone": 4, "planks": 4},
    "coal": {"stone_pickaxe": 1, "stick": 2, "planks": 4},
    "iron_ore": {"stone_pickaxe": 1, "stick": 2, "planks": 4},
    "iron_ingot": {"stone_pickaxe": 1, "furnace": 1, "coal": 2},
    "iron_pickaxe": {"stone_pickaxe": 1, "stick": 2, "furnace": 1, "coal": 2, "planks": 4},
    "diamond": {"iron_pickaxe": 1},
}
KIT_P_HI = 0.5          # kit probability while the goal is hard (no-kit success <= KIT_SUCCESS_LO)
KIT_P_LO = 0.1          # kit probability once the goal is mostly solved (no-kit success >= KIT_SUCCESS_HI)
KIT_SUCCESS_LO = 0.2
KIT_SUCCESS_HI = 0.8
KIT_UNLOCK_AT = 0.5     # frontier kit success EMA that unlocks the next goal


def build_kit_table():
    """float32[NUM_ITEMS goals, NUM_ITEMS items]: items given to an episode with that goal.
    The v2-v6 goal names: under the v7 obs (no goal curriculum) the table is empty."""
    tab = np.zeros((NUM_ITEMS, NUM_ITEMS), np.float32)
    if V7:
        return tab
    for goal, kit in KIT_SPEC.items():
        g = ITEMS.index(goal)
        for item, n in kit.items():
            if item == goal:
                raise ValueError(f"kit for {goal} contains the goal item")
            tab[g, ITEMS.index(item)] = float(n)
    return tab


KIT_TABLE = build_kit_table()


def kit_prob_for(success):
    """Kit probability for a goal given its no-kit success EMA (None/NaN = never tried)."""
    if success is None or math.isnan(success):
        return KIT_P_HI
    t = min(1.0, max(0.0, (success - KIT_SUCCESS_LO) / (KIT_SUCCESS_HI - KIT_SUCCESS_LO)))
    return KIT_P_HI + (KIT_P_LO - KIT_P_HI) * t


class GoalCurriculum:
    def __init__(self, order=GOAL_ORDER, start_unlocked=2, unlock_at=0.5, min_episodes=30,
                 zone=(0.1, 0.9), floor=0.02, low_weight=0.5, high_weight=0.25,
                 frontier_bonus=2.0):
        missing = [g for g in order if g not in ITEMS]
        if missing:
            raise ValueError(f"curriculum goals not in ITEMS: {missing}")
        self.order = list(order)
        self.order_idx = [ITEMS.index(g) for g in self.order]
        self.start_unlocked = max(1, min(int(start_unlocked), len(self.order)))
        self.unlock_at = float(unlock_at)
        self.min_episodes = int(min_episodes)
        self.zone = (float(zone[0]), float(zone[1]))
        self.floor = float(floor)
        self.low_weight = float(low_weight)
        self.high_weight = float(high_weight)
        self.frontier_bonus = float(frontier_bonus)
        self.unlocked = self.start_unlocked
        # no-kit episodes (drive weights, unlocking, goal_success)
        self.ema = np.full(NUM_ITEMS, np.nan)
        self.n = np.zeros(NUM_ITEMS, np.int64)
        # kit episodes (goal_success_kit, kit unlock branch)
        self.ema_kit = np.full(NUM_ITEMS, np.nan)
        self.n_kit = np.zeros(NUM_ITEMS, np.int64)
        # restart episodes (Go-Explore archive restarts): goal_success_restart only
        self.ema_rs = np.full(NUM_ITEMS, np.nan)
        self.n_rs = np.zeros(NUM_ITEMS, np.int64)
        # Episodes are only observed when they end. Failures end at the timeout, successes early,
        # so early estimates are biased toward success. Recording is armed once every env has lived
        # through two full episode windows (`arm`), after which finished-episode rates are unbiased.
        self.armed_at = None

    # ----- bookkeeping -----
    def arm(self, env_steps, decisions_per_env):
        """Start recording at `env_steps` + two episode windows (no-op once armed)."""
        if self.armed_at is None:
            self.armed_at = int(env_steps) + 2 * int(decisions_per_env)

    def is_armed(self, env_steps):
        return self.armed_at is not None and env_steps >= self.armed_at

    def record(self, item, success, kind=EP_SCRATCH):
        """One finished episode whose goal was `item` (ITEMS index); success = goal achieved.
        `kind`: EP_SCRATCH (drives weights/unlocks), EP_KIT or EP_RESTART (separate EMAs only)."""
        i = int(item)
        x = 1.0 if success else 0.0
        if kind == EP_KIT:
            self.n_kit[i] += 1
            self.ema_kit[i] = _ema(self.ema_kit[i], x, self.n_kit[i])
        elif kind == EP_RESTART:
            self.n_rs[i] += 1
            self.ema_rs[i] = _ema(self.ema_rs[i], x, self.n_rs[i])
        else:
            self.n[i] += 1
            self.ema[i] = _ema(self.ema[i], x, self.n[i])

    def rate(self, item):
        """No-kit success EMA of a goal, or None if it has never been an episode goal."""
        i = item if isinstance(item, (int, np.integer)) else ITEMS.index(item)
        v = self.ema[i]
        return None if math.isnan(v) else float(v)

    def maybe_unlock(self):
        """Unlock goals while the frontier is mastered. Returns the names unlocked this call."""
        out = []
        while self.unlocked < len(self.order):
            front = self.order_idx[self.unlocked - 1]
            nokit_ok = self.n[front] >= self.min_episodes and self.ema[front] >= self.unlock_at
            kit_ok = self.n_kit[front] >= self.min_episodes and self.ema_kit[front] >= KIT_UNLOCK_AT
            if nokit_ok or kit_ok:
                self.unlocked += 1
                out.append(self.order[self.unlocked - 1])
            else:
                break
        return out

    @property
    def frontier(self):
        return self.order[self.unlocked - 1]

    def stage(self):
        return f"{self.frontier} ({self.unlocked}/{len(self.order)} unlocked)"

    def goal_success(self):
        """{item: no-kit success EMA} for every goal that has been sampled at least once."""
        return {ITEMS[i]: float(self.ema[i]) for i in range(NUM_ITEMS) if self.n[i] > 0}

    def goal_success_restart(self):
        """{item: success EMA of archive-restart episodes} for goals that had restarts."""
        return {ITEMS[i]: float(self.ema_rs[i]) for i in range(NUM_ITEMS) if self.n_rs[i] > 0}

    def goal_success_kit(self):
        """{item: kit success EMA} for every goal that has had a kit episode."""
        return {ITEMS[i]: float(self.ema_kit[i]) for i in range(NUM_ITEMS) if self.n_kit[i] > 0}

    # ----- start kits -----
    def kit_prob(self):
        """float32[NUM_ITEMS] (indexed by goal item): kit probability per goal for the next episodes."""
        p = np.zeros(NUM_ITEMS, np.float32)
        for goal in KIT_SPEC:
            i = ITEMS.index(goal)
            p[i] = kit_prob_for(None if math.isnan(self.ema[i]) else float(self.ema[i]))
        return p

    @staticmethod
    def kit_table():
        return KIT_TABLE.copy()

    # ----- sampling distribution -----
    def weights(self):
        w = np.zeros(NUM_ITEMS, np.float32)
        for k in range(self.unlocked):
            i = self.order_idx[k]
            s = self.ema[i]
            if math.isnan(s):
                zone_w = 1.0                       # never tried: full weight
            elif s < self.zone[0]:
                zone_w = self.low_weight           # still too hard to learn from
            elif s > self.zone[1]:
                zone_w = self.high_weight          # mastered: keep some practice
            else:
                zone_w = 1.0                       # learning zone
            w[i] = self.floor + zone_w
        w[self.order_idx[self.unlocked - 1]] += self.frontier_bonus
        return w / w.sum()

    # ----- checkpointing -----
    def state_dict(self):
        return {
            "order": list(self.order), "unlocked": int(self.unlocked),
            "ema": [None if math.isnan(v) else float(v) for v in self.ema],
            "n": [int(x) for x in self.n], "armed_at": self.armed_at,
            "ema_kit": [None if math.isnan(v) else float(v) for v in self.ema_kit],
            "n_kit": [int(x) for x in self.n_kit],
            "ema_rs": [None if math.isnan(v) else float(v) for v in self.ema_rs],
            "n_rs": [int(x) for x in self.n_rs],
        }

    def load_state_dict(self, sd):
        if sd.get("order") != self.order:
            raise ValueError("checkpoint curriculum uses a different goal order")
        self.unlocked = int(sd["unlocked"])
        self.ema = np.array([np.nan if v is None else v for v in sd["ema"]], np.float64)
        self.n = np.array(sd["n"], np.int64)
        self.armed_at = sd.get("armed_at")
        # kit fields are absent in v2 checkpoints: default to "no kit episodes seen yet"
        ema_kit = sd.get("ema_kit")
        n_kit = sd.get("n_kit")
        if ema_kit is None or n_kit is None:
            self.ema_kit = np.full(NUM_ITEMS, np.nan)
            self.n_kit = np.zeros(NUM_ITEMS, np.int64)
        else:
            self.ema_kit = np.array([np.nan if v is None else v for v in ema_kit], np.float64)
            self.n_kit = np.array(n_kit, np.int64)
        # restart fields are absent in v2/v3 checkpoints
        ema_rs, n_rs = sd.get("ema_rs"), sd.get("n_rs")
        if ema_rs is None or n_rs is None:
            self.ema_rs = np.full(NUM_ITEMS, np.nan)
            self.n_rs = np.zeros(NUM_ITEMS, np.int64)
        else:
            self.ema_rs = np.array([np.nan if v is None else v for v in ema_rs], np.float64)
            self.n_rs = np.array(n_rs, np.int64)


def _ema(prev, x, n):
    """EMA update with alpha = max(1/n, EMA_MIN_ALPHA); the first sample sets the value."""
    if math.isnan(prev):
        return x
    alpha = max(1.0 / n, EMA_MIN_ALPHA)
    return prev + alpha * (x - prev)
