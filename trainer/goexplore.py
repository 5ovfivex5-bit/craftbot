"""Go-Explore archive (Ecoffet et al. 2019, arXiv 1901.10995) over CraftSim env snapshots.

Cell = (goal, achievement mask, depth bucket). The cell is the coarse "where and what I have
achieved" key; the archive keeps one snapshot (sim save_state blob) per cell, the one reached in the
fewest steps. Restarts pick a cell with weight (1 + tech rank) / sqrt(times chosen + 1), so rarely
chosen, late-tech-tree cells are favored, and set the restarted episode's goal to the next item up
the tech tree that the cell does not have yet.
"""
import math

import numpy as np

from .curriculum import GOAL_ORDER
from .envs import ITEMS

ACH_MASK = (1 << 14) - 1
MAX_BLOB_BYTES = 300 * 1024      # PROTOCOL section 5: snapshots are < 300 KB


def depth_bucket(vec_height):
    """Bucket of the player's height above sea level (vec[20] = (y - SEA_LEVEL) / 32), 16-block bins."""
    return int(math.floor(float(vec_height) * 2.0))


def tech_rank(mask):
    """Index in GOAL_ORDER of the furthest tech-tree item obtained, +1 (0 if none)."""
    best = 0
    for k, name in enumerate(GOAL_ORDER):
        if (mask >> ITEMS.index(name)) & 1:
            best = k + 1
    return best


class Archive:
    def __init__(self, max_cells=2000, seed=0):
        self.max_cells = int(max_cells)
        self.cells = {}          # key -> {"blob", "steps", "goal", "mask", "chosen"}
        self.bytes = 0
        self.rng = np.random.default_rng(seed)
        self.dropped = 0         # snapshots refused (archive full / blob too large)

    def __len__(self):
        return len(self.cells)

    @staticmethod
    def key(goal, mask, bucket):
        return (int(goal), int(mask) & ACH_MASK, int(bucket))

    def wants(self, key, steps):
        """Cheap pre-check before a snapshot is taken: new cell (room left) or a faster arrival."""
        c = self.cells.get(key)
        if c is None:
            return len(self.cells) < self.max_cells
        return steps < c["steps"]

    def offer(self, key, blob, steps):
        if blob is None or len(blob) > MAX_BLOB_BYTES:
            self.dropped += 1
            return False
        c = self.cells.get(key)
        if c is None:
            if len(self.cells) >= self.max_cells:
                self.dropped += 1
                return False
            self.cells[key] = {"blob": blob, "steps": int(steps), "goal": key[0], "mask": key[1],
                               "chosen": 0}
            self.bytes += len(blob)
            return True
        if steps < c["steps"]:
            self.bytes += len(blob) - len(c["blob"])
            c["blob"], c["steps"] = blob, int(steps)
            return True
        return False

    def sample(self, unlocked_goals):
        """Pick a cell and the goal for the restarted episode. Returns (blob, goal, mask) or None."""
        if not self.cells:
            return None
        unlocked = set(ITEMS.index(g) for g in unlocked_goals)
        keys, weights, goals = [], [], []
        for k, c in self.cells.items():
            goal = next_goal(c["mask"], unlocked_goals)
            if goal is None:
                continue
            keys.append(k)
            goals.append(goal)
            weights.append((1.0 + tech_rank(c["mask"])) / math.sqrt(c["chosen"] + 1.0))
        if not keys:
            return None
        w = np.asarray(weights, np.float64)
        j = int(self.rng.choice(len(keys), p=w / w.sum()))
        c = self.cells[keys[j]]
        c["chosen"] += 1
        return c["blob"], goals[j], c["mask"]


def next_goal(mask, unlocked_goals):
    """First unlocked goal (GOAL_ORDER order) whose item is not in the achievement mask."""
    for name in unlocked_goals:
        if name == "dirt":
            continue
        if not (mask >> ITEMS.index(name)) & 1:
            return ITEMS.index(name)
    return None
