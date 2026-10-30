"""TaskSampler: assigns each world a task family and task per episode, with curriculum.

Curriculum: a task is unlocked when every prerequisite task has reached
`unlock_ema_threshold` success EMA after >= `min_episodes_before_unlock` episodes.
Tasks with no prerequisites (roots, including all "social" roots) are unlocked from the start.
Within a family, unlocked tasks are picked by `weight`; tasks already `mastered`
(EMA >= mastered_ema_threshold) are down-weighted by `review_weight`.
Families are drawn from the mix restricted to families with at least one unlocked task.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from .loaders import load_envmix, load_tasks
from .schema import FAMILIES, describe as _describe, validate_envmix, validate_tasks


@dataclass(frozen=True)
class Assignment:
    world: int
    family: str
    task: str
    kit: dict = field(default_factory=dict)


class TaskSampler:
    def __init__(self, num_worlds, mix=None, seed=0, tasks=None, envmix=None):
        if num_worlds < 1:
            raise ValueError("num_worlds must be >= 1")
        self.num_worlds = int(num_worlds)
        self.envmix = copy.deepcopy(envmix) if envmix is not None else load_envmix("default")
        if isinstance(mix, str):
            self.envmix_name = mix
            mix = load_envmix(mix)["mix"]
        elif isinstance(mix, dict):
            mix = dict(mix)
        else:
            mix = dict(self.envmix["mix"])
        tasks = load_tasks() if tasks is None else list(tasks)
        errs = validate_tasks(tasks) + validate_envmix(dict(self.envmix, mix=mix))
        if errs:
            raise ValueError("invalid tasks/envmix: " + "; ".join(errs))
        self.tasks = {t["id"]: copy.deepcopy(t) for t in tasks}
        self.mix = {f: float(w) for f, w in mix.items() if float(w) > 0}
        self.by_family = {f: [] for f in FAMILIES}
        for t in tasks:
            self.by_family[t["family"]].append(t["id"])
        self.cur = dict(self.envmix.get("curriculum", {}))
        self.kits = dict(self.envmix.get("kits", {}))
        self.alpha = float(self.cur.get("ema_alpha", 0.05))
        self.unlock_thr = float(self.cur.get("unlock_ema_threshold", 0.6))
        self.master_thr = float(self.cur.get("mastered_ema_threshold", 0.9))
        self.min_eps = int(self.cur.get("min_episodes_before_unlock", 0))
        self.review_weight = float(self.cur.get("review_weight", 0.25))
        self.curriculum_on = bool(self.cur.get("enabled", True))
        self.ema = {tid: 0.0 for tid in self.tasks}
        self.episodes = {tid: 0 for tid in self.tasks}
        self.rng = random.Random(seed)
        self.current = {}

    # ---- curriculum -------------------------------------------------------
    def _mastered_prereq(self, tid):
        return self.episodes[tid] >= self.min_eps and self.ema[tid] >= self.unlock_thr

    def unlocked(self, tid):
        if not self.curriculum_on:
            return True
        return all(self._mastered_prereq(p) for p in self.tasks[tid].get("prerequisites") or [])

    def mastered(self, tid):
        return self.episodes[tid] > 0 and self.ema[tid] >= self.master_thr

    def status(self, tid):
        return {"unlocked": self.unlocked(tid), "mastered": self.mastered(tid),
                "ema": self.ema[tid], "episodes": self.episodes[tid]}

    def describe(self, task):
        tid = task if isinstance(task, str) else task["id"]
        return _describe(self.tasks[tid], status=self.status(tid))

    # ---- sampling ---------------------------------------------------------
    def _unlocked_in(self, fam):
        return [t for t in self.by_family.get(fam, []) if self.unlocked(t)]

    def _pick(self, options, weights):
        if sum(weights) <= 0:
            weights = [1.0] * len(options)
        return self.rng.choices(options, weights=weights, k=1)[0]

    def _pick_family(self):
        fams = [f for f in self.mix if self._unlocked_in(f)]
        if not fams:
            fams = [f for f in FAMILIES if self._unlocked_in(f)]
            if not fams:
                raise RuntimeError("no unlocked tasks in any family")
            return self._pick(fams, [1.0] * len(fams))
        return self._pick(fams, [self.mix[f] for f in fams])

    def _pick_task(self, family):
        opts = self._unlocked_in(family)
        weights = []
        for t in opts:
            w = float(self.tasks[t].get("weight", 1.0))
            if self.mastered(t):
                w *= self.review_weight
            weights.append(w)
        return self._pick(opts, weights)

    def _kit(self, tid, family):
        kit = {}
        rng = self.rng
        p = float(self.kits.get("start_kit_probability", 1.0))
        if self.kits.get("use_task_start_kit", True):
            skip = self.kits.get("drop_start_kit_when_mastered", True) and self.mastered(tid)
            if not skip and self.tasks[tid].get("start_kit") and rng.random() < p:
                kit.update(self.tasks[tid]["start_kit"])
        for rule in self.kits.get("rules", []):
            if rule.get("family") == family and rng.random() < float(rule.get("probability", 1.0)):
                for item, n in rule.get("kit", {}).items():
                    kit[item] = kit.get(item, 0) + int(n)
        return kit

    def sample_world(self, world):
        if not 0 <= world < self.num_worlds:
            raise IndexError(f"world {world} out of range")
        fam = self._pick_family()
        tid = self._pick_task(fam)
        a = Assignment(world=world, family=fam, task=tid, kit=self._kit(tid, fam))
        self.current[world] = a
        return a

    def sample_episode(self):
        """One assignment per world, in world order."""
        return [self.sample_world(w) for w in range(self.num_worlds)]

    # ---- learning signal --------------------------------------------------
    def update(self, world, task, success):
        """Record an episode outcome. `success` in [0, 1] (bool ok)."""
        tid = task if isinstance(task, str) else task["id"]
        if tid not in self.tasks:
            raise KeyError(f"unknown task {tid!r}")
        if not 0 <= world < self.num_worlds:
            raise IndexError(f"world {world} out of range")
        v = min(1.0, max(0.0, float(success)))
        if self.episodes[tid] == 0:
            self.ema[tid] = v
        else:
            self.ema[tid] = (1.0 - self.alpha) * self.ema[tid] + self.alpha * v
        self.episodes[tid] += 1

    # ---- persistence ------------------------------------------------------
    def state_dict(self):
        v, internal, gauss = self.rng.getstate()
        return {
            "version": 1,
            "num_worlds": self.num_worlds,
            "ema": dict(self.ema),
            "episodes": dict(self.episodes),
            "rng": [v, list(internal), gauss],
        }

    def load_state_dict(self, state):
        if state.get("version") != 1:
            raise ValueError("unsupported sampler state version")
        if state["num_worlds"] != self.num_worlds:
            raise ValueError("num_worlds mismatch")
        missing = set(self.tasks) - set(state["ema"])
        extra = set(state["ema"]) - set(self.tasks)
        if missing or extra:
            raise ValueError(f"state task mismatch: missing={sorted(missing)} extra={sorted(extra)}")
        self.ema = {k: float(v) for k, v in state["ema"].items()}
        self.episodes = {k: int(v) for k, v in state["episodes"].items()}
        v, internal, gauss = state["rng"]
        self.rng.setstate((v, tuple(internal), gauss))
