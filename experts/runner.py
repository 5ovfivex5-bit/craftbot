"""Lockstep runner: W worlds x K agents, one CraftSim.step per decision (PROTOCOL 14.2).

Each agent row runs a skill coroutine `skill(ctl, task_info)` (a generator built on core.Ctl). The Runner
assigns tasks to worlds from per-task episode quotas, sets each world's task before its episode starts,
records every decision of live agents, and closes an episode when the sim reports done for its world.
"""
import random
import time
import functools
import json
import math
import os

import numpy as np

import craftsim as cs

from experts.core import Ctl

NOOP_ROW = (0, 0, 0, 0)


class Runner:
    def __init__(self, sim, skills, tasks, episodes_per_task, seed=0, writer=None, log=None):
        if sim.action_mode != "v7":
            raise ValueError("experts need CraftSim(action_mode='v7')")
        for t in tasks:
            if t not in skills:
                raise KeyError(f"no skill registered for task {t!r}")
            if t not in cs.TASK_NAMES:
                raise KeyError(f"unknown task {t!r}")
        self.sim = sim
        self.K = sim.agents
        self.W = sim.num_envs
        self.N = sim.n
        self.skills = skills
        self.tasks = list(tasks)
        self.quota = {t: int(episodes_per_task) for t in tasks}   # episodes still to be assigned
        self.writer = writer
        self.seed = int(seed)
        self.log = log or (lambda *a: None)
        self.ctl = [Ctl(sim, i) for i in range(self.N)]
        self.gen = [None] * self.N
        self.pending = [None] * self.N       # action row waiting to be stepped, or None (noop)
        self.live = [None] * self.W          # task name of the episode in progress, None = idle world
        self.next_plan = [None] * self.W     # task drawn by the upcoming auto-reset of each world
        self.buf = [[] for _ in range(self.N)]
        self.ep_id = [-1] * self.W
        self.ep_counter = 0
        self.decisions = 0
        self.task_decisions = {t: 0 for t in tasks}
        self.stats = {t: {"episodes": 0, "successes": 0, "decisions": 0, "errors": 0} for t in tasks}
        self.errors = []
        self.wall = 0.0
        self._success = [False] * self.W
        self._ach = [0] * self.W             # last team mask seen for each world (sim ach of its last step)
        # seeds for early world resets (PROTOCOL: never reused; drawn from the runner's own stream)
        self._reset_rng = random.Random(self.seed * 7919 + 17)
        self.early_ends = 0

    # ------------------------------------------------------------------ planning

    def _take(self):
        """Assign the next episode: the task with the most unassigned episodes left (ties: list order)."""
        best = None
        for t in self.tasks:
            if self.quota[t] > 0 and (best is None or self.quota[t] > self.quota[best]):
                best = t
        if best is not None:
            self.quota[best] -= 1
        return best

    def _weights_for(self, w, task):
        v = np.ones(cs.NUM_TASKS, np.float32)
        if task is not None:
            v[:] = 0.0
            v[cs.TASK_NAMES.index(task)] = 1.0
        self.sim.set_task_weights(w, v)

    # ------------------------------------------------------------------ episodes

    def _start_world(self, w, task, vec=None):
        """Begin an episode of world w (its obs is already the reset obs). The task's start kit (tasks.json
        start_kit, part of the task definition like in training) is given before the first decision and
        patched into `vec` (the obs rows that will be recorded first) as log2(1 + n) / 7."""
        self.live[w] = task
        self.ep_id[w] = self.ep_counter
        self.ep_counter += 1
        if task is None:
            return
        self.sim.set_task(w, task)           # task of this episode; resets its counters (obs unchanged)
        for item, n in _start_kits().get(task, {}).items():
            if item not in cs.ITEMS:
                continue
            for k in range(self.K):
                self.sim.set_inventory(w * self.K + k, item, int(n))
                if vec is not None:
                    vec[w * self.K + k, cs.ITEMS.index(item)] = math.log2(1 + int(n)) / 7.0
        for k in range(self.K):
            i = w * self.K + k
            rng = random.Random(self.seed * 1000003 + self.ep_id[w] * 1009 + k)
            self.ctl[i].new_episode(task, rng)
            self.gen[i] = self.skills[task](self.ctl[i], self.ctl[i].task_info)
            self._advance(i, first=True)

    def _advance(self, i, first=False):
        try:
            self.pending[i] = next(self.gen[i]) if first else self.gen[i].send(None)
        except StopIteration:
            self.gen[i] = None
            self.pending[i] = None
        except Exception as e:  # a broken skill must not stop the other worlds
            self.gen[i] = None
            self.pending[i] = None
            w = i // self.K
            t = self.live[w]
            self.stats[t]["errors"] += 1
            self.errors.append(f"{t} agent {i}: {type(e).__name__}: {e}")
            self.log(f"skill error in {t}: {type(e).__name__}: {e}")

    def _flush(self, w):
        """Close the episode of world w: write its rows with the success flag of the sim."""
        task = self.live[w]
        ach_success = self._success[w]
        for k in range(self.K):
            i = w * self.K + k
            rows = self.buf[i]
            self.buf[i] = []
            if self.writer is not None and rows:
                self.writer.add_episode(cs.TASK_NAMES.index(task), w, k, self.ep_id[w], rows, ach_success)
            st = self.stats[task]
            st["episodes"] += 1
            st["successes"] += int(ach_success)
            st["decisions"] += len(rows)

    def _end_early(self, w, img_next, vec_next):
        """The skill of every agent of world w has returned without the sim ending the episode: close the
        episode now (recorded as a failure, with done on its last real step) and reset the world with a fresh
        seed through the sim's reset path (VecEnv::resetWorld via reset_env), instead of idling until max_steps.
        Writes the reset obs of the world's rows into img_next / vec_next (the step's obs arrays)."""
        K = self.K
        for k in range(K):
            i = w * K + k
            if self.buf[i]:
                img_b, vec_b, act_b, rew_b, _ = self.buf[i][-1]
                self.buf[i][-1] = (img_b, vec_b, act_b, rew_b, True)
        self._success[w] = bool((self._ach[w] >> 31) & 1)
        self._flush(w)
        self.early_ends += 1
        seed = self._reset_rng.getrandbits(62)
        img_r, vec_r = self.sim._env.reset_env(w * K, seed)   # raw call: craftsim.reset_env returns nothing
        img_next[w * K:(w + 1) * K] = img_r
        vec_next[w * K:(w + 1) * K] = vec_r
        self.sim._sync_rows(w * K, vec_r)
        task_next = self.next_plan[w]
        self.next_plan[w] = self._take()
        self._weights_for(w, self.next_plan[w])
        self.live[w] = None
        self._start_world(w, task_next, vec_next)

    # ------------------------------------------------------------------ main loop

    def run(self):
        sim, K, W, N = self.sim, self.K, self.W, self.N
        t0 = time.time()
        for w in range(W):
            self.live[w] = self._take()
            self._weights_for(w, self.live[w])
        img, vec = sim.reset()
        for w in range(W):
            if self.live[w] is not None and sim.task(w * K) != cs.TASK_NAMES.index(self.live[w]):
                raise RuntimeError("task weights did not select the planned task")
        for w in range(W):
            self.next_plan[w] = self._take()
            self._weights_for(w, self.next_plan[w])
        for w in range(W):
            live = self.live[w]
            self.live[w] = None
            self._start_world(w, live, vec)
        while any(t is not None for t in self.live):
            actions = np.zeros((N, 4), np.int32)
            live_rows = 0
            for w in range(W):
                if self.live[w] is None:
                    continue
                for k in range(K):
                    i = w * K + k
                    live_rows += 1
                    if self.pending[i] is not None:
                        actions[i] = self.pending[i]
                        self.task_decisions[self.live[w]] += 1
            img2, vec2, rew, done, ach = sim.step(actions)
            self.decisions += live_rows
            for w in range(W):
                if self.live[w] is None:
                    continue
                for k in range(K):
                    i = w * K + k
                    if self.writer is not None:
                        self.buf[i].append((img[i].copy(), vec[i].copy(), actions[i].copy(),
                                            float(rew[i]), bool(done[w * K])))
            for w in range(W):
                if self.live[w] is None:
                    continue
                self._ach[w] = int(ach[w * K])
                if bool(done[w * K]):
                    self._success[w] = bool((int(ach[w * K]) >> 31) & 1)
                    self._flush(w)
                    task_next = self.next_plan[w]
                    if task_next is not None and sim.task(w * K) != cs.TASK_NAMES.index(task_next):
                        raise RuntimeError("auto-reset did not draw the planned task")
                    self.next_plan[w] = self._take()
                    self._weights_for(w, self.next_plan[w])
                    self.live[w] = None
                    self._start_world(w, task_next, vec2)
                else:
                    for k in range(K):
                        i = w * K + k
                        if self.gen[i] is not None:
                            self._advance(i)
                    if all(self.gen[w * K + k] is None for k in range(K)):
                        self._end_early(w, img2, vec2)
            img, vec = img2, vec2
        self.wall = time.time() - t0
        return self.stats

    def summary(self):
        rate = self.decisions / self.wall if self.wall > 0 else 0.0
        return {"decisions": self.decisions, "wall_s": self.wall, "decisions_per_s": rate,
                "tasks": {t: dict(s, success_rate=(s["successes"] / s["episodes"] if s["episodes"] else 0.0))
                          for t, s in self.stats.items()}}


@functools.lru_cache(maxsize=1)
def _start_kits():
    """{task id: {item: count}} from data/tasks/tasks.json (empty dict when a task has no kit)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tasks", "tasks.json")
    with open(path) as f:
        data = json.load(f)
    tasks = data["tasks"] if isinstance(data, dict) else data
    return {t["id"]: dict(t.get("start_kit") or {}) for t in tasks}
