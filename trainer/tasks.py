"""v7 task scheduling: a family per world, a task per world episode (tasklib curriculum).

Timing (src/README_v7.md): the sim draws each world's task at its own auto-reset, from that world's
weights, inside the step that ends the previous episode. The trainer only sees the episode end after
that step, so the scheduler keeps one episode of look-ahead per world:

  cur[w]     task of the episode that is running now (its draw used the weights set one boundary ago)
  queued[w]  task of the NEXT episode: the sim's weights for world w are one-hot(queued[w]) now, so the
             next auto-reset draws exactly this task

At every episode end of world w: the finished episode is recorded (sampler.update with cur[w]), then
cur <- queued (already drawn by the sim), queued <- a fresh choice from the curriculum (with the updated
success EMA), and the weights are set to one-hot(queued). The curriculum therefore acts one episode
later than the outcome, which is the price of the sim's own draw (documented in README.md).

Families: each world has one family (tasklib mix proportions, largest-remainder allocation over the
families that have an unlocked task). A world switches family only when its family has no unlocked
task, or when its family is over quota after the families were rebalanced (checked once per update, the
switch happens at that world's next episode boundary).

Kits: tasklib's start kits are applied with set_inventory right after the episode starts; the vec obs
inventory block is patched to the same values (log2(1 + n) / 7), so the first decision sees the kit.
"""
import math
import random
from collections import Counter, deque

import numpy as np

import craftsim
from tasklib import TaskSampler, load_envmix, load_shaping

from . import envs as E

EP_WINDOW = 200
TASK_NAMES = list(craftsim.TASK_NAMES) if E.V7 else []
FAMILIES = list(craftsim.FAMILIES) if E.V7 else []


class WorldTasks:
    def __init__(self, num_worlds, seed, envmix="default"):
        if not E.V7:
            raise RuntimeError("WorldTasks needs the v7 observation (run with --obs v7)")
        self.W = int(num_worlds)
        self.envmix_name = envmix
        self.sampler = TaskSampler(self.W, envmix=load_envmix(envmix), seed=int(seed))
        self.rng = random.Random(int(seed) + 101)
        self.family = [None] * self.W           # world family (assignment)
        self.pending = set()                    # worlds to move to a new family at their next boundary
        self.cur = [None] * self.W              # task name of the running episode
        self.cur_family = [None] * self.W
        self.cur_kit = [{} for _ in range(self.W)]
        self.queued = [None] * self.W           # task name of the next episode (already set in the sim)
        self.queued_family = [None] * self.W
        self.queued_kit = [{} for _ in range(self.W)]
        self.window = deque(maxlen=EP_WINDOW)   # (task, family, success, return, length, kit)
        self.fam_episodes = Counter()           # cumulative world episodes per family
        self.task_index = {n: i for i, n in enumerate(TASK_NAMES)}
        self.last_task = None
        self.family_switches = 0
        self._check_kit_items()
        self.shaping = load_shaping().get("tasks", {})
        self.assign_initial()

    # ----- families -----
    def _unlocked_families(self):
        fams = [f for f in self.sampler.mix if self.sampler._unlocked_in(f)]
        if not fams:
            fams = [f for f in FAMILIES if self.sampler._unlocked_in(f)]
        return fams

    def _targets(self):
        """Largest-remainder counts of the worlds per family over the mix restricted to unlocked families."""
        fams = sorted(self._unlocked_families())
        if not fams:
            raise RuntimeError("no unlocked task in any family")
        mix = {f: float(self.sampler.mix.get(f, 0.0)) for f in fams}
        tot = sum(mix.values())
        if tot <= 0:
            mix = {f: 1.0 for f in fams}
            tot = float(len(fams))
        raw = {f: mix[f] / tot * self.W for f in fams}
        counts = {f: int(math.floor(raw[f])) for f in fams}
        left = self.W - sum(counts.values())
        for f in sorted(fams, key=lambda f: (-(raw[f] - counts[f]), f))[:left]:
            counts[f] += 1
        return counts

    def assign_initial(self):
        counts = self._targets()
        fams = [f for f in sorted(counts) for _ in range(counts[f])]
        self.rng.shuffle(fams)
        self.family = fams[:self.W]

    def family_counts(self):
        c = Counter(self.family)
        return {f: int(c.get(f, 0)) for f in FAMILIES if c.get(f, 0) or f in self._unlocked_families()}

    def rebalance(self):
        """Once per update: worlds in families over quota (or gated) move at their next boundary."""
        counts = self._targets()
        cur = Counter(self.family)
        over = {f: cur[f] - counts.get(f, 0) for f in cur if cur[f] > counts.get(f, 0)}
        for w in range(self.W):
            f = self.family[w]
            if w in self.pending:
                continue
            if not self.sampler._unlocked_in(f):
                self.pending.add(w)
            elif over.get(f, 0) > 0:
                self.pending.add(w)
                over[f] -= 1

    def _deficit_family(self, w):
        counts = self._targets()
        cur = Counter(f for i, f in enumerate(self.family) if i != w)
        best = max(sorted(counts), key=lambda f: (counts[f] - cur.get(f, 0), self.sampler.mix.get(f, 0.0)))
        return best

    # ----- choice -----
    def _choose(self, w):
        fam = self.family[w]
        if (w in self.pending) or not self.sampler._unlocked_in(fam):
            new = self._deficit_family(w)
            if new != fam:
                self.family_switches += 1
            fam = new
            self.family[w] = fam
            self.pending.discard(w)
        tid = self.sampler._pick_task(fam)
        kit = {k: int(v) for k, v in self.sampler._kit(tid, fam).items() if k in E.ITEMS}
        return tid, fam, kit

    def _onehot(self, tid):
        v = np.zeros(len(TASK_NAMES), np.float32)
        v[self.task_index[tid]] = 1.0
        return v

    # ----- sim wiring -----
    def _check_kit_items(self):
        bad = set()
        for t in self.sampler.tasks.values():
            bad |= {k for k in (t.get("start_kit") or {}) if k not in E.ITEMS}
        for r in self.sampler.kits.get("rules", []):
            bad |= {k for k in r.get("kit", {}) if k not in E.ITEMS}
        if bad:
            raise ValueError(f"start kit items not in the v7 registry: {sorted(bad)}")

    def apply_shaping(self, sim):
        """set_shaping_table for every task (empty for tasks without a shaping list)."""
        for name in TASK_NAMES:
            items = [i for i in self.shaping.get(name, {}).get("items", []) if i in E.ITEMS]
            sim.set_shaping_table(name, items)

    def plan_start(self, sim, worlds):
        """Before the first reset of a (re)built sim: the running and the next task of every world."""
        if worlds != self.W:
            self.resize(worlds)
        for w in range(self.W):
            self.cur[w], self.cur_family[w], self.cur_kit[w] = self._choose(w)
            self.queued[w], self.queued_family[w], self.queued_kit[w] = self._choose(w)
            sim.set_task_weights(w, self._onehot(self.cur[w]))

    def after_reset(self, sim, vec, K):
        """After sim.reset(): the next tasks become the sim's weights; running kits are applied."""
        for w in range(self.W):
            sim.set_task_weights(w, self._onehot(self.queued[w]))
        for w in range(self.W):
            vec = self.apply_kit(sim, vec, w, K, self.cur_kit[w])
        return vec

    def apply_kit(self, sim, vec, w, K, kit):
        if not kit:
            return vec
        vec = np.array(vec, copy=True)
        for row in range(w * K, (w + 1) * K):
            for item, n in kit.items():
                sim.set_inventory(row, item, n)
                vec[row, E.ITEMS.index(item)] = math.log2(1 + n) / 7.0
        return vec

    def advance(self, sim, w, K, vec, success, ret, length):
        """Episode of world w ended. Returns the vec with the next episode's kit applied."""
        kit = self.cur_kit[w]
        self.sampler.update(w, self.cur[w], success)
        self.window.append((self.cur[w], self.cur_family[w], float(success), float(ret), int(length),
                            bool(kit)))
        self.fam_episodes[self.cur_family[w]] += 1
        self.last_task = self.cur[w]
        self.cur[w], self.cur_family[w], self.cur_kit[w] = self.queued[w], self.queued_family[w], self.queued_kit[w]
        self.queued[w], self.queued_family[w], self.queued_kit[w] = self._choose(w)
        sim.set_task_weights(w, self._onehot(self.queued[w]))
        return self.apply_kit(sim, vec, w, K, self.cur_kit[w])

    def resize(self, W):
        """Worlds changed (num_envs rebuild): keep the curriculum, re-allocate families."""
        keep = list(self.family)[:W]
        self.W = int(W)
        self.sampler.num_worlds = self.W
        self.cur, self.cur_family = [None] * W, [None] * W
        self.cur_kit = [{} for _ in range(W)]
        self.queued, self.queued_family = [None] * W, [None] * W
        self.queued_kit = [{} for _ in range(W)]
        self.family = keep + [None] * (W - len(keep))
        counts = self._targets()
        cur = Counter(f for f in self.family if f is not None)
        for w in range(W):
            if self.family[w] is None or cur[self.family[w]] > counts.get(self.family[w], 0):
                f = max(sorted(counts), key=lambda f: (counts[f] - cur.get(f, 0), f))
                if self.family[w] is not None:
                    cur[self.family[w]] -= 1
                self.family[w] = f
                cur[f] = cur.get(f, 0) + 1
        self.pending = set()

    # ----- telemetry / persistence -----
    def status(self):
        unlocked = [t for t in TASK_NAMES if self.sampler.status(t)["unlocked"]]
        mastered = [t for t in TASK_NAMES if self.sampler.mastered(t)]
        return {"unlocked": len(unlocked), "mastered": len(mastered), "total": len(TASK_NAMES)}

    def telemetry(self):
        win = list(self.window)
        by_task, by_fam = {}, {}
        for task, fam, ok, ret, _ln, _kit in win:
            by_task.setdefault(task, []).append(ok)
            by_fam.setdefault(fam, []).append((ok, ret))
        st = self.status()
        return {
            "family_mix": {f: n / max(1, self.W) for f, n in self.family_counts().items()},
            "family_worlds": self.family_counts(),
            "family_episodes": dict(self.fam_episodes),
            "family_success": {f: float(np.mean([o for o, _ in v])) for f, v in by_fam.items()},
            "family_return": {f: float(np.mean([r for _, r in v])) for f, v in by_fam.items()},
            "task_success": {t: float(np.mean(v)) for t, v in by_task.items()},
            "task_episodes": {t: int(self.sampler.episodes[t]) for t in TASK_NAMES if self.sampler.episodes[t]},
            "task_ema": {t: float(self.sampler.ema[t]) for t in TASK_NAMES if self.sampler.episodes[t]},
            "tasks_unlocked": st["unlocked"], "tasks_mastered": st["mastered"],
            "curriculum_stage": f"{st['unlocked']}/{st['total']} tasks unlocked, {st['mastered']} mastered",
            "curriculum_task": self.last_task,
            "family_switches": self.family_switches,
        }

    def state_dict(self):
        return {"version": 1, "families": list(self.family), "fam_episodes": dict(self.fam_episodes),
                "sampler": self.sampler.state_dict(), "rng": self.rng.getstate(),
                "envmix": self.envmix_name}

    def load_state_dict(self, sd):
        if sd.get("version") != 1:
            raise ValueError("unsupported task state version")
        target = self.W
        if len(sd["families"]) != target:   # world count changed: restore at the saved size, then resize
            self.W = len(sd["families"])
            self.sampler.num_worlds = self.W
        self.family = list(sd["families"])
        self.fam_episodes = Counter(sd.get("fam_episodes", {}))
        self.sampler.load_state_dict(sd["sampler"])
        v, internal, gauss = sd["rng"]
        self.rng.setstate((v, tuple(internal), gauss))
        if self.W != target:
            self.resize(target)   # keeps the curriculum, re-allocates families to the new world count
