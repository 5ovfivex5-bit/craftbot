"""Skill registry. A skill is a generator function `skill(ctl, task)` built from core.Ctl primitives:

    @skill("gather_wood")
    def gather_wood(ctl, task):
        ...
        yield from ctl.mine_block(x, y, z)

`task` is the tasks.json entry (params, success, ...). The Runner drives it; success is decided by the sim
(the task's success predicate), so a skill only has to pursue it. Returning ends the agent's expert; the
episode then idles (noop) until the sim ends it.
"""
from experts.core import Ctl  # noqa: F401  (re-exported for skill authors)

SKILLS = {}


def skill(name):
    def register(fn):
        if name in SKILLS:
            raise ValueError(f"duplicate skill {name!r}")
        SKILLS[name] = fn
        return fn
    return register


def gather_logs(ctl, n, item="oak_log", max_fails=30):
    """Mine until the inventory holds `n` logs: nearest reachable log first. A log that cannot be reached or
    mined is skipped, and the agent explores to another area before the next try. Returns bool."""
    blacklist = set()
    fails = 0
    while ctl.count(item) < n:
        if fails > max_fails:
            return False
        before = ctl.count(item)
        found = yield from ctl.mine_nearest([item], radius=14, skip=blacklist)
        if found is None:
            fails += 1
            yield from ctl.explore()
            continue
        cell, ok = found
        if not ok or ctl.count(item) <= before:
            blacklist.add(cell)
            fails += 1
            yield from ctl.explore()
    return True


@skill("gather_wood")
def gather_wood(ctl, task):
    """Success is decided by the sim (8 oak logs mined). Keep chopping the nearest reachable log."""
    need = int(task["params"].get("count", 8))
    yield from gather_logs(ctl, need)


@skill("tools_crafting_table")
def tools_crafting_table(ctl, task):
    """Logs -> 4 planks -> crafting table (the craft ends the episode with success)."""
    if ctl.count("oak_planks") < 4:
        if not (yield from gather_logs(ctl, 1)):
            return
        yield from ctl.craft("oak_planks")
    if ctl.count("oak_planks") >= 4:
        yield from ctl.craft("crafting_table")


__all__ = ["SKILLS", "skill", "gather_logs"]


# Every other module in experts/skills/ registers its skills on import (one file per family owner).
def _load_all():
    import importlib
    import pkgutil
    for m in pkgutil.iter_modules(__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"{__name__}.{m.name}")


_load_all()
