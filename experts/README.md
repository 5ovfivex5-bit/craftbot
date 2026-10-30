# experts/ — scripted experts for the v7 sim

Scripted players that solve tasks in `craftsim` through the normal v7 action columns, and record their
decisions as demonstration shards for behaviour cloning (PROTOCOL 14). Rules (PROTOCOL 14.1): experts act only
through `CraftSim.step`; they may read privileged state (blocks, inventory, entities); no pose, inventory or
block edits during an episode.

```
experts/core.py       Ctl: primitives (one yield = one decision), A* pathing, aiming, ray helper
experts/runner.py     Runner: W worlds x K agents in lockstep, task quotas, episode boundaries
experts/record.py     ShardWriter: 14.2 shard format + meta.json
experts/generate.py   CLI
experts/skills/       SKILLS registry + reference skills (gather_wood, tools_crafting_table)
experts/test_core.py  primitive tests on fixed seeds
```

Run from the repo root inside the distrobox:

```
PYTHONPATH=. .venv/bin/python -m experts.generate --tasks gather_wood,tools_crafting_table \
    --episodes-per-task 32 --worlds 16 --agents 1 --out <dir>
PYTHONPATH=. .venv/bin/python -m experts.test_core
```

## Writing a skill

A skill is a generator `skill(ctl, task)`. `task` is the `data/tasks/tasks.json` entry (`params`, `success`,
`max_steps`, ...). The sim decides success (the task predicate); the skill only pursues it.

```python
from experts.skills import skill

@skill("my_task")
def my_task(ctl, task):
    ok = yield from ctl.craft("oak_planks")          # every primitive is used with `yield from`
    if not ok:
        return                                       # returning ends the expert; the episode idles
    yield from ctl.craft("crafting_table")
```

Registered skills are picked up by `--tasks` in `generate.py`. A skill that returns early leaves its agent idle
(noop) until the sim ends the episode; that episode is recorded with `success=False`.

## Execution model

- Each primitive is a generator. Each decision it yields one row `(primary, recipe_col, item_col, msg)` and the
  Runner steps the sim with it. When the generator resumes, the world is fresh: `state`, `region`, `inventory`
  are re-read after every decision (they are cached only within one decision).
- Primitives return values (bool, or the accept result), so use `ok = yield from ctl.x(...)`.
- One decision = `ticks` (4) sim ticks = 0.2 s. Turns and look steps are 15 degrees, one per decision.
- The item column is applied before the primary in the same decision (`item=IID[name] + 1`).
- Any primitive may spend several decisions; each yield is a decision and is counted in `ctl.decisions`.

## Ctl API

Coordinates are block cells `(x, y, z)`; a "feet cell" is the cell the agent's feet are in.

State (no decisions):

| call | returns |
|---|---|
| `ctl.state()` | the sim `state(i)` dict (x, y, z, yaw, pitch, held, health, ...); cached per decision |
| `ctl.pos()` / `ctl.cell()` / `ctl.eye()` | feet position, feet cell, eye position |
| `ctl.held()` | name of the held item or None |
| `ctl.inventory()` / `ctl.count(item)` | dict name -> count / count of one item |
| `ctl.block_at(x, y, z)` | block name (None outside the cached window) |
| `ctl.find_blocks(names, radius=12)` | `[((x, y, z), name), ...]` nearest first, Chebyshev radius |
| `ctl.near_block(name)` | a crafting table / furnace is within sim reach (`craft`/`smelt` need it) |
| `ctl.region(radius=None)` | the block window (`Region`, `.get(x, y, z)` -> id or -1 outside) |
| `ctl.task`, `ctl.task_info`, `ctl.rng` | the task name, its tasks.json entry, a seeded `random.Random` |

Primitives (generators; use `yield from`):

| call | does | returns |
|---|---|---|
| `turn_to_yaw(yaw)` | turn_left/right until the yaw matches (15 deg steps) | True |
| `look_to_pitch(pitch)` | look_up/down until the pitch matches | True |
| `aim_at(point, accept=None)` | turn + look toward a world point; with `accept(hit)` it picks the quantized orientation whose crosshair ray satisfies accept (hit = `(cell, previous_cell)` or None) | accept result (True without accept) |
| `face(x, y, z)` / `look_at_block` | aim the crosshair at block (x, y, z) (verified with a ray) | True when the ray hits that block; otherwise the closest quantized look (within ~10 deg) |
| `face_point(px, py, pz)` | aim at a point, no ray check | True |
| `walk_to(x, y, z, within=0.5, dig=False, max_decisions=600)` | walk to a standable feet cell (or one within `within` blocks of its column, dy <= 1) | True on arrival |
| `explore(min_d=6, max_d=12)` | walk to a random reachable cell 6-12 blocks away (no digging) | bool |
| `mine_block(x, y, z, walk=True)` | break the block with the best tool in inventory; clears a blocking block in reach (leaves) or walks to a stand spot with a clear view | True when the cell is air |
| `mine_nearest(names, radius=12, skip=())` | mine the nearest block of those names not in `skip` | `(cell, ok)` or None if none in range |
| `place_block(item, x, y, z, walk=True)` | place `item`'s block into the empty cell, through a visible face (aims so that the crosshair face is the one next to the cell) | True when the cell holds the block |
| `craft(recipe_name, walk=True)` | craft one recipe (craft column); table recipes first ensure a table in reach (place one or walk to one) | True if the result count rose |
| `ensure_station(name, walk=True)` | make `crafting_table` / `furnace` reachable (already near, place from inventory beside you, or walk to the nearest in the window) | bool |
| `smelt(input_item, walk=True)` | smelt one unit (smelt column); ensures a furnace in reach; fuel is taken from inventory | True if the result rose |
| `equip(item)` | select `item` as held (item column, noop primary) | True if held |
| `eat(item)` | use one `item` (eats food) | True if one was consumed |
| `attack_entity(mob=None, max_decisions=400)` | chase the nearest entity named `mob` (or any), face and attack within 2.5 blocks | True when none is left, False on timeout or when none is present |
| `flee(x, y, z, distance=10)` | walk to a standable cell away from (x, y, z) | bool |
| `wait(n=1)` | n noop decisions | True |
| `_act(prim, recipe=0, item=0, msg=0)` | one raw decision (`prim` is a name from `craftsim.PRIMARY_NAMES`; `recipe` = r + 1, `item` = i + 1) | none |

Module helpers (`experts.core`): `astar(reg, start, goal_fn, hx, hz, dig=False)`, `ray_hit(reg, origin, dir)`,
`standable(reg, x, y, z)`, `BID`/`IID` (name -> block/item id), `BLOCK_INFO`/`ITEM_INFO`/`RECIPES`
(registry JSON by name), `TASK_INFO` (tasks.json by id).

## Pathing rules

- Standable feet cell: feet and head cells are air and the cell below is solid (sim solid = any block except air,
  water, lava). Water and lava are never walked into; they are not standable and not part of any path.
- Moves: 4 horizontal neighbours; flat (cost 1); up one block by jumping (needs head room above the start);
  down 1-3 blocks (ledge; the sim deals no fall damage up to 3 blocks: damage = ceil(fall - 3)).
- `walk_to` uses A* (weighted, w=1.6) with a node budget (1200). When no open path exists it retries with digging
  (`dig`): flat tunnels and one-block staircases are mined (feet and head cells), at a cost of 4 per block.
- Walking follows the path: the next cell is reached by strafe/back/forward (no turns needed), steering toward
  the node centre from the real position; the body first centres on its lane when more than 0.25 off.
  Up-steps turn to face the cell, then `forward_jump`.
- Replans happen when the path is left (rejoins a nearby node of the path if possible) or a needed cell is solid.

## Aiming and reach

- Yaw and pitch are quantized to 15 deg; `aim_at`/`face` test the 9 quantized orientations around the exact
  angle and take the one whose crosshair ray hits the wanted cell (cheapest turn first).
- Ray = sim `target()` (DDA, reach 4.5). Water stops the ray unless the eye is in water. `ray_hit` returns the hit
  cell and the previous (air) cell, which is where `place_block` puts a block.
- `mine_block` aims at the block centre; when the view is blocked, the blocking block in reach is cleared first.
- Mining holds `attack` until the block breaks (sim: up to 80 ticks in one decision), so one decision per block
  is typical. Tool choice is by inventory (the sim uses the best tool it holds).

## Tips for skill authors

- Check preconditions before a primitive (`count`, `near_block`); the sim fails silently (a craft without
  ingredients or without a table in reach does nothing), so verify with the return value.
- Keep a per-episode blacklist of cells that failed (see `skills.gather_logs`): an unreachable target would
  otherwise be retried forever.
- Prefer `mine_nearest` / `find_blocks` over scanning the world yourself; they use the same window as the planner.
- Use `ctl.explore()` when no target is in range, so the agent moves to new ground.
- A primitive that returns before yielding costs no decision: loops must yield or change state, or they spin.
  Each retry loop needs a decision budget.
- Do not keep `state()`/`region()` results across a yield; call them again after resuming.
- Timeouts: give long searches `max_decisions`. The task's `max_steps` (tasks.json) still applies at episode level.
- Mob combat (`attack_entity`) needs the mobs module (`sim.entities(w)` is empty until it is implemented).
