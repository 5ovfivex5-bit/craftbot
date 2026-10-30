# Parity dump format (`minemind-parity-1`)

Written by the Minecraft side (`/aibot dump <bot>`, FT-4) and read by `tools/parity/compare.py`.
One JSON file per dump, UTF-8, floats as JSON numbers (write float32 values with float32 precision).
Source of truth for the sim formulas: `src/sim.cpp` (`writeObs`, `raycast`, `boxHit`, `nearBlock`,
`target`) and `craftsim.VEC_NAMES`. Layout is generic: the obs vector carries its own names, so a v7
layout change needs no change here as long as the names have sim formulas (see `simobs.py`).

The Minecraft writer must snapshot everything in the same game tick: pose, inventory, blocks, and
the obs it computed (`Obs.build`). The dump is of the bot's state AFTER the last physics tick, with the
obs computed from that same state.

## Top level

```jsonc
{
  "format": "minemind-parity-1",
  "source": "minecraft",            // or "craftsim-selftest" (written by compare.py --selftest)
  "created_utc": "2026-10-16T09:30:00Z",
  "game": {
    "mc_version": "26.2",
    "mod": "craftbot <version>",
    "level": "minecraft:overworld",
    "game_tick": 123456,
    "seed": null                     // world seed if known
  },
  "bot": {"name": "bot0", "team": 0, "index": 0},   // index = k inside its team (sim agent k)

  "pose": {                          // feet position, same convention as the sim
    "x": 10.5, "y": 64.0, "z": -3.25,   // doubles, block units; y = feet height
    "yaw": 90.0, "pitch": -15.0,        // degrees, MC convention (sim: same)
    "on_ground": true,
    "eye_height": 1.62,               // standing eye height (MC Avatar.STANDING_DIMENSIONS)
    "sneaking": false
  },

  "player": {
    "health": 20.0,                   // 0..20, float
    "fuel": 0,                        // Bot.fuel (sim Agent.fuel), int
    "mining_progress": 0.0,           // 0..1, Bot.mineProgress (sim Agent.prog)
    "mining_target": null,            // [x, y, z] of the block being mined, or null
    "step": 0,                        // decisions this episode (sim world t)
    "max_steps": 3000,
    "hunger": null,                   // FoodData level if known (sim has none), else null
    "saturation": null
  },

  "goal": {"name": "iron_ore", "index": 9},   // ITEMS name + index (sim ITEMS order)

  "inventory": {                      // mapped MineMind items, zero entries omitted
    "log": 3, "planks": 8, "stick": 2
  },
  "inventory_other": {                // everything that is not a MineMind item: vanilla id -> count
    "minecraft:gravel": 4
  },

  "teammates": [                      // same team only, ascending bot index, sim agent order
    {"index": 1, "x": 12.0, "y": 64.0, "z": -2.0, "yaw": 0.0, "pitch": 0.0,
     "msg": 0, "health": 20.0}
  ],
  "bodies": [                         // every OTHER entity the rays could hit as AGENT (non-team
    {"x": 0.0, "y": 64.0, "z": 0.0, "kind": "player"}   // players, mobs in v7). Sim has none.
  ],

  "chests": [                         // every chest block in the cube, MineMind item names
    {"x": 11, "y": 64, "z": -3, "items": {"log": 5, "dirt": 9}}
  ],

  "blocks": { ... see below ... },
  "vision": { ... see below ... },
  "obs": { "img": { ... }, "vec": { ... } }
}
```

## `blocks` (the cube around the bot)

- `radius` R (default 16). Cube = feet block `floor(x), floor(y), floor(z)` +- R on each axis, clipped to
  the build limits (y outside the limits is not stored; the `build_limit` rule below covers it).
- `origin` `[x0, y0, z0]` = minimum corner of the stored cube (after clipping in y).
- `shape` `[ny, nz, nx]`. Cells are stored in `y, z, x` order, x fastest:
  index = `((y - y0) * nz + (z - z0)) * nx + (x - x0)` (same as sim `region()` / `World::w`).
- `build_limit` `{"min_y": int, "max_y": int}`: `level.getMinY()` and `getMinY() + getHeight() - 1`.
  Rules applied by both sides for any ray cell: `y > max_y` is air, `y < min_y` is bedrock (this is
  what `Obs.cast` does and what the sim does with `WY = 64`, `y < 0`). Cells in the cube are read directly.
- `palette`: vanilla block ids (`"minecraft:oak_log"`, ...), indexed by `vanilla`.
- `vanilla`: base64 of `uint16` little-endian palette indices, one per cell. Unloaded chunks use the
  palette entry `"unloaded"` and `ours` 255.
- `ours`: base64 of `uint8`, one per cell, the MineMind block id (`Obs.blockId`, sim `Block` enum):
  `0 air 1 grass 2 dirt 3 stone 4 cobblestone 5 log 6 leaves 7 planks 8 crafting_table 9 furnace
  10 coal_ore 11 iron_ore 12 diamond_ore 13 bedrock 14 sand 15 gravel 16 water 17 deepslate 18 chest`.
  `255` = unknown (unloaded). 19 (agent) is never stored in a world.

The cube is always stored whole; rays that leave the cube before hitting anything are reported as
"unknown" by the comparator, so a radius-16 dump gives partial image coverage (reported as coverage %).
Radius 28 covers every ray (max view distance 24 + eye height).

## `vision` (what the mod's `Obs` actually used)

```json
{"res": 64, "fov_deg": 70.0, "view_dist": 24.0, "reach": 4.5,
 "ray_origin": "feet + (0, 1.62, 0)", "bodies_in_rays": true,
 "depth_encoding": "min(255, floor(dist / view_dist * 255)); sky = 255"}
```

Informational: `compare.py` always uses the sim's constants and warns when these differ.

## `obs`

- `img`: `{"shape": [res, res, 2], "order": "row-major, row 0 = top; channels = (block_id, depth)",
  "encoding": "base64-u8", "data": "<base64>"}`. Same pixel grid as `writeObs`.
- `vec`: `{"dim": 102, "names": ["inv_log", ..., "xhair_chest_chest"], "values": [float x dim]}`.
  `names` is stored WITH the vector. Names are `craftsim.VEC_NAMES` at the time of writing.
- `source`: which code produced it, e.g. `"mcmod Obs.build (v6, res 64)"`.

## Notes

- Keep the JSON valid even when a part is missing: write `null` or `[]`, never omit keys.
- Yaw/pitch/health/mining are the values the MC side used for that obs (not re-read later).
- File name convention: `<gamedir>/minemind_dumps/<yyyyMMdd-HHmmss>-<bot>.json`.
- Current draft `Bot.dump` (in progress, format string `minemind-dump-1`) differs: it stores the vanilla
  ids as an unindexed string list, `img_b64` at top level, no vec names, no teammates, chests, bodies,
  fuel, mining target, build limits or radius. `compare.py` reads only `minemind-parity-1`.
