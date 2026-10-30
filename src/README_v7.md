# CORE v7 notes (for phase-2 and phase-3 agents)

Generated data: `tools/gen_registry.py` reads `data/registry/*.json` (else `*.seed.json`) and `data/tasks/tasks.json`
(else `tasks.seed.json`) into `src/gen/registry.h`. `build.sh` runs it. Ids are file order; blocks 0..19 and items 0..14
are the v6 ids (checked). Current counts (from DATA-1/DATA-2 files as of this build):
NUM_BLOCKS 59, NUM_ITEMS 97, NUM_RECIPES 65, NUM_MOBS 9, NUM_TASKS 64, NUM_FAMILIES 9, NUM_PRIMARY 21.
Named ids exist as constants in the header: `BLK_*`, `ITM_*`, `REC_*`, `MOB_*`, `TSK_*`, and `REC_V6_*` (v6 recipes).

## Action modes
- `CraftSim(..., action_mode="v7")` (default): `step(actions int32[N, 4], msgs=None)`.
- `action_mode="v6"`: `step(actions int32[N], msgs)`, the 28 v6 ids (`ACTIONS`), 102-float obs, 15 items, 20 blocks.

### v7 action columns
Column 0 = primary (`PRIMARY_NAMES`, 21):
`0 noop, 1 forward, 2 back, 3 left, 4 right, 5 jump, 6 forward_jump, 7 sneak (toggle), 8 sprint_forward (x1.3),
9 turn_left, 10 turn_right, 11 look_up, 12 look_down, 13 attack (hold until break, max 80 ticks), 14 use (place the
selected block item on the crosshair face; flint_and_steel lights a portal; food -> survival::eat; white_bed ->
survival::sleep; ignored on table/furnace/chest), 15 craft (recipe in column 1), 16 smelt (recipe in column 1,
needs furnace), 17 deposit, 18 withdraw, 19 give, 20 drop (empties the selected stack)`.
Column 1 = recipe: 0 none, r+1 for recipe r (`RECIPE_NAMES`).
Column 2 = item select: 0 keep the selected item, i+1 select item i (sets `held`, applied before the primary).
Column 3 = message token 0..7 (`msgs` argument overrides it when given).
Sneak multiplies walk input by 0.3; sprint_forward multiplies ground acceleration by 1.3.

### v6 mapping (action_mode "v6")
Each v6 id maps to a v7 primary; placements select cobble > dirt > planks (or the table/furnace/chest item) at decision
time; craft/smelt use the REC_V6_* recipes. Mining/fuel/value use the v6 legacy tables (BINFO, ITEM_VALUE_V6,
fuel coal 8 / planks 1) so v6 numbers are unchanged. The v6 reward/goal mechanism is unchanged.

## Obs v7 layout (VEC_DIM = 295; indices are exact, names in `craftsim.VEC_NAMES`)
| range | block | content |
|---|---|---|
| 0..96 | inventory (V7_INV) | log2(1+n)/7 for all 97 items |
| 97..116 | scalars (V7_SCALARS) | 97 health/20, 98 hunger/20, 99 saturation (0), 100 on_ground, 101 pitch/90, 102 sin yaw, 103 cos yaw, 104 height (y-30)/32, 105 in_water, 106 light (survival::lightAt), 107 sin(time), 108 cos(time), 109 dimension, 110 armor (0), 111 mining progress, 112 fuel/8, 113 steps_left, 114 sneak, 115 held index (held+1)/97, 116 horizontal speed/0.3 |
| 117..180 | task one-hot (V7_TASK) | NUM_TASKS = 64 |
| 181..182 | task params (V7_TASK_PARAM) | 181 target item / 97 (success predicate, items only), 182 count/64 |
| 183..230 | teammates (V7_TEAM) | 4 slots x [present, dx/32, dy/32, dz/32, msg one-hot 8], yaw frame |
| 231..234 | nearest chest (V7_CHEST) | [present, dx/16, dy/16, dz/16] |
| 235..243 | crosshair chest (V7_XHAIR) | log2(1+count)/6 for the 9 most valuable items (`reg::XHAIR_ITEMS`) |
| 244..261 | mobs (V7_MOBS) | 3 x [present, dx/16, dy/16, dz/16, hostile, health frac], from mobs::obs |
| 262..294 | build (V7_BUILD) | 8 x [dx/8, dy/8, dz/8, item/97] + progress, from build::obs |

The image stays `res x res x 2` (block id incl. mob pseudo ids 20..58, depth).

## Hooks (all in src/<module>.h; neutral stubs in src/<module>.cpp)
Module state: `mobs::State`, `nether::State`, `build::State` are embedded in World; `survival::AgentState` is embedded in
Agent (must stay trivially copyable). Hook call sites:
- `mobs::init(World&)`: initWorld, after terrain.
- `mobs::tick(World&)`: once per world step, v7 only, before the success check.
- `mobs::obs(const World&, const Agent&, float* out18)`: writeObs, v7 only.
- `mobs::rayHit(const World&, const double o[3], const double d[3], double maxDist, float& t, uint8_t& id)`: not called yet (add to the image ray when mobs fill it).
- `mobs::attack(World&, Agent&, float damage)`: not called yet (sim has no mob attack path).
- `survival::tick(World&, Agent&)`: per live agent per decision, v7 only, after its action.
- `survival::eat(Agent&, int item)`: use on a food item (must consume the item itself).
- `survival::sleep(World&, Agent&)`: use on a white_bed.
- `survival::lightAt(const World&, int x, int y, int z)`: obs scalar 106.
- `survival::timeOfDay(const World&)`: obs scalars 107/108 and `VecEnv.time_of_day`.
- `nether::generate(World&, uint64_t seed)`: initWorld, every world generation, seed = world seed ^ 0x9E3779B97F4A7C15 (derived, never reused).
- `nether::tryLightPortal(World&, Agent&, int x, int y, int z) -> bool`: use flint_and_steel at the crosshair (sets portal_lit).
- `nether::travel(World&, Agent&)`: per live agent per decision, v7 only.
- `build::worldgen(World&, uint64_t seed)`: initWorld.
- `build::obs(const World&, const Agent&, float* out33)`: writeObs, v7 only.
- `build::progress(const World&) -> float`: build_complete predicate (currently ignores the blueprint index; SIM-4 must
  make it per blueprint, e.g. by changing the signature to take the blueprint).
- `build::placeTarget(World&, int blueprint, int x, int y, int z)`: not called yet.
Dimensions: each agent has `dim` (always 0 now). The nether block array is not implemented; `in_dimension` predicates
evaluate `dim` only. Adding a nether array means runtime dimension sizes in `get/put` (SIM-3 work).

## Tasks (v7)
- One task per world episode, drawn from `set_task_weights(w, w[NUM_TASKS])` (default uniform) at each episode start.
  `set_task(w, task, target=None, count=None)` changes the current episode (resets counters) and overrides the success
  predicate's parameters (target = item/block/mob name or index, count = count/steps/dist/n).
- Success ends the episode for the world: team reward 10 split by team share (as the v6 goal).
- Milestones: each pays its reward once per episode to the agent that satisfied it (bit per milestone, max 8).
- Predicates are evaluated per agent with per-episode counters in `Agent::st` (TaskStats): mined, placed, crafted, eaten,
  killed, distance, minY, near (decision counts at 2/4/8/16 blocks), look, follow, crouch, gave, shared, msgNear.
- `reached_y {max_y}` means the agent's feet reached y <= max_y at least once (DATA-2 assumption).
- Social: `near_teammate` uses the nearest distance bucket >= dist; `look_at_teammate` = centre ray hits an AGENT block;
  `crouch_greet` = a sneak toggle within 10 decisions of the previous toggle with a teammate in view within 4 blocks;
  `follow_teammate` = forward move with a teammate in view 3..8 blocks; `gave_item`, `shared_chest` (depositor and
  withdrawer both count), `msg_near_teammate` (non-silent token, teammate within 8).
- Knowledge shaping: `set_shaping_table(task, items)`: first time an agent obtains a listed item in that task, it gets
  0.05 * value (once per agent and item per episode).
- Unknown names in task data are inert (predicate target -1) and reported by the generator as warnings.

## Seeds
- Every world has its own RNG streams seeded from the master seed. Every auto-reset / reset_env draws a fresh 64-bit seed
  never equal to the previous one. `world_seed(w)` and `state(i)` expose it; the test checks distinct seeds across worlds and resets, and that world overworld regions differ.

## Known gaps
- Mobs, survival, nether, build modules are neutral stubs: no mob entities, no hunger, no day/night, no nether, no build
  targets. Obs slots for them are zeros.
- save_state/load_state serialize the world, agents (incl. task stats, held, sneak, dim) and task fields; module states
  (mobs/nether/build) are not serialized yet.
- VecEnv.entities(w) returns an empty list until mobs fill it.
- follow_teammate and shared_chest have no end-to-end test yet (code paths exist).
- Achievement bitmask `ach`: items are recorded for indices < 31 only (bit 31 = task success / goal).
- Items/blocks: the v7 registry has 97 items and 59 blocks; the trainer's v6 code must use action_mode "v6" and the
  *_V6 constants until it is ported.
- Performance (W=32, K=4, res=64, random actions, same run): v6 ~10.6k, v7 ~10.3k agent-steps/s (ratio 0.98).
