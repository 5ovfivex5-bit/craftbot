# Sim vs real Minecraft: gap list (FT-3)

Scope: the v6 sim (`src/sim.cpp`, `craftsim.py`) against the Minecraft mod (`mcmod/.../Obs.java`,
`Bot.java`) and the decompiled 26.2 sources (`mcsrc/`). Line numbers are as read at the time of writing.
Priority: **P0** = fix or randomize before real-MC fine-tuning results mean anything; **P1** = before the
v7 multi-task run or broad training; **P2** = small or only matters in rare cases.

"Knob" = a domain-randomization parameter for the sim (where it would live is given in brackets).
Nothing here changes the sim; this is the list to work from.

## Top 5

1. **Smelting is instant in both the sim and the bot, vanilla takes 200 ticks per ingot.** A furnace
   ingot is 10 s = 50 four-tick decisions. The iron pickaxe needs 3 ingots, so about 150 decisions of
   waiting that the policy never learns. (P0)
2. **Vertical scale and depth distribution differ.** Height obs is `(py - 30) / 32` in the sim (world
   0..63, sea 30) and `(y - 63) / 32` in `Obs.java`. Real terrain is y -64..320, so the scalar reaches about
   -4..+8 in MC against about -0.9..+1.06 in the sim. Ores also sit at different depths. (P0)
3. **Deepslate ores are 1.5x harder to mine in the real game.** Vanilla `DEEPSLATE_{IRON,COAL,DIAMOND,GOLD}_ORE`
   are `strength(4.5, 3.0)`, while the sim's coal/iron/diamond ore is 3.0 (sim.cpp BINFO lines 67-69,
   Blocks.java lines 316, 326, 1207). Below y=8 (the sim's deepslate line) every ore in the real game is the
   deepslate variant, so underground mining takes 1.5x longer in MC than in the sim. (P0)
4. **No mobs, hunger, drowning, lava or night in the sim.** `Agent` has no food or air field (sim.h), there
   is no lava block, and no mobs until SIM-1. The real game has zombies, skeleton arrows, starvation, lava
   pools and a 24000-tick day. `Obs.java` maps lava to STONE, so the policy sees lava as a wall. (P0 for
   real play, P1 for the sim, which v7 covers.)
5. **Inventory is unlimited in the sim, bounded in the real game, and tools break.** The sim keeps `int inv[]`
   counts with no slot or stack cap. Vanilla has 36 main slots, a 64-item stack limit, 1-item tools, and
   pickaxes with durability 59 / 131 / 250 (wood / stone / iron, `ToolMaterial.java` lines 23-27) that lose
   one point per block (`Item.mineBlock`, Item.java line 263). A full inventory drops items on the ground
   instead of keeping them. (P1)

## Full list

| # | Area | Sim (what it does) | Real (what MC does) | Evidence | Impact | Pri | Knob (where) |
|---|---|---|---|---|---|---|---|
| 1 | Smelting timing | `A_SMELT_IRON` gives an ingot in one step (sim.cpp 856-866) | `AbstractFurnaceBlockEntity` cooks in `cookingTotalTime`, 200 ticks default (lines 253, 297) | sim.cpp:856-866; Bot.java `smelt()` is instant too | Long plans (iron tools) far too fast | P0 | `smelt_ticks` per ingot U[150, 250]; a furnace-timer state. [sim, SIM-2 or CORE] |
| 1b | Fuel | coal = 8 smelts, plank = 1 (sim.cpp:859-860, comment "vanilla 1.5") | Planks burn 300 ticks, 1.5 ingots (Bot.java comment, not re-verified in mcsrc) | as cited | Small | P2 | `plank_smelts` in {1, 1.5}. [sim] |
| 2 | Height scalar and vertical range | `height = (py-30)/32`, world y 0..63, bedrock below 0 (sim.cpp 1006, WY=64) | `height = (y-63)/32`, world y -64..319 (Obs.java `build`, line ~466), deepslate under y≈0 | `Obs.java` header comment says sea 63 ↔ sim 30; the scale does not match | OOD input for every "how deep am I" decision | P0 | randomize `sea_level` in [30, 63] and the scalar's scale; or make height relative to sea and clip to [-1, 1] in both. [sim obs + mod Obs] |
| 3 | Ore hardness underground | BINFO: coal/iron/diamond ore = 3.0 at every depth; no deepslate ore id (sim.cpp 67-69) | Deepslate ore variants `strength(4.5, 3.0)` (Blocks.java ~316, ~326, ~1208). Deepslate itself 3.0 | `mcsrc/.../Blocks.java` | Underground mining 1.5x slower | P0 | `deep_ore_hardness_mult` in [1.0, 1.5] applied for y < 8. [sim BINFO lookup] |
| 4 | Mobs | none (mobs.h not yet filled, v7 SIM-1) | Zombies, skeletons, spiders, creepers, arrows, knockback | sim has no entity code; Bot.java `attackTick` hits blocks only | Death and damage are absent from every reward and obs | P0 real / P1 sim | `mob_density`, `hostile_on_night`, arrow dmg. [sim, SIM-1] |
| 5 | Hunger, regen, starvation | none (`Agent` has health only) | `FoodData`: exhaustion (0.1 per attack, Player.java 1239, in `attack`), regen needs food >= 18, starvation at 0. Bot never eats (FT-4 policy must act) | `mcsrc/net/minecraft/world/food/FoodData.java`, Player.java 1239 | Long episodes starve in the real game | P1 | `food_drain_mult` in [0.5, 2]; `start_food` in [6, 20]. [sim, SIM-2] |
| 6 | Air and drowning | no air meter; water is passable and slows, nothing else | Air 300 ticks, drowning damage after that | vanilla behaviour; no field in sim.h | A bot can sit in water forever in sim | P1 | `air_ticks` in [150, 300]. [sim] |
| 7 | Lava | no lava block id (sim.h `Block` enum) | Lava pools below sea; burning, death | Obs.java `blockId`: `fluid.is(LAVA)` -> STONE | Real deaths the sim cannot show; policy sees lava as stone | P1 | lava patches below y 30 with a death rule. [sim, CORE] |
| 8 | Day and night | none in v6 | 24000-tick day, darkness, monster spawns | `Bot.java` has no time; `Obs.build` has no light | Mob exposure and visibility | P1 (v7) | `time_of_day` in the obs (v7 scalar), darkness only in mobs. [sim, SIM-2] |
| 9 | Inventory slots and stacks | `int inv[NUM_ITEMS]`, no caps; chests unlimited (sim.h `Chest`) | player: 36 main slots, 64 per stack, tools 1 (`Inventory.getContainerSize` = items + equipment slots); chest: 27 slots | Bot.java `insert` / `give` use `getMaxStackSize`; overflow -> `player.drop` | Mining 64+ items diverges; overflow is lost in MC | P1 | `inv_slots` U{27..36}, `stack_cap` per item; `chest_slots` 27. [sim, CORE] |
| 10 | Tool durability | tools never break (`pickTier` uses counts) | Wood 59, stone 131, iron 250 uses; `Item.mineBlock` -1 per block with destroySpeed != 0 | ToolMaterial.java 23-27, Item.java 257-265 | After ~59 blocks a wood pickaxe is gone in MC | P1 | `tool_durability_mult` in [0.5, 1.0] (1.0 = vanilla); break event. [sim, CORE] |
| 11 | Drops from leaves, gravel, sand | leaves / sand / gravel drop nothing (BINFO drop -1) | Leaves: stick 2%, sapling 5%. Gravel: flint 10%. Sand/gravel are items | vanilla loot tables (`mcsrc` loot in jar) | A stick source the sim lacks | P2 | `leaf_stick_p` in [0, 0.02]. [sim BINFO] |
| 12 | Chest spill | chest contents go straight into the miner's inventory (sim.cpp ~703-708) | Contents drop as item entities on the ground, pickup has a delay | vanilla chest behaviour; Bot.java picks up within 1.5 blocks | Slightly faster in sim | P2 | `chest_spill_mode` drop vs direct. [sim] |
| 13 | Decision latency | one decision = exactly 4 ticks | Bot idles while the bridge reply is pending: `decide()` returns early while `pending` is not done (Bot.java ~152-175), fwd/strafe stay zero | Bot.java `decide` / `tick` | Real decisions take 4 + latency ticks | P1 | `decision_jitter_ticks` U{0..8} extra idle ticks. [sim step loop] |
| 14 | Teammate and body rays | AGENT only for same-world teammates (boxHit over `ag[k]`) | `Bot.bodies()` (Bot.java ~209) = every `level.players()` entry, so a human nearby is AGENT (19) in the image | Bot.java `bodies()` (around line 174-178) | Human players look like teammates in play mode | P1 play / P2 train | `extra_agents` 0..2 random boxes not in the team. [sim render] |
| 15 | Block shapes in rays | every block is a full cube (sim.cpp `solid`, DDA) | Stairs, slabs, fences, leaves and flowers differ; `blocksMotion` decides hits | `Obs.blockId` returns AIR for non-motion blocks, STONE otherwise | Hit distance off by up to half a block; some blocks invisible | P2 | `depth_jitter` ±1 level; `partial_block_p` (renders as a half block). [render] |
| 16 | Block id collapse | 20 ids | Obs maps gold/copper/redstone/lapis/granite/andesite/netherrack/glass/ice to STONE; podzol/mycelium to GRASS; deep-slate and tuff to DEEPSLATE | Obs.java `blockId` | Visual confusion, mostly harmless | P2 | add rare ids or randomize unknown-block colours. [sim BLOCKS] |
| 17 | Eye height when crouching | fixed 1.62 (no sneak in v6) | 1.27 crouching, 1.62 standing, 0.4 swimming/crawling | Avatar.java lines 21-30 | Only matters once sneak is an action (v7) | P2 (v7) | `eye_height` 1.27 when sneak. [sim, v7] |
| 18 | Sprint / speed | walk only (4.317 m/s, sim.cpp 22-30) | Sprint 5.6 m/s, jump boost; sprint costs hunger | vanilla | No sprint action in v6 | P2 | `walk_speed_mult` in [0.95, 1.05]. [sim physics] |
| 19 | Swim/water physics | travelInWater mirrored (sim.cpp 9-10, 490s) | Same formulas; water currents push in MC, not in sim | vanilla `LivingEntity.travelInWater` | Currents never seen in sim | P2 | `current_strength` in [0, 0.02]. [sim] |
| 20 | Worldgen and ore density | fbm terrain, ore counts scaled by area (sim.cpp header) | Vanilla ore/biome distributions, trees, caves, villages | sim.cpp header lines 13-17 | Learned search priors may not transfer | P1 | `ore_density_mult` in [0.5, 2]; `tree_density_mult` in [0.5, 2]. [sim worldgen, CORE] |
| 21 | World edge | x/z outside 0..127 is bedrock (sim.cpp `get`) | Infinite world | sim.cpp get() | Bot sees a wall at the edge | P2 | none needed for sim; avoid spawning near edge. [sim] |
| 22 | Crafting rules | pickaxe / furnace need a table in the `nearBlock` box; costs match vanilla | Table must be open (3x3 grid); planks and sticks use the 2x2 grid | `Bot.java` enforces the same box, so consistent | Rare: a bot crafting far from a table | P2 | none. |
| 23 | Mod-side bugs seen while reading (not sim) | n/a | `Obs.build` teammate slots use list position and skip >32 blocks (Obs.java 476-480); the sim never has absent teammates | Obs.java lines 476-480 | Only matters with absent teammates (other level) | P2 | keep; document. |

## Notes on what already matches

- Physics constants (gravity 0.08, drag 0.98, jump 0.42, friction 0.6*0.91, AABB 0.6 x 1.8, eye 1.62) match
  `LivingEntity` / `Player` (sim.cpp lines 1-27).
- Block break formula `speed / hardness / (canHarvest ? 30 : 100)`, airborne /5, submerged /5 matches
  `BlockBehaviour.getDestroyProgress` (BlockBehaviour.java line 348-356) and `Player.getDestroySpeed` (585-607).
- Tool speeds 2 / 4 / 6 match `ToolMaterial` (WOOD 2.0, STONE 4.0, IRON 6.0).
- Surface block hardness values in BINFO (stone, cobble, dirt, grass, sand, gravel, log, leaves, planks, table,
  furnace, chest, coal / iron / diamond ore, bedrock) agree with the vanilla values I checked. Only the deepslate
  ore variants were read in mcsrc (gap 3); the rest were not re-checked line by line.
- Vision: FOV 70 for both axes, DDA ray order, depth = dist/24*255 truncated (sim.cpp 950-990 and Obs.java
  `build`) are the same formula; `tools/parity/compare.py --selftest` shows bit-level agreement.
- Fall damage `ceil(d - 3)` matches vanilla (sim.cpp 645).

## Suggested order

1. Gaps 1, 2 and 3 (sim knobs, cheap, decide whether the policy can ever learn the real timing and depth).
2. Gap 4 with SIM-1 / SIM-2 (mobs, hunger, lava, air, night), since they change reward and death.
3. Gaps 9 and 10 (inventory and durability) before any mining-heavy v7 task.
4. Gaps 13 and 14 (latency and bodies) as part of FT-4's Bot changes.
5. Everything tagged P2 when there is time.
