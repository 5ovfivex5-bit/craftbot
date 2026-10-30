# MineMind — shared contracts

Project root: `/home/deck/claude/ai/test/craftbot`. Three independent parts:

```
src/, CMakeLists.txt, craftsim.py, tests/   C++20 simulator + pybind11 module   (owner: sim)
trainer/                                     PyTorch PPO trainer + telemetry server (owner: trainer)
dashboard/                                   C++20 SDL3 + OpenGL3 + Dear ImGui app  (owner: dashboard)
```

Data flow: `C++ sim -> batched obs -> PyTorch` (never blocked by UI).
Separately: `sim + PyTorch -> sampled telemetry -> TCP -> dashboard` (lossy, latest-wins).

All builds/runs happen inside the distrobox: `distrobox enter hi -- bash -c '...'`.
Python: `.venv/bin/python` (3.12, torch 2.14 cpu, numpy, pybind11, psutil).

---

## 1. Python simulator API (`import craftsim`)

`craftsim.py` is a facade over the native pybind11 module `_craftsim` (built into `build/`).
Names/semantics below are frozen — the trainer codes against them.

```python
NUM_ACTIONS, NUM_ITEMS, NUM_BLOCKS, VEC_DIM : int
ACTIONS: list[str]   ITEMS: list[str]   BLOCKS: list[str]   # index == id
WORLD_SIZE = (64, 48, 64)            # (sx, sy, sz)

class CraftSim:
    def __init__(self, num_envs=16, seed=0, res=32, max_steps=3000, ticks=4, threads=0): ...
    # threads=0 -> OpenMP default
    def reset(self) -> tuple[img uint8[N,res,res,2], vec float32[N,VEC_DIM]]
    def step(self, actions: int32[N]) -> (img, vec, rew float32[N], done bool[N], ach uint32[N])
        # done envs auto-reset; ach = achievement bitmask (bit i = ITEMS[i]) of the finished/current episode
    def reset_env(self, i, seed) -> None          # deterministic world from explicit seed
    def render(self, i=0, width=320, height=240) -> uint8[H,W,3]
    def inventory(self, i=0) -> dict[str,int]
    def set_inventory(self, i, item: str, count) -> None
    def state(self, i=0) -> dict  # keys: x y z vx vy vz yaw pitch on_ground health t ticks
                                  #       seed episode last_action(int) mining_progress
                                  #       target: None | {"x","y","z","block": str}
    def set_pose(self, i, yaw, pitch) -> None
    def block(self, i, x, y, z) -> str
    def region(self, i, radius=16) -> tuple[(ox,oy,oz), uint8[sy,sz,sx]]
        # full world height, XZ window of +-radius around player, clipped to world bounds
        # array index [y][z][x] relative to origin; values are block ids
    def close(self) -> None
achievements(mask) -> list[str];  save_png(path, rgb)
```

`step()`: one decision = `ticks` physics ticks, except `attack`, which holds the button until the
targeted block breaks or the target changes (max 80 ticks), like holding left-click in Minecraft.

Obs `img[...,0]` = block id hit by ray (0 = sky), `img[...,1]` = depth 0..255 (255 = far/sky).

---

## 2. Telemetry / control protocol (trainer = server, dashboard = client)

- TCP `127.0.0.1:7777` (trainer flag `--port`). Multiple clients allowed; clients may connect,
  disconnect and reconnect at any time without affecting training.
- Framing: **one JSON object per line** (`\n`-terminated, UTF-8). Binary payloads are base64.
- Server never blocks on a client: each client has a bounded outbox; for `snapshot` and
  `activations` only the newest message is kept (older dropped). `metrics`, `log`, `ack` are kept
  (bounded deque, oldest dropped on overflow).
- Every message has `"type"`. `schema` = 1.

### Server -> client

| type | when | fields |
|---|---|---|
| `hello` | on connect | `schema, run_id, state, config{...}, world_size[3], actions[], items[], blocks[]` |
| `metrics` | 2 Hz | `t, state ("idle"|"running"|"paused"|"stopping"|"stopped"), mode ("fast"|"live"|"debug"), env_steps, target_steps, sps, episodes, avg_ep_len, avg_reward, success_rate{item: 0..1}, updates, policy_loss, value_loss, entropy, approx_kl, cpu_percent, rss_mb, mem_warning (str|null), elapsed_s, eta_s (float|null = "Calculating"), latest_checkpoint (str|null), curriculum_stage, num_envs, config{...}` |
| `snapshot` | <= `snapshot_hz` while visualize && a client subscribed | `env_id, episode, seed, step, player{x,y,z,vx,vy,vz,yaw,pitch,health,on_ground}, last_action (str), last_reward, ep_reward, task, target (null|{x,y,z,block}), inventory{}, action_history[str] (last 32), origin[3], size[3] (sx,sy,sz), blocks (base64 uint8, index (y*sz+z)*sx+x), obs_res, obs (base64 uint8 res*res*2)` |
| `model` | on connect, after reset/load | `name, params_total, input{img:[res,res,2], vec:[VEC_DIM]}, gru_hidden, layers:[{name,type,params,in_shape,out_shape,trainable}], edges:[[from,to]]` |
| `activations` | <= `nn_hz` while a client subscribed to nn | `env_id, step, action_probs[NUM_ACTIONS], entropy, value, layers{name:{shape, sample[<=256 floats], mean, std, min, max, hist{edges[17], counts[16]}}}, gru{mean,std,min,max,norm}, grad_norms{name: float}, weight_stats{name:{mean,std,min,max}}, nan_inf (bool)` |
| `history` | reply to `get_history` | `rows:[metrics-like dicts]` |
| `checkpoints` | reply to `list_checkpoints` | `items:[{path, env_steps, updates, time, size_bytes}]` |
| `log` | any time | `t, level ("info"|"warn"|"error"), msg` |
| `ack` | reply to any cmd | `id, ok (bool), msg` |

### Client -> server: `{"cmd": "...", "id": <int>, ...}`

| cmd | args | semantics |
|---|---|---|
| `start` | – | begin/continue training from idle/stopped |
| `pause` / `resume` | – | pause at next env-step boundary; experience kept |
| `stop` | – | finish current update, write final checkpoint, state -> stopped |
| `reset` | – | (must be paused/stopped) fresh model+optimizer+counters |
| `save_checkpoint` | – | atomic save now (at next safe point) |
| `load_checkpoint` | `path` | (paused/stopped) restore model, optimizer, counters, RNG |
| `eval_checkpoint` | `path, episodes` | run separate eval policy on separate env; does not touch training run; reports via `log` |
| `list_checkpoints` | – | -> `checkpoints` |
| `get_history` | `last` (int) | -> `history` |
| `set_config` | `values{}` | keys: `total_steps, num_envs, lr, minibatch_size, rollout_len, ckpt_every, visualize, snapshot_hz, nn_hz, workers, mode`. Applied at the next rollout boundary (num_envs/rollout_len rebuild env + buffers after the current update completes; lr applied to optimizer param_groups). Ack says "queued". |
| `set_mode` | `mode` | fast/live/debug; never resets training |
| `select_env` | `env_id` | which env is snapshotted/inspected |
| `subscribe` | `snapshot (bool), nn (bool)` | per-client; instrumentation off when nobody subscribed |
| `debug_step` | `n, kind ("tick"|"decision")` | only when paused: steps a dedicated debug env (seeded from the selected env's episode seed) – training envs untouched; reply with a `snapshot` whose `env_id` = -1 |

"Steps" always means **environment transitions** (sum over envs). `updates` = PPO optimizer
iterations (one per rollout). ETA = (target - env_steps) / EMA(sps), null until >= 5 samples.

---

## 3. v2 changes (Minecraft-fidelity + goal-conditioned training) — FROZEN

Reference: decompiled Minecraft 26.2 sources in `mcsrc/net/minecraft/...` (read-only).

### Blocks (ids 0-13 unchanged, new ids appended) — `NUM_BLOCKS = 18`
`14 SAND, 15 GRAVEL, 16 WATER, 17 DEEPSLATE`. Deepslate ore variants stay COAL_ORE/IRON_ORE/DIAMOND_ORE.
WATER is not solid (rays/obs see it as id 16 and continue? NO: rays STOP at water surface and report 16, like
looking at a lake; the player can enter it).

### World — `WORLD_SIZE = (64, 64, 64)`, `SEA_LEVEL = 30`, deepslate below y = 8.

### Items — unchanged (14). `ITEMS` order unchanged.

### Vector obs — `VEC_DIM = 25 + 14 = 39`
Index 0..24 as before EXCEPT index 20 (height) is now `(py - SEA_LEVEL) / 32` (Minecraft side uses
`(y - 63) / 32`). Indices 25..38 = one-hot of the episode **goal item** (ITEMS index).

### Goals / reward
- Every reset the sim samples a goal item for that env from goal weights (`set_goal_weights(w[14])`,
  default = uniform over items 0..12, dirt excluded). Python: `CraftSim.set_goal_weights(np.ndarray)`,
  `CraftSim.goal(i) -> int`, `CraftSim.set_goal(i, item)` (rewrites that env's goal + obs vector).
- Reward per step = shaping + goal:
  - shaping: first time an item is obtained this episode: `0.1 * (1 + log2(MineRL value))` (dirt 0)
  - goal: `+10` the step the goal item count first becomes > 0 → episode ends (`done = True`).
- `ach` bitmask: bits 0..13 as before; **bit 31 = goal achieved** in that episode.
- `state(i)` adds key `goal` (int).

---

## 4. v3: self-improvement (no imitation) — FROZEN

Observation layout is UNCHANGED from v2 (VEC_DIM 39, NUM_BLOCKS 18) so v2 checkpoints keep working.

### Start kits (sim)
`CraftSim.set_start_kits(table: float32[14 goals, 14 items], prob: float32[14])`: at every reset, if the
sampled goal g is drawn with probability `prob[g]`, the env starts with `table[g, item]` of each item
(e.g. goal iron_ore -> stone_pickaxe 1, stick 2, planks 4). Default: prob all 0. The first obs after reset
already reflects the kit. Items given by the kit do NOT pay shaping reward and do not set ach bits.
`state(i)` adds `kit` (bool, whether this episode started with a kit).

### Curiosity bonus (sim), scaled by `CraftSim.set_novelty(scale)` (default 0 = off)
Per episode, reward `scale *` :
- `+1.0` the first time the player enters each 4x4x4 cell of the world (feet position),
- `+2.0` the first time each block type appears anywhere in the 32x32 observation,
- `+3.0` for each new minimum depth reached below SEA_LEVEL in 2-block steps.
Novelty reward is included in `rew`; `CraftSim.last_novelty()` returns float32[N] of the novelty part of
the last step (so the trainer can log it separately).

### Trainer
Curriculum also drives start kits (kit probability ~0.5 for goals beyond stone_pickaxe, annealed toward
0.1 as that goal's no-kit success rises), novelty scale schedule (start ~0.02, linear to 0 by 60% of
total steps), default BPTT 32. Metrics add `novelty_reward`, `kit_rate`, and goal success is reported
for NO-KIT episodes only (`goal_success`), plus `goal_success_kit`.

---

## 5. v4: research upgrades — FROZEN (obs layout still v2: VEC_DIM 39, NUM_BLOCKS 18)

### Sim: env snapshots for Go-Explore restarts
- `CraftSim.save_state(i) -> bytes`: full serialized env i (world blocks, player physics, inventory, goal,
  kit flag, episode counters, novelty bookkeeping, rng state, mining state). Compact (world compressed or
  raw; must be < 300 KB).
- `CraftSim.load_state(i, blob: bytes, goal: int | None = None)`: restore env i from a blob; if `goal`
  is given, overwrite the goal (and the obs one-hot); sets `t` (episode step) to 0 so the restarted
  episode gets a full time budget; marks state(i)['restart'] = True for that episode; rewrites the obs
  returned by the next call... the facade must also return fresh obs: `load_state` returns
  `(img[res,res,2], vec[VEC_DIM])` for env i, and updates the internal last-obs arrays.
- Achievements (ach bits) and first-time shaping restart from the restored state's own masks.
- Deterministic: load_state(save_state(i)) then the same actions -> identical trajectory.

### Trainer (no sim changes needed)
1. **Go-Explore archive**: cells keyed by (goal, set of items obtained, depth bucket); store a snapshot
   when a cell is first reached or reached in fewer steps; restart a fraction of resets (`--restart-frac`,
   default 0.25) from archive cells weighted toward rarely-chosen, late-tech-tree cells (count-based
   1/sqrt(n+1)); goal for the restarted episode = next item up the tech tree not yet in that cell.
   Restart episodes are reported separately (`goal_success_restart`); `goal_success` stays from-scratch
   (no kit, no restart).
2. **RND** curiosity: target/predictor MLPs on the fused encoder input (img embedding pooled + vec),
   intrinsic reward normalized by running std, separate intrinsic value head (two-headed value, gamma_int
   0.99, non-episodic), coefficient schedule like the v3 novelty (sim hand-crafted novelty default OFF
   when RND is on).
3. **HL-Gauss** value heads (classification over bins on the normalized return, sigma = 0.75*bin width).
4. **Achievement prediction** auxiliary loss: head predicting the NEXT item the agent will obtain in this
   episode (14 classes + "none"), cross-entropy on rollout data, small coef (0.1).
All new parts are flags (`--go-explore/--no-go-explore`, `--rnd/--no-rnd`, `--value-loss hlgauss|mse`,
`--aux-ach/--no-aux-ach`), defaults ON. Checkpoints keep loading v2/v3 checkpoints (new heads initialized
fresh, strict=False only for the new modules).

---

## 6. v6: teams of bots, chests, messages, bigger worlds — CURRENT

Backup of the v5 sources: `backups/pre-v6.tar.gz`. The CUDA sim (`gpusim/`) stays v5 (64^3, single agent)
and is NOT updated: the trainer must refuse `--backend gpusim` unless `--agents 1` AND a v5 layout is
requested. v6 training is CPU only.

### 6.1 World
- `WX = 128, WY = 64, WZ = 128` (`WORLD_SIZE = (128, 64, 128)`), SEA_LEVEL 30, deepslate below 8.
  Terrain/ore/tree generation scaled to fill the larger map (same densities per area).
- Each world gets its own seed: the VecEnv seeds world w's RNG from the master seed, every auto-reset
  draws a fresh 64-bit world seed from that RNG (never reused). The trainer's default master seed is
  random (time/urandom based) and logged + stored in the checkpoint; `--seed N` makes a run repeatable.
- Visited-cell novelty: 4x4x4 cells over the larger world (32 x 16 x 32 = 16384 cells).
- Height obs unchanged `(py-30)/32`. Any x/z scalar normalized by 64 is now normalized by the new size
  (keep the [0,1] or [-1,1] range the scalar had).

### 6.2 Teams
- `CraftSim(num_envs=W, agents=K, ...)`: W worlds, K agents per world (1..5, default 1).
  Agent index `i = w*K + k`. EVERY per-agent API (reset/step arrays, state, inventory, render,
  set_pose, region, goal...) is indexed by agent i, arrays have N = W*K rows. `num_envs` stays the number
  of worlds; `CraftSim.num_agents == W*K`; `CraftSim.agents == K`.
- Agents in a world share the blocks, chests and the goal. Each agent has its own body, inventory,
  health, mining state, novelty bookkeeping, and kit.
- Spawn: all K agents within 6 blocks of each other on dry grass with clear sky.
- Agents do not collide with each other (they can overlap). Rays (obs image + crosshair) DO hit other
  agents of the same world: a ray that hits another agent's box (0.6 x 1.8 x 0.6, feet at its position)
  before any block reports pseudo-block id `AGENT = 19` and its depth. Attack/place target only blocks
  (rays used for targeting ignore agents).
- One episode per WORLD: all K agents get `done=True` on the same step and the world auto-resets.
  The episode ends when ANY agent first holds the goal item, or at max_steps.

### 6.3 Blocks / items / actions (appended; old ids unchanged)
- Blocks: `18 CHEST`, `19 AGENT` (pseudo: only appears in obs, never stored in the world). `NUM_BLOCKS = 20`.
- Items: `14 chest` (`I_CHEST`). `NUM_ITEMS = 15`. ITEMS order: old 14 then "chest". MineRL-style value for
  chest shaping: 8. Chest may be a curriculum goal (it is not in the default goal weights).
- Actions appended after `craft_iron_pickaxe` (old ids 0..22 unchanged), `NUM_ACTIONS = 28`:
  - `23 craft_chest`: 8 planks -> 1 chest (needs a crafting table within 3 blocks, like the other table recipes).
  - `24 place_chest`: places a CHEST block like place_table. Each placed chest has its own 27-slot-free
    storage `int count[NUM_ITEMS]` (no slot limits, counts only). Mining a chest drops the chest item and
    spills its contents into the miner's inventory. Chests take ~2.5 s by hand like a table.
  - `25 deposit`: crosshair on a chest within reach -> moves ALL non-tool items (everything except
    wooden/stone/iron pickaxe) from the agent into that chest. No-op otherwise.
  - `26 withdraw`: crosshair on a chest within reach -> moves ALL items from that chest to the agent.
  - `27 give`: nearest teammate within 4 blocks receives half (rounded up) of each of the giver's non-tool
    stacks. No-op if nobody is in range.
- Items obtained through withdraw/give/chest-spill do NOT count as achievements and pay no shaping.

### 6.4 Messages ("chat")
- Every agent also emits one message token per step: `msg` in `0..7` (`NUM_MSGS = 8`, 0 = silent).
- `step(actions int32[N], msgs int32[N] | None)` (None = all 0). The token an agent sent on its last step
  is visible to its teammates in the next obs (6.5). Tokens have no built-in meaning; the agents learn it.
- `CraftSim.last_msgs() -> int32[N]`.

### 6.5 Vector obs — `VEC_DIM = 102`, names in `craftsim.VEC_NAMES` (list[str], len VEC_DIM)
```
0..14    inventory counts (15 items incl. chest), same scaling as before
15..25   the 11 v5 scalars, same order and meaning (old indices 14..24)
26..40   goal one-hot (15)
41..88   4 teammate slots x 12: [present, dx/32, dy/32, dz/32, msg one-hot(8)]
         slots ordered by teammate index k' != k ascending; dx = teammate - me in world coords, rotated
         into the agent's yaw frame (x = right, z = forward); missing teammates -> all zeros
89..92   nearest chest within 16 blocks: [present, dx/16, dy/16, dz/16] (yaw frame like above)
93..101  crosshair chest contents: log2(1+count)/6 for items 0..8 (log..iron_ore) — zeros unless the
         crosshair is on a chest within reach
```
Warm-start mapping old (v5, VEC_DIM 39) -> new: new[0..13]=old[0..13]; new[15..25]=old[14..24];
new[26..39]=old[25..38]; everything else is new (zero-initialized weights).

### 6.6 Reward (per agent; K=1 reproduces v5 exactly)
- Team achievements: an item counts as obtained the first time ANY agent of the world mines/crafts/smelts
  it this episode. Value `r = 0.1*(1+log2(value))` is paid as `team_share * r` to every agent of the
  world plus `(1 - team_share) * r` extra to the agent who obtained it. `CraftSim.set_team_share(x)`,
  default 0.5.
- Goal: when any agent first holds the goal item: `+10 * team_share` to every agent and
  `+10 * (1 - team_share)` extra to the agent who got it; episode ends for the world.
- Novelty: per agent, as v3 (scale `set_novelty`).
- `ach[i]` = the WORLD's team mask (same for all its agents), bit 31 = goal achieved.

### 6.7 State / save
- `state(i)` adds `world` (int), `agent` (int k), `msg` (last token int), `team` (list of the other agents'
  indices i'). New `CraftSim.chests(w) -> list[{"x","y","z","items": {name: count}}]` for world w.
- `save_state/load_state` operate on a WORLD index w (all its agents + chests), blob size limit 2 MB.
  Go-Explore in the trainer is disabled when K > 1.
- `render(i, ...)` draws other agents as colored boxes (agent color palette: k=0 blue, 1 red, 2 green,
  3 yellow, 4 purple) and chests as brown boxes.

### 6.8 Trainer
- `--agents K` (default 4 for new runs), `--team-share`, `--msg-ent-coef` (default = ent coef).
  `--envs` keeps meaning worlds; buffers/GRU states have W*K rows. Parameter sharing: one policy for all
  agents (independent PPO on per-agent rewards). "steps" (env_steps) = AGENT transitions.
- Model: `vec_dim`, `num_blocks`, `num_actions` from craftsim; new head `msg_head: Linear(hidden, 8)`.
  Action = (action, msg); log-prob and entropy are sums over the two heads.
- `--warm-start PATH`: load a v5 checkpoint's weights into the v6 model: tensors with equal shape copied;
  vec input layer columns remapped per 6.5; block embedding rows 0..17 copied (18,19 fresh); policy head
  rows 0..22 copied (23..27 fresh, bias set to the mean of old biases); everything else fresh. Counters
  and optimizer start fresh. A run started this way is a new run (new run_id), not a resume.
- Curriculum unchanged (per-world goals); success computed per world episode (count a world once, not K times).
- Metrics add: `agents`, `msg_entropy`, `msg_rate` (fraction of non-silent tokens), `chest_deposits`,
  `chest_withdraws`, `gives` (per 1k agent steps).
- Snapshot (§2) adds: `world_id, agent_id, agents_per_world`, `team: [{agent, x,y,z,yaw,pitch,health,
  last_action, msg, inventory{}}]` (ALL agents of that world incl. the selected one), `chests: [{x,y,z,
  items{}}]`, `chat: [{step, agent, msg}]` (last 32 non-silent tokens in that world). `select_env` takes an
  agent index. `hello.config` includes `agents`; `hello.world_size` = (128, 64, 128).
- Telemetry snapshot `blocks` is still the region around the selected agent (radius as before).

### 6.9 Dashboard
- World view: draw every `team` agent as a player-sized box in its color (palette 6.7) with a label
  "bot k", chests as brown boxes with a tooltip of contents; selected agent highlighted.
- Agent panel: team list (click to select that agent via `select_env`), each with inventory summary,
  last action, last msg; a "Chat" panel showing `chat` as lines "step N  bot k: m<msg>" colored per bot.
- Training panel shows `agents`, `msg_entropy`, `msg_rate`, `chest_deposits`, `chest_withdraws`, `gives`.
- Works unchanged against a v5 trainer (fields absent -> hide team UI).

### 6.10 Higher-resolution vision (added)
- Obs image resolution for v6 runs: `res = 64` (64x64 rays, was 32). `CraftSim(res=...)` stays a parameter;
  the craftsim.py default becomes 64. Same FOV as before (only more rays). Ray casting must stay fast:
  DDA voxel traversal, no per-ray allocation, teammate box test only for the <=4 teammates.
- Trainer: `--res` (default 64 for v6). The CNN keeps its 3 stride-2 convs (64 -> 8x8x32 = 2048 features
  into img_fc). Warm-start from a res-32 checkpoint: conv weights copied as-is; img_fc input weights
  expanded 4x4 -> 8x8 by giving each old spatial cell's weights to its 2x2 children divided by 4
  (equals average pooling at init, so the warm-started policy behaves like the old one at step 0).

### 6.11 Index conventions (clarification, binding for trainer + dashboard)
- snapshot `env_id` = global agent index i = world_id*agents_per_world + agent_id.
- snapshot `agent_id`, `team[].agent`, `chat[].agent` = within-world index k (0..K-1).
- `select_env.env_id` = global agent index i. metrics `agents` = agents per world (K).

---

## 7. Minecraft bot, v6 policy (single bot) — CURRENT

Mod `mcmod/` (mod id craftbot) + `mcbridge/bridge.py`. Instance: ElyPrismLauncher `aibot` (MC 26.2, Fabric
0.19.5, fabric-api 0.161.0+26.2) — NOT gtes. Build: `JAVA_HOME=~/.local/share/ElyPrismLauncher/java/java-runtime-epsilon ./gradlew build --offline`.
- `Obs.java` produces the v6 observation of §6.5/6.10 for one ServerPlayer: res 64 (same FOV/view distance as
  the sim: check src/sim.cpp writeObs for FOV, view distance, depth scaling and ray layout), NUM_BLOCKS 20
  (chest block -> 18; other players / bots -> AGENT 19 in rays), NUM_ITEMS 15 (chest item 14), VEC_DIM 102 with
  names/order exactly `craftsim.VEC_NAMES`. Teammate slots: other craftbot bots in the same level within 32
  blocks on the same team = the bots sorted by spawn order; with one bot they are zero. Nearest chest within 16
  blocks; crosshair chest contents from the real chest BlockEntity.
- `Obs` must be usable from any mod: pure static methods taking (ServerLevel/Level, position, yaw, pitch,
  inventory view, goal, step, teammates list) -> (byte[] img 64*64*2, float[] vec 102). No dependency on Bot.
- Actions 0..27 implemented in `Bot.java` with real Minecraft behavior: 23 craft_chest (8 planks, needs a
  crafting table within 3 blocks), 24 place_chest, 25 deposit (all non-tool items into the targeted chest),
  26 withdraw (all items from the targeted chest), 27 give (half of each non-tool stack to the nearest other
  bot or player within 4 blocks). Message tokens from the policy are shown above the bot's head as "m<k>"
  (nametag suffix) and fed to teammates' obs.
- Bridge: loads v6 checkpoints (and still v5/v2 ones as before), builds the model from the checkpoint's own
  obs signature/config (res, vec_dim, num_blocks, num_actions, msg head), answers
  {"action","msg","prob","value"}. Request carries "res" so the bridge can reject mismatches with a clear
  error. The mod asks the bridge for the checkpoint's res/vec layout on connect ({"cmd":"info"} ->
  {"res","vec_dim","num_blocks","num_actions","version"}) and sends that layout (v5 checkpoints: res 32,
  VEC 39, block ids >= 18 mapped: chest -> planks 7, agent -> air 0).

## 8. Human gameplay recorder (pretraining data, VPT / GEN-1.5 style) — CURRENT

Separate Fabric mod `mcrecorder/` (mod id minemind_recorder), installed in the aibot instance next to craftbot.
Purpose: record what a human player does so the policy can be pretrained on it (behavior cloning and an
inverse-dynamics model), then fine-tuned with PPO in the sim.

### 8.1 Controls
- Commands under the SAME root as the bot mod: `/aibot record start [goal]`, `/aibot record stop`, `/aibot record status` (register the `record` subtree on the `aibot` literal; Brigadier merges it with craftbot's), plus a keybind (default F8) toggling recording. HUD shows a
  red "REC 00:00 n frames" indicator while recording. goal = an ITEMS name (optional; written into the vec
  goal one-hot and meta; none = all zeros... use "none").
- Works in singleplayer (integrated server): client side reads raw input, server side computes the obs and
  game events for the same player; frames are joined by game tick.

### 8.2 Output: `<minecraft dir>/minemind_recordings/<yyyyMMdd-HHmmss>-<goal>/`
- `meta.json`: {format: "minemind-rec-1", mc_version, player, world_seed (if known), goal, start_time,
  ticks, frames, res: 64, vec_dim: 102, vec_names: [...], actions: [...28 names], items: [...15],
  blocks: [...20], frame_every_ticks: 1}
- one frame per game tick (20 Hz), fixed-size binary arrays (little endian), appended while recording:
  - `obs_img.u8`   [F,64,64,2]  the §6.5/6.10 v6 image obs (same Obs code as the bot mod; copy or share)
  - `obs_vec.f32`  [F,102]      the v6 vector obs (teammate slots zero)
  - `action.i32`   [F]          derived discrete MineMind action label for that tick (8.3), -1 = unknown
  - `input.jsonl`  one JSON object per frame: {tick, keys: {forward,back,left,right,jump,sneak,sprint,
    attack,use,drop,inventory, hotbar:1..9}, dyaw, dpitch (mouse-look delta this tick, degrees), yaw, pitch,
    x,y,z, on_ground, health, selected_slot, held_item, gui (null | "inventory" | "crafting" | "furnace" |
    "chest" | other screen class), inventory {item: count} (MineMind item names + "other")}
  - `events.jsonl` one JSON per game event: {tick, type: "break"|"place"|"craft"|"smelt"|"pickup"|"drop"|
    "chest_put"|"chest_take"|"damage"|"death"|"chat", block / item (vanilla id AND MineMind name or null),
    count, x,y,z}
- Flush at least every second; `/rec stop` (or quitting the world) finalizes meta.json (frames/ticks counts).
  A crash leaves readable arrays (F inferred from file size).

### 8.3 Action labels (human input -> one of the 28 MineMind actions, per tick; priority order)
craft/smelt/chest/give events this tick -> the matching action (craft_planks, craft_sticks, craft_table,
craft_wood_pickaxe, craft_stone_pickaxe, craft_furnace, smelt_iron, craft_iron_pickaxe, craft_chest,
deposit (chest_put), withdraw (chest_take), give (drop toward a player within 4 blocks)); place of
table/furnace/chest -> place_*; other place -> place_block; attack held while targeting a block -> attack;
|dyaw| >= 7.5 deg -> turn_left/turn_right (sim turn step: check src/sim.cpp for the per-action turn angle and
use half of it as threshold); |dpitch| >= that threshold -> look_up/look_down; forward+jump -> fwd_jump;
jump -> jump; forward/back/left/right -> fwd/back/left/right; else noop. Sim decisions are 4 ticks
(`ticks=4`); the converter (8.4) aggregates 4 frames into one sim-step label (majority, priority to rare
actions craft/place/attack).

### 8.4 Python side
- `pretrain/recordings.py`: load a recording dir (memmap arrays + jsonl), validate, aggregate to sim steps
  (every 4 ticks: obs from the first tick of the group, action from 8.3 aggregation), and iterate
  (img uint8 [T,64,64,2], vec float32 [T,102], action int64 [T]) sequences.
- `pretrain/bc.py`: behavior-cloning pretraining of the v6 MineMindPolicy (same model as trainer/model.py,
  GRU over sequence chunks of 64 steps, cross-entropy on action, optional --warm-start from a v6
  checkpoint), writes a normal trainer checkpoint (new run_id) that `trainer.run --resume`/`--warm-start`
  and the bridge can load. Report per-action accuracy. Test it on a synthetic recording generated by
  `pretrain/fake_recording.py` (writes the 8.2 format from the C++ sim with a scripted/random policy).

---

## 9. Recorder v2: free-play sessions (1-3 h), record everything, auto-classify — CURRENT

Replaces the per-task workflow of §8 (the §8 v1 format stays readable). The player just plays; nothing
needs a goal or restarting.

### 9.1 Session control
- Auto-start when the player joins a singleplayer world (config `autoStart`, default true; config file
  `config/minemind_recorder.json`). F8 = pause/resume (paused time is not recorded; HUD shows "REC" / "PAUSED").
  Commands stay: `/aibot record start|stop|pause|resume|status` (goal argument optional and ignored for labels).
- Stops on world quit / server stop / disconnect. One session dir per play session:
  `minemind_recordings/<yyyyMMdd-HHmmss>-session/`. Must run 3 h without slowing the game: measure
  per-tick cost (log a running avg every 5 min to the game log) and keep the server-thread cost < 2 ms
  average (raycast on a worker thread from a block snapshot, or every 2nd tick if needed — record which).

### 9.2 Storage (chunked + compressed so 3 h stays small)
- Frames grouped in 1-minute chunks (1200 frames): `chunks/000123.bin` = java.util.zip Deflater (zlib)
  stream of, per frame: img u8[64*64*2], vec f32[102] LE, action i32, ext_action i32 (9.4). Chunk index
  `chunks/index.jsonl`: {chunk, first_tick, frames, bytes}. A crashed session leaves all completed chunks
  readable (the current chunk is written as a temp file and renamed on completion; flush partial chunk on stop).
- `input.jsonl.gz`, `events.jsonl.gz` (GZIPOutputStream, flushed every ~5 s with SYNC_FLUSH) — same
  fields as §8.2 plus the new events below. `meta.json` rewritten every minute and finalized on stop.

### 9.3 Record everything (exact, not guessed)
Use the server player's statistics (StatsCounter / Stats.ITEM_CRAFTED, BLOCK_MINED, ITEM_USED,
ITEM_PICKED_UP, ITEM_DROPPED, ENTITY_KILLED, ENTITY_KILLED_BY, CUSTOM stats like damage dealt/taken,
jumps, eat...) diffed every tick as the source of truth for crafts/mines/uses/kills/pickups — exact
classification of every item and mob in vanilla ids. Plus Fabric events where useful (block break,
attack entity, use item). Every event has: tick, session_time_s, type, vanilla id, MineMind name (or
null), count, pos, and context (held item, gui).
Event types: `mine` (block id + drops), `craft` (item, count, via "inventory"|"crafting_table"),
`smelt` (furnace/smoker output taken), `pickup`, `drop`, `place`, `use` (item used: eat, bucket, bow,
shield...), `eat` (food item, hunger before/after), `attack` (entity type, weapon, damage), `kill` (entity
type e.g. minecraft:zombie, weapon), `killed_by`, `damage_taken` (source, amount), `death`, `respawn`,
`sleep`, `wake`, `chest_put`, `chest_take`, `trade`, `enchant`, `dimension`, `advancement` (vanilla id),
`level_up`, `chat` (player's own messages only).
### 9.4 Labels on EVERY tick (no "unknown")
- `ext_action`: a full VPT-style keyboard/mouse code that is always defined: bits 0..12 = keys
  (forward, back, left, right, jump, sneak, sprint, attack, use, drop, inventory, swap_offhand, pick),
  bits 13..16 = hotbar slot 0-8 (+1, 0 = no change), bits 17..21 = camera yaw bin (11 bins, mu-law like VPT,
  5 = none), bits 22..26 = camera pitch bin (same). Defined even when the client sample for a tick is
  missing (repeat the previous sample and mark `input.repeat=true`).
- `action` (MineMind 0..27): §8.3 mapping driven by the exact events of 9.3 (craft of a MineMind item ->
  craft_*; attack on a mob -> attack; eat/use -> noop-equivalent `use` maps to place_block only for block
  placement). Never -1: falls back to the movement/camera/noop mapping.
### 9.5 Auto-classification (Python, `pretrain/`)
- `pretrain/session.py` loads v2 sessions (and v1 dirs), exposes frames as memmaps/arrays per chunk.
- `pretrain/timeline.py <session>`: human-readable timeline + summary, e.g.
  `00:01:23 got oak_log (wood) x1`, `00:03:10 crafted wooden_axe`, `00:05:02 killed zombie with
  wooden_sword`, `00:05:03 got rotten_flesh (food)`, `00:07:40 ate cooked_beef`; categories for items
  (wood, stone, ore, tool:axe/pickaxe/sword/shovel/hoe/spear, armor, food, block, mob_drop, other) from
  vanilla ids/tags; counts per category; time spent per activity (mining / building / fighting / exploring /
  crafting / idle) segmented from events + inputs.
- Hindsight goals: for every sim step, the goal one-hot (vec[26..40]) is set to the NEXT MineMind item the
  player actually obtains in the future (within 5 min; else zeros) — so free play needs no task.
- `pretrain/bc.py`: reads v2 sessions + v1, uses hindsight goals, class-balanced loss (inverse-sqrt
  frequency weights, capped) so attack/noop don't swamp movement, keeps the chat head untouched (no
  "silent" pull) unless `--msg-silent`, sequence chunks across chunk-file boundaries.

### 9.6 Optional context notes (player-written labels)
- Recording works with no notes at all. Optionally the player can describe what they are doing:
  `/aibot context <free text>` (e.g. `/aibot context getting wood`, `/aibot context building a house`)
  starts a context segment; `/aibot context finish` ends it. A new `/aibot context <text>` while one is open
  ends the previous one and starts the new one. `/aibot context` with no args shows the current one.
  Pausing/stopping the recording ends an open context.
- HUD shows the current context under the REC indicator. Chat feedback: "Context: building a house".
- Stored as events `context_start` {text, id} and `context_end` {id, reason: "finish"|"replaced"|"stop"},
  and every frame's input.jsonl record carries `context` (text or null). meta.json lists all contexts with
  start/end tick.
- Python: session.py exposes context segments + per-step context text; timeline.py prints them as headers
  ("== 00:12:00-00:31:40 building a house ==") with the events inside and per-context summaries; bc.py keeps
  the per-step context text in the dataset (for a future language-conditioned policy; not used by the
  current model).

### 9.7 Canonical event / input field names (binding for Java writer + Python reader)
Every event: `tick` (int), `t` (session seconds, float), `type`, plus as applicable:
`item` (vanilla id "minecraft:oak_log"), `name` (MineMind item name or null), `count` (int),
`block` (vanilla block id, for mine/place), `drops` (list of {item, name, count}), `entity` (vanilla entity id,
for attack/kill/killed_by), `weapon` (vanilla id of held item or null), `damage` (float), `source` (damage
source id), `via` ("inventory"|"crafting_table"|"furnace"|...), `x`,`y`,`z` (ints), `held` (vanilla id or null),
`gui` (screen name or null), `text` + `id` (context events), `reason` (context_end).
input.jsonl(.gz) frames MUST be valid JSON on every line; a missing client sample repeats the previous
sample with `"repeat": true` (never emit partial objects like `"keys":{,`).

---

## 10. Population + breeding ("the 2 best bots breed and replace the other 2") — CURRENT

### 10.1 Population
- `--population P` (0/1 = off = today's single shared policy; P must equal `--agents` K): policy p controls
  team slot k = p in EVERY world. Each policy has its own MineMindPolicy weights, optimizer, lr and entropy
  coef; rollouts are shared, and each policy's PPO update uses only its own rows (W rows per policy). GRU
  hidden states stay per row. The env/sim is unchanged.
- Start: every policy is a copy of the start checkpoint (`--resume`/`--warm-start` of a single-policy v6
  checkpoint replicates it P times; resuming a population checkpoint restores all of it). Initial
  hyperparameter jitter: lr and ent_coef multiplied by a per-policy factor from {0.8, 1.0, 1.0, 1.25}
  (cycled), so the population is diverse from step 0.

### 10.2 Fitness and breeding
- Fitness of policy p = mean per-agent episode return of slot p (its own reward incl. team share) over the
  last evolve window, tie-break by team goal success of the episodes it played (all slots share the same
  worlds, so this is fair).
- Every `--evolve-every` agent steps (default 250,000; counted globally), rank the P policies:
  the TOP 2 are parents; each of the BOTTOM 2 is replaced by a child of the two parents:
  - weights: per-tensor uniform crossover (each named tensor taken from parent A or B with prob 0.5;
    a GRU's weight/bias tensors move as one unit so gates stay consistent), then Gaussian mutation
    `w += N(0, sigma * std(w))` with sigma = `--mutation` (default 0.01) on every floating tensor;
  - hyperparameters: lr and ent_coef from a random parent, each multiplied by a random factor in
    {0.8, 1.25}, clipped to [1e-5, 3e-3] and [1e-4, 0.05];
  - fresh optimizer state for the child; its rows' GRU states are zeroed;
  - record lineage: child id, generation, parent ids.
- With P = 4 this is exactly: 2 best breed, 2 worst are replaced. For P > 4, the bottom P/2 are replaced by
  children of random pairs from the top half.
- Log one line per evolution: ranking (id, fitness, lr, ent, generation) and which slots were replaced.

### 10.3 Telemetry / checkpoints / tools
- metrics add `population: [{slot, id, generation, parents, fitness, lr, ent_coef, success}]` and
  `generation` (number of evolutions so far). Snapshot unchanged (team[k] is policy k).
- Checkpoint: `models: [state_dict]*P`, `optimizers`, `population` metadata, plus `model` = the CURRENT BEST
  policy's state_dict so every existing loader (bridge, eval, bc warm-start, trainer --warm-start/--resume
  of single policy) keeps working unchanged and gets the best bot.
- eval.py: `--policy best|all|<slot>`; `all` runs teams with each slot driven by its own policy (default for
  population checkpoints), `best` puts the best policy in every slot.

---

## 11. Fine-tuning inside real Minecraft (online PPO through the bridge) — CURRENT

Goal: adapt a sim-trained v6 model (start: runs/pop-human/checkpoints/ckpt_000002007040_u000845.pt, its
`model` = best policy) to the real game. Heavy training stays in the sim; this is a slow, careful fine-tune.

### 11.1 In-game control (mod `craftbot`, aibot instance, singleplayer/integrated server)
- `/aibot train start [teams=2] [agents=4]` spawns `teams` teams of `agents` bots (fake players, same Bot
  machinery as `/aibot spawn`), each team at its own random surface spot >= 300 blocks from the player and
  from other teams (force-load the chunks around each team while training). `/aibot train stop` saves and
  removes the training bots. `/aibot train status` shows teams, episodes, steps/s, mean return, last checkpoint.
- `/aibot train speed <ticks-per-second>` (default 40, e.g. 20..100) via the server tick-rate manager
  (same mechanism as vanilla `/tick rate`); `/aibot train speed sprint` = vanilla tick sprint while training.
- The human player's recorder (minemind_recorder) only records the human; fake players are ignored.

### 11.2 Episodes and reward (mirror the sim, §6.6)
- Per team: goal sampled each episode from the curriculum goals up to iron (weights: equal over log, planks,
  stick, crafting_table, wooden_pickaxe, cobblestone, stone_pickaxe, furnace, coal, iron_ore; overridable by
  `/aibot train goals <item,...>`). Kits off. max 3000 decisions per episode.
- Reward per bot: team-first item achievements `0.1*(1+log2(value))` split by team_share 0.5 as in 6.6, goal
  `+10` split the same way, episode ends on goal or max steps. Items count when crafted/mined/smelted
  (inventory gain not from chest/give/pickup of another bot's drops — use the stats-diff approach of §9.3 for
  the fake players, or the mod's own action results).
- Reset: teleport the whole team to a NEW random surface spot >= 300 blocks from any used spot, clear
  inventories, full health + hunger, clear effects, daytime is not forced. Death: respawn the bot at the
  team spot and keep the episode going (its inventory drops are lost).
- Hostile mobs stay on (that's part of the real game).

### 11.3 Batched decisions (mod <-> bridge, TCP 7788)
- Each decision tick (every 4 game ticks) the mod sends ONE batched request for all training bots:
  `{"cmd":"train_step","bots":[{"id","team","img"(b64),"vec"[102],"reward","done","info":{...}}]}`, and the
  bridge answers `{"actions":[{"id","action","msg"}]}`. Reward/done are for the previous action. The play-mode
  `/aibot spawn` protocol stays unchanged and can run at the same time.

### 11.4 Bridge training mode
- `bridge.py --train --ckpt <v6 ckpt> --out runs/mc-ft [--lr 1e-5] [--rollout 128] [--kl-coef 0.1]`:
  same model + RecurrentPPO from trainer/ (reuse trainer/ppo.py, trainer/model.py, trainer/checkpoint.py);
  per-bot GRU states; collects rollouts of `rollout` decisions per bot, then one PPO update (bridge keeps
  answering; buffer the next requests during the update, or update in a background thread with a
  policy copy for acting — say which). Low lr, small clip, and a KL penalty to the frozen start policy
  (`--kl-coef`) so the sim skills are not forgotten from little data.
- Checkpoints in the normal trainer format (`runs/mc-ft/<run_id>/checkpoints/`) every 5 minutes and on stop;
  `trainer.eval` and the play bridge must load them.
- Telemetry server on 127.0.0.1:7777 (reuse trainer/telemetry.py): hello/metrics (env_steps = bot decisions,
  sps, episodes, avg_reward, success per goal, updates, losses, kl_to_start) and snapshots of a selected bot
  (team list etc. as in 6.8, blocks from the real world around the bot) so the dashboard works.
- Log a line per update: decisions, episodes, mean return, success, policy/value loss, entropy, kl_to_start.

---

# 12. v7 — "real Minecraft knowledge": data-driven registries, mobs, day/night, survival, nether, builds, multi-task worlds

Backup of all v6 sources: `backups/pre-v7.tar.gz`. Minecraft 26.2 data source (read-only): the jar
`~/.gradle/caches/fabric-loom/26.2/minecraft-common.jar` contains `data/minecraft/{recipe,advancement,
loot_table,structure,tags}/...` (1586 recipes, 1704 advancements, 1370 structure .nbt, 1389 loot tables).
Extract with `unzip -p <jar> data/minecraft/recipe/oak_planks.json`. Decompiled partial sources: `mcsrc/`.
Never modify the jar or mcsrc/.

## 12.1 Work phases and ownership (strict — only edit what you own)
Phase 1 (parallel):
- CORE (sim core agent): src/sim.h, src/sim.cpp, src/bindings.cpp, craftsim.py, tests/test_sim.py, CMakeLists.txt,
  build.sh, tools/gen_registry.py, src/gen/*. Creates module stubs `src/mobs.{h,cpp}`, `src/survival.{h,cpp}`,
  `src/nether.{h,cpp}`, `src/build.{h,cpp}` with the hook signatures of 12.4 (empty bodies) + calls them.
- DATA-1 registry: data/registry/*.json (12.2). DATA-2 tasks: data/tasks/*.json (12.5). DATA-3 builds:
  data/builds/** (12.6). DATA-4 knowledge + env mix: data/knowledge/*.json, data/envmix/*.json, tasklib/ (12.7).
- FT-1 mod train loop perf: mcmod Train.java, Obs.java, new ObsWorker.java. FT-2 bridge learning: mcbridge/mc_train.py.
  FT-3 sim-to-real parity: tools/parity/** (read-only elsewhere). FT-4 real-MC actions: mcmod Bot.java,
  CraftBotMod.java, PolicyClient.java. (13 below.)
Phase 2 (after CORE): SIM-1 mobs (src/mobs.*), SIM-2 day/night + survival (src/survival.*), SIM-3 nether +
portal + fire/lava (src/nether.*), SIM-4 builds runtime + worldgen variety (src/build.*, src/worldgen.* new).
Each phase-2 agent edits ONLY its module files + its own new test file tests/test_<module>.py; a hook it needs
that CORE did not provide is requested in its report, not hacked into sim.cpp.
Phase 3: trainer v7 + mod v7 obs/actions + dashboard palette, then a 2M-step multi-task run.

## 12.2 Registry data (DATA-1 writes, CORE consumes via tools/gen_registry.py -> src/gen/registry.h)
Scope: the subset of vanilla needed for survival through the nether (target 80-140 items, <= 120 blocks incl.
pseudo-blocks). All JSON, ids are vanilla names without "minecraft:" except where noted.
- `data/registry/blocks.json`: list of {name, id (0..), solid, transparent, liquid, hardness (vanilla seconds
  by hand base, from mcsrc/jar data), best_tool ("pickaxe"|"axe"|"shovel"|"hoe"|"sword"|null), min_tier
  (0 hand,1 wood,2 stone,3 iron,4 diamond), drops: [{item, count_min, count_max, chance, requires_tool_tier}]
  (from loot tables, simplified), light_emission, color [r,g,b] (for render/dashboard), dimension
  ("overworld"|"nether"|"both"), pseudo (true for non-world ids like mobs/agents)}. id 0 = air. Keep v6 ids
  0..19 for the same blocks where they exist (grass 1, dirt 2, stone 3, cobblestone 4, log->oak_log 5,
  leaves->oak_leaves 6, planks->oak_planks 7, crafting_table 8, furnace 9, coal_ore 10, iron_ore 11,
  diamond_ore 12, bedrock 13, sand 14, gravel 15, water 16, deepslate 17, chest 18, agent 19). Pseudo-blocks for
  mobs: one id per mob type (12.4 mobs).
- `data/registry/items.json`: {name, id, stack (1|16|64), kind ("block"|"tool"|"weapon"|"armor"|"food"|
  "material"|"misc"), places_block (block name or null), tool: {type, tier, speed, damage, durability} | null,
  food: {nutrition, saturation} | null, fuel_ticks, value (MineRL-style shaping value, powers of 2, by tech depth)}.
  Keep v6 item ids 0..14 (log=oak_log 0, planks 1, stick 2, crafting_table 3, wooden_pickaxe 4, cobblestone 5,
  stone_pickaxe 6, furnace 7, coal 8, iron_ore->raw_iron 9, iron_ingot 10, iron_pickaxe 11, diamond 12, dirt 13,
  chest 14). Wood: collapse all wood species to oak_* (log/planks/etc.) to keep the space small.
- `data/registry/recipes.json`: {id (0..), result, count, ingredients {item: n} (shapeless counts; tags
  resolved to our items, wood species -> oak), needs ("inventory" 2x2 | "crafting_table" | "furnace" |
  "smoker"/"campfire" -> treat as furnace)}. smelting recipes have needs "furnace", cook_ticks.
- `data/registry/mobs.json`: {name, pseudo_block, hostile, health, damage, speed, spawn: {light_max,
  dimension, biomes|null, day|night|any, group_min/max}, drops [{item,count_min,count_max,chance}], burns_in_sun,
  attack ("melee"|"ranged"|"explode"), size [w,h]}: at least zombie, skeleton, spider, creeper, cow, pig, sheep,
  chicken; nether: zombified_piglin, ghast(optional), blaze(optional).
- `data/registry/README.md` documents choices + exactly which jar files each entry came from.

## 12.3 Core sim changes (CORE)
- Registries generated from JSON at build time (tools/gen_registry.py -> src/gen/registry.h; build.sh runs
  it). NUM_BLOCKS/NUM_ITEMS/NUM_RECIPES/NUM_MOBS come from the data. Until DATA-1 lands, CORE ships a
  minimal data/registry/*.json seed covering v6 content (DATA-1 then overwrites/extends; keep ids stable).
- Dimensions: each world has overworld (128x64x128) and nether (96x48x96 default) block arrays; agent has a
  `dim` field; portal travel calls nether.h hooks.
- Factored actions (v7): `step(actions int32[N,4], ...)` columns = [primary, craft_recipe, item_select, msg].
  primary (`NUM_PRIMARY`): noop, forward, back, left, right, jump, forward_jump, sneak, sprint_forward,
  turn_left, turn_right, look_up, look_down, attack (hold until break / hit entity), use (place selected block /
  eat selected food / use selected item: flint_and_steel lights portal, bucket, bed -> sleep; on a crafting
  table/furnace/chest it's ignored — those use the dedicated actions), craft (recipe from column 1; needs
  checked), smelt (put recipe-1 input + best fuel into the furnace in reach; collect outputs on next smelt/
  withdraw), deposit, withdraw, give, drop. craft_recipe in 0..NUM_RECIPES (0 = none), item_select in
  0..NUM_ITEMS (0 = keep current). v6 compatibility: `CraftSim(..., action_mode="v6")` maps the 28 v6 actions.
- Obs v7 (`VEC_NAMES` generated): inventory log2(1+n)/7 for all items, selected item one-hot is NOT added
  (use an index scalar), scalars (health, hunger, saturation, on_ground, pitch, sin/cos yaw, height, in_water,
  light_level, time_of_day sin/cos, dimension, armor, mining progress, fuel, steps-left), TASK one-hot
  (NUM_TASKS from data/tasks, <= 64) + task-param slots (target item index/NUM_ITEMS, target count), teammates
  4 x 12 (as v6), nearest chest 4, crosshair-chest contents (9 most valuable items), nearest 3 mobs x
  [present, dx,dy,dz (yaw frame)/16, hostile, health frac], build-target 8 nearest unfilled cells x
  [dx,dy,dz /8, wanted item index/NUM_ITEMS] + build progress. Image stays 64x64x2 (block id incl. mob pseudo
  ids, depth).
- Tasks (12.5) loaded from data/tasks/tasks.json; each world gets ONE task per episode from the trainer's
  per-world task weights (`set_task_weights(world, w[NUM_TASKS])` and `set_task(world, task_id, params)`);
  reward = task milestones + task success + optional knowledge shaping (12.7); success ends the episode.
- Python: craftsim exposes ITEMS/BLOCKS/RECIPES/MOBS/TASKS (from the JSON), NUM_PRIMARY, `action_mode`,
  `task(i)`, `set_task_weights`, `set_task`, `entities(world)`, `time_of_day(world)`.
- Performance target: >= 5k agent-steps/s for W=32,K=4,res=64 with mobs on (sim only, 8 threads).

## 12.4 Module hooks (CORE creates stubs; phase-2 agents fill them)
```
namespace craftsim::mobs     { void init(World&); void tick(World&, Rng&); bool rayHit(const World&, ray..., float& t, uint8_t& id);
                               bool attack(World&, Agent&, float damage); void spawnTick(World&); void obs(const World&, const Agent&, float* out18); }
namespace craftsim::survival { void tick(World&, Agent&); bool eat(Agent&, int item); bool sleep(World&, Agent&); float lightAt(const World&, int x,int y,int z); }
namespace craftsim::nether   { void generate(World&, uint64_t seed); bool tryLightPortal(World&, int x,int y,int z); void travel(World&, Agent&); }
namespace craftsim::build    { void placeTarget(World&, int blueprint, int x,int y,int z); float progress(const World&); void obs(const World&, const Agent&, float* out33);
                               void worldgen(World&, uint64_t seed); }
```
(exact signatures may be refined by CORE; documented in src/<module>.h).

## 12.5 Tasks (DATA-2): `data/tasks/tasks.json`
List of {id, name, family ("gather"|"food"|"tools"|"build"|"combat"|"survival"|"nether"|"explore"),
description, params (e.g. {"item":"oak_log","count":8}), success: predicate, milestones: [{predicate, reward}],
max_steps, prerequisites [task names] (curriculum order), start_kit {item: n} (optional, for later stages),
source (which vanilla advancement/recipe inspired it)}.
Predicates (closed set CORE/SIM implement): `has_item {item, count}`, `crafted {item}`, `mined {block, count}`,
`killed {mob, count}`, `ate {food|any}`, `slept`, `survive_night`, `placed {block, count}`, `build_complete
{blueprint, min_progress}`, `portal_lit`, `in_dimension {dim}`, `reached_y {max_y}`, `distance_travelled {n}`,
`health_above {n}`. 40-64 tasks covering: gather (wood, stone, sand, gravel, coal, iron, diamonds, obsidian),
food (kill animals, cook meat, eat), tools (each tier: pickaxe/axe/sword/shovel), build (shelter, house, tower,
bridge, wall, nether portal frame, farm), combat (zombie, skeleton, spider, creeper), survival (survive night,
sleep in bed), nether (build portal, light it, enter nether, gather netherrack/quartz). Use vanilla
advancements (story/, husbandry/, adventure/, nether/) as inspiration and cite them.

## 12.6 Builds (DATA-3): `data/builds/`
- `blueprints.json`: list of {id, name, family ("shelter"|"house"|"tower"|"bridge"|"wall"|"portal"|"farm"|
  "decor"), size [x,y,z], palette {char: item_name}, layers: [[string rows]] bottom-up, anchor, required_items
  {item: n}, source}. 30-60 blueprints in 3 difficulty tiers (5-30 blocks, 30-120, 120-400), all made of items in
  12.2. Include a nether_portal frame (4x5 obsidian), simple dirt hut, cobble house, wooden house, watchtower,
  bridge, wall segment, farm plot.
- Convert a selection of vanilla structures (village houses from data/minecraft/structure/village/**.nbt —
  NBT gzip; write a small reader in tools/builds/nbt2blueprint.py) into blueprints mapped to our palette;
  record the source file.
- Procedural variants: tools/builds/variants.py generates size/material variants (scale, swap planks<->cobble).
- `data/builds/README.md`.

## 12.7 Knowledge + env mix (DATA-4)
- `data/knowledge/graph.json`: how to obtain every item: {item: {via: "mine"|"craft"|"smelt"|"kill"|"loot",
  from, tool_tier, recipe_id?, depth}} computed from 12.2 data by tools/knowledge/build_graph.py; plus
  `requirements(item)` -> full prerequisite set. Used for knowledge shaping: reward +0.05 * value the first
  time the agent obtains any item that is a prerequisite of the current task target (sim reads a generated
  table; CORE exposes `set_shaping_table(task, items[])`).
- `data/envmix/default.json`: per-world task-family mix, e.g. {"gather":0.25,"food":0.15,"tools":0.15,
  "build":0.15,"combat":0.1,"survival":0.1,"nether":0.1}, curriculum rules (unlock by prerequisites and success
  EMA), kit rules. `tasklib/` Python package: load tasks/blueprints/graph/envmix, `TaskSampler(worlds, mix)`
  assigning each world a family + task each episode, curriculum state, `describe(task)`. Pure Python, no sim
  import; unit tests in tasklib/test_tasklib.py.

# 13. Fine-tuning improvements (FT group, phase 1, v6-compatible now, layout-generic for v7)
- FT-1 (Train.java, Obs.java, ObsWorker.java): move per-bot obs raycasts off the server thread (snapshot ->
  worker pool), keep the server tick < 5 ms avg with 8 bots at speed 40; measure and log TPS/mspt every
  60 s in /aibot train status; adaptive speed (lower tick rate automatically if mspt > 45); stuck detection
  (no movement + no reward for 400 decisions -> teleport within 20 blocks); smarter far-spot selection (dry
  land, not ocean, not cave ceiling); keep §11 protocol.
- FT-2 (mc_train.py): proper PPO hygiene for online data: separate reward normalizer warmup (first 2 rollouts
  collect stats without updating policy), value-head warmup (critic-only updates for the first N updates),
  KL-to-start with adaptive coefficient targeting kl_to_start ~0.02, entropy floor, per-goal success EMA and
  automatic goal weighting (focus on goals with success between 10% and 80%), periodic checkpoint + "best"
  checkpoint by success, resume from an mc-ft checkpoint (`--resume`), dashboard metrics complete.
- FT-3 (tools/parity/): sim-to-real parity harness: (a) a Minecraft-side dump command spec + a Python tool that,
  given a dumped MC region (blocks around a bot, its pose/inventory) rebuilds the same scene in the sim
  (craftsim set blocks / pose) and compares the obs (img + vec) index by index, reporting mismatches by
  VEC_NAMES; (b) a report of the biggest sim-vs-real gaps (physics, mining times, drops, vision) with
  suggested domain-randomization knobs for the sim. Coordinate the dump format with FT-4 via the spec in your
  report; implement the Python side only.
- FT-4 (Bot.java, CraftBotMod.java, PolicyClient.java): real-MC action fidelity: attack holds until the block
  breaks (like the sim) with correct tool selection from inventory, crafting via the real recipe manager,
  placing/using items properly, eating when hungry is NOT automatic (policy must act), `/aibot dump <bot>`
  command writing the parity dump (blocks radius 16, pose, inventory, the bot's current obs) to
  minecraft/minemind_dumps/<time>.json, and a `/aibot train` safety net: bots that fall into lava/void are reset.

## 12.8 Social family (added): bots acting like real players with each other
- New task family `"social"`: worlds where the bots just behave naturally with teammates — crouch-greeting
  (sneak toggling while facing a nearby teammate, like players do), looking at each other, following a
  teammate, staying together, giving items, sharing a chest, idling/wandering together, emitting message
  tokens when close. Goal: realistic multiplayer-like behavior, not only task grinding.
- Extra predicates (CORE implements; counters per agent per episode):
  `near_teammate {dist, steps}` (within dist blocks of any teammate for >= steps decisions),
  `look_at_teammate {count}` (crosshair ray hits an AGENT pseudo-block, distinct decisions),
  `crouch_greet {count}` (>= 2 sneak toggles within 10 decisions while a teammate within 4 blocks is in view),
  `follow_teammate {steps}` (teammate in view and within 3..8 blocks while moving),
  `gave_item {count}`, `shared_chest {count}` (deposit then a teammate withdraws from the same chest),
  `msg_near_teammate {count}` (non-silent token while a teammate within 8 blocks).
  Rewards for social tasks are small and capped per episode so they can't be farmed endlessly
  (each milestone pays once, or at most N times with diminishing reward).
- DATA-2 adds 6-10 social tasks; DATA-4 adds `"social": 0.1` to default.json (renormalize the other families).

---

# 14. Scripted experts -> demonstration dataset -> v7 behavior cloning (GEN-1-style "basic game knowledge")

Goal: teach the v7 policy the basic skills it cannot discover by random exploration (choosing a recipe from 65,
aiming at moving mobs, placing blocks to a plan, eating, sleeping, social behavior) by imitating scripted
experts in the sim, then continue PPO from the cloned policy.

## 14.1 Hard rules for experts (so the demos are imitable)
- Experts control agents ONLY through the normal v7 action columns [primary, recipe, item, msg] via
  CraftSim.step. NO set_pose, set_inventory, teleports or block edits during an episode (start kits come only
  from the task definition / sim). Turning = turn_left/right and look_up/down actions (15 deg steps).
- Experts MAY read privileged state to decide (world blocks via block()/region(), entities(), state(),
  inventory(), recipes/registry, task params): teacher sees more than the student; that's fine.
- Experts must be robust: timeouts per sub-goal, recover from getting stuck (jump, back off, dig through,
  replan), give up cleanly. A failed demo episode is kept but flagged (success=false) so BC can filter.
- Deterministic given the seed; no wall-clock dependence.

## 14.2 Package `experts/` (Python)
- `experts/core.py` (EXP-1): Agent controller abstraction (`Ctl`) with primitives returning action rows:
  face(target xyz) via discrete turns, walk_to(xyz) with simple A*/greedy pathing over the block grid
  (jump 1-block steps, avoid water/lava/drops > 3), look_at_block, mine_block(xyz) (select best tool via the
  item column, hold attack), place_block(item, xyz) (select item, aim at the neighbouring face, use),
  craft(recipe_name) (recipe column; walk to/ place a crafting table or furnace when needed), smelt(input),
  eat(food), equip(item), attack_entity(mob id) (chase + aim + attack, keeping reach), flee(from xyz),
  wait. Generators/coroutines are fine (`yield action`). A `Runner` steps W worlds x K agents in lockstep,
  each agent driven by its own skill coroutine, batching one CraftSim.step per decision.
- `experts/skills/*.py` (EXP-2..4): one skill per task family/task (12.5 tasks), registered by task name:
  `SKILLS["gather_wood"] = fn(ctl, task) -> generator`. Composite skills reuse sub-skills
  (e.g. stone_pickaxe = wood -> planks -> table -> sticks -> wooden_pickaxe -> 3 cobblestone -> stone_pickaxe).
- `experts/record.py` (EXP-1): dataset writer. Shards `data/demos/<run>/shard_00000.npz` with arrays
  img uint8 [T,64,64,2], vec float32 [T,295], act int16 [T,4], done bool [T], task int16 [T], world int32 [T],
  agent int16 [T], ep int32 [T], success bool per step (filled at episode end), reward float32 [T]; plus
  `meta.json` (VEC_NAMES hash, NUM_* constants, counts per task, success rates, git-free version stamp).
  Sequences stay contiguous per (world, agent) inside a shard (split by episode) so BC can run a GRU.
- `experts/generate.py` (EXP-1): CLI `python -m experts.generate --tasks all|<list> --episodes-per-task N
  --worlds 32 --agents 4 --out data/demos/<run> [--seed S]`; reports per-task expert success and steps/s.
  Social tasks need K >= 2 (all agents in the world run social skills together).
## 14.3 BC v7 (BC-agent): `trainer/bc_v7.py`
- Loads shards, trains the v7 MineMindPolicy (trainer/model.py heads: primary/recipe/item/msg) with masked
  cross-entropy per head (same conditional/mask rules as PPO: recipe only when primary in craft/smelt, item
  only when primary in use/drop/give; heads.py masks), GRU over chunks of 64 steps, success-only filter
  option, class balancing on primary (inverse-sqrt, capped), per-task and per-head validation accuracy,
  optional --warm-start from a v7 checkpoint (runs/v7-multitask/...), writes a normal trainer checkpoint that
  `trainer.run --resume` (and population) can load. Optional mixing with human recordings later.
## 14.4 Success criteria
Expert success >= 80% on: gather_wood, gather_sand, tools_crafting_table, tools_wooden_pickaxe,
tools_stone_pickaxe, tools_wooden_axe/sword, food_hunt_* , survival_eat_any, combat_kill_zombie (night),
build_dirt_hut, social_stay_together/look/crouch_greet/follow; >= 50% on iron tools, furnace/smelting,
survival_sleep_in_bed, larger builds; nether tasks best-effort (may start from task kits).

---

# 15. Minecraft mod + bridge on v7 (play + in-game fine-tune of v7 checkpoints)

Backup: backups/pre-mod-v7.tar.gz. Target checkpoint: runs/v7-dapg2/checkpoints/ckpt_000009207808_u001124.pt (v7, population;
its `model` = best member). v6 play/fine-tune must keep working (bridge picks by checkpoint obs version; mod probes `info`).

## 15.1 Observation (Obs.java, v7 layout) — must equal the sim's VEC_NAMES/v7 layout exactly (src/README_v7.md, craftsim.VEC_NAMES)
- img 64x64x2: block ids from data/registry/blocks.json (59 ids; map real blocks by name incl. deepslate ore variants -> same ore
  ids, wood species -> oak_*; lava/water/obsidian/netherrack/... per registry; unknown -> closest structural or air);
  real mobs -> their mob pseudo-block id (zombie, skeleton, spider, creeper, cow, pig, sheep, chicken, zombified_piglin);
  other players/bots -> AGENT 19; depth like v6.
- vec 295: inventory of all 97 registry items (vanilla item -> registry item by name; wood species collapse), the 20 scalars in sim
  order (health, hunger, saturation, light, time of day sin/cos, dimension, ... — read writeObsV7 in src/sim.cpp), task one-hot
  (64) + task params, teammates 4x12, nearest chest 4, crosshair chest 9, nearest 3 mobs x6 (yaw frame), build slots 33 (zeros
  unless the episode task is a build task with a target — target support optional, zeros acceptable), with the same normalizations.
- Generated Java tables: tools/mod/gen_v7_tables.py -> mcmod/.../gen/V7Tables.java (blocks, items, recipes, tasks, VEC layout
  indices) from data/registry + data/tasks + craftsim, so Java never hand-copies ids.

## 15.2 Actions (Bot.java)
v7 action = [primary (21 names, craftsim.PRIMARY_NAMES), recipe (0 none, r+1), item (0 keep, i+1), msg]. Real-MC semantics like the
sim: movement/jump/sneak/sprint/turn 15 deg/look 15 deg; attack = hold until block breaks (best tool) or hit an entity in reach;
use = with the selected item: place block / eat food / flint_and_steel on obsidian frame / bed at night; craft = the registry
recipe -> real crafting (RecipeManager, needs table/furnace in reach as in the sim); smelt = furnace in reach; deposit/withdraw/
give/drop as v6. item_select (when != 0) selects that item into the main hand (swap from inventory to hotbar).

## 15.3 Tasks in-game (Train.java) and play
- Play: `/aibot spawn`, `/aibot task <task_id or item>` (v7: a task id from data/tasks; item names map to gather/has_item tasks).
- Train: each team gets a task from a configurable list (default: the v7 task families gather/tools/food/social + simple
  combat/survival that can be judged in-game) with success/milestones evaluated from the same predicates (has_item, crafted,
  mined, killed, ate, slept, placed, near_teammate, look_at_teammate, crouch_greet, ...) using MC stats-diff (as the recorder does)
  and positions. Reward = milestones + success like the sim (team_share 0.5). `/aibot train tasks <id,...>|default`.

## 15.4 Bridge
- Play request {"bot","reset","img","vec","res","v":7} -> {"action":[p,r,i,m],"prob","value"}; info returns
  {"version":7,"vec_dim":295,"num_blocks":59,"num_primary":21,"num_recipes":65,"num_items":97,"task_names":[...]} for v7 ckpts.
- Train mode: batched train_step as §11 with "action":[p,r,i,m] per bot; PPO with the v7 factored heads + masks (trainer/heads.py)
  via trainer's RecurrentPPO; all FT-2 features (warmups, adaptive KL to start, goal->task weighting, best checkpoint, resume).
