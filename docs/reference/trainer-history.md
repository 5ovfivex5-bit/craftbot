# Historical MineMind trainer notes

This document preserves the original development notes. Some defaults, paths
and feature descriptions refer to older observation versions. For current
commands, use [the quickstart](../quickstart.md), [training tutorial](../training.md)
and [model cards](../models.md). Local paths below are historical examples.

Goal-conditioned recurrent PPO (observation v2) over one shared policy batched across all sim envs,
a goal curriculum over the crafting tech tree, and a TCP JSON-lines telemetry server for the dashboard.
Run everything inside the distrobox:

```
distrobox enter hi -- bash -c 'cd /home/deck/claude/ai/test/craftbot && .venv/bin/python -m trainer.run --envs 64'
```

## Modules

| file | role |
|---|---|
| `run.py` | CLI + `Trainer`: state machine (idle/running/paused/stopping/stopped), command handling, metrics at 2 Hz, snapshots, boundary-queued config |
| `model.py` | `MineMindPolicy`: block embedding + depth -> small CNN, vector MLP, fused hidden layer, GRU(256), policy and value heads; `describe()` for the `model` message |
| `ppo.py` | rollout buffer, batched inference (`policy_act`), GAE, truncated-BPTT PPO update over chunks of 16 steps, `ReturnNormalizer` |
| `curriculum.py` | `GoalCurriculum`: per-goal success EMAs (scratch, kit, restart), unlock schedule, goal weights, start-kit table and probabilities |
| `goexplore.py` | Go-Explore archive: cells (goal, achievement mask, depth bucket) with sim snapshots; restart cell choice and goal |
| `eval.py` | `python -m trainer.eval`: per-goal success and decisions-to-goal on fresh seeds (no kits unless `--kits`) |
| `envs.py` | adapter over `craftsim.CraftSim`; takes VEC_DIM / NUM_BLOCKS / WORLD_SIZE / goal offset from craftsim; `obs_signature()` |
| `telemetry.py` | threaded TCP server: per-client outbox, latest-wins slots for snapshot/activations, bounded deque for the rest, command queue |
| `instrument.py` | forward hooks for one selected env at `nn_hz`; hooks registered only while a client subscribes to `nn` |
| `checkpoint.py` | atomic save (tmp, fsync, `os.replace`), load, list |
| `testclient.py` | dashboard stand-in: subscribes, prints message rates, sends commands |
| `bench.py` | throughput table vs num_envs, `--profile` (collect vs update split), and the dashboard-impact comparison |

## CLI

```
python -m trainer.run [--envs 64] [--total-steps N] [--rollout 64] [--lr 3e-4] [--minibatch 512]
                      [--ckpt-every N] [--port 7777] [--mode fast|live|debug] [--seed 0]
                      [--workers 4] [--snapshot-hz 4] [--nn-hz 2] [--no-visualize]
                      [--lr-anneal] [--ent-coef-end X]
                      [--goal-start 2] [--unlock-at 0.5] [--goal-min-episodes 30]
                      [--resume runs/<run_id>/checkpoints/ckpt_....pt]
                      [--no-server] [--autostart] [--runs-dir runs] [--run-id ID] [--threads N]
                      [--no-kits] [--novelty 0.02] [--novelty-end-frac 0.6]
                      [--go-explore|--no-go-explore] [--restart-frac 0.25]
                      [--rnd|--no-rnd] [--rnd-coef 0.1] [--value-loss hlgauss|mse]
                      [--aux-ach|--no-aux-ach] [--aux-coef 0.1] [--bptt 32]

python -m trainer.eval runs/<run>/checkpoints/ckpt_....pt [--episodes 64] [--goal all|ITEM]
                      [--seed 900001] [--envs 64] [--greedy] [--kits]
```

Fresh v2-style run (5M steps; about 25 min at ~3.5-4k env-steps/s on this 8-CPU box):

```
distrobox enter hi -- bash -c 'cd /home/deck/claude/ai/test/craftbot && .venv/bin/python -m trainer.run --no-server --envs 64 --total-steps 5000000 --lr-anneal --ent-coef-end 0.003 --run-id v2-full'
```

Recommended continuation from the v2 checkpoint with everything from section 5 on: see the
"Resuming from v2" section below.

- `--no-server` implies `--autostart`: headless training, exits when `--total-steps` is reached.
- Without `--autostart` the trainer waits in `idle` for a `start` command from a client.
- Ctrl+C / SIGTERM: finish the current update, write a final checkpoint, exit.
- `--resume` restores model, optimizer, counters (env_steps, updates, episodes), RNG states, curriculum
  and reward-normalizer state, and the config stored in the checkpoint; explicit CLI flags override it.
- Checkpoints from the v1 trainer (25-dim obs, 14 blocks, no goal) are rejected with a message naming
  the mismatch. Any checkpoint whose `obs` signature differs from the running sim is rejected the same way.

## Goal-conditioned training (obs v2)

- The sim samples one goal item per episode (weights from `set_goal_weights`); the goal is part of the
  observation (one-hot in `vec[25:39]`). Reward: shaping for first-time items, +10 and episode end on
  the goal. `ach` bit 31 = goal achieved.
- `trainer/curriculum.py` tracks a success EMA per goal (from ach bit 31 and the goal of each finished
  episode). Recording starts after every env has run two full episode windows (2 x 3000 decisions per env):
  before that, only timeouts show failures, so early success rates are biased upward. Goals are unlocked in tech-tree order: `log, planks, stick, crafting_table, wooden_pickaxe,
  cobblestone, stone_pickaxe, furnace, coal, iron_ore, iron_ingot, iron_pickaxe, diamond`. The first
  `--goal-start` (2) are unlocked at start; the next unlocks when the frontier (highest unlocked) reaches
  `--unlock-at` (0.5) success after `--goal-min-episodes` (30) episodes of that goal.
- Each rollout sets the goal weights: locked goals 0; unlocked goals get `0.02` floor plus 1.0 in the
  learning zone [0.1, 0.9], 0.5 below it (too hard so far), 0.25 above it (mastered); the frontier gets
  `+2.0` extra. Unexplored unlocked goals get full weight.
- Telemetry: `curriculum_stage` is a string such as `"cobblestone (6/13 unlocked)"` (frontier goal),
  `curriculum_task` is the frontier name, `goal_success{item: EMA}` is the per-goal success EMA (for
  goals sampled so far), `success_rate{item}` remains the item-obtained rate over the last 200 episodes.
- Returns are scaled by the running std of the discounted return (`ReturnNormalizer`, clip 10); raw
  rewards are still logged as `step_reward`. `lr` and `ent_coef` in metrics show the current schedule.
- `--lr-anneal`: linear lr decay from `lr` to 0.1*`lr` over `total_steps`. `--ent-coef-end X`: linear
  entropy-coefficient schedule from `ent_coef` (0.01) to X. Both off by default.

## v3: start kits and novelty (PROTOCOL section 4)

- `curriculum.KIT_SPEC`: per goal from `cobblestone` on, the items an episode starts with (never the goal
  item). Kit probability per goal is `kit_prob_for(no-kit success)`: 0.5 while the no-kit success is
  <= 0.2 (and for never-tried goals), linear down to 0.1 at >= 0.8.
- Kit episodes update only the kit EMA (`goal_success_kit`). Unlocking: the frontier unlocks when its no-kit
  EMA >= `unlock_at` (normal rule) OR its kit EMA >= 0.5 (`KIT_UNLOCK_AT`), so the chain does not block
  learning the next skill. Kit episodes are excluded from `success_rate` (kit items set no ach bits).
- Novelty bonus (sim): `--novelty` 0.02 decays linearly to 0 over `--novelty-end-frac` (0.6) of the steps
  left at process start (so resuming at 12M with total 13M decays over the 1M steps of that run). Off when
  RND is on.
- Checkpoints store the curriculum kit fields (`ema_kit`, `n_kit`), defaulted when absent (v2 checkpoints).

## v4: research upgrades (PROTOCOL section 5)

- **Go-Explore** (`goexplore.py`, Ecoffet et al. 2019): cell = (goal, achievement mask, depth bucket), from
  the step's `ach` mask and the height in `vec[20]` (no extra sim calls). A snapshot (`save_state`, ~270 KB)
  is kept per cell, the one reached in the fewest steps; at most 1000 cells. With `--restart-frac` (0.25)
  a finished episode restarts from a cell chosen with weight (1 + tech rank) / sqrt(times chosen + 1); the
  restarted goal is the first unlocked tech-tree item the cell lacks. Restart episodes go to
  `goal_success_restart` only. The archive is not checkpointed; it is rebuilt by each process.
- **RND** (`--rnd`, Burda et al. 2018): fixed random target and trained predictor MLP on the detached fused
  feature. The raw error is divided by its running std, clipped at 5, and scaled by `--rnd-coef` (0.1) with
  the same linear decay as the novelty schedule. Intrinsic value stream: `int_value_head` (MSE) or
  `int_value_cls` (HL-Gauss), gamma_int 0.99, non-episodic GAE; advantages are extrinsic + intrinsic.
- **HL-Gauss** (`--value-loss hlgauss`, default; Farebrother et al. 2024): 61 bins over [-30, 30] in
  scaled-return units, target = Gaussian mass per bin with sigma = 0.75 bin width, value = expectation.
  `--value-loss mse` uses the scalar `value_head` / `int_value_head`.
- **Aux achievement prediction** (`--aux-ach`, Moon et al. 2023): `aux_head` predicts the next item newly
  obtained in the same episode (14 items + none) with cross-entropy, coefficient 0.1. Targets are computed
  per rollout; an event after the rollout end counts as none.
- Checkpoints: new modules are appended after the v2 parameters; `load_state_dict(strict=False)` leaves them
  fresh, and optimizer moments are loaded for the old parameters only (`load_optimizer_state`). `bptt`
  from an old checkpoint is ignored on resume (default 32 applies unless `--bptt` is given).

Telemetry (metrics, all additive): `kit_rate`, `restart_rate`, `novelty_reward` and `novelty_scale`
(per env-step means), `rnd_reward` / `rnd_coef` / `rnd_loss`, `aux_ach_acc` / `aux_loss`,
`goal_success` (scratch only), `goal_success_kit`, `goal_success_restart`, `archive_size`, `archive_bytes`.

## v7: multi-task worlds (obs v7, default)

`--obs v7` (default) or `--obs v6` (the 102-dim, 28-action layout, unchanged). The obs version is read
from the command line before the sim is imported (`trainer/__init__.py`), so one process runs one layout.

- **Heads** (`model.py`, `heads.py`): primary (21) on the GRU output, recipe (66: 0 none, r + 1), item_select
  (98: 0 keep, i + 1) and msg (8). Recipe is active for craft/smelt; its logits are masked to recipes whose
  ingredients the inventory holds (counts decoded from vec[0:97] = log2(1+n)/7). Craft/smelt never choose "none";
  if no recipe is feasible the mask is lifted and the sim rejects the craft. Needs-table / furnace is not checked.
  Item select is active for use/drop/give, masked to items with count > 0 plus keep.
- **Inactive heads** are restricted to their neutral value (recipe 0 / item keep), so they add exactly zero to the
  log-prob and entropy and the sim receives the neutral value. Their logits are never sampled into the env. The
  masks are a function of (vec, primary), so the PPO update recomputes them. Entropy bonus = primary + active
  recipe + active item (msg is separate as in v6).
- **Tasks** (`tasks.py`, tasklib): each world has one family (largest-remainder allocation of the envmix
  proportions over the families with an unlocked task; re-allocated at update boundaries and when a family is
  gated). The world's task is chosen inside its family by the tasklib curriculum (prerequisites + success EMA).
  The sim draws a world's task at its own auto-reset from that world's weights, which happens inside the step
  that ends the previous episode, so the scheduler keeps one episode of look-ahead: the next task is queued
  and its one-hot weights are set when the previous one is queued. The curriculum therefore acts one episode
  after the outcome it is based on. Start kits (tasklib) go in with `set_inventory` and the vec inventory block
  is patched to match. Knowledge shaping: `set_shaping_table` for every task at each sim build.
- **Warm start** (`warmstart.py`, v6 checkpoint into v7): vec columns by name (75 of 295 mapped from v6's 102),
  block rows 0..19 by name, primary rows by name for noop, movement, jump, forward_jump, turn/look and attack
  (12 of 21); deposit/withdraw/give and the rest are fresh. Every other equal-shape tensor is copied; aux_head,
  spr_trans and the new recipe/item heads are fresh. The population copies the warmed policy to every slot.
- **Checkpoints**: `obs` signature with `obs: "v7"`, `tasks` (families, sampler EMA / episode counts, RNG). Loading
  a checkpoint of the other layout is refused and names the flag to use. `curriculum` is None for v7.
- **Telemetry** (additive): `family_mix`, `family_worlds`, `family_episodes`, `family_success`, `family_return`,
  `task_success`, `task_episodes`, `task_ema`, `tasks_unlocked`, `tasks_mastered`, `family_switches`, `msg_hist`,
  `recipe_entropy`, `item_entropy`, `build_progress`, `health_mean`. Snapshot: `task`, `family`, `time_of_day`,
  `entities` (from the sim; empty until the mobs module lands). Kills, deaths, eats, sleeps and portal are not
  exposed by `state()` and are not reported.
- **Evaluation**: `python -m trainer.eval CKPT --task NAME` or `--family NAME` (`--episodes` per world quota).
- v7 does not run the v6 goal curriculum, Go-Explore, SPR or the aux achievement head.

## Resuming from v2

```
distrobox enter hi -- bash -c 'cd /home/deck/claude/ai/test/craftbot && .venv/bin/python -m trainer.run --resume runs/v2-full/checkpoints/ckpt_000012001280_u002930.pt --no-server --autostart --envs 64 --total-steps 13000000 --run-id v4-full'
```

## Outputs

```
runs/<run_id>/run.json                       run metadata
runs/<run_id>/history.jsonl                  one metrics row per tick (0.5 s), append-only
runs/<run_id>/checkpoints/ckpt_<steps>_u<updates>.pt
```

Checkpoints hold: `schema_version` (2), `obs` (observation signature), `model`, `optimizer`, `config`,
`counters`, `curriculum` (unlock state, EMAs, counts), `curriculum_stage`, `reward_norm`, `rng`
(torch / numpy / python), `metadata`.

## Evaluation

```
python -m trainer.eval CKPT --episodes 64 --goal all
```

Prints a table: goal, episodes, success rate with a 95% interval, and decisions-to-goal (mean and median
over successful episodes). Episodes use worlds from seeds that training never used and sampled actions
(`--greedy` for argmax). `--goal ITEM` evaluates one goal.

## Protocol

Implements `PROTOCOL.md` section 2 (server side). Every command is acked with `ack{id, ok, msg}`.
Config changes are validated immediately and applied at the next rollout boundary (after an update,
or immediately when paused/stopped with no partial rollout). `num_envs`, `rollout_len` and `workers`
rebuild the sim and buffers at that boundary; model and optimizer are kept.

`debug_step` requires `paused`. `kind: "tick"` applies a noop action to a dedicated debug env seeded
from the selected env's seed; `kind: "decision"` samples the policy. The reply is a `snapshot` with
`env_id: -1`.

## Derived fields

- `sps` is an EMA (alpha 0.3) of env steps per second over ticks with progress. `eta_s` is
  `(target - env_steps) / sps`, null until 5 samples.
- `curriculum_stage` / `curriculum_task` come from the goal curriculum (see above), not from item rates.
- `mem_warning` appears when `num_envs x per-env estimate` exceeds 70% of available RAM.

## Test client and bench

```
python -m trainer.testclient --port 7777 --subscribe --duration 10 --send start --send pause --send resume
python -m trainer.bench                 # throughput table
python -m trainer.bench --impact        # training sps with / without a subscribed client
```

## Performance notes

- Collection is dominated by policy inference (about 4-5 ms per step at 64 envs with 4 torch threads);
  the sim step is about 1.5-2.5 ms. The PPO update is the largest share (about 2/3 of loop time).
- Block-id embedding uses `_EmbedGrad` (index_add backward), about 2.5x faster than the dense embedding
  backward on CPU, with identical gradients.
- Defaults: `minibatch_size` 512 and `ppo_epochs` 1 (8 Adam steps per 4096-step rollout; the v1 default
  of 2 epochs x 1 minibatch of 4096 took 2 steps). More torch threads did not speed up the update.

## Limits

- Debug mode uses a fixed run seed for env (re)builds. The sim's own auto-reset draws from its
  internal RNG, so episode seeds after a reset are reproducible only for the same action sequence.
- `eval_checkpoint` runs in a background thread with its own sim and policy copy; it shares the CPU
  with training while it runs.
- A `pause` received during a PPO update takes effect when that update finishes (updates are not interrupted).
- Locked goals are never sampled, so their success EMA stays at its last value (or unset); `goal_success`
  only lists goals that have been sampled.
