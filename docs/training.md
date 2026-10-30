# Train your agent

All helper commands below run Python inside `distrobox enter hi`. First complete
[setup and the native simulator build](quickstart.md).

## 1. Understand the two formats

| | v6 | v7 |
| --- | --- | --- |
| Structured observation | 102 values | 295 values |
| World image | Block-ID + depth channels | Block-ID + depth channels |
| Actions | 28 legacy actions + message | 21 primary actions + recipe/item/message heads |
| Conditioning | Goal item | Task and task parameters |
| Recommended use | Legacy crafting; real-game bridge | Current multitask research |

The default is v7. Pass `--obs` explicitly in reproducible commands. A v6
checkpoint cannot be resumed as v7; the optional warm-start mapper transfers
compatible tensors into a new v7 run.

## 2. A small first PPO run

```bash
./scripts/craftbot.sh train --obs v7 --no-server \
  --envs 8 --agents 1 --threads 2 --workers 2 \
  --total-steps 100000 --seed 42 --run-id first-agent
```

- `--envs` counts parallel **worlds**. `--agents` counts agents per world.
- Environment-step counters sum agent transitions across worlds; they are not
  optimizer-update counts.
- `--threads` controls PyTorch threads; `--workers` controls native simulator
  OpenMP threads. More threads can be slower on a small CPU.
- `--no-server` starts immediately and disables dashboard telemetry.
- Fixed seeds make experiments easier to reproduce; learning performance still
  depends on software versions, settings and evaluation protocol.

This is a plumbing run, not a promise of useful crafting performance. Look at
held-out task success, not just reward curves or training duration.

## 3. Longer training and live inspection

```bash
./scripts/craftbot.sh train --obs v7 --autostart --mode live \
  --envs 16 --agents 4 --threads 4 --workers 4 \
  --rollout 64 --minibatch 512 --bptt 32 \
  --total-steps 5000000 --seed 42 --run-id multitask-5m
```

Four agents share each world. Task families and prerequisites are defined in
`data/tasks/`, `data/envmix/` and `data/knowledge/`. The default environment mix
controls the distribution; a curriculum unlocks more tasks as evidence permits.
This command trains a shared policy unless population training is requested.

For a per-agent policy population, add `--population 4` with `--agents 4`.
Population evolution and multi-agent interaction change the experiment, so keep
them fixed when comparing checkpoints. The selected historical policies use
four population members.

Open Studio in another terminal using the quickstart instructions. Start with
the small model; the `100m` option is a different architecture, consumes much
more memory and has no recommended released result here. The current trainer
rejects `--backend gpusim` for the current observation layouts.

## 4. Find and resume a checkpoint

Output layout:

```text
runs/<run-id>/
  run.json
  history.jsonl
  checkpoints/ckpt_<agent-steps>_u<updates>.pt
```

Example continuation from the selected v7 model:

```bash
./scripts/craftbot.sh train --obs v7 --no-server \
  --resume models/best-v7.pt --run-id my-v7-continuation \
  --no-demos --total-steps 11000000 --threads 4 --workers 4
```

That checkpoint already has **9,789,440** steps. `--total-steps 11000000` means
continue to eleven million lifetime steps, not add eleven million steps. A new
run ID keeps new logs/checkpoints separate from the published model.

The release keeps full populations, optimizer state and curriculum state.
Explicit flags override restored settings. `--no-demos` disables the original
training machine's demo paths; to continue demonstration-assisted learning,
generate data locally and supply your own `--demos` paths instead. Ctrl+C
requests a graceful stop and checkpoint at an update boundary.

## 5. Generate expert demonstrations

Scripted experts control agents through normal simulator actions. They may read
privileged simulator state, so expert success is not a learned-policy score.

```bash
./scripts/craftbot.sh experts \
  --tasks gather_wood,tools_crafting_table \
  --episodes-per-task 32 --worlds 8 --agents 1 \
  --out data/demos_raw/my-experts
```

The generated directory contains `.npz` shards and metadata. Keep it out of Git.
Skills and success predicates are documented in [experts/README.md](../experts/README.md).
Add task coverage deliberately; a narrow demonstration set does not teach every
skill in the simulator.

## 6. Behavior cloning, then reinforcement learning

```bash
./scripts/craftbot.sh clone --demos data/demos_raw/my-experts \
  --out runs/bc --run-id my-clone --steps 1000 --batch 8 --chunk 64
```

The cloning trainer writes a regular checkpoint under:

```text
runs/bc/my-clone/checkpoints/ckpt_000000000000_u000000.pt
```

That file starts with zero PPO counters even though cloning has trained its
weights. Continue it with:

```bash
./scripts/craftbot.sh train --obs v7 --no-server \
  --resume runs/bc/my-clone/checkpoints/ckpt_000000000000_u000000.pt \
  --run-id clone-then-ppo --total-steps 1000000
```

To combine expert losses with PPO, add:

```bash
--demos data/demos_raw/my-experts --demo-coef 0.15 --demo-coef-end 0.02
```

Evaluate the resulting policy. The saved experiments include regressions after
cloning or further training; more updates and lower demonstration loss do not
guarantee higher task success.

## 7. Evaluate fairly

```bash
./scripts/craftbot.sh eval models/best-v7.pt --obs v7 \
  --task tools_crafting_table --episodes 64 --envs 16 \
  --agents 4 --policy all --seed 900001
```

Use separate `--task` invocations for a fixed episode budget per task. Keep seed,
agent count, action sampling, population mode and starting inventory consistent.
`--policy all` assigns each agent slot its population member; `--policy best`
repeats the saved selected member and is a different measurement. Do not compare
v6 and v7 percentages as if their tasks and worlds were identical.

The evaluator prints an approximate normal 95% interval. At 0%/100%, that
approximation understates uncertainty; use a Wilson or exact binomial interval
for formal reporting. Sixteen-episode historical tests are useful diagnostics,
not precise benchmark estimates. Repeat promising results over multiple seeds.

## 8. Keep experiments organized

Give each comparison a distinct run ID, retain its CLI/settings and choose
checkpoints by evaluation rather than filename recency. Store small curated
tables under `docs/results/`; retain bulk logs, recordings and checkpoints under
the ignored `runs/` and `data/demos_raw/` directories. Project cleanup preserves
those local artifacts instead of deleting your training history.
