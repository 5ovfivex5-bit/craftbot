# Selected model cards

Two policies are recommended because **v6 and v7 are incompatible tasks and
observation formats**. Selection uses the existing saved evaluations, not the
latest filename, largest network or training reward. The original checkpoints
and full experiment directories remain local.

## At a glance

| | `best-v6.pt` | `best-v7.pt` |
| --- | --- | --- |
| Source run | `pop-human` | `v7-dapg` |
| Source checkpoint | `ckpt_000002007040_u000845.pt` | `ckpt_000009789440_u001195.pt` |
| Agent steps | 2,007,040 | 9,789,440 |
| Optimizer updates | 845 | 1,195 |
| Parameter tensors, elements per policy | 1,806,640 | 1,891,287 |
| Population | 4 retained policies | 4 retained policies |
| Image | 64×64 block-ID/depth | 64×64 block-ID/depth |
| Vector observation | 102 | 295 |
| Native bridge | Supported v6 format | Not supported |
| Training state | Retained | Retained |

The full-training artifacts contain a selected `model`, all population
`models`, optimizer states, counters, RNG and curriculum/task state. They are
larger than a single policy's weights. File size and SHA-256 are recorded in
the release's `manifest.json`; the downloader checks both. Export removes the
original project-directory prefix from stored paths without changing tensors.

These models learned in the local simulator. A v7 warm start can transfer v6
weights; the 9.79M v7 counter is not the total training compute across every
ancestor. The names `pop-human` and `v7-dapg` identify experiments, not new
foundation-model architectures. The source includes policy construction,
training, demonstration and fine-tuning code; no pretrained language model is
required to run these policies.

## v6: recommended crafting/bridge policy

The saved comparison evaluates **64 episodes per goal**, **32 parallel worlds**,
**4 agents per world**, seed **777**, sampled actions and the default population
mode (`all`). Goal start kits are not enabled in that command.

| Goal | Earlier `team-v6` | Selected `pop-human` |
| --- | ---: | ---: |
| Wooden pickaxe | 100.0% | 100.0% |
| Stone pickaxe | 68.8% | 95.3% |
| Furnace | 96.9% | 100.0% |
| Coal | 10.9% | 18.8% |

The selected model improves the saved four-goal comparison; coal remains weak.
This is not an all-goal sweep and does not establish diamond, long-horizon
survival or real-Minecraft performance. Best among these saved comparisons is
a narrower claim than best among every checkpoint on disk.

Reproduce a goal:

```bash
./scripts/craftbot.sh eval models/best-v6.pt --obs v6 \
  --goal stone_pickaxe --episodes 64 --envs 32 --agents 4 \
  --policy all --seed 777
```

Evidence: [original saved comparison](results/v6-population.txt). The export
retains the population because evaluating only `model` would change the test.

## v7: recommended multitask research policy

The saved comparison uses **16 episodes per task**, **16 parallel worlds**,
**4 agents per world**, seed **4242**, sampled actions and population mode `all`.
The baseline is `v7-multitask` at 5,005,312 steps; selected `v7-dapg` continues
that model with demonstration-assisted PPO. The selected endpoint is the one
actually referenced in the saved evaluation.

| Task | Baseline | Selected v7 | Later DAPG2, partial test |
| --- | ---: | ---: | ---: |
| Gather wood | 81.2% | 62.5% | 25.0% |
| Gather sand | 6.2% | 56.2% | 50.0% |
| Crafting table | 93.8% | 81.2% | 37.5% |
| Wooden pickaxe | 0.0% | 12.5% | 12.5% |
| Stone pickaxe | 0.0% | 0.0% | 0.0% |
| Hunt pig | 0.0% | 0.0% | Not recorded |
| Hunt sheep | 0.0% | 0.0% | Not recorded |
| Eat any food | 0.0% | 0.0% | Not recorded |
| Kill zombie | 0.0% | 0.0% | Not recorded |
| Build dirt hut | 0.0% | 0.0% | Not recorded |
| Crouch greet | 6.2% | 25.0% | Not recorded |
| Follow teammate | 0.0% | 0.0% | Not recorded |

Selected v7 trades some wood/table reliability for better sand gathering,
wooden-pickaxe and crouch-greeting results. Its mean over these 12 rounded
percentages is approximately 19.8%, versus 15.6% for the baseline. It does not
win each task. The later DAPG2 comparison is incomplete and is worse on the
first three recorded tasks, so recency is not the selection rule.

Reproduce a task:

```bash
./scripts/craftbot.sh eval models/best-v7.pt --obs v7 \
  --task gather_sand --episodes 16 --envs 16 --agents 4 \
  --policy all --seed 4242
```

Evidence: [v7 comparison](results/v7-dapg.txt), [partial later comparison](results/v7-dapg2-partial.txt),
[separate cloning comparison](results/v7-behavior-cloning.txt). These are
historical text outputs; they are not presented as newly rerun benchmarks.

## Evaluation boundaries

- Small episode counts make percentages noisy. The text reports rounded
  percentages; small differences can be one episode.
- Policies and populations were selected after experiments. A genuinely held-out
  final benchmark should use additional seeds/tasks untouched during selection.
- The saved v6 script describes the seed and the v7 script describes task flags;
  no claim is made that every saved seed was absent from all ancestor training.
- `--task all` draws a mixture over a shared episode budget. Run each task
  separately for fixed per-task quotas.
- `--policy best`, greedy actions, another agent count, inventory kits and real
  game execution each change the protocol.
- Failure cases are part of the model card. Zero scores on food, combat or
  building must not be hidden by a favorable gathering result.
- A fresh rerun should record the source commit, Torch/NumPy versions, native
  build flags, hardware, seed and exact command. CUDA experiments and the 100M
  model option are not the source of these released CPU-policy results.

## Continue training

```bash
./scripts/craftbot.sh train --obs v7 --no-server \
  --resume models/best-v7.pt --run-id continued-v7 \
  --no-demos --total-steps 11000000
```

This preserves population/training state and continues from the saved counter.
It is not an exact bit-for-bit continuation of simulator rollout state, and
`--no-demos` deliberately changes the training recipe. Supply newly generated
demonstrations to pursue the original demonstration-assisted approach. The
[training tutorial](training.md) explains those options.
