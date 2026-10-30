# MineMind trainer

Start with the current guides:

- [Run CraftBot](../docs/quickstart.md)
- [Train, resume, clone and evaluate](../docs/training.md)
- [Selected model cards and measured results](../docs/models.md)
- [Architecture](../docs/architecture.md)

```bash
# Host entry point: runs Python inside distrobox hi.
./scripts/craftbot.sh train --obs v7 --no-server \
  --envs 8 --agents 1 --threads 2 --workers 2 \
  --total-steps 100000 --seed 42 --run-id first-agent
```

`model.py` defines the CNN/GRU agent, `ppo.py` the recurrent PPO update,
`run.py` collection/checkpointing/telemetry, `eval.py` evaluation, `tasks.py`
the v7 task scheduler, and `bc_v7.py` behavior cloning. `curriculum.py` retains
the v6 goal curriculum. `heads.py` implements the conditional v7 action heads.

Choose the observation version explicitly: v6 goal policies and v7 multitask
policies are incompatible. Full CLI help is available through the helper:

```bash
./scripts/craftbot.sh train --help
./scripts/craftbot.sh eval --help
```

The original development log is preserved in
[historical trainer notes](../docs/reference/trainer-history.md). Those notes
include earlier defaults and local paths; use the tutorials for current commands.
