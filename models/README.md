# Selected policies

Run `./scripts/craftbot.sh download-models` after setting up the project. The
public `models-v1` GitHub release supplies `best-v6.pt`, `best-v7.pt` and a
SHA-256 manifest. Weights are release assets so cloning the source stays small.

The maintainer can rebuild those release assets from the retained local runs
using `./scripts/craftbot.sh export-models`. Full policy populations, optimizer
state and counters are retained. Machine-specific project prefixes are removed.

See [the model cards](../docs/models.md) for selection criteria, historical
results, matching observation flags and limitations. Loading full training
checkpoints uses PyTorch pickle; use the project's verified release or your own
trusted local checkpoints.
