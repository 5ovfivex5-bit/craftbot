# CraftBot selected policies · models-v1

Two full training checkpoints accompany the source:

- **best-v6.pt** — `pop-human`, 2,007,040 agent steps, 1,806,640 parameter
  elements per policy, four retained population members. Recommended for legacy
  crafting goals and the v6 Minecraft bridge.
- **best-v7.pt** — `v7-dapg`, 9,789,440 agent steps, 1,891,287 parameter elements
  per policy, four retained population members. Recommended for current v7
  multitask experiments.

`manifest.json` records exact file sizes, SHA-256, dimensions and counters.
`SHA256SUMS.sha256` is also supplied. Use the repository's downloader to verify
assets automatically. Export preserves training state and removes the original
project-directory prefix from configuration paths.

Selection is based on saved historical comparisons, with mixed results and
known failures. It is not a claim of universal best performance or general
Minecraft mastery. See `docs/models.md` for the actual protocols, task tables,
provenance and limitations. Use matching `--obs v6` / `--obs v7` flags.

Code and curated checkpoints are MIT-licensed; Minecraft and dependencies retain
their own rights. Full training checkpoints require trusted PyTorch loading.
