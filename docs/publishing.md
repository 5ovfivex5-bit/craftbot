# Publish the reviewed project

The publication script targets `5ovfivex5-bit/craftbot` by default. It runs in
the same `hi` distrobox as training, so GitHub CLI authentication stays in your
normal environment. It does not copy tokens into the repository.

From the original local training workspace:

```bash
./scripts/publish.sh
```

The script checks GitHub authentication, exports both selected full checkpoints,
runs the project checks, requires a clean committed working tree, creates a
public repository if absent, pushes `main` and creates the `models-v1` release.
It refuses to change an existing private repository's visibility or overwrite
an existing model release. Passing `OWNER/REPO` selects another destination.

Before running, the project must already have a reviewed source commit. Bulk
training runs, raw recordings, models, Minecraft worlds, binaries, virtual
environments, nested dependencies and decompiled source remain excluded by
`.gitignore`. Checkpoint files are uploaded only as release assets.

To inspect publication contents:

```bash
git status --short
git ls-files
```

If a test fails, the script stops before publishing. Inspect the failure, fix it
and commit the correction. Some legacy checks have machine-dependent speed
thresholds; do not silently ignore a failed correctness assertion.

If an agent session cannot access Podman's runtime directory or GitHub's
network endpoints, run the command in your normal desktop terminal. A prepared
local repository is not described as public until the remote upload succeeds.
