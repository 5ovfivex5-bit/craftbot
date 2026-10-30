# Publication preparation checks

Prepared on 30 October 2026. These notes describe what was actually checked in
the editing session; they do not claim a newly successful training run.

## Completed

- Inspected simulator/trainer entry points, checkpoint layouts and evaluation
  scripts before writing the tutorials.
- Before the request to use Distrobox for Python, the existing local virtual
  environment successfully imported Torch 2.14.0+cpu, NumPy 2.5.3 and the already
  built native simulator (59 blocks, 64 tasks). This was not a fresh native build.
- Inspected the selected local checkpoints: v6 has a 102-value observation,
  v7 has 295 values, both retain four policy members and full training state.
- Selected models using saved task comparisons, preserving failures and the
  distinction between population evaluation and single-member bridge inference.
- Copied those selected original checkpoints to the ignored `models/` folder.
- Checked modified shell scripts with `bash -n` and the staged source with
  `git diff --cached --check`.
- Reviewed staged file paths and scanned for common credential/private-key
  signatures. Raw runs, demonstrations, worlds, environments, nested ImGui
  checkout and decompiled source are excluded.

## Blocked in the editing session

The exact `distrobox enter hi` command reports:

```text
Failed to obtain podman configuration: set sticky bit on:
chmod /run/user/1000/libpod: read-only file system
```

No new container was created. Python commands added after that instruction are
targeted at `hi`; they were not silently run with host Python instead. The new
exporter/downloader, native rebuild, project test runner and tutorial training
commands therefore still need execution in the normal desktop environment.

GitHub CLI also could not connect to `api.github.com`. Authentication validity
could not be established from that failed network check. No remote repository
or release is claimed to exist as a result of this session.

## Finish from the normal desktop terminal

```bash
cd /home/deck/claude/ai/test/craftbot
./scripts/craftbot.sh build
./scripts/craftbot.sh test --full
./scripts/publish.sh
```

The publication script exports portable model files, reruns the core checks,
pushes the committed source and uploads release assets. It stops if those
checks fail. See [publishing.md](publishing.md) for the exact behavior.
