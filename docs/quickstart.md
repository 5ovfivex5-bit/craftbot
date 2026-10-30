# Run CraftBot

The shortest path is **build the simulator → download a policy → evaluate it**.
Minecraft and the dashboard are optional.

## 1. Prepare the environment

The supported local workflow uses your existing Arch-based `hi` distrobox:

```bash
distrobox enter hi
```

Inside `hi`, install the native build tools if they are missing:

```bash
sudo pacman -S --needed base-devel cmake ninja
```

Use Python **3.12 or newer**. `uv` can create a managed 3.12 environment; without
uv the setup script expects `python3.12` on PATH. Then leave the container or
open another host terminal in the cloned `craftbot` folder:

```bash
./scripts/craftbot.sh setup
./scripts/craftbot.sh build
```

The helper executes commands through `distrobox enter hi`. Do not use a host
Python with a native module built for a different Python ABI. `JOBS=2` limits
build parallelism on memory-constrained machines.

Prefer entering the container yourself? Equivalent commands inside `hi` are:

```bash
cd /path/to/craftbot
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
./build.sh
```

On ordinary Linux without Distrobox, run that equivalent workflow in your own
build environment. Debian/Ubuntu native prerequisites are `build-essential`
and `cmake`, plus Python development headers if your Python installation lacks
them. The repository's CI uses the direct workflow.

`build.sh` generates registry/blueprint headers, locates pybind11 and produces
`build/_craftsim*.so`. It defaults to four build jobs. No game download is needed.

## 2. Get the selected models

```bash
./scripts/craftbot.sh download-models
```

The downloader obtains the `models-v1` release and verifies both file size and
SHA-256. For a fork, add `--repo OWNER/REPO`. If the release does not yet exist,
the download fails clearly; maintainers publish it using the repository script.
In the original training workspace, `./scripts/craftbot.sh export-models`
creates the same assets from the retained local checkpoints.

## 3. Run a policy without training

Current v7 multitask model:

```bash
./scripts/craftbot.sh eval models/best-v7.pt --obs v7 \
  --task gather_wood --episodes 16 --envs 8 --agents 4 --policy all
```

Legacy v6 crafting model:

```bash
./scripts/craftbot.sh eval models/best-v6.pt --obs v6 \
  --goal stone_pickaxe --episodes 64 --envs 16 --agents 4 --policy all
```

The command prints successes and decisions to success. By default actions are
sampled. `--greedy` changes the experiment to argmax actions; it is not the
historical evaluation protocol. An episode ends when the task succeeds or its
step limit is reached. Evaluation can therefore take time for unsuccessful
tasks; an idle-looking terminal is not necessarily a stalled process.

Use `--task NAME` for comparable per-task v7 tests. `--task all` mixes tasks
across the total episode budget; it does **not** give every task that many
episodes. See [model cards](models.md) before comparing numbers.

## Optional Training Studio

Studio requires SDL3, OpenGL, nlohmann/json and Dear ImGui docking. Inside `hi`:

```bash
sudo pacman -S --needed sdl3 nlohmann-json
git clone --depth 1 -b docking https://github.com/ocornut/imgui dashboard/third_party/imgui
cmake -S dashboard -B dashboard/build -DCMAKE_BUILD_TYPE=Release
cmake --build dashboard/build -j4
```

The ImGui checkout stays outside this repository's Git history. Its original
licence remains in that directory. OpenGL must be available from your graphics
driver and container integration.

From a host terminal, start a trainer with telemetry:

```bash
./scripts/craftbot.sh train --obs v7 --autostart --mode live \
  --envs 8 --agents 1 --threads 2 --workers 2 \
  --total-steps 100000 --seed 42 --run-id studio-first
```

In another terminal, from the same checkout:

```bash
./scripts/craftbot.sh studio --host 127.0.0.1 --port 7777 --visualize
```

Studio provides a world viewport, training controls, graphs, checkpoint browser,
agent inspector and network panels. It connects to an existing trainer; opening
the UI alone does not start the Python training process. Avoid `--no-server`
when using Studio because that flag disables the telemetry listener.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `_craftsim` cannot be imported | Rebuild with the same `.venv/bin/python` used to run the trainer |
| CMake cannot find pybind11 | Rerun setup, then `build.sh`; do not configure with another Python |
| Wrong observation-layout error | Match `--obs v6` or `--obs v7` to the model card |
| Studio cannot connect | Start the trainer without `--no-server`, check host/port and run IDs |
| Missing demo directory after resume | Pass `--no-demos`, or generate demos and explicitly set `--demos DIR` |
| Resume stops immediately | `--total-steps` is the absolute lifetime target; set it above the checkpoint counter |
| Distrobox reports `/run/user/1000/libpod` read-only | Run from your normal desktop terminal; a restricted agent session may not be able to enter the container |
| Download fails | Check network access, repository name and existence of the `models-v1` release |

Full training instructions: [training.md](training.md).
