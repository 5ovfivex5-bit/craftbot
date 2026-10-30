# Third-party software and project boundaries

CraftBot contains a custom voxel simulator, training code, learned policies and
tools. It uses established algorithms and libraries; it is not the Minecraft
engine and is not affiliated with Mojang or Microsoft.

| Component | Origin | Handling |
| --- | --- | --- |
| PyTorch | pytorch.org | Installed dependency; its licence applies to PyTorch |
| NumPy, psutil, pybind11 | Respective upstream projects | Installed dependencies |
| SDL3 | libsdl.org | Optional Studio dependency, zlib licence |
| Dear ImGui | github.com/ocornut/imgui, docking branch | Optional checkout, excluded from Git; retain upstream MIT licence |
| nlohmann/json | github.com/nlohmann/json | Optional Studio dependency, MIT |
| stb_image_write | github.com/nothings/stb, Sean Barrett et al. | Vendored header retains its original dual MIT/public-domain notice |
| Fabric Loader, API and Loom | fabricmc.net | Optional Java integration dependencies |
| Minecraft | Mojang/Microsoft | Separately installed game; no game binaries or decompiled source distributed |
| Converted village structure definitions | Mojang/Minecraft, converted with the local NBT tooling | Marked `vanilla:` in blueprint provenance; upstream content is not relicensed as original project work |

`mcsrc/`, local Minecraft worlds, recordings, build caches and raw experiment
directories are excluded from publication. Generated registry tables and task
definitions describe the simulator's supported subset; they are not a copy of
the complete game. `research/` and `PROTOCOL.md` contain historical design notes
and citations; plans in those documents are not claims of implemented features.

`data/builds/README.md` documents which templates are handmade, which were
converted from vanilla structures and which are procedural variants. Registry
documentation also records extracted gameplay values and assumed approximations.
The root MIT licence covers the project's own code and curated model artifacts.
It does not replace dependency licences or grant rights in Minecraft assets.
