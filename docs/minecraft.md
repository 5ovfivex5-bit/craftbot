# Run the v6 policy in Minecraft

This optional integration is separate from simulator training. You need your
own Minecraft Java **26.2** installation, Fabric Loader **0.19.3**, Fabric API
for 26.2 and Java **25**. The companion bot mod is in `mcmod/`.

Use `models/best-v6.pt`. The bridge supports v6 and older v5 observation layouts;
the v7 checkpoint cannot be used by this integration.

## 1. Build the bot mod

Enter `distrobox enter hi`, then from the repository root:

```bash
cd mcmod
JAVA_HOME="$HOME/.local/share/ElyPrismLauncher/java/java-runtime-epsilon" ./gradlew build
```

On another machine, set `JAVA_HOME` to its Java 25 installation. The output is
under `mcmod/build/libs/`; choose the normal mod JAR, not the sources JAR.

For this project's Steam Deck environment, install built mods only into:

```text
~/.local/share/ElyPrismLauncher/instances/gtes/minecraft/mods/
```

Do not copy Gradle caches, development worlds or decompiled game source into a
published repository. The source tree includes only the integration code and
Gradle wrapper necessary to build it.

## 2. Start the policy bridge

In a host terminal at the project root:

```bash
./scripts/craftbot.sh bridge --ckpt models/best-v6.pt
```

The helper enters `hi`, selects the v6 observation environment and listens on
`127.0.0.1:7788`. It loads the checkpoint's selected `model` member for inference;
the historical population evaluations use all four members, so those percentages
are not the real-game bridge's measured results.

## 3. Spawn a bot

Open the modded game and a world where commands are enabled:

```text
/aibot spawn Learner
/aibot task wooden_pickaxe Learner
/aibot status
```

Control it with `/aibot stop`, `/aibot go` and `/aibot remove`. The task command
expects the integration's goal-item names; it is not the v7 simulator task API.

## Online game fine-tuning

The bridge also has an experimental `--train` mode:

```bash
./scripts/craftbot.sh bridge --train --ckpt models/best-v6.pt --out runs/mc-ft
```

The mod exposes `/aibot train start`, `/aibot train status` and `/aibot train stop`.
This updates weights using game interaction and saves separate checkpoints.
It is a different training environment and measurement from the simulator.
See `mcbridge/mc_train.py` and `PROTOCOL.md` for the advanced controls.

The simulator has explicit parity limitations. A model that crafts reliably
in it may fail in an unfamiliar real world; judge the bridge by fresh game
observations rather than reusing simulator evaluation percentages.
