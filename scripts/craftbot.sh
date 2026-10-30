#!/usr/bin/env bash
# Steam Deck entry point: Python and native builds run inside distrobox hi.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
action="${1:-help}"
if [[ "$action" != help && "${CRAFTBOT_IN_HI:-0}" != 1 ]]; then
    exec distrobox enter hi -- env CRAFTBOT_IN_HI=1 "$ROOT/scripts/craftbot.sh" "$@"
fi
cd "$ROOT"
if (($#)); then shift; fi
PY="$ROOT/.venv/bin/python"
if [[ "$action" != setup && "$action" != help && ! -x "$PY" ]]; then
    echo "Run ./scripts/craftbot.sh setup first." >&2
    exit 1
fi
case "$action" in
    setup)
        if [[ ! -x "$PY" ]]; then
            if command -v uv >/dev/null; then
                uv venv --python 3.12 "$ROOT/.venv"
            else
                python3.12 -m venv "$ROOT/.venv"
            fi
        fi
        if command -v uv >/dev/null; then
            uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu
            uv pip install --python "$PY" -r requirements.txt "$@"
        else
            "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
            "$PY" -m pip install -r requirements.txt "$@"
        fi
        ;;
    build) exec ./build.sh "$@" ;;
    train) exec "$PY" -m trainer.run "$@" ;;
    eval) exec "$PY" -m trainer.eval "$@" ;;
    experts) exec "$PY" -m experts.generate "$@" ;;
    clone) exec "$PY" -m trainer.bc_v7 "$@" ;;
    export-models) exec "$PY" tools/export_models.py "$@" ;;
    download-models) exec "$PY" tools/download_models.py "$@" ;;
    test) exec "$PY" tools/check_project.py "$@" ;;
    bridge)
        export MINEMIND_OBS=v6
        if (($# == 0)); then
            set -- --ckpt models/best-v6.pt
        fi
        exec "$PY" mcbridge/bridge.py "$@"
        ;;
    studio)
        if [[ ! -x dashboard/build/minemind_studio ]]; then
            echo "Build Studio first: cmake -S dashboard -B dashboard/build && cmake --build dashboard/build -j4" >&2
            exit 1
        fi
        exec dashboard/build/minemind_studio "$@"
        ;;
    help)
        cat <<'HELP'
CraftBot · MineMind

  ./scripts/craftbot.sh setup                  Install Python dependencies in hi
  ./scripts/craftbot.sh build                  Build the native CPU simulator
  ./scripts/craftbot.sh download-models        Download and verify release models
  ./scripts/craftbot.sh eval MODEL [flags]     Evaluate a saved policy
  ./scripts/craftbot.sh train [flags]          Train or resume PPO
  ./scripts/craftbot.sh studio [flags]         Open the optional dashboard
  ./scripts/craftbot.sh bridge [flags]         Serve a v6 policy to the Fabric mod
  ./scripts/craftbot.sh experts [flags]        Generate scripted demonstrations
  ./scripts/craftbot.sh clone [flags]          Train behavior cloning
  ./scripts/craftbot.sh test [--full]          Run the project checks

Every command above enters distrobox hi. See docs/quickstart.md and docs/training.md.
HELP
        ;;
    *) echo "Unknown command: $action (use help)" >&2; exit 2 ;;
esac
