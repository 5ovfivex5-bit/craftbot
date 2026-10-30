#!/usr/bin/env bash
# Generate the registry tables (src/gen/registry.h from data/), configure and build the native sim module
# into build/ (run inside the distrobox). Extra args are passed to cmake, e.g. ./build.sh -DMINEMIND_BUILD_DASHBOARD=ON
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "Python environment missing. Run ./scripts/craftbot.sh setup first." >&2
    exit 1
fi
if ! command -v cmake >/dev/null || ! command -v c++ >/dev/null; then
    echo "Install CMake, a C++20 compiler and OpenMP in your build environment (see docs/quickstart.md)." >&2
    exit 1
fi
"$PY" -I "$ROOT/tools/gen_registry.py"
"$PY" -I "$ROOT/tools/gen_blueprints.py"
PB11_DIR="$("$PY" -m pybind11 --cmakedir)"
cmake -S "$ROOT" -B "$ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$PY" \
    -Dpybind11_DIR="$PB11_DIR" \
    "$@"
cmake --build "$ROOT/build" -j"${JOBS:-4}"
