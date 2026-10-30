#!/usr/bin/env bash
# VM-side setup (runs on the Colab VM from the extracted bundle; idempotent).
# Builds the native craftsim module into build/ (pybind11 + g++) when the C++ sources changed, and
# installs ninja (torch's cpp_extension JIT builds gpusim/csrc/gpusim_kernels.cu with it).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install -q pybind11 ninja >/dev/null 2>&1 || python3 -m pip install -q pybind11 ninja
EXT="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"
SO="build/_craftsim${EXT}"
STAMP="build/.src_sha256"
SHA="$(cat src/bindings.cpp src/sim.cpp src/sim.h | sha256sum | cut -d' ' -f1)"
mkdir -p build
if [ ! -f "$SO" ] || [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$SHA" ]; then
    INC="$(python3 -m pybind11 --includes)"
    g++ -O3 -march=native -std=c++20 -fPIC -shared -fopenmp -Wall $INC src/bindings.cpp src/sim.cpp -o "$SO"
    echo "$SHA" > "$STAMP"
    echo "[vm_setup] built $SO"
else
    echo "[vm_setup] $SO up to date"
fi
python3 -c "import sys; sys.path.insert(0, '.'); import craftsim; print('[vm_setup] craftsim VEC_DIM', craftsim.VEC_DIM)"
