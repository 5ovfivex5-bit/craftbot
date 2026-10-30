#!/usr/bin/env bash
# Shared helpers for the Colab GPU training scripts (sourced, not run).
#   colab_train.sh  create/attach the session, push code, resume, start training, tunnel, sync
#   colab_sync.sh   background loop: newest checkpoint (and the train log) from the VM to runs/<run>/
#   colab_stop.sh   stop everything
# The `colab` CLI (and its auth) lives in the distrobox "hi"; `dx` runs a command there, or directly
# when the script is already running inside the container.
set -euo pipefail

COLAB_ROOT="${COLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COLAB_DISTRO="${COLAB_DISTRO:-hi}"
COLAB_SSH_KEY="${COLAB_SSH_KEY:-$HOME/.ssh/id_ed25519}"
TELEMETRY_PORT="${TELEMETRY_PORT:-7777}"
VM_ROOT=/content
VM_CRAFT="$VM_ROOT/craftbot"
VM_RUNS="$VM_ROOT/runs"

dx() {
    if [ -n "${CONTAINER_ID:-}" ]; then
        "$@"
    else
        distrobox enter "$COLAB_DISTRO" -- "$@"
    fi
}

# bash on the VM (through the colab exec kernel); stdout of the command is returned
vm_sh() {
    local sess="$1" cmd="$2" tmp b64 rc=0
    tmp="$(mktemp "${TMPDIR:-/tmp}/colab_sh.XXXXXX.py")"
    b64="$(printf '%s' "$cmd" | base64 -w0)"
    cat > "$tmp" <<PYEOF
import base64, subprocess
cmd = base64.b64decode("$b64").decode()
r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
print(r.stdout, end="")
if r.returncode:
    print("[vm_sh rc=%d] %s" % (r.returncode, r.stderr[-2000:]))
PYEOF
    local pre=""
    [ -n "${CONTAINER_ID:-}" ] || pre="distrobox enter $COLAB_DISTRO --"
    timeout 600 $pre colab exec -s "$sess" -f "$tmp" --timeout 900 </dev/null || rc=$?
    rm -f "$tmp"
    return $rc
}

session_alive() {
    # `colab status` exits 0 for unknown sessions too, so look for the status line
    dx colab status -s "$1" </dev/null 2>&1 | grep -q "Status:"
}

ensure_session() {
    local sess="$1"
    if session_alive "$sess"; then
        echo "[colab] session $sess exists"
    else
        echo "[colab] creating T4 session $sess"
        dx colab new --gpu T4 -s "$sess"
    fi
}

# bundle the sources, upload, extract on the VM and build the native sim when its sources changed
push_code() {
    local sess="$1" scr
    scr="$(mktemp -d "${TMPDIR:-/tmp}/craftbot_bundle.XXXXXX")"
    mkdir -p "$scr/craftbot"
    cp -r "$COLAB_ROOT/src" "$COLAB_ROOT/craftsim.py" "$COLAB_ROOT/gpusim" "$COLAB_ROOT/trainer" "$scr/craftbot/"
    find "$scr" -name __pycache__ -prune -exec rm -rf {} +
    tar czf "$scr/bundle.tgz" -C "$scr" craftbot
    dx colab upload -s "$sess" "$scr/bundle.tgz" "$VM_ROOT/bundle.tgz" >/dev/null </dev/null
    rm -rf "$scr"
    vm_sh "$sess" "cd $VM_ROOT && tar xzf bundle.tgz && bash $VM_CRAFT/gpusim/vm_setup.sh"
}

# newest checkpoint name on the VM for run $2 ("" if none)
vm_newest_ckpt() {
    vm_sh "$1" "ls -1 $VM_RUNS/$2/checkpoints 2>/dev/null | grep -E '^ckpt_[0-9]{12}_u[0-9]{6}[.]pt\$' | sort | tail -1" | tr -d '\r' | tail -1
}

# newest local checkpoint name for run $1 ("" if none)
local_newest_ckpt() {
    ls -1 "$COLAB_ROOT/runs/$1/checkpoints" 2>/dev/null | grep -E '^ckpt_[0-9]{12}_u[0-9]{6}[.]pt$' | sort | tail -1 || true
}

# env-step count encoded in a checkpoint name (for newest-of-two comparisons)
ckpt_steps() {
    local n="$1"
    if [ -z "$n" ]; then echo 0; else echo $((10#${n:5:12})); fi
}

telemetry_listening() {
    (exec 3<>"/dev/tcp/127.0.0.1/$TELEMETRY_PORT") 2>/dev/null
}
