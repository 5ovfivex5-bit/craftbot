#!/usr/bin/env bash
# Checkpoint sync from the Colab VM to runs/<run>/ on the Deck.
#
#   gpusim/colab_sync.sh SESSION RUN_ID            loop (started with nohup by colab_train.sh)
#   gpusim/colab_sync.sh SESSION RUN_ID --once     one pass (used by colab_stop.sh)
#
# Every SYNC_SECONDS (default 180): download the newest checkpoint that is not local yet into
# runs/<run>/checkpoints/ (written as a hidden .part file, then renamed, so the trainer's listing never
# sees a partial file), keep only the newest KEEP_CKPT (default 3) local checkpoints, and copy the
# VM's train.log to runs/<run>/logs/train_remote.log. Exits when runs/<run>/pids/sync.stop exists.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=colab_common.sh
source "$HERE/colab_common.sh"

SESS="${1:?session}"
RUN="${2:?run id}"
ONCE=0
[ "${3:-}" = "--once" ] && ONCE=1
SYNC_SECONDS="${SYNC_SECONDS:-180}"
KEEP_CKPT="${KEEP_CKPT:-3}"
LOCAL_DIR="$COLAB_ROOT/runs/$RUN/checkpoints"
LOGS="$COLAB_ROOT/runs/$RUN/logs"
STOP_FILE="$COLAB_ROOT/runs/$RUN/pids/sync.stop"
mkdir -p "$LOCAL_DIR" "$LOGS"

log() { echo "[sync $(date +%H:%M:%S)] $*"; }

pass() {
    local remote name
    remote="$(vm_newest_ckpt "$SESS" "$RUN" 2>/dev/null || true)"
    if [ -n "$remote" ] && [ ! -f "$LOCAL_DIR/$remote" ]; then
        log "downloading $remote"
        local pre=""
        [ -n "${CONTAINER_ID:-}" ] || pre="distrobox enter $COLAB_DISTRO --"
        if timeout "${DOWNLOAD_TIMEOUT:-1500}" $pre colab download -s "$SESS" "$VM_RUNS/$RUN/checkpoints/$remote" "$LOCAL_DIR/.$remote.part" </dev/null \
            && mv "$LOCAL_DIR/.$remote.part" "$LOCAL_DIR/$remote"; then
            log "saved $LOCAL_DIR/$remote"
        else
            log "download failed for $remote (will retry)"
            rm -f "$LOCAL_DIR/.$remote.part"
        fi
    fi
    # prune: keep the newest KEEP_CKPT checkpoints
    ls -1 "$LOCAL_DIR" 2>/dev/null | grep -E '^ckpt_[0-9]{12}_u[0-9]{6}[.]pt$' | sort | head -n -"$KEEP_CKPT" \
        | while read -r old; do rm -f "$LOCAL_DIR/$old"; log "pruned $old"; done || true
    dx colab download -s "$SESS" "$VM_RUNS/$RUN/train.log" "$LOGS/train_remote.log" >/dev/null 2>&1 || true
}

if [ "$ONCE" = 1 ]; then
    pass
    exit 0
fi

fails=0
while [ ! -f "$STOP_FILE" ]; do
    if pass; then :; fi
    if session_alive "$SESS"; then
        fails=0
    else
        fails=$((fails + 1))
        log "session $SESS not reachable ($fails)"
        [ "$fails" -ge 5 ] && { log "giving up: session gone"; exit 1; }
    fi
    sleep "$SYNC_SECONDS"
done
log "stop file present: exiting"
