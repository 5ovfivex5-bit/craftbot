#!/usr/bin/env bash
# Weights-only sync from the Colab VM to the Deck (replaces the 1 GB full-checkpoint download).
#
#   gpusim/colab_weights_sync.sh SESSION RUN_ID          loop (run with nohup)
#   gpusim/colab_weights_sync.sh SESSION RUN_ID --once   one pass
#
# Makes sure gpusim/export_weights.py is running on the VM (fp16 weights, ~200 MB, refreshed every
# EXPORT_SECONDS), then every SYNC_SECONDS downloads weights_latest.pt into
# runs/<run>/weights/weights_<env_steps>.pt when it is newer than what we have, keeping the newest KEEP.
# Full checkpoints stay on the VM for resuming. Exits when runs/<run>/pids/weights_sync.stop exists.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=colab_common.sh
source "$HERE/colab_common.sh"

SESS="${1:?session}"
RUN="${2:?run id}"
ONCE=0
[ "${3:-}" = "--once" ] && ONCE=1
SYNC_SECONDS="${SYNC_SECONDS:-7200}"
EXPORT_SECONDS="${EXPORT_SECONDS:-600}"
KEEP="${KEEP:-3}"
OUT="$COLAB_ROOT/runs/$RUN/weights"
LOGS="$COLAB_ROOT/runs/$RUN/logs"
STOP_FILE="$COLAB_ROOT/runs/$RUN/pids/weights_sync.stop"
mkdir -p "$OUT" "$LOGS"
log() { echo "[wsync $(date +%H:%M:%S)] $*"; }

ensure_exporter() {
    # pid file, not pgrep: the remote bash command line itself contains the script name
    local pidf="$VM_RUNS/$RUN/export.pid"
    vm_sh "$SESS" "if [ -f $pidf ] && kill -0 \$(cat $pidf) 2>/dev/null; then :; else \
        cd $VM_CRAFT && setsid nohup python gpusim/export_weights.py --run-dir $VM_RUNS/$RUN --every $EXPORT_SECONDS \
        >> $VM_RUNS/$RUN/export.log 2>&1 < /dev/null & echo \$! > $pidf; disown; echo started exporter; fi" || true
}

pass() {
    ensure_exporter
    local meta="$OUT/.latest.json" steps
    curl -sf -m 60 -o "$meta" "http://127.0.0.1:${FILE_PORT:-7791}/weights_latest.json" || { log "no export yet (or tunnel down)"; return 0; }
    steps="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['counters'].get('env_steps', 0))" "$meta")"
    local name
    name="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$meta")"
    # drop partial downloads of older exports
    find "$OUT" -maxdepth 1 -name '.weights_*.pt.part' ! -name ".$name.part" -delete
    if [ -f "$OUT/$name" ]; then
        log "up to date ($name)"
        return 0
    fi
    log "downloading weights at env_steps=$steps (capped at ${RATE:-80k}/s so the dashboard keeps bandwidth)"
    # Over the SSH tunnel's file server (port 7791): full SSH speed instead of Colab's slow file API,
    # rate-capped, and resumable (-C -) so a Wi-Fi drop continues where it stopped.
    if curl -sf -C - --limit-rate "${RATE:-80k}" --retry 5 --retry-delay 10 -m "${DOWNLOAD_TIMEOUT:-5400}" \
        -o "$OUT/.$name.part" "http://127.0.0.1:${FILE_PORT:-7791}/$name" \
        && python3 -c "import hashlib,json,sys; m=json.load(open(sys.argv[1])); h=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest(); sys.exit(h != m['sha256'])" "$meta" "$OUT/.$name.part" \
        && mv "$OUT/.$name.part" "$OUT/$name"; then
        cp "$meta" "$OUT/${name%.pt}.json"
        log "saved $OUT/$name ($(du -h "$OUT/$name" | cut -f1))"
    else
        if [ -f "$OUT/.$name.part" ] && [ "$(stat -c %s "$OUT/.$name.part")" -ge "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['bytes'])" "$meta")" ]; then
            rm -f "$OUT/.$name.part"; log "complete but sha256 mismatch: discarded, will retry"
        else
            log "download incomplete (kept partial file, will resume next round)"
        fi
    fi
    ls -1 "$OUT" | grep -E '^weights_[0-9]{12}[.]pt$' | sort | head -n -"$KEEP" \
        | while read -r old; do rm -f "$OUT/$old" "$OUT/${old%.pt}.json"; log "pruned $old"; done || true
}

if [ "$ONCE" = 1 ]; then pass; exit 0; fi
while [ ! -f "$STOP_FILE" ]; do
    pass || true
    sleep "$SYNC_SECONDS"
done
log "stop file present: exiting"
