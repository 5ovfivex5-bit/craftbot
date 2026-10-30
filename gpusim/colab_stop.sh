#!/usr/bin/env bash
# Stop everything colab_train.sh started, in this order:
#   1. checkpoint sync loop (stop file + pkill), then one final sync pass (newest checkpoint local)
#   2. trainer on the VM: SIGTERM -> final checkpoint at the next update boundary -> exit (waits up to 5 min)
#   3. SSH tunnel on 127.0.0.1:7777
#   4. the Colab session (colab stop)
#   5. the dashboard on the Deck (minemind_studio), unless --keep-dashboard
#
#   gpusim/colab_stop.sh [--keep-dashboard]            (RUN / SESS env vars as in colab_train.sh)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=colab_common.sh
source "$HERE/colab_common.sh"

SESS="${SESS:-minemind-gpu}"
RUN="${RUN:-gpu-100m}"
KEEP_DASH=0
[ "${1:-}" = "--keep-dashboard" ] && KEEP_DASH=1
RUN_DIR="$COLAB_ROOT/runs/$RUN"
mkdir -p "$RUN_DIR/pids"

echo "== sync loop"
touch "$RUN_DIR/pids/sync.stop"
dx pkill -f "[c]olab_sync.sh $SESS $RUN" 2>/dev/null
touch "$RUN_DIR/pids/weights_sync.stop"
pkill -f "[c]olab_weights_sync.sh $SESS $RUN" 2>/dev/null
sleep 2

if session_alive "$SESS"; then
    echo "== trainer on the VM (graceful)"
    vm_sh "$SESS" "pkill -TERM -f '[t]rainer[.]run --backend gpusim' ; for i in \$(seq 1 150); do pgrep -f '[t]rainer[.]run --backend gpusim' >/dev/null || break; sleep 2; done; tail -n 3 $VM_RUNS/$RUN/train.log" || echo "(could not reach the VM trainer)"
    echo "== final weights export + download (fp16 weights only; full checkpoints stay on the VM)"
    vm_sh "$SESS" "cd $VM_CRAFT && python -c \"import sys; sys.path.insert(0, 'gpusim'); import export_weights as e; print(e.export(e.newest('$VM_RUNS/$RUN/checkpoints'), '$VM_RUNS/$RUN'))\"" || echo "(final export failed)"
    bash "$HERE/colab_weights_sync.sh" "$SESS" "$RUN" --once || echo "(final weights download failed)"
else
    echo "session $SESS is not alive: nothing to stop on the VM"
fi

echo "== SSH tunnel"
dx pkill -f "[s]sh -N -L $TELEMETRY_PORT:127.0.0.1:$TELEMETRY_PORT" 2>/dev/null
echo "== Colab session"
dx colab stop -s "$SESS" 2>/dev/null || echo "(no session $SESS to stop)"

if [ "$KEEP_DASH" = 0 ]; then
    echo "== dashboard"
    dx pkill -f "[m]inemind_studio" 2>/dev/null || true
    pkill -f "[m]inemind_studio" 2>/dev/null || true
fi
echo "== stopped. remaining: $(dx colab sessions 2>/dev/null | tr -d '\r' | tail -n 2 | tr '\n' ' ')"
