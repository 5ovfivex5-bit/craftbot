#!/usr/bin/env bash
# Self-healing SSH tunnel to the Colab VM: 7777 = telemetry (dashboard), 7791 = weights file server.
# Keepalives make ssh exit within ~45 s when Wi-Fi drops; the loop then reconnects.
#   gpusim/colab_tunnel.sh SESSION [RUN_ID]      (run with setsid nohup; stop: touch runs/<run>/pids/tunnel.stop)
SESS="${1:?session}"; RUN="${2:-gpu-100m}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STOP="$ROOT/runs/$RUN/pids/tunnel.stop"
while [ ! -f "$STOP" ]; do
    echo "[tunnel $(date +%H:%M:%S)] connecting"
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
        -L 7777:127.0.0.1:7777 -L 7791:127.0.0.1:7791 -i "$HOME/.ssh/id_ed25519" \
        -o ProxyCommand="distrobox enter hi -- colab ssh --proxy-mode -s $SESS" root@colab
    echo "[tunnel $(date +%H:%M:%S)] ssh exited ($?), retrying in 5 s"
    sleep 5
done
