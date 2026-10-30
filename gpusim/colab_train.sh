#!/usr/bin/env bash
# Launch or resume the long GPU training run on a Colab T4. Idempotent: re-run it after a disconnect.
#
#   gpusim/colab_train.sh                       # session minemind-gpu, run gpu-100m
#   SESS=other RUN=gpu-100m-b gpusim/colab_train.sh
#
# Steps: ensure the T4 session exists -> push sources and build the sim on the VM -> resume from the
# newest checkpoint (remote or local, whichever has more env steps) -> start `trainer.run` with nohup
# on the VM (telemetry on 127.0.0.1:7777 there) -> SSH local port forward 127.0.0.1:7777 on the Deck ->
# background checkpoint sync (colab_sync.sh). Stop everything with gpusim/colab_stop.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=colab_common.sh
source "$HERE/colab_common.sh"

SESS="${SESS:-minemind-gpu}"
RUN="${RUN:-gpu-100m}"
MODEL_SIZE="${MODEL_SIZE:-100m}"
ENVS="${ENVS:-4096}"
MINIBATCH="${MINIBATCH:-2048}"
ROLLOUT="${ROLLOUT:-64}"
TOTAL_STEPS="${TOTAL_STEPS:-250000000}"
CKPT_SECONDS="${CKPT_SECONDS:-300}"
# feature set chosen by the from-scratch comparison (see the report); override with TRAIN_FLAGS
TRAIN_FLAGS="${TRAIN_FLAGS:---curiosity rnd --spr --aux-ach --value-loss hlgauss --resblock --novelty 0.02}"

RUN_DIR="$COLAB_ROOT/runs/$RUN"
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/logs" "$RUN_DIR/pids"
rm -f "$RUN_DIR/pids/sync.stop"

echo "== session"
ensure_session "$SESS"

echo "== code"
push_code "$SESS"

echo "== resume point"
LOCAL_CK="$(local_newest_ckpt "$RUN")"
REMOTE_CK="$(vm_newest_ckpt "$SESS" "$RUN" || true)"
echo "local:  ${LOCAL_CK:-none}"
echo "remote: ${REMOTE_CK:-none}"
RESUME=""
if [ "$(ckpt_steps "$REMOTE_CK")" -ge "$(ckpt_steps "$LOCAL_CK")" ] && [ -n "$REMOTE_CK" ]; then
    RESUME="$VM_RUNS/$RUN/checkpoints/$REMOTE_CK"
elif [ -n "$LOCAL_CK" ]; then
    echo "uploading local checkpoint $LOCAL_CK"
    vm_sh "$SESS" "mkdir -p $VM_RUNS/$RUN/checkpoints"
    dx colab upload -s "$SESS" "$RUN_DIR/checkpoints/$LOCAL_CK" "$VM_RUNS/$RUN/checkpoints/$LOCAL_CK" >/dev/null
    RESUME="$VM_RUNS/$RUN/checkpoints/$LOCAL_CK"
fi
echo "resume: ${RESUME:-fresh run}"

echo "== trainer on the VM"
RUNNING="$(vm_sh "$SESS" "pgrep -f '[t]rainer[.]run --backend gpusim' >/dev/null && echo yes || echo no" | tail -1)"
if [ "$RUNNING" = "yes" ]; then
    echo "trainer already running on the VM; not starting another"
else
    RESUME_ARG=""
    [ -n "$RESUME" ] && RESUME_ARG="--resume $RESUME"
    START="mkdir -p $VM_RUNS/$RUN && cd $VM_CRAFT && setsid nohup python3 -m trainer.run --backend gpusim --device cuda --amp \
--model-size $MODEL_SIZE --envs $ENVS --minibatch $MINIBATCH --rollout $ROLLOUT --total-steps $TOTAL_STEPS \
--ckpt-every 1000000000000 --ckpt-seconds $CKPT_SECONDS --workers 1 --host 127.0.0.1 --port $TELEMETRY_PORT \
--autostart --visualize --runs-dir $VM_RUNS --run-id $RUN $RESUME_ARG $TRAIN_FLAGS \
> $VM_RUNS/$RUN/train.log 2>&1 < /dev/null & echo started"
    vm_sh "$SESS" "$START"
    echo "waiting for the telemetry server and first metrics"
    for _ in $(seq 1 60); do
        LINE="$(vm_sh "$SESS" "tail -n 3 $VM_RUNS/$RUN/train.log 2>/dev/null" | tail -n 1 || true)"
        case "$LINE" in *running*|*resumed*|*INFO*|*rror*|*Traceback*) break ;; esac
    done
    vm_sh "$SESS" "tail -n 4 $VM_RUNS/$RUN/train.log" || true
fi

echo "== SSH tunnel 127.0.0.1:$TELEMETRY_PORT -> VM"
if pgrep -f "[s]sh -N -L $TELEMETRY_PORT:127.0.0.1:$TELEMETRY_PORT" >/dev/null 2>&1 || dx pgrep -f "[s]sh -N -L $TELEMETRY_PORT:127.0.0.1:$TELEMETRY_PORT" >/dev/null 2>&1; then
    echo "tunnel already up"
else
    dx bash -c "nohup ssh -N -L $TELEMETRY_PORT:127.0.0.1:$TELEMETRY_PORT -i $COLAB_SSH_KEY \
-o ProxyCommand='colab ssh --proxy-mode -s $SESS' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
-o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ExitOnForwardFailure=yes -o BatchMode=yes \
root@colab >> $RUN_DIR/logs/tunnel.log 2>&1 < /dev/null &"
    sleep 5
fi

echo "== checkpoint sync"
if dx pgrep -f "[c]olab_sync.sh $SESS $RUN" >/dev/null 2>&1; then
    echo "sync already running"
else
    dx bash -c "cd $COLAB_ROOT && SYNC_SECONDS=${SYNC_SECONDS:-180} nohup bash gpusim/colab_sync.sh $SESS $RUN >> $RUN_DIR/logs/sync.log 2>&1 < /dev/null &"
    echo "sync started (log: $RUN_DIR/logs/sync.log)"
fi

echo "== done: telemetry on 127.0.0.1:$TELEMETRY_PORT (dashboard) ; sessions: colab sessions"
