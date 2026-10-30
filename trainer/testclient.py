"""Tiny dashboard stand-in for protocol testing.

  python -m trainer.testclient --port 7777 --subscribe --duration 10
  python -m trainer.testclient --port 7777 --send pause --send 'set_config {"values":{"num_envs":4}}' --gap 1

Prints message counts/rates per type, the hello/model summary, every ack/log, and the first
snapshot/activations sizes. --subscribe sends {"cmd":"subscribe","snapshot":true,"nn":true}.
"""
import argparse
import base64
import json
import socket
import sys
import time
from collections import Counter


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m trainer.testclient")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--subscribe", action="store_true", help="subscribe to snapshot and nn")
    p.add_argument("--send", action="append", default=[],
                   help="command: NAME or 'NAME {json args}' (repeatable, sent in order)")
    p.add_argument("--gap", type=float, default=1.0, help="seconds between sent commands")
    p.add_argument("--duration", type=float, default=5.0, help="seconds to listen")
    p.add_argument("--quiet", action="store_true", help="only print the summary")
    args = p.parse_args(argv)

    sock = socket.create_connection((args.host, args.port), timeout=5)
    sock.settimeout(0.2)
    counts = Counter()
    first = {}
    buf = b""
    next_id = 1
    to_send = []
    for item in args.send:
        name, _, rest = item.partition(" ")
        body = json.loads(rest) if rest.strip() else {}
        body["cmd"] = name
        to_send.append(body)
    if args.subscribe:
        to_send.insert(0, {"cmd": "subscribe", "snapshot": True, "nn": True})
    t0 = time.time()
    last_send = t0 + 0.5
    last_metrics = None
    acks = []

    def send(obj):
        nonlocal next_id
        obj = dict(obj)
        obj["id"] = next_id
        next_id += 1
        sock.sendall((json.dumps(obj) + "\n").encode())
        return obj

    while time.time() - t0 < args.duration or to_send:
        if to_send and time.time() - last_send >= args.gap:
            sent = send(to_send.pop(0))
            if not args.quiet:
                print(f"-> sent {sent}", flush=True)
            last_send = time.time()
        try:
            data = sock.recv(1 << 20)
            if not data:
                print("server closed connection", flush=True)
                break
            buf += data
        except socket.timeout:
            pass
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            msg = json.loads(line)
            t = msg.get("type")
            counts[t] += 1
            if t not in first:
                first[t] = time.time() - t0
            if t == "ack":
                acks.append(msg)
                if not args.quiet:
                    print(f"<- ack id={msg.get('id')} ok={msg.get('ok')} msg={msg.get('msg')}", flush=True)
            elif t == "log":
                if not args.quiet:
                    print(f"<- log [{msg.get('level')}] {msg.get('msg')}", flush=True)
            elif t == "hello":
                if not args.quiet:
                    print(f"<- hello run_id={msg.get('run_id')} state={msg.get('state')} "
                          f"schema={msg.get('schema')} world={msg.get('world_size')} "
                          f"actions={len(msg.get('actions', []))} blocks={len(msg.get('blocks', []))}",
                          flush=True)
            elif t == "model":
                if not args.quiet:
                    print(f"<- model {msg.get('name')} params={msg.get('params_total')} "
                          f"layers={len(msg.get('layers', []))} edges={len(msg.get('edges', []))}",
                          flush=True)
            elif t == "metrics":
                last_metrics = msg
                if not args.quiet:
                    eta = "Calculating" if msg.get("eta_s") is None else f"{msg['eta_s']:.0f}s"
                    print(f"<- metrics state={msg['state']} steps={msg['env_steps']} sps={msg['sps']:.0f} "
                          f"upd={msg['updates']} envs={msg['num_envs']} eta={eta} "
                          f"mem_warning={msg.get('mem_warning')}", flush=True)
            elif t == "snapshot":
                if "_seen_snapshot" not in first:
                    first["_seen_snapshot"] = True
                    blocks = msg.get("blocks")
                    print(f"<- snapshot env_id={msg.get('env_id')} ep={msg.get('episode')} "
                          f"size={msg.get('size')} blocks_b64={len(blocks) if blocks else None} "
                          f"obs_b64={len(msg.get('obs', ''))} keys={sorted(msg.keys())}", flush=True)
            elif t == "activations":
                if "_seen_act" not in first:
                    first["_seen_act"] = True
                    print(f"<- activations env_id={msg.get('env_id')} layers={sorted(msg.get('layers', {}))} "
                          f"nan_inf={msg.get('nan_inf')} entropy={msg.get('entropy'):.4f}", flush=True)
            elif t in ("checkpoints", "history"):
                n = len(msg.get("items", msg.get("rows", [])))
                if not args.quiet:
                    print(f"<- {t}: {n} entries", flush=True)
                if t == "checkpoints" and not args.quiet:
                    for it in msg.get("items", [])[:5]:
                        print(f"     {it['path']} steps={it['env_steps']} upd={it['updates']}", flush=True)
    elapsed = time.time() - t0
    print("--- summary ---")
    for t, n in sorted(counts.items(), key=lambda kv: str(kv[0])):
        print(f"{t:12s} count={n:6d} rate={n / max(elapsed, 1e-9):8.2f}/s first_at={first.get(t, 0):.2f}s")
    if last_metrics is not None:
        print(f"last metrics: state={last_metrics['state']} env_steps={last_metrics['env_steps']} "
              f"sps={last_metrics['sps']:.1f} updates={last_metrics['updates']}")
    bad = [a for a in acks if not a.get("ok")]
    print(f"acks: {len(acks)} total, {len(bad)} failed")
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
