#!/usr/bin/env python3
"""TEST HARNESS ONLY - not used by the dashboard app.

A stand-in for the trainer that speaks PROTOCOL.md section 2 (JSON lines over TCP).
It sends hello, model, metrics (2 Hz), snapshot and activations (only while subscribed),
a log line, and acks every command. Every received command is printed to stdout.

    python3 dashboard/tools/fake_server.py --port 7777 --seconds 20
"""
import argparse
import base64
import json
import math
import random
import select
import socket
import time

ACTIONS = ["NOOP", "FWD", "BACK", "LEFT", "RIGHT", "JUMP", "FWD_JUMP", "TURN_L", "TURN_R", "LOOK_UP",
           "LOOK_DOWN", "ATTACK", "PLACE_BLOCK", "PLACE_TABLE", "PLACE_FURNACE", "CRAFT_PLANKS",
           "CRAFT_STICKS", "CRAFT_TABLE", "CRAFT_WOOD_PICK", "CRAFT_STONE_PICK", "CRAFT_FURNACE",
           "SMELT_IRON", "CRAFT_IRON_PICK"]
ITEMS = ["LOG", "PLANKS", "STICK", "TABLE", "WOOD_PICK", "COBBLE", "STONE_PICK", "FURNACE", "COAL",
         "IRON_ORE", "IRON_INGOT", "IRON_PICK", "DIAMOND", "DIRT"]
BLOCKS = ["AIR", "GRASS", "DIRT", "STONE", "COBBLE", "LOG", "LEAVES", "PLANKS", "TABLE", "FURNACE",
          "COAL_ORE", "IRON_ORE", "DIAMOND_ORE", "BEDROCK"]
SX, SY, SZ = 33, 40, 33
RES = 32
VEC_DIM = len(ITEMS) + 11


def make_region():
    """Hilly grass terrain with a stone layer, a log tree and some ores. Index (y*sz+z)*sx+x."""
    data = bytearray(SX * SY * SZ)
    for z in range(SZ):
        for x in range(SX):
            h = int(14 + 4 * math.sin(x * 0.35) + 3 * math.cos(z * 0.3 + x * 0.1))
            for y in range(SY):
                if y > h:
                    b = 0
                elif y == h:
                    b = 1
                elif y > h - 3:
                    b = 2
                elif y > 2:
                    b = 3
                else:
                    b = 13
                if y == h + 1 and x == 16 and z == 16:
                    b = 5  # tree trunk marker
                if y in (h - 6, h - 7) and 8 < x < 24 and random.random() < 0.02:
                    b = 10
                data[(y * SZ + z) * SX + x] = b
    return bytes(data)


def make_obs(t):
    buf = bytearray(RES * RES * 2)
    for j in range(RES):
        for i in range(RES):
            d = 0.5 + 0.5 * math.sin(i * 0.4 + t) * math.cos(j * 0.3)
            b = 1 if j > RES * 0.45 else (0 if j < RES * 0.2 else 2)
            idx = (j * RES + i) * 2
            buf[idx] = b
            buf[idx + 1] = int(max(0, min(255, d * 180)))
    return bytes(buf)


class Client:
    def __init__(self, sock):
        self.sock = sock
        self.rbuf = b""
        self.snapshot = False
        self.nn = False
        self.selected_env = 0

    def send(self, obj):
        self.sock.sendall((json.dumps(obj) + "\n").encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--num-envs", type=int, default=16)
    args = ap.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(4)
    srv.setblocking(False)
    print(f"[fake_server] listening on 127.0.0.1:{args.port} for {args.seconds}s", flush=True)

    clients = []
    region = make_region()
    t0 = time.time()
    state = "running"
    env_steps = 0
    target = 1_000_000
    next_metrics = next_snap = next_acts = 0.0
    hist_t = 0.0
    while time.time() - t0 < args.seconds:
        now = time.time()
        rl = [srv] + [c.sock for c in clients]
        r, _, _ = select.select(rl, [], [], 0.05)
        for s in r:
            if s is srv:
                conn, addr = srv.accept()
                conn.setblocking(True)
                c = Client(conn)
                clients.append(c)
                print(f"[fake_server] client connected {addr}", flush=True)
                c.send({"type": "hello", "schema": 1, "run_id": "fake-run-0001", "state": state,
                        "config": {"total_steps": target, "num_envs": args.num_envs, "lr": 3e-4,
                                   "minibatch_size": 512, "rollout_len": 128, "ckpt_every": 50000,
                                   "visualize": False, "snapshot_hz": 10, "nn_hz": 2, "workers": 4,
                                   "mode": "live"},
                        "world_size": [64, 48, 64], "actions": ACTIONS, "items": ITEMS, "blocks": BLOCKS})
                c.send({"type": "model", "name": "CraftPolicy (fake)", "params_total": 412_000,
                        "input": {"img": [RES, RES, 2], "vec": [VEC_DIM]}, "gru_hidden": 128,
                        "layers": [
                            {"name": "conv1", "type": "Conv2d", "params": 1184, "in_shape": [2, 32, 32],
                             "out_shape": [16, 16, 16], "trainable": True},
                            {"name": "conv2", "type": "Conv2d", "params": 4640, "in_shape": [16, 16, 16],
                             "out_shape": [32, 8, 8], "trainable": True},
                            {"name": "vec_enc", "type": "Linear", "params": 2880, "in_shape": [VEC_DIM],
                             "out_shape": [64], "trainable": True},
                            {"name": "fuse", "type": "Linear", "params": 37000, "in_shape": [2048 + 64],
                             "out_shape": [128], "trainable": True},
                            {"name": "gru", "type": "GRU", "params": 99000, "in_shape": [128],
                             "out_shape": [128], "trainable": True},
                            {"name": "policy_head", "type": "Linear Head", "params": 3000, "in_shape": [128],
                             "out_shape": [len(ACTIONS)], "trainable": True},
                            {"name": "value_head", "type": "Linear Head", "params": 129, "in_shape": [128],
                             "out_shape": [1], "trainable": True}],
                        "edges": [["conv1", "conv2"], ["conv2", "fuse"], ["vec_enc", "fuse"], ["fuse", "gru"],
                                  ["gru", "policy_head"], ["gru", "value_head"]]})
                c.send({"type": "checkpoints", "items": [
                    {"path": "ckpt/run-0001/step_0050000.pt", "env_steps": 50000, "updates": 390,
                     "time": time.time() - 900, "size_bytes": 1_650_000},
                    {"path": "ckpt/run-0001/step_0100000.pt", "env_steps": 100000, "updates": 781,
                     "time": time.time() - 300, "size_bytes": 1_650_000}]})
                c.send({"type": "log", "t": time.time() - t0, "level": "info", "msg": "fake trainer ready"})
                continue
            c = next(x for x in clients if x.sock is s)
            try:
                data = s.recv(65536)
            except OSError:
                data = b""
            if not data:
                print("[fake_server] client disconnected", flush=True)
                clients.remove(c)
                s.close()
                continue
            c.rbuf += data
            while b"\n" in c.rbuf:
                line, c.rbuf = c.rbuf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError:
                    print("[fake_server] bad json:", line[:120], flush=True)
                    continue
                print("[fake_server] RECV", json.dumps(cmd), flush=True)
                name = cmd.get("cmd")
                if name == "subscribe":
                    c.snapshot = bool(cmd.get("snapshot"))
                    c.nn = bool(cmd.get("nn"))
                elif name == "select_env":
                    c.selected_env = int(cmd.get("env_id", 0))
                elif name == "start":
                    state = "running"
                elif name == "pause":
                    state = "paused"
                elif name == "resume":
                    state = "running"
                elif name == "stop":
                    state = "stopped"
                ok, msg = True, "queued" if name == "set_config" else "ok"
                if name == "load_checkpoint" and state == "running":
                    ok, msg = False, "load needs paused or stopped"
                if name == "debug_step" and state != "paused":
                    ok, msg = False, "debug step needs paused"
                c.send({"type": "ack", "id": cmd.get("id", -1), "ok": ok, "msg": msg})
                if name == "list_checkpoints":
                    c.send({"type": "checkpoints", "items": []})

        now = time.time()
        if state == "running":
            env_steps += int(4 * args.num_envs * 2)
        if now >= next_metrics:
            next_metrics = now + 0.5
            hist_t = now - t0
            sps = 2000.0 + 300 * math.sin(hist_t / 7)
            m = {"type": "metrics", "t": hist_t, "state": state, "mode": "live",
                 "env_steps": env_steps, "target_steps": target, "sps": sps,
                 "episodes": int(env_steps / 180), "avg_ep_len": 180 + 20 * math.sin(hist_t / 13),
                 "avg_reward": 0.2 + 0.1 * math.sin(hist_t / 11), "success_rate": {"LOG": 0.95, "WOOD_PICK": 0.6, "STONE_PICK": 0.2},
                 "updates": int(env_steps / 4096), "policy_loss": 0.02 * math.sin(hist_t / 5),
                 "value_loss": 0.4 + 0.05 * math.cos(hist_t / 9), "entropy": 1.9 - hist_t * 0.001,
                 "approx_kl": 0.008, "cpu_percent": 55 + 10 * math.sin(hist_t), "rss_mb": 812.5,
                 "mem_warning": None, "elapsed_s": hist_t,
                 "eta_s": (target - env_steps) / sps if hist_t > 20 else None,
                 "latest_checkpoint": "ckpt/run-0001/step_0100000.pt", "curriculum_stage": "wood",
                 "num_envs": args.num_envs,
                 "config": {"total_steps": target, "num_envs": args.num_envs, "lr": 3e-4, "minibatch_size": 512,
                            "rollout_len": 128, "ckpt_every": 50000, "workers": 4, "mode": "live"}}
            for c in clients:
                c.send(m)
        if now >= next_snap:
            next_snap = now + 0.1
            for c in clients:
                if not c.snapshot:
                    continue
                ph = hist_t * 0.6
                px = 16.5 + 3 * math.cos(ph)
                pz = 16.5 + 3 * math.sin(ph)
                py = 16.0 + 1.0
                yaw = (math.degrees(-ph) + 90) % 360
                c.send({"type": "snapshot", "env_id": c.selected_env, "episode": 3, "seed": 1234,
                        "step": int(hist_t * 20) % 3000,
                        "player": {"x": px, "y": py, "z": pz, "vx": -3 * math.sin(ph) * 0.6,
                                   "vy": 0.0, "vz": 3 * math.cos(ph) * 0.6, "yaw": yaw, "pitch": -12.0,
                                   "health": 18.0, "on_ground": True},
                        "last_action": ACTIONS[int(hist_t) % 8], "last_reward": 0.05, "ep_reward": 1.7,
                        "task": "collect wood", "target": {"x": 16, "y": 16, "z": 16, "block": "LOG"},
                        "inventory": {"LOG": 3, "PLANKS": 4, "STICK": 2},
                        "action_history": [ACTIONS[(int(hist_t * 4) + i) % len(ACTIONS)] for i in range(32)],
                        "origin": [0, 0, 0], "size": [SX, SY, SZ],
                        "blocks": base64.b64encode(region).decode(),
                        "obs_res": RES, "obs": base64.b64encode(make_obs(hist_t)).decode()})
        if now >= next_acts:
            next_acts = now + 0.5
            for c in clients:
                if not c.nn:
                    continue
                probs = [math.exp(-((i - 1 - 3 * (math.sin(hist_t / 3) + 1) / 2) ** 2) / 4) for i in range(len(ACTIONS))]
                tot = sum(probs)
                probs = [p / tot for p in probs]
                layers = {}
                for name, n in [("conv1", 256), ("conv2", 256), ("vec_enc", 64), ("fuse", 128), ("gru", 128)]:
                    samp = [math.sin(i * 0.3 + hist_t) * 1.5 for i in range(min(n, 256))]
                    layers[name] = {"shape": [n], "sample": samp, "mean": 0.0, "std": 0.9, "min": -1.5,
                                    "max": 1.5,
                                    "hist": {"edges": [-1.5 + 3 * i / 16 for i in range(17)],
                                             "counts": [abs(8 - i) * 3 + 1 for i in range(16)]}}
                c.send({"type": "activations", "env_id": c.selected_env, "step": int(hist_t * 20),
                        "action_probs": probs, "entropy": 1.9, "value": 0.42 + 0.1 * math.sin(hist_t),
                        "layers": layers,
                        "gru": {"mean": 0.01, "std": 0.4, "min": -0.9, "max": 0.9, "norm": 7.1},
                        "grad_norms": {"conv1": 0.12, "conv2": 0.09, "fuse": 0.31, "gru": 0.55, "policy_head": 0.2,
                                       "value_head": 0.07},
                        "weight_stats": {"conv1": {"mean": 0.0, "std": 0.1, "min": -0.5, "max": 0.5},
                                         "policy_head": {"mean": 0.0, "std": 0.05, "min": -0.3, "max": 0.3}},
                        "nan_inf": False})
        if int(hist_t) % 7 == 0 and int(hist_t * 10) % 10 == 0 and clients and state == "running":
            for c in clients:
                c.send({"type": "log", "t": hist_t, "level": "info", "msg": f"update {int(env_steps/4096)} done"})
    for c in clients:
        c.sock.close()
    srv.close()
    print("[fake_server] done", flush=True)


if __name__ == "__main__":
    main()
