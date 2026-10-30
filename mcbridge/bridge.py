"""Policy bridge for the Minecraft CraftBot mod.

Serves one MineMind checkpoint on 127.0.0.1 (one JSON object per line):

  {"cmd": "info"}                                  -> {"res", "vec_dim", "num_blocks", "num_actions", "version"}
  {"bot", "reset", "res", "img" (b64 uint8 res*res*2), "vec" (floats)}
                                                   -> {"action", "msg", "prob", "value"}
  any request the bridge cannot serve             -> {"error": "..."}

Layouts come from the checkpoint itself: v6 (res 64, VEC 102, 20 blocks, 28 actions, msg head) or the
legacy v5 layout (res 32, VEC 39, 18 blocks). A legacy checkpoint without a msg head always answers msg 0.
The mod sends the layout the info command reports, so a "res" mismatch is refused with an error.

    .venv/bin/python mcbridge/bridge.py [--ckpt PATH] [--port 7788] [--greedy]
    .venv/bin/python mcbridge/bridge.py --train --ckpt PATH [--port 7788] [--tele-port 7777] [--out runs/mc-ft]
                                        [--lr 1e-5] [--rollout 128] [--kl-coef 0.1] [--clip 0.1]
    .venv/bin/python mcbridge/bridge.py --train --resume <mc-ft ckpt> [--out runs/mc-ft] (FT-2 warmups, KL target,
                                        entropy floor, goal weights and best checkpoints: see mc_train.py)

Without --ckpt the newest v6 checkpoint under runs/*/checkpoints (by mtime) is used.

--train (PROTOCOL 11.4): the same policy answers play requests and also learns online from the batched
{"cmd": "train_step"} requests of `/aibot train start` (mcbridge/mc_train.py). Telemetry for the dashboard
on 127.0.0.1:--tele-port (default 7777). Checkpoints go to <out>/<run_id>/checkpoints/.
"""
import argparse
import base64
import glob
import json
import os
import signal
import socketserver
import sys
import threading

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trainer.model import MineMindPolicy  # noqa: E402
from trainer import checkpoint  # noqa: E402

V6_VEC, V6_RES, V6_BLOCKS = 102, 64, 20
V5_VEC, V5_RES, V5_BLOCKS = 39, 32, 18
DEFAULT_NUM_ACTIONS, DEFAULT_NUM_MSGS = 28, 8


def newest_checkpoint(want_vec=V6_VEC):
    """Newest (by mtime) checkpoint whose VEC_DIM is want_vec, or None."""
    paths = glob.glob(os.path.join(ROOT, "runs", "*", "checkpoints", "ckpt_*.pt"))
    for path in sorted(paths, key=os.path.getmtime, reverse=True):
        try:
            payload = checkpoint.load(path)
        except Exception:  # unreadable or other schema: skip
            continue
        if _layout_of(payload)["vec_dim"] == want_vec:
            return path
    return None


def _layout_of(payload):
    """Observation layout of a checkpoint: from its obs signature, else inferred from the weights."""
    sd = payload["model"]
    obs = payload.get("obs") or {}
    vec_dim = int(obs.get("vec_dim", sd["vec_mlp.0.weight"].shape[1]))
    if vec_dim == V6_VEC:
        res, blocks = V6_RES, V6_BLOCKS
    elif vec_dim == V5_VEC:
        res, blocks = V5_RES, V5_BLOCKS
    else:
        raise ValueError(f"unsupported checkpoint VEC_DIM={vec_dim} (expected {V5_VEC} or {V6_VEC})")
    return {
        "vec_dim": vec_dim,
        "res": int(obs.get("res", res)),
        "num_blocks": int(obs.get("num_blocks", sd["block_embed.weight"].shape[0])),
        "num_actions": int(obs.get("num_actions", sd["policy_head.weight"].shape[0])),
        "num_msgs": int(obs.get("num_msgs", sd["msg_head.weight"].shape[0] if "msg_head.weight" in sd else DEFAULT_NUM_MSGS)),
        "has_msg": "msg_head.weight" in sd,
        "version": "v6" if vec_dim == V6_VEC else "v5",
    }


class Policy:
    def __init__(self, path, greedy):
        payload = checkpoint.load(path)
        sd = payload["model"]
        self.layout = _layout_of(payload)
        cfg = payload.get("config", {}) or {}
        L = self.layout
        self.net = MineMindPolicy(res=L["res"], vec_dim=L["vec_dim"], num_actions=L["num_actions"],
                                  num_blocks=L["num_blocks"], size=cfg.get("model_size", "small"),
                                  resblock=cfg.get("resblock", True), num_msgs=L["num_msgs"])
        self.net.load_state_dict(sd, strict=False)  # unused heads of older checkpoints may be absent
        self.net.eval()
        self.greedy = greedy
        self.hidden = {}
        self.lock = threading.Lock()
        self.path = path
        counters = payload.get("counters", {})
        print(f"loaded {path}  (env_steps={counters.get('env_steps', 0):,}, {L['version']}: res={L['res']}, "
              f"VEC_DIM={L['vec_dim']}, NUM_BLOCKS={L['num_blocks']}, NUM_ACTIONS={L['num_actions']}, "
              f"msg_head={L['has_msg']})", flush=True)

    def info(self):
        L = self.layout
        return {"res": L["res"], "vec_dim": L["vec_dim"], "num_blocks": L["num_blocks"],
                "num_actions": L["num_actions"], "version": L["version"]}

    def _inputs(self, msg):
        L = self.layout
        res = int(msg.get("res", L["res"]))
        if res != L["res"]:
            raise ValueError(f"res {res} sent, checkpoint expects {L['res']} ({L['version']} layout)")
        raw = base64.b64decode(msg["img"])
        if len(raw) != res * res * 2:
            raise ValueError(f"image is {len(raw)} bytes, expected {res * res * 2}")
        img = np.frombuffer(raw, np.uint8).reshape(1, res, res, 2).copy()
        vec = np.asarray(msg["vec"], np.float32).reshape(1, -1)
        if vec.shape[1] != L["vec_dim"]:
            raise ValueError(f"vec has {vec.shape[1]} floats, checkpoint expects {L['vec_dim']}")
        img[..., 0] = np.minimum(img[..., 0], L["num_blocks"] - 1)
        return torch.from_numpy(img), torch.from_numpy(vec)

    @torch.no_grad()
    def act(self, msg):
        bot = msg.get("bot", "?")
        img, vec = self._inputs(msg)
        with self.lock:
            h = self.hidden.get(bot)
            if h is None or msg.get("reset"):
                h = self.net.zero_hidden(1)
            logits, msg_logits, value, h = self.net(img, vec, h)
            self.hidden[bot] = h
        probs = torch.softmax(logits[0], -1)
        a = int(probs.argmax()) if self.greedy else int(torch.multinomial(probs, 1))
        if self.layout["has_msg"]:
            mprobs = torch.softmax(msg_logits[0], -1)
            m = int(mprobs.argmax()) if self.greedy else int(torch.multinomial(mprobs, 1))
        else:
            m = 0
        return {"action": a, "msg": m, "prob": float(probs[a]), "value": float(value[0])}

    def handle(self, line):
        req = json.loads(line)
        if req.get("cmd") == "info":
            return self.info()
        return self.act(req)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint (default: newest v6 under runs/)")
    ap.add_argument("--port", type=int, default=7788)
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    # online fine-tuning (--train, PROTOCOL 11.4)
    ap.add_argument("--train", action="store_true", help="learn online from the mod's train_step requests")
    ap.add_argument("--tele-port", type=int, default=7777, help="telemetry for the dashboard (train mode)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "mc-ft"), help="checkpoint root (train mode)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rollout", type=int, default=128, help="decisions per bot per PPO update")
    ap.add_argument("--kl-coef", type=float, default=0.1, help="per-step KL penalty to the start policy")
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--ckpt-minutes", type=float, default=5.0)
    # FT-2 (mc_train.py): warmups, adaptive KL, entropy floor, goal weighting, best checkpoint, resume
    ap.add_argument("--resume", default=None, help="mc-ft checkpoint to continue (model, optimizer, counters); "
                                                  "the KL reference stays its original start checkpoint")
    ap.add_argument("--norm-warmup", type=int, default=2, help="first N rollouts only collect reward stats")
    ap.add_argument("--critic-warmup", type=int, default=3, help="first N PPO updates train only the value head")
    ap.add_argument("--kl-target", type=float, default=0.02, help="target kl_to_start (adapts --kl-coef)")
    ap.add_argument("--kl-gain", type=float, default=0.5, help="log-proportional gain of the KL coefficient")
    ap.add_argument("--entropy-floor", type=float, default=0.5, help="raise ent_coef below this action entropy")
    ap.add_argument("--succ-ema", type=float, default=0.05, help="per-goal success EMA step (per episode)")
    ap.add_argument("--goal-min-eps", type=int, default=10, help="episodes before a goal's weight is adapted")
    ap.add_argument("--goal-lo", type=float, default=0.1, help="goal weight 1.0 inside [goal-lo, goal-hi]")
    ap.add_argument("--goal-hi", type=float, default=0.8)
    ap.add_argument("--best-window", type=int, default=200, help="episodes for the recent success (best ckpt)")
    ap.add_argument("--best-min-eps", type=int, default=30, help="episodes before a best checkpoint is saved")
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.resume and not args.train:
        sys.exit("--resume needs --train")
    path = args.resume or args.ckpt or newest_checkpoint()
    if not path:
        sys.exit("no v6 checkpoint found under runs/ (pass --ckpt)")
    if args.train and args.greedy:
        sys.exit("--train samples its actions (drop --greedy)")
    policy = Policy(path, args.greedy)
    trainer = None
    if args.train:
        from mc_train import MCTrainer  # noqa: E402  (mcbridge/ is this script's directory)
        trainer = MCTrainer(policy, args, args.tele_port, args.out)

    def dispatch(line):
        if trainer is not None:
            req = json.loads(line)
            if req.get("cmd") == "train_step":
                return trainer.step(req)
            if req.get("cmd") == "train_end":
                trainer.end(req.get("reason", ""))
                return {"ok": True}
        return policy.handle(line)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            print(f"mod connected from {self.client_address}", flush=True)
            for line in self.rfile:
                try:
                    reply = dispatch(line)
                except Exception as e:  # keep the connection alive; the mod reports the error
                    reply = {"error": f"{type(e).__name__}: {e}"}
                self.wfile.write((json.dumps(reply) + "\n").encode())
            print("mod disconnected", flush=True)

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    if trainer is not None:
        # a client that is still connected must not hold up shutdown (Ctrl-C saves the checkpoint first)
        socketserver.ThreadingTCPServer.daemon_threads = True
        import faulthandler  # SIGUSR1 dumps every thread's stack to the log (diagnostics)
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), Handler) as srv:
        print(f"policy bridge listening on 127.0.0.1:{args.port}"
              + (" (online training)" if trainer else ""), flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if trainer is not None:
                print("[mc-ft] shutting down: saving the final checkpoint", flush=True)
                trainer.close()


if __name__ == "__main__":
    main()
