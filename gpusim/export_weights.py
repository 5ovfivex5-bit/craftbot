"""Runs on the Colab VM next to the trainer: every --every seconds, turns the newest full checkpoint
(~1 GB with optimizer state) into a small fp16 weights-only file the Deck can download quickly.

    python gpusim/export_weights.py --run-dir /content/runs/gpu-100m [--every 600]

Writes <run-dir>/weights_<env_steps>.pt (newest 3 kept) and weights_latest.json (name, counters,
size, sha256) atomically. The file keeps
every checkpoint field except optimizer/normalizer state, so trainer.checkpoint.load() and eval accept it;
it cannot resume training (no optimizer), only evaluation / playing / warm-starting.
"""
import argparse
import glob
import hashlib
import json
import os
import time

import torch

DROP = {"optimizer", "scaler", "rng"}


def newest(ckpt_dir):
    paths = [p for p in glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")) if not os.path.basename(p).startswith(".")]
    return max(paths, key=os.path.getmtime) if paths else None


def export(src, run_dir):
    payload = torch.load(src, map_location="cpu", weights_only=False)
    out = {k: v for k, v in payload.items() if k not in DROP}
    out["model"] = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in payload["model"].items()}
    out["weights_only_export"] = {"source": os.path.basename(src), "dtype": "float16", "time": time.time()}
    steps = int(payload.get("counters", {}).get("env_steps", 0))
    name = f"weights_{steps:012d}.pt"
    tmp = os.path.join(run_dir, "." + name + ".tmp")
    torch.save(out, tmp)
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    os.replace(tmp, os.path.join(run_dir, name))
    meta = {"name": name, "source": os.path.basename(src), "counters": payload.get("counters", {}),
            "bytes": os.path.getsize(os.path.join(run_dir, name)), "sha256": h.hexdigest(), "time": time.time()}
    tmpj = os.path.join(run_dir, ".weights_latest.json.tmp")
    with open(tmpj, "w") as f:
        json.dump(meta, f)
    os.replace(tmpj, os.path.join(run_dir, "weights_latest.json"))
    # keep the newest 3 exports on the VM so an in-progress download is never deleted under it
    for old in sorted(glob.glob(os.path.join(run_dir, "weights_[0-9]*.pt")))[:-3]:
        os.remove(old)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--every", type=float, default=600)
    args = ap.parse_args()
    torch.set_num_threads(1)  # stay out of the trainer's way on the 2-vCPU VM
    done = None
    while True:
        src = newest(os.path.join(args.run_dir, "checkpoints"))
        # skip a checkpoint written in the last 20 s (may still be renaming)
        if src and src != done and time.time() - os.path.getmtime(src) > 20:
            try:
                meta = export(src, args.run_dir)
                done = src
                print(f"[export {time.strftime('%H:%M:%S')}] {meta['source']} -> {meta['bytes'] / 1e6:.0f} MB", flush=True)
            except Exception as e:  # a checkpoint pruned mid-read etc.: try again next round
                print(f"[export {time.strftime('%H:%M:%S')}] failed on {src}: {e}", flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
