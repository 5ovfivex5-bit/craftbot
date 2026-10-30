"""Atomic checkpoints under runs/<run_id>/checkpoints/.

Write path: tmp file in the same directory -> flush + fsync -> os.replace (atomic on POSIX).
Filename: ckpt_<env_steps:012d>_u<updates:06d>.pt (list() parses counters from the name).
"""
import os
import platform
import random
import re
import sys
import time

import numpy as np
import torch

SCHEMA_VERSION = 2   # 2: observation signature, curriculum and reward-normalizer state (obs v2)
_NAME_RE = re.compile(r"^ckpt_(\d{12})_u(\d{6})\.pt$")


def checkpoint_name(env_steps, updates):
    return f"ckpt_{env_steps:012d}_u{updates:06d}.pt"


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state):
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])


def save(ckpt_dir, payload):
    """Atomically write payload to ckpt_dir. Returns the final path."""
    os.makedirs(ckpt_dir, exist_ok=True)
    name = checkpoint_name(payload["counters"]["env_steps"], payload["counters"]["updates"])
    final = os.path.join(ckpt_dir, name)
    tmp = os.path.join(ckpt_dir, "." + name + f".{os.getpid()}.tmp")
    body = dict(payload)
    body["schema_version"] = SCHEMA_VERSION
    body["metadata"] = {
        "saved_at": time.time(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **payload.get("metadata", {}),
    }
    try:
        with open(tmp, "wb") as f:
            torch.save(body, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    dfd = os.open(ckpt_dir, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return final


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ver = payload.get("schema_version")
    if ver == 1 or (ver is None and "model" in payload):
        raise ValueError(f"{os.path.basename(path)} is a v1 checkpoint (obs v1: 25-dim vec, 14 blocks, "
                         f"no goal). It cannot be trained on by the v2 trainer; start a new run.")
    if ver != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema_version={ver} (expected {SCHEMA_VERSION})")
    return payload


def load_for_training(path):
    """load() plus the observation-layout check: refuses checkpoints whose policy inputs/outputs
    do not match the running sim (the error names the mismatched fields)."""
    from .envs import obs_mismatch
    payload = load(path)
    why = obs_mismatch(payload.get("obs"))
    if why is not None:
        raise ValueError(f"cannot use {os.path.basename(path)}: {why}")
    return payload


def list_checkpoints(runs_dir):
    """All checkpoints under runs_dir/*/checkpoints, newest first."""
    items = []
    if not os.path.isdir(runs_dir):
        return items
    for run in sorted(os.listdir(runs_dir)):
        cdir = os.path.join(runs_dir, run, "checkpoints")
        if not os.path.isdir(cdir):
            continue
        for name in os.listdir(cdir):
            m = _NAME_RE.match(name)
            if not m:
                continue
            path = os.path.join(cdir, name)
            st = os.stat(path)
            items.append({
                "path": path,
                "run_id": run,
                "env_steps": int(m.group(1)),
                "updates": int(m.group(2)),
                "time": st.st_mtime,
                "size_bytes": st.st_size,
            })
    items.sort(key=lambda x: x["time"], reverse=True)
    return items
