#!/usr/bin/env python3
"""Export the two locally selected, trusted checkpoints as portable release assets.

Run inside distrobox hi. Full policy populations and optimizer state are retained
so the historical population evaluations and continued training remain possible.
"""
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = {
    "best-v6.pt": "runs/pop-human/checkpoints/ckpt_000002007040_u000845.pt",
    "best-v7.pt": "runs/v7-dapg/checkpoints/ckpt_000009789440_u001195.pt",
}


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(portable(item) for item in value)
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "")
    return value


def main():
    output = ROOT / "models"
    output.mkdir(exist_ok=True)
    manifest = {"release": "models-v1", "selection": "historical task evaluations; see docs/models.md",
                "checkpoints": []}
    for filename, source in SELECTIONS.items():
        path = ROOT / source
        if not path.is_file():
            raise SystemExit(f"Local training checkpoint missing: {source}. Use download-models on a fresh clone.")
        # These are the project's own local full-training checkpoints, not arbitrary input files.
        payload = portable(torch.load(path, map_location="cpu", weights_only=False))
        destination = output / filename
        temporary = output / (filename + ".tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        record = {"filename": filename, "source": source,
                  "sha256": digest, "bytes": destination.stat().st_size,
                  "parameters_per_policy": sum(t.numel() for t in payload["model"].values()),
                  "policies": len(payload.get("models", [])) or 1,
                  "obs": payload["obs"], "counters": payload["counters"]}
        manifest["checkpoints"].append(record)
        print(f"{filename}: {record['bytes'] / 2**20:.1f} MiB, SHA-256 {digest}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "SHA256SUMS.sha256").write_text("".join(
        f"{item['sha256']}  {item['filename']}\n" for item in manifest["checkpoints"]))


if __name__ == "__main__":
    main()
