#!/usr/bin/env python3
"""Download the selected public model release and verify each SHA-256 digest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "5ovfivex5-bit/craftbot"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub OWNER/REPO")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("--repo must be OWNER/REPO")
    base = f"https://github.com/{args.repo}/releases/download/models-v1"
    output = ROOT / "models"
    output.mkdir(exist_ok=True)
    try:
        with urllib.request.urlopen(base + "/manifest.json", timeout=60) as response:
            manifest = json.load(response)
        items = manifest["checkpoints"]
        if {item["filename"] for item in items} != {"best-v6.pt", "best-v7.pt"} or len(items) != 2:
            raise ValueError("Unexpected model list in release manifest")
        for item in items:
            filename, expected = item["filename"], item["sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"Invalid SHA-256 for {filename}")
            destination = output / filename
            if destination.is_file() and digest(destination) == expected:
                print(f"Verified existing {filename}")
                continue
            temporary = output / (filename + ".tmp")
            try:
                print(f"Downloading {filename} ({item['bytes'] / 2**20:.1f} MiB)", flush=True)
                with urllib.request.urlopen(base + "/" + filename, timeout=60) as response:
                    with temporary.open("wb") as stream:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            stream.write(block)
                if temporary.stat().st_size != item["bytes"] or digest(temporary) != expected:
                    raise ValueError(f"Size or SHA-256 mismatch for {filename}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Both models verified. See docs/models.md for the matching --obs flag.")
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Model download failed: {exc}\nCheck the models-v1 release on {args.repo}.")


if __name__ == "__main__":
    main()
