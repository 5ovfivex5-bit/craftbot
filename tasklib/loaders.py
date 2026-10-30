"""File loaders for tasks, blueprints, the knowledge graph and env mixes. Pure Python."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_tasks(path=None):
    path = Path(path) if path else DATA / "tasks" / "tasks.json"
    data = _read(path)
    return list(data.get("tasks", [])) if isinstance(data, dict) else list(data)


def load_blueprints(path=None):
    path = Path(path) if path else DATA / "builds" / "blueprints.json"
    data = _read(path)
    return list(data.get("blueprints", [])) if isinstance(data, dict) else list(data)


def load_graph(path=None):
    path = Path(path) if path else DATA / "knowledge" / "graph.json"
    return _read(path)


def load_shaping(path=None):
    path = Path(path) if path else DATA / "knowledge" / "shaping.json"
    return _read(path)


def load_envmix(name_or_path="default"):
    """`name` -> data/envmix/<name>.json; or pass a path to a .json file."""
    p = Path(name_or_path)
    if p.suffix == ".json" or p.exists():
        return _read(p)
    return _read(DATA / "envmix" / f"{name_or_path}.json")
