"""Helpers over data/knowledge/graph.json (built by tools/knowledge/build_graph.py).

requirements(item)  -> frozenset of every item that must be obtained (transitively) first.
recipe_path(item)   -> ordered list of steps, raw inputs first, ending with `item`.
"""
from __future__ import annotations

from .loaders import load_graph

_default = None


class KnowledgeGraph:
    def __init__(self, data):
        self.data = data
        self.items = data["items"]

    @classmethod
    def load(cls, path=None):
        return cls(load_graph(path))

    def entry(self, item):
        if item not in self.items:
            raise KeyError(f"unknown item {item!r}")
        return self.items[item]

    def obtainable(self, item):
        return self.entry(item)["depth"] is not None

    def requirements(self, item):
        e = self.entry(item)
        return frozenset(e["requirements"])

    def recipe_path(self, item):
        if not self.obtainable(item):
            raise ValueError(f"{item!r} has no known acquisition source")
        order, seen = [], set()

        def visit(n):
            if n in seen:
                return
            seen.add(n)
            for p in self.items[n]["prereqs"]:
                visit(p)
            order.append(n)

        visit(item)
        steps = []
        for n in order:
            e = self.items[n]
            step = {"item": n, "via": e["via"], "from": e["from"],
                    "recipe_id": e.get("recipe_id"), "tool_tier": e.get("tool_tier"),
                    "depth": e["depth"]}
            if e.get("recipe"):
                step["recipe"] = e["recipe"]
            steps.append(step)
        return steps


def default_graph(path=None):
    global _default
    if path is not None:
        return KnowledgeGraph.load(path)
    if _default is None:
        _default = KnowledgeGraph.load()
    return _default


def requirements(item, graph=None):
    return (graph or default_graph()).requirements(item)


def recipe_path(item, graph=None):
    return (graph or default_graph()).recipe_path(item)
