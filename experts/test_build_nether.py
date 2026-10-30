"""Tests for experts/skills/build_nether.py (pure helpers, no sim): run with
    PYTHONPATH=. .venv/bin/python -m experts.test_build_nether
"""
import random

from experts.skills import build_nether as bn


def test_blueprint_counts():
    for name in ("dirt_hut", "wall_segment", "farm_plot", "bridge", "nether_portal"):
        bpd = bn.load_blueprint(name)
        counts = {}
        for _, _, _, item in bpd["solids"]:
            counts[item] = counts.get(item, 0) + 1
        assert counts == bpd["required"], (name, counts, bpd["required"])


def test_rot_is_bijection():
    bpd = bn.load_blueprint("bridge")
    for rot in range(4):
        cells = {bn.rot_xz(bpd["sx"], bpd["sz"], rot, x, z) for x in range(bpd["sx"]) for z in range(bpd["sz"])}
        assert len(cells) == bpd["sx"] * bpd["sz"]


def test_solve_roundtrip():
    """Place a blueprint at a random pose, synthesize the 8 nearest-unfilled obs slots, and solve back."""
    rng = random.Random(3)
    for name in ("dirt_hut", "wall_segment", "farm_plot"):
        bpd = bn.load_blueprint(name)
        rot, ox, oy, oz = rng.randrange(4), 60, 44, 50
        solids, _, _ = bn.place_cells(bpd, rot, ox, oy, oz)
        px, py, pz = 50.5, 44.0, 45.5
        d = sorted(solids, key=lambda c: (c[0] + 0.5 - px) ** 2 + (c[1] + 0.5 - py) ** 2 + (c[2] + 0.5 - pz) ** 2)
        obs = d[:8]
        got = bn.solve_target(bpd, obs, (px, py, pz))
        assert got is not None, name
        s2, _, _ = bn.place_cells(bpd, *got)
        assert sorted(s2) == sorted(solids), name


if __name__ == "__main__":
    test_blueprint_counts()
    test_rot_is_bijection()
    test_solve_roundtrip()
    print("build_nether tests ok")
