#!/usr/bin/env python3
"""Unit tests for tasklib and tools/knowledge/build_graph.py. Plain asserts; run:
    .venv/bin/python tasklib/test_tasklib.py
Uses tools/knowledge/fixtures so they pass before the real registry exists. Real-data checks
run only when data/registry/items.json and data/tasks/tasks.json are present.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "knowledge"))

import build_graph as bg  # noqa: E402
from tasklib import (TaskSampler, describe, recipe_path, requirements, validate_tasks,  # noqa: E402
                     validate_envmix, normalize_predicates, load_envmix, load_tasks,
                     KnowledgeGraph, FAMILIES)
from tasklib.loaders import load_blueprints  # noqa: E402

FIX = REPO / "tools" / "knowledge" / "fixtures"
REG = FIX / "registry"


def _fixture_registry():
    return bg.load_registry(REG)


def _fixture_graph():
    reg = _fixture_registry()
    graph, warnings = bg.build_graph(reg)
    return graph, reg, warnings


def _fixture_tasks():
    return load_tasks(FIX / "tasks.json")


def _fixture_envmix():
    return {
        "name": "fixture",
        "mix": {"gather": 0.3, "tools": 0.2, "food": 0.1, "build": 0.1, "social": 0.3},
        "curriculum": {"enabled": True, "ema_alpha": 0.5, "unlock_ema_threshold": 0.5,
                       "mastered_ema_threshold": 0.9, "min_episodes_before_unlock": 3,
                       "review_weight": 0.0},
        "kits": {"use_task_start_kit": True, "start_kit_probability": 1.0,
                 "drop_start_kit_when_mastered": True,
                 "rules": [{"family": "tools", "kit": {"stick": 2}, "probability": 1.0}]},
    }


# ---------------------------------------------------------------- graph
def test_graph_depths_and_sources():
    graph, _, _ = _fixture_graph()
    items = graph["items"]
    assert items["oak_log"]["via"] == "mine" and items["oak_log"]["depth"] == 0
    assert items["oak_planks"]["via"] == "craft" and items["oak_planks"]["depth"] == 1
    assert items["stick"]["depth"] == 2
    assert items["crafting_table"]["depth"] == 2
    assert items["wooden_pickaxe"]["depth"] == 3
    assert items["cobblestone"]["via"] == "mine"
    assert items["cobblestone"]["tool_tier"] == 1 and items["cobblestone"]["depth"] == 4
    assert items["furnace"]["depth"] == 5
    assert items["raw_iron"]["depth"] == 6
    assert items["iron_ingot"]["via"] == "smelt" and items["iron_ingot"]["depth"] == 7
    assert items["iron_pickaxe"]["depth"] == 8
    assert items["diamond"]["depth"] == 9
    assert items["beef"]["via"] == "kill" and items["beef"]["depth"] == 0
    assert items["cooked_beef"]["via"] == "smelt" and items["cooked_beef"]["depth"] == 6
    assert items["netherite_ingot"]["depth"] is None
    assert graph["unobtainable"] == ["netherite_ingot"]


def test_graph_requirements_closure():
    graph, _, _ = _fixture_graph()
    req = set(graph["items"]["diamond"]["requirements"])
    expect = {"iron_pickaxe", "iron_ingot", "raw_iron", "furnace", "cobblestone", "stone_pickaxe",
              "wooden_pickaxe", "stick", "oak_planks", "oak_log", "crafting_table"}
    assert req == expect, req ^ expect
    assert "diamond" not in req
    assert set(graph["items"]["oak_log"]["requirements"]) == set()


def test_graph_recipe_and_station_prereqs():
    graph, _, _ = _fixture_graph()
    assert graph["items"]["iron_ingot"]["prereqs"] == ["furnace", "raw_iron"]
    assert graph["items"]["stick"]["recipe_id"] == 1


def test_recipe_path_order():
    graph, _, _ = _fixture_graph()
    kg = KnowledgeGraph(graph)
    steps = recipe_path("diamond", graph=kg)
    names = [s["item"] for s in steps]
    assert names[-1] == "diamond"
    seen = set()
    for s in steps:
        for p in kg.entry(s["item"])["prereqs"]:
            assert p in seen, (s["item"], p)
        seen.add(s["item"])
    assert set(names) == set(graph["items"]["diamond"]["requirements"]) | {"diamond"}
    try:
        recipe_path("netherite_ingot", graph=kg)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_requirements_helper():
    graph, _, _ = _fixture_graph()
    assert "furnace" in requirements("iron_ingot", graph=KnowledgeGraph(graph))


def test_graph_is_deterministic():
    g1, _, _ = _fixture_graph()
    g2, _, _ = _fixture_graph()
    assert json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True)


def test_graph_warns_on_bad_refs():
    reg = _fixture_registry()
    reg["recipes"].append({"id": 99, "result": "stick", "ingredients": {"ghost": 1}, "needs": "inventory"})
    _, warnings = bg.build_graph(reg)
    assert any("unknown ingredients" in w for w in warnings)


# ---------------------------------------------------------------- shaping
def test_shaping_targets_and_prereqs():
    graph, reg, _ = _fixture_graph()
    sh = bg.build_shaping(graph, reg, _fixture_tasks(), load_blueprints(FIX / "blueprints.json"))
    d = sh["tasks"]["mine_diamond"]
    assert d["target_items"] == ["diamond"]
    assert "furnace" in d["items"] and "diamond" not in d["items"]
    assert sh["tasks"]["kill_cow"]["target_items"] == ["beef"]
    hut = sh["tasks"]["build_hut"]
    assert set(hut["target_items"]) == {"oak_planks", "chest"}
    assert "crafting_table" in hut["items"] and "oak_planks" not in hut["items"]
    assert sh["tasks"]["social_wave"]["items"] == []
    assert set(sh["tasks"]) == {t["id"] for t in _fixture_tasks()}


# ---------------------------------------------------------------- schema
def test_validate_fixture_tasks_ok():
    assert validate_tasks(_fixture_tasks()) == []


def test_validate_catches_problems():
    bad = [
        {"id": "a", "family": "gather", "success": {"type": "has_item", "item": "x"}, "prerequisites": ["b"]},
        {"id": "b", "family": "nope", "success": {"type": "mined"}, "prerequisites": ["a"]},
        {"id": "c", "family": "social", "success": {"type": "bogus_predicate"}},
    ]
    errs = validate_tasks(bad)
    assert any("unknown family" in e for e in errs)
    assert any("cycle" in e for e in errs)
    assert any("no recognised success" in e for e in errs)


def test_normalize_predicate_forms():
    forms = [
        {"type": "has_item", "item": "a", "count": 2},
        {"has_item": {"item": "a", "count": 2}},
        [{"predicate": {"has_item": {"item": "a", "count": 2}}, "reward": 1}],
    ]
    for f in forms:
        got = list(normalize_predicates(f))
        assert got == [("has_item", {"item": "a", "count": 2})], (f, got)


def test_social_family_and_predicates():
    assert "social" in FAMILIES
    got = list(normalize_predicates({"type": "crouch_greet", "count": 2}))
    assert got == [("crouch_greet", {"count": 2})]
    assert list(normalize_predicates({"type": "gave_item", "count": 1})) == [("gave_item", {"count": 1})]


def test_validate_envmix_shipped_files():
    for name in ("default", "gather_heavy", "build_heavy"):
        env = load_envmix(name)
        assert validate_envmix(env) == [], name
        assert abs(sum(env["mix"].values()) - 1.0) < 1e-6, name
        assert abs(env["mix"]["social"] - 0.1) < 1e-9, name


def test_validate_envmix_rejects_bad():
    assert validate_envmix({"mix": {"nope": 1.0}})
    assert validate_envmix({"mix": {"gather": 1.0}, "curriculum": {"ema_alpha": 0}})


# ---------------------------------------------------------------- sampler
def _sampler(seed=1, num_worlds=4):
    return TaskSampler(num_worlds, seed=seed, tasks=_fixture_tasks(), envmix=_fixture_envmix())


def test_initial_unlocks_are_roots_only():
    s = _sampler()
    assert s.unlocked("gather_wood") and s.unlocked("kill_cow") and s.unlocked("social_wave")
    assert not s.unlocked("make_wood_pickaxe") and not s.unlocked("social_follow")


def test_sampler_only_picks_unlocked_and_covers_families():
    s = _sampler(seed=3, num_worlds=8)
    seen_fams = set()
    for _ in range(300):
        for a in s.sample_episode():
            assert s.unlocked(a.task), a
            seen_fams.add(a.family)
            assert a.family == s.tasks[a.task]["family"]
            s.update(a.world, a.task, 0.0)  # keep everything locked behind roots
    assert "social" in seen_fams and "gather" in seen_fams


def test_curriculum_unlocks_after_prereq_mastery():
    s = _sampler()
    for _ in range(3):
        s.update(0, "gather_wood", 1.0)
    assert s.unlocked("make_wood_pickaxe") and s.unlocked("build_chest")
    assert not s.unlocked("make_stone_pickaxe")  # still needs gather_stone
    for _ in range(3):
        s.update(0, "gather_stone", 1.0)
        s.update(0, "make_wood_pickaxe", 1.0)
    assert s.unlocked("make_stone_pickaxe")


def test_social_chain_unlocks():
    s = _sampler()
    assert not s.unlocked("social_follow")
    for _ in range(3):
        s.update(1, "social_wave", 1.0)
    assert s.unlocked("social_follow")


def test_ema_updates():
    s = _sampler()
    s.update(0, "gather_wood", 1.0)
    assert s.ema["gather_wood"] == 1.0
    s.update(0, "gather_wood", 0.0)
    assert abs(s.ema["gather_wood"] - 0.5) < 1e-9
    assert s.episodes["gather_wood"] == 2


def test_kit_rules_and_start_kit():
    s = _sampler(seed=5)
    kits = {}
    for _ in range(200):
        for a in s.sample_episode():
            kits.setdefault(a.task, set()).update(a.kit)
    if "build_chest" in kits:
        assert "oak_planks" in kits["build_chest"]
    s2 = _sampler(seed=5)
    for _ in range(3):
        s2.update(0, "build_chest", 1.0)  # mastered -> start kit dropped
    assert s2._kit("build_chest", "build") == {}


def test_state_roundtrip_is_deterministic():
    a = _sampler(seed=7)
    for _ in range(20):
        for x in a.sample_episode():
            a.update(x.world, x.task, 1.0 if x.family == "gather" else 0.0)
    state = json.loads(json.dumps(a.state_dict()))  # must be JSON-serialisable
    b = _sampler(seed=99)
    b.load_state_dict(state)
    assert b.ema == a.ema and b.episodes == a.episodes
    seq_a = [x.task for _ in range(30) for x in a.sample_episode()]
    seq_b = [x.task for _ in range(30) for x in b.sample_episode()]
    assert seq_a == seq_b


def test_state_mismatch_rejected():
    a = _sampler()
    st = a.state_dict()
    st["ema"].pop("gather_wood")
    try:
        _sampler().load_state_dict(st)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_mix_string_and_dict_overrides():
    s = TaskSampler(2, mix={"food": 1.0}, seed=0, tasks=_fixture_tasks(), envmix=_fixture_envmix())
    fams = {x.family for _ in range(50) for x in s.sample_episode()}
    assert fams == {"food"}


def test_describe_contains_key_fields():
    s = _sampler()
    line = s.describe("make_wood_pickaxe")
    assert "make_wood_pickaxe" in line and "[tools]" in line and "crafted" in line
    assert "unlocked=False" in line
    assert "[social]" in describe(next(t for t in _fixture_tasks() if t["family"] == "social"))


# ---------------------------------------------------------------- real data (optional)
def test_real_data_if_present():
    reg_dir = REPO / "data" / "registry"
    tasks_path = REPO / "data" / "tasks" / "tasks.json"
    if not ((reg_dir / "items.json").exists() and tasks_path.exists()):
        print("  (real data not present; skipped)")
        return
    reg = bg.load_registry(reg_dir)
    graph, warnings = bg.build_graph(reg)
    tasks = load_tasks(tasks_path)
    errs = validate_tasks(tasks)
    assert not errs, errs
    for w in warnings[:10]:
        print("  warn:", w)
    kg = KnowledgeGraph(graph)
    for t in tasks:
        for name, params in normalize_predicates(t["success"]):
            item = params.get("item")
            if item:
                assert item in graph["items"], (t["id"], item)
    for env_name in ("default", "gather_heavy", "build_heavy"):
        assert validate_envmix(load_envmix(env_name)) == []
    s = TaskSampler(32, mix="default", seed=0, tasks=tasks)
    for _ in range(20):
        for a in s.sample_episode():
            assert s.unlocked(a.task)
    print(f"  real data: {graph['counts']}")
    if (REPO / "data" / "knowledge" / "graph.json").exists():
        on_disk = json.loads((REPO / "data" / "knowledge" / "graph.json").read_text())
        assert on_disk["items"].keys() == graph["items"].keys(), "graph.json is stale"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
