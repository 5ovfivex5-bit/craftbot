"""Common loader for human-gameplay sessions: v2 (PROTOCOL 9.2) and v1 dirs (8.2, via recordings.py).

One interface for both (see `open_session`):
  s.read(t0, t1)       -> Ticks: per-tick arrays for ticks [t0, t1)
                          img uint8 [n,64,64,2], vec float32 [n,102], action int32 [n] (-1 unknown),
                          ext int32 [n] (VPT-style code, 9.4) or None for v1 sessions
  s.n_ticks, s.meta, s.goal, s.path, s.version, s.warnings
  s.input_records()    -> iterator of per-tick input dicts (tolerant parse, 8.2/9.2 input fields)
  s.events()           -> list of normalized events (see normalize_event)
  s.chunk_bounds()     -> [(t0, t1)] storage chunk ranges (v1: fixed 1200-tick ranges)
  s.actions()          -> int32 [n_ticks] per-tick MineMind labels (whole session, small)
  s.sim_steps(ticks=4) -> (first [T] int64 = first tick of each step, labels [T] int64, 4 ticks/step)

v2 chunk record (9.2), per frame, interleaved, little endian:
  img u8[64*64*2] | vec f32[102] | action i32 | ext_action i32      (8608 bytes/frame)
Chunk files are zlib (java.util.zip.Deflater) streams; `chunks/index.jsonl` lines are
{chunk, first_tick, frames, bytes}. A crashed or still-recording session may have a truncated last
chunk, a `.tmp` chunk, a partial index line, and an unfinalized meta.json: all tolerated, with warnings.

Field names for events are not fully fixed by 9.3, so normalize_event accepts several spellings
(vanilla id: item/block/id/vanilla; MineMind name: name/minemind; position: x,y,z or pos).
"""
import gzip
import json
import os
import re
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from pretrain import recordings as rec_lib

RES = 64
VEC_DIM = 102
TICK_RATE = 20                 # game ticks per second (20 Hz)
TICKS_PER_STEP = rec_lib.TICKS_PER_STEP
CHUNK_TICKS = 1200             # 1 minute of frames per storage chunk (9.2)
HINDSIGHT_TICKS = 5 * 60 * TICK_RATE   # 5 minutes (9.5)
FRAME_DTYPE = np.dtype([("img", "u1", (RES, RES, 2)), ("vec", "<f4", (VEC_DIM,)),
                        ("action", "<i4"), ("ext", "<i4")])
FRAME_BYTES = FRAME_DTYPE.itemsize   # 8608
_CHUNK_RE = re.compile(r"^(\d+)\.(bin|bin\.tmp|tmp)$")

OBTAIN_TYPES = {"pickup", "craft", "smelt"}          # hindsight goals: not chest/give/withdraw (9.5)
FIGHT_TYPES = {"attack", "kill", "killed_by", "damage_taken"}
V1_TYPE_MAP = {"break": "mine", "damage": "damage_taken"}
MOVE_KEYS = ("forward", "back", "left", "right", "jump")

# Vanilla item/block names that map to a MineMind ITEMS name (ITEMS are coarser than vanilla ids).
_MM_TABLE = {
    "stick": "stick", "crafting_table": "crafting_table", "wooden_pickaxe": "wooden_pickaxe",
    "cobblestone": "cobblestone", "stone_pickaxe": "stone_pickaxe", "furnace": "furnace",
    "coal": "coal", "coal_ore": "coal", "iron_ore": "iron_ore", "raw_iron": "iron_ore",
    "iron_ingot": "iron_ingot", "iron_pickaxe": "iron_pickaxe", "diamond": "diamond",
    "diamond_ore": "diamond", "dirt": "dirt", "chest": "chest", "planks": "planks",
}

# ---- item categories (vanilla ids, 9.5) ----
FOOD = {
    "apple", "golden_apple", "enchanted_golden_apple", "bread", "porkchop", "beef", "mutton", "chicken",
    "rabbit", "cod", "salmon", "tropical_fish", "pufferfish", "potato", "baked_potato", "poisonous_potato",
    "carrot", "golden_carrot", "beetroot", "melon_slice", "sweet_berries", "glow_berries", "dried_kelp",
    "cookie", "pumpkin_pie", "cake", "chorus_fruit", "honey_bottle", "rotten_flesh", "mushroom_stew",
    "beetroot_soup", "rabbit_stew", "suspicious_stew",
}
MOB_DROPS = {
    "bone", "arrow", "string", "spider_eye", "gunpowder", "ender_pearl", "blaze_rod", "blaze_powder",
    "slime_ball", "leather", "feather", "egg", "ink_sac", "glow_ink_sac", "phantom_membrane",
    "prismarine_shard", "prismarine_crystals", "rabbit_hide", "rabbit_foot", "ghast_tear", "magma_cream",
    "honeycomb", "wither_skeleton_skull", "nautilus_shell", "heart_of_the_sea", "shulker_shell",
    "dragon_breath", "scute", "armadillo_scute", "bone_block_placeholder",
}
TOOL_TYPES = ("pickaxe", "axe", "shovel", "hoe", "sword", "spear", "bow", "crossbow", "trident",
              "shears", "fishing_rod", "flint_and_steel")
ARMOR_SUFFIX = ("_helmet", "_chestplate", "_leggings", "_boots")
ARMOR_EXTRA = {"shield", "turtle_helmet", "elytra"}
ORE_ITEMS = {"coal", "charcoal", "diamond", "emerald", "lapis_lazuli", "redstone", "quartz",
             "amethyst_shard", "netherite_scrap"}
STONE_ITEMS = {"stone", "cobblestone", "andesite", "diorite", "granite", "deepslate", "cobbled_deepslate",
               "tuff", "calcite", "blackstone", "basalt", "sandstone"}
BLOCK_ITEMS = {"dirt", "grass_block", "sand", "gravel", "glass", "torch", "chest", "crafting_table",
               "furnace", "ladder", "bricks", "clay", "mud", "terracotta", "bookshelf", "tnt", "hay_block"}
BLOCK_SUFFIX = ("_block", "_slab", "_stairs", "_wall", "_fence", "_door", "_trapdoor", "_glass", "_pane")


def _id(v):
    """Vanilla id without the namespace; dicts use their 'id' or 'name'."""
    if isinstance(v, dict):
        v = v.get("id") or v.get("name")
    if not isinstance(v, str) or not v:
        return None
    return v.split(":")[-1]


def items():
    import craftsim
    return list(craftsim.ITEMS)


def mm_of(vid):
    """MineMind item name for a vanilla id, or None."""
    if not vid:
        return None
    if vid in _MM_TABLE:
        return _MM_TABLE[vid]
    if vid.endswith("_log"):
        return "log"
    if vid.endswith("_planks"):
        return "planks"
    return None


def categorize(vid):
    """(category, subtype) for a vanilla id. Categories: wood, stone, ore, tool, armor, food, mob_drop,
    block, other. Tools: subtype = tool type (pickaxe, axe, sword, spear, ...)."""
    if not vid:
        return ("other", None)
    n = vid
    if n in FOOD or n.startswith("cooked_") or n.startswith("raw_") and n[4:] in FOOD:
        return ("food", None)
    if n in ARMOR_EXTRA or n.endswith(ARMOR_SUFFIX):
        return ("armor", None)
    for t in TOOL_TYPES:
        if n == t or n.endswith("_" + t):
            return ("tool", t)
    if n in MOB_DROPS:
        return ("mob_drop", None)
    if n in ("stick", "planks", "log") or n.endswith(("_log", "_wood", "_planks", "_stem")) \
            or n.startswith("stripped_"):
        return ("wood", None)
    if n.endswith("_ore") or n.startswith("raw_") or n in ORE_ITEMS or n.endswith(("_ingot", "_nugget")):
        return ("ore", None)
    if n in STONE_ITEMS:
        return ("stone", None)
    if n in BLOCK_ITEMS or n.endswith(BLOCK_SUFFIX):
        return ("block", None)
    return ("other", None)


def _got(vid, mm, count):
    return {"vanilla": vid, "item": mm if mm else mm_of(vid), "count": int(count or 1)}


def normalize_event(raw, v1=False):
    """Normalized event dict: tick, t (s), type, vanilla (item or mined block id), block, entity,
    weapon, count, got (list of {vanilla, item, count} obtained by this event), pos, gui."""
    typ = raw.get("type") or "unknown"
    if v1:
        typ = V1_TYPE_MAP.get(typ, typ)
    tick = int(raw.get("tick") or 0)
    t = raw.get("session_time_s")
    t = float(t) if t is not None else tick / TICK_RATE
    mm_raw = raw.get("name") or raw.get("minemind") or raw.get("mm")
    vid = _id(raw.get("item")) or _id(raw.get("vanilla")) or _id(raw.get("id"))
    if vid is None and mm_raw in items():
        vid = mm_raw
    block = _id(raw.get("block")) or (vid if typ == "mine" else None)
    entity = _id(raw.get("entity") or raw.get("mob"))
    weapon = _id(raw.get("weapon")) or _id(raw.get("held_item"))
    if typ in FIGHT_TYPES and weapon is None:
        weapon = "hand"
    count = int(raw.get("count") or 1)
    got = []
    if typ in OBTAIN_TYPES and vid:
        got.append(_got(vid, mm_raw if mm_raw in items() else None, count))
    if typ == "mine" and isinstance(raw.get("drops"), list):
        for d in raw["drops"]:
            if isinstance(d, dict):
                did = _id(d.get("item") or d.get("id"))
                got.append(_got(did, d.get("name"), d.get("count", 1)))
    pos = None
    if "x" in raw and "y" in raw and "z" in raw:
        pos = [raw["x"], raw["y"], raw["z"]]
    elif isinstance(raw.get("pos"), list) and len(raw["pos"]) == 3:
        pos = raw["pos"]
    out = {"tick": tick, "t": t, "type": typ, "vanilla": vid, "block": block, "entity": entity,
           "weapon": weapon, "count": count, "got": [g for g in got if g["item"] or g["vanilla"]],
           "pos": pos, "gui": raw.get("gui")}
    if typ in ("context_start", "context_end"):      # 9.6 player-written context notes
        out["ctx_id"] = raw.get("id")
        out["text"] = raw.get("text")
        out["reason"] = raw.get("reason")
    return out


def _parse_json_line(line):
    s = line.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except ValueError:
        pass
    # v1 recorder bug: missing_input samples are written as `"keys":{,` (see the real v1 files)
    try:
        return json.loads(s.replace('"keys":{,', '"keys":{'))
    except ValueError:
        return None


def _iter_lines(path):
    """Lines of a plain or gzip file. A truncated gzip (crash mid-write, SYNC_FLUSH) yields every line
    that decompresses; a corrupt tail stops the stream instead of raising."""
    if not os.path.isfile(path):
        return
    if not path.endswith(".gz"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line
        return
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = b""
    with open(path, "rb") as f:
        while True:
            raw = f.read(1 << 20)
            if not raw:
                break
            try:
                data = d.decompress(raw)
            except zlib.error:
                break
            pending += data
            parts = pending.split(b"\n")
            pending = parts.pop()
            for p in parts:
                yield p.decode("utf-8", "replace")
            if d.eof:
                break
    if pending:
        yield pending.decode("utf-8", "replace")


def _read_meta(path):
    p = os.path.join(path, "meta.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except ValueError:
        return {"_broken": True}


def _inflate(raw):
    """zlib stream -> bytes, keeping every decodable prefix (truncated or corrupt tail is dropped),
    cut to whole frames."""
    d = zlib.decompressobj()
    out = []
    step = 1 << 16
    broke = False
    for i in range(0, len(raw), step):
        try:
            out.append(d.decompress(raw[i:i + step]))
        except zlib.error:
            broke = True
            break
    if not broke:
        try:
            out.append(d.flush())
        except zlib.error:
            pass
    data = b"".join(out)
    return data[: len(data) // FRAME_BYTES * FRAME_BYTES]


@dataclass
class Ticks:
    t0: int
    img: np.ndarray
    vec: np.ndarray
    action: np.ndarray
    ext: Optional[np.ndarray]


class Session:
    version = None

    def __init__(self, path, meta):
        self.path = path
        self.meta = meta or {}
        self.warnings = []
        self.bad_input_lines = 0
        self._events = None
        self._actions = None

    @property
    def goal(self):
        return self.meta.get("goal")

    def _warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def actions(self):
        if self._actions is None:
            out = np.empty(self.n_ticks, np.int32)
            for c0, c1 in self.chunk_bounds():
                out[c0:c1] = self.read(c0, c1).action
            self._actions = out
        return self._actions

    def sim_steps(self, ticks=TICKS_PER_STEP):
        T = self.n_ticks // ticks
        first = np.arange(T, dtype=np.int64) * ticks
        labels = rec_lib.step_labels(self.actions()[:T * ticks], ticks)
        return first, labels

    def events(self):
        if self._events is None:
            self._events = self._load_events()
        return self._events

    def contexts(self):
        """Context segments (9.6): list of {id, text, start_tick, end_tick, reason}, sorted by start.
        Built from context_start/context_end events; meta.json `contexts` is the fallback when the
        events are missing. An open segment ends at the session end with reason "open"."""
        starts, ends = {}, {}
        for e in self.events():
            if e["type"] == "context_start":
                starts.setdefault(e["ctx_id"], e)
            elif e["type"] == "context_end":
                ends.setdefault(e["ctx_id"], e)
        out = []
        if starts:
            for cid, s in starts.items():
                en = ends.get(cid)
                out.append({"id": cid, "text": s["text"], "start_tick": s["tick"],
                            "end_tick": en["tick"] if en else self.n_ticks,
                            "reason": en["reason"] if en else "open"})
            for cid in ends:
                if cid not in starts:
                    self._warn(f"context_end {cid} without context_start")
        else:
            for c in self.meta.get("contexts") or []:
                out.append({"id": c.get("id"), "text": c.get("text"),
                            "start_tick": int(c.get("start_tick", 0)),
                            "end_tick": int(c["end_tick"]) if c.get("end_tick") is not None else self.n_ticks,
                            "reason": c.get("reason")})
        out.sort(key=lambda c: c["start_tick"])
        return out

    def step_context(self, first):
        """Per sim step (first tick of each step): the active context text, or None."""
        text = [None] * len(first)
        for c in self.contexts():
            lo = int(np.searchsorted(first, c["start_tick"], side="left"))
            hi = int(np.searchsorted(first, c["end_tick"], side="left"))
            for i in range(lo, hi):
                text[i] = c["text"]
        return text

    def describe(self):
        return {"path": self.path, "version": self.version, "ticks": self.n_ticks,
                "sim_steps": self.n_ticks // TICKS_PER_STEP, "goal": self.goal,
                "warnings": self.warnings, "bad_input_lines": self.bad_input_lines}


class V2Session(Session):
    version = 2
    CACHE = 6

    def __init__(self, path, meta):
        super().__init__(path, meta)
        if meta and meta.get("_broken"):
            self._warn("meta.json is not valid JSON (crash while writing?)")
            self.meta = {}
        if self.meta and not self.meta.get("finalized", True):
            self._warn("meta.json not finalized (crash or still recording)")
        if not self.meta:
            self._warn("no meta.json")
        if int(self.meta.get("res", RES)) != RES or int(self.meta.get("vec_dim", VEC_DIM)) != VEC_DIM:
            raise ValueError(f"{path}: res/vec_dim {self.meta.get('res')}/{self.meta.get('vec_dim')}, "
                             f"expected {RES}/{VEC_DIM}")
        self._cache = OrderedDict()
        self._chunks = self._scan_chunks()
        self.n_ticks = sum(c["frames"] for c in self._chunks)
        self._starts = np.array([c["first"] for c in self._chunks], np.int64)

    def _scan_chunks(self):
        cdir = os.path.join(self.path, "chunks")
        index = {}
        ip = os.path.join(cdir, "index.jsonl")
        if os.path.isfile(ip):
            with open(ip) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        self._warn("index.jsonl has an unparseable line (crash?)")
                        continue
                    index[int(e["chunk"])] = e
        files = {}
        if os.path.isdir(cdir):
            for name in os.listdir(cdir):
                m = _CHUNK_RE.match(name)
                if not m:
                    continue
                num = int(m.group(1))
                # prefer the finished .bin over a leftover temp file
                if num not in files or name.endswith(".bin"):
                    files[num] = (os.path.join(cdir, name), name)
                if not name.endswith(".bin"):
                    self._warn(f"temp chunk {name} present (crash?) and read as partial")
        for num in index:
            if num not in files:
                self._warn(f"chunk {num:06d} is in index.jsonl but its file is missing")
        chunks = []
        prev_end = 0
        for num in sorted(files):
            path, name = files[num]
            e = index.get(num)
            size = os.path.getsize(path)
            if e is not None and e.get("bytes") == size:
                frames = int(e["frames"])
                first = int(e.get("first_tick", prev_end))
            else:
                # unindexed or size mismatch: count the frames that really decode
                with open(path, "rb") as f:
                    data = _inflate(f.read())
                frames = len(data) // FRAME_BYTES
                first = int(e["first_tick"]) if e is not None and "first_tick" in e else prev_end
                why = "not in index.jsonl" if e is None else f"size {size} != index bytes {e.get('bytes')}"
                self._warn(f"chunk {num:06d} partial/unindexed ({why}); {frames} frames decoded")
            if first != prev_end and chunks:
                self._warn(f"chunk {num:06d} first_tick {first} != previous end {prev_end}")
            chunks.append({"num": num, "file": path, "first": first, "frames": frames})
            prev_end = first + frames
        return chunks

    def _load_chunk(self, c):
        key = c["num"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with open(c["file"], "rb") as f:
            data = _inflate(f.read())
        rec = np.frombuffer(data, dtype=FRAME_DTYPE)
        if len(rec) < c["frames"]:
            self._warn(f"chunk {key:06d} shorter than expected: {len(rec)} < {c['frames']} frames")
        self._cache[key] = rec
        while len(self._cache) > self.CACHE:
            self._cache.popitem(last=False)
        return rec

    def chunk_bounds(self):
        return [(c["first"], c["first"] + c["frames"]) for c in self._chunks]

    def read(self, t0, t1):
        t0 = max(0, int(t0))
        t1 = min(self.n_ticks, int(t1))
        if t1 <= t0:
            return Ticks(t0, np.zeros((0, RES, RES, 2), np.uint8), np.zeros((0, VEC_DIM), np.float32),
                         np.zeros(0, np.int32), np.zeros(0, np.int32))
        parts = []
        for c in self._chunks:
            a, b = c["first"], c["first"] + c["frames"]
            if b <= t0 or a >= t1:
                continue
            rec = self._load_chunk(c)
            lo, hi = max(t0, a) - a, min(t1, b) - a
            hi = min(hi, len(rec))
            if hi > lo:
                parts.append(rec[lo:hi])
        rec = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return Ticks(t0, rec["img"].copy(), rec["vec"].astype(np.float32, copy=True),
                     rec["action"].astype(np.int32, copy=True), rec["ext"].astype(np.int32, copy=True))

    def input_records(self):
        self.bad_input_lines = 0
        for line in _iter_lines(os.path.join(self.path, "input.jsonl.gz")) or []:
            rec = _parse_json_line(line)
            if rec is None:
                self.bad_input_lines += 1
                continue
            yield rec
        if self.bad_input_lines:
            self._warn(f"input: {self.bad_input_lines} unparseable lines skipped")

    def _load_events(self):
        out = []
        bad = 0
        for line in _iter_lines(os.path.join(self.path, "events.jsonl.gz")) or []:
            rec = _parse_json_line(line)
            if rec is None:
                bad += 1
                continue
            out.append(normalize_event(rec, v1=False))
        if bad:
            self._warn(f"events: {bad} unparseable lines skipped")
        return out


class V1Session(Session):
    version = 1

    def __init__(self, path):
        meta = _read_meta(path) or {}
        super().__init__(path, meta)
        self.rec = rec_lib.load(path, strict=False)
        self.meta = self.rec.meta
        self.warnings.extend(self.rec.warnings)
        self.n_ticks = self.rec.frames
        self._chunk = CHUNK_TICKS

    def chunk_bounds(self):
        return [(a, min(self.n_ticks, a + self._chunk)) for a in range(0, self.n_ticks, self._chunk)]

    def read(self, t0, t1):
        t0 = max(0, int(t0))
        t1 = min(self.n_ticks, int(t1))
        img = np.array(self.rec.img[t0:t1])
        vec = np.array(self.rec.vec[t0:t1], np.float32)
        act = np.array(self.rec.action[t0:t1], np.int32)
        return Ticks(t0, img, vec, act, None)

    def input_records(self):
        self.bad_input_lines = 0
        for line in _iter_lines(os.path.join(self.path, "input.jsonl")) or []:
            rec = _parse_json_line(line)
            if rec is None:
                self.bad_input_lines += 1
                continue
            yield rec
        if self.bad_input_lines:
            self._warn(f"input.jsonl: {self.bad_input_lines} unparseable lines skipped "
                       f"(v1 writes `\"keys\":{{,` for missing input; repaired where possible)")

    def _load_events(self):
        out = []
        for line in _iter_lines(os.path.join(self.path, "events.jsonl")) or []:
            rec = _parse_json_line(line)
            if rec is not None:
                out.append(normalize_event(rec, v1=True))
        return out


def is_session(path):
    if not os.path.isdir(path):
        return False
    if os.path.isdir(os.path.join(path, "chunks")):
        return True
    return os.path.isfile(os.path.join(path, "meta.json")) or os.path.isfile(os.path.join(path, "action.i32"))


def open_session(path):
    path = os.path.abspath(path)
    meta = _read_meta(path) or {}
    fmt = str(meta.get("format", ""))
    if os.path.isdir(os.path.join(path, "chunks")) or fmt.startswith("minemind-rec-2"):
        return V2Session(path, meta)
    if fmt == rec_lib.FORMAT or os.path.isfile(os.path.join(path, "action.i32")):
        return V1Session(path)
    raise ValueError(f"{path}: not a v1 or v2 session")


def find_sessions(root):
    """root itself if it is a session, else the session dirs directly inside it (sorted)."""
    if is_session(root):
        return [os.path.abspath(root)]
    out = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if is_session(p):
                out.append(os.path.abspath(p))
    return out


def hindsight_goals(sess, first, horizon=HINDSIGHT_TICKS):
    """Per sim step: index of the next MineMind item the player obtains strictly after the step's first
    tick, within `horizon` ticks (9.5). -1 = none (zero goal). Only craft/smelt/pickup (and mine drops)
    count; chest, withdraw and give do not."""
    names = items()
    pts = []
    for e in sess.events():
        if e["type"] not in OBTAIN_TYPES and e["type"] != "mine":
            continue
        for g in e["got"]:
            if g["item"] in names:
                pts.append((e["tick"], names.index(g["item"])))
    goal = np.full(len(first), -1, np.int64)
    if not pts or len(first) == 0:
        return goal
    pts.sort()
    ticks = np.array([p[0] for p in pts], np.int64)
    idx = np.array([p[1] for p in pts], np.int64)
    j = np.searchsorted(ticks, first, side="right")
    jj = np.minimum(j, len(ticks) - 1)
    ok = (j < len(ticks)) & (ticks[jj] - first <= horizon)
    goal[ok] = idx[jj][ok]
    return goal


def decode_ext(ext):
    """9.4 ext_action code -> dict: key bits, hotbar (0 = none, 1..9), yaw/pitch camera bins (5 = none)."""
    keys = ["forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use", "drop",
            "inventory", "swap_offhand", "pick"]
    return {"keys": [k for i, k in enumerate(keys) if ext >> i & 1],
            "hotbar": (ext >> 13 & 15), "yaw_bin": (ext >> 17) & 31, "pitch_bin": (ext >> 22) & 31}
