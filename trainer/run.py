"""MineMind trainer: `python -m trainer.run`.

Single training loop (main thread). Commands from dashboard clients are polled between env steps
and between PPO minibatches; config changes are queued to rollout boundaries; the socket threads
never touch training state.
"""
import argparse
import base64
import copy
import json
import math
import os
import platform
import signal
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime

import numpy as np
import psutil
import torch

from . import checkpoint as ckpt
from . import envs as E
from .curriculum import EP_KIT, EP_RESTART, EP_SCRATCH, GOAL_ORDER, GoalCurriculum
from .envs import (ACTIONS, BLOCKS, HAS_GOALS, HAS_REGION, HAS_TASKS, ITEMS, MAX_AGENTS, NUM_ACTIONS,
                   NUM_BLOCKS, NUM_ITEM_CLASSES, NUM_ITEMS, NUM_MSGS, NUM_RECIPE_CLASSES, OBS, RES, V7,
                   VEC_DIM, VEC_NAMES, SimAdapter, obs_signature, per_env_bytes, primary_of)
from .tasks import TASK_NAMES, WorldTasks
from .instrument import Instrument
from .model import MineMindPolicy
from .eval import build_model, run_worlds
from .goexplore import Archive, depth_bucket
from .ppo import (RecurrentPPO, ReturnNormalizer, Rollout, RunningStd, act, act_by_slot,
                  load_optimizer_state, policy_act)
from .telemetry import TelemetryServer
from .demos import DemoData, DemoSampler
from .warmstart import load_warm_start

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS = os.path.join(ROOT, "runs")
SCHEMA = 1
MODES = ("fast", "live", "debug")
MODE_PRESETS = {"fast": {}, "live": {"snapshot_hz": 4.0, "nn_hz": 2.0},
                "debug": {"snapshot_hz": 10.0, "nn_hz": 5.0}}
DEFAULTS = {
    # num_envs = worlds W; every world holds `agents` agents (rollout rows = W * agents)
    "total_steps": 1_000_000, "num_envs": 64, "lr": 3e-4, "minibatch_size": 512,
    "rollout_len": 64, "ckpt_every": 100_000, "visualize": True, "snapshot_hz": 4.0,
    "nn_hz": 2.0, "workers": 4, "mode": "live",
    # seed None = random master seed (os.urandom) at the start of a fresh run; logged and checkpointed
    "seed": None,
    # v6 (PROTOCOL 6.8-6.10): teams of agents per world, reward split, chat-token entropy, image res,
    # and an optional v5 checkpoint to warm start from (msg_ent_coef None = the action entropy coef)
    "agents": 4, "team_share": 0.5, "msg_ent_coef": None, "res": RES, "warm_start": None,
    # algorithm constants (not settable at runtime)
    "gamma": 0.995, "gae_lambda": 0.95, "clip": 0.2, "ent_coef": 0.01, "ppo_epochs": 1,
    "bptt": 32,
    # v3: start kits (curriculum-driven) and the novelty bonus schedule. novelty decays linearly from
    # `novelty` to 0 over `novelty_end_frac` of the steps remaining at process start.
    "kits": True, "novelty": 0.02, "novelty_end_frac": 0.6,
    # v4 (PROTOCOL section 5): Go-Explore restarts (single-agent worlds only), RND curiosity,
    # HL-Gauss values, aux achievement loss. v6 defaults: all of the v4/v5 heads OFF (they measured
    # worse); the flags still work.
    "go_explore": True, "restart_frac": 0.25,
    "curiosity": "none", "rnd_coef": 0.1,
    # v5: SPR representation loss, AdamW-style decay, cosine LR with warmup, SimBa residual block
    "spr": False, "spr_coef": 1.0, "spr_k": 5, "spr_tau": 0.99,
    "weight_decay": 1e-4, "lr_cosine": True, "resblock": False,
    "value_loss": "mse", "aux_ach": False, "aux_coef": 0.1,
    # schedules: lr_anneal -> linear to 10% of lr over total_steps; ent_coef_end -> linear
    # from ent_coef to ent_coef_end over total_steps (None = constant)
    "lr_anneal": False, "ent_coef_end": None,
    # goal curriculum
    "goal_start": 2, "unlock_at": 0.5, "goal_min_episodes": 30,
    # backend: cpu (the v6 sim; gpusim is refused); model_size small | 100m (architecture, fixed by a
    # checkpoint); amp is unused on the cpu backend
    "backend": "cpu", "device": "cpu", "amp": False, "model_size": "small",
    # wall-clock checkpoint interval in seconds (0 = off); on top of ckpt_every (env steps)
    "ckpt_seconds": 0.0,
    # v7 (PROTOCOL 10): population of per-slot policies (0/1 = off, else == agents), evolution period in
    # agent steps, and the Gaussian mutation scale of the weights (relative to each tensor's std)
    "population": 0, "evolve_every": 250_000, "mutation": 0.01,
    # v7 (src/README_v7.md): observation version (fixed per process, see trainer/__init__.py) and the env
    # mix (tasklib, data/envmix/<name>.json) that assigns each world its task family
    "obs": OBS, "envmix": "default",
    # DAPG-style demo loss (trainer/demos.py): expert shard dirs (None = off), the masked multi-head CE weight
    # (linear from demo_coef to demo_coef_end over total_steps), sequences per PPO minibatch, successful episodes
    # only, and the idle filter (noop / sneak runs longer than demo_max_idle rows are removed; < 0 = off)
    "demos": None, "demo_coef": 0.5, "demo_coef_end": 0.05, "demo_batch": 4, "demo_success_only": True,
    "demo_max_idle": 8,
    # primary actions the demo CE is applied to: "all", "decisions" (craft/smelt/use/deposit/withdraw/give/drop:
    # what to do with items, not how to walk), or a comma list of primary names
    "demo_actions": "all",
}
LR_ANNEAL_END_FRAC = 0.1
EP_MAX_DECISIONS = 3000  # sim episode cap (max_steps)
GOAL_BIT = 31  # ach bit: goal achieved in this episode
SET_KEYS = {"total_steps", "num_envs", "lr", "minibatch_size", "rollout_len", "ckpt_every",
            "visualize", "snapshot_hz", "nn_hz", "workers", "mode"}
EP_WINDOW = 200
ACTION_HISTORY = 32
CHAT_LEN = 32          # last non-silent message tokens kept per world (snapshot `chat`)
POP_JITTER = (0.8, 1.0, 1.0, 1.25)   # PROTOCOL 10.1: initial lr / ent factor per slot (cycled)
POP_HP_FACTORS = (0.8, 1.25)         # PROTOCOL 10.2: child hyperparameter factors
POP_LR_RANGE = (1e-5, 3e-3)
POP_ENT_RANGE = (1e-4, 0.05)
POP_LINEAGE_KEEP = 200
DEPOSIT, WITHDRAW, GIVE = ACTIONS.index("deposit"), ACTIONS.index("withdraw"), ACTIONS.index("give")
HEIGHT_IDX = E.HEIGHT_IDX   # vec index of the player height (y - SEA_LEVEL) / 32, by name
BUILD_IDX = VEC_NAMES.index("build_progress") if V7 else None
HEALTH_IDX = VEC_NAMES.index("health") if V7 else None
ACT_COLS = 3 if V7 else 0    # v7 action columns kept in the rollout: primary, recipe, item
RND_REWARD_CLIP = 5.0
MAX_HISTORY_ROWS = 20000
MEM_WARN_FRACTION = 0.7
# Commands that must not run while an update is in flight (they touch params / envs).
DEFER_DURING_UPDATE = {"save_checkpoint", "load_checkpoint", "reset", "eval_checkpoint", "debug_step"}


def _now():
    return datetime.now().strftime("%H:%M:%S")


def _np(x):
    """Host numpy view of a device tensor (identity for numpy / lists)."""
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _random_seed():
    """Master seed from the OS entropy pool (62 bits, positive int64)."""
    return int.from_bytes(os.urandom(8), "little") & ((1 << 62) - 1)


class Member:
    """One population policy (team slot). lr_base / ent_base are the hyperparameters before the
    schedules; the effective values are set at every update (PROTOCOL 10.1/10.2)."""

    def __init__(self, mid, model, ppo, lr_base, ent_base, gen=0, parents=None):
        self.id, self.model, self.ppo = mid, model, ppo
        self.lr_base, self.ent_base = float(lr_base), float(ent_base)
        self.gen, self.parents = int(gen), (None if parents is None else [int(x) for x in parents])


class _RolloutSlice:
    """View of one population slot's columns of a Rollout ([T, N] -> [T, W] with rows p, p+K, ...).
    Duck-typed for RecurrentPPO.update, which only reads these fields."""

    def __init__(self, r, sl):
        self.T = r.T
        for name in ("img", "vec", "hid", "act", "msg", "logp", "val", "val_int", "rnd", "rew", "done",
                     "mask", "prev"):
            setattr(self, name, getattr(r, name)[:, sl])
        self.N = self.act.shape[1]


def _load_state(model, ppo, sd):
    """Load a policy state dict (strict=False for the v4+ heads). Returns the top-level modules missing from
    it (initialized fresh). Unexpected parameters are an error. SPR target = EMA copy of the loaded model."""
    res = model.load_state_dict(sd, strict=False)
    if res.unexpected_keys:
        raise ValueError(f"checkpoint has unexpected parameters: {res.unexpected_keys[:5]}")
    if ppo is not None and ppo.spr:
        ppo.reset_target()
    return sorted({k.split(".")[0] for k in res.missing_keys})


def _breed(sa, sb, rng, sigma):
    """Child state dict (PROTOCOL 10.2): per-tensor uniform crossover of two parents (the GRU's tensors
    move as one unit, so its gates stay consistent), then Gaussian mutation of every floating tensor:
    w += N(0, sigma * std(w))."""
    gen = torch.Generator().manual_seed(int(rng.integers(2**62)))
    pick, child = {}, {}
    for k in sa:
        g = "gru" if k.startswith("gru.") else k
        if g not in pick:
            pick[g] = sa if rng.random() < 0.5 else sb
        t = pick[g][k].detach().clone()
        if t.is_floating_point() and sigma > 0 and t.numel() > 1:
            sd_ = float(t.float().std())
            if sd_ > 0:
                t = t + torch.randn(t.shape, generator=gen, dtype=t.dtype) * (sigma * sd_)
        child[k] = t
    return child


def _new_window():
    """Counters over the rollout being collected (per-1k-agent-step stats are derived at each update)."""
    return {"agent_steps": 0, "msg_nonsilent": 0, "deposit": 0, "withdraw": 0, "give": 0,
            "msg_counts": np.zeros(NUM_MSGS, np.int64)}


class Trainer:
    def __init__(self, args):
        self.args = args
        self.cfg = dict(DEFAULTS)
        self.counters = {"env_steps": 0, "updates": 0, "episodes": 0}
        self.resume_payload = None
        if args.backend == "gpusim":
            raise ValueError("--backend gpusim is not supported for v6 (teams, chat, chests and 128x64x128 "
                             "worlds exist only in the CPU sim). v6 training is CPU only.")
        if args.resume and args.warm_start:
            raise ValueError("--resume and --warm-start are exclusive: a warm start begins a new run")
        if args.resume:
            self.resume_payload = ckpt.load_for_training(args.resume)
            stored = dict(self.resume_payload.get("config", {}))
            stored.pop("bptt", None)   # v3 default (32) applies on resume unless --bptt is given
            self.cfg.update(stored)
            self.run_id = args.run_id or self.resume_payload["run_id"]   # --run-id forks the run
        else:
            self.run_id = args.run_id or (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        # explicit CLI flags override the (possibly restored) config
        overrides = {"novelty": args.novelty, "novelty_end_frac": args.novelty_end_frac,
                     "restart_frac": args.restart_frac, "rnd_coef": args.rnd_coef,
                     "value_loss": args.value_loss, "aux_coef": args.aux_coef,
                     "go_explore": args.go_explore, "aux_ach": args.aux_ach,
                     "spr": args.spr, "spr_coef": args.spr_coef, "spr_k": args.spr_k,
                     "spr_tau": args.spr_tau, "weight_decay": args.weight_decay,
                     "lr_cosine": args.lr_cosine, "resblock": args.resblock,
                     "total_steps": args.total_steps, "num_envs": args.envs, "lr": args.lr,
                     "minibatch_size": args.minibatch, "rollout_len": args.rollout,
                     "ckpt_every": args.ckpt_every, "mode": args.mode, "seed": args.seed,
                     "workers": args.workers, "snapshot_hz": args.snapshot_hz, "nn_hz": args.nn_hz,
                     "ent_coef_end": args.ent_coef_end, "goal_start": args.goal_start,
                     "unlock_at": args.unlock_at, "goal_min_episodes": args.goal_min_episodes,
                     "ppo_epochs": args.epochs, "bptt": args.bptt,
                     "ckpt_seconds": args.ckpt_seconds, "agents": args.agents,
                     "team_share": args.team_share, "msg_ent_coef": args.msg_ent_coef,
                     "res": args.res, "population": args.population, "evolve_every": args.evolve_every,
                     "mutation": args.mutation, "envmix": args.envmix,
                     "demo_coef": args.demo_coef, "demo_coef_end": args.demo_coef_end,
                     "demo_batch": args.demo_batch, "demo_success_only": args.demo_success_only,
                     "demo_max_idle": args.demo_max_idle, "demo_actions": args.demo_actions}
        for k, v in overrides.items():
            if v is not None:
                self.cfg[k] = v
        if args.demos and args.no_demos:
            raise ValueError("--demos and --no-demos are exclusive")
        if args.demos:
            self.cfg["demos"] = [os.path.abspath(d) for d in args.demos]
        if args.no_demos:      # a resumed checkpoint keeps its stored demo config unless this switches it off
            self.cfg["demos"] = None
        if args.obs is not None and args.obs != OBS:
            raise ValueError(f"--obs {args.obs} but this process was started with obs {OBS} (the obs version "
                             f"is read before the sim is imported: pass --obs {args.obs} from the start)")
        self.cfg["obs"] = OBS          # the running layout (a checkpoint of another obs is refused on load)
        if args.visualize is not None:
            self.cfg["visualize"] = args.visualize
        if args.lr_anneal:
            self.cfg["lr_anneal"] = True
        if args.no_kits:
            self.cfg["kits"] = False
        if args.curiosity is not None:
            self.cfg["curiosity"] = args.curiosity
        elif args.rnd is not None:
            self.cfg["curiosity"] = "rnd" if args.rnd else "none"
        if not V7 and not HAS_GOALS:
            raise RuntimeError("craftsim has no goal API (set_goal_weights/goal): the v6 trainer needs the "
                               "v2 goal-conditioned sim; rebuild with ./build.sh")
        if V7 and not HAS_TASKS:
            raise RuntimeError("craftsim has no task API (set_task_weights): the v7 trainer needs the v7 sim")
        if self.resume_payload is not None and int(self.cfg["res"]) != int(
                self.resume_payload.get("obs", {}).get("res", self.cfg["res"])):
            raise ValueError("--res cannot change on resume: the checkpoint's image size is part of the network")
        if self.resume_payload is None:
            if args.seed is None:
                self.cfg["seed"] = _random_seed()
                seed_source = "random (os.urandom)"
            else:
                seed_source = "from --seed"
        else:
            seed_source = "from checkpoint"
        self.seed_source = seed_source
        K = int(self.cfg["agents"])
        if not (1 <= K <= MAX_AGENTS):
            raise ValueError(f"--agents must be in [1, {MAX_AGENTS}] (got {K})")
        P = int(self.cfg["population"])
        self.pop_on = P >= 2
        self.P = P if self.pop_on else 0
        if self.pop_on:
            if P != K:
                raise ValueError(f"--population must equal --agents (policy p controls team slot p): "
                                 f"got population={P}, agents={K}")
            if self.cfg["curiosity"] == "spr":
                raise ValueError("--population does not support --curiosity spr (its latent error is "
                                 "computed for one policy); use rnd or none")
        if args.warm_start is not None:
            if not os.path.isfile(args.warm_start):
                raise ValueError(f"--warm-start: no such checkpoint: {args.warm_start}")
            self.cfg["warm_start"] = os.path.abspath(args.warm_start)
        self.cfg["backend"] = "cpu"
        self.cfg["device"] = "cpu"
        self.dev = torch.device("cpu")
        self.cfg["amp"] = False         # AMP is a GPU-backend option; the v6 sim is CPU only
        self.amp = False
        self._gpu_notes = []
        if (K > 1 or V7) and self.cfg["go_explore"]:
            self.cfg["go_explore"] = False
            self._gpu_notes.append(("go-explore disabled: v7 (the archive is keyed by the v6 goal)" if V7 else
                                    f"go-explore disabled: agents={K} per world (snapshots are per world and "
                                    "the archive keys one agent per world; PROTOCOL 6.7)"))
        self.runs_dir = os.path.abspath(args.runs_dir)
        self.run_dir = os.path.join(self.runs_dir, self.run_id)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.history_path = os.path.join(self.run_dir, "history.jsonl")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.tasks = self._new_tasks(int(self.cfg["num_envs"]))
        self.logs = deque(maxlen=500)
        self.server = None      # set in main() when the telemetry server is up (log() reads it)
        self.demo, self.demo_data = None, None
        if self.cfg.get("demos"):
            self.demo, self.demo_data = self._new_demo()
        self._new_policies()
        self.state = "idle"
        self.quit = False
        self.stop_requested = False
        self._signal_stop = False
        self.pending = {}
        self.deferred = deque()
        self.in_update = False
        self.last_ckpt_steps = 0
        self._last_ckpt_wall = time.time()
        self.latest_ckpt = None
        self.ep_hist = deque(maxlen=EP_WINDOW)  # (reward, length, achievement mask, goal, kind)
        self.kind_window = deque(maxlen=EP_WINDOW)  # episode kind (EP_*) of recent episodes
        self.nov_sum = 0.0    # novelty reward summed over agent steps since the last update
        self.nov_n = 0        # agent steps since the last update
        self.win = _new_window()
        self.cur = self._new_curriculum()
        self.archive = self._new_archive()
        self.restart_rng = np.random.default_rng(self.cfg["seed"] + 7)
        self.ret_norm = ReturnNormalizer(self.cfg["gamma"])
        self.rnd_norm = RunningStd()
        self.warm_report = None
        self.metrics_hist = deque(maxlen=MAX_HISTORY_ROWS)
        self.last_stats = {}
        self.elapsed = 0.0
        self.ema_sps = None
        self.sps_samples = 0
        self._m_t = time.time()
        self._m_steps = 0
        self._last_metrics = 0.0
        self._last_snap = 0.0
        self._last_nn = 0.0
        self._last_console = 0.0
        self._history_warned = False
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)
        self._eval_thread = None
        self._reset_count = 0
        self._env_epoch = 0
        self.sim = None
        self.debug = None
        self.debug_key = None
        self.sel = 0
        self.server = None

        pop_meta = None
        if self.resume_payload is not None:
            pop_meta = self._restore_policies(self.resume_payload)
            self.counters = dict(self.resume_payload["counters"])
            self.last_ckpt_steps = self.counters["env_steps"]
            ckpt.restore_rng(self.resume_payload.get("rng"))
            self._restore_curriculum(self.resume_payload)
            self.latest_ckpt = None
            self.log("info", f"resumed run {self.run_id} from {args.resume} "
                             f"(env_steps={self.counters['env_steps']}, updates={self.counters['updates']})")
        else:
            torch.manual_seed(self.cfg["seed"])
            np.random.seed(self.cfg["seed"] % (2**32))
            if self.cfg.get("warm_start"):
                self._apply_warm_start(self.cfg["warm_start"])
        self._init_pop_schedule(pop_meta)
        # novelty schedule origin: the step this process started at (0 for a fresh run)
        self.novelty_t0 = self.counters["env_steps"]
        self._build_envs()
        self._write_run_meta()
        self.log("info", f"master seed {self.cfg['seed']} ({self.seed_source})")
        self.log("info", f"backend=cpu model={self.cfg['model_size']} res={self.cfg['res']} "
                         f"worlds={self.cfg['num_envs']} agents={self.cfg['agents']} "
                         f"({sum(p.numel() for p in self.model.parameters()):,} params)")
        if self.pop_on:
            self.log("info", f"population {self.P} (evolve every {self.cfg['evolve_every']:,} agent steps, "
                             f"mutation {self.cfg['mutation']}): lr/ent start factors "
                             + "/".join(str(POP_JITTER[i % len(POP_JITTER)]) for i in range(self.P)))
        for note in self._gpu_notes:
            self.log("warn", note)
        if self.warm_report is not None:
            self.log("info", f"warm start from {self.cfg['warm_start']}: {len(self.warm_report['copied'])} "
                             f"tensors copied, remapped {', '.join(self.warm_report['remapped'])}, "
                             f"fresh {', '.join(self.warm_report['fresh'])}")
            if "vec_columns" in self.warm_report:
                vc, wr = self.warm_report["vec_columns"], self.warm_report
                self.log("info", f"warm start v6->v7: vec columns mapped by name {vc['mapped']}/{vc['v7']} "
                                 f"(v6 {vc['v6']}), block rows {wr['block_rows']}, primary rows "
                                 f"{wr['primary_rows']}/{wr['primary_total']}")
        if V7:
            fc = self.tasks.family_counts()
            self.log("info", f"obs v7: envmix {self.cfg['envmix']}; world families "
                             + ", ".join(f"{f} {n}" for f, n in sorted(fc.items(), key=lambda kv: -kv[1]))
                             + f"; tasks unlocked {self.tasks.status()['unlocked']}/{self.tasks.status()['total']}")
        self.log("info", f"{'task kits' if V7 else 'start kits'} {'on' if self.cfg['kits'] else 'off'}; novelty "
                         f"{self.cfg['novelty']} -> 0 over {self._novelty_span():,} steps "
                         f"from env_steps={self.novelty_t0:,}")

    # ----- setup helpers -----
    def _new_demo(self):
        """DAPG-style demo loss (trainer/demos.py): load the expert shards with the idle filter, log what it removed.
        Returns (sampler, data); the sampler's coef follows _demo_coef() each update."""
        if not V7:
            raise ValueError("--demos needs the v7 observation (the demo shards are v7 rows)")
        data = DemoData(self.cfg["demos"], success_only=bool(self.cfg["demo_success_only"]),
                        max_idle=int(self.cfg["demo_max_idle"]))
        self.log("info", f"demos: {len(data.paths)} shards from {', '.join(self.cfg['demos'])}; "
                         f"coef {self.cfg['demo_coef']} -> {self.cfg['demo_coef_end']}, "
                         f"batch {self.cfg['demo_batch']} x 64 rows per minibatch")
        for line in data.describe():
            self.log("info", line.strip())
        sampler = DemoSampler(data, int(self.cfg["demo_batch"]), seed=int(self.cfg["seed"]),
                              coef=float(self.cfg["demo_coef"]), actions=self.cfg.get("demo_actions", "all"))
        self.log("info", f"demo CE on primary actions: {sampler.describe_actions()}")
        return sampler, data

    def _demo_coef(self):
        """Demo loss weight: linear from demo_coef to demo_coef_end over the steps left at process start (from
        novelty_t0 to total_steps), so a resumed run starts again at demo_coef."""
        t0 = self.novelty_t0
        frac = min(1.0, max(0.0, (self.counters["env_steps"] - t0) / max(1, int(self.cfg["total_steps"]) - t0)))
        c0, c1 = float(self.cfg["demo_coef"]), float(self.cfg["demo_coef_end"])
        return c0 + (c1 - c0) * frac

    # ----- setup helpers -----
    def _new_model(self):
        return MineMindPolicy(res=int(self.cfg["res"]), vec_dim=VEC_DIM, num_actions=NUM_ACTIONS,
                              num_blocks=NUM_BLOCKS, num_msgs=NUM_MSGS, size=self.cfg["model_size"],
                              resblock=bool(self.cfg["resblock"]),
                              num_recipe_classes=NUM_RECIPE_CLASSES, num_item_classes=NUM_ITEM_CLASSES
                              ).to(self.dev)

    def _new_tasks(self, W):
        """v7: the world task scheduler (families, tasklib curriculum); None for v6."""
        if not V7:
            return None
        return WorldTasks(W, seed=int(self.cfg["seed"]), envmix=str(self.cfg.get("envmix") or "default"))

    def _new_policies(self):
        """Fresh policies (PROTOCOL 10.1). Single policy: self.model / self.ppo / self.inst. Population:
        P members that all start as copies of one init, with lr / ent jitter by slot; self.model etc. point
        at the best member (slot 0 until the first evolution)."""
        self.model = self._new_model()
        self.ppo = self._make_ppo()
        self.members = []
        self.evolutions, self.lineage, self.next_id, self.best_slot = 0, [], self.P, 0
        if not self.pop_on:
            self.inst = Instrument(self.model)
            self.insts = [self.inst]
            return
        base_lr, base_ent = float(self.cfg["lr"]), float(self.cfg["ent_coef"])
        for p in range(self.P):
            if p == 0:
                model, ppo = self.model, self.ppo
            else:
                model = self._new_model()
                model.load_state_dict(self.model.state_dict())
                ppo = self._make_ppo(model)
            f = POP_JITTER[p % len(POP_JITTER)]
            self.members.append(Member(p, model, ppo, base_lr * f, base_ent * f))
        self.insts = [Instrument(mb.model) for mb in self.members]
        self._set_best(0)

    def _set_best(self, p):
        self.best_slot = int(p)
        mb = self.members[p]
        self.model, self.ppo, self.inst = mb.model, mb.ppo, self.insts[p]

    def _init_pop_schedule(self, meta):
        """Population counters for a run that starts (fresh, resumed or reset) at the current counters."""
        self.pop_win = [{"returns": [], "succ": 0, "rsum": 0.0, "rn": 0} for _ in range(self.P)] \
            if self.pop_on else []
        if self.pop_on:
            meta = meta or {}
            self.next_evolve = int(meta.get("next_evolve",
                                            self.counters["env_steps"] + int(self.cfg["evolve_every"])))

    def _restore_policies(self, payload):
        """Load a checkpoint into the policies. A population checkpoint restores every member, its
        optimizer and the lineage. Anything else loads its `model` (the single policy, or the best one of a
        population checkpoint) and, with population on, replicates it to every slot. Returns the stored
        population metadata (None when there is none)."""
        if self.pop_on and "models" in payload:
            models = payload["models"]
            if len(models) != self.P:
                raise ValueError(f"checkpoint has {len(models)} policies; --population is {self.P}")
            for mb, sd in zip(self.members, models):
                _load_state(mb.model, mb.ppo, sd)
            for mb, osd in zip(self.members, payload.get("optimizers") or []):
                load_optimizer_state(mb.ppo.opt, osd)
            meta = payload.get("population") or {}
            for rec in meta.get("members", []):
                mb = self.members[int(rec["slot"])]
                mb.id, mb.gen = int(rec["id"]), int(rec["gen"])
                mb.parents = rec.get("parents")
                mb.lr_base, mb.ent_base = float(rec["lr_base"]), float(rec["ent_base"])
            self.evolutions = int(meta.get("evolutions", 0))
            self.next_id = int(meta.get("next_id", self.P))
            self.lineage = list(meta.get("lineage", []))[-POP_LINEAGE_KEEP:]
            self._set_best(int(meta.get("best", 0)))
            return meta
        self._load_model(payload)
        load_optimizer_state(self.ppo.opt, payload["optimizer"])
        self.ppo.set_lr(self.cfg["lr"])
        if self.pop_on:
            self._sync_members(payload["optimizer"])
        return None

    def _sync_members(self, opt_sd=None):
        """Copy the best policy's weights into every other member (and its optimizer state if given)."""
        src = self.model.state_dict()
        for mb in self.members:
            if mb.model is not self.model:
                _load_state(mb.model, mb.ppo, src)
            if opt_sd is not None:
                load_optimizer_state(mb.ppo.opt, opt_sd)

    def _population_meta(self):
        return {"size": self.P, "best": self.best_slot, "evolutions": self.evolutions,
                "next_evolve": self.next_evolve, "next_id": self.next_id,
                "members": [{"slot": p, "id": mb.id, "gen": mb.gen, "parents": mb.parents,
                             "lr_base": mb.lr_base, "ent_base": mb.ent_base}
                            for p, mb in enumerate(self.members)],
                "lineage": self.lineage[-POP_LINEAGE_KEEP:]}

    def _fitness(self, p):
        """Mean per-agent episode return of slot p over the evolve window; if no episode finished in the
        window, the mean per-step reward (every slot plays the same worlds, so the basis is shared)."""
        w = self.pop_win[p]
        if w["returns"]:
            return float(np.mean(w["returns"]))
        return float(w["rsum"] / w["rn"]) if w["rn"] else None

    def _success(self, p):
        w = self.pop_win[p]
        return float(w["succ"] / len(w["returns"])) if w["returns"] else None

    def _population_rows(self):
        rows = []
        for p, mb in enumerate(self.members):
            rows.append({"slot": p, "id": mb.id, "generation": mb.gen, "parents": mb.parents,
                         "fitness": self._fitness(p),
                         "lr": float(mb.ppo.opt.param_groups[0]["lr"]), "ent_coef": float(mb.ppo.ent_coef),
                         "success": self._success(p)})
        return rows

    def _forward_on(self, img, vec, h, sample=True, rnd=False):
        """(action, msg, logp, v_ext, v_int, h_new, rnd_err) for the rows of img/vec/h: the shared policy,
        or with population on each slot's policy on its own rows (PROTOCOL 10.1)."""
        vm = self.cfg["value_loss"]
        if self.pop_on:
            return act_by_slot([mb.model for mb in self.members], img, vec, h, sample=sample,
                               value_mode=vm, rnd=rnd)
        return act(self.model, img, vec, h, sample=sample, value_mode=vm, rnd=rnd)

    def _new_curriculum(self):
        if V7:          # v7 curriculum: WorldTasks (tasklib); the v6 goal curriculum is not used
            return None
        return GoalCurriculum(GOAL_ORDER, start_unlocked=self.cfg["goal_start"],
                              unlock_at=self.cfg["unlock_at"],
                              min_episodes=self.cfg["goal_min_episodes"])

    def _restore_curriculum(self, payload):
        if V7:
            if "tasks" in payload and self.tasks is not None:
                self.tasks.load_state_dict(payload["tasks"])
        elif "curriculum" in payload:
            self.cur.load_state_dict(payload["curriculum"])
        if "reward_norm" in payload:
            self.ret_norm.load_state_dict(payload["reward_norm"])
        if "rnd_norm" in payload:
            self.rnd_norm.load_state_dict(payload["rnd_norm"])

    def _write_run_meta(self):
        meta = {"run_id": self.run_id, "created": time.time(), "python": sys.version.split()[0],
                "torch": torch.__version__, "platform": platform.platform(), "schema": SCHEMA}
        path = os.path.join(self.run_dir, "run.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(meta, f, indent=2)

    def _build_envs(self):
        """Create the training sim + rollout buffers (only at boundaries / init). num_envs = worlds W,
        agents = K per world; every per-agent buffer has N = W * K rows.
        v7: the task of every world's first episode and the next one are set before / after the first reset
        (tasks.py), and the kits are applied. v6: goal weights and start kits as before."""
        W = int(self.cfg["num_envs"])
        K = int(self.cfg["agents"])
        N = W * K
        T = int(self.cfg["rollout_len"])
        res = int(self.cfg["res"])
        if self.sim is not None:
            self.sim.close()
        # debug mode: fixed seed so the run is reproducible; otherwise vary per rebuild
        seed = self.cfg["seed"] if self.cfg["mode"] == "debug" else self.cfg["seed"] + self._env_epoch
        self._env_epoch += 1
        self.sim = SimAdapter(W, seed=seed, threads=self.cfg["workers"], res=res, agents=K)
        if V7:
            self.tasks.apply_shaping(self.sim)          # knowledge shaping lists, every task
            self._apply_sim_config()
            self.tasks.plan_start(self.sim, W)          # running task of every world, the next one queued
            self.img, self.vec = self.sim.reset()
            self.vec = self.tasks.after_reset(self.sim, self.vec, K)
            self.ep_goal = np.full(N, -1, np.int64)
            self.ep_kind = np.array([EP_KIT if self.tasks.cur_kit[i // K] else EP_SCRATCH for i in range(N)],
                                    np.int64)
        else:
            self.cur.arm(self.counters["env_steps"], EP_MAX_DECISIONS * N)   # decisions per env * agent rows
            self._apply_sim_config()   # goal weights, start kits, team share and novelty before the first reset
            self.img, self.vec = self.sim.reset()
            self.ep_goal = np.array([self.sim.goal(i) for i in range(N)], np.int64)
            self.ep_kind = np.array([EP_KIT if self.sim.kit(i) else EP_SCRATCH for i in range(N)], np.int64)
        self.start_mask = np.zeros(N, np.int64)     # achievement mask an episode started with
        self.prev_env = np.zeros(N, np.int64)       # mask the next step's events are measured against
        self.ret_norm.reset_envs(N)
        self.h = self.model.zero_hidden(N)
        self.roll = Rollout(T, N, res, VEC_DIM, self.model.gru_hidden, act_cols=ACT_COLS)
        self.roll_t = 0
        self.ep_rew = np.zeros(N, np.float64)
        self.ep_len = np.zeros(N, np.int64)
        self.ep_index = np.zeros(N, np.int64)
        self.env_steps_n = np.zeros(N, np.int64)
        self.last_rew = np.zeros(N, np.float64)
        self.last_act = np.full(N, -1, np.int64)
        self.last_msg = np.zeros(N, np.int64)
        self.act_hist = np.full((N, ACTION_HISTORY), -1, np.int64)
        self.act_ptr = np.zeros(N, np.int64)
        self.world_chat = [deque(maxlen=CHAT_LEN) for _ in range(W)]   # last non-silent tokens per world
        self.win = _new_window()
        self.sel = min(self.sel, N - 1)
        self.debug_key = None
        if self.debug is not None:
            self.debug.close()
            self.debug = None

    def _mem_check(self, worlds, T):
        """Warning string when `worlds` worlds of the current agent count and rollout length would not
        fit (None otherwise)."""
        est = worlds * per_env_bytes(T, self.model.gru_hidden, agents=int(self.cfg["agents"]),
                                     res=int(self.cfg["res"]))
        avail = psutil.virtual_memory().available
        if est > MEM_WARN_FRACTION * avail:
            return (f"estimated {est / 2**30:.2f} GiB for {worlds} worlds exceeds "
                    f"{int(MEM_WARN_FRACTION * 100)}% of available {avail / 2**30:.2f} GiB")
        return None

    # ----- v4 helpers -----
    def _make_ppo(self, model=None):
        ppo = self._make_ppo_base(model)
        ppo.demo = self.demo
        return ppo

    def _make_ppo_base(self, model=None):
        return RecurrentPPO(self.model if model is None else model, lr=self.cfg["lr"], gamma=self.cfg["gamma"],
                            lam=self.cfg["gae_lambda"], clip=self.cfg["clip"],
                            ent_coef=self.cfg["ent_coef"], epochs=self.cfg["ppo_epochs"],
                            bptt=self.cfg["bptt"], minibatch_size=self.cfg["minibatch_size"],
                            value_mode=self.cfg["value_loss"], rnd=self._int_on(),
                            curiosity=self.cfg["curiosity"], spr=bool(self.cfg["spr"]),
                            spr_coef=float(self.cfg["spr_coef"]), spr_k=int(self.cfg["spr_k"]),
                            spr_tau=float(self.cfg["spr_tau"]), weight_decay=float(self.cfg["weight_decay"]),
                            aux=bool(self.cfg["aux_ach"]), aux_coef=float(self.cfg["aux_coef"]),
                            amp=self.amp, msg_ent_coef=self._msg_ent_base())

    def _msg_ent_base(self):
        """Chat-token entropy coefficient before the schedule (default: the action entropy coef)."""
        v = self.cfg.get("msg_ent_coef")
        return float(self.cfg["ent_coef"] if v is None else v)

    def _new_archive(self):
        return Archive(max_cells=1000, seed=int(self.cfg["seed"]) + 11)

    def _load_model(self, payload):
        """Load a checkpoint's policy. Modules missing from it (v4 heads on a v2/v3 checkpoint) keep
        their fresh initialization; anything else that differs is an error."""
        ckpt_size = payload.get("config", {}).get("model_size", "small")
        if ckpt_size != self.cfg["model_size"]:
            raise ValueError(f"checkpoint model_size={ckpt_size} differs from the running model_size="
                             f"{self.cfg['model_size']}: start the run with --model-size {ckpt_size}")
        mods = _load_state(self.model, self.ppo, payload["model"])
        if mods:
            self.log("info", f"checkpoint has no {', '.join(mods)}: initialized fresh")

    def _apply_warm_start(self, path):
        """v5 -> v6 warm start (PROTOCOL 6.8/6.10) into the fresh model. Also carries over the reward
        normalizer, so the copied value heads keep their return scale."""
        src, report = load_warm_start(path, self.model)
        if "reward_norm" in src:
            self.ret_norm.load_state_dict(src["reward_norm"])
        self.warm_report = report
        if self.ppo.spr:
            self.ppo.reset_target()
        if self.pop_on:          # members were built before the warm start: copy the warmed policy to all
            self._sync_members()

    def _sched(self, start):
        """Linear decay from `start` to 0 over novelty_end_frac of the steps left at process start."""
        span = self._novelty_span()
        if start <= 0 or span <= 0:
            return 0.0
        frac = (self.counters["env_steps"] - self.novelty_t0) / span
        return start * max(0.0, 1.0 - frac)

    def _rnd_coef(self):
        return self._sched(float(self.cfg["rnd_coef"]))

    def _maybe_restart(self, i):
        """Go-Explore: restart the world of agent row i (which just finished an episode) from an archive
        cell with prob restart_frac. Single-agent worlds only (go-explore is off when agents > 1).
        Returns True when the world was restarted."""
        if not (self.cfg["go_explore"] and E.HAS_STATE and len(self.archive)):
            return False
        if self.restart_rng.random() >= float(self.cfg["restart_frac"]):
            return False
        pick = self.archive.sample(GOAL_ORDER[:self.cur.unlocked])
        if pick is None:
            return False
        blob, goal, mask = pick
        out = self.sim.load_state(i, blob, goal)     # K == 1: world index == agent row
        if out is None:
            return False
        img_i, vec_i = out
        self.img = np.array(self.img, copy=True)    # sim-returned arrays may be read-only views
        self.vec = np.array(self.vec, copy=True)
        self.img[i] = img_i[0]
        self.vec[i] = vec_i[0]
        self.ep_goal[i] = goal
        self.ep_kind[i] = EP_RESTART
        self.start_mask[i] = mask
        self.prev_env[i] = mask
        return True

    def _archive_update(self, ach, done):
        """Store snapshots of running envs whose (goal, mask, depth) cell is new or reached faster."""
        heights = self.vec[:, HEIGHT_IDX]
        for i in range(self.sim.n):
            if done[i]:
                continue
            key = Archive.key(self.ep_goal[i], ach[i], depth_bucket(heights[i]))
            steps = int(self.ep_len[i])
            if self.archive.wants(key, steps):
                self.archive.offer(key, self.sim.save_state(i), steps)

    # ----- v3: start kits and novelty schedule -----
    def _novelty_span(self):
        """Steps over which the novelty scale decays (end_frac of the steps left at process start)."""
        return int(float(self.cfg["novelty_end_frac"]) * max(1, int(self.cfg["total_steps"]) - self.novelty_t0))

    def _novelty_scale(self):
        if self._int_on():          # RND replaces the hand-crafted bonus (PROTOCOL section 5)
            return 0.0
        return self._sched(float(self.cfg["novelty"]))

    def _apply_sim_config(self):
        """Push curriculum-driven sim settings. Takes effect for episodes started from now on.
        v7: team share and novelty only (the tasks are set per world by tasks.py; no goal kits)."""
        if V7:
            self.sim.set_team_share(float(self.cfg["team_share"]))
            self.sim.set_novelty(self._novelty_scale())
            return
        self.sim.set_goal_weights(self.cur.weights())
        self.sim.set_team_share(float(self.cfg["team_share"]))
        if self.cfg["kits"]:
            self.sim.set_start_kits(self.cur.kit_table(), self.cur.kit_prob())
        else:
            zeros = np.zeros(NUM_ITEMS, np.float32)
            self.sim.set_start_kits(np.zeros((NUM_ITEMS, NUM_ITEMS), np.float32), zeros)
        self.sim.set_novelty(self._novelty_scale())

    # ----- logging / telemetry -----
    def log(self, level, msg):
        rec = {"t": time.time(), "level": level, "msg": msg}
        self.logs.append(rec)
        print(f"[{_now()}] {level.upper():5s} {msg}", flush=True)
        if self.server is not None:
            self.server.broadcast("log", rec)

    def _ack(self, cid, mid, ok, msg):
        if self.server is not None and cid is not None:
            self.server.send_to(cid, "ack", {"id": mid, "ok": bool(ok), "msg": msg})

    def _model_msg(self):
        return self.model.describe()

    def _public_config(self):
        return {k: self.cfg[k] for k in DEFAULTS}

    def _send_hello(self, cid):
        self.server.send_to(cid, "hello", {
            "schema": SCHEMA, "run_id": self.run_id, "state": self.state,
            "config": self._public_config(), "world_size": list(E.WORLD_SIZE),
            "actions": ACTIONS, "items": ITEMS, "blocks": BLOCKS,
        })
        self.server.send_to(cid, "model", self._model_msg())
        self._emit_metrics(time.time(), to=cid)

    # ----- metrics -----
    def _success_rates(self):
        """Item-obtained rate per item over the episode window (any goal)."""
        # from-scratch episodes only: kit items do not set achievement bits, restarts begin mid-tree
        eps = [e for e in self.ep_hist if V7 or e[4] == EP_SCRATCH]
        if not eps:
            return {}
        n = len(eps)
        out = {}
        for i, name in enumerate(ITEMS):
            c = sum(1 for e in eps if (e[2] >> i) & 1)
            out[name] = c / n
        return out

    def _int_on(self):
        return self.cfg["curiosity"] != "none"

    def _cosine_factor(self):
        """LR multiplier: linear warmup over the first 2% of the steps left at process start, then a
        cosine decay to 10% at total_steps."""
        t0 = self.novelty_t0
        total = max(1, int(self.cfg["total_steps"]) - t0)
        f = min(1.0, max(0.0, (self.counters["env_steps"] - t0) / total))
        warm = 0.02
        if f < warm:
            return max(f, 0.01) / warm
        p = (f - warm) / (1.0 - warm)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))

    def _kind_rate(self, kind):
        if not self.kind_window:
            return None
        return float(sum(1 for k in self.kind_window if k == kind) / len(self.kind_window))

    def _curriculum(self):
        """Goal curriculum view: `stage` (unlocked count), `task` (frontier goal), `text`. v7: the task
        scheduler's view (`task` = the last finished episode's task)."""
        if V7:
            t = self.tasks.telemetry()
            return {"stage": t["tasks_unlocked"], "task": t["curriculum_task"], "text": t["curriculum_stage"]}
        return {"stage": self.cur.unlocked, "task": self.cur.frontier, "text": self.cur.stage()}

    def _cur_metrics(self):
        """Per-goal (v6) or per-task (v7) success numbers for the metrics row."""
        if V7:
            return {"goal_success": self.tasks.telemetry()["task_ema"], "goal_success_restart": None,
                    "goal_success_kit": None}
        return {"goal_success": self.cur.goal_success(), "goal_success_restart": self.cur.goal_success_restart(),
                "goal_success_kit": self.cur.goal_success_kit()}

    def _task_of_row(self, i):
        return self.tasks.cur[int(i) // int(self.cfg["agents"])] if V7 else None

    def _family_of_row(self, i):
        return self.tasks.cur_family[int(i) // int(self.cfg["agents"])] if V7 else None

    def _metrics(self, now):
        steps = self.counters["env_steps"]
        dt = now - self._m_t
        dsteps = steps - self._m_steps
        if self.state in ("running", "stopping") and dt > 0:
            self.elapsed += dt
        if self.state in ("running", "stopping") and dsteps > 0 and dt >= 0.2:
            inst = dsteps / dt
            self.ema_sps = inst if self.ema_sps is None else 0.3 * inst + 0.7 * self.ema_sps
            self.sps_samples += 1
        if dt >= 0.2:
            self._m_t, self._m_steps = now, steps
        sps = self.ema_sps if (self.ema_sps is not None and self.state in ("running", "stopping")) else 0.0
        target = int(self.cfg["total_steps"])
        if self.sps_samples >= 5 and self.ema_sps and self.ema_sps > 0:
            eta = max(0.0, (target - steps) / self.ema_sps)
        else:
            eta = None
        eps = list(self.ep_hist)[-100:]
        avg_len = float(np.mean([e[1] for e in eps])) if eps else None
        avg_rew = float(np.mean([e[0] for e in eps])) if eps else None
        pending_n = self.pending.get("num_envs", self.cfg["num_envs"])
        mem_warn = self._mem_check(int(pending_n), int(self.cfg["rollout_len"]))
        cur = self._curriculum()
        cur_m = self._cur_metrics()
        st = self.last_stats or {}
        v7m = self._v7_metrics() if V7 else {}
        return {
            "t": now, "state": self.state, "mode": self.cfg["mode"],
            "env_steps": steps, "target_steps": target, "sps": sps,
            "agents": int(self.cfg["agents"]),
            "msg_entropy": st.get("msg_entropy"), "msg_rate": st.get("msg_rate"),
            "chest_deposits": st.get("chest_deposits"), "chest_withdraws": st.get("chest_withdraws"),
            "gives": st.get("gives"),
            "episodes": self.counters["episodes"], "avg_ep_len": avg_len, "avg_reward": avg_rew,
            "success_rate": self._success_rates(), "updates": self.counters["updates"],
            "policy_loss": st.get("policy_loss"), "value_loss": st.get("value_loss"),
            "entropy": st.get("entropy"), "approx_kl": st.get("approx_kl"),
            "explained_var": st.get("explained_var"), "step_reward": st.get("step_reward"),
            "cpu_percent": self._proc.cpu_percent(None),
            "rss_mb": self._proc.memory_info().rss / 2**20,
            "mem_warning": mem_warn, "elapsed_s": self.elapsed, "eta_s": eta,
            "latest_checkpoint": self.latest_ckpt, "curriculum_stage": cur["text"],
            "curriculum_task": cur["task"], "goal_success": cur_m["goal_success"],
            "lr": float(self.ppo.opt.param_groups[0]["lr"]), "ent_coef": float(self.ppo.ent_coef),
            "reward_scale": float(1.0 / math.sqrt(self.ret_norm.var + 1e-8)),
            "num_envs": int(self.sim.worlds) if self.sim else 0,
            "novelty_reward": st.get("novelty_reward"), "novelty_scale": self._novelty_scale(),
            "kit_rate": self._kind_rate(EP_KIT),
            "restart_rate": self._kind_rate(EP_RESTART),
            "rnd_reward": st.get("rnd_reward"),
            "rnd_coef": self._rnd_coef() if self._int_on() else 0.0,
            "rnd_loss": st.get("rnd_loss"), "aux_loss": st.get("aux_loss"),
            "aux_ach_acc": st.get("aux_ach_acc"), "spr_loss": st.get("spr_loss"),
            "goal_success_restart": cur_m["goal_success_restart"],
            "archive_size": len(self.archive), "archive_bytes": int(self.archive.bytes),
            "goal_success_kit": cur_m["goal_success_kit"],
            "population": self._population_rows() if self.pop_on else None,
            "generation": self.evolutions if self.pop_on else None,
            "config": self._public_config(),
            "obs": OBS, "msg_hist": st.get("msg_hist"),
            "recipe_entropy": st.get("recipe_entropy"), "item_entropy": st.get("item_entropy"),
            "build_progress": st.get("build_progress"), "health_mean": st.get("health_mean"),
            "demo_ce": st.get("demo_ce"), "demo_acc_primary": st.get("demo_acc_primary"),
            "demo_acc_recipe": st.get("demo_acc_recipe"), "demo_acc_item": st.get("demo_acc_item"),
            "demo_coef": st.get("demo_coef"), "demo_rows_kept": self._demo_rows("rows_kept"),
            "demo_rows_dropped": self._demo_rows("idle_rows_dropped"),
            **v7m,
        }

    def _demo_rows(self, key):
        """Demo rows kept / dropped by the idle filter (static, from the startup load); None without --demos."""
        if self.demo_data is None:
            return None
        return int(self.demo_data.totals()[key])

    def _v7_metrics(self):
        """v7 telemetry (additive): per-task / per-family success and return, family mix, curriculum."""
        t = self.tasks.telemetry()
        return {"family_mix": t["family_mix"], "family_worlds": t["family_worlds"],
                "family_episodes": t["family_episodes"], "family_success": t["family_success"],
                "family_return": t["family_return"], "task_success": t["task_success"],
                "task_episodes": t["task_episodes"], "task_ema": t["task_ema"],
                "tasks_unlocked": t["tasks_unlocked"], "tasks_mastered": t["tasks_mastered"],
                "family_switches": t["family_switches"]}

    def _emit_metrics(self, now, to=None):
        m = self._metrics(now)
        self._last_metrics = now
        self.metrics_hist.append(m)
        try:
            with open(self.history_path, "a") as f:
                f.write(json.dumps(m, separators=(",", ":")) + "\n")
        except OSError as e:
            if not self._history_warned:
                self._history_warned = True
                self.log("error", f"history write failed: {e}")
        if self.server is not None:
            if to is None:
                self.server.broadcast("metrics", m)
            else:
                self.server.send_to(to, "metrics", m)
        if now - self._last_console >= 10.0:
            self._last_console = now
            eta = "Calculating" if m["eta_s"] is None else f"{m['eta_s']:.0f}s"
            print(f"[{_now()}] {m['state']:8s} steps={m['env_steps']:,}/{m['target_steps']:,} "
                  f"sps={m['sps']:.0f} upd={m['updates']} ep={m['episodes']} "
                  + (f"demo_ce={m['demo_ce']:.3f} " if m.get("demo_ce") is not None else "") +
                  f"rew={m['avg_reward'] if m['avg_reward'] is None else round(m['avg_reward'], 3)} "
                  f"eta={eta} rss={m['rss_mb']:.0f}MB", flush=True)

    # ----- snapshots -----
    def _meta_for(self, i):
        hist = [int(x) for x in _np(self.act_hist[i])]
        ptr = int(self.act_ptr[i])
        ordered = []
        for k in range(ACTION_HISTORY):
            v = hist[(ptr + k) % ACTION_HISTORY]
            if v >= 0:
                ordered.append(v)
        return {
            "episode": int(self.ep_index[i]), "seed": None, "step": int(self.ep_len[i]),
            "last_action": int(self.last_act[i]), "last_reward": float(self.last_rew[i]),
            "ep_reward": float(self.ep_rew[i]),
            "task": self._task_of_row(i) if V7 else self._curriculum()["task"],
            "family": self._family_of_row(i),
            "hist": ordered, "obs_img": self.img[i],
        }

    def _snapshot_payload(self, adapter, idx, env_id, meta, chat, last_act, last_msg):
        """Snapshot of agent row `idx` of `adapter` (its world's blocks, chests, team and chat).
        env_id = global agent index; world_id = idx // K; agent_id = idx % K (PROTOCOL 6.11)."""
        K = int(adapter.agents)
        w, k = idx // K, idx % K
        st = adapter.state(idx)
        player = {
            "x": float(st.get("x", 0.0)), "y": float(st.get("y", 0.0)), "z": float(st.get("z", 0.0)),
            "vx": float(st.get("vx", 0.0)), "vy": float(st.get("vy", 0.0)), "vz": float(st.get("vz", 0.0)),
            "yaw": float(st.get("yaw", 0.0)), "pitch": float(st.get("pitch", 0.0)),
            "health": float(st.get("health", 0.0)), "on_ground": bool(st.get("on_ground", False)),
        }
        team = []
        for kk in range(K):
            row = w * K + kk
            s_k = adapter.state(row) if row != idx else st
            la = int(last_act[row]) if last_act is not None else -1
            team.append({
                "agent": kk,
                "x": float(s_k.get("x", 0.0)), "y": float(s_k.get("y", 0.0)), "z": float(s_k.get("z", 0.0)),
                "yaw": float(s_k.get("yaw", 0.0)), "pitch": float(s_k.get("pitch", 0.0)),
                "health": float(s_k.get("health", 0.0)),
                "last_action": ACTIONS[la] if 0 <= la < len(ACTIONS) else None,
                "msg": int(last_msg[row]) if last_msg is not None else 0,
                "inventory": adapter.inventory(row),
            })
        chests = [{"x": int(c["x"]), "y": int(c["y"]), "z": int(c["z"]), "items": c["items"]}
                  for c in adapter.chests(w)]
        reg = adapter.region(idx, 16)
        if reg is not None:
            (ox, oy, oz), arr = reg
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
            origin = [int(ox), int(oy), int(oz)]
            size = [int(arr.shape[2]), int(arr.shape[0]), int(arr.shape[1])]  # sx, sy, sz
            blocks = base64.b64encode(arr.tobytes()).decode("ascii")
        else:
            origin, size, blocks = None, list(E.WORLD_SIZE), None
        obs = np.ascontiguousarray(_np(meta["obs_img"]), dtype=np.uint8)
        la = meta["last_action"]
        seed = st.get("seed", meta.get("seed"))
        return {
            "env_id": int(env_id), "world_id": int(w), "agent_id": int(k),
            "agents_per_world": K,
            "episode": int(meta["episode"]),
            "seed": None if seed is None else int(seed), "step": int(meta["step"]),
            "player": player, "last_action": ACTIONS[la] if 0 <= la < len(ACTIONS) else None,
            "last_reward": float(meta["last_reward"]), "ep_reward": float(meta["ep_reward"]),
            "task": meta["task"], "family": meta.get("family"), "target": st.get("target"),
            "time_of_day": adapter.time_of_day(w), "entities": adapter.entities(w),
            "inventory": adapter.inventory(idx), "action_history": [ACTIONS[a] for a in meta["hist"]],
            "team": team, "chests": chests,
            "chat": [dict(c) for c in list(chat)[-CHAT_LEN:]],
            "origin": origin, "size": size, "blocks": blocks, "blocks_available": reg is not None,
            "obs_res": obs.shape[0], "obs": base64.b64encode(obs.tobytes()).decode("ascii"),
        }

    def _maybe_emit_snapshot(self, now):
        if self.server is None or self.sim is None:
            return
        if not self.cfg["visualize"] or self.cfg["mode"] == "fast":
            return
        if not self.server.any_subscribed("snapshot"):
            return
        if now - self._last_snap < 1.0 / max(0.1, float(self.cfg["snapshot_hz"])):
            return
        self._last_snap = now
        sel = self.sel
        K = int(self.cfg["agents"])
        payload = self._snapshot_payload(self.sim, sel, sel, self._meta_for(sel), self.world_chat[sel // K],
                                         self.last_act, self.last_msg)
        self.server.broadcast("snapshot", payload, need="snapshot")

    def _maybe_emit_nn(self, now):
        if self.server is None or self.sim is None:
            return
        if self.cfg["mode"] == "fast" or not self.server.any_subscribed("nn"):
            return
        if now - self._last_nn < 1.0 / max(0.1, float(self.cfg["nn_hz"])):
            return
        self._last_nn = now
        sel = self.sel
        inst = self.insts[sel % len(self.insts)]   # the policy of the selected row's slot
        payload, _hn = inst.probe(self.img[sel], self.vec[sel], self.h[sel:sel + 1], sel,
                                  int(self.env_steps_n[sel]))
        if payload is not None:
            self.server.broadcast("activations", payload, need="nn")

    def _sync_instrument(self):
        want = (self.server is not None and self.cfg["mode"] != "fast"
                and self.server.any_subscribed("nn"))
        for inst in self.insts:
            inst.set_active(want)

    # ----- environment stepping -----
    def _collect_step(self):
        """One decision for every agent row: act, step the sim with (action, msg), store the transitions,
        and close the episode of every world that ended (once per world, not per agent)."""
        r = self.roll
        t = self.roll_t
        K = int(self.cfg["agents"])
        W = self.sim.worlds
        a, m, logp, v, vi, hn, err = self._forward_on(self.img, self.vec, self.h, sample=True,
                                                      rnd=self.cfg["curiosity"] == "rnd")
        r.img[t] = self.img
        r.vec[t] = self.vec
        r.hid[t] = self.h.numpy()
        r.act[t] = a
        r.msg[t] = m
        r.logp[t] = logp
        r.val[t] = v
        r.val_int[t] = vi
        r.rnd[t] = err
        r.prev[t] = self.prev_env
        img, vec, rew, done, ach = self.sim.step(a.astype(np.int32), m.astype(np.int32))
        ach = np.asarray(ach, np.int64)
        r.mask[t] = ach
        rew = np.asarray(rew, np.float32)
        d = np.asarray(done, np.float32)
        r.rew[t] = rew
        r.done[t] = d
        self.h = hn * torch.from_numpy(1.0 - d)[:, None]
        self.img, self.vec = img, vec
        N = self.sim.n
        nov = self.sim.last_novelty()
        if nov is not None:
            self.nov_sum += float(nov.sum())
            self.nov_n += N
        self.ep_rew += rew
        if self.pop_on:
            for k in range(K):
                self.pop_win[k]["rsum"] += float(rew[k::K].sum())
                self.pop_win[k]["rn"] += W
        self.ep_len += 1
        self.env_steps_n += 1
        self.last_rew = rew.astype(np.float64)
        prim = primary_of(a).astype(np.int64)
        self.last_act = prim
        self.last_msg = m.astype(np.int64)
        idx = np.arange(N)
        self.act_hist[idx, self.act_ptr % ACTION_HISTORY] = prim
        self.act_ptr += 1
        self.prev_env = ach.copy()
        # chat: every non-silent token goes into its world's ring buffer (PROTOCOL 6.8)
        for i in np.nonzero(m)[0]:
            self.world_chat[i // K].append({"step": int(self.ep_len[i]), "agent": int(i % K), "msg": int(m[i])})
        win = self.win
        win["agent_steps"] += N
        win["msg_nonsilent"] += int(np.count_nonzero(m))
        win["msg_counts"] += np.bincount(m, minlength=NUM_MSGS)
        counts = np.bincount(prim, minlength=NUM_ACTIONS)
        win["deposit"] += int(counts[DEPOSIT])
        win["withdraw"] += int(counts[WITHDRAW])
        win["give"] += int(counts[GIVE])
        # world episodes: all K agents of a world end on the same step
        for w in np.nonzero(d.reshape(W, K)[:, 0] > 0)[0]:
            self._end_world_episode(int(w), ach)
        if self.cfg["go_explore"] and E.HAS_STATE and K == 1:
            self._archive_update(ach, d)
        self.counters["env_steps"] += N
        self.roll_t += 1

    def _end_world_episode_v7(self, w, ach):
        """v7: record world w's finished episode (its task, family, success, return), then the scheduler starts
        the next one: the sim already drew it (queued task); its kit is applied to the vec obs here."""
        K = int(self.cfg["agents"])
        rows = slice(w * K, (w + 1) * K)
        mask = int(ach[w * K])
        success = bool((mask >> GOAL_BIT) & 1)
        ret = float(self.ep_rew[rows].mean())
        length = int(self.ep_len[w * K])
        kind = EP_KIT if self.tasks.cur_kit[w] else EP_SCRATCH
        self.ep_hist.append((ret, length, mask, -1, kind))
        if self.pop_on:   # per-slot fitness: every world hosts all slots, so tasks are shared fairly
            for k in range(K):
                self.pop_win[k]["returns"].append(float(self.ep_rew[w * K + k]))
                self.pop_win[k]["succ"] += int(success)
        self.kind_window.append(kind)
        self.counters["episodes"] += 1
        self.ep_index[rows] += 1
        self.ep_rew[rows] = 0.0
        self.ep_len[rows] = 0
        self.vec = self.tasks.advance(self.sim, w, K, self.vec, success, ret, length)
        for i in range(w * K, (w + 1) * K):
            self.ep_goal[i] = -1
            self.ep_kind[i] = EP_KIT if self.tasks.cur_kit[w] else EP_SCRATCH
        self.start_mask[rows] = 0
        self.prev_env[rows] = 0

    def _end_world_episode(self, w, ach):
        if V7:
            return self._end_world_episode_v7(w, ach)
        """Record the episode of world w once (goal, success, length, mean reward over its agents), then
        start the next one: the sim already drew a fresh world (goal, kit); a Go-Explore restart may
        override it."""
        K = int(self.cfg["agents"])
        rows = slice(w * K, (w + 1) * K)
        mask = int(ach[w * K])
        g = int(self.ep_goal[w * K])
        kind = int(self.ep_kind[w * K])
        achieved = bool((mask >> GOAL_BIT) & 1)
        self.ep_hist.append((float(self.ep_rew[rows].mean()), int(self.ep_len[w * K]), mask, g, kind))
        if self.pop_on:   # per-slot fitness: each slot's own return of this world episode
            for k in range(K):
                self.pop_win[k]["returns"].append(float(self.ep_rew[w * K + k]))
                self.pop_win[k]["succ"] += int(achieved)
        self.kind_window.append(kind)
        if self.cur.is_armed(self.counters["env_steps"]):
            self.cur.record(g, achieved, kind=kind)
        self.counters["episodes"] += 1
        self.ep_index[rows] += 1
        self.ep_rew[rows] = 0.0
        self.ep_len[rows] = 0
        for i in range(w * K, (w + 1) * K):
            self.ep_goal[i] = self.sim.goal(i)
            self.ep_kind[i] = EP_KIT if self.sim.kit(i) else EP_SCRATCH
        self.start_mask[rows] = 0
        self.prev_env[rows] = 0
        self._maybe_restart(w * K)
        self.prev_env[rows] = self.start_mask[rows]

    def _schedules(self):
        """lr and entropy coefficient for the update about to run (linear over total_steps)."""
        frac = min(1.0, self.counters["env_steps"] / max(1, int(self.cfg["total_steps"])))
        lr = self.cfg["lr"]
        if self.cfg["lr_cosine"]:
            lr = lr * self._cosine_factor()
        elif self.cfg["lr_anneal"]:
            lr = lr * (1.0 - (1.0 - LR_ANNEAL_END_FRAC) * frac)
        ent = self.cfg["ent_coef"]
        if self.cfg["ent_coef_end"] is not None:
            ent = ent + (float(self.cfg["ent_coef_end"]) - ent) * frac
        return lr, ent

    def _set_member_hparams(self, lr, ent):
        """Population: each member's lr / ent_coef = its own base value times the schedule's factor."""
        base_lr, base_ent = float(self.cfg["lr"]), float(self.cfg["ent_coef"])
        lr_mult = lr / base_lr if base_lr > 0 else 1.0
        ent_mult = ent / base_ent if base_ent > 0 else 1.0
        for mb in self.members:
            e = mb.ent_base * ent_mult
            mb.ppo.set_lr(mb.lr_base * lr_mult)
            mb.ppo.ent_coef = e
            mb.ppo.msg_ent_coef = self._msg_ent_base() * (e / base_ent if base_ent > 0 else 1.0)

    def _update_members(self, last_val, last_val_int, rew_scaled, rew_int):
        """One PPO update per member on its own slot's rows of the shared rollout (PROTOCOL 10.1)."""
        K = self.P
        outs = []
        for p, mb in enumerate(self.members):
            sl = slice(p, None, K)
            inst = self.insts[p]
            outs.append(mb.ppo.update(_RolloutSlice(self.roll, sl), last_val[sl], last_val_int[sl],
                                      pump=self._pump,
                                      grad_hook=inst.record_grads if inst.active else None,
                                      rew=rew_scaled[:, sl],
                                      rew_int=None if rew_int is None else rew_int[:, sl]))
        stats = {}
        for key in outs[0]:
            vals = [o[key] for o in outs if o.get(key) is not None]
            if key == "skipped_nonfinite":
                stats[key] = int(sum(vals))
            elif vals:
                stats[key] = float(np.mean(vals))
            else:
                stats[key] = None
        return stats

    def _do_update(self):
        _, _, _, last_val, last_val_int, _, _ = self._forward_on(self.img, self.vec, self.h, sample=False)
        lr, ent = self._schedules()
        if self.pop_on:
            self._set_member_hparams(lr, ent)
        else:
            self.ppo.set_lr(lr)
            self.ppo.ent_coef = ent
            # the chat-token entropy coefficient follows the action entropy schedule (same relative scale)
            base = float(self.cfg["ent_coef"])
            self.ppo.msg_ent_coef = self._msg_ent_base() * (ent / base if base > 0 else 1.0)
        if self.demo is not None:
            self.demo.coef = self._demo_coef()
        self.ret_norm.update(_np(self.roll.rew), _np(self.roll.done))
        rew_scaled = self.ret_norm.scale(self.roll.rew)
        rew_int, rnd_mean = None, None
        if self._int_on():
            # intrinsic reward: RND error divided by its running std, times the decaying coefficient
            if self.cfg["curiosity"] == "spr":
                self.roll.rnd = self.ppo.spr_intrinsic(self.roll, self.img, self.vec)
            rnd_np = _np(self.roll.rnd)
            self.rnd_norm.update(rnd_np)
            rew_int = np.clip(self.rnd_norm.scale(rnd_np), 0.0, RND_REWARD_CLIP)
            rew_int = (self._rnd_coef() * rew_int).astype(np.float32)
            rnd_mean = float(rew_int.mean())
        self.in_update = True
        try:
            if self.pop_on:
                stats = self._update_members(last_val, last_val_int, rew_scaled, rew_int)
            else:
                grad_hook = self.inst.record_grads if self.inst.active else None
                stats = self.ppo.update(self.roll, last_val, last_val_int, pump=self._pump,
                                        grad_hook=grad_hook, rew=rew_scaled, rew_int=rew_int)
        finally:
            self.in_update = False
        self.counters["updates"] += 1
        nov_mean = float(self.nov_sum) / self.nov_n if self.nov_n else 0.0
        self.nov_sum, self.nov_n = 0.0, 0
        # team stats over the rollout just trained on: per 1k agent steps for the counts
        win = self.win
        steps = max(1, win["agent_steps"])
        team = {"msg_rate": win["msg_nonsilent"] / steps,
                "chest_deposits": 1000.0 * win["deposit"] / steps,
                "chest_withdraws": 1000.0 * win["withdraw"] / steps,
                "gives": 1000.0 * win["give"] / steps,
                "msg_hist": (win["msg_counts"] / steps).tolist()}
        if V7:
            team["build_progress"] = float(np.mean(self.vec[:, BUILD_IDX]))
            team["health_mean"] = float(np.mean(self.vec[:, HEALTH_IDX])) * 20.0
        self.win = _new_window()
        self.last_stats = dict(stats, step_reward=float(self.roll.rew.mean()), lr=lr, ent_coef=ent,
                               demo_coef=self.demo.coef if self.demo is not None else None,
                               msg_ent_coef=float(self.ppo.msg_ent_coef),
                               novelty_reward=nov_mean, rnd_reward=rnd_mean, **team)
        if stats.get("skipped_nonfinite"):
            self.log("warn", f"update {self.counters['updates']}: skipped "
                             f"{stats['skipped_nonfinite']} non-finite minibatches")
        for inst in self.insts:
            if inst.active:
                inst.refresh_weights()
        self.roll_t = 0

    def _after_update(self):
        if V7:
            before = self.tasks.status()["unlocked"]
            self.tasks.rebalance()
            now = self.tasks.status()["unlocked"]
            if now != before:
                self.log("info", f"curriculum: {now}/{self.tasks.status()['total']} tasks unlocked")
        else:
            for name in self.cur.maybe_unlock():
                self.log("info", f"curriculum: unlocked {name} -> {self.cur.stage()}")
        self._apply_sim_config()   # takes effect for episodes started from now on
        self._maybe_apply_pending()
        if self.pop_on and self.counters["env_steps"] >= self.next_evolve:
            self._evolve()
        due_steps = self.counters["env_steps"] - self.last_ckpt_steps >= int(self.cfg["ckpt_every"])
        secs = float(self.cfg.get("ckpt_seconds") or 0.0)
        due_time = secs > 0 and self.counters["env_steps"] > self.last_ckpt_steps and \
            time.time() - self._last_ckpt_wall >= secs
        if due_steps or due_time:
            self.save_checkpoint("periodic")
        if self.state == "stopping":
            self._finalize("stop requested")
        elif self.counters["env_steps"] >= int(self.cfg["total_steps"]):
            self._finalize("target reached")

    # ----- population (PROTOCOL 10.2) -----
    def _evolve(self):
        """Rank the P policies by fitness (then team success); the top half are parents; each bottom-half
        slot is replaced by a child of two parents (the two best for P = 4). Children: crossover + mutation
        of the weights, hyperparameters from a random parent times {0.8, 1.25}, fresh optimizer, zeroed GRU
        state in their rows."""
        P, K = self.P, self.P
        steps = self.counters["env_steps"]
        fit = [self._fitness(p) for p in range(P)]
        succ = [self._success(p) or 0.0 for p in range(P)]
        basis = "episode return" if all(self.pop_win[p]["returns"] for p in range(P)) else "step reward"
        n_eps = len(self.pop_win[0]["returns"])
        rank = sorted(range(P), key=lambda p: (-np.inf if fit[p] is None else fit[p], succ[p]), reverse=True)
        ranking = [{"slot": p, "id": self.members[p].id, "fitness": fit[p], "lr": self.members[p].lr_base,
                    "ent": self.members[p].ent_base, "gen": self.members[p].gen} for p in rank]
        gen = self.evolutions + 1
        rng = np.random.default_rng([int(self.cfg["seed"]) % 2**32, gen])
        n_rep = P // 2
        pool, victims = rank[:P - n_rep], rank[P - n_rep:]
        replaced = []
        for s in victims:
            if len(pool) >= 2:
                a, b = (int(x) for x in rng.choice(pool, size=2, replace=False))
            else:
                a = b = pool[0]
            ma, mb_ = self.members[a], self.members[b]
            hp = ma if rng.random() < 0.5 else mb_
            child = _breed(ma.model.state_dict(), mb_.model.state_dict(), rng, float(self.cfg["mutation"]))
            lr = float(np.clip(hp.lr_base * POP_HP_FACTORS[int(rng.integers(2))], *POP_LR_RANGE))
            ent = float(np.clip(hp.ent_base * POP_HP_FACTORS[int(rng.integers(2))], *POP_ENT_RANGE))
            m = self.members[s]
            old_id = m.id
            m.model.load_state_dict(child)
            m.id, m.gen, m.parents = self.next_id, gen, [ma.id, mb_.id]
            self.next_id += 1
            m.lr_base, m.ent_base = lr, ent
            m.ppo.opt = torch.optim.Adam(m.model.parameters(), lr=lr, eps=1e-5)   # fresh optimizer
            if m.ppo.spr:
                m.ppo.reset_target()
            self.h[s::K] = 0.0                                                    # GRU state of its rows
            replaced.append({"slot": s, "old_id": old_id, "new_id": m.id, "parents": m.parents})
        self.evolutions = gen
        every = int(self.cfg["evolve_every"])
        while self.next_evolve <= steps:
            self.next_evolve += every
        self._reset_pop_window()
        self._set_best(rank[0])
        rec = {"evolution": gen, "env_steps": steps, "basis": basis, "ranking": ranking,
               "replaced": replaced}
        self.lineage = (self.lineage + [rec])[-POP_LINEAGE_KEEP:]
        rk = " | ".join(f"id{r['id']} slot{r['slot']} f={r['fitness']:.4g} lr={r['lr']:.2e} "
                        f"ent={r['ent']:.3g} gen={r['gen']}" for r in ranking)
        rp = "; ".join(f"slot {r['slot']} (id {r['old_id']} -> {r['new_id']}, child of {r['parents'][0]}+"
                       f"{r['parents'][1]})" for r in replaced)
        self.log("info", f"evolution {gen} at env_steps={steps:,} (fitness = {basis}, {n_eps} episodes/slot) "
                         f"ranking [{rk}]; replaced {rp}")

    def _reset_pop_window(self):
        self.pop_win = [{"returns": [], "succ": 0, "rsum": 0.0, "rn": 0} for _ in range(self.P)] \
            if self.pop_on else []

    # ----- checkpoints -----
    def _payload(self, reason):
        body = self._payload_policy(reason)
        if self.pop_on:
            body["models"] = [mb.model.state_dict() for mb in self.members]
            body["optimizers"] = [mb.ppo.opt.state_dict() for mb in self.members]
            body["population"] = self._population_meta()
        return body

    def _payload_policy(self, reason):
        return {
            "run_id": self.run_id,
            "obs": obs_signature(int(self.cfg["res"])),
            "model": self.model.state_dict(),
            "optimizer": self.ppo.opt.state_dict(),
            "config": dict(self.cfg),
            "counters": dict(self.counters),
            "curriculum": self.cur.state_dict() if self.cur is not None else None,
            "curriculum_stage": self.cur.stage() if self.cur is not None else self.tasks.telemetry()["curriculum_stage"],
            "tasks": self.tasks.state_dict() if self.tasks is not None else None,
            "reward_norm": self.ret_norm.state_dict(),
            "rnd_norm": self.rnd_norm.state_dict(),
            "rng": ckpt.rng_state(),
            "metadata": {"reason": reason, "run_id": self.run_id, "mode": self.cfg["mode"]},
        }

    def save_checkpoint(self, reason):
        path = ckpt.save(self.ckpt_dir, self._payload(reason))
        self.latest_ckpt = path
        self.last_ckpt_steps = self.counters["env_steps"]
        self._last_ckpt_wall = time.time()
        self.log("info", f"checkpoint saved ({reason}): {path}")
        return path

    def _finalize(self, reason):
        if self.counters["env_steps"] > 0:
            self.save_checkpoint(f"final: {reason}")
        self.state = "stopped"
        self.stop_requested = False
        self.log("info", f"stopped ({reason})")
        if reason == "target reached" and self.server is None:
            self.quit = True

    def _request_stop(self, quit_after=False):
        if quit_after:
            self.quit = True
        if self.state == "running":
            self.state = "stopping"
            self.stop_requested = True
            self.log("info", "stop requested: finishing current update")
        elif self.state in ("idle", "paused"):
            self.roll_t = 0
            self._finalize("stop")
        elif self.state == "stopping" and quit_after:
            pass

    # ----- config / boundary -----
    def _maybe_apply_pending(self):
        if not self.pending or self.in_update or self.roll_t != 0:
            return
        changes, self.pending = self.pending, {}
        rebuild = False
        for k, v in changes.items():
            if k == "lr":
                if self.pop_on:   # members keep their own base lr; scale them by the new / old value
                    old = float(self.cfg["lr"]) or 1.0
                    for mb in self.members:
                        mb.lr_base *= v / old
                self.cfg["lr"] = v
                self.ppo.set_lr(v)
            elif k == "minibatch_size":
                self.cfg[k] = v
                self.ppo.minibatch_size = v
            elif k in ("num_envs", "rollout_len", "workers"):
                self.cfg[k] = v
                rebuild = True
            elif k == "mode":
                self._set_mode(v)
            else:
                self.cfg[k] = v
        if rebuild:
            self._build_envs()
        self.log("info", f"applied queued config at env_steps={self.counters['env_steps']}: "
                         f"{', '.join(sorted(changes))}")

    def _set_mode(self, mode):
        self.cfg["mode"] = mode
        self.cfg.update(MODE_PRESETS[mode])
        self._sync_instrument()

    # ----- command handling -----
    def _pump(self):
        if self.server is not None:
            for msg in self.server.poll():
                self._dispatch(msg)
        if self.deferred and not self.in_update:
            pending = list(self.deferred)
            self.deferred.clear()
            for msg in pending:
                self._dispatch(msg)
        self._sync_instrument()
        now = time.time()
        if now - self._last_metrics >= 0.5:
            self._emit_metrics(now)
        self._maybe_emit_snapshot(now)
        self._maybe_emit_nn(now)
        self._maybe_apply_pending()

    def _dispatch(self, msg):
        cmd = msg.get("cmd")
        cid = msg.get("_client")
        mid = msg.get("id")
        if cmd == "_connect":
            if self.server is not None:
                self._send_hello(cid)
            return
        if cmd == "_disconnect":
            return
        if self.in_update and cmd in DEFER_DURING_UPDATE:
            self.deferred.append(msg)
            return
        handler = getattr(self, "_cmd_" + str(cmd), None)
        if handler is None:
            self._ack(cid, mid, False, f"unknown command {cmd!r}")
            return
        try:
            ok, text = handler(msg, cid)
        except Exception as e:  # keep the loop alive on bad input
            ok, text = False, f"{type(e).__name__}: {e}"
        self._ack(cid, mid, ok, text)

    def _cmd_start(self, msg, cid):
        if self.state in ("idle", "stopped"):
            self.state = "running"
            return True, "running"
        if self.state == "paused":
            self.state = "running"
            return True, "resumed"
        if self.state == "running":
            return True, "already running"
        return False, "stopping: wait for the final checkpoint"

    def _cmd_resume(self, msg, cid):
        if self.state == "paused":
            self.state = "running"
            return True, "resumed"
        if self.state == "running":
            return True, "already running"
        return False, f"cannot resume from {self.state}"

    def _cmd_pause(self, msg, cid):
        if self.state == "running":
            self.state = "paused"
            return True, "paused"
        if self.state == "paused":
            return True, "already paused"
        return False, f"cannot pause from {self.state}"

    def _cmd_stop(self, msg, cid):
        if self.state == "stopped":
            return True, "already stopped"
        if self.state == "stopping":
            return True, "already stopping"
        self._request_stop()
        return True, "stopping" if self.state == "stopping" else "stopped"

    def _cmd_reset(self, msg, cid):
        if self.state not in ("paused", "stopped"):
            return False, "reset requires paused or stopped"
        self._reset_count += 1
        torch.manual_seed(self.cfg["seed"] + self._reset_count)
        for inst in self.insts:
            inst.set_active(False)
        self._new_policies()
        self.counters = {"env_steps": 0, "updates": 0, "episodes": 0}
        self._init_pop_schedule(None)
        self.last_ckpt_steps = 0
        self.cur = self._new_curriculum()
        self.tasks = self._new_tasks(int(self.cfg["num_envs"]))
        self.archive = self._new_archive()
        self.restart_rng = np.random.default_rng(self.cfg["seed"] + 7)
        self.ret_norm = ReturnNormalizer(self.cfg["gamma"])
        self.rnd_norm = RunningStd()
        self.ep_hist.clear()
        self.kind_window.clear()
        self.novelty_t0 = 0
        self.last_stats = {}
        self.elapsed = 0.0
        self.ema_sps = None
        self.sps_samples = 0
        self._m_steps = 0
        self._m_t = time.time()
        self.warm_report = None
        if self.cfg.get("warm_start") and os.path.isfile(self.cfg["warm_start"]):
            self._apply_warm_start(self.cfg["warm_start"])   # the run's initial state, as it was started
        self._build_envs()
        self.log("info", "reset: fresh model, optimizer and counters")
        if self.server is not None:
            self.server.broadcast("model", self._model_msg())
        return True, "reset"

    def _cmd_save_checkpoint(self, msg, cid):
        path = self.save_checkpoint("manual")
        return True, path

    def _cmd_load_checkpoint(self, msg, cid):
        if self.state not in ("paused", "stopped"):
            return False, "load_checkpoint requires paused or stopped"
        path = msg["path"]
        payload = ckpt.load_for_training(path)
        pop_meta = self._restore_policies(payload)
        self.counters = dict(payload["counters"])
        self._init_pop_schedule(pop_meta)
        self.last_ckpt_steps = self.counters["env_steps"]
        ckpt.restore_rng(payload.get("rng"))
        self.cur = self._new_curriculum()
        self.tasks = self._new_tasks(int(self.cfg["num_envs"]))
        self.archive = self._new_archive()
        self.restart_rng = np.random.default_rng(self.cfg["seed"] + 7)
        self.ret_norm = ReturnNormalizer(self.cfg["gamma"])
        self.rnd_norm = RunningStd()
        self._restore_curriculum(payload)
        self.ep_hist.clear()
        self.kind_window.clear()
        self.novelty_t0 = self.counters["env_steps"]
        self.last_stats = {}
        self._build_envs()
        self.log("info", f"loaded checkpoint {path} (env_steps={self.counters['env_steps']}, "
                         f"updates={self.counters['updates']})")
        if self.server is not None:
            self.server.broadcast("model", self._model_msg())
        return True, f"loaded env_steps={self.counters['env_steps']}"

    def _cmd_eval_checkpoint(self, msg, cid):
        path = msg["path"]
        episodes = int(msg.get("episodes", 8))
        if not (1 <= episodes <= 200):
            return False, "episodes must be 1..200"
        if not os.path.isfile(path):
            return False, f"no such checkpoint: {path}"
        if self._eval_thread is not None and self._eval_thread.is_alive():
            return False, "an evaluation is already running"
        self._eval_thread = threading.Thread(target=self._run_eval, args=(path, episodes),
                                             name="eval", daemon=True)
        self._eval_thread.start()
        return True, f"evaluation started ({episodes} episodes)"

    def _run_eval(self, path, episodes):
        """Separate policy copy on a separate sim (teams as in the checkpoint); never touches training."""
        try:
            pl = ckpt.load_for_training(path)
            cfg_pl = pl.get("config", {})
            model = build_model(pl)                       # CPU copy for eval
            model.load_state_dict(pl["model"], strict=False)
            model.eval()
            K = int(cfg_pl.get("agents", 1))
            W = min(episodes, 16)                         # worlds; each finishes ceil(episodes / W) episodes
            quota = math.ceil(episodes / W)
            sim = SimAdapter(W, seed=int(time.time()) % 100000, threads=1, res=int(pl["obs"]["res"]), agents=K)
            worlds = run_worlds(model, sim, quota)
            sim.close()
            eps = [e for ep in worlds for e in ep][:episodes]
            masks = [e["mask"] for e in eps]
            rewards = [e["reward"] for e in eps]
            lens = [e["length"] for e in eps]
            rates = {ITEMS[i]: sum(1 for m in masks if (m >> i) & 1) / max(1, len(masks))
                     for i in range(NUM_ITEMS)}
            rates = {k2: v for k2, v in rates.items() if v > 0}
            self.log("info", f"eval {os.path.basename(path)} (env_steps={pl['counters']['env_steps']}, "
                             f"agents={K}): episodes={len(rewards)} avg_reward={np.mean(rewards):.3f} "
                             f"avg_len={np.mean(lens):.1f} success={json.dumps(rates)}")
        except Exception as e:
            self.log("error", f"eval failed for {path}: {type(e).__name__}: {e}")

    def _cmd_list_checkpoints(self, msg, cid):
        items = ckpt.list_checkpoints(self.runs_dir)
        self.server.send_to(cid, "checkpoints", {"items": items})
        return True, f"{len(items)} checkpoints"

    def _cmd_get_history(self, msg, cid):
        last = int(msg.get("last", 500))
        rows = list(self.metrics_hist)[-max(1, last):]
        self.server.send_to(cid, "history", {"rows": rows})
        return True, f"{len(rows)} rows"

    def _cmd_set_config(self, msg, cid):
        values = msg.get("values")
        if not isinstance(values, dict) or not values:
            return False, "values must be a non-empty object"
        clean = {}
        for k, v in values.items():
            if k not in SET_KEYS:
                return False, f"unknown or read-only key {k!r}"
            clean[k] = self._coerce(k, v)
        self.pending.update(clean)
        msg_text = f"queued: {', '.join(sorted(clean))} (applied at next rollout boundary)"
        if "num_envs" in clean or "rollout_len" in clean:
            warn = self._mem_check(int(clean.get("num_envs", self.cfg["num_envs"])),
                                   int(clean.get("rollout_len", self.cfg["rollout_len"])))
            if warn:
                msg_text += f"; WARNING: {warn}"
                self.log("warn", warn)
        return True, msg_text

    def _coerce(self, k, v):
        if k == "visualize":
            if not isinstance(v, bool):
                raise ValueError("visualize must be bool")
            return v
        if k == "mode":
            if v not in MODES:
                raise ValueError(f"mode must be one of {MODES}")
            return v
        if k in ("snapshot_hz", "nn_hz", "lr"):
            v = float(v)
            if not (0 < v <= 1000):
                raise ValueError(f"{k} out of range")
            return v
        v = int(v)
        ranges = {"total_steps": (0, 10**12), "num_envs": (1, 4096), "minibatch_size": (1, 10**7),
                  "rollout_len": (1, 4096), "ckpt_every": (1, 10**12), "workers": (0, 64)}
        lo, hi = ranges[k]
        if not (lo <= v <= hi):
            raise ValueError(f"{k} must be in [{lo}, {hi}]")
        return v

    def _cmd_set_mode(self, msg, cid):
        mode = msg.get("mode")
        if mode not in MODES:
            return False, f"mode must be one of {MODES}"
        self._set_mode(mode)
        self.log("info", f"mode={mode}")
        return True, f"mode={mode}"

    def _cmd_select_env(self, msg, cid):
        env_id = int(msg.get("env_id", -1))
        if not (0 <= env_id < self.sim.n):
            return False, f"env_id must be in [0, {self.sim.n})"
        self.sel = env_id
        return True, f"selected env {env_id}"

    def _cmd_subscribe(self, msg, cid):
        snap = bool(msg.get("snapshot", False))
        nn = bool(msg.get("nn", False))
        if self.server is None or not self.server.set_subscription(cid, snapshot=snap, nn=nn):
            return False, "unknown client"
        if "nn_layers" in msg:
            for inst in self.insts:
                inst.set_layers(msg.get("nn_layers") or [])
        self._sync_instrument()
        return True, f"subscribed snapshot={snap} nn={nn}"

    def _cmd_debug_step(self, msg, cid):
        if self.state != "paused":
            return False, "debug_step requires paused state"
        if not E.HAS_RESET_ENV:
            return False, "sim has no reset_env(): debug stepping unavailable"
        n = int(msg.get("n", 1))
        kind = msg.get("kind", "decision")
        if not (1 <= n <= 1000):
            return False, "n must be 1..1000"
        if kind not in ("tick", "decision"):
            return False, "kind must be 'tick' or 'decision'"
        sel = self.sel
        K = int(self.cfg["agents"])
        seed = self.sim.state(sel).get("seed")
        if seed is None:
            return False, "sim does not report the selected env's seed"
        key = (sel // K, int(seed))
        if self.debug is None or self.debug_key != key:
            if self.debug is not None:
                self.debug.close()
            # one debug world with the same K agents, rebuilt from the selected agent's world seed
            self.debug = SimAdapter(1, seed=0, threads=1, res=int(self.cfg["res"]), agents=K)
            self.debug.reset_env(0, int(seed))
            # reset_env returns no observation; one noop tick yields the first obs of every agent
            zeros = np.zeros(K, np.int32)
            self.debug.step(zeros, zeros)
            self.debug_key = key
            self.debug_h = self.model.zero_hidden(K)
            self.debug_last_act = np.full(K, -1, np.int64)
            self.debug_last_msg = np.zeros(K, np.int64)
            self.debug_chat = deque(maxlen=CHAT_LEN)
            self.debug_meta = {"episode": 0, "seed": int(seed), "step": 1, "last_action": -1,
                               "last_reward": 0.0, "ep_reward": 0.0, "hist": [], "obs_img": None}
        meta = self.debug_meta
        dk = sel % K
        for _ in range(n):
            if kind == "tick":
                a = np.zeros((K, 3) if V7 else K, np.int64)
                m = np.zeros(K, np.int64)
                hn = self.debug_h
            else:
                if self.pop_on:
                    a, m, _, _, _, hn, _ = act_by_slot([mb.model for mb in self.members], self.debug.img,
                                                       self.debug.vec, self.debug_h, sample=True)
                else:
                    a, m, _, _, hn = policy_act(self.model, self.debug.img, self.debug.vec, self.debug_h,
                                                sample=True)
            img, vec, rew, done, _ach = self.debug.step(a.astype(np.int32), m.astype(np.int32))
            self.debug_h = hn * torch.from_numpy(1.0 - np.asarray(done, np.float32))[:, None]
            ap = primary_of(a).astype(np.int64)
            self.debug_last_act = ap
            self.debug_last_msg = m.astype(np.int64)
            for k in np.nonzero(m)[0]:
                self.debug_chat.append({"step": int(meta["step"]), "agent": int(k), "msg": int(m[k])})
            meta["step"] += 1
            meta["last_action"] = int(ap[dk])
            meta["last_reward"] = float(rew[dk])
            meta["ep_reward"] += float(rew[dk])
            meta["hist"] = (meta["hist"] + [int(ap[dk])])[-ACTION_HISTORY:]
            if done[dk]:
                meta["episode"] += 1
                meta["ep_reward"] = 0.0
        meta["obs_img"] = self.debug.img[dk]
        meta["task"] = TASK_NAMES[self.debug.task(0)] if V7 else self._curriculum()["task"]
        meta["family"] = None
        payload = self._snapshot_payload(self.debug, dk, -1, meta, self.debug_chat,
                                         self.debug_last_act, self.debug_last_msg)
        self.server.send_to(cid, "snapshot", payload)
        return True, f"debug {kind} x{n}"

    # ----- main loop -----
    def run(self):
        last_status = time.time()
        while True:
            self._pump()
            if self._signal_stop:
                self._signal_stop = False
                self._request_stop(quit_after=True)
            if self.quit and self.state == "stopped":
                break
            if self.state in ("running", "stopping"):
                if self.roll_t < self.roll.T:
                    self._collect_step()
                else:
                    self._do_update()
                    self._after_update()
            else:
                time.sleep(0.02)
            if self.quit and self.state in ("idle", "paused"):
                self._request_stop(quit_after=True)
        self.shutdown()

    def shutdown(self):
        self._emit_metrics(time.time())
        if self.server is not None:
            self.server.close()
        if self.sim is not None:
            self.sim.close()
        if self.debug is not None:
            self.debug.close()
        self.log("info", f"exit: run_id={self.run_id} env_steps={self.counters['env_steps']} "
                         f"updates={self.counters['updates']} checkpoint={self.latest_ckpt}")

    def request_signal_stop(self):
        self._signal_stop = True


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m trainer.run", description="MineMind PPO trainer")
    p.add_argument("--envs", type=int, default=None, help="number of parallel worlds (num_envs)")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--rollout", type=int, default=None, help="rollout length per env")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--minibatch", type=int, default=None, help="transitions per minibatch")
    p.add_argument("--ckpt-every", type=int, default=None, help="env steps between checkpoints")
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--mode", choices=MODES, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="master seed (default: random, logged and stored in the checkpoint)")
    p.add_argument("--workers", type=int, default=None, help="sim OpenMP threads (0 = default)")
    p.add_argument("--snapshot-hz", type=float, default=None)
    p.add_argument("--nn-hz", type=float, default=None)
    vis = p.add_mutually_exclusive_group()
    vis.add_argument("--visualize", dest="visualize", action="store_true", default=None)
    vis.add_argument("--no-visualize", dest="visualize", action="store_false")
    p.add_argument("--resume", metavar="PATH", default=None, help="checkpoint to resume from")
    p.add_argument("--no-server", action="store_true", help="headless: no TCP server, autostart")
    p.add_argument("--autostart", action="store_true", help="start training without a start command")
    p.add_argument("--runs-dir", default=DEFAULT_RUNS)
    p.add_argument("--run-id", default=None)
    p.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    p.add_argument("--lr-anneal", action="store_true", help="linear lr decay to 10%% over total_steps")
    p.add_argument("--ent-coef-end", type=float, default=None,
                   help="linear entropy-coef schedule from ent_coef to this value over total_steps")
    p.add_argument("--goal-start", type=int, default=None, help="goals unlocked at the start")
    p.add_argument("--unlock-at", type=float, default=None, help="frontier success EMA to unlock next goal")
    p.add_argument("--goal-min-episodes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="PPO epochs per rollout (default 1)")
    p.add_argument("--bptt", type=int, default=None, help="truncated-BPTT chunk length (default 32)")
    p.add_argument("--novelty", type=float, default=None,
                   help="hand-crafted novelty bonus start scale, linear to 0 (default 0.02; off with --rnd)")
    p.add_argument("--novelty-end-frac", type=float, default=None,
                   help="fraction of the steps left at process start over which novelty / rnd_coef decay")
    p.add_argument("--no-kits", action="store_true", help="never start episodes with a kit")
    p.add_argument("--go-explore", dest="go_explore", action="store_true", default=None)
    p.add_argument("--no-go-explore", dest="go_explore", action="store_false")
    p.add_argument("--restart-frac", type=float, default=None, help="Go-Explore restart fraction (0.25)")
    p.add_argument("--rnd", dest="rnd", action="store_true", default=None)
    p.add_argument("--no-rnd", dest="rnd", action="store_false")
    p.add_argument("--rnd-coef", type=float, default=None, help="RND reward start coefficient (0.1)")
    p.add_argument("--value-loss", choices=("hlgauss", "mse"), default=None)
    p.add_argument("--aux-ach", dest="aux_ach", action="store_true", default=None)
    p.add_argument("--no-aux-ach", dest="aux_ach", action="store_false")
    p.add_argument("--aux-coef", type=float, default=None)
    p.add_argument("--curiosity", choices=("rnd", "spr", "none"), default=None,
                   help="intrinsic reward source (default rnd); --rnd / --no-rnd set rnd / none")
    p.add_argument("--spr", dest="spr", action="store_true", default=None)
    p.add_argument("--no-spr", dest="spr", action="store_false")
    p.add_argument("--spr-coef", type=float, default=None)
    p.add_argument("--spr-k", type=int, default=None, help="SPR prediction horizon K (default 5)")
    p.add_argument("--spr-tau", type=float, default=None, help="SPR target EMA rate (default 0.99)")
    p.add_argument("--weight-decay", type=float, default=None, help="decoupled weight decay (1e-4)")
    p.add_argument("--lr-cosine", dest="lr_cosine", action="store_true", default=None)
    p.add_argument("--no-lr-cosine", dest="lr_cosine", action="store_false")
    p.add_argument("--resblock", dest="resblock", action="store_true", default=None)
    p.add_argument("--no-resblock", dest="resblock", action="store_false")
    p.add_argument("--backend", choices=("cpu", "gpusim"), default=None,
                   help="cpu (craftsim, the only v6 backend; gpusim is refused)")
    p.add_argument("--model-size", dest="model_size", choices=("small", "100m"), default=None,
                   help="policy architecture: small (v5 network) or 100m (~101M parameters)")
    p.add_argument("--ckpt-seconds", type=float, default=None,
                   help="also checkpoint every N wall-clock seconds (0 = off)")
    p.add_argument("--agents", type=int, default=None,
                   help="agents per world K (default 4; 1 = single-agent v5-style worlds)")
    p.add_argument("--team-share", type=float, default=None,
                   help="fraction of each team reward paid to every agent of the world (default 0.5)")
    p.add_argument("--msg-ent-coef", type=float, default=None,
                   help="entropy bonus of the chat-token head (default = the action entropy coef)")
    p.add_argument("--res", type=int, default=None,
                   help="observation image resolution (default 64; the CNN has 3 stride-2 convs)")
    p.add_argument("--population", type=int, default=None,
                   help="PROTOCOL 10: policies per team slot (0/1 = off; else must equal --agents)")
    p.add_argument("--evolve-every", type=int, default=None,
                   help="population: evolve (2 best breed, 2 worst replaced) every N agent steps (250000)")
    p.add_argument("--mutation", type=float, default=None,
                   help="population: Gaussian mutation scale, relative to each tensor's std (0.01)")
    p.add_argument("--obs", choices=("v6", "v7"), default=None,
                   help="observation / action layout: v7 (default; 295-dim obs, 21 primary actions with "
                        "recipe / item / msg heads, per-world tasks) or v6 (102-dim obs, 28 actions). Read "
                        "before the sim is imported; a checkpoint must match it")
    p.add_argument("--envmix", default=None,
                   help="v7: env mix name in data/envmix (family proportions; default 'default')")
    p.add_argument("--warm-start", metavar="PATH", default=None,
                   help="v5 checkpoint to warm start from (starts a new run; counters and optimizer fresh)")
    p.add_argument("--demos", nargs="+", default=None, metavar="DIR",
                   help="expert shard dirs (PROTOCOL 14.2): DAPG-style auxiliary masked CE loss inside PPO")
    p.add_argument("--no-demos", action="store_true", help="switch the demo loss off (e.g. on resume)")
    p.add_argument("--demo-coef", type=float, default=None,
                   help="demo loss weight at the start (default 0.5)")
    p.add_argument("--demo-coef-end", type=float, default=None,
                   help="demo loss weight at total_steps, linear (default 0.05)")
    p.add_argument("--demo-batch", type=int, default=None,
                   help="demo sequences (of 64 rows) per PPO minibatch (default 4)")
    p.add_argument("--demo-success-only", dest="demo_success_only", action=argparse.BooleanOptionalAction,
                   default=None, help="demos: successful episodes only (default on)")
    p.add_argument("--demo-actions", default=None,
                   help="demos: primary actions the demo loss trains: all | decisions | comma list (default all)")
    p.add_argument("--demo-max-idle", type=int, default=None,
                   help="demos: drop runs of more than N consecutive noop / sneak rows, keep the rest (default 8; "
                        "< 0 = off)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cpu = os.cpu_count() or 1
    trainer = None
    # torch threads depend on the sim workers setting (sim default 4 workers, torch gets the rest)
    workers = args.workers if args.workers is not None else DEFAULTS["workers"]
    torch.set_num_threads(args.threads or (max(1, cpu - workers) if workers > 0 else max(1, cpu // 2)))
    try:
        trainer = Trainer(args)
    except ValueError as e:   # e.g. a v1 or mismatched checkpoint passed to --resume
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not args.no_server:
        try:
            trainer.server = TelemetryServer(args.host, args.port)
        except OSError as e:
            print(f"cannot bind {args.host}:{args.port}: {e}", file=sys.stderr)
            return 2
        trainer.log("info", f"telemetry listening on {args.host}:{trainer.server.port}")
    if args.no_server or args.autostart:
        trainer.state = "running"

    def on_signal(signum, _frame):
        trainer.request_signal_stop()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    trainer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
