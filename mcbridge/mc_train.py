"""Online PPO fine-tuning of the MineMind policy inside Minecraft (PROTOCOL section 11.4, FT-2 of section 13).

The CraftBot mod (`/aibot train start`) sends ONE batched request per decision tick for all training bots:

  {"cmd": "train_step", "new_session"?: true,
   "bots": [{"id", "team", "slot", "img" (b64 uint8 res*res*2), "vec" [VEC_DIM],
             "reward" (for the previous action), "done" (episode ended after the previous action),
             "info": {goal, episode, step, ach, success, ep_ret, ep_len, died, msg, hp, yaw, pitch, pos, inv,
                      region?}}]}
  -> {"actions": [{"id", "action", "msg"}], "select": <bot id or null>, "stats": {...},
      "goal_weights": {item: w}}

  {"cmd": "train_end", "reason"}   -> the trainer saves a checkpoint after the queued updates.

goal_weights (FT-2, optional for the mod): relative sampling weight per goal item for the next episodes, mean 1
over the goals seen so far. Goals whose success EMA is inside [--goal-lo, --goal-hi] get weight 1.0, goals outside
that band (too hard or mastered) get 0.5, goals with fewer than --goal-min-eps finished episodes stay at 1.0.
The mod may ignore the field until it reads it (FT-4/FT-1); the bridge always sends it.

Data flow. Each bot is one row of a time-major rollout (T = --rollout decisions). A transition is completed
when the next request of that bot arrives (its reward and done flag are the previous action's). The GRU state
of each bot is kept here and is zeroed when its episode ended (the same rule as PPO's update). When a rollout
is full, it is handed to a background thread that runs one RecurrentPPO update (trainer/ppo.py) on a private
copy of the model; afterwards the acting copy is refreshed under a lock. Requests keep being answered during
the update (they use the last finished weights), so the game never waits on training.

Warmups (phases, counted in rollouts / PPO updates):
  1. norm_warmup: the first --norm-warmup rollouts only update the reward-return normalizer; the policy is not
     updated and the rollouts are discarded.
  2. critic_warmup: the next updates (until --critic-warmup updates in total) train only the value head
     (grad hook zeroes every other gradient), so the acting policy is bit-identical during this phase.
  3. ppo: full PPO updates.

KL to the frozen start policy: each transition's reward is reduced by kl_coef * (log pi(a|s) - log pi_ref(a|s))
for the sampled action (the usual per-step KL penalty, folded into the reward so PPO itself is unchanged).
kl_coef is adapted after every PPO-phase update, log-proportional (PI-style, gain --kl-gain, step clipped to x/2):
  kl_coef *= exp(clip(gain * ln(kl_to_start / --kl-target), -ln2, ln2)).
Entropy floor: when the action entropy drops below --entropy-floor, ent_coef is raised (x1.5 per update, up to
0.05); when it is above 1.5x the floor it decays back to --ent-coef.

Checkpoints: trainer format (trainer/checkpoint.py) under <out>/<run_id>/checkpoints/, every 5 minutes with new
updates, and on train_end. The best checkpoint by recent success (mean success of the last --best-window finished
episodes, needs --best-min-eps episodes) goes to <out>/<run_id>/best/. Each checkpoint also carries `mc_ft`
(phase counters, kl_coef, ent_coef, per-goal success counts/EMA, recent successes, best score) so `--resume`
continues exactly: model, optimizer, reward normalizer, counters, rng and the start checkpoint (the KL reference
stays the ORIGINAL start policy recorded in metadata.start_checkpoint). trainer.eval.load_policy loads them
unchanged.
"""
import base64
import math
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from collections import deque

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainer import checkpoint as ckpt  # noqa: E402
from trainer.envs import ACTIONS, BLOCKS, ITEMS  # noqa: E402
from trainer.eval import build_model  # noqa: E402
from trainer.ppo import RecurrentPPO, ReturnNormalizer, Rollout, act, load_optimizer_state  # noqa: E402
from trainer.telemetry import TelemetryServer  # noqa: E402

GAMMA = 0.995
BPTT = 16
MINIBATCH = 512
MAX_STEPS = 3000
LN2 = math.log(2.0)
KL_MIN, KL_MAX = 1e-3, 10.0
ENT_MAX = 0.05


def _value_only(model):
    """grad hook for critic-only updates: keep only the value head's gradient (the actor and the shared
    trunk stay exactly where they are; zero grads give Adam zero steps while its moments are zero)."""
    for name, p in model.named_parameters():
        if p.grad is not None and not name.startswith("value_head."):
            p.grad.zero_()


class MCTrainer:
    def __init__(self, policy, args, tele_port, out_dir, host="127.0.0.1"):
        self.policy = policy                       # bridge.Policy: acting net (policy.net), lock, layout
        self.args = args
        self.norm_warmup = int(getattr(args, "norm_warmup", 2))
        self.critic_warmup = int(getattr(args, "critic_warmup", 3))
        self.kl_target = float(getattr(args, "kl_target", 0.02))
        self.kl_gain = float(getattr(args, "kl_gain", 0.5))
        self.ent_floor = float(getattr(args, "entropy_floor", 0.5))
        self.succ_alpha = float(getattr(args, "succ_ema", 0.05))
        self.goal_min_eps = int(getattr(args, "goal_min_eps", 10))
        self.goal_lo = float(getattr(args, "goal_lo", 0.1))
        self.goal_hi = float(getattr(args, "goal_hi", 0.8))
        self.best_min_eps = int(getattr(args, "best_min_eps", 30))

        # --resume: the mc-ft checkpoint gives the acting weights (policy.net is loaded from it by the bridge),
        # the optimizer, normalizers and counters. The KL reference stays the original start checkpoint.
        self.resumed_from = None
        rp = None
        self.start_path = policy.path
        resume = getattr(args, "resume", None)
        if resume:
            rp = ckpt.load(resume)
            meta = rp.get("metadata", {}) or {}
            if meta.get("mode") != "mc-ft" or "optimizer" not in rp:
                raise ValueError(f"--resume needs an mc-ft checkpoint ({os.path.basename(resume)} is not one)")
            self.resumed_from = resume
            self.start_path = meta.get("start_checkpoint") or getattr(args, "ckpt", None)
            if not self.start_path or not os.path.exists(self.start_path):
                raise ValueError("--resume: the start checkpoint of this run is missing "
                                 f"({self.start_path}); pass --ckpt with the v6 start checkpoint")
        payload = ckpt.load_for_training(self.start_path)
        if payload.get("obs", {}).get("vec_dim") != 102:
            raise ValueError("online fine-tuning needs a v6 checkpoint (VEC_DIM 102)")
        self.start = payload
        self.res = policy.layout["res"]
        self.vec_dim = policy.layout["vec_dim"]
        self.num_blocks = policy.layout["num_blocks"]
        self.lock = threading.Lock()               # buffers, sessions, bot state (request thread + monitor)
        self.model = build_model(payload)
        self.model.load_state_dict(payload["model"], strict=False)
        if rp is not None:
            self.model.load_state_dict(rp["model"], strict=False)
        self.model.eval()
        self.ref = build_model(payload)
        self.ref.load_state_dict(payload["model"], strict=False)
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.gru = self.model.gru_hidden
        self.ppo = RecurrentPPO(self.model, lr=args.lr, gamma=GAMMA, lam=0.95, clip=args.clip,
                                ent_coef=args.ent_coef, epochs=args.epochs, bptt=BPTT,
                                minibatch_size=MINIBATCH, value_mode="mse", rnd=False,
                                weight_decay=0.0)
        self.ret_norm = ReturnNormalizer(GAMMA)
        self.kl_coef = float(args.kl_coef)
        self.T = int(args.rollout)
        self.run_id = time.strftime("mcft-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.run_dir = os.path.join(out_dir, self.run_id)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.best_dir = os.path.join(self.run_dir, "best")
        self.ckpt_every = float(args.ckpt_minutes) * 60.0

        self.env_steps = 0
        self.updates = 0
        self.episodes = 0
        self.rollouts = 0
        self.goal_eps = [0] * len(ITEMS)
        self.goal_succ = [0] * len(ITEMS)
        self.goal_ema = [None] * len(ITEMS)
        self.recent = deque(maxlen=int(getattr(args, "best_window", 200)))
        self.best_score = None
        self.best_path = None
        if rp is not None:
            if "reward_norm" in rp:
                self.ret_norm.load_state_dict(rp["reward_norm"])
            load_optimizer_state(self.ppo.opt, rp["optimizer"])
            for g in self.ppo.opt.param_groups:
                g["lr"] = float(args.lr)               # the CLI lr wins over the saved one
            c = rp.get("counters", {}) or {}
            self.env_steps = int(c.get("env_steps", 0))
            self.updates = int(c.get("updates", 0))
            self.episodes = int(c.get("episodes", 0))
            st = rp.get("mc_ft", {}) or {}
            self.rollouts = int(st.get("rollouts", self.updates + self.norm_warmup if self.updates else 0))
            self.kl_coef = float(st.get("kl_coef", self.kl_coef))
            self.ppo.ent_coef = float(st.get("ent_coef", self.ppo.ent_coef))
            if st.get("goal_eps"):
                self.goal_eps = list(st["goal_eps"])
                self.goal_succ = list(st["goal_succ"])
                self.goal_ema = list(st["goal_ema"])
            self.recent.extend(st.get("recent", []))
            self.best_score = st.get("best_score")
            self.best_path = st.get("best_path")
            if "rng" in rp:
                ckpt.restore_rng(rp["rng"])
        elif "reward_norm" in payload:
            self.ret_norm.load_state_dict(payload["reward_norm"])

        self.session = None
        self.roll = None
        self.t = 0
        self.raw = None
        self.kl = None
        self.hid = None
        self.ref_hid = None
        self.sps = 0.0
        self._sps_mark = (time.time(), 0)
        self.returns = deque(maxlen=200)
        self.cur_goal = {}
        self._team_returns = {}
        self.bot_state = {}                       # id -> last request data (for the dashboard)
        self.by_slot = {}                         # (team, slot) -> id
        self.selected = None
        self.K = 1
        self.dropped_sessions = 0
        self.last_stats = {}
        self.saved_updates = self.updates
        self.latest_ckpt = None
        self.last_ckpt_time = time.time()
        self.closing = False
        self.queue = queue.Queue()
        self.trainer_thread = threading.Thread(target=self._train_loop, name="mc-ft-trainer", daemon=True)
        self.trainer_thread.start()
        self.tele = TelemetryServer(host=host, port=tele_port)
        self.mon_thread = threading.Thread(target=self._monitor, name="mc-ft-telemetry", daemon=True)
        self.mon_thread.start()
        print(f"[mc-ft] start checkpoint {self.start_path}", flush=True)
        if self.resumed_from:
            print(f"[mc-ft] resumed {self.resumed_from}: updates {self.updates}, decisions {self.env_steps}, "
                  f"episodes {self.episodes}, phase {self.phase()}, kl_coef {self.kl_coef:g}", flush=True)
        print(f"[mc-ft] run {self.run_id}: checkpoints -> {self.ckpt_dir} (every {args.ckpt_minutes:g} min "
              f"and on train_end); lr={args.lr:g} kl_coef={self.kl_coef:g} kl_target={self.kl_target:g} "
              f"rollout={self.T} clip={args.clip:g} epochs={args.epochs} norm_warmup={self.norm_warmup} "
              f"critic_warmup={self.critic_warmup} entropy_floor={self.ent_floor:g}", flush=True)
        print(f"[mc-ft] telemetry on {host}:{self.tele.port}", flush=True)

    def phase(self):
        if self.rollouts <= self.norm_warmup:
            return "norm_warmup"
        if self.updates < self.critic_warmup:
            return "critic_warmup"
        return "ppo"

    # ------------------------------------------------------------------ requests (bridge thread)

    def _new_session(self, ids):
        if self.session is not None and self.t:
            self.dropped_sessions += 1
            print(f"[mc-ft] new session: dropped a partial rollout of {self.t} steps", flush=True)
        N = len(ids)
        self.session = {"ids": list(ids), "row": {b: i for i, b in enumerate(ids)}, "N": N, "pend": None}
        self.roll = Rollout(self.T, N, self.res, self.vec_dim, self.gru)
        self.raw = np.zeros((self.T, N), np.float32)
        self.kl = np.zeros((self.T, N), np.float32)
        self.t = 0
        self.hid = torch.zeros(N, self.gru)
        self.ref_hid = torch.zeros(N, self.gru)
        self.ret_norm.reset_envs(N)
        self.cur_goal = {}
        print(f"[mc-ft] session: {N} bots", flush=True)

    def _ready_roll(self):
        N = self.session["N"]
        self.roll = Rollout(self.T, N, self.res, self.vec_dim, self.gru)
        self.raw = np.zeros((self.T, N), np.float32)
        self.kl = np.zeros((self.T, N), np.float32)
        self.t = 0

    def step(self, req):
        """One decision tick for all training bots -> actions. Never blocks on training."""
        bots = req["bots"]
        ids = [b["id"] for b in bots]
        with self.lock:
            sess = self.session
            if (req.get("new_session") or sess is None or len(set(ids)) != len(ids)
                    or set(ids) != set(sess["ids"])):
                self._new_session(ids)
                sess = self.session
            N, res, nb = sess["N"], self.res, self.num_blocks
            img = np.zeros((N, res, res, 2), np.uint8)
            vec = np.zeros((N, self.vec_dim), np.float32)
            reward = np.zeros(N, np.float32)
            done = np.zeros(N, bool)
            order = []
            for b in bots:
                r = sess["row"][b["id"]]
                order.append(r)
                raw = base64.b64decode(b["img"])
                if len(raw) != res * res * 2:
                    raise ValueError(f"image is {len(raw)} bytes, expected {res * res * 2}")
                img[r] = np.frombuffer(raw, np.uint8).reshape(res, res, 2)
                v = np.asarray(b["vec"], np.float32)
                if v.shape[0] != self.vec_dim:
                    raise ValueError(f"vec has {v.shape[0]} floats, expected {self.vec_dim}")
                vec[r] = v
                reward[r] = float(b.get("reward", 0.0))
                done[r] = bool(b.get("done", False))
            img[..., 0] = np.minimum(img[..., 0], nb - 1)

            # 1) the previous action's transition is complete: reward and done arrived with this request
            ended_goal = {}                           # team -> goal of the episode that just ended
            for b in bots:
                team = int(b.get("team", 0))
                if b.get("done"):
                    ended_goal.setdefault(team, self.cur_goal.get(team))
            for b in bots:
                if "goal" in (b.get("info") or {}):
                    self.cur_goal[int(b.get("team", 0))] = int(b["info"]["goal"])
            p = sess["pend"]
            if p is not None:
                i = self.t
                roll = self.roll
                roll.img[i] = p["img"]
                roll.vec[i] = p["vec"]
                roll.hid[i] = p["hid"]
                roll.act[i] = p["act"]
                roll.msg[i] = p["msg"]
                roll.logp[i] = p["logp"]
                roll.val[i] = p["val"]
                roll.rew[i] = reward - self.kl_coef * p["kl"]
                roll.done[i] = done.astype(np.float32)
                self.raw[i] = reward
                self.kl[i] = p["kl"]
                self.t += 1
                self.env_steps += N
                for b in bots:                       # episode bookkeeping (dashboard + metrics)
                    self._note_info(b, ended_goal)
                for team, rets in self._team_returns.items():
                    if rets:
                        self.returns.append(float(np.mean(rets)))
                        rets.clear()
            # episodes that just ended start from a zero GRU state (acting and reference)
            if done.any():
                mask = torch.from_numpy(done)
                self.hid[mask] = 0.0
                self.ref_hid[mask] = 0.0

            # 2) act on the current observations
            h_in = self.hid.clone()
            with self.policy.lock:
                a, m, lp, v, _vi, hn, _err = act(self.policy.net, img, vec, self.hid, sample=True,
                                                 value_mode="mse")
            self.hid = hn.clone()
            lp_ref, self.ref_hid = self._ref_logp(img, vec, self.ref_hid, a, m)
            kl = (lp - lp_ref).astype(np.float32)

            # 3) a full rollout goes to the trainer; v is the bootstrap value of these observations
            if self.t == self.T:
                self.queue.put(("update", (self.roll, np.asarray(v, np.float32).copy(), self.raw, self.kl)))
                self._ready_roll()

            sess["pend"] = {"img": img.copy(), "vec": vec.copy(), "hid": h_in.numpy().copy(),
                            "act": a.copy(), "msg": m.copy(), "logp": lp.copy(),
                            "val": np.asarray(v, np.float32).copy(), "kl": kl}
            actions = []
            for b, r in zip(bots, order):
                actions.append({"id": b["id"], "action": int(a[r]), "msg": int(m[r])})
                info = b.get("info", {}) or {}
                st = self.bot_state.setdefault(b["id"], {"actions": deque(maxlen=32), "acc": 0.0})
                st.update(team=int(b.get("team", 0)), slot=int(b.get("slot", 0)), img=raw_img(img[r]),
                          vec=vec[r].tolist(), info=info, last_action=ACTIONS[int(a[r])],
                          last_reward=float(reward[r]), kl=float(kl[r]))
                if "region" in info:
                    st["region"] = info["region"]
                st["actions"].append(ACTIONS[int(a[r])])
                st["acc"] = 0.0 if bool(b.get("done", False)) else st["acc"] + float(reward[r])
                self.by_slot[(st["team"], st["slot"])] = b["id"]
                self.K = max(self.K, st["slot"] + 1)
            reply = {"actions": actions, "select": self.selected, "stats": self.stats(),
                     "goal_weights": self._goal_weights()}
        return reply

    def _note_info(self, b, ended_goal):
        """Episode bookkeeping for one bot of a request (ended_goal: team -> goal of the episode that ended)."""
        info = b.get("info") or {}
        team, slot = int(b.get("team", 0)), int(b.get("slot", 0))
        if b.get("done"):
            if slot == 0:
                goal = ended_goal.get(team)
                if goal is not None:
                    ok = 1 if info.get("success") else 0
                    self.goal_eps[goal] += 1
                    self.goal_succ[goal] += ok
                    prev = self.goal_ema[goal]
                    self.goal_ema[goal] = float(ok) if prev is None else prev + self.succ_alpha * (ok - prev)
                    self.recent.append(bool(ok))
                self.episodes += 1
            self._team_returns.setdefault(team, []).append(float(info.get("ep_ret", 0.0)))

    def _goal_weights(self):
        """{item: w}: 1.0 inside the success band [goal_lo, goal_hi], 0.5 outside (too hard or mastered),
        1.0 for goals with fewer than goal_min_eps episodes; mean 1 over the goals with enough episodes."""
        raw = {}
        for g in range(len(ITEMS)):
            if self.goal_eps[g] < self.goal_min_eps or self.goal_ema[g] is None:
                continue
            e = self.goal_ema[g]
            raw[g] = 1.0 if self.goal_lo <= e <= self.goal_hi else 0.5
        mean = sum(raw.values()) / len(raw) if raw else 1.0
        return {ITEMS[g]: round(raw[g] / mean, 4) if g in raw else 1.0 for g in range(len(ITEMS))}

    @torch.no_grad()
    def _ref_logp(self, img, vec, h, a, m):
        feat = self.ref.encode(torch.from_numpy(img), torch.from_numpy(vec))
        out, hn = self.ref.recur(feat, h)
        lg = torch.log_softmax(self.ref.policy_head(out), -1).gather(-1, torch.from_numpy(a)[:, None])[:, 0]
        ml = torch.log_softmax(self.ref.msg_head(out), -1).gather(-1, torch.from_numpy(m)[:, None])[:, 0]
        return (lg + ml).numpy().astype(np.float32), hn.clone()

    def stats(self):
        return {"updates": self.updates, "env_steps": self.env_steps, "episodes": self.episodes,
                "rollouts": self.rollouts, "phase": self.phase(), "kl_coef": self.kl_coef,
                "latest_checkpoint": self.latest_ckpt, "best_checkpoint": self.best_path,
                "kl_to_start": self.last_stats.get("kl_to_start"), "best_success": self.best_score}

    def end(self, reason):
        """train_end: drop the partial rollout, save after the queued updates."""
        with self.lock:
            if self.session is not None and self.t:
                self.dropped_sessions += 1
            self.session = None
        self.queue.put(("save", f"train_end: {reason}"))
        print(f"[mc-ft] train_end ({reason}): checkpoint queued", flush=True)

    # ------------------------------------------------------------------ trainer thread

    def _train_loop(self):
        while True:
            try:
                kind, item = self.queue.get(timeout=1.0)
            except queue.Empty:
                kind, item = None, None
            try:
                if kind == "update":
                    self._update(*item)
                elif kind == "save":
                    self._save(item)
                elif kind == "stop":
                    self._save("bridge shutdown")
                    return
            except Exception:  # never let the trainer thread die silently
                traceback.print_exc()
                print("[mc-ft] update failed; the acting policy keeps the last good weights", flush=True)
            if kind is None and self.updates != self.saved_updates and \
                    time.time() - self.last_ckpt_time >= self.ckpt_every:
                self._save("periodic")

    def _update(self, roll, last_val, raw, kl):
        t0 = time.time()
        pen = roll.rew.astype(np.float64)                       # reward - kl penalty (what PPO sees)
        self.ret_norm.update(pen, roll.done)
        self.rollouts += 1
        if self.rollouts <= self.norm_warmup:                  # 1) normalizer warmup: stats only
            self.last_stats = dict(self.last_stats, kl_to_start=float(kl.mean()), kl_max=float(kl.max()),
                                   raw_step=float(raw.mean()), update_s=time.time() - t0)
            print(f"[mc-ft] norm warmup {self.rollouts}/{self.norm_warmup} (reward stats only) | decisions "
                  f"{self.env_steps} | return scale {math.sqrt(self.ret_norm.var):.3f} | "
                  f"step reward {raw.mean():+.4f}", flush=True)
            return
        scaled = self.ret_norm.scale(pen).astype(np.float32)
        critic_only = self.updates < self.critic_warmup         # 2) critic-only warmup updates
        if critic_only:
            s = self.ppo.update(roll, last_val, rew=scaled, grad_hook=_value_only)
        else:                                                    # 3) full PPO
            s = self.ppo.update(roll, last_val, rew=scaled)
        with self.policy.lock:                                  # acting copy = trained weights
            self.policy.net.load_state_dict(self.model.state_dict())
            self.policy.net.eval()
        self.updates += 1
        if not critic_only:
            self._adapt(kl, s)
        dt = max(1e-3, time.time() - t0)
        self.last_stats = dict(s, kl_to_start=float(kl.mean()), kl_max=float(kl.max()),
                               raw_step=float(raw.mean()), update_s=dt, phase="critic_warmup" if critic_only
                               else "ppo", kl_coef=self.kl_coef, ent_coef=self.ppo.ent_coef)
        with self.lock:
            mean_ret = float(np.mean(self.returns)) if self.returns else float("nan")
            succ = sum(self.goal_succ) / max(1, sum(self.goal_eps))
            recent = (sum(self.recent) / len(self.recent)) if self.recent else float("nan")
            recent_n = len(self.recent)
        print(f"[mc-ft] {'critic' if critic_only else 'ppo'} update {self.updates} | decisions {self.env_steps} | "
              f"episodes {self.episodes} | return {mean_ret:+.3f} | step reward {raw.mean():+.4f} | "
              f"success {succ:.2f} (recent {recent:.2f}/{recent_n}) | "
              f"pi_loss {s['policy_loss']:+.4f} v_loss {s['value_loss']:.4f} entropy {s['entropy']:.3f} "
              f"(floor {self.ent_floor:g}, ent_coef {self.ppo.ent_coef:.4f}) | "
              f"kl_to_start {kl.mean():.4f} (max {kl.max():.3f}, coef {self.kl_coef:.4f}) "
              f"ppo_kl {s['approx_kl']:.4f} | update {dt:.2f}s", flush=True)
        if not critic_only and recent_n >= self.best_min_eps and not math.isnan(recent):
            if self.best_score is None or recent > self.best_score:
                self._save_best(recent)

    def _adapt(self, kl, s):
        """KL-to-start coefficient (log-proportional, step clipped to x2 or /2) and entropy floor."""
        kl_mean = max(float(kl.mean()), 1e-4)
        err = math.log(kl_mean / self.kl_target)
        factor = math.exp(max(-LN2, min(LN2, self.kl_gain * err)))
        self.kl_coef = min(KL_MAX, max(KL_MIN, self.kl_coef * factor))
        ent = s.get("entropy")
        if ent is not None:
            base = max(float(self.args.ent_coef), 1e-3)
            cur = self.ppo.ent_coef
            if ent < self.ent_floor:
                cur = min(ENT_MAX, max(cur, base) * 1.5)
            elif ent > 1.5 * self.ent_floor and cur > base:
                cur = max(base, cur / 1.5)
            self.ppo.ent_coef = cur

    def _payload(self, reason):
        with self.lock:
            goal_state = {"goal_eps": list(self.goal_eps), "goal_succ": list(self.goal_succ),
                          "goal_ema": list(self.goal_ema)}
            recent = list(self.recent)
        payload = {
            "run_id": self.run_id,
            "obs": self.start["obs"],
            "model": self.model.state_dict(),
            "optimizer": self.ppo.opt.state_dict(),
            "config": dict(self.start.get("config", {}) or {}),
            "counters": {"env_steps": int(self.env_steps), "updates": int(self.updates),
                         "episodes": int(self.episodes)},
            "reward_norm": self.ret_norm.state_dict(),
            "rnd_norm": self.start.get("rnd_norm", {"mean": 0.0, "var": 1.0, "count": 1e-4}),
            "rng": ckpt.rng_state(),
            "mc_ft": dict(goal_state, rollouts=int(self.rollouts), kl_coef=float(self.kl_coef),
                          ent_coef=float(self.ppo.ent_coef), recent=recent,
                          best_score=self.best_score, best_path=self.best_path, phase=self.phase()),
            "metadata": {"reason": reason, "run_id": self.run_id, "mode": "mc-ft",
                         "start_checkpoint": self.start_path, "resumed_from": self.resumed_from,
                         "lr": self.args.lr, "kl_coef": self.kl_coef, "rollout": self.T,
                         "source": "minecraft online PPO"},
        }
        payload["config"]["mc_finetune"] = {"lr": self.args.lr, "kl_coef": self.kl_coef, "rollout": self.T,
                                            "clip": self.args.clip, "epochs": self.args.epochs,
                                            "kl_target": self.kl_target, "norm_warmup": self.norm_warmup,
                                            "critic_warmup": self.critic_warmup}
        return payload

    def _save(self, reason):
        path = ckpt.save(self.ckpt_dir, self._payload(reason))
        self.latest_ckpt = path
        self.saved_updates = self.updates
        self.last_ckpt_time = time.time()
        print(f"[mc-ft] checkpoint ({reason}): {path}", flush=True)

    def _save_best(self, score):
        """Best by recent success: one file under <run>/best/ (the previous best is removed)."""
        payload = self._payload(f"best by success {score:.3f}")
        payload["metadata"]["best_success"] = float(score)
        old = self.best_path
        path = ckpt.save(self.best_dir, payload)
        self.best_score = float(score)
        self.best_path = path
        if old and old != path and os.path.exists(old):
            try:
                os.remove(old)
            except OSError:
                pass
        print(f"[mc-ft] best checkpoint (recent success {score:.3f}): {path}", flush=True)

    def close(self):
        """Bridge shutdown: queued updates run, then a final checkpoint."""
        self.closing = True
        self.queue.put(("stop", None))
        self.trainer_thread.join(timeout=600)
        self.tele.close()

    # ------------------------------------------------------------------ telemetry (127.0.0.1:7777)

    def _monitor(self):
        last_metrics = 0.0
        last_snap = 0.0
        while not self.closing:
            time.sleep(0.05)
            try:
                self._handle_commands()
                now = time.time()
                if now - last_metrics >= 0.5:
                    last_metrics = now
                    self.tele.broadcast("metrics", self._metrics(now))
                if now - last_snap >= 0.25 and self.tele.any_subscribed("snapshot"):
                    last_snap = now
                    snap = self._snapshot()
                    if snap is not None:
                        self.tele.broadcast("snapshot", snap, need="snapshot")
            except Exception:
                traceback.print_exc()

    def _handle_commands(self):
        for msg in self.tele.poll():
            cmd = msg.get("cmd")
            cid = msg.get("_client")
            if cmd == "_connect":
                self.tele.send_to(cid, "hello", self._hello())
                self.tele.send_to(cid, "model", self.model.describe())
                self.tele.send_to(cid, "metrics", self._metrics(time.time()))
            elif cmd == "_disconnect":
                continue
            elif cmd == "subscribe":
                self.tele.set_subscription(cid, snapshot=msg.get("snapshot"), nn=msg.get("nn"))
                self.tele.send_to(cid, "ack", {"id": msg.get("id"), "ok": True, "msg": "subscribed"})
            elif cmd == "select_env":
                env_id = int(msg.get("env_id", -1))
                with self.lock:
                    team, slot = divmod(env_id, max(1, self.K))
                    name = self.by_slot.get((team, slot))
                    self.selected = name
                ok = name is not None
                self.tele.send_to(cid, "ack", {"id": msg.get("id"), "ok": ok,
                                               "msg": f"selected {name}" if ok else "no such bot"})
            else:
                self.tele.send_to(cid, "ack", {"id": msg.get("id"), "ok": False,
                                               "msg": "online fine-tune: control it in game with /aibot train"})

    def _hello(self):
        return {"schema": 1, "run_id": self.run_id, "state": "running",
                "config": self._config(), "world_size": [128, 64, 128],
                "actions": list(ACTIONS), "items": list(ITEMS), "blocks": list(BLOCKS)}

    def _config(self):
        return {"mode": "mc-ft", "lr": self.args.lr, "kl_coef": self.kl_coef, "kl_target": self.kl_target,
                "rollout": self.T, "clip": self.args.clip, "epochs": self.args.epochs, "agents": self.K,
                "res": self.res, "start_checkpoint": self.start_path, "resumed_from": self.resumed_from,
                "norm_warmup": self.norm_warmup, "critic_warmup": self.critic_warmup,
                "entropy_floor": self.ent_floor, "ent_coef": self.args.ent_coef}

    def _metrics(self, now):
        s = self.last_stats
        with self.lock:
            teams = len({t for (t, _s) in self.by_slot})
            n_eps = max(1, self.episodes)
            weights = self._goal_weights()
            ema = {ITEMS[g]: self.goal_ema[g] for g in range(len(ITEMS)) if self.goal_ema[g] is not None}
            recent = (float(sum(self.recent)) / len(self.recent)) if self.recent else None
        t0, n0 = self._sps_mark
        if now - t0 >= 2.0:
            self.sps = (self.env_steps - n0) / (now - t0)
            self._sps_mark = (now, self.env_steps)
        success = {ITEMS[g]: self.goal_succ[g] / self.goal_eps[g] for g in range(len(ITEMS)) if self.goal_eps[g]}
        mean_ret = float(np.mean(self.returns)) if self.returns else None
        return {
            "t": now, "state": "running" if self.session is not None else "idle", "mode": "live",
            "env_steps": self.env_steps, "target_steps": None, "sps": self.sps,
            "decisions_per_s": self.sps, "agents": self.K,
            "episodes": self.episodes, "avg_ep_len": None, "avg_reward": mean_ret,
            "success_rate": success, "success_ema": ema, "success_recent": recent,
            "goal_weights": weights, "updates": self.updates, "rollouts": self.rollouts,
            "phase": self.phase(),
            "policy_loss": s.get("policy_loss"), "value_loss": s.get("value_loss"),
            "entropy": s.get("entropy"), "approx_kl": s.get("approx_kl"),
            "explained_var": s.get("explained_var"), "kl_to_start": s.get("kl_to_start"),
            "kl_coef": self.kl_coef, "kl_target": self.kl_target, "kl_max": s.get("kl_max"),
            "ent_coef": self.ppo.ent_coef, "entropy_floor": self.ent_floor,
            "msg_entropy": s.get("msg_entropy"), "step_reward": s.get("raw_step"),
            "cpu_percent": None, "rss_mb": None, "mem_warning": None, "elapsed_s": None, "eta_s": None,
            "latest_checkpoint": self.latest_ckpt, "best_checkpoint": self.best_path,
            "best_success": self.best_score, "resumed_from": self.resumed_from,
            "curriculum_stage": "mc fine-tune", "curriculum_task": None,
            "lr": float(self.args.lr), "reward_scale": float(1.0 / np.sqrt(self.ret_norm.var + 1e-8)),
            "num_envs": teams, "population": None, "generation": None, "config": self._config(),
            "episodes_seen": n_eps,
        }

    def _snapshot(self):
        with self.lock:
            name = self.selected
            st = self.bot_state.get(name) if name else None
            if st is None:
                return None
            info = st["info"]
            team = st.get("team", 0)
            members = []
            for other, s2 in self.bot_state.items():
                if s2.get("team") != team:
                    continue
                i2 = s2["info"]
                members.append({"agent": s2.get("slot", 0), "x": i2.get("pos", [0, 0, 0])[0],
                                "y": i2.get("pos", [0, 0, 0])[1], "z": i2.get("pos", [0, 0, 0])[2],
                                "yaw": i2.get("yaw", 0.0), "pitch": i2.get("pitch", 0.0),
                                "health": i2.get("hp", 0.0), "last_action": s2.get("last_action"),
                                "msg": int(i2.get("msg", 0)), "inventory": i2.get("inv", {})})
            pos = info.get("pos", [0.0, 0.0, 0.0])
            region = st.get("region")
            return {
                "env_id": team * max(1, self.K) + int(st.get("slot", 0)), "world_id": team,
                "agent_id": int(st.get("slot", 0)), "agents_per_world": max(1, self.K),
                "episode": int(info.get("episode", 0)), "seed": None, "step": int(info.get("step", 0)),
                "player": {"x": pos[0], "y": pos[1], "z": pos[2], "vx": 0.0, "vy": 0.0, "vz": 0.0,
                           "yaw": float(info.get("yaw", 0.0)), "pitch": float(info.get("pitch", 0.0)),
                           "health": float(info.get("hp", 0.0)), "on_ground": False},
                "last_action": st.get("last_action"), "last_reward": st.get("last_reward", 0.0),
                "ep_reward": st.get("acc", 0.0), "task": ITEMS[int(info.get("goal", 0))],
                "target": None, "inventory": info.get("inv", {}),
                "action_history": list(st["actions"]), "team": sorted(members, key=lambda d: d["agent"]),
                "chests": [], "chat": [],
                "origin": region["origin"] if region else None,
                "size": region["size"] if region else None,
                "blocks": region["blocks"] if region else None,
                "blocks_available": bool(region),
                "obs_res": self.res, "obs": base64.b64encode(st["img"]).decode("ascii"),
            }


def raw_img(arr):
    return np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
