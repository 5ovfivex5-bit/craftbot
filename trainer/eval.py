"""Per-goal evaluation of a v2 checkpoint on fresh seeds (sampled actions by default).

  python -m trainer.eval runs/<run>/checkpoints/ckpt_....pt --episodes 64 --goal all
  python -m trainer.eval CKPT --goal diamond --episodes 200 --seed 42 --greedy

v6: an episode is a WORLD episode (its K agents share one world and end together); each world episode
is counted once. --agents defaults to the checkpoint's agents-per-world.

For each goal: `episodes` episodes in which that goal is the only one sampled. By default no start kit
is given (`--kits` enables the goal's kit, i.e. the curriculum's helper, for diagnostics only), so the
success is the real from-scratch skill. Reports the goal
success rate (ach bit 31) with a 95% binomial interval, and the decisions-to-goal (episode length
of the successful episodes: mean and median). Worlds come from seeds that training never used
(default base 900001, one block of seeds per goal), so the numbers are not training-set numbers.
"""
import argparse
import math
import sys
import time

import numpy as np
import torch

from . import checkpoint as ckpt
from .curriculum import GOAL_ORDER, KIT_TABLE
import craftsim

from .envs import ITEMS, NUM_ACTIONS, NUM_BLOCKS, NUM_ITEMS, NUM_MSGS, RES, V7, VEC_DIM, SimAdapter
from .model import MineMindPolicy
from .ppo import act_by_slot, policy_act

GOAL_BIT = 31
DEFAULT_SEED = 900001


def build_model(payload):
    """Policy with the layout of a checkpoint (its observation signature and model config)."""
    cfg = payload.get("config", {})
    obs = payload.get("obs", {})
    return MineMindPolicy(res=obs.get("res", RES), vec_dim=obs.get("vec_dim", VEC_DIM),
                          num_actions=obs.get("num_actions", NUM_ACTIONS),
                          num_blocks=obs.get("num_blocks", NUM_BLOCKS),
                          size=cfg.get("model_size", "small"), resblock=cfg.get("resblock", True),
                          num_msgs=obs.get("num_msgs", NUM_MSGS),
                          num_recipe_classes=obs.get("num_recipe_classes", 0),
                          num_item_classes=obs.get("num_item_classes", 0))


def load_policy(path):
    payload = ckpt.load_for_training(path)
    model = build_model(payload)
    model.load_state_dict(payload["model"], strict=False)   # unused v4/v5 heads may be absent
    model.eval()
    return model, payload


def load_policies(path, policy=None):
    """PROTOCOL 10.3. policy 'best' -> one model (the checkpoint's `model`, the best member); 'all' -> the
    list of P member models, slot k driven by member k; an int slot -> that member in every slot (a list
    of K copies). Default: 'all' for a population checkpoint, else 'best'.
    Returns (models_or_model, payload, resolved_policy)."""
    payload = ckpt.load_for_training(path)
    best = build_model(payload)
    best.load_state_dict(payload["model"], strict=False)
    best.eval()
    members = payload.get("models")
    if policy is None:
        policy = "all" if members is not None else "best"
    if policy == "best" or (members is None and policy == "all"):
        return best, payload, "best"
    if members is None:
        raise ValueError(f"--policy {policy} needs a population checkpoint (this one has a single policy)")
    models = []
    for sd in members:
        m = build_model(payload)
        m.load_state_dict(sd, strict=False)
        m.eval()
        models.append(m)
    if policy == "all":
        return models, payload, "all"
    slot = int(policy)
    if not (0 <= slot < len(models)):
        raise ValueError(f"--policy slot must be in [0, {len(models)})")
    return [models[slot]] * len(models), payload, f"slot{slot}"


def run_worlds(model, sim, episodes_per_world, sample=True):
    """`model` is one policy, or a list of K policies (slot k of every world is driven by list[k])."""
    """Steps `sim` (W worlds x K agents, policy acting for every agent) until every world has finished
    `episodes_per_world` episodes. Episodes are WORLD episodes: counted once per world, reward is the
    mean over the world's agents. Returns one list of dicts per world:
    {"reward", "length", "mask" (team achievement mask), "goal"}."""
    W, K, N = sim.worlds, sim.agents, sim.n
    img, vec = sim.reset()
    h = (model[0] if isinstance(model, (list, tuple)) else model).zero_hidden(N)
    ep_r = np.zeros(N, np.float64)
    ep_l = np.zeros(W, np.int64)
    goal = [sim.goal(w * K) for w in range(W)]
    task = [sim.task(w * K) if V7 else -1 for w in range(W)]   # v7: the task of each world's episode
    out = [[] for _ in range(W)]
    done_w = np.zeros(W, np.int64)
    while done_w.min() < episodes_per_world:
        if isinstance(model, (list, tuple)):
            a, m, _, _, _, hn, _ = act_by_slot(list(model), img, vec, h, sample=sample)
        else:
            a, m, _, _, hn = policy_act(model, img, vec, h, sample=sample)
        img, vec, rew, done, ach = sim.step(a.astype(np.int32), m.astype(np.int32))
        ep_r += rew
        ep_l += 1
        world_done = np.asarray(done, bool).reshape(W, K)[:, 0]
        for w in np.nonzero(world_done)[0]:
            rows = slice(w * K, (w + 1) * K)
            if done_w[w] < episodes_per_world:
                out[w].append({"reward": float(ep_r[rows].mean()), "length": int(ep_l[w]),
                               "mask": int(ach[w * K]), "goal": goal[w],
                               "task": craftsim.TASK_NAMES[task[w]] if V7 else None})
                done_w[w] += 1
            ep_r[rows] = 0.0
            ep_l[w] = 0
            goal[w] = sim.goal(w * K)
            task[w] = sim.task(w * K) if V7 else -1
        h = hn * torch.from_numpy(1.0 - np.asarray(done, np.float32))[:, None]
    return out


def evaluate_goal(model, goal_idx, episodes, seed, envs=64, agents=1, sample=True, threads=4,
                  kits=False):
    """`episodes` world episodes with goal `goal_idx`, split over `envs` parallel worlds of `agents`
    agents each. Success = the team achieved the goal (bit 31)."""
    k = max(1, min(int(envs), int(episodes)))
    quota = math.ceil(episodes / k)          # world episodes per world
    sim = SimAdapter(k, seed=int(seed), threads=threads, agents=agents)
    w = np.zeros(NUM_ITEMS, np.float32)
    w[goal_idx] = 1.0
    sim.set_goal_weights(w)
    # no kit unless asked for: the sim's default kit probability is 0, set it explicitly anyway
    kit_p = np.zeros(NUM_ITEMS, np.float32)
    if kits:
        kit_p[goal_idx] = 1.0
    sim.set_start_kits(KIT_TABLE, kit_p)
    sim.set_novelty(0.0)   # hand-crafted novelty is never part of the from-scratch measurement
    sim.reset()            # goal weights only apply to newly generated worlds
    assert all(sim.goal(i) == goal_idx for i in range(sim.n)), "goal weights did not pin the goal"
    torch.manual_seed(int(seed))
    worlds = run_worlds(model, sim, quota, sample=sample)
    sim.close()
    results = [(bool((e["mask"] >> GOAL_BIT) & 1), e["length"]) for ep in worlds for e in ep][:episodes]
    succ = [n for ok, n in results if ok]
    p = len(succ) / len(results)
    se = math.sqrt(max(p * (1 - p), 1e-12) / len(results))
    return {
        "episodes": len(results), "success": p, "ci95": 1.96 * se,
        "decisions_mean": float(np.mean(succ)) if succ else None,
        "decisions_median": float(np.median(succ)) if succ else None,
    }


def evaluate_tasks(model, task_names, episodes, seed, envs=64, agents=1, sample=True, threads=4):
    """v7: episodes of the given tasks. Each world draws from the weights set here (uniform over
    `task_names`; a single name gives that task only). `episodes` per world-quota: every world finishes
    ceil(episodes / envs) episodes. Returns {task: {episodes, success, ci95, decisions_mean, decisions_median}}
    over the episodes that ran that task (pooled over the group's draws)."""
    if not V7:
        raise ValueError("task evaluation needs the v7 observation (run with --obs v7)")
    k = max(1, min(int(envs), int(episodes)))
    quota = math.ceil(episodes / k)
    sim = SimAdapter(k, seed=int(seed), threads=threads, agents=agents)
    idx = [craftsim.TASK_NAMES.index(t) for t in task_names]
    w = np.zeros(craftsim.NUM_TASKS, np.float32)
    w[idx] = 1.0 / len(idx)
    for world in range(k):
        sim.set_task_weights(world, w)
    sim.reset()
    torch.manual_seed(int(seed))
    worlds = run_worlds(model, sim, quota, sample=sample)
    sim.close()
    by = {}
    for ep in worlds:
        for e in ep:
            by.setdefault(e["task"], []).append(e)
    out = {}
    for t in task_names:
        eps = by.get(t, [])
        if not eps:
            continue
        succ = [e["length"] for e in eps if (e["mask"] >> GOAL_BIT) & 1]
        p = len(succ) / len(eps)
        se = math.sqrt(max(p * (1 - p), 1e-12) / len(eps))
        out[t] = {"episodes": len(eps), "success": p, "ci95": 1.96 * se,
                  "decisions_mean": float(np.mean(succ)) if succ else None,
                  "decisions_median": float(np.median(succ)) if succ else None}
    return out


def format_tasks(rows, meta):
    head = f"{'task':<30} {'family':<9} {'episodes':>8} {'success':>16} {'decisions-to-success':>24}"
    lines = [f"checkpoint: {meta['path']}",
             f"env_steps={meta['env_steps']:,} updates={meta['updates']} sampling={meta['sampling']} "
             f"seeds={meta['seed']}+ (fresh)", head, "-" * len(head)]
    for name, r in rows:
        fam = craftsim.FAMILIES[craftsim.TASKS[craftsim.TASK_NAMES.index(name)]["family"]]
        succ = f"{100 * r['success']:6.1f}% +-{100 * r['ci95']:4.1f}"
        dec = "-" if r["decisions_mean"] is None else f"mean {r['decisions_mean']:7.0f}  median {r['decisions_median']:6.0f}"
        lines.append(f"{name:<30} {fam:<9} {r['episodes']:>8} {succ:>16} {dec:>24}")
    if rows:
        lines.append("-" * len(head))
        lines.append(f"{'mean over tasks':<30} {'':<9} {'':>8} {100 * np.mean([r['success'] for _, r in rows]):>15.1f}%")
    return "\n".join(lines)


def format_table(rows, meta):
    head = (f"{'goal':<16} {'episodes':>8} {'success':>16} {'decisions-to-goal':>26}")
    lines = [f"checkpoint: {meta['path']}",
             f"env_steps={meta['env_steps']:,} updates={meta['updates']} sampling={meta['sampling']} "
             f"seeds={meta['seed']}+ (fresh, per goal)", head, "-" * len(head)]
    for name, r in rows:
        if r is None:
            continue
        succ = f"{100 * r['success']:6.1f}% ±{100 * r['ci95']:4.1f}"
        if r["decisions_mean"] is None:
            dec = "-"
        else:
            dec = f"mean {r['decisions_mean']:7.0f}  median {r['decisions_median']:6.0f}"
        lines.append(f"{name:<16} {r['episodes']:>8} {succ:>16} {dec:>26}")
    done = [r for _, r in rows if r is not None]
    if done:
        mean_succ = np.mean([r["success"] for r in done])
        lines.append("-" * len(head))
        lines.append(f"{'mean over goals':<16} {'':>8} {100 * mean_succ:>15.1f}%")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m trainer.eval",
                                description="per-goal success and decisions-to-goal for a checkpoint")
    p.add_argument("ckpt")
    p.add_argument("--episodes", type=int, default=64, help="episodes per goal")
    p.add_argument("--goal", default="all", help="an ITEMS name, or 'all' (the curriculum goals)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base seed for the fresh worlds")
    p.add_argument("--envs", type=int, default=64, help="parallel worlds per goal")
    p.add_argument("--agents", type=int, default=None,
                   help="agents per world (default: the checkpoint's agents, 1 for older configs)")
    p.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    p.add_argument("--threads", type=int, default=4, help="sim OpenMP threads")
    p.add_argument("--kits", action="store_true", help="start each episode with the goal's kit (diagnostic)")
    p.add_argument("--policy", default=None,
                   help="population checkpoints: 'all' (default; slot k driven by its own policy), 'best' "
                        "(the best policy in every slot) or a slot index (that policy in every slot)")
    p.add_argument("--obs", choices=("v6", "v7"), default=None,
                   help="must match the checkpoint's observation version (read before the sim is imported)")
    p.add_argument("--task", default=None, help="v7: a task name (or 'all', the default)")
    p.add_argument("--family", default=None, help="v7: evaluate a family (uniform over its tasks)")
    args = p.parse_args(argv)
    if args.episodes < 1:
        p.error("--episodes must be >= 1")
    if args.obs is not None and args.obs != ("v7" if V7 else "v6"):
        p.error(f"--obs {args.obs} but this process runs obs {'v7' if V7 else 'v6'}")

    if V7:
        torch.set_num_threads(max(1, torch.get_num_threads()))
        model, payload, policy = load_policies(args.ckpt, args.policy)
        agents = args.agents if args.agents is not None else int(payload.get("config", {}).get("agents", 1))
        if isinstance(model, list) and len(model) != agents:
            p.error(f"population checkpoint has {len(model)} policies but {agents} agents per world")
        names = list(craftsim.TASK_NAMES)
        if args.family:
            fams = list(craftsim.FAMILIES)
            if args.family not in fams:
                p.error(f"--family must be one of {fams}")
            groups = [[n for n in names if craftsim.FAMILIES[craftsim.TASKS[names.index(n)]["family"]] == args.family]]
            label = args.family
        elif args.task and args.task != "all":
            if args.task not in names:
                p.error("--task must be a task name (see data/tasks/tasks.json)")
            groups = [[args.task]]
            label = args.task
        else:
            groups = [names]
            label = "all"
        rows = []
        t0 = time.time()
        for tasks in groups:
            res = evaluate_tasks(model, tasks, args.episodes, args.seed, envs=args.envs, agents=agents,
                                 sample=not args.greedy, threads=args.threads)
            rows.extend(sorted(res.items(), key=lambda kv: names.index(kv[0])))
            print(f"  {label}: {len(res)} tasks ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        meta = {"path": args.ckpt, "env_steps": payload["counters"]["env_steps"],
                "updates": payload["counters"]["updates"],
                "sampling": ("greedy" if args.greedy else "sampled") + f" agents={agents} policy={policy}"
                + f" scope={label}", "seed": args.seed}
        print(format_tasks(rows, meta))
        return 0

    if args.goal == "all":
        goals = list(GOAL_ORDER)
    elif args.goal in ITEMS and args.goal != "dirt":
        goals = [args.goal]
    else:
        p.error(f"--goal must be 'all' or one of {[g for g in ITEMS if g != 'dirt']}")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, payload, policy = load_policies(args.ckpt, args.policy)
    agents = args.agents if args.agents is not None else int(payload.get("config", {}).get("agents", 1))
    if isinstance(model, list) and len(model) != agents:
        p.error(f"population checkpoint has {len(model)} policies but {agents} agents per world")
    t0 = time.time()
    rows = []
    for gi, name in enumerate(goals):
        gidx = ITEMS.index(name)
        r = evaluate_goal(model, gidx, args.episodes, args.seed + 1000 * gidx, envs=args.envs,
                          agents=agents, sample=not args.greedy, threads=args.threads, kits=args.kits)
        rows.append((name, r))
        print(f"  {name:<16} success {100 * r['success']:5.1f}%  episodes={r['episodes']}  "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    meta = {"path": args.ckpt, "env_steps": payload["counters"]["env_steps"],
            "updates": payload["counters"]["updates"],
            "sampling": ("greedy" if args.greedy else "sampled") + ("+kits" if args.kits else "")
            + f" agents={agents} policy={policy}",
            "seed": args.seed}
    print(format_table(rows, meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
