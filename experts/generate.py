"""CLI: run the expert skills in the sim and write a demonstration dataset (PROTOCOL 14.2).

    PYTHONPATH=. python -m experts.generate --tasks all|a,b --episodes-per-task N --worlds W --agents K \
        --out data/demos/<run> [--seed S] [--max-steps 3000] [--shard-rows 20000] [--no-compress]

Reports per-task expert success (sim success of each episode) and decisions/s (agent decisions per wall second).
"""
import argparse
import datetime
import os
import sys

import craftsim as cs
from experts.record import ShardWriter
from experts.runner import Runner
from experts.skills import SKILLS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default="all", help="'all' or a comma list of task names with a registered skill")
    p.add_argument("--episodes-per-task", type=int, default=32)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--out", default=None, help="output directory (default data/demos/expert-<timestamp>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=3000, help="sim episode cap (task max_steps still applies)")
    p.add_argument("--shard-rows", type=int, default=20000)
    p.add_argument("--compress", dest="compress", action="store_true", default=True,
                   help="savez_compressed (default on; use --no-compress for plain npz)")
    p.add_argument("--no-compress", dest="compress", action="store_false", help="plain np.savez (faster, larger)")
    p.add_argument("--threads", type=int, default=4,
                   help="OpenMP threads of the sim (0 = default; 4 measured fastest on the 8-CPU dev box)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.tasks == "all":
        tasks = sorted(SKILLS)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in SKILLS:
            sys.exit(f"no skill registered for {t!r}; registered: {', '.join(sorted(SKILLS))}")
    out = args.out or os.path.join("data", "demos", "expert-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    from craftsim import CraftSim
    sim = CraftSim(num_envs=args.worlds, agents=args.agents, seed=args.seed, res=64,
                   max_steps=args.max_steps, threads=args.threads, action_mode="v7")
    writer = ShardWriter(out, shard_rows=args.shard_rows, compress=args.compress)
    runner = Runner(sim, SKILLS, tasks, args.episodes_per_task, seed=args.seed, writer=writer,
                    log=lambda msg: print("  !", msg, file=sys.stderr))
    runner.run()
    summ = runner.summary()
    meta = writer.close(extra={"run": {"tasks": tasks, "episodes_per_task": args.episodes_per_task,
                                       "worlds": args.worlds, "agents": args.agents, "seed": args.seed,
                                       "max_steps": args.max_steps, "threads": args.threads, "decisions": summ["decisions"],
                                       "wall_s": round(summ["wall_s"], 2),
                                       "decisions_per_s": round(summ["decisions_per_s"], 1),
                                       "errors": runner.errors[:20]}})
    print(f"out: {out}  shards: {len(meta['shards'])}  rows: {meta['total_rows']}")
    print(f"{'task':28s} {'episodes':>8s} {'success':>8s} {'rate':>7s} {'decisions':>10s}")
    for t in tasks:
        s = summ["tasks"][t]
        print(f"{t:28s} {s['episodes']:8d} {s['successes']:8d} {100 * s['success_rate']:6.1f}% {s['decisions']:10d}")
    print(f"decisions: {summ['decisions']}  wall: {summ['wall_s']:.1f}s  decisions/s: {summ['decisions_per_s']:.1f}"
          f"  (worlds={args.worlds} agents={args.agents})  early episode ends: {runner.early_ends}")
    if runner.errors:
        print(f"skill errors: {len(runner.errors)} (first: {runner.errors[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
