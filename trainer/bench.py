"""Throughput / overhead benchmark for the v2 (goal-conditioned) trainer.

  python -m trainer.bench                           # sim-only and full-training sps vs num_envs + memory
  python -m trainer.bench --envs 64 --profile       # collect vs update time split at one num_envs
  python -m trainer.bench --impact                  # training sps with / without a subscribed client

Full training = collection (batched inference + sim.step) + PPO update, measured with the same
Trainer code the CLI uses (goal curriculum, return normalization and schedules included, no server).
Sim geometry (VEC_DIM, NUM_BLOCKS, WORLD_SIZE, goal offset) is read from craftsim, never hardcoded.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import psutil
import torch

from . import envs as E
from .envs import RES, SimAdapter, VEC_DIM
from .ppo import Rollout
from .run import DEFAULT_RUNS, ROOT, Trainer, parse_args

NUM_ENVS = [1, 4, 16, 64, 256]


def sim_only(n, seconds, threads):
    sim = SimAdapter(n, seed=1, threads=threads)
    sim.reset()
    rng = np.random.default_rng(0)
    for _ in range(5):
        sim.step(rng.integers(0, E.NUM_ACTIONS, n).astype(np.int32))
    steps = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        sim.step(rng.integers(0, E.NUM_ACTIONS, n).astype(np.int32))
        steps += n
    dt = time.time() - t0
    sim.close()
    return steps / dt


def _drive(trainer, seconds):
    """The run loop without commands. Returns (env steps/s, seconds in collection, seconds in update)."""
    trainer.state = "running"
    start_steps = trainer.counters["env_steps"]
    t_collect = t_update = 0.0
    t0 = time.time()
    while time.time() - t0 < seconds:
        trainer._pump()
        if trainer.roll_t < trainer.roll.T:
            a = time.perf_counter()
            trainer._collect_step()
            t_collect += time.perf_counter() - a
        else:
            a = time.perf_counter()
            trainer._do_update()
            trainer._after_update()
            t_update += time.perf_counter() - a
    dt = time.time() - t0
    return (trainer.counters["env_steps"] - start_steps) / dt, t_collect, t_update


def make_trainer(n, runs_dir, rollout=64, workers=4, extra=()):
    args = parse_args(["--envs", str(n), "--rollout", str(rollout), "--total-steps", "100000000",
                       "--ckpt-every", "1000000000", "--runs-dir", runs_dir,
                       "--workers", str(workers), "--mode", "live", *extra])
    return Trainer(args)


def full_training(n, seconds, runs_dir, workers=4):
    t = make_trainer(n, runs_dir, workers=workers)
    _drive(t, 1.0)  # warm-up (allocations, first update)
    sps, tc, tu = _drive(t, seconds)
    t.shutdown()
    return sps, tc, tu


def memory_per_env():
    """RSS delta per env for the sim world + rollout buffers (used by the mem_warning estimate)."""
    p = psutil.Process()
    rss0 = p.memory_info().rss
    sim = SimAdapter(256, seed=3, threads=4)
    sim.reset()
    rss_sim = p.memory_info().rss
    roll = Rollout(64, 256, RES, VEC_DIM, 256)
    rss_roll = p.memory_info().rss
    sim.close()
    sim_per = (rss_sim - rss0) / 256
    roll_per = (rss_roll - rss_sim) / 256
    return sim_per, roll_per


def impact(n, seconds, runs_dir):
    """Training sps with no client vs with a subscribed dummy client (snapshot + nn)."""
    results = {}
    for label in ("no client", "subscribed client"):
        t = make_trainer(n, runs_dir)
        proc = None
        if label != "no client":
            from .telemetry import TelemetryServer
            t.server = TelemetryServer("127.0.0.1", 0)
            port = t.server.port
            proc = subprocess.Popen(
                [sys.executable, "-m", "trainer.testclient", "--port", str(port), "--subscribe",
                 "--duration", str(seconds + 6), "--quiet"],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.0)
            t.server.broadcast("log", {"t": time.time(), "level": "info", "msg": "bench"})
        _drive(t, 2.0)
        # subscribe-driven instrumentation is only active when the client's subscribe was processed
        sps, _, _ = _drive(t, seconds)
        results[label] = sps
        if proc is not None:
            proc.wait(timeout=seconds + 30)
        if t.server is not None:
            t.server.close()
        t.shutdown()
    return results


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m trainer.bench")
    p.add_argument("--seconds", type=float, default=12.0, help="measurement window per row")
    p.add_argument("--sim-seconds", type=float, default=4.0)
    p.add_argument("--envs", type=str, default=",".join(str(n) for n in NUM_ENVS))
    p.add_argument("--workers", type=int, default=4, help="sim OpenMP threads for the bench")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="torch intra-op threads (default: cpus - workers, as in trainer.run)")
    p.add_argument("--profile", action="store_true",
                   help="only the training loop at --envs (one value): sps and collect/update split")
    p.add_argument("--impact", action="store_true", help="run the dashboard-impact comparison")
    p.add_argument("--impact-envs", type=int, default=64)
    p.add_argument("--impact-seconds", type=float, default=20.0)
    p.add_argument("--runs-dir", default=os.path.join(DEFAULT_RUNS, "bench"))
    args = p.parse_args(argv)
    os.makedirs(args.runs_dir, exist_ok=True)

    cpu = os.cpu_count() or 1
    torch.set_num_threads(args.torch_threads or max(1, cpu - args.workers))
    sig = E.obs_signature()
    print(f"obs v{E.OBS_VERSION}: vec_dim={sig['vec_dim']} blocks={sig['num_blocks']} "
          f"world={tuple(sig['world_size'])} goal_offset={sig['goal_offset']}", flush=True)
    print(f"cpus={cpu} sim_workers={args.workers} torch_threads={torch.get_num_threads()} "
          f"craftsim_threads_arg={'yes' if 'threads' in E._INIT_PARAMS else 'no'}", flush=True)

    if args.impact:
        r = impact(args.impact_envs, args.impact_seconds, args.runs_dir)
        base = r["no client"]
        sub = r["subscribed client"]
        print(f"dashboard impact at num_envs={args.impact_envs} over {args.impact_seconds:.0f}s windows")
        print(f"  training sps, no client         : {base:10.0f}")
        print(f"  training sps, subscribed client : {sub:10.0f}")
        print(f"  slowdown                        : {100.0 * (1 - sub / base):9.1f} %")
        return 0

    if args.profile:
        n = int(args.envs.split(",")[0])
        sps, tc, tu = full_training(n, args.seconds, args.runs_dir, workers=args.workers)
        tot = tc + tu
        print(f"training num_envs={n}: {sps:,.0f} env-steps/s over {tot:.1f}s of loop time")
        print(f"  collect (policy inference + sim.step): {tc:7.2f}s  {100 * tc / tot:5.1f} %")
        print(f"  PPO update                           : {tu:7.2f}s  {100 * tu / tot:5.1f} %")
        return 0

    envs = [int(x) for x in args.envs.split(",")]
    sim_pe, roll_pe = memory_per_env()
    print(f"measured per env: sim RSS {sim_pe / 2**10:.1f} KiB, rollout buffers (T=64) "
          f"{roll_pe / 2**10:.1f} KiB; estimate used by mem_warning: "
          f"{E.per_env_bytes(64) / 2**10:.1f} KiB", flush=True)
    print(f"{'num_envs':>9} {'sim-only sps':>14} {'training sps':>14} {'train/sim':>10} {'collect%':>9}")
    for n in envs:
        s = sim_only(n, args.sim_seconds, args.workers)
        tr, tc, tu = full_training(n, args.seconds, args.runs_dir, workers=args.workers)
        print(f"{n:9d} {s:14,.0f} {tr:14,.0f} {tr / s:10.2f} {100 * tc / max(tc + tu, 1e-9):8.1f}%",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
