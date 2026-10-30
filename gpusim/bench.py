"""Throughput benchmark for gpusim on CUDA, with the CPU sim on the same machine for reference.

    python gpusim/bench.py [--ns 256 1024 4096 8192] [--secs 4] [--policy]

Reports env-steps/s (step + obs, random actions), GPU memory, and with --policy the end-to-end
collection rate with a dummy MineMindPolicy forward + categorical sampling per step (read-only use
of trainer/model.py; weights are random, fp32, no_grad).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# drop this script's own directory: gpusim/gpusim.py would shadow the gpusim package
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, ROOT)
import craftsim as cs  # noqa: E402
from gpusim.gpusim import GpuCraftSim  # noqa: E402


def timed(fn, secs, sync):
    """Runs fn() repeatedly for ~secs seconds (after 3 warm-up calls). Returns (calls, seconds)."""
    for _ in range(3):
        fn()
    sync()
    t0 = time.perf_counter()
    calls = 0
    while True:
        fn()
        calls += 1
        if calls % 4 == 0:
            sync()
            if time.perf_counter() - t0 >= secs:
                break
    sync()
    return calls, time.perf_counter() - t0


def bench_cpu(n, secs):
    sim = cs.CraftSim(n, seed=0, res=32, max_steps=3000, ticks=4, threads=0)
    sim.reset()
    acts = np.random.default_rng(0).integers(0, cs.NUM_ACTIONS, n).astype(np.int32)
    calls, dt = timed(lambda: sim.step(acts), secs, lambda: None)
    return calls * n / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[256, 1024, 4096, 8192])
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--policy", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="also time the CPU sim on this machine")
    args = ap.parse_args()

    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"CPU threads: {os.cpu_count()}")

    if args.cpu:
        for n in (256, 1024):
            print(f"CPU craftsim  N={n:5d}: {bench_cpu(n, args.secs):10.0f} steps/s")

    policy = None
    if args.policy:
        from trainer.model import MineMindPolicy
        policy = MineMindPolicy(res=32).to(dev).eval()
        print(f"policy params: {sum(p.numel() for p in policy.parameters())}")

    rows = []
    for n in args.ns:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        sim = GpuCraftSim(num_envs=n, seed=0, res=32, max_steps=3000, ticks=4, device="cuda",
                          pool_size=args.pool, refresh_every=0)
        sim.reset()
        build_s = time.perf_counter() - t0
        gen = torch.Generator(device=dev).manual_seed(0)
        acts = torch.randint(0, cs.NUM_ACTIONS, (n,), device=dev, generator=gen, dtype=torch.int32)
        sync = torch.cuda.synchronize

        calls, dt = timed(lambda: sim.step(acts), args.secs, sync)
        sps = calls * n / dt
        mem_sim = (torch.cuda.max_memory_allocated() - base) / 2**20
        row = {"N": n, "sim_sps": sps, "sim_ms": 1e3 * dt / calls, "mem_MiB": mem_sim, "build_s": build_s}

        if policy is not None:
            h = policy.zero_hidden(n, device=dev)
            img, vec = sim.img, sim.vec
            torch.cuda.reset_peak_memory_stats()
            base2 = torch.cuda.memory_allocated()
            with torch.no_grad():
                def collect():
                    nonlocal h
                    logits, _v, h = policy(img, vec, h)
                    a = torch.distributions.Categorical(logits=logits).sample()
                    sim.step(a)
                calls2, dt2 = timed(collect, args.secs, sync)
            row["e2e_sps"] = calls2 * n / dt2
            row["policy_mem_MiB"] = (torch.cuda.max_memory_allocated() - base2) / 2**20

        rows.append(row)
        print(row, flush=True)
        sim.close()
        del sim
        torch.cuda.empty_cache()

    print("\n| N | sim steps/s | sim ms/step | sim mem MiB | build s | " + ("e2e steps/s (policy) | policy-path mem MiB |" if args.policy else ""))
    print("|---|---|---|---|---|" + ("---|---|" if args.policy else ""))
    for r in rows:
        extra = f" {r['e2e_sps']:.0f} | {r['policy_mem_MiB']:.0f} |" if args.policy else ""
        print(f"| {r['N']} | {r['sim_sps']:.0f} | {r['sim_ms']:.2f} | {r['mem_MiB']:.0f} | {r['build_s']:.1f} |{extra}")
    print(f"max GPU reserved: {torch.cuda.max_memory_reserved() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
