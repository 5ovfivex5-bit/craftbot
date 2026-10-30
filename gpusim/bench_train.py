"""End-to-end training throughput of the gpusim backend (run on the CUDA machine).

    python gpusim/bench_train.py --model-size 100m --envs 1024 4096 --minibatch 512 2048 --updates 2

For each (envs, minibatch) pair it builds the real trainer (trainer.run.Trainer, gpusim backend), runs
one warm-up rollout+update and then `--updates` timed ones, and prints: parameter count, peak CUDA
allocation, collection steps/s (policy + sim, no update), update seconds, and end-to-end steps/s.
Nothing is written under runs/ (a scratch --runs-dir is used).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# drop this script's own directory: gpusim/gpusim.py would shadow the gpusim package
sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from trainer.run import Trainer, parse_args  # noqa: E402


def run_one(model_size, envs, minibatch, rollout, updates, amp, bptt, flags, runs_dir):
    argv = ["--backend", "gpusim", "--device", "cuda", "--model-size", model_size,
            "--envs", str(envs), "--minibatch", str(minibatch), "--rollout", str(rollout),
            "--bptt", str(bptt), "--no-server", "--runs-dir", runs_dir,
            "--run-id", f"bench-{model_size}-{envs}-{minibatch}-{int(time.time())}",
            "--ckpt-every", "1000000000000"] + (["--amp"] if amp else ["--no-amp"]) + flags
    args = parse_args(argv)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    tr = Trainer(args)
    tr.state = "running"
    sync = torch.cuda.synchronize
    params = sum(p.numel() for p in tr.model.parameters())
    T = tr.roll.T
    col, upd = [], []
    for u in range(updates + 1):
        sync()
        t0 = time.perf_counter()
        while tr.roll_t < T:
            tr._collect_step()
        sync()
        t1 = time.perf_counter()
        tr._do_update()
        tr._after_update()
        sync()
        t2 = time.perf_counter()
        if u > 0:                     # first rollout is warm-up (kernel JIT, allocator)
            col.append(t1 - t0)
            upd.append(t2 - t1)
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    steps = envs * T
    c = steps / (sum(col) / len(col))
    u_s = sum(upd) / len(upd)
    e2e = steps / (sum(col) / len(col) + u_s)
    # sanity: the gpu goal read-back agrees with the sim's per-env goal
    ok = all(int(tr.sim.goals()[i]) == tr.sim.goal(i) for i in range(0, envs, max(1, envs // 8)))
    row = {"model": model_size, "params": params, "envs": envs, "minibatch": minibatch, "rollout": T,
           "amp": amp, "peak_alloc_GiB": round(peak, 2), "reserved_GiB": round(reserved, 2),
           "collect_sps": round(c), "collect_s": round(sum(col) / len(col), 3),
           "update_s": round(u_s, 2), "e2e_sps": round(e2e), "goal_readback_ok": ok}
    print(row, flush=True)
    tr.sim.close()
    del tr
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="100m", choices=("small", "100m"))
    ap.add_argument("--envs", type=int, nargs="+", default=[1024])
    ap.add_argument("--minibatch", type=int, nargs="+", default=[512])
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--bptt", type=int, default=32)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--flags", nargs=argparse.REMAINDER, default=[],
                    help="extra trainer flags, e.g. --no-rnd --no-spr (must come last)")
    ap.add_argument("--runs-dir", default="/tmp/bench_runs")
    a = ap.parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    rows = []
    for n in a.envs:
        for mb in a.minibatch:
            rows.append(run_one(a.model_size, n, mb, a.rollout, a.updates, a.amp, a.bptt, a.flags, a.runs_dir))
    print("\n| model | params | envs | minibatch | amp | peak GiB | reserved GiB | collect sps | update s | e2e sps |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['params']:,} | {r['envs']} | {r['minibatch']} | {r['amp']} | "
              f"{r['peak_alloc_GiB']} | {r['reserved_GiB']} | {r['collect_sps']:,} | {r['update_s']} | "
              f"{r['e2e_sps']:,} |")


if __name__ == "__main__":
    main()
