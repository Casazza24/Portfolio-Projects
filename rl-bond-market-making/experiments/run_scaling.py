"""Scaling benchmark: exact coupled DP vs separable DP vs RL (shared actor).

The original notebook claimed "FD time grows exponentially with bonds while
RL grows linearly; the shared actor has constant parameter count". This
script measures each piece separately so the claim can be checked:

  * coupled DP (solve_coupled): exact optimum on (2 q_max + 1)^n states.
    Exponential in n. Timed at a reduced q_max (--coupled-qmax, default 3)
    so that n = 4 is feasible.
  * separable DP (solve_separable): identical bonds -> one single-bond solve,
    constant in n. "separable_heterogeneous" times n independent
    solve_single_bond calls (bond i gets sigma * (1 + 0.01 i)), the O(n) cost
    if bonds differed. Neither is the coupled optimum when rho != 0.
  * RL (train_shared): wall time per training episode. train_shared is run
    for warmup + E episodes; a subclass of MultiBondEnv (patched into
    mm.train's namespace, no repo code is changed) timestamps every reset(),
    so each episode's duration (rollout + both updates) is measured directly
    and setup cost (env, networks, optimizers) is excluded. The first
    `warmup` episodes are discarded. Actor parameter count is constant;
    critic MLPCritic(n + 1, 128) grows with n.

Every timing is repeated --repeats times (median, min, max, std reported).
Peak memory is measured in a separate run under tracemalloc (it slows code
down, so it is never active during timing): it covers Python heap
allocations, including numpy buffers, but excludes torch C++ allocations.

Usage:  python experiments/run_scaling.py [--repeats 5] [--quick]
Writes results/scaling.json.
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
import tracemalloc

# Single-threaded numerics: keeps the laptop responsive and makes timings
# comparable (BLAS threads otherwise oversubscribe the O(n^2) products).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import mm.train  # noqa: E402
from mm.env import MultiBondEnv  # noqa: E402
from mm.model import MarketParams  # noqa: E402
from mm.dp import solve_single_bond, solve_separable, solve_coupled  # noqa: E402
from mm.policy import SharedActor, MLPCritic, count_params  # noqa: E402
from mm.train import train_shared  # noqa: E402

P = MarketParams()
MEM_LABEL = "tracemalloc peak (Python heap; excludes torch C++ allocations)"
RL_WORK_NOTE = ("Per step: actor forward on (n, 4) features is O(n) (shared weights, batch of n); "
                "critic forward on the (n + 1) flat observation is O(n) (first layer n+1 -> 128); "
                "env covariance mat-vecs (bond_features C q, risk penalty Sigma q) and the "
                "Cholesky price shock are O(n^2). Episode length is fixed at N = T/dt steps.")


def stats(times):
    t = np.asarray(times, dtype=float)
    return {"median": float(np.median(t)), "min": float(t.min()), "max": float(t.max()),
            "std": float(t.std(ddof=1)) if len(t) > 1 else 0.0, "n": int(len(t))}


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def peak_mem_mb(fn):
    """tracemalloc peak (MB) of one call."""
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 2 ** 20


def bench_coupled(sizes, q_max, rho, repeats):
    p = P.with_(q_max=q_max)
    out = []
    for n in sizes:
        fn = lambda: solve_coupled(p, n, rho)  # noqa: E731
        times = [timed(fn) for _ in range(repeats)]
        row = {"n_bonds": n, "q_max": q_max, "states": (2 * q_max + 1) ** n,
               "moves_per_state": 3 ** n, "seconds_raw": times, "seconds": stats(times),
               "peak_mem_mb": peak_mem_mb(fn)}
        out.append(row)
        print(f"  coupled n={n}: median {row['seconds']['median']:.3f}s", flush=True)
    return out


def _heterogeneous(n):
    for i in range(n):
        solve_single_bond(P.with_(sigma=P.sigma * (1 + 0.01 * i)))


def bench_separable(sizes, repeats):
    out = []
    for n in sizes:
        row = {"n_bonds": n, "q_max": P.q_max}
        for name, fn in [("separable_identical", lambda: solve_separable(P, n)),
                         ("separable_heterogeneous", lambda: _heterogeneous(n))]:
            times = [timed(fn) for _ in range(repeats)]
            row[name] = {"seconds_raw": times, "seconds": stats(times), "peak_mem_mb": peak_mem_mb(fn)}
        out.append(row)
        print(f"  separable n={n}: identical {row['separable_identical']['seconds']['median']:.3f}s  "
              f"heterogeneous {row['separable_heterogeneous']['seconds']['median']:.3f}s", flush=True)
    return out


class _TimedEnv(MultiBondEnv):
    """MultiBondEnv that records perf_counter() at every reset()."""
    stamps = []

    def reset(self, seed=None):
        _TimedEnv.stamps.append(time.perf_counter())
        return super().reset(seed)


def episode_times(n, rho, n_episodes, seed):
    """Wall time of each train_shared episode (reset to next reset / return)."""
    mm.train.MultiBondEnv = _TimedEnv
    try:
        train_shared(P, n, rho, n_episodes=n_episodes, seed=seed)
        end = time.perf_counter()
    finally:
        mm.train.MultiBondEnv = MultiBondEnv
    stamps = _TimedEnv.stamps[1:] + [end]       # stamps[0] is the reset in __init__
    _TimedEnv.stamps = []
    return list(np.diff(stamps))


def bench_rl(sizes, rho, episodes, warmup, repeats):
    out = []
    actor_params = count_params(SharedActor(4, 128, P.delta_min, P.delta_max))
    for n in sizes:
        per_ep, all_eps = [], []
        for r in range(repeats):
            eps = episode_times(n, rho, warmup + episodes, seed=r)[warmup:]
            all_eps.append(eps)
            per_ep.append(float(np.mean(eps)))
        row = {"n_bonds": n, "episodes_timed": episodes, "warmup_episodes": warmup,
               "steps_per_episode": P.n_steps,
               "episode_seconds_raw": all_eps,
               "seconds_per_episode_raw": per_ep,
               "seconds_per_episode": stats(per_ep),
               "seconds_single_episode": stats(np.concatenate(all_eps)),
               "peak_mem_mb_one_episode": peak_mem_mb(lambda: train_shared(P, n, rho, n_episodes=1, seed=0)),
               "actor_params": actor_params, "critic_params": count_params(MLPCritic(n + 1, 128))}
        out.append(row)
        print(f"  rl n={n}: median {row['seconds_per_episode']['median']:.3f}s/episode", flush=True)
    return out


def loglog_slope(xs, ys):
    """Least-squares slope of log y on log x (growth exponent); None if < 2 points."""
    if len(xs) < 2 or min(ys) <= 0:
        return None
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--coupled-qmax", type=int, default=3)
    ap.add_argument("--coupled-sizes", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--separable-sizes", type=int, nargs="+", default=[1, 10, 100, 500])
    ap.add_argument("--rl-sizes", type=int, nargs="+", default=[1, 2, 5, 10, 50, 100, 500])
    ap.add_argument("--rl-episodes", type=int, default=20)
    ap.add_argument("--rl-warmup", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: repeats=1, rl sizes 1 2 5, rl-episodes 3, coupled 1 2 3, separable 1 10")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "scaling.json"))
    args = ap.parse_args()
    if args.quick:
        args.repeats, args.rl_sizes, args.rl_episodes = 1, [1, 2, 5], 3
        args.coupled_sizes, args.separable_sizes = [1, 2, 3], [1, 10]

    torch.set_num_threads(1)
    solve_single_bond(P)                     # untimed: first-call / import warm-up
    solve_coupled(P.with_(q_max=1), 1, args.rho)
    train_shared(P, 1, args.rho, n_episodes=1)
    print(f"Coupled DP (q_max={args.coupled_qmax}, rho={args.rho})", flush=True)
    coupled = bench_coupled(args.coupled_sizes, args.coupled_qmax, args.rho, args.repeats)
    print(f"Separable DP (q_max={P.q_max})", flush=True)
    separable = bench_separable(args.separable_sizes, args.repeats)
    print(f"RL train_shared (rho={args.rho}, {args.rl_episodes} timed episodes after {args.rl_warmup} warm-up)",
          flush=True)
    rl = bench_rl(args.rl_sizes, args.rho, args.rl_episodes, args.rl_warmup, args.repeats)

    summary = {
        "coupled_growth_ratio_per_bond": [
            coupled[i + 1]["seconds"]["median"] / coupled[i]["seconds"]["median"] for i in range(len(coupled) - 1)],
        "separable_identical_loglog_slope": loglog_slope(
            [r["n_bonds"] for r in separable], [r["separable_identical"]["seconds"]["median"] for r in separable]),
        "separable_heterogeneous_loglog_slope": loglog_slope(
            [r["n_bonds"] for r in separable], [r["separable_heterogeneous"]["seconds"]["median"] for r in separable]),
        "rl_loglog_slope": loglog_slope(
            [r["n_bonds"] for r in rl], [r["seconds_per_episode"]["median"] for r in rl]),
        "actor_params_constant": len({r["actor_params"] for r in rl}) == 1,
        "critic_params_by_n": {str(r["n_bonds"]): r["critic_params"] for r in rl},
    }
    out = {
        "params": P.__dict__, "rho": args.rho, "repeats": args.repeats, "quick": args.quick,
        "torch_num_threads": torch.get_num_threads(),
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "torch": torch.__version__, "numpy": np.__version__,
                    "load_avg_at_end": list(os.getloadavg())},
        "memory_label": MEM_LABEL,
        "process_ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "process_ru_maxrss_note": "whole-process high-water mark (bytes on macOS, KiB on Linux); not per measurement",
        "rl_work_note": RL_WORK_NOTE,
        "coupled": coupled, "separable": separable, "rl": rl, "summary": summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    def fmt(s):
        return f"{s['median']:9.4f} [{s['min']:.4f}, {s['max']:.4f}] sd {s['std']:.4f}"

    print(f"\nTimes in seconds: median [min, max] sd over {args.repeats} repeat(s). Memory = {MEM_LABEL}.")
    print(f"\nExact coupled DP, q_max={args.coupled_qmax}")
    print(f"  {'n':>3} {'states':>8}  {'time':<42} {'mem MB':>8}")
    for r in coupled:
        print(f"  {r['n_bonds']:>3} {r['states']:>8}  {fmt(r['seconds']):<42} {r['peak_mem_mb']:8.2f}")
    print(f"  growth ratio per added bond: {[round(x, 1) for x in summary['coupled_growth_ratio_per_bond']]}")

    print(f"\nSeparable DP, q_max={P.q_max}")
    print(f"  {'n':>4}  {'identical (1 solve)':<42} {'heterogeneous (n solves)':<42} {'mem MB (het)':>12}")
    for r in separable:
        a, b = r["separable_identical"], r["separable_heterogeneous"]
        print(f"  {r['n_bonds']:>4}  {fmt(a['seconds']):<42} {fmt(b['seconds']):<42} {b['peak_mem_mb']:12.2f}")

    print(f"\nRL train_shared, seconds per episode ({P.n_steps} steps; stats over per-repeat means), "
          f"torch threads = {torch.get_num_threads()}")
    print(f"  {'n':>4}  {'time / episode':<42} {'mem MB':>8} {'actor params':>13} {'critic params':>14}")
    for r in rl:
        print(f"  {r['n_bonds']:>4}  {fmt(r['seconds_per_episode']):<42} {r['peak_mem_mb_one_episode']:8.2f} "
              f"{r['actor_params']:>13} {r['critic_params']:>14}")
    print(f"  log-log slope of time vs n: {summary['rl_loglog_slope']}")
    print(f"  {RL_WORK_NOTE}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
