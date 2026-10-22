"""Multi-bond experiments against exact coupled-DP benchmarks.

Part A (notebook 2.4): parameter-shared actor vs the coupled optimum and the
    separable (correlation-ignoring) policy, n = 2, 3 bonds.
Part C (notebook 2.5): large portfolios, where exact DP is impossible. The
    shared-actor policy is compared with the separable DP policy by Monte
    Carlo on common random numbers (same evaluation seeds), giving a paired
    difference with a confidence interval. 500 bonds is optional
    (--large-sizes 50 to skip it).
Part B (notebook Section 3): linear vs quadratic vs neural-network critics.
    The notebook compared average *training* reward; here each frozen
    policy is also scored exactly under the coupled objective.

Every frozen policy is scored exactly (coupled-DP policy evaluation) where
the grid allows, plus Monte Carlo on evaluation seeds never used in training.

Resource use: numerical libraries are limited to one thread and jobs run
sequentially in-process by default (--workers 1). Each finished job is saved
to results/runs/<part>/<config>.json and skipped on re-runs, so an
interrupted experiment resumes where it stopped. Part B also saves trained
weights to results/models/ for the notebook.

Usage:  python experiments/run_multi_bond.py --part A|B|C [--seeds 3] [--workers 1]
Writes results/multi_bond_{A,B,C}.json.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mm.model import MarketParams  # noqa: E402
from mm.dp import (solve_single_bond, solve_coupled, evaluate_coupled_policy,  # noqa: E402
                   separable_policy, state_index)
from mm.env import simulate_multi_bond  # noqa: E402
from mm.evaluate import shared_actor_policy, multi_actor_policy, summarize  # noqa: E402

P = MarketParams()
EVAL_EPISODES = 5_000
LARGE_EVAL_EPISODES = 200
EVAL_SEED_OFFSET = 2_000_000
RESULTS = os.path.join(ROOT, "results")


# ----------------------------------------------------------- cached solves

@lru_cache(maxsize=None)
def single_solution():
    return solve_single_bond(P)


@lru_cache(maxsize=None)
def benchmarks(n_bonds, rho):
    """(coupled optimum, separable-policy value) from q = 0; solved once per config."""
    V, _, info = solve_coupled(P, n_bonds, rho)
    _, b1, a1 = single_solution()
    Vsep, _ = evaluate_coupled_policy(P, n_bonds, rho, separable_policy(P, b1, a1))
    i0 = state_index(info["Q"], [0] * n_bonds)
    return float(V[0, i0]), float(Vsep[0, i0])


def exact_value(policy, n_bonds, rho):
    V, Q = evaluate_coupled_policy(P, n_bonds, rho, policy)
    return float(V[0, state_index(Q, [0] * n_bonds)])


def scored(policy, n_bonds, rho, eval_seed):
    J_opt, J_sep = benchmarks(n_bonds, rho)
    J = exact_value(policy, n_bonds, rho)
    mc, _ = simulate_multi_bond(P, n_bonds, rho, policy, EVAL_EPISODES, seed=eval_seed)
    return {"exact_objective": J, "gap_pct": 100 * (J_opt - J) / abs(J_opt),
            "vs_separable": J - J_sep, "mc_objective": summarize(mc)}


def _torch():
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # can only be set once per process; already 1 on later jobs
    return torch


# --------------------------------------------------------------------- jobs

def job_A(args):
    n_bonds, rho, episodes, seed = args
    _torch()
    from mm.train import train_shared
    t0 = time.time()
    actor, _, hist = train_shared(P, n_bonds, rho, episodes, seed=seed)
    out = scored(shared_actor_policy(actor, P, n_bonds, rho), n_bonds, rho, EVAL_SEED_OFFSET + seed)
    out.update({"n_bonds": n_bonds, "rho": rho, "episodes": episodes, "seed": seed,
                "train_seconds": time.time() - t0, "train_objective_last100": float(np.mean(hist[-100:]))})
    return out


def job_B(args):
    critic, n_bonds, rho, episodes, seed = args
    torch = _torch()
    from mm.train import train_with_critic
    t0 = time.time()
    actor, crit, hist, losses = train_with_critic(critic, P, n_bonds, rho, episodes, seed=seed)
    train_s = time.time() - t0
    os.makedirs(os.path.join(RESULTS, "models"), exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": crit.state_dict(), "history": hist, "losses": losses},
               os.path.join(RESULTS, "models", f"B_{critic}_n{n_bonds}_rho{rho}_s{seed}_ep{episodes}.pt"))
    out = scored(multi_actor_policy(actor, P, n_bonds), n_bonds, rho, EVAL_SEED_OFFSET + seed)
    out.update({"critic": critic, "n_bonds": n_bonds, "rho": rho, "episodes": episodes, "seed": seed,
                "train_seconds": train_s,
                "train_objective_last200": float(np.mean(hist[-200:])),
                "critic_loss_last200": float(np.mean(losses[-200:]))})
    return out


def job_C(args):
    n_bonds, rho, episodes, seed = args
    _torch()
    from mm.train import train_shared
    t0 = time.time()
    actor, _, hist = train_shared(P, n_bonds, rho, episodes, seed=seed)
    train_s = time.time() - t0
    _, b1, a1 = single_solution()
    eval_seed = EVAL_SEED_OFFSET + 10_000 + seed
    rl, q_rl = simulate_multi_bond(P, n_bonds, rho, shared_actor_policy(actor, P, n_bonds, rho),
                                   LARGE_EVAL_EPISODES, seed=eval_seed)
    sep, q_sep = simulate_multi_bond(P, n_bonds, rho, separable_policy(P, b1, a1),
                                     LARGE_EVAL_EPISODES, seed=eval_seed)
    return {"n_bonds": n_bonds, "rho": rho, "episodes": episodes, "seed": seed, "train_seconds": train_s,
            "rl_per_bond": summarize(rl / n_bonds), "separable_per_bond": summarize(sep / n_bonds),
            "paired_diff_per_bond": summarize((rl - sep) / n_bonds),
            "rl_final_abs_inventory": float(np.abs(q_rl).mean()),
            "separable_final_abs_inventory": float(np.abs(q_sep).mean()),
            "train_objective_last100_per_bond": float(np.mean(hist[-100:]) / n_bonds)}


JOBS = {"A": job_A, "B": job_B, "C": job_C}
CONFIG_KEYS = {"A": ("n_bonds", "rho", "episodes", "seed"),
               "B": ("critic", "n_bonds", "rho", "episodes", "seed"),
               "C": ("n_bonds", "rho", "episodes", "seed")}


# ------------------------------------------------------------ resumability

def run_path(part, config):
    name = "_".join(f"{k}{v}" for k, v in zip(CONFIG_KEYS[part], config))
    return os.path.join(RESULTS, "runs", part, name + ".json")


def import_aggregate(part):
    """Seed the per-run cache from an earlier aggregate results file so its
    completed configurations are never recomputed."""
    path = os.path.join(RESULTS, f"multi_bond_{part}.json")
    if not os.path.exists(path):
        return 0
    n = 0
    for r in json.load(open(path)).get("runs", []):
        dest = run_path(part, tuple(r[k] for k in CONFIG_KEYS[part]))
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            json.dump(r, open(dest, "w"), indent=1)
            n += 1
    return n


def run_jobs(part, configs, workers):
    todo = [c for c in configs if not os.path.exists(run_path(part, c))]
    print(f"Part {part}: {len(configs) - len(todo)} of {len(configs)} configurations already done; running {len(todo)}",
          flush=True)

    def save(config, result):
        dest = run_path(part, config)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        json.dump(result, open(dest, "w"), indent=1)
        print(f"  done {os.path.basename(dest)}", flush=True)

    if workers <= 1:
        for c in todo:
            save(c, JOBS[part](c))
    else:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for c, r in zip(todo, pool.imap(JOBS[part], todo)):
                save(c, r)
    # Aggregate every saved run of this part, including configurations from
    # earlier invocations (e.g. extra seeds), so validated results are kept.
    run_dir = os.path.join(RESULTS, "runs", part)
    return [json.load(open(os.path.join(run_dir, f))) for f in sorted(os.listdir(run_dir)) if f.endswith(".json")]


def group(runs, keys, fields):
    out = {}
    for r in runs:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return [{**dict(zip(keys, k)), **{f: summarize([r[f] for r in rs]) for f in fields}}
            for k, rs in sorted(out.items())]


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--large-sizes", type=int, nargs="+", default=[50, 500])
    ap.add_argument("--sweep-rhos", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    imported = import_aggregate(args.part)
    if imported:
        print(f"imported {imported} completed runs from results/multi_bond_{args.part}.json")
    seeds = range(args.seeds)

    if args.part == "A":
        configs = [(n, rho, ep, s) for n in (2, 3) for rho in (0.0, 0.6, 0.9) for ep in (500, 2000) for s in seeds]
        runs = run_jobs("A", configs, args.workers)
        bench = {f"n{n}_rho{rho}": dict(zip(("optimum", "separable_policy"), benchmarks(n, rho)))
                 for n in (2, 3) for rho in (0.0, 0.6, 0.9)}
        summary = group(runs, ["n_bonds", "rho", "episodes"],
                        ["exact_objective", "gap_pct", "vs_separable", "train_objective_last100"])
        json.dump({"params": P.__dict__, "benchmarks": bench, "summary": summary, "runs": runs},
                  open(os.path.join(RESULTS, "multi_bond_A.json"), "w"), indent=1)
        for s in summary:
            b = bench[f"n{s['n_bonds']}_rho{s['rho']}"]
            g, v = s["gap_pct"], s["vs_separable"]
            print(f"  n={s['n_bonds']} rho={s['rho']} ep={s['episodes']:>4}: optimum {b['optimum']:.3f} "
                  f"separable {b['separable_policy']:.3f} | RL J = {s['exact_objective']['mean']:.3f} "
                  f"± {s['exact_objective']['ci95']:.3f}  gap {g['mean']:.2f}% ± {g['ci95']:.2f}%  "
                  f"RL - separable = {v['mean']:+.3f} ± {v['ci95']:.3f}")

    if args.part == "C":
        configs = [(n, 0.6, 1000, s) for n in args.large_sizes for s in range(min(args.seeds, 3))]
        runs = run_jobs("C", configs, args.workers)
        json.dump({"params": P.__dict__, "eval_episodes": LARGE_EVAL_EPISODES, "runs": runs},
                  open(os.path.join(RESULTS, "multi_bond_C.json"), "w"), indent=1)
        for r in runs:
            d = r["paired_diff_per_bond"]
            print(f"  n={r['n_bonds']} seed={r['seed']}: RL {r['rl_per_bond']['mean']:.3f}  separable "
                  f"{r['separable_per_bond']['mean']:.3f}  RL - separable = {d['mean']:+.3f} ± {d['ci95']:.3f}  "
                  f"train {r['train_seconds']:.0f}s")

    if args.part == "B":
        configs = [(c, 3, 0.7, 2500, s) for c in ("linear", "quadratic", "neural_network") for s in seeds]
        configs += [(c, 2, rho, 1500, s) for c in ("linear", "neural_network")
                    for rho in args.sweep_rhos for s in seeds]
        runs = run_jobs("B", configs, args.workers)
        summary = group(runs, ["critic", "n_bonds", "rho", "episodes"],
                        ["exact_objective", "gap_pct", "train_objective_last200", "critic_loss_last200"])
        bench = {f"n{n}_rho{rho}": dict(zip(("optimum", "separable_policy"), benchmarks(n, rho)))
                 for n, rho in sorted({(r["n_bonds"], r["rho"]) for r in runs})}
        json.dump({"params": P.__dict__, "benchmarks": bench, "summary": summary, "runs": runs},
                  open(os.path.join(RESULTS, "multi_bond_B.json"), "w"), indent=1)
        for s in summary:
            e, t = s["exact_objective"], s["train_objective_last200"]
            print(f"  {s['critic']:<15} n={s['n_bonds']} rho={s['rho']}: exact J = {e['mean']:.3f} ± {e['ci95']:.3f} "
                  f"(gap {s['gap_pct']['mean']:.2f}%)  train-reward(last200) = {t['mean']:.3f} ± {t['ci95']:.3f}  "
                  f"critic loss = {s['critic_loss_last200']['mean']:.1f}")


if __name__ == "__main__":
    main()
