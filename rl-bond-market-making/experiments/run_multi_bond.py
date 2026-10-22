"""Multi-bond experiments against exact coupled-DP benchmarks.

Part A (notebook 2.4): parameter-shared actor vs the coupled optimum and the
    separable (correlation-ignoring) policy, n = 2, 3 bonds.
Part C (notebook 2.5): large portfolios (50, 500 bonds), where exact DP is
    impossible. The shared-actor policy is compared with the separable DP
    policy by Monte Carlo on common random numbers (same evaluation seeds),
    giving a paired difference with a confidence interval.
Part B (notebook Section 3): linear vs quadratic vs neural-network critics.
    The notebook compared average *training* reward; here each frozen
    policy is also scored exactly under the coupled objective.

Every frozen policy is scored exactly (coupled-DP policy evaluation), plus
Monte Carlo on evaluation seeds never used in training.

Usage:  python experiments/run_multi_bond.py --part A|B|C|all [--seeds 5]
Writes results/multi_bond_{A,B,C}.json.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mm.model import MarketParams  # noqa: E402
from mm.dp import (solve_single_bond, solve_coupled, evaluate_coupled_policy,  # noqa: E402
                   separable_policy, state_index)
from mm.env import simulate_multi_bond  # noqa: E402
from mm.evaluate import shared_actor_policy, multi_actor_policy, summarize  # noqa: E402

P = MarketParams()
EVAL_EPISODES = 5_000
EVAL_SEED_OFFSET = 2_000_000
_CACHE = {}


def benchmarks(n_bonds, rho):
    key = (n_bonds, rho)
    if key not in _CACHE:
        V, _, info = solve_coupled(P, n_bonds, rho)
        _, b1, a1 = solve_single_bond(P)
        Vsep, _ = evaluate_coupled_policy(P, n_bonds, rho, separable_policy(P, b1, a1))
        i0 = state_index(info["Q"], [0] * n_bonds)
        _CACHE[key] = (float(V[0, i0]), float(Vsep[0, i0]), i0)
    return _CACHE[key]


def exact_value(policy, n_bonds, rho):
    V, Q = evaluate_coupled_policy(P, n_bonds, rho, policy)
    return float(V[0, state_index(Q, [0] * n_bonds)])


def scored(policy, n_bonds, rho, eval_seed):
    J_opt, J_sep, _ = benchmarks(n_bonds, rho)
    J = exact_value(policy, n_bonds, rho)
    mc, _ = simulate_multi_bond(P, n_bonds, rho, policy, EVAL_EPISODES, seed=eval_seed)
    return {"exact_objective": J, "gap_pct": 100 * (J_opt - J) / abs(J_opt),
            "vs_separable": J - J_sep, "mc_objective": summarize(mc)}


def job_A(args):
    n_bonds, rho, episodes, seed = args
    import torch
    torch.set_num_threads(1)
    from mm.train import train_shared
    t0 = time.time()
    actor, _, hist = train_shared(P, n_bonds, rho, episodes, seed=seed)
    out = scored(shared_actor_policy(actor, P, n_bonds, rho), n_bonds, rho, EVAL_SEED_OFFSET + seed)
    out.update({"n_bonds": n_bonds, "rho": rho, "episodes": episodes, "seed": seed,
                "train_seconds": time.time() - t0, "train_objective_last100": float(np.mean(hist[-100:]))})
    return out


def job_B(args):
    critic, n_bonds, rho, episodes, seed = args
    import torch
    torch.set_num_threads(1)
    from mm.train import train_with_critic
    t0 = time.time()
    actor, _, hist, losses = train_with_critic(critic, P, n_bonds, rho, episodes, seed=seed)
    out = scored(multi_actor_policy(actor, P, n_bonds), n_bonds, rho, EVAL_SEED_OFFSET + seed)
    out.update({"critic": critic, "n_bonds": n_bonds, "rho": rho, "episodes": episodes, "seed": seed,
                "train_seconds": time.time() - t0,
                "train_objective_last200": float(np.mean(hist[-200:])),
                "critic_loss_last200": float(np.mean(losses[-200:]))})
    return out


LARGE_EVAL_EPISODES = 200


def job_C(args):
    n_bonds, rho, episodes, seed = args
    import torch
    torch.set_num_threads(1)
    from mm.train import train_shared
    t0 = time.time()
    actor, _, hist = train_shared(P, n_bonds, rho, episodes, seed=seed)
    train_s = time.time() - t0
    _, b1, a1 = solve_single_bond(P)
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


def group(runs, keys, fields):
    out = {}
    for r in runs:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return [{**dict(zip(keys, k)), **{f: summarize([r[f] for r in rs]) for f in fields}}
            for k, rs in sorted(out.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["A", "B", "C", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    seeds = range(args.seeds)

    if args.part in ("A", "all"):
        jobs = [(n, rho, ep, s) for n in (2, 3) for rho in (0.0, 0.6, 0.9) for ep in (500, 2000) for s in seeds]
        with Pool(args.workers) as pool:
            runs = pool.map(job_A, jobs)
        bench = {f"n{n}_rho{rho}": dict(zip(("optimum", "separable_policy"), benchmarks(n, rho)[:2]))
                 for n in (2, 3) for rho in (0.0, 0.6, 0.9)}
        summary = group(runs, ["n_bonds", "rho", "episodes"],
                        ["exact_objective", "gap_pct", "vs_separable", "train_objective_last100"])
        json.dump({"params": P.__dict__, "benchmarks": bench, "summary": summary, "runs": runs},
                  open(os.path.join(ROOT, "results", "multi_bond_A.json"), "w"), indent=1)
        print("Part A: shared actor vs coupled optimum")
        for s in summary:
            b = bench[f"n{s['n_bonds']}_rho{s['rho']}"]
            g, v = s["gap_pct"], s["vs_separable"]
            print(f"  n={s['n_bonds']} rho={s['rho']} ep={s['episodes']:>4}: optimum {b['optimum']:.3f} "
                  f"separable {b['separable_policy']:.3f} | RL J = {s['exact_objective']['mean']:.3f} "
                  f"± {s['exact_objective']['ci95']:.3f}  gap {g['mean']:.2f}% ± {g['ci95']:.2f}%  "
                  f"RL - separable = {v['mean']:+.3f} ± {v['ci95']:.3f}")

    if args.part in ("C", "all"):
        jobs = [(n, 0.6, 1000, s) for n in (50, 500) for s in range(min(args.seeds, 3))]
        with Pool(args.workers) as pool:
            runs = pool.map(job_C, jobs)
        json.dump({"params": P.__dict__, "eval_episodes": LARGE_EVAL_EPISODES, "runs": runs},
                  open(os.path.join(ROOT, "results", "multi_bond_C.json"), "w"), indent=1)
        print("Part C: shared actor vs separable DP policy, Monte Carlo on common random numbers (per bond)")
        for r in runs:
            d = r["paired_diff_per_bond"]
            print(f"  n={r['n_bonds']} seed={r['seed']}: RL {r['rl_per_bond']['mean']:.3f}  separable "
                  f"{r['separable_per_bond']['mean']:.3f}  RL - separable = {d['mean']:+.3f} ± {d['ci95']:.3f}  "
                  f"train {r['train_seconds']:.0f}s")

    if args.part in ("B", "all"):
        critics = ("linear", "quadratic", "neural_network")
        jobs = [(c, 3, 0.7, 2500, s) for c in critics for s in seeds]
        jobs += [(c, 2, rho, 1500, s) for c in ("linear", "neural_network")
                 for rho in (0.0, 0.3, 0.5, 0.7, 0.9) for s in seeds]
        with Pool(args.workers) as pool:
            runs = pool.map(job_B, jobs)
        summary = group(runs, ["critic", "n_bonds", "rho", "episodes"],
                        ["exact_objective", "gap_pct", "train_objective_last200", "critic_loss_last200"])
        bench = {f"n{n}_rho{rho}": dict(zip(("optimum", "separable_policy"), benchmarks(n, rho)[:2]))
                 for n, rho in {(r["n_bonds"], r["rho"]) for r in runs}}
        json.dump({"params": P.__dict__, "benchmarks": bench, "summary": summary, "runs": runs},
                  open(os.path.join(ROOT, "results", "multi_bond_B.json"), "w"), indent=1)
        print("Part B: critic comparison (frozen policies scored exactly)")
        for s in summary:
            e, t = s["exact_objective"], s["train_objective_last200"]
            print(f"  {s['critic']:<15} n={s['n_bonds']} rho={s['rho']}: exact J = {e['mean']:.3f} ± {e['ci95']:.3f} "
                  f"(gap {s['gap_pct']['mean']:.2f}%)  train-reward(last200) = {t['mean']:.3f} ± {t['ci95']:.3f}  "
                  f"critic loss = {s['critic_loss_last200']['mean']:.1f}")


if __name__ == "__main__":
    main()
