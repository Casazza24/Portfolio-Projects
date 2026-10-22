"""Single-bond experiment: trained RL policies vs the exact DP optimum.

For each training seed, the frozen (deterministic) policy is scored
  * exactly, by DP policy evaluation (no Monte Carlo error), and
  * by Monte Carlo on independent evaluation seeds (never used in training),
and compared with the optimum and with historical/naive baselines.

Usage:  python experiments/run_single_bond.py [--seeds 5] [--episodes 2000 8000]
Writes results/single_bond.json.
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
from mm.dp import solve_single_bond, evaluate_single_bond_policy  # noqa: E402
from mm.env import simulate_single_bond  # noqa: E402
from mm.evaluate import single_actor_policy, summarize  # noqa: E402
from mm.legacy import legacy_fd_policy, legacy_analytic_policy  # noqa: E402

P = MarketParams()
EVAL_EPISODES = 20_000
EVAL_SEED_OFFSET = 1_000_000


def dp_policy(bid, ask):
    b, a = np.nan_to_num(bid, nan=P.delta_max), np.nan_to_num(ask, nan=P.delta_max)
    return lambda n, q: (b[n][np.asarray(q) + P.q_max], a[n][np.asarray(q) + P.q_max])


def quote_errors(policy, bid, ask):
    """RMSE of the policy's quotes vs the DP optimum over all (t, q) where the
    side can trade, and at t = 0 only."""
    q = P.q_grid
    diffs, diffs0 = [], []
    for n in range(P.n_steps):
        b, a = policy(n, q)
        d = np.concatenate([(b - bid[n])[q < P.q_max], (a - ask[n])[q > -P.q_max]])
        diffs.append(d)
        if n == 0:
            diffs0 = d
    allv = np.concatenate(diffs)
    return float(np.sqrt(np.mean(allv ** 2))), float(np.sqrt(np.mean(np.asarray(diffs0) ** 2)))


def score(policy, V_opt, bid, ask, eval_seed):
    J = float(evaluate_single_bond_policy(P, policy)[0, P.q_max])
    mc = simulate_single_bond(P, policy, EVAL_EPISODES, seed=eval_seed)
    rmse_all, rmse_0 = quote_errors(policy, bid, ask)
    J_opt = float(V_opt[0, P.q_max])
    return {
        "exact_objective": J,
        "gap_to_optimum": J_opt - J,
        "gap_pct": 100 * (J_opt - J) / abs(J_opt),
        "mc_objective": summarize(mc),
        "quote_rmse_all_t": rmse_all,
        "quote_rmse_t0": rmse_0,
    }


def run_seed(args):
    seed, episodes = args
    import torch
    torch.set_num_threads(1)
    from mm.train import train_single
    V_opt, bid, ask = solve_single_bond(P)
    t0 = time.time()
    actor, _, history = train_single(P, episodes, seed=seed)
    elapsed = time.time() - t0
    out = score(single_actor_policy(actor, P), V_opt, bid, ask, EVAL_SEED_OFFSET + seed)
    out.update({"seed": seed, "episodes": episodes, "train_seconds": elapsed,
                "train_objective_last100": float(np.mean(history[-100:])),
                "learning_curve_every50": [float(np.mean(history[i:i + 50])) for i in range(0, len(history), 50)]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, nargs="+", default=[2000, 8000])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    V_opt, bid, ask = solve_single_bond(P)
    baselines = {
        "dp_optimal": score(dp_policy(bid, ask), V_opt, bid, ask, EVAL_SEED_OFFSET - 1),
        "legacy_fd_policy": score(legacy_fd_policy(P), V_opt, bid, ask, EVAL_SEED_OFFSET - 1),
        "legacy_analytic_formula": score(legacy_analytic_policy(P), V_opt, bid, ask, EVAL_SEED_OFFSET - 1),
        "constant_1_over_k": score(lambda n, q: (np.full(len(q), 1 / P.k), np.full(len(q), 1 / P.k)),
                                   V_opt, bid, ask, EVAL_SEED_OFFSET - 1),
    }

    jobs = [(s, e) for e in args.episodes for s in range(args.seeds)]
    with Pool(args.workers) as pool:
        runs = pool.map(run_seed, jobs)

    summary = {}
    for e in args.episodes:
        rs = [r for r in runs if r["episodes"] == e]
        summary[str(e)] = {
            "exact_objective": summarize([r["exact_objective"] for r in rs]),
            "gap_pct": summarize([r["gap_pct"] for r in rs]),
            "quote_rmse_all_t": summarize([r["quote_rmse_all_t"] for r in rs]),
            "train_objective_last100": summarize([r["train_objective_last100"] for r in rs]),
            "train_seconds": summarize([r["train_seconds"] for r in rs]),
        }

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = {"params": P.__dict__, "optimum": float(V_opt[0, P.q_max]), "eval_episodes": EVAL_EPISODES,
           "baselines": baselines, "rl_summary": summary, "rl_runs": runs}
    with open(os.path.join(ROOT, "results", "single_bond.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"Optimum J* = {out['optimum']:.4f}")
    for k, b in baselines.items():
        print(f"  {k:<26} J = {b['exact_objective']:.4f}  gap = {b['gap_pct']:.3f}%  quote RMSE = {b['quote_rmse_all_t']:.4f}")
    for e, s in summary.items():
        g, j = s["gap_pct"], s["exact_objective"]
        print(f"  RL {e:>5} episodes ({j['n']} seeds): J = {j['mean']:.4f} ± {j['ci95']:.4f}  "
              f"gap = {g['mean']:.3f}% ± {g['ci95']:.3f}% [{g['min']:.3f}, {g['max']:.3f}]  "
              f"quote RMSE = {s['quote_rmse_all_t']['mean']:.4f}  train-reward(last100) = {s['train_objective_last100']['mean']:.3f}")


if __name__ == "__main__":
    main()
