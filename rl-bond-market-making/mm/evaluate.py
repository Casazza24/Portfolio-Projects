"""Evaluate frozen policies against the numerical benchmarks.

Two complementary measurements:
  * exact: dp.evaluate_*_policy gives the policy's expected objective with
    no Monte Carlo error (feasible for 1 bond and small coupled grids);
  * Monte Carlo: independent simulation episodes (seeds never used in
    training) with confidence intervals, feasible at any number of bonds.
Policies are evaluated deterministically (the actor's mean quote).
"""

import numpy as np
import torch
from scipy import stats

from .env import bond_features
from .model import correlation_matrix


def _infer(actor, x):
    with torch.no_grad():
        return actor.deterministic(x)


def mean_ci(x, level=0.95):
    """Mean and t-based confidence half-width."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    half = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)


def summarize(values, level=0.95):
    m, h = mean_ci(values, level)
    v = np.asarray(values, dtype=float)
    return {"mean": m, "ci95": h, "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def single_actor_policy(actor, params):
    """policy(n, q) -> (bid, ask) from a single-bond Actor (observation (tau, q/q_max))."""
    def policy(n, q):
        q = np.asarray(q)
        tau = params.T - n * params.dt
        x = torch.as_tensor(np.stack([np.full(q.shape, tau), q / params.q_max], -1), dtype=torch.float32)
        out = _infer(actor, x).numpy().astype(float)
        return out[..., 0], out[..., 1]
    return policy


def shared_actor_policy(actor, params, n_bonds, rho):
    """policy(n, Q[(E, n)]) -> (E, n, 2) from a parameter-shared actor."""
    C = correlation_matrix(n_bonds, rho)
    idx = (np.arange(n_bonds) / max(n_bonds - 1, 1)).astype(np.float32)

    def policy(n, Q):
        tau = params.T - n * params.dt
        feats = bond_features(np.full(len(Q), tau), Q, C, params.q_max, idx)
        out = _infer(actor, torch.from_numpy(feats.reshape(-1, 4))).numpy().astype(float)
        return out.reshape(len(Q), n_bonds, 2)
    return policy


def multi_actor_policy(actor, params, n_bonds):
    """policy(n, Q[(E, n)]) -> (E, n, 2) from a flat-state MultiActor."""
    def policy(n, Q):
        tau = params.T - n * params.dt
        x = np.concatenate([np.full((len(Q), 1), tau), Q / params.q_max], axis=1)
        out = _infer(actor, torch.as_tensor(x, dtype=torch.float32)).numpy().astype(float)
        return out.reshape(len(Q), n_bonds, 2)
    return policy


def as_single(policy_multi):
    """Adapt an (E, 1, 2) multi-bond policy to the single-bond (bid, ask) form."""
    def policy(n, q):
        out = policy_multi(n, np.asarray(q).reshape(-1, 1))
        return out[:, 0, 0], out[:, 0, 1]
    return policy
