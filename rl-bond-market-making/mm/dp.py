"""Exact dynamic-programming benchmarks for the problem in model.py.

The original notebooks call this the finite-difference (FD) solution. Here
it is the backward Bellman recursion of the *discrete-time* simulator, so
its value is the exact optimal objective of the environment the RL agent
trains in, not an approximation of a different continuous-time problem.

Single bond, at step n and inventory q, with W(q') = V_{n+1}(q') - phi q'^2 dt:

    V_n(q) = max_{d_b, d_a}  p_b d_b + p_a d_a + E[ W(q + B - A) ],
    V_N(q) = -ell q^2,

where B ~ Bern(p_b), A ~ Bern(p_a) independent, p_b = 0 at q = q_max and
p_a = 0 at q = -q_max. The expectation includes the both-sides-fill outcome
(probability p_b p_a), which the original solver dropped.

Optimization. With p(d) = c exp(-k d), the objective in d_b alone has the
form p(d_b) (d_b + C_b) + const, where C_b = E[W | B=1] - E[W | B=0] is
taken over the ask fill. This is unimodal with maximizer d_b = 1/k - C_b,
so clipping to the admissible box gives the coordinate-wise optimum. The
same holds for d_a. Because C_b depends on p_a (and vice versa), the two
coordinates are iterated to a fixed point. tests/test_dp.py checks the
fixed point against brute-force search over the quote box.
"""

import numpy as np
from scipy.linalg import expm

from .model import MarketParams


def _bounds(p):
    """Admissible quote range, raised where p(d) would be capped at 1
    (below ln(A dt)/k the fill probability is flat, so wider is better)."""
    lo = p.delta_min
    if p.A * p.dt > 1:
        lo = max(lo, np.log(p.A * p.dt) / p.k)
    return lo, p.delta_max


def _bellman_step(p, W, can_buy, can_sell, d_bid=None, d_ask=None, tol=1e-12, max_iter=200):
    """One backward step for a vector of states.

    W:          values W(q-1), W(q), W(q+1) as three arrays (w_dn, w_0, w_up)
    d_bid/ask:  fixed quotes (policy evaluation) or None (optimize)
    Returns (value, d_bid, d_ask).
    """
    w_dn, w_0, w_up = W
    lo, hi = _bounds(p)
    optimize = d_bid is None
    if optimize:
        d_bid = np.full_like(w_0, 1.0 / p.k)
        d_ask = np.full_like(w_0, 1.0 / p.k)
    else:
        d_bid, d_ask = p.clip_delta(d_bid), p.clip_delta(d_ask)

    def probs(db, da):
        return p.fill_prob(db) * can_buy, p.fill_prob(da) * can_sell

    if optimize:
        for _ in range(max_iter):
            _, pa = probs(d_bid, d_ask)
            c_bid = ((1 - pa) * w_up + pa * w_0) - ((1 - pa) * w_0 + pa * w_dn)
            new_bid = np.clip(1.0 / p.k - c_bid, lo, hi)
            pb, _ = probs(new_bid, d_ask)
            c_ask = ((1 - pb) * w_dn + pb * w_0) - ((1 - pb) * w_0 + pb * w_up)
            new_ask = np.clip(1.0 / p.k - c_ask, lo, hi)
            change = max(np.max(np.abs(new_bid - d_bid)), np.max(np.abs(new_ask - d_ask)))
            d_bid, d_ask = new_bid, new_ask
            if change < tol:
                break

    pb, pa = probs(d_bid, d_ask)
    value = (pb * d_bid + pa * d_ask
             + pb * (1 - pa) * w_up + pa * (1 - pb) * w_dn
             + (pb * pa + (1 - pb) * (1 - pa)) * w_0)
    return value, d_bid, d_ask


def _neighbors(v):
    """(v[q-1], v[q], v[q+1]) with edge padding; padded entries are only
    ever multiplied by a zero fill probability."""
    padded = np.concatenate([[v[0]], v, [v[-1]]])
    return padded[:-2], v, padded[2:]


def _single_bond_recursion(p, policy=None):
    q = p.q_grid
    N = p.n_steps
    V = np.empty((N + 1, len(q)))
    V[N] = -p.terminal_penalty * q ** 2
    bid = np.empty((N, len(q)))
    ask = np.empty((N, len(q)))
    can_buy = (q < p.q_max).astype(float)
    can_sell = (q > -p.q_max).astype(float)
    for n in range(N - 1, -1, -1):
        W = V[n + 1] - p.running_penalty * q ** 2 * p.dt
        fixed = (None, None) if policy is None else policy(n, q)
        V[n], bid[n], ask[n] = _bellman_step(p, _neighbors(W), can_buy, can_sell, *fixed)
    # Quotes on a side that cannot trade are irrelevant; mark them.
    bid[:, q == p.q_max] = np.nan
    ask[:, q == -p.q_max] = np.nan
    return V, bid, ask


def solve_single_bond(params=MarketParams()):
    """Optimal value V[n, q] and quotes bid[n, q], ask[n, q] (q index = q + q_max)."""
    return _single_bond_recursion(params)


def evaluate_single_bond_policy(params, policy):
    """Exact expected objective of a Markov policy.

    policy(n, q_array) -> (d_bid, d_ask) arrays over the inventory grid.
    Returns V^pi[n, q]; V^pi[0, q_max] is the expected objective from q=0.
    """
    V, _, _ = _single_bond_recursion(params, policy)
    return V


def continuous_time_solution(params=MarketParams(), t=0.0):
    """Closed-form solution of the continuous-time limit (dt -> 0) without
    quote bounds, for an independent check of the discrete recursion.

    HJB:  dh/dt - phi q^2 + sum_sides max_d A e^{-k d} (d + h(q +- 1) - h(q)) = 0,
          h(T, q) = -ell q^2.
    The inner max is (A / (e k)) exp(k [h(q+-1) - h(q)]), so h = (1/k) log w
    turns the HJB into the linear ODE  dw/dt = -M w,  with M tridiagonal:
    M_qq = -k phi q^2 and M_{q,q+-1} = A/e (zero beyond the inventory limits).
    Hence w(t) = expm(M (T - t)) w(T).  (Cartea, Jaimungal & Penalva 2015, ch. 10.)

    Returns (h, d_bid, d_ask) on the inventory grid at time t.
    """
    p = params
    q = p.q_grid
    n = len(q)
    M = np.diag(-p.k * p.running_penalty * q.astype(float) ** 2)
    off = p.A / np.e
    M += np.diag(np.full(n - 1, off), 1) + np.diag(np.full(n - 1, off), -1)
    w_T = np.exp(-p.k * p.terminal_penalty * q.astype(float) ** 2)
    w = expm(M * (p.T - t)) @ w_T
    h = np.log(w) / p.k
    h_dn, _, h_up = _neighbors(h)
    d_bid = 1.0 / p.k - (h_up - h)
    d_ask = 1.0 / p.k - (h_dn - h)
    d_bid[q == p.q_max] = np.nan
    d_ask[q == -p.q_max] = np.nan
    return h, d_bid, d_ask
