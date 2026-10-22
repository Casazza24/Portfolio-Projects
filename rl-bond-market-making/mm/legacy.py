"""Historical benchmarks from the original notebooks, kept unchanged so they
can be scored under the corrected objective. Do not use as ground truth.

legacy_fd_single: the notebook's "FD" solver. Its objective has only a
    terminal penalty -gamma q_T^2 (no running penalty, so sigma never enters),
    and it drops the both-sides-fill term of the discrete simulator.
legacy_analytic_quotes: the closed form quoted in model_reference.md,
    delta = (1/k) ln(1 + k/gamma) -/+ gamma (T - t) q. This is not the
    Avellaneda-Stoikov solution (which has gamma sigma^2 in the skew and
    (1/gamma) ln(1 + gamma/k) in the spread) and is shown only for history.
"""

import numpy as np


def legacy_fd_single(T=1.0, dt=0.005, gamma=0.1, k=1.5, A=140.0, q_max=5):
    """Verbatim logic of solve_fd_single_bond from the notebooks (sigma unused)."""
    n_steps = int(T / dt)
    q_range = np.arange(-q_max, q_max + 1)
    n_q = len(q_range)
    V = np.zeros((n_steps + 1, n_q))
    V[-1, :] = -gamma * q_range ** 2
    bid = np.full((n_steps, n_q), 5.0)
    ask = np.full((n_steps, n_q), 5.0)
    for t in range(n_steps - 1, -1, -1):
        for j in range(n_q):
            val = V[t + 1, j]
            if j + 1 < n_q:
                d = V[t + 1, j + 1] - V[t + 1, j]
                db = max(1.0 / k - d, 0.01)
                val += A * np.exp(-k * db) * dt * (d + db)
                bid[t, j] = db
            if j - 1 >= 0:
                d = V[t + 1, j - 1] - V[t + 1, j]
                da = max(1.0 / k - d, 0.01)
                val += A * np.exp(-k * da) * dt * (d + da)
                ask[t, j] = da
            V[t, j] = val
    return V, bid, ask


def legacy_fd_policy(params):
    _, bid, ask = legacy_fd_single(params.T, params.dt, params.gamma, params.k, params.A, params.q_max)
    return lambda n, q: (bid[n][np.asarray(q) + params.q_max], ask[n][np.asarray(q) + params.q_max])


def legacy_analytic_policy(params):
    def policy(n, q):
        q = np.asarray(q, dtype=float)
        tau = params.T - n * params.dt
        base = np.log(1 + params.k / params.gamma) / params.k
        skew = params.gamma * tau * q
        return base + skew, base - skew
    return policy
