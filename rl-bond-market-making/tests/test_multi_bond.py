"""Multi-bond benchmarks: separable baseline and coupled (covariance) DP."""

import numpy as np
import pytest
from scipy.optimize import minimize

from mm.model import MarketParams, covariance_matrix
from mm.env import MultiBondEnv, simulate_multi_bond
from mm.dp import (
    solve_single_bond, solve_coupled, evaluate_coupled_policy, solve_separable,
    separable_policy, state_index,
)

P = MarketParams()


@pytest.fixture(scope="module")
def single():
    return solve_single_bond(P)


def grid_policy(p, quotes, n_bonds):
    m = 2 * p.q_max + 1
    def pol(n, q):
        idx = np.ravel_multi_index(tuple((q + p.q_max).T), (m,) * n_bonds)
        return quotes[n][idx]
    return pol


def test_one_bond_coupled_equals_single(single):
    V1, b1, _ = single
    V, quotes, _ = solve_coupled(P, 1, 0.0)
    assert np.allclose(V, V1, atol=1e-10)
    assert np.allclose(quotes[..., 0, 0][:, :-1], b1[:, :-1])


def test_zero_correlation_decomposes(single):
    V1, _, _ = single
    V, _, info = solve_coupled(P, 2, 0.0)
    Q = info["Q"]
    expected = V1[0][Q[:, 0] + P.q_max] + V1[0][Q[:, 1] + P.q_max]
    assert np.allclose(V[0], expected, atol=1e-9)


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
def test_coupled_beats_separable_policy_under_correlation(single, rho):
    _, b1, a1 = single
    V, _, info = solve_coupled(P, 2, rho)
    Vsep, _ = evaluate_coupled_policy(P, 2, rho, separable_policy(P, b1, a1))
    assert np.all(V[0] >= Vsep[0] - 1e-9)
    i0 = state_index(info["Q"], [0, 0])
    assert V[0, i0] > Vsep[0, i0]


def test_separable_policy_value_is_independent_of_correlation(single):
    # Under the separable policy inventories are independent with mean zero,
    # so the expected cross-risk term vanishes.
    _, b1, a1 = single
    vals = []
    for rho in (0.0, 0.6):
        V, Q = evaluate_coupled_policy(P, 2, rho, separable_policy(P, b1, a1))
        vals.append(V[0, state_index(Q, [0, 0])])
    assert vals[0] == pytest.approx(vals[1], abs=1e-9)


def test_coupled_value_matches_monte_carlo():
    rho = 0.6
    V, quotes, info = solve_coupled(P, 2, rho)
    x, _ = simulate_multi_bond(P, 2, rho, grid_policy(P, quotes, 2), 60_000, seed=99)
    target = V[0, state_index(info["Q"], [0, 0])]
    assert abs(x.mean() - target) < 3 * x.std() / np.sqrt(len(x))


def test_coupled_bellman_step_matches_numerical_optimizer():
    p = P.with_(T=0.01, dt=0.005)          # two steps
    rho = 0.8
    V, quotes, info = solve_coupled(p, 2, rho)
    Q = info["Q"]
    cov = covariance_matrix(p, 2, rho)
    risk = 0.5 * p.gamma * np.einsum("si,ij,sj->s", Q, cov, Q) * p.dt
    W = dict(zip(map(tuple, Q), V[1] - risk))

    def value(x, q):
        db, da = p.clip_delta(x[:2]), p.clip_delta(x[2:])
        pb = p.fill_prob(db) * (np.array(q) < p.q_max)
        pa = p.fill_prob(da) * (np.array(q) > -p.q_max)
        total = np.sum(pb * db + pa * da)
        for m1 in (-1, 0, 1):
            for m2 in (-1, 0, 1):
                pr = 1.0
                for j, mj in enumerate((m1, m2)):
                    pr *= {1: pb[j] * (1 - pa[j]), 0: pb[j] * pa[j] + (1 - pb[j]) * (1 - pa[j]), -1: pa[j] * (1 - pb[j])}[mj]
                if pr:
                    total += pr * W[(q[0] + m1, q[1] + m2)]
        return total

    rng = np.random.default_rng(0)
    bounds = [(p.delta_min, p.delta_max)] * 4
    for q in [(0, 0), (3, -2), (5, 4), (-4, -4)]:
        s = state_index(Q, q)
        # Independent check: bounded quasi-Newton from 30 random starts
        best = max(-minimize(lambda x: -value(x, q), rng.uniform(0.01, 3.0, 4),
                             method="L-BFGS-B", bounds=bounds).fun for _ in range(30))
        assert V[0, s] >= best - 1e-9
        assert V[0, s] == pytest.approx(best, abs=1e-8)


def test_correlated_inventory_skews_the_other_bond():
    # Long bond 1 with rho > 0: bond 2 should quote a wider bid (avoid adding
    # correlated risk) and a tighter ask (offsetting short is attractive).
    V, quotes, info = solve_coupled(P, 2, 0.6)
    Q = info["Q"]
    flat, long1 = state_index(Q, [0, 0]), state_index(Q, [3, 0])
    assert quotes[0, long1, 1, 0] > quotes[0, flat, 1, 0]
    assert quotes[0, long1, 1, 1] < quotes[0, flat, 1, 1]


def test_quotes_symmetric_under_bond_swap():
    V, quotes, info = solve_coupled(P, 2, 0.6)
    Q = info["Q"]
    s, t = state_index(Q, [2, -1]), state_index(Q, [-1, 2])
    assert V[0, s] == pytest.approx(V[0, t])
    assert np.allclose(quotes[0, s, 0], quotes[0, t, 1])


def test_env_risk_penalty_is_portfolio_variance():
    env = MultiBondEnv(P, n_bonds=3, correlation=0.5, seed=0)
    env.q = np.array([2, -1, 3])
    env.rng = type("R", (), {"random": lambda self, s: np.ones(s), "standard_normal": lambda self, n: np.zeros(n)})()
    _, r, _ = env.step(np.ones((3, 2)))
    cov = covariance_matrix(P, 3, 0.5)
    assert r == pytest.approx(-0.5 * P.gamma * env.q @ cov @ env.q * P.dt)


def test_separable_solver_is_the_single_bond_solution(single):
    V1, _, _ = single
    Vs, _, _ = solve_separable(P, 500)
    assert np.allclose(Vs, V1)
