"""Single-bond environment and DP benchmark tests.

The DP is validated three independent ways: brute-force search over the
quote box, Monte Carlo of the simulator, and the closed-form continuous-time
limit. None of these use RL output.
"""

import numpy as np
import pytest

from mm.model import MarketParams
from mm.env import MultiBondEnv, SingleBondEnv, simulate_single_bond
from mm.dp import (
    solve_single_bond, evaluate_single_bond_policy, continuous_time_solution,
    _bellman_step, _neighbors,
)

P = MarketParams()


class ScriptedRNG:
    """Stand-in for np.random.Generator that returns scripted draws."""

    def __init__(self, uniforms, normals):
        self.u, self.z = list(uniforms), list(normals)

    def random(self, shape):
        return np.array(self.u.pop(0), dtype=float).reshape(shape)

    def standard_normal(self, n):
        return np.array(self.z.pop(0), dtype=float).reshape(n)


def scripted_env(uniforms, normals, params=P, q0=0):
    env = SingleBondEnv(params)
    env.rng = ScriptedRNG(uniforms, normals)
    env.q = np.array([q0])
    return env


# --- Inventory transitions and rewards ----------------------------------

def test_bid_fill_buys_one_unit_and_earns_half_spread():
    env = scripted_env([[[0.0], [1.0]]], [[0.0]])
    _, r, _ = env.step([[0.5, 0.7]])
    assert env.q[0] == 1
    assert env.X == pytest.approx(-(100 - 0.5))
    assert r == pytest.approx(0.5 - P.running_penalty * 1 * P.dt)


def test_both_sides_fill_leaves_inventory_unchanged():
    env = scripted_env([[[0.0], [0.0]]], [[0.0]])
    _, r, _ = env.step([[0.5, 0.7]])
    assert env.q[0] == 0
    assert r == pytest.approx(1.2)


def test_inventory_limits_use_start_of_step_inventory():
    env = scripted_env([[[0.0], [1.0]]], [[0.0]], q0=P.q_max)
    env.step([[0.5, 0.7]])
    assert env.q[0] == P.q_max  # bid cannot fill at +q_max
    env = scripted_env([[[0.0], [0.0]]], [[0.0]], q0=P.q_max)
    env.step([[0.5, 0.7]])
    assert env.q[0] == P.q_max - 1  # ask still fills; inventory stays in range


def test_price_move_marks_post_fill_inventory():
    env = scripted_env([[[0.0], [1.0]]], [[1.0]], q0=2)
    _, r, _ = env.step([[0.5, 0.7]])
    dS = P.sigma * np.sqrt(P.dt)
    assert r == pytest.approx(0.5 + 3 * dS - P.running_penalty * 9 * P.dt)


def test_quotes_are_clipped_to_admissible_box():
    env = scripted_env([[[0.0], [1.0]]], [[0.0]])
    _, r, _ = env.step([[-3.0, 99.0]])
    assert r == pytest.approx(P.delta_min - P.running_penalty * P.dt)


def test_terminal_step_charges_liquidation_penalty():
    p = P.with_(T=0.01, dt=0.005)
    env = scripted_env([[[0.0], [1.0]], [[0.0], [1.0]]], [[0.0], [0.0]], params=p)
    env.step([[0.5, 0.7]])
    _, r, done = env.step([[0.5, 0.7]])
    assert done
    assert r == pytest.approx(0.5 - p.running_penalty * 4 * p.dt - p.terminal_penalty * 4)


def test_rewards_sum_to_the_objective():
    env = SingleBondEnv(P, seed=3)
    total = 0.0
    while not env.done:
        _, r, _ = env.step([[0.6, 0.6]])
        total += r
    assert total == pytest.approx(env.objective_realized())


def test_batch_simulator_reproduces_env_path():
    pol = lambda n, q: (np.full(q.shape, 0.6), np.full(q.shape, 0.8))
    env = SingleBondEnv(P, seed=11)
    total = 0.0
    while not env.done:
        _, r, _ = env.step([[0.6, 0.8]])
        total += r
    assert simulate_single_bond(P, pol, 1, seed=11)[0] == pytest.approx(total)


def test_multibond_env_with_one_bond_matches_single():
    a, b = SingleBondEnv(P, seed=5), MultiBondEnv(P, n_bonds=1, seed=5)
    for _ in range(50):
        ra = a.step([[0.4, 0.9]])[1]
        rb = b.step([[0.4, 0.9]])[1]
        assert ra == pytest.approx(rb)


# --- DP benchmark ----------------------------------------------------------

@pytest.fixture(scope="module")
def solution():
    return solve_single_bond(P)


def test_terminal_condition(solution):
    V, _, _ = solution
    assert np.allclose(V[-1], -P.terminal_penalty * P.q_grid ** 2)


@pytest.mark.parametrize("n, q", [(0, 0), (100, 3), (199, -5), (150, 4), (50, -2)])
def test_dp_matches_brute_force_quote_search(solution, n, q):
    V, _, _ = solution
    W = V[n + 1] - P.running_penalty * P.q_grid ** 2 * P.dt
    dn, w0, up = _neighbors(W)
    i = q + P.q_max
    g = np.linspace(P.delta_min, P.delta_max, 1501)
    DB, DA = np.meshgrid(g, g, indexing="ij")
    full = lambda x: np.full(DB.shape, x)
    val, _, _ = _bellman_step(P, (full(dn[i]), full(w0[i]), full(up[i])),
                              full(float(q < P.q_max)), full(float(q > -P.q_max)), DB, DA)
    assert V[n, i] >= val.max() - 1e-9
    assert V[n, i] == pytest.approx(val.max(), abs=1e-5)


def test_dp_value_matches_monte_carlo(solution):
    V, bid, ask = solution
    pol = lambda n, q: (np.nan_to_num(bid[n][q + P.q_max], nan=P.delta_max),
                        np.nan_to_num(ask[n][q + P.q_max], nan=P.delta_max))
    x = simulate_single_bond(P, pol, 100_000, seed=2024)
    half_width = 3.0 * x.std() / np.sqrt(len(x))  # ~99.7% interval
    assert abs(x.mean() - V[0, P.q_max]) < half_width


def test_dp_converges_to_continuous_time_closed_form():
    errors = []
    for dt in (0.004, 0.001, 0.00025):
        p = P.with_(dt=dt, delta_min=-50.0, delta_max=50.0)
        V, _, _ = solve_single_bond(p)
        h, _, _ = continuous_time_solution(p)
        errors.append(np.max(np.abs(V[0] - h)))
    # First-order convergence: error shrinks ~4x per 4x smaller dt
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.25)
    assert errors[2] < 0.06


def test_policy_evaluation_of_optimal_policy_reproduces_optimum(solution):
    V, bid, ask = solution
    Vpi = evaluate_single_bond_policy(P, lambda n, q: (np.nan_to_num(bid[n], nan=1.0), np.nan_to_num(ask[n], nan=1.0)))
    assert np.allclose(Vpi, V)


def test_other_policies_score_below_optimum(solution):
    V, _, _ = solution
    const = evaluate_single_bond_policy(P, lambda n, q: (np.full(q.shape, 1 / P.k), np.full(q.shape, 1 / P.k)))
    assert const[0, P.q_max] < V[0, P.q_max]


# --- Optimal quoting behavior ---------------------------------------------

def test_quotes_are_symmetric_in_inventory(solution):
    _, bid, ask = solution
    assert np.allclose(bid[:, :-1], ask[:, ::-1][:, :-1], equal_nan=True)


def test_inventory_skew_direction(solution):
    _, bid, ask = solution
    b, a = bid[0, :-1], ask[0, 1:]
    assert np.all(np.diff(b) > 0)   # longer -> less eager to buy
    assert np.all(np.diff(a) < 0)   # longer -> more eager to sell


def test_more_risk_aversion_means_more_skew():
    skew = []
    for gamma in (0.05, 0.1, 0.2):
        _, bid, ask = solve_single_bond(P.with_(gamma=gamma))
        i = P.q_max + 2
        skew.append(bid[0, i] - ask[0, i])
    assert skew[0] < skew[1] < skew[2]


def test_volatility_enters_through_the_risk_penalty():
    _, bid_lo, _ = solve_single_bond(P.with_(sigma=1.0))
    _, bid_hi, _ = solve_single_bond(P.with_(sigma=3.0))
    i = P.q_max + 2
    assert bid_hi[0, i] > bid_lo[0, i]
