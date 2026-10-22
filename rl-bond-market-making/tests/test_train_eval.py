"""Training loops (mm/train.py) and evaluation helpers (mm/evaluate.py).

Training runs use a 10-step horizon and a handful of episodes: these tests
check plumbing (seeding, parameter updates, credit assignment, adapters),
not learning quality.
"""

import numpy as np
import pytest
import torch
from scipy import stats

import mm.train as train
from mm.model import MarketParams
from mm.env import MultiBondEnv, simulate_single_bond
from mm.dp import evaluate_single_bond_policy, evaluate_coupled_policy, state_index
from mm.policy import Actor, SharedActor, MultiActor
from mm.evaluate import (
    mean_ci, summarize, single_actor_policy, shared_actor_policy, multi_actor_policy, as_single,
)

torch.set_num_threads(1)

P = MarketParams().with_(T=0.05, dt=0.005)   # 10 steps
EPISODES = 4


def weights(model):
    return [t.detach().clone() for t in model.state_dict().values()]


def same_weights(a, b):
    return all(torch.equal(x, y) for x, y in zip(weights(a), weights(b)))


def run(kind, seed, n_episodes=EPISODES):
    """Return (actor, history) for one of the three trainers."""
    if kind == "single":
        actor, _, hist = train.train_single(P, n_episodes=n_episodes, seed=seed, hidden=16)
    elif kind == "shared":
        actor, _, hist = train.train_shared(P, n_bonds=2, rho=0.6, n_episodes=n_episodes, seed=seed, hidden=16)
    else:
        actor, _, hist, _ = train.train_with_critic(kind, P, n_bonds=2, rho=0.7, n_episodes=n_episodes, seed=seed)
    return actor, hist


KINDS = ["single", "shared", "linear", "quadratic", "neural_network"]


# --- Reproducibility and parameter updates ------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_same_seed_reproduces_history_and_weights(kind):
    """Same seed -> bit-identical histories and actor weights."""
    a1, h1 = run(kind, seed=3)
    torch.manual_seed(999)          # disturb global RNG state in between
    np.random.seed(999)
    a2, h2 = run(kind, seed=3)
    assert h1 == h2
    assert same_weights(a1, a2)


@pytest.mark.parametrize("kind", ["single", "shared", "quadratic"])
def test_different_seed_differs(kind):
    """Different seeds -> different histories and initial/final weights."""
    a1, h1 = run(kind, seed=0)
    a2, h2 = run(kind, seed=1)
    assert h1 != h2
    assert not same_weights(a1, a2)


@pytest.mark.parametrize("kind", ["single", "shared", "quadratic"])
def test_training_updates_actor(kind):
    """After a few episodes the actor differs from its seed-matched initialization."""
    a0, _ = run(kind, seed=5, n_episodes=0)
    a1, _ = run(kind, seed=5, n_episodes=EPISODES)
    assert len(weights(a0)) == len(weights(a1))
    changed = [not torch.equal(x, y) for x, y in zip(weights(a0), weights(a1))]
    assert any(changed)
    # The mean head and log_std both receive gradient in every trainer.
    names = list(a0.state_dict().keys())
    assert changed[names.index("log_std")]
    assert changed[names.index("mean_head.bias")]


# --- History equals the realized objective ------------------------------

class RecordingEnv(MultiBondEnv):
    """MultiBondEnv that records objective_realized() and summed rewards per episode."""

    log = []

    def reset(self, seed=None):
        self._rewards = []
        self._bond_sum = 0.0
        return super().reset(seed)

    def step(self, actions):
        out = super().step(actions)
        self._rewards.append(out[1])
        self._bond_sum += float(self.bond_rewards.sum())
        if self.done:
            RecordingEnv.log.append((self.objective_realized(), sum(self._rewards), self._bond_sum))
        return out


@pytest.fixture
def recording_env(monkeypatch):
    RecordingEnv.log = []
    monkeypatch.setattr(train, "MultiBondEnv", RecordingEnv)
    return RecordingEnv.log


@pytest.mark.parametrize("kind", ["single", "shared", "linear"])
def test_history_is_realized_episode_objective(kind, recording_env):
    """history[e] = sum over bonds of first-step return-to-go
    = X_N + q.S_N - penalties, computed independently from the env's books."""
    _, hist = run(kind, seed=2, n_episodes=6)
    log = recording_env
    assert len(log) == len(hist) == 6
    for h, (realized, reward_sum, bond_sum) in zip(hist, log):
        assert reward_sum == pytest.approx(realized, abs=1e-9)
        assert bond_sum == pytest.approx(realized, abs=1e-9)
        # history goes through float32 cumulative sums
        assert h == pytest.approx(realized, rel=1e-5, abs=1e-4)


def test_shared_history_not_trivially_constant():
    """Sanity: the episode objectives vary (fills actually happen)."""
    _, hist = run("shared", seed=4, n_episodes=6)
    assert len(set(hist)) > 1


# --- Policy adapters -----------------------------------------------------

def constant_actor(cls, value, *args):
    """Actor whose mean head is zeroed so softplus(bias) = value everywhere."""
    actor = cls(*args, low=P.delta_min, high=P.delta_max)
    with torch.no_grad():
        actor.mean_head.weight.zero_()
        actor.mean_head.bias.fill_(float(np.log(np.expm1(value))) if value < 30 else float(value))
    return actor


def test_single_actor_policy_shape_and_constant():
    pol = single_actor_policy(constant_actor(Actor, 0.8, 2, 8), P)
    bid, ask = pol(3, P.q_grid)
    assert bid.shape == ask.shape == P.q_grid.shape
    assert bid.dtype == np.float64
    assert np.allclose(bid, 0.8, atol=1e-6) and np.allclose(ask, 0.8, atol=1e-6)


@pytest.mark.parametrize("raw, expected", [(100.0, P.delta_max), (1e-6, P.delta_min)])
def test_single_actor_policy_clipped(raw, expected):
    """Means outside [delta_min, delta_max] are clipped exactly to the bound."""
    actor = Actor(2, 8, P.delta_min, P.delta_max)
    with torch.no_grad():
        actor.mean_head.weight.zero_()
        actor.mean_head.bias.fill_(raw if raw > 1 else -20.0)
    bid, ask = single_actor_policy(actor, P)(0, P.q_grid)
    assert np.allclose(bid, expected) and np.allclose(ask, expected)


def test_single_actor_policy_respects_bounds_random_init():
    torch.manual_seed(0)
    bid, ask = single_actor_policy(Actor(2, 8, P.delta_min, P.delta_max), P)(0, P.q_grid)
    for d in (bid, ask):
        assert np.all(d >= P.delta_min - 1e-7) and np.all(d <= P.delta_max + 1e-7)


def test_single_actor_policy_feeds_tau_and_scaled_q():
    """The adapter builds the same (tau, q/q_max) observation as the env."""
    seen = []

    class Spy(Actor):
        def deterministic(self, x):
            seen.append(x.clone())
            return super().deterministic(x)

    single_actor_policy(Spy(2, 8, P.delta_min, P.delta_max), P)(4, P.q_grid)
    x = seen[0].numpy()
    assert np.allclose(x[:, 0], P.T - 4 * P.dt)
    assert np.allclose(x[:, 1], P.q_grid / P.q_max)
    env = MultiBondEnv(P, 1, seed=0)
    env.step_count = 4
    env.q = np.array([2])
    assert np.allclose(env.observation(), x[P.q_max + 2])


def test_shared_and_multi_actor_policy_shapes_and_constant():
    Q = np.array([[0, 0], [1, -2], [5, -5]])
    shared = shared_actor_policy(constant_actor(SharedActor, 0.6, 4, 8), P, 2, 0.5)
    multi = multi_actor_policy(constant_actor(MultiActor, 1.3, 3, 4, 8), P, 2)
    out_s, out_m = shared(2, Q), multi(2, Q)
    assert out_s.shape == out_m.shape == (3, 2, 2)
    assert np.allclose(out_s, 0.6, atol=1e-6)
    assert np.allclose(out_m, 1.3, atol=1e-6)


def test_shared_actor_policy_matches_env_features():
    """Batched adapter == actor applied to env.bond_features() at the same state."""
    torch.manual_seed(1)
    actor = SharedActor(4, 8, P.delta_min, P.delta_max)
    with torch.no_grad():
        actor.mean_head.weight.normal_()   # make output state-dependent
    env = MultiBondEnv(P, 3, correlation=0.4, seed=0)
    env.step_count, env.q = 2, np.array([1, -3, 0])
    expected = actor.deterministic(torch.from_numpy(env.bond_features())).detach().numpy()
    got = shared_actor_policy(actor, P, 3, 0.4)(2, env.q[None, :])
    assert np.allclose(got[0], expected, atol=1e-6)


def test_multi_actor_policy_matches_env_observation():
    torch.manual_seed(2)
    actor = MultiActor(3, 4, 8, P.delta_min, P.delta_max)
    env = MultiBondEnv(P, 2, correlation=0.4, seed=0)
    env.step_count, env.q = 7, np.array([-1, 4])
    expected = actor.deterministic(torch.from_numpy(env.observation()).unsqueeze(0)).detach().numpy()
    got = multi_actor_policy(actor, P, 2)(7, env.q[None, :])
    assert np.allclose(got.reshape(1, 4), expected, atol=1e-6)


def test_as_single_round_trips():
    """as_single(shared adapter with n=1) gives the same quotes as a (bid, ask) policy."""
    torch.manual_seed(3)
    actor = SharedActor(4, 8, P.delta_min, P.delta_max)
    with torch.no_grad():
        actor.mean_head.weight.normal_()
    multi = shared_actor_policy(actor, P, 1, 0.0)
    bid, ask = as_single(multi)(5, P.q_grid)
    full = multi(5, P.q_grid.reshape(-1, 1))
    assert bid.shape == ask.shape == P.q_grid.shape
    assert np.array_equal(bid, full[:, 0, 0]) and np.array_equal(ask, full[:, 0, 1])
    # and back: wrapping the single form as multi reproduces the original
    rewrapped = np.stack(as_single(multi)(5, P.q_grid), axis=-1)[:, None, :]
    assert np.array_equal(rewrapped, full)


# --- Exact evaluation vs Monte Carlo ------------------------------------

def constant_policy(bid, ask):
    def pol(n, q):
        q = np.asarray(q)
        return np.full(q.shape, bid), np.full(q.shape, ask)
    return pol


def test_exact_policy_value_matches_monte_carlo():
    """evaluate_single_bond_policy V[0, q=0] within 3 SE of the simulator mean."""
    p = MarketParams().with_(T=0.5, dt=0.005)
    pol = constant_policy(0.9, 1.1)
    V = evaluate_single_bond_policy(p, pol)
    exact = V[0, p.q_max]
    mc = simulate_single_bond(p, pol, n_episodes=20000, seed=11)
    se = mc.std(ddof=1) / np.sqrt(len(mc))
    assert abs(mc.mean() - exact) < 3 * se
    assert exact > 0  # spread income dominates over this horizon


def test_exact_actor_value_matches_monte_carlo():
    """Same check through the actor adapter (state-dependent quotes)."""
    torch.manual_seed(4)
    actor = Actor(2, 8, P.delta_min, P.delta_max)
    p = MarketParams().with_(T=0.25, dt=0.005)
    pol = single_actor_policy(actor, p)
    exact = evaluate_single_bond_policy(p, pol)[0, p.q_max]
    mc = simulate_single_bond(p, pol, n_episodes=20000, seed=12)
    assert abs(mc.mean() - exact) < 3 * mc.std(ddof=1) / np.sqrt(len(mc))


def test_coupled_one_bond_equals_single_bond_evaluation():
    p = MarketParams().with_(T=0.1, dt=0.005)
    single = constant_policy(0.7, 1.4)
    V1 = evaluate_single_bond_policy(p, single)

    def multi(n, Q):
        b, a = single(n, Q[:, 0])
        return np.stack([b, a], axis=-1)[:, None, :]

    V, Q = evaluate_coupled_policy(p, 1, 0.0, multi)
    assert np.allclose(V, V1, atol=1e-12)
    assert V[0, state_index(Q, [0])] == pytest.approx(V1[0, p.q_max], abs=1e-12)


# --- Confidence intervals ------------------------------------------------

SAMPLE = np.array([1.0, 2.5, -0.5, 3.0, 4.0, 0.0, 2.0])


@pytest.mark.parametrize("level", [0.9, 0.95, 0.99])
def test_mean_ci_matches_scipy(level):
    m, h = mean_ci(SAMPLE, level)
    lo, hi = stats.t.interval(level, len(SAMPLE) - 1, loc=SAMPLE.mean(), scale=stats.sem(SAMPLE))
    assert m == pytest.approx(12.0 / 7)
    assert m - h == pytest.approx(lo, rel=1e-12)
    assert m + h == pytest.approx(hi, rel=1e-12)


def test_mean_ci_single_value_is_nan():
    m, h = mean_ci([2.5])
    assert m == 2.5 and np.isnan(h)


def test_summarize():
    s = summarize(SAMPLE)
    m, h = mean_ci(SAMPLE)
    assert s == {"mean": m, "ci95": h, "std": pytest.approx(SAMPLE.std(ddof=1)),
                 "min": -0.5, "max": 4.0, "n": 7}
    one = summarize([3.0])
    assert one["n"] == 1 and one["mean"] == one["min"] == one["max"] == 3.0
    assert np.isnan(one["ci95"]) and np.isnan(one["std"])
