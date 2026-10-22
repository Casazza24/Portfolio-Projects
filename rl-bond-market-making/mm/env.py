"""Simulators that implement exactly the problem in model.py."""

import numpy as np

from .model import MarketParams, correlation_matrix


class MultiBondEnv:
    """n-bond market-making simulator. n_bonds=1 is the single-bond problem.

    Observations exclude the mid price: with driftless prices the optimal
    quotes depend only on (time remaining, inventory), so the price would be
    a spurious input.

    Each step draws exactly 2n uniforms and n normals from the env's own
    Generator, so a seed fixes the whole path of random numbers regardless
    of the quotes posted (common random numbers across policies).
    """

    def __init__(self, params=MarketParams(), n_bonds=1, correlation=0.0, seed=None, S0=100.0):
        self.p = params
        self.n = n_bonds
        self.rho = correlation
        self.C = correlation_matrix(n_bonds, correlation)
        self.cov = params.sigma ** 2 * self.C
        self.chol = np.linalg.cholesky(self.C)
        self.S0 = S0
        self.rng = np.random.default_rng(seed)
        self.bond_index = (np.arange(n_bonds) / max(n_bonds - 1, 1)).astype(np.float32)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.q = np.zeros(self.n, dtype=int)
        self.S = np.full(self.n, self.S0)
        self.X = 0.0
        self.running_penalty_paid = 0.0
        self.done = False
        self.bond_rewards = np.zeros(self.n)
        return self.observation()

    @property
    def tau(self):
        return self.p.T - self.step_count * self.p.dt

    # ------------------------------------------------------------ features
    def observation(self):
        """Flat portfolio observation: (tau, q_1/q_max, ..., q_n/q_max)."""
        return np.concatenate([[self.tau], self.q / self.p.q_max]).astype(np.float32)

    def bond_features(self):
        """(n, 4) per-bond features for a parameter-shared actor:
        (tau, q_i/q_max, covariance-weighted risk (C q)_i/(q_max n), bond index)."""
        return bond_features(self.tau, self.q, self.C, self.p.q_max, self.bond_index)

    # ---------------------------------------------------------------- step
    def step(self, actions):
        """actions: (n, 2) array of (delta_bid, delta_ask); clipped to the
        admissible range. Returns (observation, reward, done)."""
        p = self.p
        if self.done:
            raise RuntimeError("step() called on a finished episode")
        actions = p.clip_delta(np.asarray(actions, dtype=float).reshape(self.n, 2))
        d_bid, d_ask = actions[:, 0], actions[:, 1]

        u = self.rng.random((2, self.n))
        z = self.rng.standard_normal(self.n)

        can_buy = self.q < p.q_max
        can_sell = self.q > -p.q_max
        buy = (u[0] < p.fill_prob(d_bid)) & can_buy
        sell = (u[1] < p.fill_prob(d_ask)) & can_sell

        self.X += np.sum((self.S + d_ask) * sell) - np.sum((self.S - d_bid) * buy)
        self.q = self.q + buy.astype(int) - sell.astype(int)

        dS = np.sqrt(p.dt) * p.sigma * (self.chol @ z)
        self.S = self.S + dS

        # Per-bond share of the risk penalty: sums to (gamma/2) q^T Sigma q dt
        risk_share = 0.5 * p.gamma * self.q * (self.cov @ self.q) * p.dt
        self.running_penalty_paid += risk_share.sum()
        self.bond_rewards = d_bid * buy + d_ask * sell + self.q * dS - risk_share

        self.step_count += 1
        if self.step_count >= p.n_steps:
            self.done = True
            self.bond_rewards = self.bond_rewards - p.terminal_penalty * self.q ** 2

        return self.observation(), float(self.bond_rewards.sum()), self.done

    def objective_realized(self):
        """X_N + q.S_N - S_0 q_0 - running penalties - terminal penalty (q_0 = 0)."""
        out = self.X + self.q @ self.S - self.running_penalty_paid
        if self.done:
            out -= self.p.terminal_penalty * np.sum(self.q ** 2)
        return out


def SingleBondEnv(params=MarketParams(), seed=None):
    """Single-bond problem: MultiBondEnv with one bond."""
    return MultiBondEnv(params, n_bonds=1, correlation=0.0, seed=seed)


def bond_features(tau, q, C, q_max, bond_index):
    """Per-bond features from (tau, inventory vector). Works on one state
    (q of shape (n,)) or a batch of states (q of shape (m, n))."""
    q = np.asarray(q, dtype=float)
    n = q.shape[-1]
    risk = (q @ C.T) / (q_max * n)
    tau = np.asarray(tau, dtype=float)
    tau_arr = np.broadcast_to(tau.reshape(tau.shape + (1,)), q.shape)
    idx = np.broadcast_to(bond_index, q.shape)
    return np.stack([tau_arr, q / q_max, risk, idx], axis=-1).astype(np.float32)


def simulate_single_bond(params, policy, n_episodes, seed=None):
    """Vectorized Monte Carlo of the single-bond problem over many episodes.

    policy(n, q_array) -> (d_bid, d_ask) arrays. Draws random numbers in the
    same order as MultiBondEnv.step (2 uniforms then 1 normal per episode and
    step), so with n_episodes=1 it reproduces MultiBondEnv(n_bonds=1) exactly
    for the same seed. Returns the realized objective of each episode.
    """
    p = params
    rng = np.random.default_rng(seed)
    q = np.zeros(n_episodes, dtype=int)
    total = np.zeros(n_episodes)
    for n in range(p.n_steps):
        d_bid, d_ask = policy(n, q)
        d_bid = p.clip_delta(np.broadcast_to(d_bid, q.shape))
        d_ask = p.clip_delta(np.broadcast_to(d_ask, q.shape))
        u = rng.random((n_episodes, 2))  # same stream as env's (2, 1) draw when n_episodes=1
        z = rng.standard_normal(n_episodes)
        buy = (u[:, 0] < p.fill_prob(d_bid)) & (q < p.q_max)
        sell = (u[:, 1] < p.fill_prob(d_ask)) & (q > -p.q_max)
        q = q + buy.astype(int) - sell.astype(int)
        dS = np.sqrt(p.dt) * p.sigma * z
        total += d_bid * buy + d_ask * sell + q * dS - p.running_penalty * q ** 2 * p.dt
    total -= p.terminal_penalty * q ** 2
    return total


def simulate_multi_bond(params, n_bonds, rho, policy, n_episodes, seed=None):
    """Vectorized Monte Carlo of the n-bond problem.

    policy(n, Q[(E, n)]) -> (E, n, 2) quotes. Returns (objective per episode,
    final inventories). Same per-step objective as MultiBondEnv.
    """
    p = params
    rng = np.random.default_rng(seed)
    C = correlation_matrix(n_bonds, rho)
    cov = p.sigma ** 2 * C
    L = np.linalg.cholesky(C)
    q = np.zeros((n_episodes, n_bonds), dtype=int)
    total = np.zeros(n_episodes)
    for n in range(p.n_steps):
        quotes = p.clip_delta(policy(n, q))
        u = rng.random((2, n_episodes, n_bonds))
        z = rng.standard_normal((n_episodes, n_bonds))
        buy = (u[0] < p.fill_prob(quotes[..., 0])) & (q < p.q_max)
        sell = (u[1] < p.fill_prob(quotes[..., 1])) & (q > -p.q_max)
        q = q + buy.astype(int) - sell.astype(int)
        dS = np.sqrt(p.dt) * p.sigma * (z @ L.T)
        total += (np.sum(quotes[..., 0] * buy + quotes[..., 1] * sell + q * dS, axis=1)
                  - 0.5 * p.gamma * np.einsum("ei,ij,ej->e", q, cov, q) * p.dt)
    total -= p.terminal_penalty * np.sum(q ** 2, axis=1)
    return total, q
