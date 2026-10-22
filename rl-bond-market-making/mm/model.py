"""The market-making control problem, defined once and shared by the
simulator (env.py), the dynamic-programming solvers (dp.py) and the RL code.

Discrete-time problem
---------------------
Time steps n = 0, ..., N-1 with N = T / dt. At step n the dealer holds
inventory q_n (an integer vector, |q_i| <= q_max) and posts, for every bond i,
a bid distance d^b_i and ask distance d^a_i from the mid price, both clipped
to [delta_min, delta_max].

Fills. Independently across bonds and sides,
    B_i ~ Bernoulli(p(d^b_i)) if q_i <  q_max, else 0
    A_i ~ Bernoulli(p(d^a_i)) if q_i > -q_max, else 0
    p(d) = min(1, A exp(-k d) dt)
and inventory moves to q_{n+1} = q_n + B - A. Both boundary checks use the
start-of-step inventory, so q_{n+1} always stays in [-q_max, q_max].

Prices. S_{n+1} = S_n + sqrt(dt) L Z, Z ~ N(0, I), where Sigma = L L^T is the
covariance of price changes per unit time, Sigma = sigma^2 * C with C the
correlation matrix.

Objective. The dealer maximizes

    J = E[ X_N + q_N . S_N
           - sum_n (gamma / 2) q_{n+1}^T Sigma q_{n+1} dt      (running risk)
           - ell * |q_N|^2 ]                                   (terminal)

X is cash. The running term is the mean-variance penalty: over one step the
mark-to-market P&L of holding q_{n+1} has variance q^T Sigma q dt, so
gamma is a genuine risk-aversion coefficient and volatility enters the
objective through Sigma. ell is a terminal liquidation cost per unit^2.

Per-step reward used by the simulator (sums exactly to the objective,
since X_0 = 0, q_0 = 0):

    r_n = B . d^b + A . d^a + q_{n+1} . (S_{n+1} - S_n)
          - (gamma/2) q_{n+1}^T Sigma q_{n+1} dt  [ - ell |q_N|^2 at n = N-1 ]

The price term has zero mean, so the Bellman recursion only needs spread
income and penalties; the price noise matters only for Monte Carlo variance.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class MarketParams:
    T: float = 1.0
    dt: float = 0.005
    sigma: float = 2.0            # price volatility per sqrt(unit time)
    gamma: float = 0.1            # risk aversion (mean-variance)
    k: float = 1.5                # arrival decay in the quote distance
    A: float = 140.0              # arrival intensity at zero distance
    q_max: int = 5
    terminal_penalty: float = 0.1  # ell
    delta_min: float = 0.01
    delta_max: float = 5.0

    @property
    def n_steps(self):
        n = int(round(self.T / self.dt))
        if abs(n * self.dt - self.T) > 1e-9:
            raise ValueError("T must be an integer multiple of dt")
        return n

    @property
    def running_penalty(self):
        """phi = gamma * sigma^2 / 2: the single-bond running coefficient on q^2."""
        return 0.5 * self.gamma * self.sigma ** 2

    @property
    def q_grid(self):
        return np.arange(-self.q_max, self.q_max + 1)

    def fill_prob(self, delta):
        """Per-step fill probability for a quote at distance delta."""
        return np.minimum(1.0, self.A * np.exp(-self.k * np.asarray(delta, dtype=float)) * self.dt)

    def clip_delta(self, delta):
        return np.clip(delta, self.delta_min, self.delta_max)

    def with_(self, **kw):
        return replace(self, **kw)


def correlation_matrix(n_bonds, rho):
    """Equicorrelation matrix with off-diagonal rho."""
    C = np.full((n_bonds, n_bonds), float(rho))
    np.fill_diagonal(C, 1.0)
    return C


def covariance_matrix(params, n_bonds, rho):
    return params.sigma ** 2 * correlation_matrix(n_bonds, rho)
