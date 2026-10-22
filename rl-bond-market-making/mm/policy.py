"""Policy and value networks.

Action distribution
-------------------
A Gaussian policy samples a raw quote vector a ~ N(mu(s), diag(std^2)).
The executed quote is clip(a, delta_min, delta_max). The clip is treated as
part of the environment's transition, so the policy is the Gaussian over
the raw action and the score-function (REINFORCE) estimator

    grad J = E[ R * grad log N(a; mu, std) ]      evaluated at the RAW a

is unbiased for the return of exactly this sampling procedure. Evaluating
the density at clip(a) instead (the original notebooks' bug) scores a point
that was not sampled; whenever the clip binds this biases the gradient
(see tests/test_policy.py).
"""

import numpy as np
import torch
import torch.nn as nn


def sample_clipped_gaussian(mean, std, low, high, generator=None):
    """Return (raw sample, executed clipped action, log-prob of the raw sample)."""
    noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
    raw = mean + std * noise
    log_prob = torch.distributions.Normal(mean, std).log_prob(raw.detach()).sum(-1)
    return raw.detach(), torch.clamp(raw.detach(), low, high), log_prob


class _GaussianActor(nn.Module):
    """MLP trunk -> softplus mean quotes, state-independent learnable std."""

    def __init__(self, in_dim, out_dim, hidden, log_std_init, low, high, zero_init=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean_head = nn.Linear(hidden, out_dim)
        if zero_init:
            nn.init.zeros_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.full((out_dim,), float(log_std_init)))
        self.low, self.high = low, high

    def forward(self, x):
        mean = nn.functional.softplus(self.mean_head(self.net(x)))
        return mean, torch.exp(self.log_std).expand_as(mean)

    def act(self, x, generator=None):
        mean, std = self(x)
        return sample_clipped_gaussian(mean, std, self.low, self.high, generator)

    def deterministic(self, x):
        mean, _ = self(x)
        return torch.clamp(mean, self.low, self.high)


class Actor(_GaussianActor):
    """Single-bond policy: (tau, q/q_max) -> (bid, ask) distances."""

    def __init__(self, in_dim=2, hidden=64, low=0.01, high=5.0):
        super().__init__(in_dim, 2, hidden, -1.0, low, high)


class SharedActor(_GaussianActor):
    """Parameter-shared multi-bond policy: per-bond features (n, 4) -> (n, 2).
    Same weights for every bond, so the parameter count does not depend on n."""

    def __init__(self, in_dim=4, hidden=128, low=0.01, high=5.0):
        super().__init__(in_dim, 2, hidden, np.log(0.3), low, high, zero_init=True)


class MultiActor(_GaussianActor):
    """Baseline multi-bond policy: flat state (1 + n) -> 2n quotes."""

    def __init__(self, in_dim, out_dim, hidden=128, low=0.01, high=5.0):
        super().__init__(in_dim, out_dim, hidden, 0.0, low, high)


class MLPCritic(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


class LinearCritic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)


class QuadraticCritic(nn.Module):
    """Linear in [x, x_i x_j (i <= j)]: can represent any quadratic exactly."""

    def __init__(self, in_dim):
        super().__init__()
        self.idx = torch.triu_indices(in_dim, in_dim)
        self.linear = nn.Linear(in_dim + self.idx.shape[1], 1)

    def forward(self, x):
        quad = x[:, self.idx[0]] * x[:, self.idx[1]]
        return self.linear(torch.cat([x, quad], dim=1))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
