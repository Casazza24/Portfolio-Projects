"""Actor-critic training, extracted from rl_market_making_PyTorch.ipynb.

The algorithms are the notebook's: Monte Carlo policy gradient with a
learned state-value baseline (REINFORCE with baseline), one update per
episode, undiscounted returns (finite horizon), gradient-norm clipping.
What changed is what they train on: the shared objective in model.py, the
corrected action log-probabilities in policy.py, and explicit seeds.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .env import MultiBondEnv
from .model import MarketParams
from .policy import Actor, SharedActor, MultiActor, MLPCritic, LinearCritic, QuadraticCritic


def _seed(seed):
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _returns_to_go(rewards):
    r = torch.as_tensor(np.asarray(rewards), dtype=torch.float32)
    return torch.flip(torch.cumsum(torch.flip(r, [0]), 0), [0])


def _update(opt, loss, params, clip=1.0):
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, clip)
    opt.step()


def train_single(params=MarketParams(), n_episodes=2000, seed=0, lr_actor=3e-4, lr_critic=1e-3, hidden=64):
    """Single-bond actor-critic. Returns (actor, critic, episode objectives)."""
    _seed(seed)
    env = MultiBondEnv(params, n_bonds=1, seed=seed)
    actor = Actor(2, hidden, params.delta_min, params.delta_max)
    critic = MLPCritic(2, hidden)
    opt_a, opt_c = optim.Adam(actor.parameters(), lr=lr_actor), optim.Adam(critic.parameters(), lr=lr_critic)
    history = []
    for _ in range(n_episodes):
        obs = env.reset()
        logps, values, rewards = [], [], []
        while not env.done:
            x = torch.from_numpy(obs).unsqueeze(0)
            _, action, logp = actor.act(x)
            values.append(critic(x).squeeze())
            logps.append(logp.squeeze())
            obs, r, _ = env.step(action.numpy())
            rewards.append(r)
        G = _returns_to_go(rewards)
        V = torch.stack(values)
        _update(opt_a, -(torch.stack(logps) * (G - V.detach())).mean(), actor.parameters())
        _update(opt_c, nn.functional.mse_loss(V, G), critic.parameters())
        history.append(float(G[0]))
    return actor, critic, history


def train_shared(params=MarketParams(), n_bonds=2, rho=0.6, n_episodes=500, seed=0, hidden=128):
    """Parameter-shared actor with per-bond credit assignment (notebook 2.4).

    Each bond's log-prob is weighted by its own return-to-go (its spread
    income, mark-to-market and share of the risk penalty, which sum to the
    portfolio reward) minus the portfolio critic's per-bond baseline.
    Returns (actor, critic, episode objectives).
    """
    _seed(seed)
    env = MultiBondEnv(params, n_bonds=n_bonds, correlation=rho, seed=seed)
    actor = SharedActor(4, hidden, params.delta_min, params.delta_max)
    critic = MLPCritic(n_bonds + 1, hidden)
    opt_a, opt_c = optim.Adam(actor.parameters(), lr=3e-4), optim.Adam(critic.parameters(), lr=1e-3)
    history = []
    for _ in range(n_episodes):
        obs = env.reset()
        logps, values, bond_rewards = [], [], []
        while not env.done:
            feats = torch.from_numpy(env.bond_features())
            _, action, logp = actor.act(feats)
            values.append(critic(torch.from_numpy(obs).unsqueeze(0)).squeeze())
            logps.append(logp)
            obs, _, _ = env.step(action.numpy())
            bond_rewards.append(env.bond_rewards.copy())
        G = _returns_to_go(bond_rewards)                  # (T, n)
        V = torch.stack(values)                           # (T,)
        _update(opt_a, -(torch.stack(logps) * (G - V.detach().unsqueeze(1))).mean(), actor.parameters())
        _update(opt_c, nn.functional.mse_loss(V, G.mean(dim=1)), critic.parameters())
        history.append(float(G[0].sum()))
    return actor, critic, history


CRITICS = {
    "linear": lambda d: LinearCritic(d),
    "quadratic": lambda d: QuadraticCritic(d),
    "neural_network": lambda d: MLPCritic(d, 128),
}


def train_with_critic(critic_type, params=MarketParams(), n_bonds=3, rho=0.7, n_episodes=2500, seed=0):
    """Notebook Section 3: flat-state MultiActor with a choice of critic.
    Returns (actor, critic, episode objectives, critic losses)."""
    _seed(seed)
    env = MultiBondEnv(params, n_bonds=n_bonds, correlation=rho, seed=seed)
    actor = MultiActor(1 + n_bonds, 2 * n_bonds, 128, params.delta_min, params.delta_max)
    critic = CRITICS[critic_type](1 + n_bonds)
    opt_a, opt_c = optim.Adam(actor.parameters(), lr=3e-4), optim.Adam(critic.parameters(), lr=1e-3)
    history, losses = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        logps, values, rewards = [], [], []
        while not env.done:
            x = torch.from_numpy(obs).unsqueeze(0)
            _, action, logp = actor.act(x)
            values.append(critic(x).squeeze())
            logps.append(logp.squeeze())
            obs, r, _ = env.step(action.numpy().reshape(n_bonds, 2))
            rewards.append(r)
        G = _returns_to_go(rewards)
        V = torch.stack(values)
        _update(opt_a, -(torch.stack(logps) * (G - V.detach())).mean(), actor.parameters())
        loss = nn.functional.mse_loss(V, G)
        _update(opt_c, loss, critic.parameters())
        history.append(float(G[0]))
        losses.append(loss.item())
    return actor, critic, history, losses
