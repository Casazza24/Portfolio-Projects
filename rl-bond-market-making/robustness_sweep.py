"""Robustness sweep: retrain single-bond agents under different parameters.

Designed to be called from the notebook via ProcessPoolExecutor.
All classes are self-contained so they can be pickled across processes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---- Duplicated env/model classes (must be top-level for pickling) ----

class MarketMakingEnv:
    def __init__(self, T=1.0, dt=0.005, sigma=2.0, gamma=0.1, k=1.5, A=140.0, q_max=10):
        self.T = T
        self.dt = dt
        self.sigma = sigma
        self.gamma = gamma
        self.k = k
        self.A = A
        self.q_max = q_max
        self.n_steps = int(T / dt)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.S = 100.0
        self.q = 0
        self.X = 0.0
        self.step_count = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        tau = self.T - self.t
        return np.array([tau, self.q / self.q_max, self.S / 100.0], dtype=np.float32)

    def step(self, action):
        delta_bid, delta_ask = action
        delta_bid = max(delta_bid, 0.01)
        delta_ask = max(delta_ask, 0.01)

        lambda_bid = self.A * np.exp(-self.k * delta_bid) * self.dt
        lambda_ask = self.A * np.exp(-self.k * delta_ask) * self.dt

        bid_fill = np.random.random() < lambda_bid
        ask_fill = np.random.random() < lambda_ask

        if bid_fill and self.q < self.q_max:
            self.q += 1
            self.X -= (self.S - delta_bid)
        if ask_fill and self.q > -self.q_max:
            self.q -= 1
            self.X += (self.S + delta_ask)

        self.S += self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.t += self.dt
        self.step_count += 1

        if self.step_count >= self.n_steps:
            self.done = True

        reward = -self.gamma * (self.q ** 2) * self.dt
        if self.done:
            reward += self.X + self.q * self.S - self.gamma * (self.q ** 2)

        return self._get_state(), reward, self.done


class Actor(nn.Module):
    def __init__(self, state_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), -1.0))

    def forward(self, state):
        x = self.net(state)
        mean = torch.nn.functional.softplus(self.mean_head(x))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(self, state, deterministic=False):
        mean, std = self.forward(state)
        if deterministic:
            return mean
        dist = torch.distributions.Normal(mean, std)
        raw = dist.sample()
        # Score the sampled (raw) action. The clip is part of execution, not of the
        # policy density; log_prob(clamped) would bias the policy gradient.
        log_prob = dist.log_prob(raw).sum(-1)
        action = torch.clamp(raw, 0.01, 5.0)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state)


# ---- Training ----

def train_actor_critic(env, n_episodes=2000):
    actor = Actor(state_dim=3, hidden=64)
    critic = Critic(state_dim=3, hidden=64)
    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    for ep in range(n_episodes):
        state = env.reset()
        states, log_probs, values, rewards = [], [], [], []

        while not env.done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = actor.get_action(state_t)
            value = critic(state_t)
            next_state, reward, done = env.step(action.detach().numpy().flatten())

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        values_t = torch.cat(values).squeeze()
        log_probs_t = torch.stack(log_probs)
        advantages = returns - values_t.detach()

        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = nn.MSELoss()(values_t, returns)

        opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt_actor.step()

        opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt_critic.step()

    return actor


# ---- Single sweep point (called by each worker) ----

def run_single_sweep(args):
    """Train one agent and evaluate. args = (param_name, param_value, n_episodes)."""
    param_name, param_value, n_episodes = args

    base = {'T': 1.0, 'dt': 0.005, 'sigma': 2.0, 'gamma': 0.1,
            'k': 1.5, 'A': 140.0, 'q_max': 5}
    base[param_name] = param_value

    env = MarketMakingEnv(**base)
    actor = train_actor_critic(env, n_episodes=n_episodes)

    with torch.no_grad():
        spreads_collected = []
        for tau in [0.3, 0.5, 0.7]:
            state = torch.tensor([tau, 0.0, 1.0], dtype=torch.float32).unsqueeze(0)
            mean, _ = actor(state)
            spreads_collected.append(mean.squeeze().numpy())
        avg_spreads = np.mean(spreads_collected, axis=0)

    result = {
        'param_name': param_name,
        'param_value': param_value,
        'mean_bid_spread': float(avg_spreads[0]),
        'mean_ask_spread': float(avg_spreads[1]),
        'mean_total_spread': float(avg_spreads[0] + avg_spreads[1]),
    }
    print(f"  {param_name}={param_value}: bid={avg_spreads[0]:.4f}, ask={avg_spreads[1]:.4f}")
    return result


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    import json

    jobs = []
    for val in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        jobs.append(('gamma', val, 2500))
    for val in [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        jobs.append(('sigma', val, 2500))
    for val in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        jobs.append(('k', val, 2000))
    print(f"Running {len(jobs)} sweep jobs across available cores...")
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_single_sweep, jobs))

    with open('sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to sweep_results.json")
