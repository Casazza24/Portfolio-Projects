# Section 4 (Draft) — Competitive Market Making with Game Theory

## Overview

Sections 1-3 assume a **monopolist market maker** — order arrival depends only on your own quoted spread. In reality, multiple dealers compete for the same order flow. When a competitor tightens their spread, your fill rate drops even if you haven't changed your quotes.

This section extends our RL framework to model **strategic interaction** between competing market makers using concepts from game theory.

---

## 4.1 The Competition Problem

### Why It Matters

In the single-agent model, the arrival intensity is:

$$\lambda(\delta) = A e^{-k\delta}$$

This only depends on *your* spread $\delta$. But with $N$ competing dealers, a customer picks the best quote. Your effective arrival rate now depends on **everyone's quotes**:

$$\lambda_i(\delta_i, \delta_{-i}) = A \cdot \frac{e^{-k\delta_i}}{\sum_{j=1}^{N} e^{-k\delta_j}}$$

This is a **softmax competition model** — your share of order flow is proportional to how competitive your spread is relative to other dealers. If all dealers quote the same spread, flow is split equally. If you undercut, you get more flow but earn less per trade.

### The Game Theory Connection

This is a **simultaneous-move game** where:
- **Players:** $N$ market makers
- **Strategies:** Each player chooses spreads $(\delta_i^{\text{bid}}, \delta_i^{\text{ask}})$ at each time step
- **Payoffs:** Terminal wealth minus inventory penalty (same as Section 1, but with competitive arrival rates)

The solution concept is a **Nash Equilibrium** — a set of strategies where no single market maker can improve their payoff by unilaterally changing their quotes, given what everyone else is doing.

---

## 4.2 Multi-Agent Environment

```python
class CompetitiveMarketMakingEnv:
    """Market-making environment with N competing dealers."""
    
    def __init__(self, n_dealers=3, T=1.0, dt=0.01, sigma=2.0, 
                 gamma=0.1, k=1.5, A=140.0, q_max=5):
        self.n_dealers = n_dealers
        self.T = T
        self.dt = dt
        self.sigma = sigma
        self.gamma = gamma
        self.k = k
        self.A = A
        self.q_max = q_max
        self.n_steps = int(T / dt)
    
    def reset(self):
        self.t = 0.0
        self.S = 100.0
        # Each dealer has their own inventory and cash
        self.q = np.zeros(self.n_dealers, dtype=int)
        self.X = np.zeros(self.n_dealers)
        self.step_count = 0
        self.done = False
        return self._get_states()
    
    def _get_states(self):
        """Return state for each dealer (they see their own inventory + mid price)."""
        tau = self.T - self.t
        states = []
        for i in range(self.n_dealers):
            states.append(np.array([
                tau, self.q[i] / self.q_max, self.S / 100.0
            ], dtype=np.float32))
        return states
    
    def step(self, actions):
        """
        actions: list of (delta_bid, delta_ask) for each dealer.
        Competition: order flow allocated via softmax over spreads.
        """
        bid_spreads = np.array([a[0] for a in actions])
        ask_spreads = np.array([a[1] for a in actions])
        
        # Softmax competition for bid orders (customer selling to dealer)
        bid_scores = np.exp(-self.k * bid_spreads)
        bid_probs = bid_scores / bid_scores.sum()
        
        # Softmax competition for ask orders (customer buying from dealer)
        ask_scores = np.exp(-self.k * ask_spreads)
        ask_probs = ask_scores / ask_scores.sum()
        
        # Total arrival rate, then allocate to dealers
        total_bid_intensity = self.A * self.dt
        total_ask_intensity = self.A * self.dt
        
        # Simulate which dealer (if any) gets the fill
        if np.random.random() < total_bid_intensity * np.exp(-self.k * bid_spreads.min()):
            winner = np.random.choice(self.n_dealers, p=bid_probs)
            if self.q[winner] < self.q_max:
                self.q[winner] += 1
                self.X[winner] -= (self.S - bid_spreads[winner])
        
        if np.random.random() < total_ask_intensity * np.exp(-self.k * ask_spreads.min()):
            winner = np.random.choice(self.n_dealers, p=ask_probs)
            if self.q[winner] > -self.q_max:
                self.q[winner] -= 1
                self.X[winner] += (self.S + ask_spreads[winner])
        
        # Price evolution
        self.S += self.sigma * np.sqrt(self.dt) * np.random.randn()
        self.t += self.dt
        self.step_count += 1
        
        if self.step_count >= self.n_steps:
            self.done = True
        
        # Rewards for each dealer
        rewards = -self.gamma * self.q**2 * self.dt
        if self.done:
            rewards += self.X + self.q * self.S - self.gamma * self.q**2
        
        return self._get_states(), rewards, self.done
```

---

## 4.3 Finding Nash Equilibrium with Multi-Agent RL

### Approach: Independent Learners

The simplest multi-agent RL approach: each dealer runs its own actor-critic, treating the other dealers as part of the environment. This is called **Independent Learners (IL)**.

```python
def train_competitive(n_dealers=3, n_episodes=3000):
    """Train multiple competing market makers simultaneously."""
    env = CompetitiveMarketMakingEnv(n_dealers=n_dealers)
    
    # Each dealer has its own actor and critic
    actors = [Actor(state_dim=3, hidden=64) for _ in range(n_dealers)]
    critics = [Critic(state_dim=3, hidden=64) for _ in range(n_dealers)]
    # ... training loop where all dealers act simultaneously ...
```

**Key insight:** If independent learners converge to a stable joint policy, that policy approximates a Nash Equilibrium — no single dealer can improve by deviating.

### What We Expect to See

1. **Spread compression:** Competition drives spreads tighter than the monopolist solution. With more dealers, each earns less per trade but must quote aggressively to get any flow at all.

2. **Inventory management is harder:** In monopoly, you control your own fill rate. In competition, aggressive competitors can force fills on you (by quoting wide and pushing flow your way when you're tight).

3. **Emergence of strategies:** Some dealers may specialize — quoting tight on bid but wide on ask, effectively becoming directional. This emerges naturally from the RL training without being programmed.

---

## 4.4 Experiments

### Experiment 1: Monopoly vs Duopoly vs Triopoly

Compare optimal spreads and PnL as the number of competing dealers increases:

| Metric | 1 Dealer (Monopoly) | 2 Dealers | 3 Dealers | 5 Dealers |
|--------|-------------------|-----------|-----------|-----------|
| Avg Bid Spread | (widest) | ... | ... | (tightest) |
| Avg Ask Spread | (widest) | ... | ... | (tightest) |
| Avg PnL per Dealer | (highest) | ... | ... | (lowest) |
| Avg Inventory Risk | (lowest) | ... | ... | (highest) |

**Hypothesis:** Spreads decrease and per-dealer PnL decreases as competition increases — consistent with Bertrand competition from microeconomics.

### Experiment 2: Asymmetric Dealers

What if one dealer has a better model (bigger network, more training) while others use simple strategies?

- Train 1 RL dealer against 2 dealers using the analytical Avellaneda-Stoikov quotes
- Does the RL dealer learn to exploit the fixed-strategy competitors?
- This connects to the CSCI 3202 concept of **best response** in game theory

### Experiment 3: Convergence to Nash Equilibrium

Track whether the joint policy stabilizes:
- Plot each dealer's average spread over training
- If they converge to the same strategy, that's a symmetric Nash Equilibrium
- Measure "exploitability" — how much a best-response deviator could gain against the learned strategies

---

## 4.5 Connection to Course Concepts

This section bridges **three CSCI 3202 topics**:

1. **Game Theory:** Nash Equilibrium, best response, simultaneous games — the competitive market making problem is a repeated simultaneous game where dealers choose spreads each period

2. **Multi-Agent RL:** Independent learners, non-stationarity (each agent's environment changes as others learn), convergence guarantees

3. **Mechanism Design:** The softmax competition model is itself a mechanism — how the "market" allocates order flow determines what strategies are optimal. Different allocation rules lead to different equilibria.

---

## 4.6 Why This Is Hard (and Interesting)

The fundamental challenge: **non-stationarity**. In single-agent RL, the environment is fixed — the same action in the same state always has the same distribution of outcomes. In multi-agent RL, the environment includes other learning agents, so it changes as they learn. This can cause:

- **Oscillation:** Dealer A tightens → Dealer B tightens → both lose money → both widen → cycle repeats
- **Failure to converge:** Policies chase each other without settling
- **Multiple equilibria:** There may be several Nash Equilibria, and different training runs find different ones

These are open research problems, which makes them perfect for a "Future Work" discussion.

---

## References

- Cont, R. & de Larrard, A. (2013). "Price dynamics in a Markovian limit order market." *SIAM J. Financial Mathematics*.
- Baldacci, B., Bergault, P., & Guéant, O. (2021). "Market making and mean-field games." *Mathematics and Financial Economics*.
- Lowe, R. et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." *NeurIPS*.
- Lanctot, M. et al. (2017). "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning." *NeurIPS*.
