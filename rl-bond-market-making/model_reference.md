# Model Reference — RL Market Making

A complete reference for all variables, equations, and notation used in `rl_market_making.ipynb`.

---

## Core Variables

| Symbol | Name | Description |
|--------|------|-------------|
| $S_t$ | Mid-price | The "fair" price of the bond at time $t$. Midpoint between best bid and best ask in the market. |
| $q_t$ | Inventory | How many units of the bond the market maker currently holds. Positive = long, negative = short. |
| $X_t$ | Cash | Accumulated cash from trading. Goes up when you sell, down when you buy. |
| $T$ | Terminal time | End of the trading session. At $T$, remaining inventory is liquidated. |
| $\tau = T - t$ | Time remaining | How much time is left. Drives urgency in quoting. |
| $\Delta t$ (dt) | Time step | Discrete simulation step size. Smaller = more realistic but slower. |

---

## Quoting Variables

| Symbol | Name | Description |
|--------|------|-------------|
| $\delta^{\text{bid}}$ | Bid spread | Distance **below** mid-price where the market maker posts a buy order. Bid price = $S_t - \delta^{\text{bid}}$. |
| $\delta^{\text{ask}}$ | Ask spread | Distance **above** mid-price where the market maker posts a sell order. Ask price = $S_t + \delta^{\text{ask}}$. |

**Intuition:** Wider spreads = safer (more profit per trade, fewer fills). Narrower spreads = riskier (less profit per trade, more fills, faster inventory accumulation).

---

## Model Parameters

| Symbol | Name | Notebook Value | Description |
|--------|------|---------------|-------------|
| $\sigma$ | Volatility | 2.0 | How much the mid-price moves per unit time. Higher = more inventory risk. |
| $\gamma$ | Risk aversion | 0.1 | How much the market maker dislikes holding inventory. Higher = more aggressive inventory reduction. |
| $k$ | Arrival sensitivity | 1.5 | How quickly order arrival drops off as spread widens. Higher = more sensitive customers. |
| $A$ | Base arrival rate | 140.0 | Maximum possible order intensity (when spread → 0). |
| $q_{\max}$ | Inventory limit | 5 or 10 | Hard cap on inventory position (prevents blowup). |
| $\rho$ | Correlation | 0.6–0.7 | Correlation between bond price movements (multi-bond only). |

---

## Price Dynamics

$$dS_t = \sigma \, dW_t$$

- This is **arithmetic Brownian motion** (no drift, constant volatility).
- $W_t$ is a Wiener process (standard Brownian motion) — continuous random walk.
- In discrete time: $S_{t+\Delta t} = S_t + \sigma \sqrt{\Delta t} \cdot Z$, where $Z \sim \mathcal{N}(0,1)$.
- **Why no drift?** For market making over short horizons, we assume no directional view — the market maker profits from the spread, not from price prediction.

**Multi-bond version:** For $n$ bonds with correlation matrix $\Sigma$:

$$\mathbf{S}_{t+\Delta t} = \mathbf{S}_t + \sigma \sqrt{\Delta t} \cdot L \mathbf{Z}$$

where $L$ is the Cholesky decomposition of the correlation matrix ($\Sigma = LL^T$) and $\mathbf{Z}$ is a vector of independent standard normals. This ensures the price changes are correlated.

---

## Order Arrival Model

$$\lambda(\delta) = A \cdot e^{-k\delta}$$

- $\lambda(\delta)$ is the **intensity** (expected arrivals per unit time) given a spread of $\delta$.
- Exponential decay: as you widen the spread, fewer customers are willing to trade.
- In discrete time, the probability of a fill in one step: $P(\text{fill}) = \lambda(\delta) \cdot \Delta t = A e^{-k\delta} \Delta t$.

**Why exponential?** It's a standard model in market microstructure (Avellaneda-Stoikov). It captures the empirical observation that order flow is highly sensitive to quoted prices near the market, but insensitive far away.

---

## Objective Function

The market maker maximizes:

$$\max_{\delta^{\text{bid}}, \delta^{\text{ask}}} \; \mathbb{E}\left[ X_T + q_T S_T - \gamma q_T^2 \right]$$

Breaking this down:
- $X_T$ — total cash accumulated from trading
- $q_T S_T$ — mark-to-market value of remaining inventory
- $-\gamma q_T^2$ — **penalty for holding inventory at terminal time**

The penalty term is crucial: without it, the market maker could accumulate huge positions. The quadratic form means the penalty grows rapidly — holding 5 units is 25× worse than holding 1 unit.

---

## The HJB Equation

The Hamilton-Jacobi-Bellman equation for the value function $V(t, q)$:

$$\frac{\partial V}{\partial t} + \frac{\sigma^2}{2}\frac{\partial^2 V}{\partial S^2} + \max_{\delta^a, \delta^b} \left[ \lambda(\delta^a)\left(V(q-1) - V(q) + \delta^a\right) + \lambda(\delta^b)\left(V(q+1) - V(q) + \delta^b\right) \right] = 0$$

**Term by term:**
- $\frac{\partial V}{\partial t}$ — how value changes with time (time decay)
- $\frac{\sigma^2}{2}\frac{\partial^2 V}{\partial S^2}$ — effect of price volatility on value (diffusion term)
- $\lambda(\delta^a)(V(q-1) - V(q) + \delta^a)$ — expected gain from an ask fill: you sell one unit (inventory drops by 1), capture spread $\delta^a$, and your value function shifts accordingly
- $\lambda(\delta^b)(V(q+1) - V(q) + \delta^b)$ — same logic for a bid fill (buy one unit)

**Terminal condition:** $V(T, q) = -\gamma q^2$ (just the inventory penalty at expiry).

---

## Optimal Quotes (Analytical Solution)

From the first-order conditions of the HJB equation:

$$\delta^{\text{bid}*} = \frac{1}{k} \ln\left(1 + \frac{k}{\gamma}\right) + \gamma(T-t) \cdot q$$

$$\delta^{\text{ask}*} = \frac{1}{k} \ln\left(1 + \frac{k}{\gamma}\right) - \gamma(T-t) \cdot q$$

**Interpretation:**

- **Base spread:** $\frac{1}{k} \ln\left(1 + \frac{k}{\gamma}\right)$ — the "symmetric" component. Depends only on parameters, not state. This is the spread you'd quote with zero inventory.
- **Skew term:** $\pm \gamma(T-t) \cdot q$ — the inventory-dependent adjustment.
  - When $q > 0$ (long): bid spread **widens** (less eager to buy more), ask spread **narrows** (eager to sell)
  - When $q < 0$ (short): opposite
  - As $T-t \to 0$ (near terminal): skew increases — urgency to flatten inventory before the penalty hits

---

## Finite Difference Method

**What it does:** Solves the HJB equation numerically by discretizing time and inventory onto a grid, then working backward from the terminal condition.

**Grid:** 
- Time axis: $t \in \{0, \Delta t, 2\Delta t, \ldots, T\}$ — $N_t$ points
- Inventory axis: $q \in \{-q_{\max}, \ldots, -1, 0, 1, \ldots, q_{\max}\}$ — $N_q = 2q_{\max}+1$ points

**Algorithm:**
1. Set $V(T, q) = -\gamma q^2$ for all $q$ (terminal condition)
2. For $t = T-\Delta t, T-2\Delta t, \ldots, 0$:
   - For each inventory $q$:
     - Search over all $(\delta^{\text{bid}}, \delta^{\text{ask}})$ combinations
     - Compute expected value using the Bellman equation
     - Store the best value and corresponding optimal quotes

**Curse of dimensionality:** For $n$ bonds, the inventory grid has $(2q_{\max}+1)^n$ points. With $q_{\max}=5$:
- 1 bond: 11 states
- 2 bonds: 121 states
- 3 bonds: 1,331 states
- 5 bonds: 161,051 states
- 10 bonds: ~25 billion states (impossible)

---

## Actor-Critic RL

### State Representation

The RL agent observes:

$$s = (\tau, \; q/q_{\max}, \; S/100)$$

- $\tau$: time remaining (normalized between 0 and $T$)
- $q/q_{\max}$: inventory normalized to $[-1, 1]$
- $S/100$: price normalized around initial value

For multi-bond: $s = (\tau, \; q_1/q_{\max}, \ldots, q_n/q_{\max}, \; S_1/100, \ldots, S_n/100)$

### Actor (Policy Network)

Maps state → action (bid/ask spreads):

$$\pi_\theta(s) \to (\mu_{\text{bid}}, \mu_{\text{ask}})$$

- Output passes through **softplus** ($\ln(1 + e^x)$) to ensure spreads are always positive
- Actions are sampled from $\mathcal{N}(\mu, \sigma^2)$ during training (exploration)
- $\sigma$ is a learnable parameter (log-std)

### Critic (Value Network)

Maps state → scalar value estimate:

$$V_\phi(s) \to \hat{V}(s)$$

This approximates the true value function from the HJB equation.

### Training Update

**Returns:** For an episode with rewards $r_0, r_1, \ldots, r_T$:

$$G_t = r_t + \gamma_{\text{disc}} \cdot r_{t+1} + \gamma_{\text{disc}}^2 \cdot r_{t+2} + \ldots$$

where $\gamma_{\text{disc}} = 0.99$ is the discount factor (not the same as risk aversion $\gamma$!).

**Advantage:** How much better the actual return was vs. what the critic predicted:

$$A_t = G_t - V_\phi(s_t)$$

**Actor update (policy gradient):**

$$\nabla_\theta J = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t \right]$$

Intuition: if an action led to better-than-expected returns ($A_t > 0$), increase its probability.

**Critic update (MSE):**

$$L_\phi = \frac{1}{N} \sum_t (G_t - V_\phi(s_t))^2$$

Intuition: train the critic to predict actual returns accurately.

---

## Reward Function

**Per-step reward (during episode):**

$$r_t = -\gamma \cdot q_t^2 \cdot \Delta t$$

This is a running inventory penalty — discourages accumulating large positions even mid-episode.

**Terminal reward (at $t = T$):**

$$r_T = X_T + q_T S_T - \gamma q_T^2$$

Total wealth (cash + inventory value) minus final inventory penalty.

**Multi-bond terminal reward:**

$$r_T = X_T + \sum_i q_i S_i - \gamma \sum_i q_i^2 - \gamma \sum_{i<j} \rho_{ij} q_i q_j$$

The cross-term $\rho_{ij} q_i q_j$ penalizes holding correlated positions in the same direction — this is what makes the value function nonlinear and is why linear approximation fails.

---

## Section 3: Linear vs Neural Network

### Linear Critic

$$\hat{V}(s) = \mathbf{w}^T \mathbf{s} + b$$

This is literally linear regression on the state features. It can only represent hyperplanes — it cannot capture the bowl-shaped (quadratic) value function.

### Quadratic Feature Critic

$$\hat{V}(s) = \mathbf{w}^T [\mathbf{s}, \; s_i s_j \; \forall \; i \leq j] + b$$

We manually add all pairwise products as features. This gives linear regression its "best shot" — it can now represent quadratic surfaces. But it still struggles because the true value function has complex time-dependent and boundary interactions that aren't purely quadratic.

### Neural Network Critic

$$\hat{V}(s) = f_\phi(s)$$

A 2-hidden-layer neural network with ReLU activations. Can approximate any continuous function (universal approximation theorem). Automatically discovers the right nonlinear features.

### Why NN Wins

The true value function has:
- **Quadratic** individual inventory terms: $q_i^2$
- **Cross terms** from correlation: $q_i q_j$
- **Time-dependent** curvature: the penalty structure changes as $\tau \to 0$
- **Boundary effects**: behavior near $q_{\max}$ is discontinuous (can't buy/sell more)

A linear model captures none of these. A quadratic feature model captures the first two but not the last two. The neural network captures all of them.

---

## Key Parameter Relationships

| If you increase... | Effect on optimal strategy |
|---|---|
| $\gamma$ (risk aversion) | Narrower base spread, stronger inventory skewing |
| $k$ (arrival sensitivity) | Wider base spread (customers are price-sensitive) |
| $\sigma$ (volatility) | More aggressive inventory reduction (holding is riskier) |
| $\rho$ (correlation) | More cross-hedging between bonds, stronger nonlinearity |
| $A$ (base arrival) | Can afford wider spreads (still get fills) |
| $T-t$ (time left) | Less urgency → weaker skewing. Near terminal → aggressive flattening |

---

## Notation Summary

| Context | $\gamma$ meaning |
|---------|-----------------|
| Model parameter | Risk aversion coefficient (inventory penalty weight) |
| RL discount | $\gamma_{\text{disc}} = 0.99$ (how much future rewards are discounted) |

These are different! The notebook uses `gamma` for risk aversion and `gamma_disc` for the RL discount factor.
