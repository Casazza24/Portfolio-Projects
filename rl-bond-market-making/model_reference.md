# Model Reference: RL Market Making

Notation and equations for the market-making problem studied in this repo.

**Which code this describes.** `rl_market_making_PyTorch.ipynb` is the current, maintained implementation and imports the model, simulator, solvers and policies from the `mm/` package. `rl_market_making.ipynb` (TensorFlow) is an earlier iteration kept for history. The section "Current model" below matches `mm/`. The section "Historical formulation" records what the original notebooks did and what was wrong with it.

---

# Current model

The single source of truth is `mm/model.py` (problem definition), `mm/env.py` (simulator), `mm/dp.py` (exact benchmarks) and `mm/policy.py` (action distribution and gradient estimator).

## Notation

### State and time

| Symbol | Name | Description |
|--------|------|-------------|
| $n = 0, \ldots, N-1$ | Step index | Discrete time steps, $N = T / \Delta t$ ($T$ must be an integer multiple of $\Delta t$). |
| $T$ | Horizon | End of the session. Remaining inventory is marked to market and charged a liquidation cost. |
| $\Delta t$ (`dt`) | Time step | Simulation step size. |
| $\tau = T - n\Delta t$ | Time remaining | Drives the urgency of flattening inventory. |
| $\mathbf{S}_n$ | Mid prices | Vector of bond mid prices. |
| $\mathbf{q}_n$ | Inventory | Integer vector, $\lvert q_i \rvert \le q_{\max}$. Positive is long, negative is short. $\mathbf{q}_0 = 0$. |
| $X_n$ | Cash | Increases on a sale, decreases on a purchase. $X_0 = 0$. |

### Quotes

| Symbol | Name | Description |
|--------|------|-------------|
| $\delta^b_i$ | Bid distance | Bid price is $S_i - \delta^b_i$. |
| $\delta^a_i$ | Ask distance | Ask price is $S_i + \delta^a_i$. |
| $[\delta_{\min}, \delta_{\max}]$ | Admissible box | Executed quotes are clipped to this range. |

Wider quotes earn more per fill but fill less often, so inventory changes more slowly.

### Parameters (defaults from `MarketParams`)

| Symbol | Code name | Default | Description |
|--------|-----------|---------|-------------|
| $T$ | `T` | 1.0 | Horizon. |
| $\Delta t$ | `dt` | 0.005 | Step size ($N = 200$). |
| $\sigma$ | `sigma` | 2.0 | Price volatility per $\sqrt{\text{unit time}}$. |
| $\gamma$ | `gamma` | 0.1 | Risk aversion (mean-variance coefficient). |
| $k$ | `k` | 1.5 | Decay of arrival intensity in the quote distance. |
| $A$ | `A` | 140.0 | Arrival intensity at zero distance. |
| $q_{\max}$ | `q_max` | 5 | Inventory limit per bond. |
| $\ell$ | `terminal_penalty` | 0.1 | Terminal liquidation cost per unit$^2$. |
| $\delta_{\min}, \delta_{\max}$ | `delta_min`, `delta_max` | 0.01, 5.0 | Quote box. |
| $\rho$ | `correlation` | 0 | Pairwise price correlation (multi-bond; experiments use 0.6 to 0.7). |
| $\phi = \gamma\sigma^2/2$ | `running_penalty` | 0.2 | Single-bond running coefficient on $q^2$. |

**Notation note.** $\gamma$ is always risk aversion. The RL code uses undiscounted returns (finite horizon), so there is no discount factor in the current model. $A$ (intensity) and $A_i$ (ask fill indicator) are distinguished by the subscript.

## Price dynamics

Arithmetic Brownian motion with no drift: $d\mathbf{S}_t = L\, d\mathbf{W}_t$ with covariance per unit time

$$\Sigma = \sigma^2 C, \qquad C_{ii} = 1,\; C_{ij} = \rho \;(i \ne j).$$

In discrete time,

$$\mathbf{S}_{n+1} = \mathbf{S}_n + \sigma\sqrt{\Delta t}\, L_C \mathbf{Z}_n, \qquad \mathbf{Z}_n \sim \mathcal{N}(0, I),\quad C = L_C L_C^T,$$

so $\operatorname{Cov}(\mathbf{S}_{n+1} - \mathbf{S}_n) = \Sigma\,\Delta t$. No drift means the dealer has no directional view and earns from the spread. Because prices are driftless, the optimal quotes depend only on $(\tau, \mathbf{q})$, and the observation given to the RL agent excludes prices.

## Fill model

The arrival intensity at distance $\delta$ is $\lambda(\delta) = A e^{-k\delta}$. The per-step fill probability is

$$p(\delta) = \min\left(1,\; A e^{-k\delta}\,\Delta t\right).$$

For each bond $i$, independently across bonds and across the two sides,

$$B_i \sim \text{Bern}\big(p(\delta^b_i)\big) \text{ if } q_{n,i} < q_{\max}, \text{ else } 0, \qquad A_i \sim \text{Bern}\big(p(\delta^a_i)\big) \text{ if } q_{n,i} > -q_{\max}, \text{ else } 0,$$

$$\mathbf{q}_{n+1} = \mathbf{q}_n + \mathbf{B} - \mathbf{A}.$$

- Both limit checks use the start-of-step inventory $\mathbf{q}_n$, so $\mathbf{q}_{n+1}$ always stays in $[-q_{\max}, q_{\max}]$.
- Both sides can fill in the same step (probability $p(\delta^b)p(\delta^a)$). Inventory is then unchanged and the dealer earns $\delta^b + \delta^a$.
- At the defaults $A\Delta t = 0.7 < 1$, so the cap at 1 never binds on the quote box.

## Objective

The dealer maximizes

$$J = \mathbb{E}\left[ X_N + \mathbf{q}_N \cdot \mathbf{S}_N \;-\; \sum_{n=0}^{N-1} \frac{\gamma}{2}\, \mathbf{q}_{n+1}^T \Sigma\, \mathbf{q}_{n+1}\, \Delta t \;-\; \ell\, \lvert \mathbf{q}_N \rvert^2 \right].$$

- $X_N + \mathbf{q}_N \cdot \mathbf{S}_N$: terminal wealth (cash plus inventory marked at mid).
- Running term: a mean-variance penalty. Over one step, holding $\mathbf{q}_{n+1}$ has mark-to-market P&L $\mathbf{q}_{n+1} \cdot (\mathbf{S}_{n+1} - \mathbf{S}_n)$ with variance $\mathbf{q}_{n+1}^T \Sigma\, \mathbf{q}_{n+1}\,\Delta t$. Summing $\frac{\gamma}{2}\times$ variance over steps makes $\gamma$ a genuine risk-aversion coefficient and is how volatility and correlation enter the objective. For one bond the term is $\phi\, q_{n+1}^2 \Delta t$ with $\phi = \gamma\sigma^2/2$.
- $\ell \lvert \mathbf{q}_N \rvert^2$: terminal liquidation cost, separate from risk aversion.

### Per-step reward

The simulator pays

$$r_n = \mathbf{B}\cdot\boldsymbol{\delta}^b + \mathbf{A}\cdot\boldsymbol{\delta}^a + \mathbf{q}_{n+1}\cdot(\mathbf{S}_{n+1} - \mathbf{S}_n) - \frac{\gamma}{2}\,\mathbf{q}_{n+1}^T \Sigma\, \mathbf{q}_{n+1}\,\Delta t \;\;\big[-\,\ell\,\lvert\mathbf{q}_N\rvert^2 \text{ at } n = N-1\big].$$

These rewards telescope to the objective. The cash change in a step is $\Delta X_n = \mathbf{B}\cdot\boldsymbol{\delta}^b + \mathbf{A}\cdot\boldsymbol{\delta}^a - \mathbf{S}_n\cdot(\mathbf{q}_{n+1} - \mathbf{q}_n)$, and $\Delta(\mathbf{q}\cdot\mathbf{S}) = \mathbf{q}_{n+1}\cdot(\mathbf{S}_{n+1} - \mathbf{S}_n) + \mathbf{S}_n\cdot(\mathbf{q}_{n+1} - \mathbf{q}_n)$. Adding them, the $\mathbf{S}_n$ terms cancel, and with $X_0 = 0$, $\mathbf{q}_0 = 0$,

$$\sum_{n=0}^{N-1} r_n = X_N + \mathbf{q}_N\cdot\mathbf{S}_N - \sum_n \frac{\gamma}{2}\,\mathbf{q}_{n+1}^T\Sigma\,\mathbf{q}_{n+1}\Delta t - \ell\lvert\mathbf{q}_N\rvert^2.$$

The price term has zero conditional mean, so it drops out of the Bellman recursion and only adds Monte Carlo variance.

For the multi-bond shared actor the reward is split per bond, with bond $i$ charged the risk share $\frac{\gamma}{2} q_i (\Sigma \mathbf{q})_i \Delta t$. The shares sum to the portfolio penalty.

## Single bond: discrete Bellman recursion

This is the exact optimum of the simulator the RL agent trains in (`dp.solve_single_bond`). The original notebooks called it the finite-difference (FD) solution. Define the post-fill value net of the running penalty

$$W_{n+1}(q') = V_{n+1}(q') - \phi\, q'^2\,\Delta t, \qquad V_N(q) = -\ell q^2.$$

With $p_b = p(\delta^b)\,\mathbb{1}[q < q_{\max}]$ and $p_a = p(\delta^a)\,\mathbb{1}[q > -q_{\max}]$,

$$V_n(q) = \max_{\delta^b, \delta^a} \Big[ p_b\delta^b + p_a\delta^a + p_b(1-p_a)\,W_{n+1}(q+1) + p_a(1-p_b)\,W_{n+1}(q-1) + \big(p_b p_a + (1-p_b)(1-p_a)\big)\,W_{n+1}(q) \Big].$$

The last coefficient includes the both-sides-fill outcome $p_b p_a$, which leaves inventory unchanged. $V_0(0)$ is the optimal expected objective; at the defaults it is about 66.2.

### Per-quote maximizer

Holding $\delta^a$ fixed, the bracket depends on $\delta^b$ as $p_b(\delta^b + C_b) + \text{const}$, where

$$C_b = \big[(1-p_a)W(q+1) + p_a W(q)\big] - \big[(1-p_a)W(q) + p_a W(q-1)\big]$$

is the expected gain in continuation value from a bid fill, averaged over the ask outcome. Symmetrically $C_a = \big[(1-p_b)W(q-1) + p_b W(q)\big] - \big[(1-p_b)W(q) + p_b W(q+1)\big]$. Where the cap does not bind, $p(\delta) = A\Delta t\, e^{-k\delta}$, and

$$f(\delta) = e^{-k\delta}(\delta + C), \qquad f'(\delta) = e^{-k\delta}\big(1 - k(\delta + C)\big),$$

which is positive for $\delta < 1/k - C$ and negative above it. $f$ is unimodal, so its maximum over an interval is the unconstrained maximizer clipped to the interval:

$$\delta^{b*} = \operatorname{clip}\!\left(\frac{1}{k} - C_b,\; \delta_{\min}, \delta_{\max}\right), \qquad \delta^{a*} = \operatorname{clip}\!\left(\frac{1}{k} - C_a,\; \delta_{\min}, \delta_{\max}\right).$$

Because $C_b$ depends on $p_a$ and $C_a$ on $p_b$, the solver iterates the two coordinates to a fixed point; `tests/test_single_bond.py` checks the result against brute-force search over the quote box. If $A\Delta t > 1$, the solver raises the lower bound to $\ln(A\Delta t)/k$, since below it $p = 1$ is flat and a wider quote is strictly better.

The same recursion with fixed quotes (`dp.evaluate_single_bond_policy`) gives the exact expected objective of any Markov policy, with no Monte Carlo error.

## Single bond: continuous-time limit

As $\Delta t \to 0$, without quote bounds, write the value as $x + qS + h(t, q)$. The $S$-diffusion term vanishes (the value is linear in $S$), so $\sigma$ enters only through $\phi$. The HJB equation is

$$\frac{\partial h}{\partial t} - \phi q^2 + \max_{\delta^b} A e^{-k\delta^b}\big(\delta^b + h(q+1) - h(q)\big) + \max_{\delta^a} A e^{-k\delta^a}\big(\delta^a + h(q-1) - h(q)\big) = 0, \qquad h(T, q) = -\ell q^2,$$

with the bid term absent at $q = q_{\max}$ and the ask term absent at $q = -q_{\max}$. Each inner maximum is attained at $\delta^* = 1/k - \big(h(q\pm1) - h(q)\big)$ with value $\frac{A}{e k}\exp\big(k[h(q\pm1) - h(q)]\big)$. The substitution $h = \frac{1}{k}\ln w$ makes the equation linear:

$$\frac{dw}{dt} = -M w, \qquad M_{qq} = -k\phi q^2, \quad M_{q,q\pm1} = \frac{A}{e},$$

with $M$ tridiagonal and zero beyond the inventory limits. Hence

$$w(t) = \exp\big(M(T - t)\big)\, w(T), \qquad w(T, q) = e^{-k\ell q^2},$$

$$\delta^{b*}(t, q) = \frac{1}{k} - \big(h(t, q+1) - h(t, q)\big), \qquad \delta^{a*}(t, q) = \frac{1}{k} - \big(h(t, q-1) - h(t, q)\big).$$

(`dp.continuous_time_solution`; Cartea, Jaimungal and Penalva 2015, ch. 10.) This is an independent check on the discrete recursion, not the same object at finite $\Delta t$: at the defaults $h(0, 0) \approx 65.2$ versus $V_0(0) \approx 66.2$. With a wide quote box the discrete value converges to it at first order in $\Delta t$ (`test_dp_converges_to_continuous_time_closed_form`).

## Multi-bond

For $n$ bonds the running penalty is

$$\frac{\gamma}{2}\,\mathbf{q}^T\Sigma\,\mathbf{q}\,\Delta t = \phi\Big(\sum_i q_i^2 + 2\rho\sum_{i<j} q_i q_j\Big)\Delta t.$$

### Separable formulation (`dp.solve_separable`)

Quote every bond with the single-bond optimum, ignoring cross-bond covariance. When $\rho = 0$, fills, prices and penalties decompose bond by bond, so $V(\mathbf{q}) = \sum_i V_1(q_i)$ and this policy is exact. For $\rho \ne 0$ it is suboptimal (the coupled solver strictly beats it from $\mathbf{q} = 0$ in the two-bond tests for $\rho \in \{0.3, 0.6, 0.9\}$). Its value does not depend on $\rho$ when starting from $\mathbf{q}_0 = 0$: under this policy inventories evolve independently, and by the bid/ask symmetry of the single-bond solution each has mean zero, so $\mathbb{E}[q_i q_j] = \mathbb{E}[q_i]\,\mathbb{E}[q_j] = 0$ and the expected cross term vanishes. Its cost is one single-bond solve, independent of $n$ for identical bonds.

### Coupled formulation (`dp.solve_coupled`)

The exact optimum with the full covariance penalty, on the grid of $(2q_{\max}+1)^n$ inventory vectors. The recursion is the multi-bond version of the one above, with $W_{n+1}(\mathbf{q}') = V_{n+1}(\mathbf{q}') - \frac{\gamma}{2}\mathbf{q}'^T\Sigma\,\mathbf{q}'\,\Delta t$, $V_N(\mathbf{q}) = -\ell\lvert\mathbf{q}\rvert^2$, and the expectation over all $3^n$ net moves in $\{-1, 0, 1\}^n$ (per bond: $+1$ with probability $p_b(1-p_a)$, $0$ with $p_b p_a + (1-p_b)(1-p_a)$, $-1$ with $p_a(1-p_b)$). The $2n$ quotes per state are found by coordinate ascent, each coordinate using the same clipped maximizer $\operatorname{clip}(1/k - C)$, with $C$ now averaged over the other bonds' moves as well. `dp.evaluate_coupled_policy` scores any Markov policy exactly under the coupled objective.

**Curse of dimensionality.** With $q_{\max} = 5$ the grid has $11^n$ states: 11 (1 bond), 121 (2), 1,331 (3), 161,051 (5), about 25.9 billion (10). The coupled solver is intended as ground truth for $n \le 3$; beyond that, RL and Monte Carlo evaluation are used.

## Actor-critic RL

### Observations

- Single bond and flat multi-bond actor: $s = (\tau,\; q_1/q_{\max}, \ldots, q_n/q_{\max})$. Prices are excluded (see Price dynamics).
- Parameter-shared multi-bond actor: per-bond features $\big(\tau,\; q_i/q_{\max},\; (C\mathbf{q})_i/(q_{\max} n),\; i/(n-1)\big)$, with the same weights for every bond.

### Actor and action distribution

The network outputs a mean quote $\mu_\theta(s)$ through a softplus, and a state-independent learnable log-std. A raw action is sampled as $a \sim \mathcal{N}(\mu_\theta(s), \operatorname{diag}(\text{std}^2))$, and the executed quote is $\operatorname{clip}(a, \delta_{\min}, \delta_{\max})$. The clip is treated as part of the environment, so the policy is the Gaussian over the raw action and the score-function estimator is

$$\nabla_\theta J = \mathbb{E}\Big[\nabla_\theta \log \mathcal{N}\big(a;\, \mu_\theta(s), \text{std}\big)\cdot \hat{A}\Big], \quad \text{evaluated at the raw (unclipped) sample } a.$$

This is unbiased for the return of exactly this sampling procedure. Evaluated deterministically, the policy executes $\operatorname{clip}(\mu_\theta(s))$.

### Training

Monte Carlo policy gradient with a learned state-value baseline (REINFORCE with baseline), one update per episode, gradient-norm clipping at 1.0:

$$G_n = \sum_{m \ge n} r_m \;\;(\text{undiscounted}), \qquad \hat{A}_n = G_n - V_\psi(s_n), \qquad L_\psi = \frac{1}{N}\sum_n \big(G_n - V_\psi(s_n)\big)^2.$$

The shared multi-bond actor weights bond $i$'s log-probability by its own return-to-go (computed from the per-bond reward split above) minus the critic baseline.

### Critics compared (Section 3 of the notebook)

- Linear: $\hat V(s) = \mathbf{w}^T s + b$. Can represent only hyperplanes, so not the bowl shape of the value in $\mathbf{q}$.
- Quadratic features: $\hat V(s) = \mathbf{w}^T[s,\; s_i s_j\;\forall\, i \le j] + b$. Can represent any quadratic in the features exactly, including the $q_i q_j$ cross terms.
- Neural network: a 2-hidden-layer ReLU MLP $f_\psi(s)$.

The exact value function is not purely quadratic: its curvature changes with $\tau$ and it has boundary effects at $\lvert q_i\rvert = q_{\max}$ (one side cannot trade), which the quadratic critic cannot represent exactly.

## Parameter relationships

Computed with `dp.solve_single_bond` by changing one default at a time (spread is $\delta^b + \delta^a$ at $q = 0$, $t = 0$; skew is $\delta^b(q{=}1) - \delta^b(q{=}0)$):

| If you increase... | Effect on the optimal quotes |
|---|---|
| $\gamma$ (risk aversion) | Stronger inventory skew; spread at $q = 0$ slightly wider. |
| $\sigma$ (volatility) | Same direction as $\gamma$, since both enter only through $\phi = \gamma\sigma^2/2$. |
| $k$ (arrival decay) | Narrower spread (the base distance scales like $1/k$) and weaker skew. |
| $A$ (arrival intensity) | Slightly narrower spread and weaker skew. |
| $\ell$ (terminal cost) | Stronger skew near $T$; little effect early in the session. |
| $\rho$ (correlation) | Inventory in one bond skews the optimal quotes of the others (`test_correlated_inventory_skews_the_other_bond`); the separable policy ignores this. |
| $\tau$ (time left) | At the defaults the skew is about 0.06 at $\tau = T$ and 0.19 one step before $T$. |

---

# Historical formulation (original notebooks)

This section records the formulation used by the original notebooks (TensorFlow first, then the first PyTorch version) before the correctness pass. It is kept for history and is **not** the current model. `mm/legacy.py` reproduces the old FD solver and the old closed-form quotes so they can be scored under the corrected objective.

## Objective (historical)

$$\max_{\delta^{\text{bid}}, \delta^{\text{ask}}} \; \mathbb{E}\left[ X_T + q_T S_T - \gamma q_T^2 \right]$$

- $X_T$: total cash accumulated from trading
- $q_T S_T$: mark-to-market value of remaining inventory
- $-\gamma q_T^2$: penalty for holding inventory at terminal time, the only penalty in this objective

The RL environment, however, paid a running penalty during the episode and a terminal reward at the end:

$$r_t = -\gamma\, q_t^2\, \Delta t, \qquad r_T = X_T + q_T S_T - \gamma q_T^2.$$

Multi-bond terminal reward:

$$r_T = X_T + \sum_i q_i S_i - \gamma \sum_i q_i^2 - \gamma \sum_{i<j} \rho_{ij} q_i q_j$$

The PyTorch notebook used undiscounted returns; the TensorFlow multi-bond trainer discounted with $\gamma_{\text{disc}} = 0.99$. The state included normalized prices $S/100$.

## HJB equation (historical text)

The Hamilton-Jacobi-Bellman equation for the value function $V(t, q)$:

$$\frac{\partial V}{\partial t} + \frac{\sigma^2}{2}\frac{\partial^2 V}{\partial S^2} + \max_{\delta^a, \delta^b} \left[ \lambda(\delta^a)\left(V(q-1) - V(q) + \delta^a\right) + \lambda(\delta^b)\left(V(q+1) - V(q) + \delta^b\right) \right] = 0$$

**Term by term:**
- $\frac{\partial V}{\partial t}$: how value changes with time
- $\frac{\sigma^2}{2}\frac{\partial^2 V}{\partial S^2}$: effect of price volatility on value (diffusion term)
- $\lambda(\delta^a)(V(q-1) - V(q) + \delta^a)$: expected gain from an ask fill (sell one unit, capture $\delta^a$)
- $\lambda(\delta^b)(V(q+1) - V(q) + \delta^b)$: same for a bid fill (buy one unit)

**Terminal condition:** $V(T, q) = -\gamma q^2$.

The FD solver discretized this backward from the terminal condition on the grid $t \in \{0, \Delta t, \ldots, T\}$, $q \in \{-q_{\max}, \ldots, q_{\max}\}$, choosing quotes at each state by the first-order condition.

## "Analytical" optimal quotes (historical)

$$\delta^{\text{bid}*} = \frac{1}{k} \ln\left(1 + \frac{k}{\gamma}\right) + \gamma(T-t) \cdot q, \qquad \delta^{\text{ask}*} = \frac{1}{k} \ln\left(1 + \frac{k}{\gamma}\right) - \gamma(T-t) \cdot q$$

These were described as a base spread $\frac{1}{k}\ln(1 + k/\gamma)$ plus an inventory skew $\pm\gamma(T-t)q$.

## What was wrong

a. **Two different objectives.** The RL environment used a running plus terminal penalty, while the FD solver used the terminal penalty only. The RL agent and the benchmark it was compared against were optimizing different problems.

b. **Volatility did not enter.** $\sigma$ appeared in the HJB text but not in the FD recursion or in the objective. The value is linear in $S$, so the $\frac{\sigma^2}{2}V_{SS}$ term is zero, and with no variance penalty the solution was the same for every $\sigma$.

c. **Missing both-fill term.** The FD recursion added the bid and ask terms separately, which drops the probability that both sides fill in the same step. The simulator allowed both fills, so the recursion was not the Bellman equation of the simulator.

d. **The closed form is not Avellaneda-Stoikov.** The Avellaneda-Stoikov approximation has a spread term $\frac{1}{\gamma}\ln(1 + \gamma/k)$ (not $\frac{1}{k}\ln(1 + k/\gamma)$) and an inventory skew $\gamma\sigma^2(T-t)q$ (not $\gamma(T-t)q$). Scored exactly under the corrected objective at the defaults, the historical formula earns about 30.5, against an optimum of about 66.2.

e. **Multi-bond penalty and solver.** The penalty $\gamma\big(\sum_i q_i^2 + \sum_{i<j}\rho\, q_i q_j\big)$ gives the cross terms half the weight they have in portfolio variance, $\sum_i q_i^2 + 2\rho\sum_{i<j} q_i q_j$ (times $\sigma^2$). The multi-bond FD solver ignored correlation entirely.

f. **Biased policy gradient.** The log-probability was evaluated at the clamped action instead of the raw Gaussian sample. Whenever the clamp binds this scores a point that was not sampled. In a test near the lower bound ($\mu = 0.05$, std $= 0.3$, about 45% of samples clamped), the estimated gradient with respect to the mean had the wrong sign (true about $+0.04$, estimate about $-0.05$; `tests/test_policy.py`).

---

# References

- Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224. Source of the exponential arrival model $\lambda(\delta) = A e^{-k\delta}$.
- Cartea, Á., Jaimungal, S. and Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press, ch. 10. Source of the running inventory penalty and the matrix-exponential closed form used in `dp.continuous_time_solution`.
