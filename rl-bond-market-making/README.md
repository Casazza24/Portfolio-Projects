# RL Bond Market Making

**[Live Demo](https://casazza24.github.io/Portfolio-Projects/rl-bond-market-making/demo.html)**

A reinforcement learning approach to optimal market making, benchmarked directly against the analytical and finite-difference solution to a classical stochastic control problem. Built as a final project for CSCI 3202 (Artificial Intelligence). A single-agent actor-critic learns to quote bid and ask spreads on a simulated bond, is checked against the theoretical optimum, and is then extended to a five-bond portfolio where the state space makes exact solution infeasible and only the learned policy remains tractable.

## Problem Statement

Market making is the business of continuously quoting a bid price (where you will buy) and an ask price (where you will sell) around a security's mid-price, earning the spread between the two while managing the inventory risk that comes from being filled unevenly on either side. The classical formulation of this problem is the Avellaneda-Stoikov framework: the mid-price follows arithmetic Brownian motion, $dS_t = \sigma \, dW_t$, order arrivals on each side occur with intensity $\lambda(\delta) = A e^{-k\delta}$ that decays exponentially in the quoted spread $\delta$, and the market maker chooses bid and ask spreads at each instant to maximize terminal wealth minus a quadratic penalty on ending inventory.

This problem has a closed-form solution for a single bond via the Hamilton-Jacobi-Bellman (HJB) equation, and that solution can also be recovered numerically with a finite-difference (FD) scheme that discretizes time and inventory and works backward from the terminal condition. But the FD approach suffers from the curse of dimensionality: the inventory grid for $n$ bonds has $(2q_{\max}+1)^n$ states. With an inventory cap of 5, one bond has 11 states, five bonds have 161,051 states, and ten bonds would require roughly 25 billion states, which is computationally impossible to solve exactly. This is exactly the setting where a function-approximation-based method, like an RL agent parameterized by a neural network, becomes necessary rather than optional. The project's core question is whether an RL agent can learn a policy that matches the FD/analytical optimum where it exists (single bond), and then scale to a regime where FD cannot go (multiple correlated bonds).

## Methodology

**Finite-difference solver.** The FD solver discretizes the HJB equation over a time grid and an inventory grid ($q \in \{-q_{\max}, \ldots, q_{\max}\}$), sets the terminal condition $V(T, q) = -\gamma q^2$, and works backward in time. At each grid point it searches over candidate bid/ask spread pairs, computes the expected value under the Bellman equation using the order arrival model, and stores the value-maximizing quotes. This produces both a numerical value function $V(t, q)$ and the corresponding optimal quoting policy, which also has a known closed form:

$$\delta^{\text{bid}\ast} = \frac{1}{k}\ln\left(1 + \frac{k}{\gamma}\right) + \gamma(T-t)q, \qquad \delta^{\text{ask}\ast} = \frac{1}{k}\ln\left(1 + \frac{k}{\gamma}\right) - \gamma(T-t)q$$

**RL agent.** The agent is an actor-critic architecture. The actor is a policy network that maps the state (time remaining, normalized inventory, normalized price) to the parameters of a Gaussian distribution over bid/ask spreads, passed through a softplus so quotes are always positive; actions are sampled from this distribution during training to encourage exploration. The critic is a separate value network trained to predict expected return from a given state, used to compute the advantage that drives the actor's policy gradient update. Training simulates episodes under the same price dynamics and order arrival model as the FD solver, accumulates per-step inventory penalties and a terminal wealth-minus-penalty reward, and updates both networks via policy gradient (actor) and mean-squared-error regression to realized returns (critic).

**Single-bond and multi-bond settings.** The single-bond environment is used to validate the RL agent against the known-correct FD/analytical solution. The multi-bond environment extends this to five correlated bonds, with price paths driven by a Cholesky-decomposed correlation matrix so that price shocks are correlated across bonds, and a terminal reward that includes cross terms $\rho_{ij} q_i q_j$ penalizing correlated inventory positions in the same direction. This cross-term is what makes the true value function nonlinear in a way that a linear or purely quadratic function approximator cannot fully capture, and is the reason the choice of critic architecture matters so much in this setting. Full notation for every variable, parameter, and equation used across both settings is in `model_reference.md`.

## Results

- **Single-bond convergence:** the RL agent converges to a reward of approximately 57, against a theoretical (FD/analytical) optimum of approximately 66, or about 86 percent of optimum. This gap is expected: a sampled, function-approximated policy trained via policy gradient will not reach the exact analytical optimum, but 86 percent is close enough to demonstrate the agent has learned the right qualitative behavior (inventory-skewed quoting that tightens as risk aversion, urgency, and inventory level dictate).
- **Critic architecture ranking:** across the three critic designs tested in Section 3 (linear, quadratic-features, and neural network), the neural network critic reaches approximately 106 in its evaluation metric, the quadratic-feature critic reaches approximately 57, and the linear critic reaches approximately 44. This ordering is consistent with theory: the true value function has quadratic inventory terms, cross terms from correlation, time-dependent curvature, and boundary effects near the inventory cap, none of which a linear model can represent, only some of which a quadratic-feature model can represent, and all of which a neural network can approximate.

## What Was Fixed

A number of issues surfaced and were corrected during development, in the interest of making the FD-versus-RL comparison actually apples-to-apples:

- **Matched time steps.** The FD solver and the RL agent were originally running on different `dt` values (0.005 versus 0.01), which meant the two were not solving the same discretized problem. Both now use the same `dt`, so their outputs are directly comparable.
- **Undiscounted critic targets.** The critic was retrained on undiscounted real profits (rather than a discounted RL return) so that its value estimates are directly comparable, in the same units, to the FD solver's value function output.
- **Gradient clipping.** Added to stabilize actor and critic updates, which had previously shown occasional instability during training.
- **Reduced initial exploration noise.** The starting variance on the actor's sampled actions was lowered, which reduced early-training instability without preventing the agent from exploring.
- **Averaged sensitivity plots.** Sensitivity analyses (how results vary with a given parameter) now average 3 runs per point rather than reporting a single noisy run, giving more reliable curves.
- **Stable Section 3 results.** With the fixes above in place, Section 3 (the critic architecture comparison) now produces a stable and correct ranking (neural network > quadratic > linear) across repeated runs, rather than the noisy or inconsistent orderings seen before these fixes.

## Technical Stack

Python, TensorFlow, PyTorch, NumPy, and Matplotlib.

## Two Notebooks: TensorFlow and PyTorch

The project includes two Jupyter notebooks implementing the same experiments: `rl_market_making.ipynb` (TensorFlow) and `rl_market_making_PyTorch.ipynb` (PyTorch). Both notebooks are methodologically identical: same environment, same state representation, same actor-critic architecture, same training procedure, same evaluation metrics. The point of maintaining both is a direct framework comparison, confirming that the results are a property of the method and not an artifact of a particular deep learning library's defaults, numerical behavior, or optimizer implementation.

## Reference

See `model_reference.md` for a complete reference of every variable, parameter, and equation used in the notebooks, including the price dynamics, order arrival model, HJB equation, optimal quote formulas, the finite-difference algorithm, and the actor-critic state, action, and reward definitions.

`section4_draft.md` is a draft extension (not yet implemented in the notebooks) exploring competitive market making among multiple dealers using a softmax order-flow allocation model and multi-agent RL, framed around Nash equilibrium and game-theoretic concepts from CSCI 3202.
