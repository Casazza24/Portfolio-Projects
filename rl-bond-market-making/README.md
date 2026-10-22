# RL Bond Market Making

**[Live Demo](https://casazza24.github.io/Portfolio-Projects/rl-bond-market-making/demo.html)** (built from the original, pre-correction version; see [Project history](#project-history))

A reinforcement learning agent that learns how to be a bond market maker: setting buy and sell prices to earn the spread while managing the risk of holding too much inventory. Built as a final project for CSCI 3202 (Artificial Intelligence) at CU Boulder, then revisited with a correctness and validation pass.

## What Is Market Making?

A market maker always offers to buy and sell a security. They post a "bid" price (what they'll pay to buy) and an "ask" price (what they'll charge to sell). The gap between the two, the spread, is how they make money. The catch is that if they end up buying much more than they sell (or vice versa), they hold inventory that can lose value. The core challenge is how wide to set the prices, and how to skew them based on current inventory.

## The Research Question

For one bond the optimal quotes can be computed exactly by dynamic programming over (time, inventory). For many correlated bonds the exact grid grows as $(2q_{\max}+1)^n$: 11 inventory states for one bond, about 26 billion for ten, more than the atoms in the universe for 500. Following Guéant & Manziuk (2019), the project asks whether actor-critic RL can learn good quotes where exact methods cannot, and whether a neural-network value function is needed to do it.

## What the Validated Experiments Show

All numbers below come from frozen (deterministic) policies trained with several seeds, scored **exactly** by dynamic-programming policy evaluation where the grid allows and otherwise by Monte Carlo on independent evaluation episodes. Intervals are 95% t-intervals across training seeds. Raw results are in `results/`.

**1. Single bond: RL learns the right behavior but does not reach the optimum at the original budget.**

| Policy | Expected objective | Gap to optimum |
|---|---|---|
| Exact optimum (DP) | 66.22 | 0 |
| RL, 2,000 episodes (original budget), 5 seeds | 62.81 ± 0.78 | 5.1% ± 1.2% |
| RL, 8,000 episodes, 5 seeds | 64.94 ± 0.95 | 1.9% ± 1.4% |
| Constant quotes that ignore inventory | 61.77 | 6.7% |

The learned policy skews its quotes with inventory in the right direction, but at the original budget it is only slightly better than ignoring inventory. Four times more training closes most of the gap, with large variation between seeds.

**2. Many bonds: correlation-aware quoting is worth little here, and a simple baseline beats RL.**

- The *separable* policy (each bond quoted by the exact single-bond optimum) is within 0.1% of the exact coupled optimum for 2 bonds and within 0.15% for 3 bonds, even at correlation 0.9. Its cost does not grow with the number of identical bonds.
- The parameter-shared RL actor is about 6% below the coupled optimum for 2 and 3 bonds at the original 500-episode budget (5 seeds, every correlation tested), and 1.5% to 5% below at 2,000 episodes with wide intervals.
- At 50 and 500 bonds (3 seeds, paired Monte Carlo on common random numbers), the RL policy scores 0.6 to 0.9 per bond (about 1.0% to 1.4%) **below** the separable DP policy in every run, with intervals that exclude zero.

So the curse of dimensionality is real for the *exact coupled* solver, but in this model it does not stop a near-optimal solution: the separable policy is cheap and close to optimal, and RL does not beat it.

**3. Critics: no reliable advantage for the neural network on policy quality (3 seeds, weak precision).**

The original notebook reported a neural-network critic beating the best linear critic by about 60% on average *training* reward from single runs. Re-run with 3 seeds:

- On the original metric (training reward of the exploring policy), the NN critic still leads, by about 12% at 3 bonds.
- On the frozen policies' exact objective, linear, quadratic and NN critics are **not distinguishable**: seed-to-seed spread (for example 127 to 178 for the linear critic) is larger than any difference between critic types.
- The NN critic fits returns much better (about 25 times lower critic loss), but that does not translate into better policies here.

Three seeds give wide intervals, so these results rule out a large, reliable advantage rather than establish that the critics are equal.

**4. Scaling (single-threaded, median of 3 repeats; `results/scaling.json`).**

| Method | 1 bond | 4 bonds | 100 bonds | 500 bonds |
|---|---|---|---|---|
| Exact coupled DP ($q_{\max}=3$) | 0.05 s | 2.5 s (2,401 states, 37 MB) | infeasible | infeasible |
| Separable DP, identical bonds | 0.04 s | 0.04 s | 0.04 s | 0.04 s |
| Separable DP, heterogeneous bonds | 0.04 s | | 4.1 s | (linear) |
| RL, **one** training episode | 0.045 s | | 0.065 s | 0.13 s |

The exact coupled solver's cost per added bond keeps rising (2.5x, 3.0x, 6.3x from 1 to 4 bonds), as the $(2q_{\max}+1)^n$ grid predicts. The shared actor's parameter count is constant (17,412), but that is not the algorithm's total cost: the critic grows with the number of bonds (17,025 to 80,897 parameters at 500), each step evaluates the actor on every bond, and the environment's covariance products are $O(n^2)$. Training also needs hundreds to thousands of episodes (about 6 minutes for 1,000 episodes at 500 bonds in part C), and still scores below the separable policy, which takes 0.04 s.

## What Changed in the Correctness Pass

The original comparisons were not apples to apples. The main fixes (details in `model_reference.md` and `historical/README.md`):

- **One objective.** RL environment and benchmark now maximize the same explicitly defined objective, $\mathbb{E}[X_N + q_N S_N - \sum \tfrac{\gamma}{2} q^\top \Sigma q\, \Delta t - \ell |q_N|^2]$. Previously the RL environment had a running and a terminal inventory penalty while the "ground truth" had only the terminal one, and volatility did not enter the benchmark at all.
- **Exact, independently validated benchmark.** The dynamic program is exact for the discrete simulator (including both sides filling in one step) and is checked against Monte Carlo, a brute-force quote search, and the closed-form continuous-time solution.
- **Correct policy gradients.** Log-probabilities are evaluated at the raw sampled action; the original clamped version could point the gradient the wrong way.
- **Genuine multi-bond benchmarks.** A separable solver and an exact covariance-coupled solver replace the old multi-bond FD, which ignored correlation while enumerating the full grid.
- **Honest evaluation.** Frozen policies, independent evaluation episodes, multiple seeds, confidence intervals, exact scoring where possible.

## Repository Layout

| Path | Contents |
|---|---|
| `rl_market_making_PyTorch.ipynb` | **Current notebook.** Interactive walkthrough; imports from `mm/` and loads saved results (no retraining). |
| `mm/` | Package: objective (`model.py`), simulators (`env.py`), exact DP benchmarks (`dp.py`), networks (`policy.py`), training (`train.py`), evaluation (`evaluate.py`), historical baselines (`legacy.py`) |
| `tests/` | 76 tests: transitions, rewards, DP validation, multi-bond benchmarks, policy-gradient estimator, training reproducibility |
| `experiments/` | Scripts that produce `results/` (resumable, single-threaded by default) |
| `results/` | Saved experiment results and the model weights the notebook loads |
| `model_reference.md` | Notation, objective, derivations; historical formulation and what was wrong with it |
| `historical/` | Original notebook with its outputs, and notes on why its results are superseded |
| `rl_market_making.ipynb` | Earlier TensorFlow iteration, kept for history (only the log-probability fix applied) |

## Reproducing

```bash
conda activate myenv            # Python 3.10, torch 2.5, numpy 1.23, scipy 1.13
cd rl-bond-market-making
python -m pytest -q tests       # ~15 s
python experiments/run_single_bond.py              # single bond, 5 seeds x {2000, 8000} episodes
python experiments/run_multi_bond.py --part A --seeds 5
python experiments/run_multi_bond.py --part C      # add --large-sizes 50 to skip 500 bonds
python experiments/run_multi_bond.py --part B --seeds 3
python experiments/train_demo_models.py            # weights the notebook loads
python experiments/run_scaling.py --repeats 3
```

The multi-bond runner limits numerical libraries to one thread, runs jobs sequentially, and resumes from saved per-configuration results. Training is deterministic given the seed.

## Limitations

- Poisson fills with exponential intensity; no adverse selection, market impact, or limit order book.
- Identical bonds with equicorrelation. Heterogeneous bonds would make the separable benchmark cost grow linearly and could change how close it is to optimal.
- Vanilla REINFORCE with a learned baseline and one episode per update; more sample-efficient methods (PPO, SAC) were not tried, so the results describe this algorithm, not RL in general.
- Few seeds (3 to 5); small differences are not resolvable.
- Exact evaluation of RL policies is only possible up to three bonds; larger portfolios are compared with the separable policy, not with the true optimum.

## Project History

The original version (July 2026, commit `abae89b`) reported that RL matched the exact solution for one bond, scaled where FD could not, and that a neural-network critic beat linear critics by about 60%. Those results came from mismatched objectives and single training runs; they are preserved in `historical/` and in the live demo, but are superseded by the results above.

## References

- Guéant & Manziuk (2019), "Deep reinforcement learning for market making in corporate bonds: beating the curse of dimensionality"
- Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
- Cartea, Jaimungal & Penalva (2015), *Algorithmic and High-Frequency Trading*, ch. 10
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*

---

*Note: Claude (Anthropic) was used to help with file organization, writing documentation, code review, and the correctness pass (objective definition, benchmarks, tests and experiment scripts).*
