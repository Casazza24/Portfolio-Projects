# RL Bond Market Making

**[Live Demo](https://casazza24.github.io/Portfolio-Projects/rl-bond-market-making/demo.html)**

A reinforcement learning agent that learns how to be a bond market maker — setting buy and sell prices to earn the spread while managing the risk of holding too much inventory. Built as a final project for CSCI 3202 (Artificial Intelligence) at CU Boulder.

## Demo Video

[Watch the demo video](demo_video.webm)

## What Is Market Making?

A market maker is someone who always offers to buy and sell a security. They post a "bid" price (what they'll pay to buy) and an "ask" price (what they'll charge to sell). The gap between the two — the spread — is how they make money. The catch is that if they end up buying way more than they sell (or vice versa), they're stuck holding a bunch of inventory that could lose value. So the core challenge is: how wide or narrow should you set your prices, and how should you adjust them based on how much inventory you're currently holding?

## What This Project Does

There's a well-known math solution to this problem for a single bond (the Avellaneda-Stoikov model). You can solve it exactly using a technique called finite differences, which basically works backward through time on a grid of possible states to find the best action at each point.

The problem is that this grid approach falls apart fast. For one bond with an inventory cap of 5, you only have 11 possible inventory states. For five bonds, that jumps to 161,051 states. For ten bonds, you'd need roughly 25 billion states — totally impossible to compute.

This project trains an RL agent (specifically, an actor-critic with neural networks) to learn the same pricing strategy. The key findings:

1. **For a single bond, the RL agent matches the known-correct solution** — it learns to widen its buy price when it's already holding a lot, and tighten its sell price to offload inventory, just like the math says it should.

2. **The RL agent scales to five bonds where the exact solution can't go.** Training time for the RL agent grows roughly linearly with the number of bonds, while the grid method's time grows exponentially.

3. **Neural networks outperform simpler models when bonds are correlated.** When bond prices move together, the interactions between inventory positions create a value landscape that a simple linear model can't capture. A neural network handles this naturally, roughly doubling the performance of a linear model.

## How It Works

- **Section 1** validates the RL agent against the known solution for one bond — making sure the learning actually works before scaling up.
- **Section 2** shows what happens when you try to scale the grid method to multiple bonds (it breaks) and that RL handles it fine.
- **Section 3** compares three different ways of estimating how good a given state is (linear, quadratic features, and neural network) and shows why you need a neural network when bonds are correlated.

## Tech Stack

Python, TensorFlow, PyTorch, NumPy, Matplotlib

## Notebooks

There are two notebooks that run the same experiments:

- `rl_market_making.ipynb` — TensorFlow version
- `rl_market_making_PyTorch.ipynb` — PyTorch version

Both produce the same results, confirming the findings aren't just an artifact of one framework's defaults.

## References

- Gueant & Manziuk (2019), "Deep reinforcement learning for market making in corporate bonds: beating the curse of dimensionality" — the main paper this project builds on
- Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book" — the original market making model
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*

## Additional Files

- `model_reference.md` — full notation and equations used in the notebooks
- `section4_draft.md` — a draft extension exploring competitive market making with multiple dealers (not yet implemented)

---

*Note: Claude (Anthropic) was used to help with file organization, writing this README, and pushing code to GitHub. The project itself — all modeling, implementation, and analysis — is original work.*
