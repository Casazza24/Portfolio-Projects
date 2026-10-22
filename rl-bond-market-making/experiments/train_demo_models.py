"""Train and save the single-bond demonstration agent used by the notebook.

Training is deterministic given the seed, so this reproduces run
(seed 0, 2000 episodes) of experiments/run_single_bond.py exactly; the script
checks that the exact objective matches results/single_bond.json. Skips if
the model file already exists. Single-threaded.

Usage:  python experiments/train_demo_models.py
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mm.model import MarketParams  # noqa: E402
from mm.train import train_single  # noqa: E402
from mm.dp import evaluate_single_bond_policy  # noqa: E402
from mm.evaluate import single_actor_policy  # noqa: E402

P = MarketParams()
SEED, EPISODES = 0, 2000
OUT = os.path.join(ROOT, "results", "models", f"single_bond_s{SEED}_ep{EPISODES}.pt")


def main():
    torch.set_num_threads(1)
    if os.path.exists(OUT):
        print("exists:", OUT)
        return
    actor, critic, history = train_single(P, EPISODES, seed=SEED)
    J = float(evaluate_single_bond_policy(P, single_actor_policy(actor, P))[0, P.q_max])
    saved = [r for r in json.load(open(os.path.join(ROOT, "results", "single_bond.json")))["rl_runs"]
             if r["seed"] == SEED and r["episodes"] == EPISODES][0]["exact_objective"]
    print(f"exact objective {J:.6f}; results/single_bond.json has {saved:.6f}")
    if abs(J - saved) > 1e-6:
        raise SystemExit("mismatch: training is not reproducing the recorded run")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "history": history,
                "seed": SEED, "episodes": EPISODES}, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
