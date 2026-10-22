"""The policy-gradient estimator must match the implemented sampling procedure.

Toy problem: a ~ N(mu, s), executed x = clip(a, lo, hi), reward f(x).
The true gradient of E[f(clip(a))] is computed by numerical integration and
compared with score-function estimates that use log N(raw a) (correct) or
log N(clip(a)) (the original notebooks).
"""

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from mm.policy import sample_clipped_gaussian, Actor, SharedActor, MultiActor, QuadraticCritic, count_params

LO, HI = 0.01, 5.0
f = lambda x: -(x - 0.3) ** 2


def expected_reward(mu, s):
    """E[f(clip(a))] by quadrature, with the clip's point masses at the bounds."""
    inner, _ = integrate.quad(lambda a: f(a) * stats.norm.pdf(a, mu, s), LO, HI)
    return inner + f(LO) * stats.norm.cdf(LO, mu, s) + f(HI) * stats.norm.sf(HI, mu, s)


def true_grad(mu, s, h=1e-5):
    return np.array([
        (expected_reward(mu + h, s) - expected_reward(mu - h, s)) / (2 * h),
        (expected_reward(mu, s * np.exp(h)) - expected_reward(mu, s * np.exp(-h))) / (2 * h),  # d/d log s
    ])


def score_estimate(mu, s, n, use_raw, seed=0):
    g = torch.Generator().manual_seed(seed)
    mean = torch.full((n, 1), mu, dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor(np.log(s), dtype=torch.float64, requires_grad=True)
    std = torch.exp(log_std).expand_as(mean)
    raw, executed, logp_raw = sample_clipped_gaussian(mean, std, LO, HI, g)
    logp = logp_raw if use_raw else torch.distributions.Normal(mean, std).log_prob(executed).sum(-1)
    R = f(executed).squeeze(-1)
    (logp * R).mean().backward()
    return np.array([mean.grad.sum().item(), log_std.grad.item()])


@pytest.mark.parametrize("mu, s", [(0.05, 0.3), (0.2, 0.5), (1.0, 0.37)])
def test_raw_sample_score_is_unbiased(mu, s):
    est = score_estimate(mu, s, 400_000, use_raw=True)
    assert est == pytest.approx(true_grad(mu, s), abs=0.01)


def test_clamped_log_prob_is_biased_when_clip_binds():
    mu, s = 0.05, 0.3   # about 45% of samples hit the lower clip
    truth = true_grad(mu, s)
    est = score_estimate(mu, s, 400_000, use_raw=False)
    # Not just noisy: the mean-gradient has the wrong sign (true ~ +0.04,
    # clamped estimator ~ -0.05; Monte Carlo standard error ~ 0.001).
    assert truth[0] > 0 > est[0]
    assert abs(est[0] - truth[0]) > 0.05


def test_no_clipping_means_both_estimators_agree():
    mu, s = 2.0, 0.2   # clip essentially never binds
    a = score_estimate(mu, s, 200_000, use_raw=True, seed=1)
    b = score_estimate(mu, s, 200_000, use_raw=False, seed=1)
    assert a == pytest.approx(b, abs=1e-6)


def test_executed_action_is_clip_of_raw_and_log_prob_is_of_raw():
    g = torch.Generator().manual_seed(3)
    mean, std = torch.tensor([[0.0, 4.9]]), torch.tensor([[1.0, 1.0]])
    raw, executed, logp = sample_clipped_gaussian(mean, std, LO, HI, g)
    assert torch.allclose(executed, raw.clamp(LO, HI))
    assert torch.allclose(logp, torch.distributions.Normal(mean, std).log_prob(raw).sum(-1))


def test_actor_shapes_and_deterministic_bounds():
    torch.manual_seed(0)
    a = Actor()
    x = torch.zeros(7, 2)
    raw, executed, logp = a.act(x)
    assert executed.shape == (7, 2) and logp.shape == (7,)
    det = a.deterministic(x)
    assert torch.all(det >= LO) and torch.all(det <= HI)


def test_shared_actor_size_is_independent_of_bond_count():
    a = SharedActor()
    assert a(torch.zeros(3, 4))[0].shape == (3, 2)
    assert a(torch.zeros(500, 4))[0].shape == (500, 2)
    assert count_params(MultiActor(1 + 500, 1000)) > count_params(MultiActor(1 + 5, 10))


def test_quadratic_critic_can_fit_a_quadratic_exactly():
    torch.manual_seed(0)
    x = torch.randn(512, 3)
    y = (-(x[:, 0] ** 2) - 0.7 * x[:, 0] * x[:, 1] + 0.5 * x[:, 2]).unsqueeze(1)
    c = QuadraticCritic(3)
    opt = torch.optim.Adam(c.parameters(), lr=0.05)
    for _ in range(1500):
        opt.zero_grad()
        loss = ((c(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-4
