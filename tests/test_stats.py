"""Statistics used by the report, checked on synthetic data with known answers."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import numpy as np

from utils.plot_results import (INF_BUDGET, bootstrap_saturation, budget_of, fit_saturation, holm,
                                mannwhitney_p, nonparametric_b90, spearman)


def test_holm():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])
    assert np.allclose(holm([0.5]), [0.5])


def test_spearman():
    r, p = spearman([1, 2, 3, 4, 5, 6], [2, 4, 5, 9, 10, 30])
    assert np.isclose(r, 1.0) and p == 0.0
    r, p = spearman([1, 2, 3, 4, 5, 6, 7, 8], [3, 1, 4, 1, 5, 9, 2, 6])
    assert -1 < r < 1 and 0 < p < 1


def test_mannwhitney_exact_small():
    assert np.isclose(mannwhitney_p([1, 2, 3], [4, 5, 6]), 0.1)   # 2 / C(6,3)


def test_saturation_fit_recovers_b90():
    rng = np.random.default_rng(0)
    y0, yinf, beta = 40.0, 80.0, 3.0
    groups = {}
    for b in (0, 1, 2, 4, 8, 16, INF_BUDGET):
        mean = yinf if np.isinf(b) else yinf - (yinf - y0) * math.exp(-b / beta)
        groups[b] = list(mean + rng.normal(0, 2, 10))
    budgets = [b for b, v in groups.items() for _ in v]
    scores = [s for v in groups.values() for s in v]
    fit = fit_saturation(budgets, scores)
    assert abs(fit["beta"] - beta) < 0.6 and abs(fit["y0"] - y0) < 2 and abs(fit["y_inf"] - yinf) < 2
    assert np.isclose(fit["b90"], fit["beta"] * math.log(10))
    assert nonparametric_b90(groups) == 8          # 90% of the gain first reached at B = 8 (true B90 ~ 6.9)
    bs = bootstrap_saturation(groups, n_boot=200)
    assert bs["b90_ci"][0] < beta * math.log(10) < bs["b90_ci"][1]


def test_no_gain_has_no_saturation_budget():
    groups = {0: [50, 51, 52], 4: [50, 49, 51], INF_BUDGET: [49, 50, 48]}
    fit = fit_saturation([b for b, v in groups.items() for _ in v], [s for v in groups.values() for s in v])
    assert fit["b90"] is None and nonparametric_b90(groups) is None


def test_budget_of_counts_usable_fixed_sends():
    h = dict(sender="fixed", max_steps=80, comm_every=40, comm_budget=None)
    assert budget_of(h) == 2
    assert budget_of(dict(h, comm_every=20)) == 4 and budget_of(dict(h, comm_every=1)) == INF_BUDGET
    assert budget_of(dict(sender="rule", comm_budget=6)) == 6 and budget_of(dict(sender="none")) == 0
