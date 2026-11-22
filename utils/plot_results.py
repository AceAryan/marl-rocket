"""
Loading, grouping and statistics for sweep results (experiments/run_all.py).

Writes <dir>/summary.csv with one row per configuration. Statistics are
plain numpy/math (no scipy): Student-t intervals, Welch and exact or
approximate Mann-Whitney U tests, Holm correction, Spearman trend tests, and
a saturating budget curve fitted with a seed bootstrap.

Usage:
    python utils/plot_results.py --dir checkpoints/sweep_v5
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import csv
import glob
import json
import math
import numpy as np


# ---- Student-t statistics in plain numpy/math (no scipy dependency) ----

def _betacf(a, b, x, iters=200, eps=1e-12):
    """Continued fraction for the regularized incomplete beta (Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((a + m2 - 1) * (a + m2)),
                   -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf2(t, df):
    """Two-sided p-value P(|T| > |t|) for Student t with df degrees of freedom."""
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def t_ppf975(df):
    """97.5% quantile of Student t (bisection on the two-sided tail)."""
    lo, hi = 0.0, 1000.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t_sf2(mid, df) > 0.05 else (lo, mid)
    return 0.5 * (lo + hi)


def welch_p(a, b):
    """Two-sided Welch t-test p-value (unequal variances)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return 1.0 if a.mean() == b.mean() else 0.0
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t_sf2(t, df))


def _ranks(x):
    """Ranks starting at 1, ties get their average rank."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def mannwhitney_p(a, b, exact_limit=200_000):
    """Two-sided Mann-Whitney U (rank-sum) p-value. Robust to the occasional
    collapsed seed, which can dominate a t-test on means. Exact (enumerating
    every split of the pooled ranks, ties included) when the number of splits
    is at most `exact_limit`, e.g. 10 vs 10 seeds; normal approximation with
    tie and continuity corrections otherwise."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1:
        return np.nan
    ranks = _ranks(np.concatenate([a, b]))
    r1 = ranks[:n1].sum()
    mean_r1 = n1 * (n1 + n2 + 1) / 2.0
    dev = abs(r1 - mean_r1)
    if math.comb(n1 + n2, n1) <= exact_limit:
        from itertools import combinations
        count = total = 0
        for idx in combinations(range(n1 + n2), n1):
            total += 1
            if abs(ranks[list(idx)].sum() - mean_r1) >= dev - 1e-9:
                count += 1
        return count / total
    n = n1 + n2
    _, tie_counts = np.unique(ranks, return_counts=True)
    var = n1 * n2 / 12.0 * ((n + 1) - np.sum(tie_counts ** 3 - tie_counts) / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = max(dev - 0.5, 0.0) / math.sqrt(var)
    return float(math.erfc(z / math.sqrt(2.0)))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaNs are left out and kept)."""
    p = np.asarray(pvals, np.float64)
    out = np.full(len(p), np.nan)
    idx = [i for i in range(len(p)) if not np.isnan(p[i])]
    m = len(idx)
    running = 0.0
    for k, i in enumerate(sorted(idx, key=lambda i: p[i])):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def spearman(x, y):
    """Spearman rank correlation and its two-sided p-value (t approximation)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0, 1.0
    r = float(np.corrcoef(rx, ry)[0, 1])
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return r, float(t_sf2(t, n - 2))


def _ci95(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    se = values.std(ddof=1) / np.sqrt(len(values))
    return float(t_ppf975(len(values) - 1) * se)


# ---- saturating budget curve ----

INF_BUDGET = float("inf")


def _fit_given_beta(B, y, beta):
    e = np.where(np.isinf(B), 0.0, np.exp(-np.where(np.isinf(B), 0.0, B) / beta))
    X = np.stack([e, 1.0 - e], axis=1)          # y = y0 * e + y_inf * (1 - e)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sse = float(np.sum((X @ coef - y) ** 2))
    return coef, sse


BETAS = np.exp(np.linspace(np.log(0.05), np.log(500.0), 400))


def fit_saturation(budgets, scores):
    """Least-squares fit of y(B) = y_inf - (y_inf - y0) exp(-B / beta) to
    per-seed points (budget 0 = no communication, inf = unlimited).
    Returns dict(y0, y_inf, beta, b90) where b90 = beta ln 10 is the budget
    that reaches 90% of the gain; b90 is None when there is no gain."""
    B = np.asarray(budgets, np.float64)
    y = np.asarray(scores, np.float64)
    best = None
    for beta in BETAS:
        coef, sse = _fit_given_beta(B, y, beta)
        if best is None or sse < best[0]:
            best = (sse, beta, coef)
    _, beta, (y0, y_inf) = best
    return {"y0": float(y0), "y_inf": float(y_inf), "beta": float(beta),
            "b90": float(beta * math.log(10)) if y_inf > y0 else None}


def nonparametric_b90(groups):
    """Smallest finite budget whose median reaches 90% of the gain from budget
    0 to unlimited. groups: {budget: [scores]} including 0 and inf."""
    med = {b: float(np.median(v)) for b, v in groups.items() if len(v)}
    if 0 not in med or INF_BUDGET not in med or med[INF_BUDGET] <= med[0]:
        return None
    goal = med[0] + 0.9 * (med[INF_BUDGET] - med[0])
    for b in sorted(k for k in med if np.isfinite(k) and k > 0):
        if med[b] >= goal:
            return b
    return INF_BUDGET


def bootstrap_saturation(groups, n_boot=1000, seed=0):
    """Point estimates and 95% percentile intervals (resampling seeds within
    each budget) of the fitted B90 and the nonparametric B90."""
    rng = np.random.default_rng(seed)

    def flat(g):
        bs, ys = [], []
        for b, v in g.items():
            bs += [b] * len(v)
            ys += list(v)
        return bs, ys

    fit = fit_saturation(*flat(groups))
    np90 = nonparametric_b90(groups)
    fits, nps = [], []
    for _ in range(n_boot):
        g = {b: rng.choice(np.asarray(v), size=len(v), replace=True) for b, v in groups.items() if len(v)}
        f = fit_saturation(*flat(g))
        fits.append(np.inf if f["b90"] is None else f["b90"])
        n = nonparametric_b90(g)
        nps.append(np.inf if n is None else n)

    def ci(xs):
        xs = np.sort(np.asarray(xs, np.float64))   # order statistics: inf (no saturation) stays inf
        return [float(xs[int(0.025 * (len(xs) - 1))]), float(xs[int(np.ceil(0.975 * (len(xs) - 1)))])]
    return {"fit": fit, "b90_ci": ci(fits), "np_b90": np90, "np_b90_ci": ci(nps)}


# ---- loading ----

def load_histories(base_dir: str):
    histories = []
    for path in sorted(glob.glob(os.path.join(base_dir, "**", "history.json"), recursive=True)):
        try:
            with open(path) as f:
                h = json.load(f)
        except (OSError, ValueError):
            print(f"skipping unreadable {path}")
            continue
        if "final_eval" in h and "label" in h:
            histories.append(h)
    return histories


def mode_of(h):
    return "match" if h.get("match") else "single"


def group_runs(histories):
    """{(mode, obs_frac, comm_delay, label): [history, ...]}"""
    groups = {}
    for h in histories:
        groups.setdefault((mode_of(h), h["obs_frac"], h["comm_delay"], h["label"]), []).append(h)
    return groups


def score(h, field="final_eval"):
    """Primary metric: goals per episode (single-goal mode: goal rate), in %."""
    return 100.0 * h[field]["goals_mean"]


def auc(h):
    """Learning speed: mean validation goals per episode over training (%)."""
    return 100.0 * float(np.mean([e["goals_mean"] for e in h["eval"]]))


def budget_of(h):
    """Messages per agent per episode allowed (0, k or inf), for budget curves."""
    if h["sender"] == "none":
        return 0.0
    if h["sender"] == "fixed":
        steps = h["max_steps"]
        k = h["comm_every"]
        return INF_BUDGET if k == 1 else float(len(range((k + 1) // 2, steps, k)))
    return INF_BUDGET if h["comm_budget"] is None else float(h["comm_budget"])


SUMMARY_FIELDS = ["mode", "obs_frac", "delay", "label", "sender", "budget", "seeds", "goals_median",
                  "goals_q25", "goals_q75", "goals_mean", "goals_ci95", "last_median", "collapsed_last",
                  "goal_rate", "time_to_goal", "msgs_per_agent", "auc", "shuffled_median", "role_sep"]


def summarize(base_dir: str):
    groups = group_runs(load_histories(base_dir))
    rows = []
    for key in sorted(groups, key=lambda k: (k[0], -k[1], k[2], k[3])):
        runs = groups[key]
        g = np.array([score(h) for h in runs])
        last = np.array([score(h, "final_eval_last") for h in runs])
        sh = [score(h, "final_eval_shuffled") for h in runs if "final_eval_shuffled" in h]
        ttg = [h["final_eval"]["time_to_goal"] for h in runs if h["final_eval"]["time_to_goal"] is not None]
        q25, med, q75 = np.percentile(g, [25, 50, 75])
        rows.append({
            "mode": key[0], "obs_frac": key[1], "delay": key[2], "label": key[3], "sender": runs[0]["sender"],
            "budget": budget_of(runs[0]), "seeds": len(runs), "goals_median": med, "goals_q25": q25,
            "goals_q75": q75, "goals_mean": g.mean(), "goals_ci95": _ci95(g), "last_median": np.median(last),
            "collapsed_last": int(np.sum(last < 0.5 * g)),
            "goal_rate": 100 * np.mean([h["final_eval"]["goal_rate"] for h in runs]),
            "time_to_goal": np.mean(ttg) if ttg else np.nan,
            "msgs_per_agent": np.mean([h["final_eval"]["msgs_mean"] for h in runs]) / 2,
            "auc": np.mean([auc(h) for h in runs]),
            "shuffled_median": np.median(sh) if sh else np.nan,
            "role_sep": np.mean([h["final_eval"]["role_separation"] for h in runs]),
        })
    if rows:
        out = os.path.join(base_dir, "summary.csv")
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints/sweep_v5")
    args = parser.parse_args()
    for r in summarize(args.dir):
        print(f"{r['mode']:>6} {r['obs_frac']:>6} d{r['delay']} {r['label']:>10} n={r['seeds']:2d} "
              f"goals {r['goals_median']:6.1f} [{r['goals_q25']:5.1f}-{r['goals_q75']:5.1f}] "
              f"last {r['last_median']:6.1f} msgs {r['msgs_per_agent']:5.1f} auc {r['auc']:5.1f}")
