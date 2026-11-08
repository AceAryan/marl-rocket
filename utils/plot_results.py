"""
Summarise and plot sweep results saved by experiments/run_all.py.

Usage:
    python utils/plot_results.py --dir checkpoints/sweep --save
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import csv
import glob
import json
import numpy as np

import math


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

# x-axis order of the saturation curve: (setting, budget label)
BUDGET_ORDER = [("A", "0"), ("D5", "5"), ("D10", "10"), ("D", "20"), ("D50", "50"), ("B", "inf")]
SETTING_ORDER = ["A", "D5", "D10", "D", "D50", "C", "B", "F5", "F10", "F20", "F50", "Finf"]
# Fixed-schedule senders on the same budget axis (A = 0 messages)
FIXED_ORDER = [("A", "0"), ("F5", "5"), ("F10", "10"), ("F20", "20"), ("F50", "50"), ("Finf", "inf")]


def load_histories(base_dir: str):
    histories = []
    for path in sorted(glob.glob(os.path.join(base_dir, "**", "history.json"), recursive=True)):
        with open(path) as f:
            h = json.load(f)
        if "final_eval" in h:
            histories.append(h)
    return histories


def group_runs(histories):
    groups = {}
    for h in histories:
        groups.setdefault((h["obs_frac"], h["comm_delay"], h["setting"]), []).append(h)
    return groups


def _ci95(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    se = values.std(ddof=1) / np.sqrt(len(values))
    t = t_ppf975(len(values) - 1)
    return float(t * se)


def summarize(base_dir: str):
    groups = group_runs(load_histories(base_dir))
    if not groups:
        print(f"No finished runs found in {base_dir}")
        return []

    def final(runs, key, field="final_eval"):
        return [h[field].get(key, np.nan) for h in runs if field in h]

    rows = []
    for (obs, delay, setting), runs in sorted(
        groups.items(), key=lambda kv: (-kv[0][0], kv[0][1], SETTING_ORDER.index(kv[0][2]))
    ):
        returns = final(runs, "return_mean")
        # Area under the learning curve: mean periodic greedy return (sample efficiency)
        aucs = [np.mean([e["return_mean"] for e in h["eval"]]) for h in runs]
        row = {
            "obs_frac": obs, "delay": delay, "setting": setting, "seeds": len(runs),
            "return": np.mean(returns), "return_ci95": _ci95(returns),
            "goal_rate": np.mean(final(runs, "goal_rate")),
            "concede_rate": np.mean(final(runs, "concede_rate")),
            "goal_diff": np.mean(final(runs, "goal_diff")),
            "msgs_per_ep": np.mean(final(runs, "msgs_mean")),
            "role_sep": np.mean(final(runs, "role_separation")),
            "ep_len": np.mean(final(runs, "episode_length")),
            "auc": np.mean(aucs), "auc_ci95": _ci95(aucs),
            "silenced_drop": np.nan, "vs_A_diff": np.nan, "vs_A_p": np.nan,
            "auc_vs_A_p": np.nan,
        }
        silenced = final(runs, "return_mean", "final_eval_silenced")
        if silenced:
            row["silenced_drop"] = np.mean(returns) - np.mean(silenced)
        baseline = groups.get((obs, 0, "A"))
        if baseline and setting != "A":
            base = final(baseline, "return_mean")
            row["vs_A_diff"] = np.mean(returns) - np.mean(base)
            base_auc = [np.mean([e["return_mean"] for e in h["eval"]]) for h in baseline]
            row["vs_A_p"] = welch_p(returns, base)
            row["auc_vs_A_p"] = welch_p(aucs, base_auc)
        rows.append(row)

    header = (f"{'obs':>5} {'delay':>5} {'set':>4} {'n':>2} | {'return':>15} | {'goal':>5} "
              f"{'conc':>5} {'GD':>6} {'epLen':>6} | {'msgs':>6} {'roleSep':>7} | {'silDrop':>7} | "
              f"{'vsA':>7} {'p':>6} | {'AUC':>14} {'p':>6}")
    print("\n=== FINAL GREEDY EVALUATION (mean over seeds, 95% CI) ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['obs_frac']:>5} {r['delay']:>5} {r['setting']:>4} {r['seeds']:>2} | "
            f"{r['return']:6.1f} +/- {r['return_ci95']:5.1f} | {r['goal_rate']:5.2f} "
            f"{r['concede_rate']:5.2f} {r['goal_diff']:6.2f} {r['ep_len']:6.1f} | {r['msgs_per_ep']:6.1f} "
            f"{r['role_sep']:7.3f} | {r['silenced_drop']:7.1f} | "
            f"{r['vs_A_diff']:7.1f} {r['vs_A_p']:6.3f} | "
            f"{r['auc']:5.1f} +/- {r['auc_ci95']:5.1f} {r['auc_vs_A_p']:6.3f}"
        )
    out = os.path.join(base_dir, "summary.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out}")
    return rows


def plot(base_dir: str, save: bool = False):
    try:
        import matplotlib
        if save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return
    groups = group_runs(load_histories(base_dir))
    if not groups:
        return

    def stat(runs):
        v = [h["final_eval"]["return_mean"] for h in runs]
        return np.mean(v), _ci95(v)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    # (1) Saturation curve: final return vs per-agent budget
    ax = axes[0]
    for obs in sorted({k[0] for k in groups}, reverse=True):
        xs, ms, cs = [], [], []
        for x, (setting, _) in enumerate(BUDGET_ORDER):
            if (obs, 0, setting) in groups:
                m, c = stat(groups[(obs, 0, setting)])
                xs.append(x); ms.append(m); cs.append(c)
        if len(xs) > 1:
            line = ax.errorbar(xs, ms, yerr=cs, marker="o", capsize=3, label=f"learned, rho={obs}")
        fx, fm, fc = [], [], []
        for x, (setting, _) in enumerate(FIXED_ORDER):
            if (obs, 0, setting) in groups:
                m, c = stat(groups[(obs, 0, setting)])
                fx.append(x); fm.append(m); fc.append(c)
        if len(fx) > 2:
            ax.errorbar(fx, fm, yerr=fc, marker="s", ls="--", capsize=3,
                        color=line[0].get_color() if len(xs) > 1 else None,
                        label=f"fixed schedule, rho={obs}")
    ax.set_xticks(range(len(BUDGET_ORDER)), [b for _, b in BUDGET_ORDER])
    ax.set_xlabel("Message budget B per agent per episode")
    ax.set_ylabel("Final greedy return (95% CI)")
    ax.set_title("RQ1: budget saturation")

    # (2) Learning curves (periodic greedy eval), partial obs if present
    ax = axes[1]
    rhos_avail = {k[0] for k in groups}
    obs_for_curves = 0.15 if 0.15 in rhos_avail else min(rhos_avail)
    for setting in SETTING_ORDER:
        runs = groups.get((obs_for_curves, 0, setting))
        if not runs:
            continue
        length = min(len(h["eval"]) for h in runs)
        values = np.array([[e["return_mean"] for e in h["eval"][:length]] for h in runs])
        xs = runs[0]["eval_episode"][:length]
        mean = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else 0 * mean
        ax.plot(xs, mean, label=setting)
        ax.fill_between(xs, mean - se, mean + se, alpha=0.15)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy eval return (mean +/- SE)")
    ax.set_title(f"Learning curves (rho={obs_for_curves})")

    # (3) RQ2: delay on B (partial obs)
    ax = axes[2]
    delayed = sorted({(k[0], k[1]) for k in groups if k[2] == "B" and k[1] > 0})
    if delayed:
        rho = delayed[0][0]
        delays = sorted({d for r, d in delayed if r == rho} | {0})
        delays = [d for d in delays if (rho, d, "B") in groups]
        ms, cs = zip(*[stat(groups[(rho, d, "B")]) for d in delays])
        ax.errorbar(delays, ms, yerr=cs, marker="o", capsize=3, label=f"B (learned), rho={rho}")
        fdel = [d for d in delays if (rho, d, "Finf") in groups]
        if fdel:
            fm, fc = zip(*[stat(groups[(rho, d, "Finf")]) for d in fdel])
            ax.errorbar(fdel, fm, yerr=fc, marker="s", ls="--", capsize=3, label=f"Finf (fixed), rho={rho}")
        if (rho, 0, "A") in groups:
            ax.axhline(stat(groups[(rho, 0, "A")])[0], ls="--", color="gray", label="A (no comm)")
        ax.set_xlabel("Message delay (steps)")
    ax.set_ylabel("Final greedy return (95% CI)")
    ax.set_title("RQ2: communication delay")

    # (4) RQ3: A vs B across observability
    ax = axes[3]
    rhos = sorted(k[0] for k in groups if k[1] == 0 and k[2] == "A" and (k[0], 0, "B") in groups)
    if rhos:
        for setting in ("A", "B"):
            ms, cs = zip(*[stat(groups[(r, 0, setting)]) for r in rhos])
            ax.errorbar(rhos, ms, yerr=cs, marker="o", capsize=3, label=setting)
        frhos = [r for r in rhos if (r, 0, "Finf") in groups]
        if frhos:
            fm, fc = zip(*[stat(groups[(r, 0, "Finf")]) for r in frhos])
            ax.errorbar(frhos, fm, yerr=fc, marker="s", ls="--", capsize=3, label="Finf (fixed)")
    ax.set_xlabel("Observation radius rho (fraction of field diagonal)")
    ax.set_ylabel("Final greedy return (95% CI)")
    ax.set_title("RQ3: A vs B across observability")

    for ax in axes:
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        out = os.path.join(base_dir, "results.png")
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints/sweep_final")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    summarize(args.dir)
    plot(args.dir, args.save)
