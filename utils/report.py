"""
Report-ready figures and tables for a finished (or partial) sweep.

Writes <dir>/results.md, <dir>/summary.csv and <dir>/figures/*.png:
    rq1_budget.png         goals vs message budget, learned / rule / fixed senders,
                           fitted saturation curves and B90 (budget reaching 90% of the gain)
    budget_awareness.png   learned sender with and without the budget features (feedback 2)
    rq2_delay.png          goals vs message delay
    rq3_observability.png  goals and communication gain vs observation radius
    learning_curves.png    validation goals during training
    send_timing.png        when in the episode messages are sent

Primary metric: goals per episode on 200 held-out episodes (single-goal mode:
the share of episodes with an assisted goal), for the policy snapshot with
the best validation score; the last policy is reported next to it. Per
configuration: median over seeds with its interquartile range, and the mean
with a 95% t-interval. Tests are two-sided Mann-Whitney U on per-seed scores,
Holm-corrected within each table; trends use Spearman's rank correlation.

Usage:
    python -m utils.report --dir checkpoints/sweep_v5
    python -m utils.report --dir checkpoints/pilot_v6 --pilot
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import math
import re
import numpy as np

from experiments import config as C
from utils.plot_results import (INF_BUDGET, _ci95, auc, bootstrap_saturation, budget_of, group_runs, holm,
                                load_histories, mannwhitney_p, score, spearman, summarize)

GRAY, BLUE, AQUA, ORANGE, PURPLE = "#52514e", "#2a78d6", "#1baf7a", "#eb6834", "#8e5bd0"
KIND = {"learned": ("L", BLUE, "o", "Learned gate"), "rule": ("R", AQUA, "^", "Rule: send on sight"),
        "fixed": ("F", ORANGE, "s", "Fixed schedule")}
MIN_SEEDS_FOR_STARS = 5


# ------------------------------------------------------------------ helpers

def _scores(runs, field="final_eval"):
    return [score(h, field) for h in runs if field in h]


def _med_iqr(values):
    q25, med, q75 = np.percentile(values, [25, 50, 75])
    return float(med), float(q25), float(q75)


def _fmt(v, f=".1f", dash="-"):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return dash
    if isinstance(v, float) and np.isinf(v):
        return "inf"
    return format(v, f)


def _star(p, n_min):
    if p is None or np.isnan(p) or n_min < MIN_SEEDS_FOR_STARS:
        return ""
    return " **" if p < 0.01 else " *" if p < 0.05 else ""


def _budget_label(b):
    return "inf" if np.isinf(b) else str(int(b))


def sender_groups(groups, mode, rho, kind, delay=0):
    """{budget: runs} for one sender kind at (mode, rho, delay); budget 0 = A."""
    prefix = KIND[kind][0]
    pat = re.compile(rf"^{prefix}(\d+|inf)$")
    out = {}
    for (m, r, d, label), runs in groups.items():
        if m != mode or r != rho or d != delay:
            continue
        if label == "A":
            out[0.0] = runs
        elif pat.match(label):
            out[budget_of(runs[0])] = runs
    return out


def _main_key(groups):
    """(mode, rho) of the budget sweep: from config, else the most populated."""
    if any(k[0] == C.MAIN_MODE and k[1] == C.MAIN_RHO for k in groups):
        return C.MAIN_MODE, C.MAIN_RHO
    counts = {}
    for (m, r, d, label) in groups:
        counts[(m, r)] = counts.get((m, r), 0) + 1
    return max(counts, key=counts.get) if counts else (C.MAIN_MODE, C.MAIN_RHO)


# ------------------------------------------------------------------ tables

ROW_HEADER = ("| Configuration | Seeds | Goals % median [IQR] | Mean ± 95% CI | Δ vs A | p vs A | Δ vs unlimited "
              "| p vs unlimited | Last policy median | Collapsed (last) | Msgs/agent | Time to goal | AUC "
              "| Shuffled-content drop |\n|" + "---|" * 14)


def _row(label, runs, base=None, unlimited=None, p_a=np.nan, p_u=np.nan):
    g = np.asarray(_scores(runs))
    med, q25, q75 = _med_iqr(g)
    last = np.asarray(_scores(runs, "final_eval_last"))
    d_a = med - np.median(_scores(base)) if base else np.nan
    d_u = med - np.median(_scores(unlimited)) if unlimited else np.nan
    n_min_a = min(len(g), len(base)) if base else 0
    n_min_u = min(len(g), len(unlimited)) if unlimited else 0
    ttg = [h["final_eval"]["time_to_goal"] for h in runs if h["final_eval"]["time_to_goal"] is not None]
    msgs = np.mean([h["final_eval"]["msgs_mean"] for h in runs]) / 2
    sh = _scores(runs, "final_eval_shuffled")
    drop = med - np.median(sh) if sh else np.nan
    return (f"| {label} | {len(g)} | {med:.1f} [{q25:.0f}-{q75:.0f}] | {g.mean():.1f} ± {_ci95(g):.1f} | "
            f"{_fmt(d_a, '+.1f')}{_star(p_a, n_min_a)} | {_fmt(p_a, '.3f')} | {_fmt(d_u, '+.1f')}{_star(p_u, n_min_u)} | "
            f"{_fmt(p_u, '.3f')} | {np.median(last):.1f} | {int(np.sum(last < 0.5 * g))} | {msgs:.1f} | "
            f"{_fmt(float(np.mean(ttg)) if ttg else None, '.0f')} | {np.mean([auc(h) for h in runs]):.1f} | "
            f"{_fmt(drop, '+.1f')} |")


def rq1_tables(groups, mode, rho):
    lines, sat = [], {}
    for kind in ("learned", "rule", "fixed"):
        sg = sender_groups(groups, mode, rho, kind)
        finite = sorted(b for b in sg if b > 0)
        if not finite:
            continue
        base, unl = sg.get(0.0), sg.get(INF_BUDGET)
        keys = [b for b in finite if b in sg]
        pa = [mannwhitney_p(_scores(sg[b]), _scores(base)) if base else np.nan for b in keys]
        pu = [mannwhitney_p(_scores(sg[b]), _scores(unl)) if unl and not np.isinf(b) else np.nan for b in keys]
        adj = holm(pa + pu)
        pa, pu = adj[:len(keys)], adj[len(keys):]
        prefix = KIND[kind][0]
        rows = []
        if base:
            rows.append(_row("A: no communication", base))
        for b, p1, p2 in zip(keys, pa, pu):
            rows.append(_row(f"{prefix}{_budget_label(b)}", sg[b], base, None if np.isinf(b) else unl, p1, p2))
        lines += [f"### {KIND[kind][3]} (budget = messages per agent per episode)", "", ROW_HEADER] + rows + [""]
        pts = [(math.log2(1 + b), s) for b in sg if not np.isinf(b) for s in _scores(sg[b])]
        if len(pts) >= 3:
            r, p = spearman(*zip(*pts))
            lines.append(f"Trend over finite budgets (incl. 0): Spearman rho = {r:+.2f}, p = {_fmt(p, '.3g')} "
                         f"(n = {len(pts)} runs).")
        if base and unl:
            bs = bootstrap_saturation({b: _scores(v) for b, v in sg.items()})
            sat[kind] = bs
            fit = bs["fit"]
            if fit["b90"] is None:
                lines.append("Saturation: no gain from A to unlimited, so no saturation budget.")
            else:
                lines.append(
                    f"Saturation fit y(B) = y_inf - (y_inf - y0)·exp(-B/beta): y0 = {fit['y0']:.1f}, "
                    f"y_inf = {fit['y_inf']:.1f}, **B90 = {fit['b90']:.1f}** messages "
                    f"(95% seed-bootstrap CI {_fmt(bs['b90_ci'][0])}-{_fmt(bs['b90_ci'][1])}); "
                    f"smallest tested budget whose median reaches 90% of the gain: "
                    f"**{_fmt(bs['np_b90'], '.0f')}** (CI {_fmt(bs['np_b90_ci'][0], '.0f')}-"
                    f"{_fmt(bs['np_b90_ci'][1], '.0f')}).")
        lines.append("")
    return lines, sat


def awareness_table(groups, mode, rho):
    pairs = []
    for (m, r, d, label), runs in groups.items():
        if m == mode and r == rho and d == 0 and label.endswith("-blind"):
            aware = groups.get((m, r, d, label[:-len("-blind")]))
            if aware:
                pairs.append((budget_of(runs[0]), label[:-len("-blind")], aware, runs))
    if not pairs:
        return []
    pairs.sort()
    ps = holm([mannwhitney_p(_scores(a), _scores(b)) for _, _, a, b in pairs])
    lines = ["| Budget | Aware median [IQR] | Blind median [IQR] | Δ (aware - blind) | p (Holm) | "
             "Msgs/agent aware / blind | Last-quarter sends aware / blind |", "|---|---|---|---|---|---|---|"]
    for (b, label, aware, blind), p in zip(pairs, ps):
        ma, qa1, qa3 = _med_iqr(_scores(aware))
        mb, qb1, qb3 = _med_iqr(_scores(blind))

        def late(runs):
            h = np.sum([r["final_eval"]["send_hist"] for r in runs], axis=0)
            return 100 * h[-2:].sum() / max(1, h.sum())
        lines.append(f"| {label} | {ma:.1f} [{qa1:.0f}-{qa3:.0f}] | {mb:.1f} [{qb1:.0f}-{qb3:.0f}] | "
                     f"{ma - mb:+.1f}{_star(p, min(len(aware), len(blind)))} | {_fmt(p, '.3f')} | "
                     f"{np.mean([r['final_eval']['msgs_mean'] for r in aware]) / 2:.1f} / "
                     f"{np.mean([r['final_eval']['msgs_mean'] for r in blind]) / 2:.1f} | "
                     f"{late(aware):.0f}% / {late(blind):.0f}% |")
    return lines + [""]


def delay_table(groups, mode, rho):
    lines = []
    for label in ("Linf", "Rinf", "Finf"):
        rows = sorted((k[2], v) for k, v in groups.items() if k[0] == mode and k[1] == rho and k[3] == label)
        if len(rows) < 2 or rows[0][0] != 0:
            continue
        base = rows[0][1]
        ps = holm([mannwhitney_p(_scores(v), _scores(base)) for d, v in rows[1:]])
        lines += [f"**{label}**", "", "| Delay | Seeds | Goals % median [IQR] | Δ vs delay 0 | p (Holm) | Msgs/agent |",
                  "|---|---|---|---|---|---|"]
        m0 = np.median(_scores(base))
        for i, (d, v) in enumerate(rows):
            med, q1, q3 = _med_iqr(_scores(v))
            p = ps[i - 1] if i else np.nan
            lines.append(f"| {d} | {len(v)} | {med:.1f} [{q1:.0f}-{q3:.0f}] | {_fmt(med - m0 if i else None, '+.1f')}"
                         f"{_star(p, min(len(v), len(base)))} | {_fmt(p, '.3f')} | "
                         f"{np.mean([h['final_eval']['msgs_mean'] for h in v]) / 2:.1f} |")
        pts = [(d, s) for d, v in rows for s in _scores(v)]
        r, p = spearman(*zip(*pts))
        lines += ["", f"Trend over delay: Spearman rho = {r:+.2f}, p = {_fmt(p, '.3g')}.", ""]
    return lines


def _boot_median_diff(a, b, rng, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    d = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(n)]
    return float(np.median(a) - np.median(b)), np.percentile(d, [2.5, 97.5])


def observability_table(groups, mode):
    rhos = sorted({k[1] for k in groups if k[0] == mode and k[2] == 0 and k[3] == "A"}, reverse=True)
    if len(rhos) < 2:
        return []
    rng = np.random.default_rng(0)
    lines = ["| rho | A median [IQR] | Linf median [IQR] | Rinf median [IQR] | Linf - A (95% CI) | Rinf - A (95% CI) |",
             "|---|---|---|---|---|---|"]
    gaps = {"Linf": [], "Rinf": []}
    for r in rhos:
        a = groups.get((mode, r, 0, "A"))
        cells = []
        for label in ("A", "Linf", "Rinf"):
            v = groups.get((mode, r, 0, label))
            if label == "Rinf" and r >= 1.0:   # with full view the rule sender is the fixed every-step sender
                v = v or groups.get((mode, r, 0, "Finf"))
            if v:
                m, q1, q3 = _med_iqr(_scores(v))
                cells.append(f"{m:.1f} [{q1:.0f}-{q3:.0f}] (n={len(v)})")
            else:
                cells.append("-")
        for label in ("Linf", "Rinf"):
            v = groups.get((mode, r, 0, label))
            if v and a:
                d, ci = _boot_median_diff(_scores(v), _scores(a), rng)
                gaps[label].append((r, _scores(v), _scores(a)))
                cells.append(f"{d:+.1f} ({ci[0]:+.1f} to {ci[1]:+.1f})")
            else:
                cells.append("-")
        lines.append(f"| {r:g} | " + " | ".join(cells) + " |")
    lines.append("")
    for label, rows in gaps.items():
        if len(rows) < 3:
            continue
        x = np.log([r for r, _, _ in rows])
        slopes = []
        for _ in range(2000):
            y = [np.median(rng.choice(v, len(v))) - np.median(rng.choice(a, len(a))) for _, v, a in rows]
            slopes.append(np.polyfit(x, y, 1)[0])
        y0 = [np.median(v) - np.median(a) for _, v, a in rows]
        s0 = np.polyfit(x, y0, 1)[0]
        lo, hi = np.percentile(slopes, [2.5, 97.5])
        lines.append(f"{label}: slope of the gain against log(rho) = {s0:+.1f} pp per log unit "
                     f"(95% bootstrap CI {lo:+.1f} to {hi:+.1f}; negative = communication helps more as view shrinks).")
    return lines + [""]


def simple_table(groups, mode, rho, labels, ref="L4"):
    rows = [(l, groups.get((mode, rho, 0, l))) for l in labels]
    rows = [(l, v) for l, v in rows if v]
    if not rows:
        return []
    base = groups.get((mode, rho, 0, ref))
    lines = [f"| Configuration | Seeds | Goals % median [IQR] | Δ vs {ref} | p |", "|---|---|---|---|---|"]
    for l, v in rows:
        m, q1, q3 = _med_iqr(_scores(v))
        p = mannwhitney_p(_scores(v), _scores(base)) if base and l != ref else np.nan
        d = m - np.median(_scores(base)) if base and l != ref else None
        lines.append(f"| {l} | {len(v)} | {m:.1f} [{q1:.0f}-{q3:.0f}] | {_fmt(d, '+.1f')} | {_fmt(p, '.3f')} |")
    return lines + [""]


# ------------------------------------------------------------------ figures

def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#0b0b0b")
    ax.set_xlabel(xlabel, color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e")


def _bx(b, inf_x):
    return inf_x if np.isinf(b) else math.log2(1 + b)


def fig_budget(plt, groups, mode, rho, sat, refs, out, ylabel):
    budgets = sorted({b for kind in KIND for b in sender_groups(groups, mode, rho, kind)})
    finite = [b for b in budgets if not np.isinf(b)]
    if len(finite) < 2:
        return
    inf_x = math.log2(1 + max(finite)) + 1.0
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    base = groups.get((mode, rho, 0, "A"))
    if base:
        m, q1, q3 = _med_iqr(_scores(base))
        ax.axhline(m, color=GRAY, ls="--", lw=1.4, label=f"A: no communication ({m:.0f})")
        ax.axhspan(q1, q3, color=GRAY, alpha=0.08, lw=0)
    for name, (val, ls) in refs.items():
        ax.axhline(val, color="#9b9a95", ls=ls, lw=1.0, label=name)
    offsets = {"learned": -0.06, "rule": 0.0, "fixed": 0.06}
    for kind, (prefix, color, marker, label) in KIND.items():
        sg = sender_groups(groups, mode, rho, kind)
        pts = sorted(b for b in sg if b > 0)
        if len(pts) < 2:
            continue
        xs = [_bx(b, inf_x) + offsets[kind] for b in pts]
        cells = [_med_iqr(_scores(sg[b])) for b in pts]
        ax.errorbar(xs, [c[0] for c in cells], yerr=[[c[0] - c[1] for c in cells], [c[2] - c[0] for c in cells]],
                    color=color, marker=marker, ms=7, lw=0, elinewidth=1.6, capsize=3, label=label,
                    markeredgecolor="white", markeredgewidth=1.0)
        fit = sat.get(kind, {}).get("fit")
        if fit and fit["b90"] is not None:
            bb = np.linspace(0, max(finite), 200)
            yy = fit["y_inf"] - (fit["y_inf"] - fit["y0"]) * np.exp(-bb / fit["beta"])
            ax.plot(np.log2(1 + bb), yy, color=color, lw=1.4, alpha=0.8)
            b90 = fit["b90"]
            if b90 <= max(finite):
                ax.axvline(math.log2(1 + b90), color=color, ls=":", lw=1.2)
                lo, hi = sat[kind]["b90_ci"]
                ax.axvspan(math.log2(1 + lo), math.log2(1 + min(hi, max(finite))), color=color, alpha=0.07, lw=0)
    ticks = [b for b in finite]
    ax.set_xticks([_bx(b, inf_x) for b in ticks] + [inf_x], [_budget_label(b) for b in ticks] + ["inf"])
    _style(ax, f"RQ1  Goals vs message budget ({mode}, rho={rho:g})",
           "Messages per agent per episode (log scale; dotted = fitted B90 with 95% CI)", ylabel)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_awareness(plt, groups, mode, rho, out, ylabel):
    pairs = sorted((budget_of(v[0]), l[:-6], v) for (m, r, d, l), v in groups.items()
                   if m == mode and r == rho and d == 0 and l.endswith("-blind"))
    pairs = [(b, l, groups.get((mode, rho, 0, l)), v) for b, l, v in pairs if groups.get((mode, rho, 0, l))]
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k, (b, l, aware, blind) in enumerate(pairs):
        for dx, runs, color, name in ((-0.18, aware, BLUE, "Sees its budget"), (0.18, blind, PURPLE, "Budget hidden")):
            m, q1, q3 = _med_iqr(_scores(runs))
            ax.bar(k + dx, m, width=0.34, color=color, alpha=0.85, label=name if k == 0 else None)
            ax.errorbar(k + dx, m, yerr=[[m - q1], [q3 - m]], color="#0b0b0b", capsize=3, lw=1)
    ax.set_xticks(range(len(pairs)), [l for _, l, _, _ in pairs])
    _style(ax, f"Budget awareness (learned gate, rho={rho:g})", "Setting", ylabel)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_delay(plt, groups, mode, rho, out, ylabel):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    drawn = False
    base = groups.get((mode, rho, 0, "A"))
    if base:
        ax.axhline(np.median(_scores(base)), color=GRAY, ls="--", lw=1.4, label="A: no communication")
    for label, color, marker in (("Linf", BLUE, "o"), ("Rinf", AQUA, "^"), ("Finf", ORANGE, "s")):
        rows = sorted((k[2], v) for k, v in groups.items() if k[0] == mode and k[1] == rho and k[3] == label)
        if len(rows) < 2:
            continue
        cells = [_med_iqr(_scores(v)) for _, v in rows]
        ax.errorbar([d for d, _ in rows], [c[0] for c in cells],
                    yerr=[[c[0] - c[1] for c in cells], [c[2] - c[0] for c in cells]],
                    color=color, marker=marker, lw=2, capsize=3, label=label)
        drawn = True
    if drawn:
        _style(ax, f"RQ2  Goals vs message delay (rho={rho:g})", "Delay (steps)", ylabel)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_obs(plt, groups, mode, out, ylabel):
    rhos = sorted({k[1] for k in groups if k[0] == mode and k[2] == 0 and k[3] == "A"}, reverse=True)
    if len(rhos) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xs = range(len(rhos))
    for label, color, marker, ls in (("A", GRAY, "D", "--"), ("Linf", BLUE, "o", "-"), ("Rinf", AQUA, "^", "-")):
        pts = [(x, groups.get((mode, r, 0, label)) or (groups.get((mode, r, 0, "Finf")) if label == "Rinf" and r >= 1 else None))
               for x, r in zip(xs, rhos)]
        pts = [(x, v) for x, v in pts if v]
        if len(pts) < 2:
            continue
        cells = [_med_iqr(_scores(v)) for _, v in pts]
        ax.errorbar([x for x, _ in pts], [c[0] for c in cells],
                    yerr=[[c[0] - c[1] for c in cells], [c[2] - c[0] for c in cells]],
                    color=color, marker=marker, ls=ls, lw=2, capsize=3, label=label)
    ax.set_xticks(list(xs), [f"{r:g}" for r in rhos])
    _style(ax, "RQ3  Goals vs observation radius", "rho (fraction of the field diagonal; right = less visible)", ylabel)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_curves(plt, groups, mode, rho, out, ylabel):
    series = [("A", GRAY, "--"), ("L4", BLUE, "-"), ("R4", AQUA, "-"), ("Linf", BLUE, ":"), ("Rinf", AQUA, ":"),
              ("Finf", ORANGE, ":")]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    drawn = False
    for label, color, ls in series:
        runs = groups.get((mode, rho, 0, label))
        if not runs:
            continue
        n = min(len(h["eval"]) for h in runs)
        v = 100.0 * np.array([[e["goals_mean"] for e in h["eval"][:n]] for h in runs])
        x = runs[0]["eval_episode"][:n]
        med = np.median(v, axis=0)
        ax.plot(x, med, color=color, ls=ls, lw=2, label=f"{label} (n={len(runs)})")
        ax.fill_between(x, np.percentile(v, 25, axis=0), np.percentile(v, 75, axis=0), color=color, alpha=0.10, lw=0)
        drawn = True
    if drawn:
        _style(ax, f"Validation goals during training (rho={rho:g}, median and IQR)", "Training episode", ylabel)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_timing(plt, groups, mode, rho, out):
    series = [(l, c) for l, c in (("L2", BLUE), ("R2", AQUA), ("L4", "#7aa9e6"), ("R4", "#79d6b3"),
                                  ("Linf", "#1b4f8f"), ("Rinf", "#0f6e4d")) if groups.get((mode, rho, 0, l))]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.8 / len(series)
    for k, (label, color) in enumerate(series):
        h = np.sum([r["final_eval"]["send_hist"] for r in groups[(mode, rho, 0, label)]], axis=0).astype(float)
        share = 100 * h / max(1.0, h.sum())
        ax.bar(np.arange(len(share)) + k * width, share, width=width * 0.95, color=color, label=label)
    T = groups[(mode, rho, 0, series[0][0])][0]["max_steps"]
    ax.set_xticks(np.arange(8) + 0.4, [f"{int(T * i / 8)}-{int(T * (i + 1) / 8)}" for i in range(8)], fontsize=8)
    _style(ax, f"When messages are sent (greedy policies, rho={rho:g})", "Episode step", "Share of messages (%)")
    ax.legend(frameon=False, fontsize=8.5, ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ reference teams

def reference_lines(base, mode, rho):
    """Random and hand-coded teams on the same held-out episodes (cached)."""
    path = os.path.join(base, "baselines.json")
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            cache = json.load(f)
    key = f"{mode}|{rho}"
    if key not in cache:
        from experiments.scripted import evaluate_reference
        cache[key] = {k: evaluate_reference(k, rho, match=(mode == "match"), n_eval=200)
                      for k in ("random", "scripted", "scripted_comm")}
        with open(path, "w") as f:
            json.dump(cache, f, indent=1)
    return cache[key]


# ------------------------------------------------------------------ main report

def build_report(base, references=True):
    histories = load_histories(base)
    groups = group_runs(histories)
    if not groups:
        print(f"No finished runs in {base}")
        return
    summarize(base)
    mode, rho = _main_key(groups)
    ylabel = "Goals per episode (%)" if mode == "single" else "Goals per match (x100)"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig_dir = os.path.join(base, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    refs, ref_lines = {}, []
    if references:
        try:
            r = reference_lines(base, mode, rho)
            refs = {f"Hand-coded team, no messages ({100 * r['scripted']['goals_mean']:.0f})": (100 * r["scripted"]["goals_mean"], ":"),
                    f"Hand-coded team, rule messages ({100 * r['scripted_comm']['goals_mean']:.0f})": (100 * r["scripted_comm"]["goals_mean"], "-.")}
            ref_lines = ["| Reference team | Goals % | Time to goal |", "|---|---|---|"] + [
                f"| {k} | {100 * v['goals_mean']:.1f} | {_fmt(v['time_to_goal'], '.0f')} |" for k, v in r.items()] + [""]
        except Exception as e:   # references are a nicety; never block the report
            print(f"reference teams skipped: {e}")

    rq1, sat = rq1_tables(groups, mode, rho)
    fig_budget(plt, groups, mode, rho, sat, refs, os.path.join(fig_dir, "rq1_budget.png"), ylabel)
    fig_awareness(plt, groups, mode, rho, os.path.join(fig_dir, "budget_awareness.png"), ylabel)
    fig_delay(plt, groups, mode, rho, os.path.join(fig_dir, "rq2_delay.png"), ylabel)
    fig_obs(plt, groups, mode, os.path.join(fig_dir, "rq3_observability.png"), ylabel)
    fig_curves(plt, groups, mode, rho, os.path.join(fig_dir, "learning_curves.png"), ylabel)
    fig_timing(plt, groups, mode, rho, os.path.join(fig_dir, "send_timing.png"))

    lines = ["# Results", "",
             f"Main regime: **{mode}-goal episodes, rho = {rho:g}**. Primary metric: goals per episode on 200 held-out "
             "episodes (single-goal mode: share of episodes with an assisted goal), in %, for the policy snapshot with the "
             "best validation score; *Last policy* is the end-of-training policy on the same episodes (*Collapsed* = "
             "seeds whose last policy scores under half of their selected snapshot). Median over seeds with "
             "interquartile range, and mean with a 95% t-interval. p: two-sided Mann-Whitney U, **Holm-corrected "
             "within each table** (* p<0.05, ** p<0.01; no stars with fewer than 5 seeds per group). AUC: mean "
             "validation score over training (learning speed). *Shuffled-content drop*: score minus the score when "
             "messages are delivered at the same times but carry another episode's content. Time to goal: first "
             "goal step among scoring episodes.", ""]
    if ref_lines:
        lines += ["## Reference teams (no learning)", ""] + ref_lines
    lines += ["## RQ1: message budget", ""] + rq1
    aw = awareness_table(groups, mode, rho)
    if aw:
        lines += ["## Budget awareness (feedback 2)", "",
                  "Learned gate with the budget features (own messages left, teammate's inferred messages left, "
                  "fraction left) vs the same setting with them hidden.", ""] + aw
    dl = delay_table(groups, mode, rho)
    if dl:
        lines += [f"## RQ2: message delay (rho = {rho:g})", ""] + dl
    ob = observability_table(groups, mode)
    if ob:
        lines += ["## RQ3: observability", ""] + ob
    second = [r for r in {k[1] for k in groups if k[0] == mode} if r != rho]
    for r2 in sorted(second, reverse=True):
        sg = sender_groups(groups, mode, r2, "rule")
        if len([b for b in sg if 0 < b < INF_BUDGET]) >= 3 and 0.0 in sg and INF_BUDGET in sg:
            t2, _ = rq1_tables(groups, mode, r2)
            lines += [f"## Saturation at a second radius (rho = {r2:g})", ""] + t2
    t4 = simple_table(groups, mode, rho, ["L4", "L4-team", "L4-rollout", "C", "Linf"])
    if t4:
        lines += ["## Other learned-sender variants", ""] + t4
    out = os.path.join(base, "results.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved figures to {fig_dir} and tables to {out}")
    return sat


# ------------------------------------------------------------------ pilot decisions

def pilot_report(base):
    """Apply the pre-registered pilot rules (README, "Pilot") and write pilot.md."""
    groups = group_runs(load_histories(base))
    med = lambda runs, f="final_eval": float(np.median(_scores(runs, f)))
    lines = ["# Pilot results and decisions", "",
             "Seeds 100-102. Scores: goals per episode (%), best validation snapshot on 200 held-out episodes "
             "(match mode: 100). Rules were fixed before the pilot ran (README, Pilot).", ""]

    # P1: parameter sharing
    lines += ["## P1: parameter sharing (single-goal, rho 0.075)", "",
              "| Setting | Shared median | Independent median | Shared AUC | Independent AUC |", "|---|---|---|---|---|"]
    share_ok = True
    for s in ("A", "Finf"):
        a, b = groups.get(("single", 0.075, 0, s)), groups.get(("single", 0.075, 0, s + "-indep"))
        if not a or not b:
            continue
        ga, gb = med(a), med(b)
        aa, ab = np.median([auc(h) for h in a]), np.median([auc(h) for h in b])
        if ga < gb - 2 or aa < ab - 2:
            share_ok = False
        lines.append(f"| {s} | {ga:.1f} | {gb:.1f} | {aa:.1f} | {ab:.1f} |")
    lines += ["", f"Decision: **{'share parameters' if share_ok else 'independent learners'}** "
              "(sharing is kept unless it is > 2 pp worse in median score or AUC for A or Finf).", ""]

    # P2: regime
    lines += ["## P2: regime", "", "| Mode | rho | A median | Finf median | Relative gap | A >= 25% of Finf | "
              "Collapsed Finf seeds | Eligible |", "|---|---|---|---|---|---|---|---|"]
    best = {}
    for (m, r, d, l), runs in sorted(groups.items()):
        if l != "A" or d != 0:
            continue
        f = groups.get((m, r, 0, "Finf"))
        if not f:
            continue
        ga, gf = med(runs), med(f)
        gap = (gf - ga) / ga if ga > 0 else np.inf
        collapsed = int(np.sum(np.asarray(_scores(f)) < 0.5 * gf))
        ok = ga >= 0.25 * gf and collapsed == 0 and gf > 0
        lines.append(f"| {m} | {r:g} | {ga:.1f} | {gf:.1f} | {gap:+.2f} | {'yes' if ga >= 0.25 * gf else 'no'} | "
                     f"{collapsed} | {'yes' if ok else 'no'} |")
        if ok and (m not in best or gap > best[m][0]):
            best[m] = (gap, r)
    if "single" in best and "match" in best:
        mode = "match" if best["match"][0] >= 1.5 * best["single"][0] else "single"
    else:
        mode = next(iter(best), "single")
    rho = best.get(mode, (None, 0.075))[1]
    lines += ["", f"Decision: **{mode} mode at rho = {rho:g}** (largest relative gap among eligible regimes; "
              "single-goal kept unless match mode's gap is >= 1.5x larger).", ""]

    # P3: senders and credit
    b = 4 if mode == "single" else 10
    lines += [f"## P3: senders ({mode}, rho 0.075)", "", "| Setting | Seeds | Median | Last-policy median | Msgs/agent |",
              "|---|---|---|---|---|"]
    for l in ("A", "Finf", f"R{b}", "Rinf", f"L{b}", f"L{b}-team", "Linf"):
        v = groups.get((mode, 0.075, 0, l))
        if v:
            lines.append(f"| {l} | {len(v)} | {med(v):.1f} | {med(v, 'final_eval_last'):.1f} | "
                         f"{np.mean([h['final_eval']['msgs_mean'] for h in v]) / 2:.1f} |")
    lr, lt = groups.get((mode, 0.075, 0, f"L{b}")), groups.get((mode, 0.075, 0, f"L{b}-team"))
    credit = "rollout" if not lt or (lr and med(lr) >= med(lt)) else "team"
    lines += ["", f"Decision: credit = **{credit}** (higher median at L{b}).", ""]

    # Episodes (convergence) and collapse
    pilot_n = 15_000 if mode == "single" else 5_000
    cands = (10_000, 15_000) if mode == "single" else (3_500, 5_000)
    chosen = cands[-1]
    for n in cands:
        moves = []
        for l in ("A", "Finf", f"L{b}"):
            runs = groups.get((mode, rho, 0, l)) or groups.get((mode, 0.075, 0, l))
            if not runs:
                continue
            ep = np.asarray(runs[0]["eval_episode"])
            v = 100 * np.median([[e["goals_mean"] for e in h["eval"]] for h in runs], axis=0)
            end = v[ep <= n][-3:].mean()
            start = v[ep <= 0.8 * n][-3:].mean()
            moves.append(end - start)
        if moves and max(abs(x) for x in moves) < 2.0:
            chosen = n
            break
    all_runs = [h for runs in groups.values() for h in runs]
    collapse = np.mean([score(h, "final_eval_last") < 0.5 * score(h) for h in all_runs]) if all_runs else 0
    lines += ["## Training length and stability", "",
              f"Episodes: **{chosen}** (smallest of {cands} where the median validation curves of A, Finf and L{b} "
              f"move < 2 pp over the last 20% of training; pilot ran {pilot_n}).",
              f"Runs whose last policy scores under half of their selected snapshot: {100 * collapse:.0f}% "
              f"-> LR decay **{'on' if collapse > 0.10 else 'off'}** (rule: on if > 10%).", ""]

    # Budget grid from the unlimited senders' usage
    usage = [np.mean([h["final_eval"]["msgs_mean"] for h in groups[(mode, 0.075, 0, l)]]) / 2
             for l in ("Rinf", "Linf") if groups.get((mode, 0.075, 0, l))]
    top = max(2.0, float(np.median(usage))) if usage else 24.0
    grid = sorted({int(round(x)) for x in np.geomspace(1, top, 9)} | {1, 2})
    lines += [f"Budget grid: unlimited senders use a median of {top:.1f} messages per agent per episode -> "
              f"finite budgets **{grid}** (roughly log-spaced from 1 to that usage).", ""]
    decisions = {"share": share_ok, "mode": mode, "rho": rho, "credit": credit, "episodes": chosen,
                 "lr_decay": bool(collapse > 0.10), "budgets": grid}
    lines += ["```json", json.dumps(decisions, indent=1), "```", ""]
    with open(os.path.join(base, "pilot.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return decisions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints/sweep_v5")
    parser.add_argument("--pilot", action="store_true", help="apply the pre-registered pilot rules")
    parser.add_argument("--no_references", action="store_true")
    args = parser.parse_args()
    if args.pilot:
        pilot_report(args.dir)
    else:
        build_report(args.dir, references=not args.no_references)
