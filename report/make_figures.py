"""Figures of the report drawn for print: the v3 and v4 development sweeps and the efficiency plot
(vector PDF, Times, 8 pt). The main-sweep figures are written by utils/report.py.

Usage (from the repository root):
    python report/make_figures.py
"""

import os
import sys
import glob
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.report import GRAY, BLUE, AQUA, ORANGE, KIND                         # noqa: E402

OUT = os.path.join(ROOT, "report", "figures")
FULL = 7.0                # page width (inches)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.handlelength": 1.8, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def grid(ax):
    ax.grid(axis="y", color="#e4e3df", lw=0.5)
    ax.set_axisbelow(True)


def errbar(ax, x, cells, **kw):
    kw.setdefault("capsize", 1.8)
    kw.setdefault("ms", 3.5)
    kw.setdefault("elinewidth", 0.8)
    ax.errorbar(x, [c[0] for c in cells], yerr=[[c[0] - c[1] for c in cells], [c[2] - c[0] for c in cells]], **kw)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


# ------------------------------------------------------------------ main sweep (from its summary)

def fig_efficiency():
    """Goal rate against the messages actually sent, one point per main-sweep configuration."""
    import csv
    import re
    base = os.path.join(ROOT, "checkpoints", "sweep_v5")
    rows = [r for r in csv.DictReader(open(os.path.join(base, "summary.csv")))
            if r["obs_frac"] == "0.06" and r["delay"] == "0"]
    a = next(float(r["goals_median"]) for r in rows if r["label"] == "A-indep")
    scripted = 100 * json.load(open(os.path.join(base, "baselines.json")))["single|0.06"]["scripted"]["goals_mean"]
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    ax.axhline(a, color=GRAY, ls="--", lw=0.9, label="No messages (A)")
    ax.axhline(scripted, color="#9b9a95", ls=":", lw=0.9, label="Hand-coded team")
    for kind, (prefix, color, marker, label) in KIND.items():
        pts = sorted((float(r["msgs_per_agent"]), float(r["goals_median"]), float(r["goals_q25"]),
                      float(r["goals_q75"]), r["label"][:-6]) for r in rows
                     if re.fullmatch(rf"{prefix}(\d+|inf)-indep", r["label"]))
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], color=color, marker=marker,
                    ms=3.5, lw=0, elinewidth=0.6, capsize=1.5, markeredgecolor="white", markeredgewidth=0.4,
                    label=label.replace("Rule: send on sight", "Rule"))
        for x, y, _, _, name in pts:
            if name.endswith("inf") and kind != "rule":
                ax.annotate(f"{name}", (x, y), xytext={"learned": (5, 2), "rule": (8, -14), "fixed": (-8, 6)}[kind],
                            textcoords="offset points", fontsize=7, color=color, family="monospace")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32], ["0.5", "1", "2", "4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlim(0.45, 70)
    ax.set_ylim(0, 85)
    ax.set_xlabel("Messages actually sent per agent per episode (log scale)")
    ax.set_ylabel("Goal rate (%)")
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0)
    grid(ax)
    save(fig, "efficiency.pdf")


# ------------------------------------------------------------------ development runs

def _runs(root, obs, setting, delay=0):
    return [json.load(open(f)) for f in
            sorted(glob.glob(f"{root}/obs{obs}/delay{delay}/setting_{setting}/seed_*/history.json"))]


def _mean_ci(root, obs, setting):
    v = np.array([h["final_eval"]["goal_rate"] * 100 for h in _runs(root, obs, setting)])
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - ci, v.mean() + ci


def fig_v3():
    root = os.path.join(ROOT, "checkpoints", "sweep_v3")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0), sharey=True)
    names = {"learned": ["D2", "D5", "D10", "D", "B"], "rule": ["R2", "R5", "R10", "R20", "Rinf"],
             "fixed": ["F2", "F5", "F10", "F20", "Finf"]}
    xs = np.arange(5)
    for ax, obs, tag in zip(axes[:2], ["0.15", "0.1"], ["(a)", "(b)"]):
        m, lo, hi = _mean_ci(root, obs, "A")
        ax.axhspan(lo, hi, color=GRAY, alpha=0.10, lw=0)
        ax.axhline(m, color=GRAY, ls="--", lw=0.9, label="No messages (A)")
        for i, (kind, (_, color, marker, label)) in enumerate(KIND.items()):
            errbar(ax, xs + (i - 1) * 0.1, [_mean_ci(root, obs, s) for s in names[kind]], color=color,
                   marker=marker, label=label.replace("Rule: send on sight", "Rule"))
        ax.set_xticks(xs, ["2", "5", "10", "20", r"$\infty$"])
        ax.set_xlabel(rf"Message budget $B$ ($\rho={obs}$)")
        ax.text(0.0, 1.03, tag, transform=ax.transAxes, fontweight="bold")
    obs = ["1.0", "0.25", "0.15", "0.1"]
    for i, (s, color, marker, ls, name) in enumerate([("A", GRAY, "D", "--", "No messages (A)"),
                                                      ("B", BLUE, "o", "-", "Learned, unlimited"),
                                                      ("Rinf", AQUA, "^", "-", "Rule, unlimited"),
                                                      ("Finf", ORANGE, "s", "-", "Every step")]):
        errbar(axes[2], np.arange(4) + (i - 1.5) * 0.07, [_mean_ci(root, o, s) for o in obs], color=color,
               marker=marker, ls=ls, label=name)
    axes[2].set_xticks(range(4), obs)
    axes[2].set_xlabel(r"View radius $\rho$ (unlimited budgets)")
    axes[2].text(0.0, 1.03, "(c)", transform=axes[2].transAxes, fontweight="bold")
    axes[0].set_ylabel("Goal rate (%)")
    axes[0].legend(loc="lower left")
    axes[2].legend(loc="lower left")
    for ax in axes:
        ax.set_ylim(0, 100)
        grid(ax)
    fig.tight_layout(w_pad=1.2)
    save(fig, os.path.join("dev", "v3_dev.pdf"))


def fig_v4():
    on, off = (os.path.join(ROOT, "checkpoints", d) for d in ("sweep_v4", "sweep_v4_offplan"))
    panels = [("Budget 5", [(on, "A", GRAY, "--", "No messages (A)"), (off, "D5", BLUE, "-", "Learned, $B=5$"),
                            (off, "R5", AQUA, "-", "Rule, $B=5$"), (off, "F5", ORANGE, "-", "Fixed, every 16 steps")]),
              ("Unlimited", [(on, "A", GRAY, "--", "No messages (A)"), (on, "B", BLUE, "-", "Learned, unlimited"),
                             (on, "Rinf", AQUA, "-", "Rule, unlimited"), (on, "Finf", ORANGE, "-", "Every step")])]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45), sharey=True)
    for ax, (title, series), tag in zip(axes, panels, ["(a)", "(b)"]):
        for root, s, color, ls, name in series:
            runs = _runs(root, "0.15", s)
            n = min(len(h["eval"]) for h in runs)
            v = 100.0 * np.array([[e["goal_rate"] for e in h["eval"][:n]] for h in runs])
            x = np.array(runs[0]["eval_episode"][:n]) / 1000
            m, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(runs))
            ax.plot(x, m, color=color, ls=ls, lw=0.9, label=name)
            ax.fill_between(x, m - se, m + se, color=color, alpha=0.13, lw=0)
        ax.set_xlabel("Training episode (thousands)")
        ax.set_xlim(0, x[-1])
        ax.set_ylim(0, 100)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, columnspacing=1.5, handlelength=1.5)
        ax.text(0.0, 1.03, f"{tag} {title}", transform=ax.transAxes, fontweight="bold")
        grid(ax)
    axes[0].set_ylabel("Validation goal rate (%)")
    fig.tight_layout(w_pad=1.2)
    save(fig, os.path.join("dev", "v4_learning_curves.pdf"))


if __name__ == "__main__":
    fig_efficiency()
    fig_v3()
    fig_v4()
