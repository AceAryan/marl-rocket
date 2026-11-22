"""Figure 2 of the report: the v3 development sweep (budget at two view radii, and observability).

Usage: python report/make_dev_figure.py checkpoints/sweep_v3 report/figures/dev/v3_dev.png
"""
import json, glob, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root, out = sys.argv[1], sys.argv[2]
def runs(obs, setting, delay=0):
    v = [json.load(open(f))["final_eval"]["goal_rate"] * 100
         for f in glob.glob(f"{root}/obs{obs}/delay{delay}/setting_{setting}/seed_*/history.json")]
    v = np.array(v); return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)), len(v)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False})
C = {"L": "#1f77d4", "R": "#2ca58d", "F": "#f26b38", "A": "#555555"}
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), sharey=True, constrained_layout=True)
budgets = [2, 5, 10, 20, "inf"]; xs = np.arange(len(budgets))
names = {"L": ["D2", "D5", "D10", "D", "B"], "R": ["R2", "R5", "R10", "R20", "Rinf"], "F": ["F2", "F5", "F10", "F20", "Finf"]}
label = {"L": "Learned gate", "R": "Rule", "F": "Fixed schedule"}
for ax, obs in zip(axes[:2], ["0.15", "0.1"]):
    m, ci, _ = runs(obs, "A")
    ax.axhspan(m - ci, m + ci, color=C["A"], alpha=0.12, lw=0)
    ax.axhline(m, color=C["A"], ls="--", lw=1, label="No messages (A)")
    for i, k in enumerate("LRF"):
        r = [runs(obs, s) for s in names[k]]
        ax.errorbar(xs + (i - 1) * 0.09, [a for a, _, _ in r], [b for _, b, _ in r], color=C[k], marker="o",
                    ms=3, lw=1.2, capsize=2, label=label[k])
    ax.set_xticks(xs, ["2", "5", "10", "20", r"$\infty$"])
    ax.set_xlabel("Message budget per agent")
    ax.set_title(rf"Budget, $\rho={obs}$")
axes[0].set_ylabel("Goal rate (%)")
obs = ["1.0", "0.25", "0.15", "0.1"]; xo = np.arange(len(obs))
for i, (k, s, lab) in enumerate([("A", "A", "No messages (A)"), ("L", "B", "Learned, unlimited"),
                                 ("R", "Rinf", "Rule, unlimited"), ("F", "Finf", "Fixed, every step")]):
    r = [runs(o, s) for o in obs]
    axes[2].errorbar(xo + (i - 1.5) * 0.07, [a for a, _, _ in r], [b for _, b, _ in r], color=C[k], marker="o",
                     ms=3, lw=1.2, capsize=2, ls="--" if k == "A" else "-", label=lab)
axes[2].set_xticks(xo, obs); axes[2].set_xlabel(r"View radius $\rho$ (smaller = sees less)")
axes[2].set_title("Observability, unlimited budgets")
axes[0].legend(loc="lower left", frameon=False)
axes[2].legend(loc="lower left", frameon=False)
for ax in axes: ax.set_ylim(0, 100); ax.grid(axis="y", lw=0.4, alpha=0.4)
fig.savefig(out, dpi=300)
