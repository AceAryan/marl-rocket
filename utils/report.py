"""
Report-ready figures and tables for a finished sweep.

Writes into <dir>/figures/:
    rq1_budget.png, rq2_delay.png, rq3_observability.png,
    learning_curves.png, send_timing.png
and <dir>/results.md (one table per research question).

The primary metric is the final greedy goal rate (share of 200 held-out
episodes with an assisted goal). Per configuration: the median over seeds with
its interquartile range (robust to the occasional collapsed seed), plus the
mean with a 95% t-interval; p-values are two-sided Mann-Whitney U tests
against A (no communication) at the same rho.

Usage:
    python -m utils.report --dir checkpoints/sweep_v4 [--before <earlier sweep>]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np

from utils.plot_results import load_histories, group_runs, _ci95, mannwhitney_p

PARTIAL = 0.15
LOW = 0.1   # lowest observation radius, where shared sightings matter most
# One encoding everywhere: gray = no communication, blue = learned sender,
# aqua = send-on-sight rule, orange = fixed schedule.
GRAY, BLUE, ORANGE, AQUA = "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
LEARNED_BUDGET = [("A", 0), ("D2", 2), ("D5", 5), ("D10", 10), ("D", 20), ("B", "inf")]
FIXED_BUDGET = [("A", 0), ("F2", 2), ("F5", 5), ("F10", 10), ("F20", 20), ("Finf", "inf")]
RULE_BUDGET = [("A", 0), ("R2", 2), ("R5", 5), ("R10", 10), ("R20", 20), ("Rinf", "inf")]
LABELS = {
    "A": "A: no communication", "B": "B: learned, unlimited", "C": "C: learned, every 5 steps",
    "D2": "D2: learned, budget 2", "D5": "D5: learned, budget 5", "D10": "D10: learned, budget 10",
    "D": "D: learned, budget 20", "F2": "F2: fixed, every 40 steps", "F5": "F5: fixed, every 16 steps",
    "F10": "F10: fixed, every 8 steps", "F20": "F20: fixed, every 4 steps", "Finf": "Finf: fixed, every step",
    "R2": "R2: rule, budget 2", "R5": "R5: rule, budget 5", "R10": "R10: rule, budget 10",
    "R20": "R20: rule, budget 20", "Rinf": "Rinf: rule, unlimited",
}


# ------------------------------------------------------------------ stats

def goal_rates(runs, field="final_eval"):
    return [100.0 * h[field]["goal_rate"] for h in runs if field in h]


COLLAPSED = 50.0   # a run below this final goal rate (%) counts as collapsed


def cell(groups, key):
    """(median, median - q25, q75 - median, n) of the final goal rate (%) for
    one configuration, or None."""
    runs = groups.get(key)
    if not runs:
        return None
    g = np.asarray(goal_rates(runs))
    q25, med, q75 = np.percentile(g, [25, 50, 75])
    return float(med), float(med - q25), float(q75 - med), len(g)


def p_vs_a(groups, key):
    runs, base = groups.get(key), groups.get((key[0], 0, "A"))
    if not runs or not base or key[2] == "A":
        return np.nan
    return mannwhitney_p(goal_rates(runs), goal_rates(base))


# ------------------------------------------------------------------ figures

def _style(ax, title, xlabel, ylabel="Final goal rate (%), median and IQR"):
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
    if "goal rate" in ylabel.lower() or ylabel == "":
        ax.set_ylim(0, 105)


def _series(ax, xs, cells, color, marker, label, ls="-"):
    pts = [(x, c) for x, c in zip(xs, cells) if c is not None]
    if len(pts) < 2:   # a lone point (e.g. only the shared A anchor) is not a series
        return
    x = [p[0] for p in pts]
    m = [p[1][0] for p in pts]
    e = [[p[1][1] for p in pts], [p[1][2] for p in pts]]   # interquartile range
    ax.errorbar(x, m, yerr=e, color=color, marker=marker, ms=8, lw=2, ls=ls, capsize=4,
                label=label, markeredgecolor="white", markeredgewidth=1.2)


def _a_line(ax, groups, rho, xmax, ls="--", suffix=""):
    c = cell(groups, (rho, 0, "A"))
    if c:
        ax.axhline(c[0], color=GRAY, ls=ls, lw=1.5, label=f"A: no communication{suffix} ({c[0]:.0f}%)")
        ax.axhspan(c[0] - c[1], c[0] + c[2], color=GRAY, alpha=0.08, lw=0)


def fig_budget(plt, groups, out):
    rhos = [r for r in (PARTIAL, LOW) if (r, 0, "A") in groups]
    if not rhos:
        return
    fig, axes = plt.subplots(1, len(rhos), figsize=(6.2 * len(rhos), 4.6), sharey=True, squeeze=False)
    xs = range(len(LEARNED_BUDGET))
    for ax, rho in zip(axes[0], rhos):
        _a_line(ax, groups, rho, len(xs))
        for budget, color, marker, label in ((LEARNED_BUDGET, BLUE, "o", "Learned sender"),
                                             (RULE_BUDGET, AQUA, "^", "Rule: send on sight"),
                                             (FIXED_BUDGET, ORANGE, "s", "Fixed schedule")):
            _series(ax, xs, [cell(groups, (rho, 0, s)) for s, _ in budget], color, marker, label)
        ax.set_xticks(list(xs), [str(b) for _, b in LEARNED_BUDGET])
        _style(ax, f"RQ1  Goal rate vs message budget (rho={rho})",
               "Messages per agent per 80-step episode (fixed: 80 / period)",
               "Final goal rate (%)" if ax is axes[0][0] else "")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_delay(plt, groups, out):
    delays = sorted({k[1] for k in groups if k[0] == PARTIAL and k[2] in ("B", "Rinf", "Finf")})
    if not delays:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    _a_line(ax, groups, PARTIAL, len(delays))
    _series(ax, delays, [cell(groups, (PARTIAL, d, "Finf")) for d in delays], ORANGE, "s",
            "Fixed, every step (Finf)")
    _series(ax, delays, [cell(groups, (PARTIAL, d, "Rinf")) for d in delays], AQUA, "^",
            "Rule, unlimited (Rinf)")
    _series(ax, delays, [cell(groups, (PARTIAL, d, "B")) for d in delays], BLUE, "o",
            "Learned, unlimited (B)")
    ax.set_xticks(delays)
    _style(ax, f"RQ2  Goal rate vs message delay (rho={PARTIAL})", "Delay (steps)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_obs(plt, groups, out):
    rhos = sorted({k[0] for k in groups if k[1] == 0 and k[2] == "A"})
    if len(rhos) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    xs = range(len(rhos))
    _series(ax, xs, [cell(groups, (r, 0, "A")) for r in rhos], GRAY, "D", "A: no communication", ls="--")
    _series(ax, xs, [cell(groups, (r, 0, "Finf")) for r in rhos], ORANGE, "s", "Fixed, every step (Finf)")
    _series(ax, xs, [cell(groups, (r, 0, "Rinf")) for r in rhos], AQUA, "^", "Rule, unlimited (Rinf)")
    _series(ax, xs, [cell(groups, (r, 0, "B")) for r in rhos], BLUE, "o", "Learned, unlimited (B)")
    ax.set_xticks(list(xs), [f"{r:g}" for r in rhos])
    _style(ax, "RQ3  Goal rate vs observation radius",
           "Observation radius rho (fraction of field diagonal; smaller = less visible)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _curve(ax, runs, color, label, ls="-"):
    length = min(len(h["eval"]) for h in runs)
    v = 100.0 * np.array([[e["goal_rate"] for e in h["eval"][:length]] for h in runs])
    xs = runs[0]["eval_episode"][:length]
    mean = v.mean(0)
    se = v.std(0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else 0 * mean
    ax.plot(xs, mean, color=color, lw=2, ls=ls, label=f"{label} (n={len(runs)})")
    ax.fill_between(xs, mean - se, mean + se, color=color, alpha=0.15, lw=0)


def fig_curves(plt, groups, out):
    panels = [("Budget 5", [("A", GRAY, "--"), ("D5", BLUE, "-"), ("R5", AQUA, "-"), ("F5", ORANGE, "-")]),
              ("Unlimited", [("A", GRAY, "--"), ("B", BLUE, "-"), ("Rinf", AQUA, "-"), ("Finf", ORANGE, "-")])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, series) in zip(axes, panels):
        for s, color, ls in series:
            runs = groups.get((PARTIAL, 0, s))
            if runs:
                _curve(ax, runs, color, LABELS[s], ls)
        _style(ax, f"{title} (rho={PARTIAL})", "Training episode",
               "Greedy goal rate (%), mean ± SE" if ax is axes[0] else "")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def send_timing(run_dirs, n_episodes=40, seed=30_000, legacy=False):
    """Replay greedy policies and record when each message is sent and whether
    the sender saw the ball. run_dirs: list of (setting, rho, seed_dir).
    legacy=True replays the pre-fix rule (a learned decision at every step
    the agent may send), for checkpoints trained before event triggering."""
    from experiments.train import make_env, choose_messages, sender_kind, BALL_VIS, HEAD_MESSAGES
    from agents.dqn_agent import DQNAgent
    from env.rocket_env import N_ACTIONS, N_MESSAGES
    steps, visible = [], []
    for setting, rho, path in run_dirs:
        env = make_env(setting, rho)
        agents = [DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=N_ACTIONS,
                           n_messages=N_MESSAGES if legacy else HEAD_MESSAGES, agent_id=i)
                  for i in range(2)]
        for a in agents:
            a.load(path)
            a.eps = 0.0
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            t = 0
            while True:
                acts = [a.select_action(o) for a, o in zip(agents, obs)]
                if legacy:
                    msgs = [a.select_message(o, budget_exhausted=not c)
                            for a, o, c in zip(agents, obs, info["can_send"])]
                else:
                    msgs, _ = choose_messages(agents, obs, info["can_send"], sender_kind(setting))
                for i, m in enumerate(msgs):
                    if m and info["can_send"][i]:
                        steps.append(t)
                        visible.append(obs[i][BALL_VIS] > 0)
                obs, _, term, trunc, info = env.step(acts, msgs)
                t += 1
                if term or trunc:
                    break
    return np.array(steps), np.array(visible, dtype=bool)


def _seed_dirs(base, setting, rho=PARTIAL):
    d = os.path.join(base, f"obs{rho}", "delay0", f"setting_{setting}")
    if not os.path.isdir(d):
        return []
    return [(setting, rho, os.path.join(d, s)) for s in sorted(os.listdir(d))
            if os.path.exists(os.path.join(d, s, "agent0_msg.npz"))]


def fig_send_timing(plt, base, before, out, setting="D5", fixed="F5"):
    series = []
    if before:
        series.append((f"Learned {setting}, before fix", _seed_dirs(before, setting), GRAY, True))
    series.append((f"Learned {setting}", _seed_dirs(base, setting), BLUE, False))
    series.append((f"Rule R{setting[1:]}", _seed_dirs(base, "R" + setting[1:]), AQUA, False))
    series.append((f"Fixed {fixed}", _seed_dirs(base, fixed), ORANGE, False))
    series = [s for s in series if s[1]]
    if not series:
        return None
    bins = np.arange(0, 81, 5)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rows = []
    width = 5 / (len(series) + 0.5)
    for k, (label, dirs, color, legacy) in enumerate(series):
        steps, vis = send_timing(dirs, legacy=legacy)
        if len(steps) == 0:
            continue
        hist, _ = np.histogram(steps, bins=bins)
        share = 100.0 * hist / len(steps)
        ax.bar(bins[:-1] + k * width, share, width=width * 0.9, align="edge", color=color,
               label=f"{label}: median step {np.median(steps):.0f}, ball seen {100 * vis.mean():.0f}%")
        rows.append((label, len(dirs), np.median(steps), 100 * np.mean(steps < 10), 100 * vis.mean()))
    _style(ax, f"When are messages sent? (rho={PARTIAL}, greedy policies)",
           "Episode step", "Share of messages (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return rows


# ------------------------------------------------------------------ tables

def _row(groups, key, label=None):
    runs = groups[key]
    g = np.asarray(goal_rates(runs))
    sil = goal_rates(runs, "final_eval_silenced")
    a = groups.get((key[0], 0, "A"))
    diff = np.median(g) - np.median(goal_rates(a)) if a and key[2] != "A" else np.nan
    q25, med, q75 = np.percentile(g, [25, 50, 75])
    p = p_vs_a(groups, key)
    ep_len = np.mean([h["final_eval"]["episode_length"] for h in runs])
    msgs = np.mean([h["final_eval"]["msgs_mean"] for h in runs]) / 2
    fmt = lambda v, f: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, f)
    star = "" if np.isnan(p) else (" **" if p < 0.01 else " *" if p < 0.05 else "")
    return (f"| {label or LABELS.get(key[2], key[2])} | {len(g)} | {med:.1f} [{q25:.0f}-{q75:.0f}] | "
            f"{np.mean(g):.1f} ± {_ci95(g):.1f} | {fmt(diff, '+.1f')}{star} | {fmt(p, '.3f')} | "
            f"{int(np.sum(g < COLLAPSED))} | {ep_len:.0f} | {msgs:.1f} | "
            f"{fmt(float(np.median(sil)) if sil else None, '.1f')} |")


HEADER = ("| Setting | Seeds | Goal % median [IQR] | Mean ± 95% CI | Δ median vs A (pp) | p | "
          "Collapsed | Steps to goal | Msgs/agent | Goal % silenced |\n"
          "|---|---|---|---|---|---|---|---|---|---|")


def write_results_md(groups, base, timing_rows=None):
    lines = ["# Results", "",
             "Each run keeps the policy snapshot with the best goal rate on the periodic "
             "validation episodes; it is then scored on 200 separate held-out episodes. "
             "Goal rate = share of "
             "episodes with an assisted goal; median over seeds with its interquartile range, "
             "and the mean with a 95% t-interval. Δ: difference in medians from A at the same "
             "observation radius; p: two-sided Mann-Whitney U test against A (* p<0.05, "
             "** p<0.01). Collapsed = seeds below 50% goals. Steps to goal = mean episode length "
             "(80 = no goal). *Goal % silenced* = the same policy evaluated with the channel "
             "muted (how much it relies on messages).", ""]

    def table(title, keys, show_delay=False):
        keys = [k for k in keys if k in groups]
        if not keys:
            return
        rows = [_row(groups, k, f"{LABELS.get(k[2], k[2])}, delay {k[1]}"
                     if show_delay and k[2] != "A" else None) for k in keys]
        lines.extend([f"## {title}", "", HEADER] + rows + [""])

    order = ["A", "D2", "D5", "D10", "D", "C", "B", "R2", "R5", "R10", "R20", "Rinf",
             "F2", "F5", "F10", "F20", "Finf"]
    table(f"RQ1: message budget (rho = {PARTIAL})", [(PARTIAL, 0, s) for s in order])
    table(f"RQ1: message budget (rho = {LOW})", [(LOW, 0, s) for s in order])
    table("RQ1: message budget (full view, rho = 1.0)", [(1.0, 0, s) for s in order])
    delays = sorted({k[1] for k in groups if k[0] == PARTIAL})
    table(f"RQ2: message delay (rho = {PARTIAL})",
          [(PARTIAL, 0, "A")] + [(PARTIAL, d, s) for s in ("B", "Rinf", "Finf") for d in delays],
          show_delay=True)
    rhos = sorted({k[0] for k in groups if k[1] == 0 and k[2] == "A"}, reverse=True)
    if len(rhos) > 1:
        lines.extend(["## RQ3: observability", "",
                      "| rho | " + " | ".join(LABELS[s] for s in ("A", "B", "Rinf", "Finf")) + " |",
                      "|---|---|---|---|---|"])
        for r in rhos:
            cells = []
            for s in ("A", "B", "Rinf", "Finf"):
                c = cell(groups, (r, 0, s))
                p = p_vs_a(groups, (r, 0, s))
                star = "" if np.isnan(p) else (" **" if p < 0.01 else " *" if p < 0.05 else "")
                cells.append("-" if c is None else f"{c[0]:.1f} [{c[0] - c[1]:.0f}-{c[0] + c[2]:.0f}]{star}")
            lines.append(f"| {r:g} | " + " | ".join(cells) + " |")
        lines.append("")
    if timing_rows:
        lines.extend(["## Message timing (greedy, rho = 0.15)", "",
                      "| Policy | Seeds | Median send step | % sent in first 10 steps | % sent while seeing ball |",
                      "|---|---|---|---|---|"])
        for label, n, med, early, vis in timing_rows:
            lines.append(f"| {label} | {n} | {med:.0f} | {early:.0f} | {vis:.0f} |")
        lines.append("")
    out = os.path.join(base, "results.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def build_report(base, before=None):
    groups = group_runs(load_histories(base))
    if not groups:
        print(f"No finished runs in {base}")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig_dir = os.path.join(base, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_budget(plt, groups, os.path.join(fig_dir, "rq1_budget.png"))
    fig_delay(plt, groups, os.path.join(fig_dir, "rq2_delay.png"))
    fig_obs(plt, groups, os.path.join(fig_dir, "rq3_observability.png"))
    fig_curves(plt, groups, os.path.join(fig_dir, "learning_curves.png"))
    timing = fig_send_timing(plt, base, before, os.path.join(fig_dir, "send_timing.png"))
    out = write_results_md(groups, base, timing)
    print(f"Saved figures to {fig_dir} and tables to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints/sweep_v4")
    parser.add_argument("--before", default=None,
                        help="earlier sweep with the same observation layout, shown as 'before fix' "
                             "in the send-timing figure")
    args = parser.parse_args()
    build_report(args.dir, args.before if args.before and os.path.isdir(args.before) else None)
