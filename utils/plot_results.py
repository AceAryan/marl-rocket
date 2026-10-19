"""
Plot training curves from saved history.json files.
Requires matplotlib.

Usage:
    python utils/plot_results.py --dir checkpoints
"""

import argparse
import json
import os
import glob
import numpy as np


def smooth(x, w=30):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode='valid').tolist()


def load_histories(base_dir: str):
    results = {}
    for path in sorted(glob.glob(os.path.join(base_dir, "**", "history.json"), recursive=True)):
        with open(path) as f:
            h = json.load(f)
        key = os.path.relpath(os.path.dirname(path), base_dir).replace(os.sep, "/")
        results[key] = h
    return results


def plot_saturation(results, ax):
    budget_settings = {"A": 0, "B1": 5, "B2": 10, "B3": 20, "B4": 50, "B5": 100}
    observations = (1.0, 0.25)
    scores = {
        obs_frac: {setting: [] for setting in (*budget_settings, "D")}
        for obs_frac in observations
    }

    for history in results.values():
        setting = history.get("setting")
        obs_frac = history.get("obs_frac")
        if obs_frac not in scores or setting not in scores[obs_frac]:
            continue

        eval_rewards = history.get("eval_reward_mean", [])
        train_rewards = history.get("episode_reward", [])
        if eval_rewards:
            scores[obs_frac][setting].append(float(eval_rewards[-1]))
        elif train_rewards:
            tail = train_rewards[-min(200, len(train_rewards)):]
            scores[obs_frac][setting].append(float(np.mean(tail)))

    for obs_frac in observations:
        budgets = []
        means = []
        stds = []
        for setting, budget in budget_settings.items():
            values = scores[obs_frac][setting]
            if values:
                budgets.append(budget)
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values)))
        if budgets:
            marker = "o" if obs_frac == 1.0 else "s"
            ax.errorbar(
                budgets,
                means,
                yerr=stds,
                marker=marker,
                capsize=3,
                label=f"obs_frac={obs_frac}",
            )

        unlimited = scores[obs_frac]["D"]
        if unlimited:
            mean_reward = float(np.mean(unlimited))
            std_reward = float(np.std(unlimited))
            ax.axhline(mean_reward, linestyle="--", alpha=0.65,
                       label=f"unlimited, obs_frac={obs_frac}")
            ax.axhspan(mean_reward - std_reward, mean_reward + std_reward, alpha=0.08)

    ax.set_title("Reward vs. communication budget")
    ax.set_xlabel("Messages per agent per episode")
    ax.set_ylabel("Mean episode reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot(base_dir: str, smooth_w: int = 30, save: bool = False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    results = load_histories(base_dir)
    if not results:
        print(f"No history files found in {base_dir}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Communication Budget Experiment", fontsize=13)

    groups = {}
    for key, history in results.items():
        group = (history.get("setting", key), history.get("obs_frac", "unknown"))
        groups.setdefault(group, []).append(history)

    for (setting, obs_frac), histories in sorted(groups.items()):
        label = f"{setting} / obs={obs_frac}"
        for ax, field in zip(axes, ("episode_reward", "score_team", "msgs_sent")):
            curves = [smooth(history[field], smooth_w) for history in histories if history.get(field)]
            if not curves:
                continue
            length = min(map(len, curves))
            values = np.asarray([curve[:length] for curve in curves], dtype=np.float64)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            xs = np.arange(length)
            ax.plot(xs, mean, label=label)
            if len(curves) > 1:
                ax.fill_between(xs, mean - std, mean + std, alpha=0.15)

    saturation_fig, saturation_ax = plt.subplots(figsize=(7, 4))
    plot_saturation(results, saturation_ax)

    axes[0].set_title("Episode Reward")
    axes[1].set_title("Goals Scored (Team)")
    axes[2].set_title("Messages Sent / Episode")

    for ax in axes:
        ax.legend(fontsize=7)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    if save:
        out = os.path.join(base_dir, "results.png")
        fig.savefig(out, dpi=150)
        print(f"Saved to {out}")
        saturation_out = os.path.join(base_dir, "saturation_curve.png")
        saturation_fig.savefig(saturation_out, dpi=150)
        print(f"Saved to {saturation_out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",    default="checkpoints")
    parser.add_argument("--smooth", type=int, default=30)
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()
    plot(args.dir, args.smooth, args.save)
