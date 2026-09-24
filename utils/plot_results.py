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
    for path in sorted(glob.glob(os.path.join(base_dir, "*/history.json"))):
        with open(path) as f:
            h = json.load(f)
        key = os.path.basename(os.path.dirname(path))
        results[key] = h
    return results


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

    labels = {
        "setting_A_obs1.0": "A: No comm",
        "setting_B_obs1.0": "B: Every step",
        "setting_C_obs1.0": "C: Every 5 steps",
        "setting_D_obs1.0": "D: Budget=20",
    }

    for key, h in results.items():
        label = labels.get(key, key)
        r  = smooth(h["episode_reward"], smooth_w)
        g  = smooth(h["score_team"],     smooth_w)
        m  = smooth(h["msgs_sent"],      smooth_w)
        xs = range(len(r))

        axes[0].plot(xs, r, label=label)
        axes[1].plot(xs, g, label=label)
        axes[2].plot(xs, m, label=label)

    axes[0].set_title("Episode Reward")
    axes[1].set_title("Goals Scored (Team)")
    axes[2].set_title("Messages Sent / Episode")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        out = os.path.join(base_dir, "results.png")
        plt.savefig(out, dpi=150)
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",    default="checkpoints")
    parser.add_argument("--smooth", type=int, default=30)
    parser.add_argument("--save",   action="store_true")
    args = parser.parse_args()
    plot(args.dir, args.smooth, args.save)
