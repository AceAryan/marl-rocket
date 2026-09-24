"""
Run all communication experiments and produce comparison plots.

Usage:
    python -m experiments.run_all --episodes 500
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
from experiments.train import train, COMM_SETTINGS


def smooth(x, w=20):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='valid')


def run_all(episodes: int = 1000, obs_frac: float = 1.0, seed: int = 42):
    results = {}
    for setting in COMM_SETTINGS:
        print(f"\n{'='*50}")
        print(f"  RUNNING SETTING {setting}")
        print(f"{'='*50}")
        h = train(
            setting=setting,
            n_episodes=episodes,
            obs_frac=obs_frac,
            seed=seed,
        )
        results[setting] = h

    # --- Print summary table ---
    print("\n\n=== RESULTS SUMMARY ===")
    print(f"{'Setting':<10} {'Avg Reward':>12} {'Team Goals':>12} {'Opp Goals':>12} {'Msgs/Ep':>10}")
    print("-" * 58)
    for s, h in results.items():
        tail = slice(-min(200, episodes), None)
        r  = np.mean(h["episode_reward"][tail])
        g  = np.mean(h["score_team"][tail])
        ag = np.mean(h["score_opp"][tail])
        m  = np.mean(h["msgs_sent"][tail])
        desc = {
            "A": "No comm",
            "B": "Every step",
            "C": "Every 5 steps",
            "D": "Budget=20",
        }[s]
        print(f"{s} ({desc:<13}) | {r:>10.1f} | {g:>10.2f} | {ag:>10.2f} | {m:>8.1f}")

    # Try to save a simple ASCII plot
    print("\n\n=== SMOOTHED REWARD CURVES (terminal) ===")
    colors = {"A": "●", "B": "■", "C": "▲", "D": "★"}
    for s, h in results.items():
        curve = smooth(h["episode_reward"], w=max(1, episodes // 50))
        max_r = max(abs(v) for v in curve) + 1e-8
        bar = "".join(
            colors[s] if v > 0 else "·"
            for v in curve[::max(1, len(curve)//60)]
        )
        print(f"  {s}: {bar}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,   default=1000)
    parser.add_argument("--obs_frac", type=float, default=1.0)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()
    run_all(episodes=args.episodes, obs_frac=args.obs_frac, seed=args.seed)
