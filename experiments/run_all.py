"""
Run all communication experiments and produce comparison plots.

Usage:
    python -m experiments.run_all --episodes 500
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
from experiments.train import train, COMM_SETTINGS

DEFAULT_SEEDS = (42, 123, 7, 99, 17)


def run_budget_sweep(
    episodes: int,
    obs_frac: float = 1.0,
    seeds=DEFAULT_SEEDS,
    max_steps: int = 200,
    eval_max_steps: int = 500,
    save_dir: str = "checkpoints",
    eval_every: int = 100,
    n_eval_episodes: int = 20,
):
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("At least one random seed is required")
    results = {}
    for setting in COMM_SETTINGS:
        results[setting] = []
        for seed in seeds:
            print(f"\n{'='*50}")
            print(f"  RUNNING SETTING {setting} (obs_frac={obs_frac}, seed={seed})")
            print(f"{'='*50}")
            h = train(
                setting=setting,
                n_episodes=episodes,
                obs_frac=obs_frac,
                seed=seed,
                max_steps=max_steps,
                eval_max_steps=eval_max_steps,
                save_dir=os.path.join(save_dir, f"seed_{seed}"),
                eval_every=eval_every,
                n_eval_episodes=n_eval_episodes,
            )
            results[setting].append(h)

    print(f"\n\n=== RESULTS SUMMARY (obs_frac={obs_frac}) ===")
    print(f"{'Setting':<10} {'Eval Reward':>19} {'Team Goals':>19} {'Opp Goals':>19}")
    print("-" * 72)
    for setting, histories in results.items():
        rewards = [h["eval_reward_mean"][-1] for h in histories]
        team_goals = [h["eval_score_team_mean"][-1] for h in histories]
        opponent_goals = [h["eval_score_opp_mean"][-1] for h in histories]
        comm_cfg = COMM_SETTINGS[setting]
        if comm_cfg["comm_every"] > 1:
            desc = f"every {comm_cfg['comm_every']} steps"
        elif comm_cfg["comm_budget"] is None:
            desc = "unlimited"
        else:
            desc = f"budget={comm_cfg['comm_budget']}"
        print(
            f"{setting} ({desc:<13}) | "
            f"{np.mean(rewards):>8.2f} +/- {np.std(rewards):<7.2f} | "
            f"{np.mean(team_goals):>8.3f} +/- {np.std(team_goals):<7.3f} | "
            f"{np.mean(opponent_goals):>8.3f} +/- {np.std(opponent_goals):<7.3f}"
        )

    return results


def run_all(
    episodes: int = 1000,
    seeds=DEFAULT_SEEDS,
    max_steps: int = 200,
    eval_max_steps: int = 500,
    save_dir: str = "checkpoints",
    eval_every: int = 100,
    n_eval_episodes: int = 20,
):
    return {
        obs_frac: run_budget_sweep(
            episodes,
            obs_frac=obs_frac,
            seeds=seeds,
            max_steps=max_steps,
            eval_max_steps=eval_max_steps,
            save_dir=save_dir,
            eval_every=eval_every,
            n_eval_episodes=n_eval_episodes,
        )
        for obs_frac in (1.0, 0.25)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,   default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max_steps", type=int,  default=200)
    parser.add_argument("--eval_max_steps", type=int, default=500)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    args = parser.parse_args()
    run_all(
        episodes=args.episodes,
        seeds=args.seeds,
        max_steps=args.max_steps,
        eval_max_steps=args.eval_max_steps,
        save_dir=args.save_dir,
        eval_every=args.eval_every,
        n_eval_episodes=args.n_eval_episodes,
    )
