"""Train and plot a greedy two-agent goal-rate sanity check."""

import argparse
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from env.rocket_env import N_ACTIONS, N_MESSAGES, SHAPING_GAMMA, RocketEnv

RANDOM_BASELINE_GOAL_RATE = 0.12


def evaluate(agents: list[DQNAgent], env: RocketEnv, episodes: int, seed: int) -> dict:
    previous_epsilons = [agent.eps for agent in agents]
    previous_random_state = random.getstate()
    for agent in agents:
        agent.eps = 0.0
    goals_per_episode = []
    ball_progress_per_episode = []
    try:
        for episode in range(episodes):
            observations, info = env.reset(seed=seed + episode)
            previous_ball_y = float(info["ball_pos"][1])
            ball_progress = 0.0
            for _ in range(env.max_steps):
                actions = [
                    agent.select_action(observations[i])
                    for i, agent in enumerate(agents)
                ]
                observations, _, terminated, truncated, info = env.step(actions)
                current_ball_y = float(info["ball_pos"][1])
                ball_progress += current_ball_y - previous_ball_y
                previous_ball_y = current_ball_y
                if terminated or truncated:
                    break
            goals_per_episode.append(info["score_team"])
            ball_progress_per_episode.append(ball_progress)
    finally:
        for agent, epsilon in zip(agents, previous_epsilons):
            agent.eps = epsilon
        random.setstate(previous_random_state)

    return {
        "mean_goals": float(np.mean(goals_per_episode)),
        "goal_rate": float(np.mean(np.asarray(goals_per_episode) > 0)),
        "ball_progress": float(np.mean(ball_progress_per_episode)),
    }


def train(
    episodes: int = 3000,
    max_steps: int = 200,
    eval_episodes: int = 50,
    eval_every: int = 100,
    seed: int = 42,
    output_dir: str = "checkpoints/two_agent_sanity",
) -> dict:
    if episodes <= 0 or max_steps <= 0 or eval_episodes <= 0 or eval_every <= 0:
        raise ValueError("episode counts, max_steps, and eval_every must be positive")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    env = RocketEnv(
        n_agents=2,
        curriculum="easy",
        opponent=True,
        comm_budget=0,
        max_steps=max_steps,
    )
    eval_env = RocketEnv(
        n_agents=2,
        curriculum="easy",
        opponent=True,
        comm_budget=0,
        max_steps=max_steps,
    )
    agents = [
        DQNAgent(
            obs_dim=env.observation_space.shape[0],
            n_actions=N_ACTIONS,
            n_messages=N_MESSAGES,
            hidden=128,
            lr=3e-4,  # was 1e-3 — overrode the dqn_agent.py fix; this is the actual bug
            gamma=SHAPING_GAMMA,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=max(1, episodes * max_steps // 4),
            batch_size=64,
            target_update=1500,
            agent_id=i,
        )
        for i in range(env.n_agents)
    ]

    history = {
        "eval_episode": [],
        "eval_mean_goals": [],
        "eval_goal_rate": [],
        "eval_ball_progress": [],
        "random_baseline_goal_rate": RANDOM_BASELINE_GOAL_RATE,
        "epsilon": [],
        "goal_buffer_size": [],
    }
    best_goal_rate = -1.0
    best_episode = None
    started = time.time()
    for episode in range(1, episodes + 1):
        observations, _ = env.reset(seed=seed if episode == 1 else None)
        for _ in range(max_steps):
            actions = [
                agent.select_action(observations[i])
                for i, agent in enumerate(agents)
            ]
            next_observations, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            for i, agent in enumerate(agents):
                agent.store(
                    observations[i],
                    actions[i],
                    reward,
                    next_observations[i],
                    terminated,
                    goal_transition=terminated and info["score_team"] > 0,
                )
                agent.update()
            observations = next_observations
            if done:
                break

        if episode % eval_every == 0 or episode == episodes:
            metrics = evaluate(
                agents,
                eval_env,
                eval_episodes,
                seed + episode * eval_episodes,
            )
            history["eval_episode"].append(episode)
            history["eval_mean_goals"].append(metrics["mean_goals"])
            history["eval_goal_rate"].append(metrics["goal_rate"])
            history["eval_ball_progress"].append(metrics["ball_progress"])
            history["epsilon"].append(agents[0].eps)
            history["goal_buffer_size"].append(len(agents[0].goal_buffer))
            if metrics["goal_rate"] > best_goal_rate:
                best_goal_rate = metrics["goal_rate"]
                best_episode = episode
                for agent in agents:
                    agent.save(os.path.join(output_dir, "best"))
            print(
                f"Episode {episode:4d}/{episodes} | "
                f"greedy goals/episode={metrics['mean_goals']:.2f} | "
                f"goal rate={metrics['goal_rate']:.0%} | "
                f"baseline_cleared="
                f"{metrics['goal_rate'] >= RANDOM_BASELINE_GOAL_RATE} | "
                f"ball_progress={metrics['ball_progress']:.2f} | "
                f"epsilon={agents[0].eps:.3f} | "
                f"goal_buffer={len(agents[0].goal_buffer)} | "
                f"elapsed={time.time() - started:.0f}s"
            )

    os.makedirs(output_dir, exist_ok=True)
    for agent in agents:
        agent.save(output_dir)
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as history_file:
        json.dump(history, history_file, indent=2)

    figure, (goal_axis, progress_axis) = plt.subplots(2, 1, figsize=(7, 7))
    goal_axis.plot(history["eval_episode"], history["eval_goal_rate"], marker="o")
    goal_axis.axhline(
        RANDOM_BASELINE_GOAL_RATE,
        color="gray",
        linestyle="--",
        label=f"random baseline ({RANDOM_BASELINE_GOAL_RATE:.2f})",
    )
    goal_axis.set(
        title="Two-agent greedy evaluation",
        ylabel="Episodes scoring a goal",
        ylim=(0, 1),
    )
    goal_axis.legend()
    goal_axis.grid(alpha=0.3)
    progress_axis.plot(history["eval_episode"], history["eval_ball_progress"], marker="o")
    progress_axis.set(
        xlabel="Training episode",
        ylabel="Mean net ball y-displacement",
    )
    progress_axis.grid(alpha=0.3)
    figure.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation.png")
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(f"Saved evaluation history to {history_path}")
    print(f"Saved goal-rate and ball-progress plot to {plot_path}")
    print(f"Best checkpoint: episode {best_episode}, goal_rate={best_goal_rate:.2f} "
          f"(saved to {os.path.join(output_dir, 'best')})")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", default="checkpoints/two_agent_sanity"
    )
    args = parser.parse_args()
    train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        eval_episodes=args.eval_episodes,
        eval_every=args.eval_every,
        seed=args.seed,
        output_dir=args.output_dir,
    )