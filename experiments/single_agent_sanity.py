"""Train and plot a greedy single-agent goal-rate sanity check."""

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


def evaluate(agent: DQNAgent, env: RocketEnv, episodes: int, seed: int) -> dict:
    previous_epsilon = agent.eps
    agent.eps = 0.0
    goals_per_episode = []
    try:
        for episode in range(episodes):
            observations, _ = env.reset(seed=seed + episode)
            for _ in range(env.max_steps):
                action = agent.select_action(observations[0])
                observations, _, terminated, truncated, info = env.step([action])
                if terminated or truncated:
                    break
            goals_per_episode.append(info["score_team"])
    finally:
        agent.eps = previous_epsilon

    return {
        "mean_goals": float(np.mean(goals_per_episode)),
        "goal_rate": float(np.mean(np.asarray(goals_per_episode) > 0)),
    }


def train(
    episodes: int = 1000,
    max_steps: int = 200,
    eval_episodes: int = 20,
    eval_every: int = 100,
    seed: int = 42,
    output_dir: str = "checkpoints/single_agent_sanity",
) -> dict:
    if episodes <= 0 or max_steps <= 0 or eval_episodes <= 0 or eval_every <= 0:
        raise ValueError("episode counts, max_steps, and eval_every must be positive")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    env = RocketEnv(
        n_agents=1,
        curriculum="easy",
        opponent=False,
        comm_budget=0,
        max_steps=max_steps,
    )
    eval_env = RocketEnv(
        n_agents=1,
        curriculum="easy",
        opponent=False,
        comm_budget=0,
        max_steps=max_steps,
    )
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=N_ACTIONS,
        n_messages=N_MESSAGES,
        hidden=128,
        lr=1e-3,
        gamma=SHAPING_GAMMA,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=max(1, episodes * max_steps // 4),
        batch_size=64,
        target_update=1500,  # was 300 — hard sync every ~1.5 episodes was too fast
    )

    history = {
        "eval_episode": [],
        "eval_mean_goals": [],
        "eval_goal_rate": [],
        "epsilon": [],
        "goal_buffer_size": [],
    }
    started = time.time()
    for episode in range(1, episodes + 1):
        observations, _ = env.reset(seed=seed if episode == 1 else None)
        for _ in range(max_steps):
            action = agent.select_action(observations[0])
            next_observations, reward, terminated, truncated, info = env.step([action])
            done = terminated or truncated
            agent.store(
                observations[0],
                action,
                reward,
                next_observations[0],
                done,
                # was: abs(info["task_reward"]) >= 100.0
                # that also matched large negative timeout/failure penalties,
                # polluting the goal buffer with non-scoring transitions.
                goal_transition=info["score_team"] > 0,
            )
            agent.update()
            observations = next_observations
            if done:
                break

        if episode % eval_every == 0 or episode == episodes:
            metrics = evaluate(agent, eval_env, eval_episodes, seed + episode * eval_episodes)
            history["eval_episode"].append(episode)
            history["eval_mean_goals"].append(metrics["mean_goals"])
            history["eval_goal_rate"].append(metrics["goal_rate"])
            history["epsilon"].append(agent.eps)
            history["goal_buffer_size"].append(len(agent.goal_buffer))
            print(
                f"Episode {episode:4d}/{episodes} | "
                f"greedy goals/episode={metrics['mean_goals']:.2f} | "
                f"goal rate={metrics['goal_rate']:.0%} | "
                f"epsilon={agent.eps:.3f} | "
                f"goal_buffer={len(agent.goal_buffer)} | "
                f"elapsed={time.time() - started:.0f}s"
            )

    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as history_file:
        json.dump(history, history_file, indent=2)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["eval_episode"], history["eval_goal_rate"], marker="o")
    axis.set(
        title="Single-agent greedy evaluation goal rate",
        xlabel="Training episode",
        ylabel="Evaluation episodes scoring a goal",
        ylim=(0, 1),
    )
    axis.grid(alpha=0.3)
    figure.tight_layout()
    plot_path = os.path.join(output_dir, "goal_rate.png")
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(f"Saved evaluation history to {history_path}")
    print(f"Saved goal-rate plot to {plot_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", default="checkpoints/single_agent_sanity"
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