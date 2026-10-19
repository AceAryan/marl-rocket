"""
Training loop for multi-agent DQN with configurable communication.

Usage:
    python -m experiments.train --setting A          # no comm
    python -m experiments.train --setting B1         # tight budget
    python -m experiments.train --setting C          # every 5 steps
    python -m experiments.train --setting D          # unlimited comm
    python -m experiments.train --episodes 2000 --render
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import random
import time
import numpy as np
import torch
from env.rocket_env import RocketEnv, N_ACTIONS, N_MESSAGES, SHAPING_GAMMA
from agents.dqn_agent import DQNAgent

# Communication presets from the project design doc
COMM_SETTINGS = {
    "A":  dict(comm_budget=0,    comm_delay=0, comm_every=1),  # none
    "B1": dict(comm_budget=5,    comm_delay=0, comm_every=1),  # very tight
    "B2": dict(comm_budget=10,   comm_delay=0, comm_every=1),  # tight
    "B3": dict(comm_budget=20,   comm_delay=0, comm_every=1),  # moderate
    "B4": dict(comm_budget=50,   comm_delay=0, comm_every=1),  # loose
    "B5": dict(comm_budget=100,  comm_delay=0, comm_every=1),  # very loose
    "C":  dict(comm_budget=None, comm_delay=0, comm_every=5),  # periodic
    "D":  dict(comm_budget=None, comm_delay=0, comm_every=1),  # unlimited
}


def make_env(
    comm_setting: dict,
    obs_frac: float = 1.0,
    max_steps: int = 500,
    curriculum: str = "full",
):
    return RocketEnv(
        n_agents=2,
        obs_frac=obs_frac,
        max_steps=max_steps,
        curriculum=curriculum,
        **comm_setting,
    )


def evaluate(agents, env, n_eval: int = 20, seed: int = 99):
    """Evaluate greedy policies without storing transitions or updating agents."""
    saved_eps = [agent.eps for agent in agents]
    saved_random_state = random.getstate()
    for agent in agents:
        agent.eps = 0.0

    rewards = []
    shaped_rewards = []
    team_goals = []
    opponent_goals = []
    try:
        for episode in range(n_eval):
            obs_list, info = env.reset(seed=seed + episode)
            episode_reward = 0.0
            shaped_episode_reward = 0.0
            for _ in range(env.max_steps):
                actions = [agents[i].select_action(obs_list[i]) for i in range(len(agents))]
                message_due = (info.get("step", 0) + 1) % env.comm_every == 0
                messages = [
                    agents[i].select_message(
                        obs_list[i],
                        budget_exhausted=(
                            not message_due
                            or (
                                env.comm_budget is not None
                                and info["msgs_sent"][i] >= env.comm_budget
                            )
                        ),
                    )
                    for i in range(len(agents))
                ]
                obs_list, reward, terminated, truncated, info = env.step(actions, messages)
                episode_reward += info["task_reward"]
                shaped_episode_reward += reward
                if terminated or truncated:
                    break

            rewards.append(episode_reward)
            shaped_rewards.append(shaped_episode_reward)
            team_goals.append(info["score_team"])
            opponent_goals.append(info["score_opp"])
    finally:
        for agent, epsilon in zip(agents, saved_eps):
            agent.eps = epsilon
        random.setstate(saved_random_state)

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "shaped_reward_mean": float(np.mean(shaped_rewards)),
        "score_team_mean": float(np.mean(team_goals)),
        "score_opp_mean": float(np.mean(opponent_goals)),
    }


def train(
    setting: str = "D",
    n_episodes: int = 1000,
    obs_frac: float = 1.0,
    max_steps: int = 200,
    eval_max_steps: int = 500,
    curriculum_episodes: int = 500,
    save_dir: str = "checkpoints",
    render: bool = False,
    seed: int = 42,
    log_every: int = 50,
    eval_every: int = 100,
    n_eval_episodes: int = 20,
):
    if n_episodes <= 0 or max_steps <= 0 or eval_max_steps <= 0:
        raise ValueError("n_episodes, max_steps, and eval_max_steps must be positive")
    if curriculum_episodes < 0:
        raise ValueError("curriculum_episodes cannot be negative")
    if log_every <= 0 or eval_every <= 0 or n_eval_episodes <= 0:
        raise ValueError("log_every, eval_every, and n_eval_episodes must be positive")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    comm_cfg = COMM_SETTINGS[setting]
    curriculum = "easy" if curriculum_episodes > 0 else "full"
    env = make_env(
        comm_cfg,
        obs_frac=obs_frac,
        max_steps=max_steps,
        curriculum=curriculum,
    )
    eval_env = make_env(
        comm_cfg,
        obs_frac=obs_frac,
        max_steps=eval_max_steps,
        curriculum="full",
    )

    obs_dim = env.observation_space.shape[0]

    agents = [
        DQNAgent(
            obs_dim=obs_dim,
            n_actions=N_ACTIONS,
            n_messages=N_MESSAGES,
            hidden=128,
            lr=1e-3,
            gamma=SHAPING_GAMMA,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=max(1, n_episodes * max_steps // 4),
            batch_size=64,
            target_update=300,
            agent_id=i,
        )
        for i in range(2)
    ]

    history = {
        "episode_reward": [],
        "shaped_episode_reward": [],
        "score_team": [],
        "score_opp": [],
        "msgs_sent": [],
        "eval_episode": [],
        "eval_reward_mean": [],
        "eval_reward_std": [],
        "eval_shaped_reward_mean": [],
        "eval_score_team_mean": [],
        "eval_score_opp_mean": [],
        "training_curriculum": [],
        "setting": setting,
        "obs_frac": obs_frac,
        "comm_budget": comm_cfg["comm_budget"],
        "training_max_steps": max_steps,
        "eval_max_steps": eval_max_steps,
        "curriculum_episodes": curriculum_episodes,
    }

    t0 = time.time()
    for ep in range(1, n_episodes + 1):
        if ep == curriculum_episodes + 1 and curriculum == "easy":
            curriculum = "full"
            env = make_env(
                comm_cfg,
                obs_frac=obs_frac,
                max_steps=max_steps,
                curriculum=curriculum,
            )
            obs_list, info = env.reset(seed=seed + ep)
        elif ep == 1:
            obs_list, info = env.reset(seed=seed)
        else:
            obs_list, info = env.reset()
        history["training_curriculum"].append(curriculum)
        ep_reward = 0.0
        ep_shaped_reward = 0.0
        for step in range(max_steps):
            # Select actions + messages
            actions = [agents[i].select_action(obs_list[i]) for i in range(2)]
            message_due = (info.get("step", 0) + 1) % comm_cfg["comm_every"] == 0
            had_message_choice = [
                message_due
                and (
                    comm_cfg["comm_budget"] is None
                    or info["msgs_sent"][i] < comm_cfg["comm_budget"]
                )
                for i in range(2)
            ]
            messages = [
                agents[i].select_message(
                    obs_list[i],
                    budget_exhausted=not had_message_choice[i],
                )
                for i in range(2)
            ]

            # Step environment
            next_obs_list, reward, terminated, truncated, info = env.step(actions, messages)
            done = terminated or truncated
            ep_reward += info["task_reward"]
            ep_shaped_reward += reward

            # Store transitions
            for i in range(2):
                task_reward = info["task_reward"]
                agents[i].store(
                    obs_list[i],
                    actions[i],
                    reward,
                    next_obs_list[i],
                    done,
                    goal_transition=abs(task_reward) >= 100.0,
                )
                agents[i].update()
                if had_message_choice[i]:
                    agents[i].store_message(
                        obs_list[i], messages[i], reward, next_obs_list[i], done
                    )
                    agents[i].update_message_head()

            obs_list = next_obs_list

            if render:
                env.render()

            if done:
                break

        history["episode_reward"].append(ep_reward)
        history["shaped_episode_reward"].append(ep_shaped_reward)
        history["score_team"].append(info["score_team"])
        history["score_opp"].append(info["score_opp"])
        history["msgs_sent"].append(sum(info["msgs_sent"]))

        if ep % log_every == 0:
            recent = history["episode_reward"][-log_every:]
            avg_r  = np.mean(recent)
            avg_g  = np.mean(history["score_team"][-log_every:])
            avg_ag = np.mean(history["score_opp"][-log_every:])
            eps    = agents[0].eps
            elapsed = time.time() - t0
            print(
                f"[{setting}] Ep {ep:5d}/{n_episodes} | "
                f"{curriculum} | "
                f"AvgR={avg_r:7.1f} | Goals {avg_g:.2f}-{avg_ag:.2f} | "
                f"eps={eps:.3f} | {elapsed:.0f}s"
            )

        if ep % eval_every == 0 or ep == n_episodes:
            metrics = evaluate(
                agents,
                eval_env,
                n_eval=n_eval_episodes,
                seed=seed + ep * n_eval_episodes,
            )
            history["eval_episode"].append(ep)
            history["eval_reward_mean"].append(metrics["reward_mean"])
            history["eval_reward_std"].append(metrics["reward_std"])
            history["eval_shaped_reward_mean"].append(metrics["shaped_reward_mean"])
            history["eval_score_team_mean"].append(metrics["score_team_mean"])
            history["eval_score_opp_mean"].append(metrics["score_opp_mean"])
            print(
                f"[{setting}] Eval Ep {ep:5d} | "
                f"Reward={metrics['reward_mean']:7.2f} +/- {metrics['reward_std']:.2f} | "
                f"Goals {metrics['score_team_mean']:.2f}-{metrics['score_opp_mean']:.2f}"
            )

    # Save
    ckpt_path = os.path.join(save_dir, f"obs{obs_frac}", f"setting_{setting}")
    os.makedirs(ckpt_path, exist_ok=True)
    for a in agents:
        a.save(ckpt_path)
    def _to_py(obj):
        if isinstance(obj, (np.floating, np.integer)): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _to_py(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_to_py(v) for v in obj]
        return obj
    with open(os.path.join(ckpt_path, "history.json"), "w") as f:
        json.dump(_to_py(history), f)

    print(f"\nSaved to {ckpt_path}")
    return history


# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting",  default="D", choices=list(COMM_SETTINGS))
    parser.add_argument("--episodes", type=int,   default=1000)
    parser.add_argument("--obs_frac", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--eval_max_steps", type=int, default=500)
    parser.add_argument("--curriculum_episodes", type=int, default=500)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--log_every",type=int,   default=50)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    args = parser.parse_args()

    train(
        setting=args.setting,
        n_episodes=args.episodes,
        obs_frac=args.obs_frac,
        max_steps=args.max_steps,
        eval_max_steps=args.eval_max_steps,
        curriculum_episodes=args.curriculum_episodes,
        render=args.render,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        n_eval_episodes=args.n_eval_episodes,
    )
