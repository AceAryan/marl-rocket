"""
Training loop for multi-agent DQN with configurable communication.

Usage:
    python -m experiments.train --setting A          # no comm
    python -m experiments.train --setting B          # unlimited comm
    python -m experiments.train --setting C          # every 5 steps
    python -m experiments.train --setting D          # budget=20/ep
    python -m experiments.train --episodes 2000 --render
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import time
import numpy as np
from env.rocket_env import RocketEnv, N_ACTIONS, N_MESSAGES
from agents.dqn_agent import DQNAgent

# Communication presets from the project design doc
COMM_SETTINGS = {
    "A": dict(comm_budget=0,    comm_delay=0, comm_every=1),   # None
    "B": dict(comm_budget=None, comm_delay=0, comm_every=1),   # Every step
    "C": dict(comm_budget=None, comm_delay=0, comm_every=5),   # Every 5 steps
    "D": dict(comm_budget=20,   comm_delay=0, comm_every=1),   # Budget 20/ep
}


def make_env(comm_setting: dict, obs_frac: float = 1.0, max_steps: int = 500):
    return RocketEnv(
        n_agents=2,
        obs_frac=obs_frac,
        max_steps=max_steps,
        **comm_setting,
    )


def train(
    setting: str = "B",
    n_episodes: int = 1000,
    obs_frac: float = 1.0,
    max_steps: int = 500,
    save_dir: str = "checkpoints",
    render: bool = False,
    seed: int = 42,
    log_every: int = 50,
):
    np.random.seed(seed)

    comm_cfg = COMM_SETTINGS[setting]
    env = make_env(comm_cfg, obs_frac=obs_frac, max_steps=max_steps)
    env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]

    agents = [
        DQNAgent(
            obs_dim=obs_dim,
            n_actions=N_ACTIONS,
            n_messages=N_MESSAGES,
            hidden=128,
            lr=1e-3,
            gamma=0.99,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay=n_episodes * max_steps // 4,
            batch_size=64,
            target_update=300,
            agent_id=i,
        )
        for i in range(2)
    ]

    history = {
        "episode_reward": [],
        "score_team": [],
        "score_opp": [],
        "msgs_sent": [],
        "setting": setting,
        "obs_frac": obs_frac,
    }

    t0 = time.time()
    for ep in range(1, n_episodes + 1):
        obs_list, info = env.reset()
        ep_reward = 0.0
        prev_obs = obs_list

        for step in range(max_steps):
            # Select actions + messages
            actions = [agents[i].select_action(obs_list[i]) for i in range(2)]
            messages = [
                agents[i].select_message(
                    obs_list[i],
                    budget_exhausted=(
                        comm_cfg["comm_budget"] is not None
                        and info.get("msgs_sent", [0, 0])[i] >= comm_cfg["comm_budget"]
                    ),
                )
                for i in range(2)
            ]

            # Step environment
            next_obs_list, reward, terminated, truncated, info = env.step(actions, messages)
            done = terminated or truncated
            ep_reward += reward

            # Store transitions
            for i in range(2):
                agents[i].store(obs_list[i], actions[i], reward, next_obs_list[i], done)
                agents[i].update()

            obs_list = next_obs_list

            if render:
                env.render()

            if done:
                break

        history["episode_reward"].append(ep_reward)
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
                f"AvgR={avg_r:7.1f} | Goals {avg_g:.2f}-{avg_ag:.2f} | "
                f"eps={eps:.3f} | {elapsed:.0f}s"
            )

    # Save
    ckpt_path = os.path.join(save_dir, f"setting_{setting}_obs{obs_frac}")
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
    parser.add_argument("--setting",  default="B", choices=["A","B","C","D"])
    parser.add_argument("--episodes", type=int,   default=1000)
    parser.add_argument("--obs_frac", type=float, default=1.0)
    parser.add_argument("--max_steps",type=int,   default=500)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--log_every",type=int,   default=50)
    args = parser.parse_args()

    train(
        setting=args.setting,
        n_episodes=args.episodes,
        obs_frac=args.obs_frac,
        max_steps=args.max_steps,
        render=args.render,
        seed=args.seed,
        log_every=args.log_every,
    )
