"""
Training loop for independent DQN (IQL-Comm) with configurable communication.

Usage:
    python -m experiments.train --setting A          # no comm
    python -m experiments.train --setting B          # unlimited, every step
    python -m experiments.train --setting C          # unlimited, every 5 steps
    python -m experiments.train --setting D          # budget 20 / episode
    python -m experiments.train --setting B --obs_frac 0.25 --comm_delay 2
"""

import sys, os
# Tiny matmuls: multithreaded BLAS only adds overhead (and oversubscribes parallel sweeps)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import random
import time
import numpy as np
from env.rocket_env import RocketEnv, N_ACTIONS, N_MESSAGES, FIELD_D, OBS_SLICES

BALL_VIS = OBS_SLICES["ball_vis"].start
from agents.dqn_agent import DQNAgent

# Communication presets. A-D follow the proposal's Table 1; D5/D10/D50 add
# budget points for the saturation curve (RQ1).
COMM_SETTINGS = {
    "A":   dict(comm_budget=0,    comm_every=1),  # no communication
    "D5":  dict(comm_budget=5,    comm_every=1),
    "D10": dict(comm_budget=10,   comm_every=1),
    "D":   dict(comm_budget=20,   comm_every=1),  # fixed budget (proposal D)
    "D50": dict(comm_budget=50,   comm_every=1),
    "C":   dict(comm_budget=None, comm_every=5),  # unlimited, every 5 steps
    "B":   dict(comm_budget=None, comm_every=1),  # unlimited, every step
}
# Fixed-schedule senders (no learned message head): each agent broadcasts its
# sighting every k steps, i.e. 200/k messages per 200-step episode. These
# separate the value of the information from the cost of learning when to send.
FIXED_SENDERS = {"F5": 40, "F10": 20, "F20": 10, "F50": 4, "Finf": 1}
for _name, _k in FIXED_SENDERS.items():
    COMM_SETTINGS[_name] = dict(comm_budget=None, comm_every=_k)

REWARD_SCALE = 0.1        # task units are +/-100 per goal; train on +/-10
EVAL_SEED_BASE = 10_000   # every eval uses the same episodes (common random numbers)


def make_env(
    setting: str,
    obs_frac: float = 1.0,
    comm_delay: int = 0,
    max_steps: int = 200,
    curriculum: str = "full",
    opponent: bool = False,
):
    return RocketEnv(
        n_agents=2,
        obs_frac=obs_frac,
        comm_delay=comm_delay,
        max_steps=max_steps,
        curriculum=curriculum,
        opponent=opponent,
        **COMM_SETTINGS[setting],
    )


def decision_points(obs_list, can_send):
    """Event-triggered communication: a learned agent decides whether to
    message only when it may send AND currently sees the ball (otherwise the
    message would carry no sighting). At all other steps it stays silent."""
    return [bool(allowed) and o[BALL_VIS] > 0 for o, allowed in zip(obs_list, can_send)]


def choose_messages(agents, obs_list, can_send, fixed: bool, silence: bool = False):
    """Message tokens for this step (0 = stay silent) and the decision mask
    (which agents made a learned message decision this step)."""
    if fixed:
        return [0 if silence or not allowed else 1 for allowed in can_send], [False] * len(agents)
    decide = decision_points(obs_list, can_send)
    messages = [
        agent.select_message(o) if d and not silence else 0
        for agent, o, d in zip(agents, obs_list, decide)
    ]
    return messages, decide


def evaluate(agents, env, n_eval: int = 50, seed: int = EVAL_SEED_BASE, silence: bool = False,
             fixed: bool = False):
    """Greedy evaluation without storing transitions or updating agents.

    silence=True blocks every message (coordination ablation: the return drop
    measures how much the learned policy relies on the channel)."""
    saved_eps = [agent.eps for agent in agents]
    saved_random_state = random.getstate()
    for agent in agents:
        agent.eps = 0.0

    returns, team_goals, opp_goals, msgs, separation, lengths = [], [], [], [], [], []
    try:
        for episode in range(n_eval):
            obs_list, info = env.reset(seed=seed + episode)
            ep_return = 0.0
            sep = []
            while True:
                actions = [agent.select_action(o) for agent, o in zip(agents, obs_list)]
                messages, _ = choose_messages(agents, obs_list, info["can_send"], fixed, silence)
                obs_list, _, terminated, truncated, info = env.step(actions, messages)
                ep_return += info["task_reward"]
                d = np.linalg.norm(info["agent_pos"] - info["ball_pos"], axis=1)
                sep.append(abs(d[0] - d[1]) / FIELD_D)
                if terminated or truncated:
                    break
            returns.append(ep_return)
            team_goals.append(info["score_team"])
            opp_goals.append(info["score_opp"])
            msgs.append(sum(info["msgs_sent"]))
            separation.append(np.mean(sep))
            lengths.append(info["step"])
    finally:
        for agent, epsilon in zip(agents, saved_eps):
            agent.eps = epsilon
        random.setstate(saved_random_state)

    team_goals = np.asarray(team_goals, dtype=np.float64)
    opp_goals = np.asarray(opp_goals, dtype=np.float64)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "goal_rate": float(np.mean(team_goals > 0)),
        "concede_rate": float(np.mean(opp_goals > 0)),
        "goal_diff": float(np.mean(team_goals - opp_goals)),
        "msgs_mean": float(np.mean(msgs)),
        "role_separation": float(np.mean(separation)),
        "episode_length": float(np.mean(lengths)),   # lower = faster goals
    }


def _snapshot(agents):
    return [([p.copy() for p in a.q_net.params], [p.copy() for p in a.msg_net.params]) for a in agents]


def _restore(agents, snap):
    for a, (q, m) in zip(agents, snap):
        a.q_net.params = [p.copy() for p in q]
        a.msg_net.params = [p.copy() for p in m]


def train(
    setting: str = "B",
    n_episodes: int = 4000,
    obs_frac: float = 1.0,
    comm_delay: int = 0,
    max_steps: int = 200,
    save_dir: str = "checkpoints/manual",
    render: bool = False,
    seed: int = 42,
    log_every: int = 100,
    eval_every: int = 250,
    n_eval_episodes: int = 100,
    n_final_eval: int = 200,
    explore_frac: float = 0.1,
    train_every: int = 2,
    assist_curriculum: float = 0.0,
    opponent: bool = False,
    verbose: bool = True,
):
    if n_episodes <= 0 or max_steps <= 0:
        raise ValueError("n_episodes and max_steps must be positive")
    if log_every <= 0 or eval_every <= 0 or n_eval_episodes <= 0 or train_every <= 0:
        raise ValueError("log_every, eval_every, n_eval_episodes, train_every must be positive")

    np.random.seed(seed)
    random.seed(seed)

    comm_cfg = COMM_SETTINGS[setting]
    uses_comm = comm_cfg["comm_budget"] != 0
    fixed = setting in FIXED_SENDERS
    learns_msgs = uses_comm and not fixed
    # No keeper by default: with assist-based scoring, the keeper made the task
    # unlearnable for independent DQN within 4000 episodes (0-5% goals).
    env = make_env(setting, obs_frac, comm_delay, max_steps, opponent=opponent)
    eval_env = make_env(setting, obs_frac, comm_delay, max_steps, opponent=opponent)
    obs_dim = env.observation_space.shape[0]

    agents = [
        DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, n_messages=N_MESSAGES, agent_id=i)
        for i in range(2)
    ]
    explore_episodes = max(1, int(explore_frac * n_episodes))

    history = {
        "setting": setting,
        "obs_frac": obs_frac,
        "comm_delay": comm_delay,
        "comm_budget": comm_cfg["comm_budget"],
        "comm_every": comm_cfg["comm_every"],
        "fixed_sender": fixed,
        "event_triggered": learns_msgs,
        "seed": seed,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "assist_curriculum": assist_curriculum,
        "opponent": opponent,
        "episode_return": [],
        "score_team": [],
        "score_opp": [],
        "msgs_sent": [],
        "eval_episode": [],
        "eval": [],
    }

    run_dir = save_dir
    best = None   # (validation score, episode, network snapshot)
    t0 = time.time()
    total_steps = 0
    for ep in range(1, n_episodes + 1):
        eps = max(0.05, 1.0 - (1.0 - 0.05) * (ep - 1) / explore_episodes)
        for agent in agents:
            agent.eps = eps
        # Assist curriculum: the share of episodes requiring assisted goals ramps
        # 0 -> 1 over the first `assist_curriculum` of training (learn to score
        # first, then to pass before shooting). Evaluation always requires them.
        ramp = 1.0 if assist_curriculum <= 0 else min(1.0, (ep - 1) / (assist_curriculum * n_episodes))
        env.assist_required = random.random() < ramp
        obs_list, info = env.reset(seed=seed * 1_000_003 + ep)
        ep_return = 0.0
        while True:
            actions = [agent.select_action(o) for agent, o in zip(agents, obs_list)]
            messages = [0, 0]
            if uses_comm:
                messages, decided = choose_messages(agents, obs_list, info["can_send"], fixed)
                for i, agent in enumerate(agents):
                    if decided[i]:
                        agent.message_decision(obs_list[i], messages[i])

            next_obs_list, reward, terminated, truncated, info = env.step(actions, messages)
            ep_return += info["task_reward"]
            r = reward * REWARD_SCALE
            total_steps += 1

            for i, agent in enumerate(agents):
                agent.store(obs_list[i], actions[i], r, next_obs_list[i], terminated, truncated)
                if learns_msgs:
                    agent.message_reward(r)
                if total_steps % train_every == 0:
                    agent.update()
                    if learns_msgs:
                        agent.update_message_head()

            obs_list = next_obs_list
            if render:
                env.render()
            if terminated or truncated:
                break

        for i, agent in enumerate(agents):
            agent.end_episode(obs_list[i])

        history["episode_return"].append(ep_return)
        history["score_team"].append(info["score_team"])
        history["score_opp"].append(info["score_opp"])
        history["msgs_sent"].append(sum(info["msgs_sent"]))

        if verbose and ep % log_every == 0:
            recent = slice(-log_every, None)
            print(
                f"[{setting} obs={obs_frac} d={comm_delay} s={seed}] Ep {ep:5d}/{n_episodes} | "
                f"R={np.mean(history['episode_return'][recent]):7.1f} | "
                f"Goals {np.mean(history['score_team'][recent]):.2f}-"
                f"{np.mean(history['score_opp'][recent]):.2f} | "
                f"msgs={np.mean(history['msgs_sent'][recent]):5.1f} | "
                f"eps={eps:.3f} | {time.time() - t0:.0f}s",
                flush=True,
            )

        if ep % eval_every == 0 or ep == n_episodes:
            metrics = evaluate(agents, eval_env, n_eval=n_eval_episodes, fixed=fixed)
            history["eval_episode"].append(ep)
            history["eval"].append(metrics)
            # Model selection on the validation episodes: keep the best snapshot
            score = (metrics["goal_rate"], metrics["return_mean"])
            if best is None or score > best[0]:
                best = (score, ep, _snapshot(agents))
            if verbose:
                print(
                    f"[{setting} obs={obs_frac} d={comm_delay} s={seed}] Eval {ep:5d} | "
                    f"R={metrics['return_mean']:7.1f} | goal={metrics['goal_rate']:.2f} "
                    f"concede={metrics['concede_rate']:.2f} | msgs={metrics['msgs_mean']:.1f}",
                    flush=True,
                )

    # Final, larger evaluation on held-out seeds (disjoint from the validation
    # episodes used for model selection). The last policy is recorded too.
    final_seed = EVAL_SEED_BASE + 100_000
    history["final_eval_last"] = evaluate(agents, eval_env, n_eval=n_final_eval, seed=final_seed, fixed=fixed)
    _restore(agents, best[2])
    history["best_episode"] = best[1]
    history["final_eval"] = evaluate(agents, eval_env, n_eval=n_final_eval, seed=final_seed, fixed=fixed)
    if uses_comm:
        history["final_eval_silenced"] = evaluate(
            agents, eval_env, n_eval=n_final_eval, seed=final_seed, silence=True, fixed=fixed
        )
    history["train_seconds"] = time.time() - t0
    history["total_steps"] = total_steps

    os.makedirs(run_dir, exist_ok=True)
    for agent in agents:
        agent.save(run_dir)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f)

    if verbose:
        fe = history["final_eval"]
        print(
            f"[{setting} obs={obs_frac} d={comm_delay} s={seed}] FINAL ({n_final_eval} eps) | "
            f"R={fe['return_mean']:.1f} | goal={fe['goal_rate']:.2f} concede={fe['concede_rate']:.2f}"
            f" | saved to {run_dir}"
        )
    return history


# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="B", choices=list(COMM_SETTINGS))
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--obs_frac", type=float, default=1.0)
    parser.add_argument("--comm_delay", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    parser.add_argument("--n_final_eval", type=int, default=200)
    parser.add_argument("--explore_frac", type=float, default=0.1)
    parser.add_argument("--train_every", type=int, default=2)
    parser.add_argument("--assist_curriculum", type=float, default=0.0)
    parser.add_argument("--opponent", action="store_true", help="add the scripted keeper")
    args = parser.parse_args()

    train(
        setting=args.setting,
        n_episodes=args.episodes,
        obs_frac=args.obs_frac,
        comm_delay=args.comm_delay,
        max_steps=args.max_steps,
        save_dir=args.save_dir or os.path.join(
            "checkpoints", "manual", f"obs{args.obs_frac}_delay{args.comm_delay}",
            f"setting_{args.setting}", f"seed_{args.seed}",
        ),
        render=args.render,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        n_eval_episodes=args.n_eval_episodes,
        n_final_eval=args.n_final_eval,
        explore_frac=args.explore_frac,
        train_every=args.train_every,
        assist_curriculum=args.assist_curriculum,
        opponent=args.opponent,
    )
