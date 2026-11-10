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
from env.rocket_env import RocketEnv, N_ACTIONS, FIELD_D, OBS_SLICES, MAX_STEPS

BALL_VIS = OBS_SLICES["ball_vis"].start
NO_MSG = OBS_SLICES["msg_token"].start   # one-hot slot of the "no message" token
from agents.dqn_agent import DQNAgent

# Communication presets. A-D follow the proposal's Table 1; D2/D5/D10 add
# budget points for the saturation curve (RQ1). D*, C and B use the learned sender.
COMM_SETTINGS = {
    "A":   dict(comm_budget=0,    comm_every=1),  # no communication
    "D2":  dict(comm_budget=2,    comm_every=1),
    "D5":  dict(comm_budget=5,    comm_every=1),
    "D10": dict(comm_budget=10,   comm_every=1),
    "D":   dict(comm_budget=20,   comm_every=1),  # fixed budget (proposal D)
    "C":   dict(comm_budget=None, comm_every=5),  # unlimited, every 5 steps
    "B":   dict(comm_budget=None, comm_every=1),  # unlimited, every step
}
# Fixed-schedule senders (no learned message head): each agent broadcasts its
# sighting every k steps, i.e. 80/k messages per 80-step episode, matching the
# learned budgets. These separate the value of the information from the cost
# of learning when to send.
FIXED_SENDERS = {"F2": 40, "F5": 16, "F10": 8, "F20": 4, "Finf": 1}
for _name, _k in FIXED_SENDERS.items():
    COMM_SETTINGS[_name] = dict(comm_budget=None, comm_every=_k)
# Rule senders (no learned message head): send whenever allowed and seeing the
# ball, until the per-episode budget is spent. The hand-coded counterpart of
# the learned event-triggered sender at the same budgets.
RULE_SENDERS = {"R2": 2, "R5": 5, "R10": 10, "R20": 20, "Rinf": None}
for _name, _b in RULE_SENDERS.items():
    COMM_SETTINGS[_name] = dict(comm_budget=_b, comm_every=1)

# The learned message head chooses silent (0) or send (1). A sent message
# carries the sender's ball sighting and position; the token itself carries no
# extra content, and letting the head pick among interchangeable tokens fed the
# receiver an arbitrary, drifting one-hot that slowed its learning.
HEAD_MESSAGES = 2


def sender_kind(setting: str) -> str:
    """'none', 'fixed', 'rule' or 'learned'."""
    if COMM_SETTINGS[setting]["comm_budget"] == 0:
        return "none"
    if setting in FIXED_SENDERS:
        return "fixed"
    if setting in RULE_SENDERS:
        return "rule"
    return "learned"


REWARD_SCALE = 0.1        # task units are +/-100 per goal; train on +/-10
EVAL_SEED_BASE = 10_000   # every eval uses the same episodes (common random numbers)


def make_env(
    setting: str,
    obs_frac: float = 1.0,
    comm_delay: int = 0,
    max_steps: int = MAX_STEPS,
    curriculum: str = "full",
    opponent: bool = False,
    memory: bool = True,
):
    return RocketEnv(
        n_agents=2,
        obs_frac=obs_frac,
        comm_delay=comm_delay,
        max_steps=max_steps,
        curriculum=curriculum,
        opponent=opponent,
        memory=memory,
        **COMM_SETTINGS[setting],
    )


def decision_points(obs_list, can_send):
    """Event-triggered communication: a learned agent decides whether to
    message only when it may send AND currently sees the ball (otherwise the
    message would carry no sighting). At all other steps it stays silent."""
    return [bool(allowed) and o[BALL_VIS] > 0 for o, allowed in zip(obs_list, can_send)]


def choose_messages(agents, obs_list, can_send, sender: str, silence: bool = False):
    """Message tokens for this step (0 = stay silent) and the decision mask
    (which agents made a learned message decision this step).
    sender: 'fixed' (send whenever allowed), 'rule' (send whenever allowed and
    seeing the ball) or 'learned' (the message head decides at those points)."""
    if sender in ("none", "fixed"):
        return [0 if silence or not allowed or sender == "none" else 1 for allowed in can_send], [False] * len(agents)
    decide = decision_points(obs_list, can_send)
    if sender == "rule":
        return [0 if silence or not d else 1 for d in decide], [False] * len(agents)
    if sender != "learned":
        raise ValueError(f"unknown sender kind {sender!r}")
    messages = [
        agent.select_message(o) if d and not silence else 0
        for agent, o, d in zip(agents, obs_list, decide)
    ]
    return messages, decide


def mask_message(obs):
    """The receiver's observation as if it held no message (the env's "no
    message" encoding): the counterfactual for the value of information."""
    o = obs.copy()
    for key in ("msg_token", "msg_ball", "msg_ball_vis", "msg_sender"):
        o[OBS_SLICES[key]] = 0.0
    o[NO_MSG] = 1.0
    o[OBS_SLICES["msg_age"]] = 1.0
    return o


def message_value(receiver, obs) -> float:
    """Value of the message the receiver currently holds, judged by the
    receiver's own Q-function: Q(o, a*) - Q(o, a_cf), where a* is its greedy
    action with the message and a_cf the action it would take without it.
    Zero when the message does not change its decision (or none is held)."""
    if obs[NO_MSG] > 0:
        return 0.0
    q = receiver.q_net(obs)
    a_cf = int(np.argmax(receiver.q_net(mask_message(obs))))
    return float(q.max() - q[a_cf])


def evaluate(agents, env, n_eval: int = 50, seed: int = EVAL_SEED_BASE, silence: bool = False,
             sender: str = "learned"):
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
                messages, _ = choose_messages(agents, obs_list, info["can_send"], sender, silence)
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
    n_episodes: int = 10_000,
    obs_frac: float = 1.0,
    comm_delay: int = 0,
    max_steps: int = MAX_STEPS,
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
    msg_credit: str = "voi",
    msg_cost: float = 0.05,
    msg_handover: float = 0.5,
    memory: bool = True,
    verbose: bool = True,
):
    """msg_credit: how the learned message head is rewarded.
      "voi"  - value of information: each step the teammate holds my message,
               I am credited with how much it improves the teammate's decision
               (message_value, in the teammate's Q units), minus msg_cost per
               message sent. Direct, low-variance credit for *when* to send.
      "team" - the shared team reward (the original IQL-Comm credit).
    msg_handover: rule-to-learned handover for the learned sender. At each
      decision point during training, the send-on-sight rule decides with
      probability p, which falls linearly from 1 to 0 over the first
      `msg_handover` of training; otherwise the head decides. The head learns
      off-policy from every decision. Without it, sender and receiver co-adapt:
      the receiver learns against a send pattern that keeps shifting, and runs
      were unstable (some seeds collapsed). Evaluation always uses the head."""
    if msg_credit not in ("voi", "team"):
        raise ValueError("msg_credit must be 'voi' or 'team'")
    if n_episodes <= 0 or max_steps <= 0:
        raise ValueError("n_episodes and max_steps must be positive")
    if log_every <= 0 or eval_every <= 0 or n_eval_episodes <= 0 or train_every <= 0:
        raise ValueError("log_every, eval_every, n_eval_episodes, train_every must be positive")

    np.random.seed(seed)
    random.seed(seed)

    comm_cfg = COMM_SETTINGS[setting]
    sender = sender_kind(setting)
    uses_comm = sender != "none"
    learns_msgs = sender == "learned"
    # No keeper by default: with assist-based scoring, the keeper made the task
    # unlearnable for independent DQN within 4000 episodes (0-5% goals).
    env = make_env(setting, obs_frac, comm_delay, max_steps, opponent=opponent, memory=memory)
    eval_env = make_env(setting, obs_frac, comm_delay, max_steps, opponent=opponent, memory=memory)
    obs_dim = env.observation_space.shape[0]

    agents = [
        DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, n_messages=HEAD_MESSAGES, agent_id=i)
        for i in range(2)
    ]
    explore_episodes = max(1, int(explore_frac * n_episodes))

    history = {
        "setting": setting,
        "obs_frac": obs_frac,
        "comm_delay": comm_delay,
        "comm_budget": comm_cfg["comm_budget"],
        "comm_every": comm_cfg["comm_every"],
        "sender": sender,
        "fixed_sender": sender == "fixed",
        "event_triggered": sender in ("rule", "learned"),
        "seed": seed,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "assist_curriculum": assist_curriculum,
        "opponent": opponent,
        "memory": memory,
        "msg_credit": msg_credit if learns_msgs else None,
        "msg_cost": msg_cost if learns_msgs and msg_credit == "voi" else None,
        "msg_handover": msg_handover if learns_msgs else None,
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
        p_rule = 0.0
        if learns_msgs and msg_handover > 0:
            p_rule = max(0.0, 1.0 - (ep - 1) / (msg_handover * n_episodes))
        while True:
            actions = [agent.select_action(o) for agent, o in zip(agents, obs_list)]
            messages = [0, 0]
            if uses_comm:
                messages, decided = choose_messages(agents, obs_list, info["can_send"], sender)
                if p_rule > 0:
                    messages = [1 if d and random.random() < p_rule else m
                                for m, d in zip(messages, decided)]
                for i, agent in enumerate(agents):
                    if decided[i]:
                        agent.message_decision(obs_list[i], messages[i])

            next_obs_list, reward, terminated, truncated, info = env.step(actions, messages)
            ep_return += info["task_reward"]
            r = reward * REWARD_SCALE
            total_steps += 1

            if learns_msgs:
                if msg_credit == "voi":
                    # Agent i is credited with the value of the message agent j holds
                    # (with 2 agents, j's message always comes from i), minus the
                    # cost of sending. A message is sent only at a decision point,
                    # so messages[i] != 0 means agent i sent one this step.
                    msg_r = [message_value(agents[1 - i], next_obs_list[1 - i])
                             - (msg_cost if messages[i] else 0.0) for i in range(2)]
                else:
                    msg_r = [r, r]

            for i, agent in enumerate(agents):
                agent.store(obs_list[i], actions[i], r, next_obs_list[i], terminated, truncated)
                if learns_msgs:
                    agent.message_reward(msg_r[i])
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
            metrics = evaluate(agents, eval_env, n_eval=n_eval_episodes, sender=sender)
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
    history["final_eval_last"] = evaluate(agents, eval_env, n_eval=n_final_eval, seed=final_seed, sender=sender)
    _restore(agents, best[2])
    history["best_episode"] = best[1]
    history["final_eval"] = evaluate(agents, eval_env, n_eval=n_final_eval, seed=final_seed, sender=sender)
    if uses_comm:
        history["final_eval_silenced"] = evaluate(
            agents, eval_env, n_eval=n_final_eval, seed=final_seed, silence=True, sender=sender
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
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--obs_frac", type=float, default=1.0)
    parser.add_argument("--comm_delay", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
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
    parser.add_argument("--msg_credit", default="voi", choices=["voi", "team"])
    parser.add_argument("--msg_cost", type=float, default=0.05)
    parser.add_argument("--msg_handover", type=float, default=0.5)
    parser.add_argument("--no_memory", action="store_true", help="ablation: blank the memory inputs")
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
        msg_credit=args.msg_credit,
        msg_cost=args.msg_cost,
        msg_handover=args.msg_handover,
        memory=not args.no_memory,
    )
