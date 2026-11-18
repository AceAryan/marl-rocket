"""
Training loop for a two-agent DQN team with configurable communication.

Settings (parse_setting):
    A        no communication
    L<B>     learned gate, budget B messages per agent per episode (B = 1, 2, ... or inf)
    R<B>     rule: send on sight when the teammate's copy of my last message has expired
    F<B>     fixed schedule: every ceil(T/B) steps, sighting or not
    B, C, D  proposal aliases: Linf; learned gate, every 5 steps; L20

Usage:
    python -m experiments.train --setting A --obs_frac 0.075
    python -m experiments.train --setting L4 --obs_frac 0.075 --msg_credit rollout
    python -m experiments.train --setting Rinf --obs_frac 0.075 --match
"""

import sys, os
# Tiny matmuls: multithreaded BLAS only adds overhead (and oversubscribes parallel sweeps)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import math
import random
import subprocess
import time
import numpy as np
from env.rocket_env import (RocketEnv, N_ACTIONS, FIELD_D, OBS_SLICES, MAX_STEPS, ASSIST_WINDOW,
                            NO_TOUCH, PAYLOAD, PAYLOAD_DIM)
from agents.dqn_agent import make_agents, unique

BALL_VIS = OBS_SLICES["ball_vis"].start
SEND_AGE = OBS_SLICES["send_age"].start
REWARD_SCALE = 0.1        # task units are +/-100 per goal; train on +/-10
EVAL_SEED_BASE = 10_000   # every validation eval uses the same episodes (common random numbers)
FINAL_SEED_BASE = EVAL_SEED_BASE + 100_000   # held-out episodes for the final evaluation
MATCH_STEPS = 200         # match-mode episode length
ALIASES = {"B": "Linf", "D": "L20"}


def parse_setting(name: str, max_steps: int = MAX_STEPS) -> dict:
    """Sender kind and channel parameters of a setting name."""
    if name == "A":
        return dict(sender="none", comm_budget=0, comm_every=1)
    if name == "C":
        return dict(sender="learned", comm_budget=None, comm_every=5)
    base = ALIASES.get(name, name)
    kind = {"L": "learned", "R": "rule", "F": "fixed"}.get(base[:1])
    rest = base[1:]
    if kind is None or not (rest == "inf" or (rest.isdigit() and int(rest) > 0)):
        raise ValueError(f"unknown setting {name!r}")
    budget = None if rest == "inf" else int(rest)
    if kind == "fixed":
        every = 1 if budget is None else max(1, math.ceil(max_steps / budget))
        return dict(sender="fixed", comm_budget=None, comm_every=every)
    return dict(sender=kind, comm_budget=budget, comm_every=1)


def sender_kind(setting: str) -> str:
    """'none', 'fixed', 'rule' or 'learned'."""
    return parse_setting(setting)["sender"]


def run_label(setting, budget_obs=True, msg_credit="rollout", share=True, lr_decay=False):
    """Name of a configuration: the setting plus any non-default variant."""
    label = setting
    if sender_kind(setting) == "learned" and msg_credit != "rollout":
        label += f"-{msg_credit}"
    if not budget_obs:
        label += "-blind"
    if not share:
        label += "-indep"
    if lr_decay:
        label += "-lrdecay"
    return label


def make_env(setting, obs_frac=1.0, comm_delay=0, max_steps=None, match=False, opponent=False,
             memory=True, budget_obs=True, curriculum="full"):
    max_steps = max_steps or (MATCH_STEPS if match else MAX_STEPS)
    cfg = parse_setting(setting, max_steps)
    return RocketEnv(
        n_agents=2, obs_frac=obs_frac, comm_delay=comm_delay, max_steps=max_steps, curriculum=curriculum,
        opponent=opponent, memory=memory, budget_obs=budget_obs, match=match,
        comm_budget=cfg["comm_budget"], comm_every=cfg["comm_every"],
    )


# ------------------------------------------------------------------ acting

def _shared(objs):
    return len(unique(objs)) == 1


def q_values(agents, obs_lists):
    """Q-values for several worlds at once: returns [world][agent] -> array."""
    if _shared([a.learner for a in agents]):
        X = np.stack([a.x(o) for obs in obs_lists for a, o in zip(agents, obs)])
        Q = agents[0].learner.q_net(X)
        n = len(agents)
        return [Q[w * n:(w + 1) * n] for w in range(len(obs_lists))]
    return [[a.q_values(o) for a, o in zip(agents, obs)] for obs in obs_lists]


def choose_actions(agents, obs_list, greedy=False):
    q = q_values(agents, [obs_list])[0]
    acts = []
    for a, qa in zip(agents, q):
        if not greedy and random.random() < a.eps:
            acts.append(random.randrange(a.n_actions))
        else:
            acts.append(int(np.argmax(qa)))
    return acts


def gate_scores(agents, obs_list, idx):
    if _shared([a.gate for a in agents]):
        X = np.stack([agents[i].x(obs_list[i]) for i in idx])
        return agents[0].gate.score(X).tolist()
    return [agents[i].gate_score(obs_list[i]) for i in idx]


def decision_points(obs_list, can_send):
    """Event-triggered communication: a learned agent decides whether to
    message only when it may send AND currently sees the ball (otherwise the
    message would carry no sighting). At all other steps it stays silent."""
    return [bool(allowed) and o[BALL_VIS] > 0 for o, allowed in zip(obs_list, can_send)]


def choose_messages(agents, obs_list, can_send, sender, greedy=False, p_rule=0.0, msg_cost=0.0,
                    silence=False):
    """Message tokens for this step (0 = silent, 1 = send) and the decision mask
    (which agents made a learned gate decision this step).
    sender: 'none'; 'fixed' (send whenever allowed); 'rule' (send when allowed,
    seeing the ball, and the teammate's copy of my last message has expired);
    'learned' (the gate decides at the event-triggered decision points; with
    probability p_rule the rule decides instead, during the handover)."""
    n = len(agents)
    if silence or sender == "none":
        return [0] * n, [False] * n
    if sender == "fixed":
        return [1 if allowed else 0 for allowed in can_send], [False] * n
    decide = decision_points(obs_list, can_send)
    rule = [1 if d and o[SEND_AGE] >= 1.0 else 0 for d, o in zip(decide, obs_list)]
    if sender == "rule":
        return rule, [False] * n
    if sender != "learned":
        raise ValueError(f"unknown sender kind {sender!r}")
    messages = [0] * n
    idx = [i for i in range(n) if decide[i]]
    if idx:
        scores = gate_scores(agents, obs_list, idx)
        for score, i in zip(scores, idx):
            if p_rule > 0 and random.random() < p_rule:
                messages[i] = rule[i]
            elif not greedy and random.random() < agents[i].eps:
                messages[i] = random.randrange(2)   # silence-biased: half the exploratory picks are silent
            else:
                messages[i] = 1 if score > msg_cost else 0
    return messages, decide


# ------------------------------------------------------------------ gate credit

def rollout_advantage(env, agents, actions, messages, i, sender, msg_cost, horizon, gamma=0.99):
    """Counterfactual value of agent i sending now: clone the game, play this
    step once with agent i silent and once with it sending (everything else
    equal), continue both copies with the current greedy policies (actions and
    later gate decisions, same budgets), and return the difference in
    discounted team return (training reward units) over `horizon` steps."""
    worlds = [env.clone(), env.clone()]
    variants = [list(messages), list(messages)]
    variants[0][i], variants[1][i] = 0, 1
    returns = [0.0, 0.0]
    alive = [True, True]
    obs, info = [None, None], [None, None]
    disc = 1.0
    for h in range(horizon):
        live = [w for w in (0, 1) if alive[w]]
        if not live:
            break
        if h == 0:
            acts = {w: actions for w in live}
            msgs = {w: variants[w] for w in live}
        else:
            q = q_values(agents, [obs[w] for w in live])
            acts = {w: [int(np.argmax(qa)) for qa in qw] for w, qw in zip(live, q)}
            msgs = {w: choose_messages(agents, obs[w], info[w]["can_send"], sender, greedy=True,
                                       msg_cost=msg_cost)[0] for w in live}
        for w in live:
            obs[w], reward, term, trunc, info[w] = worlds[w].step(acts[w], msgs[w])
            returns[w] += disc * reward * REWARD_SCALE
            if term or trunc:
                alive[w] = False
        disc *= gamma
    return returns[1] - returns[0]


# ------------------------------------------------------------------ evaluation

class _ContentRecorder:
    """payload_hook that records every message's content per (episode, agent)."""

    def __init__(self):
        self.sends = {}
        self.episode = 0

    def __call__(self, i, payload, touch, step):
        age = step - touch if touch > NO_TOUCH else None
        self.sends.setdefault((self.episode, i), []).append((payload.copy(), age))
        return payload, touch


class _ContentShuffler:
    """payload_hook that replaces the k-th message of agent i in episode e by
    the k-th message agent i sent in the next evaluation episode that has one
    (the proposal's shuffled-content ablation: same timing, wrong content)."""

    def __init__(self, sends, n_episodes):
        self.sends, self.n = sends, n_episodes
        self.episode = 0
        self.count = {}

    def __call__(self, i, payload, touch, step):
        k = self.count.get(i, 0)
        self.count[i] = k + 1
        for shift in range(1, self.n):
            donor = self.sends.get(((self.episode + shift) % self.n, i), [])
            if len(donor) > k:
                p, age = donor[k]
                return p.copy(), (step - age if age is not None and age <= ASSIST_WINDOW else NO_TOUCH)
        empty = np.zeros(PAYLOAD_DIM, np.float32)
        empty[PAYLOAD["touch"]] = 1.0
        return empty, NO_TOUCH

    def new_episode(self, e):
        self.episode, self.count = e, {}


def evaluate(agents, env, n_eval: int = 50, seed: int = EVAL_SEED_BASE, sender: str = "learned",
             msg_cost: float = 0.05, silence: bool = False, shuffle: bool = False):
    """Greedy evaluation without storing transitions or updating agents.

    silence=True blocks every message. shuffle=True delivers messages at the
    same decision points but with the content of messages from other
    evaluation episodes (the drop measures reliance on what messages say)."""
    saved_eps = [agent.eps for agent in agents]
    saved_random_state = random.getstate()
    saved_hook = env.payload_hook
    for agent in agents:
        agent.eps = 0.0
    hook = None
    if shuffle:
        recorder = _ContentRecorder()
        env.payload_hook = recorder
        for e in range(n_eval):
            recorder.episode = e
            _play(agents, env, seed + e, sender, msg_cost, silence)
        hook = _ContentShuffler(recorder.sends, n_eval)
    env.payload_hook = hook
    rows = []
    try:
        for e in range(n_eval):
            if hook is not None:
                hook.new_episode(e)
            rows.append(_play(agents, env, seed + e, sender, msg_cost, silence))
    finally:
        env.payload_hook = saved_hook
        for agent, epsilon in zip(agents, saved_eps):
            agent.eps = epsilon
        random.setstate(saved_random_state)

    goals = np.array([r["goals"] for r in rows], dtype=np.float64)
    first = [r["first_goal"] for r in rows if r["first_goal"] is not None]
    send_steps = [s for r in rows for s in r["send_steps"]]
    bins = np.linspace(0, env.max_steps, 9)
    hist = np.histogram(send_steps, bins=bins)[0] if send_steps else np.zeros(8)
    return {
        "goals_mean": float(goals.mean()),
        "goal_rate": float(np.mean(goals > 0)),
        "return_mean": float(np.mean([r["return"] for r in rows])),
        "return_std": float(np.std([r["return"] for r in rows])),
        "concede_rate": float(np.mean([r["conceded"] > 0 for r in rows])),
        "goal_diff": float(np.mean([r["goals"] - r["conceded"] for r in rows])),
        "msgs_mean": float(np.mean([r["msgs"] for r in rows])),   # both agents, per episode
        "role_separation": float(np.mean([r["separation"] for r in rows])),
        "episode_length": float(np.mean([r["length"] for r in rows])),
        "time_to_goal": float(np.mean(first)) if first else None,   # first goal, scoring episodes only
        "send_hist": [int(c) for c in hist],                         # sends per eighth of the episode
    }


def _play(agents, env, seed, sender, msg_cost, silence):
    obs_list, info = env.reset(seed=seed)
    ep_return, sep, send_steps = 0.0, [], []
    while True:
        actions = choose_actions(agents, obs_list, greedy=True)
        messages, _ = choose_messages(agents, obs_list, info["can_send"], sender, greedy=True,
                                      msg_cost=msg_cost, silence=silence)
        for i, m in enumerate(messages):
            if m and info["can_send"][i]:
                send_steps.append(env._step_count)
        obs_list, _, terminated, truncated, info = env.step(actions, messages)
        ep_return += info["task_reward"]
        d = np.linalg.norm(info["agent_pos"] - info["ball_pos"], axis=1)
        sep.append(abs(d[0] - d[1]) / FIELD_D)
        if terminated or truncated:
            break
    return {
        "goals": info["score_team"], "conceded": info["score_opp"], "return": ep_return,
        "msgs": sum(info["msgs_sent"]), "separation": float(np.mean(sep)), "length": info["step"],
        "first_goal": info["goal_steps"][0] if info["goal_steps"] else None, "send_steps": send_steps,
    }


# ------------------------------------------------------------------ training

def _snapshot(agents):
    return ([l.q_net.flat.copy() for l in unique([a.learner for a in agents])],
            [g.net.flat.copy() for g in unique([a.gate for a in agents])])


def _restore(agents, snap):
    for l, flat in zip(unique([a.learner for a in agents]), snap[0]):
        l.q_net.flat[:] = flat
    for g, flat in zip(unique([a.gate for a in agents]), snap[1]):
        g.net.flat[:] = flat


_GIT = None


def git_commit():
    global _GIT
    if _GIT is None:
        try:
            out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(__file__)),
                                 capture_output=True, text=True, timeout=10)
            _GIT = out.stdout.strip() or "unknown"
        except Exception:
            _GIT = "unknown"
    return _GIT


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def train(
    setting: str = "L4",
    n_episodes: int = 10_000,
    obs_frac: float = 0.075,
    comm_delay: int = 0,
    max_steps: int = None,
    match: bool = False,
    save_dir: str = "checkpoints/manual",
    render: bool = False,
    seed: int = 42,
    log_every: int = 100,
    eval_every: int = 250,
    n_eval_episodes: int = 100,
    n_final_eval: int = 200,
    explore_frac: float = 0.1,
    train_every: int = 2,
    gate_every: int = 8,
    share: bool = True,
    msg_credit: str = "rollout",
    msg_cost: float = 0.05,
    msg_handover: float = 0.3,
    rollouts_per_episode: int = 2,
    rollout_prob: float = 0.2,
    rollout_horizon: int = 40,
    budget_obs: bool = True,
    memory: bool = True,
    lr_decay: bool = False,
    opponent: bool = False,
    verbose: bool = True,
):
    """Train a two-agent team in one setting and save its history and weights.

    Message gate (learned settings), msg_credit:
      "rollout" - at up to `rollouts_per_episode` decision points per agent per
                  episode (each chosen with probability `rollout_prob`), the game
                  is cloned and played on with the message sent and withheld;
                  the gate regresses the difference in discounted team return
                  (rollout_advantage). Centralised training, team reward only.
      "team"    - the gate regresses Q(o, a) on the Monte-Carlo team return that
                  followed each decision (the proposal's original credit).
    The agent sends when the gate's score exceeds msg_cost. During the first
    `msg_handover` of training the paced rule decides with probability falling
    from 1 to 0, so receivers first learn against a sensible sender."""
    if msg_credit not in ("rollout", "team"):
        raise ValueError("msg_credit must be 'rollout' or 'team'")
    if n_episodes <= 0 or (max_steps is not None and max_steps <= 0):
        raise ValueError("n_episodes and max_steps must be positive")
    if log_every <= 0 or eval_every <= 0 or n_eval_episodes <= 0 or train_every <= 0 or gate_every <= 0:
        raise ValueError("log_every, eval_every, n_eval_episodes, train_every, gate_every must be positive")
    max_steps = max_steps or (MATCH_STEPS if match else MAX_STEPS)
    config = {k: v for k, v in locals().items()
              if k not in ("save_dir", "render", "log_every", "verbose")}

    np.random.seed(seed)
    random.seed(seed)

    cfg = parse_setting(setting, max_steps)
    sender = cfg["sender"]
    uses_comm = sender != "none"
    learns_msgs = sender == "learned"
    env_kw = dict(obs_frac=obs_frac, comm_delay=comm_delay, max_steps=max_steps, match=match,
                  opponent=opponent, memory=memory, budget_obs=budget_obs)
    env = make_env(setting, **env_kw)
    eval_env = make_env(setting, **env_kw)
    obs_dim = env.observation_space.shape[0]

    gate_mode = "advantage" if msg_credit == "rollout" else "q"
    agents = make_agents(obs_dim, N_ACTIONS, n_agents=2, share=share, gate_mode=gate_mode)
    learners = unique([a.learner for a in agents])
    gates = unique([a.gate for a in agents])
    lr0 = learners[0].opt.lr
    explore_episodes = max(1, int(explore_frac * n_episodes))
    gamma = agents[0].gamma

    history = {
        "setting": setting,
        "label": run_label(setting, budget_obs, msg_credit, share, lr_decay),
        "sender": sender,
        "comm_budget": cfg["comm_budget"],
        "comm_every": cfg["comm_every"],
        "obs_frac": obs_frac,
        "comm_delay": comm_delay,
        "match": match,
        "max_steps": max_steps,
        "seed": seed,
        "n_episodes": n_episodes,
        "share": share,
        "budget_obs": budget_obs,
        "memory": memory,
        "msg_credit": msg_credit if learns_msgs else None,
        "msg_cost": msg_cost if learns_msgs else None,
        "msg_handover": msg_handover if learns_msgs else None,
        "config": config,
        "git_commit": git_commit(),
        "episode_return": [],
        "episode_goals": [],
        "msgs_sent": [],
        "rollouts": 0,
        "eval_episode": [],
        "eval": [],
    }

    best = None   # (validation score, episode, network snapshot)
    t0 = time.time()
    total_steps = 0
    for ep in range(1, n_episodes + 1):
        eps = max(0.05, 1.0 - (1.0 - 0.05) * (ep - 1) / explore_episodes)
        for agent in agents:
            agent.eps = eps
        if lr_decay:
            frac = (ep - 1) / max(1, n_episodes - 1)
            for l in learners:
                l.opt.lr = lr0 * (1.0 - 0.8 * frac)   # linear decay to 20% (5e-4 -> 1e-4)
        p_rule = 0.0
        if learns_msgs and msg_handover > 0:
            p_rule = max(0.0, 1.0 - (ep - 1) / (msg_handover * n_episodes))

        obs_list, info = env.reset(seed=seed * 1_000_003 + ep)
        ep_return = 0.0
        rewards = []                       # scaled training rewards, for Monte-Carlo gate targets
        decisions = []                     # (step index, agent, gate input, message) for msg_credit="team"
        rollouts_left = [rollouts_per_episode] * len(agents)
        while True:
            actions = choose_actions(agents, obs_list)
            messages = [0, 0]
            if uses_comm:
                messages, decided = choose_messages(agents, obs_list, info["can_send"], sender,
                                                    p_rule=p_rule, msg_cost=msg_cost)
                if learns_msgs:
                    for i, agent in enumerate(agents):
                        if not decided[i]:
                            continue
                        if msg_credit == "team":
                            decisions.append((len(rewards), i, agent.x(obs_list[i]), messages[i]))
                        elif rollouts_left[i] > 0 and random.random() < rollout_prob:
                            adv = rollout_advantage(env, agents, actions, messages, i, sender, msg_cost,
                                                    rollout_horizon, gamma)
                            agent.gate.push(agent.x(obs_list[i]), 1, adv)
                            rollouts_left[i] -= 1
                            history["rollouts"] += 1

            next_obs_list, reward, terminated, truncated, info = env.step(actions, messages)
            ep_return += info["task_reward"]
            r = reward * REWARD_SCALE
            rewards.append(r)
            total_steps += 1

            for i, agent in enumerate(agents):
                agent.store(obs_list[i], actions[i], r, next_obs_list[i], terminated, truncated)
            if total_steps % train_every == 0:
                for l in learners:
                    l.update()
            if learns_msgs and total_steps % gate_every == 0:
                for g in gates:
                    g.update()

            obs_list = next_obs_list
            if render:
                env.render()
            if terminated or truncated:
                break

        for agent in agents:
            agent.end_episode()
        if decisions:   # Monte-Carlo team return from each decision to the end of the episode
            returns = np.zeros(len(rewards) + 1)
            for t in range(len(rewards) - 1, -1, -1):
                returns[t] = rewards[t] + gamma * returns[t + 1]
            for t, i, x, m in decisions:
                agents[i].gate.push(x, m, returns[t])

        history["episode_return"].append(ep_return)
        history["episode_goals"].append(info["score_team"])
        history["msgs_sent"].append(sum(info["msgs_sent"]))

        if verbose and ep % log_every == 0:
            recent = slice(-log_every, None)
            print(
                f"[{setting} obs={obs_frac} d={comm_delay} s={seed}] Ep {ep:5d}/{n_episodes} | "
                f"R={np.mean(history['episode_return'][recent]):7.1f} | "
                f"goals={np.mean(history['episode_goals'][recent]):.2f} | "
                f"msgs={np.mean(history['msgs_sent'][recent]):5.1f} | "
                f"eps={eps:.3f} | {time.time() - t0:.0f}s",
                flush=True,
            )

        if ep % eval_every == 0 or ep == n_episodes:
            metrics = evaluate(agents, eval_env, n_eval=n_eval_episodes, sender=sender, msg_cost=msg_cost)
            history["eval_episode"].append(ep)
            history["eval"].append(metrics)
            # Model selection on the validation episodes: keep the best snapshot
            score = (metrics["goals_mean"], metrics["return_mean"])
            if best is None or score > best[0]:
                best = (score, ep, _snapshot(agents))
            if verbose:
                print(
                    f"[{setting} obs={obs_frac} d={comm_delay} s={seed}] Eval {ep:5d} | "
                    f"goals={metrics['goals_mean']:.2f} rate={metrics['goal_rate']:.2f} "
                    f"| msgs={metrics['msgs_mean']:.1f}",
                    flush=True,
                )

    # Final, larger evaluation on held-out episodes (disjoint from the validation
    # episodes used for model selection), for the last policy and the selected one.
    ev = dict(n_eval=n_final_eval, seed=FINAL_SEED_BASE, sender=sender, msg_cost=msg_cost)
    history["final_eval_last"] = evaluate(agents, eval_env, **ev)
    _restore(agents, best[2])
    history["best_episode"] = best[1]
    history["final_eval"] = evaluate(agents, eval_env, **ev)
    if uses_comm:
        history["final_eval_shuffled"] = evaluate(agents, eval_env, shuffle=True, **ev)
    history["train_seconds"] = time.time() - t0
    history["total_steps"] = total_steps

    os.makedirs(save_dir, exist_ok=True)
    for agent in agents:
        agent.save(save_dir)
    write_json_atomic(os.path.join(save_dir, "history.json"), history)

    if verbose:
        fe = history["final_eval"]
        print(
            f"[{setting} obs={obs_frac} d={comm_delay} s={seed}] FINAL ({n_final_eval} eps) | "
            f"goals={fe['goals_mean']:.2f} rate={fe['goal_rate']:.2f} | saved to {save_dir}"
        )
    return history


# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="L4")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--obs_frac", type=float, default=0.075)
    parser.add_argument("--comm_delay", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--match", action="store_true", help="timed multi-goal matches")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    parser.add_argument("--n_final_eval", type=int, default=200)
    parser.add_argument("--explore_frac", type=float, default=0.1)
    parser.add_argument("--train_every", type=int, default=2)
    parser.add_argument("--independent", action="store_true", help="no parameter sharing")
    parser.add_argument("--msg_credit", default="rollout", choices=["rollout", "team"])
    parser.add_argument("--msg_cost", type=float, default=0.05)
    parser.add_argument("--msg_handover", type=float, default=0.3)
    parser.add_argument("--budget_blind", action="store_true", help="ablation: hide the budget features")
    parser.add_argument("--no_memory", action="store_true", help="ablation: blank the memory inputs")
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--opponent", action="store_true", help="add the scripted keeper")
    args = parser.parse_args()

    train(
        setting=args.setting,
        n_episodes=args.episodes,
        obs_frac=args.obs_frac,
        comm_delay=args.comm_delay,
        max_steps=args.max_steps,
        match=args.match,
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
        share=not args.independent,
        msg_credit=args.msg_credit,
        msg_cost=args.msg_cost,
        msg_handover=args.msg_handover,
        budget_obs=not args.budget_blind,
        memory=not args.no_memory,
        lr_decay=args.lr_decay,
        opponent=args.opponent,
    )
