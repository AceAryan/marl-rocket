"""
Non-learning reference teams (proposal section 4.2): a random policy and a
hand-coded team, evaluated on the same held-out episodes as the trained agents.

The hand-coded team acts only on its own observation (partial view, memory,
messages). Each agent sweeps its half of the ball's spawn band, then waits in
front of the goal mouth (a meeting point that needs no communication); whenever
it knows where the ball is (own sighting, memory or a teammate's message) it
goes there and pushes the ball toward the goal. With messages it uses the paced
rule sender (send on sight when the teammate's copy has expired).

Usage:
    python -m experiments.scripted --obs_frac 0.075 [--match]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import math
import numpy as np

from env.rocket_env import OBS_SLICES as S, FIELD_W, FIELD_H, FIELD_D, CONTACT_DIST, _denorm_pos

GOAL = np.array([FIELD_W / 2, FIELD_H])
MOUTH = np.array([FIELD_W / 2, FIELD_H - 3.5])
SPAWN_BAND = (3.0, FIELD_W - 3.0, 0.4 * FIELD_H, 0.6 * FIELD_H)


def _move(me, target, tol=0.3):
    d = target - me
    if abs(d[0]) < tol and abs(d[1]) < tol:
        return 0
    if abs(d[1]) >= abs(d[0]):
        if d[1] > 0:
            return 5 if d[1] > 1.3 else 1
        return 2
    return 4 if d[0] > 0 else 3


def _push(me, ball, target=GOAL):
    u = target - ball
    n = np.linalg.norm(u)
    u = u / n if n > 1e-6 else np.array([0.0, 1.0])
    rel = me - ball
    if rel @ u > -0.3 and np.linalg.norm(rel) < 2.2:          # beside or in front: go around
        perp = np.array([-u[1], u[0]])
        side = 1.0 if rel @ perp >= 0 else -1.0
        return _move(me, ball + perp * side * 1.7 - u * 0.8)
    behind = ball - u * (CONTACT_DIST + 0.4)
    if np.linalg.norm(me - behind) > 0.7:
        return _move(me, behind)
    return _move(me, ball + u, 0.1)


def _sweep(x0, x1, y0, y1, r, left_first):
    rows = max(1, int(math.ceil((y1 - y0) / max(1.6 * r, 0.5))))
    pts, left = [], left_first
    for k in range(rows):
        y = y0 + (k + 0.5) * (y1 - y0) / rows
        a, b = (x0 + 0.6 * r, x1 - 0.6 * r) if x1 - x0 > 1.2 * r else ((x0 + x1) / 2,) * 2
        pts += [np.array([a, y]), np.array([b, y])] if left else [np.array([b, y]), np.array([a, y])]
        left = not left
    return pts


class ScriptedAgent:
    def __init__(self, i, obs_frac):
        self.i, self.r = i, obs_frac * FIELD_D
        self.reset()

    def reset(self):
        x0, x1, y0, y1 = SPAWN_BAND
        mid = (x0 + x1) / 2
        half = (x0, mid) if self.i == 0 else (mid, x1)
        self.waypoints = _sweep(half[0], half[1], y0, y1, self.r, left_first=(self.i == 0))
        self.k = 0
        self.stale = []

    def act(self, o):
        me = _denorm_pos(o[S["self"]])
        if o[S["ball_vis"]][0] > 0:
            self.stale = []
            return _push(me, _denorm_pos(o[S["ball"]]))
        if o[S["mem_ball_age"]][0] < 1.0:
            mem = _denorm_pos(o[S["mem_ball"]])
            if not any(np.linalg.norm(mem - s) < 0.5 for s in self.stale):
                if np.linalg.norm(me - mem) < 0.7 * self.r:
                    self.stale.append(mem)          # got there and it is gone
                else:
                    return _move(me, mem)
        while self.k < len(self.waypoints) and np.linalg.norm(me - self.waypoints[self.k]) < 0.6:
            self.k += 1
        if self.k < len(self.waypoints):
            return _move(me, self.waypoints[self.k])
        return _move(me, MOUTH)


def evaluate_reference(kind, obs_frac, match=False, n_eval=200, seed=None, max_steps=None):
    """kind: 'random', 'scripted' (no messages) or 'scripted_comm' (paced rule sender)."""
    from experiments.train import FINAL_SEED_BASE, make_env, choose_messages
    seed = FINAL_SEED_BASE if seed is None else seed
    setting = "Rinf" if kind == "scripted_comm" else "A"
    env = make_env(setting, obs_frac=obs_frac, match=match, max_steps=max_steps)
    rng = np.random.default_rng(0)
    goals, first = [], []
    for e in range(n_eval):
        obs, info = env.reset(seed=seed + e)
        bots = [ScriptedAgent(i, obs_frac) for i in range(2)]
        scored = 0
        while True:
            if kind == "random":
                actions = rng.integers(6, size=2).tolist()
            else:
                actions = [b.act(o) for b, o in zip(bots, obs)]
            msgs = choose_messages(bots, obs, info["can_send"], "rule")[0] if kind == "scripted_comm" else [0, 0]
            obs, _, term, trunc, info = env.step(actions, msgs)
            if info["score_team"] > scored:
                scored = info["score_team"]
                for b in bots:
                    b.reset()
            if term or trunc:
                break
        goals.append(info["score_team"])
        first.append(info["goal_steps"][0] if info["goal_steps"] else None)
    g = np.asarray(goals, np.float64)
    ttg = [f for f in first if f is not None]
    return {"goals_mean": float(g.mean()), "goal_rate": float(np.mean(g > 0)),
            "time_to_goal": float(np.mean(ttg)) if ttg else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_frac", type=float, default=0.075)
    parser.add_argument("--match", action="store_true")
    parser.add_argument("--n_eval", type=int, default=200)
    args = parser.parse_args()
    for kind in ("random", "scripted", "scripted_comm"):
        print(kind, evaluate_reference(kind, args.obs_frac, args.match, args.n_eval))
