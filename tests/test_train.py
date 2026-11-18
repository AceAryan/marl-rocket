"""Tests for the training-side message logic, the learners and an end-to-end train() smoke."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import math
import random
import tempfile
import numpy as np

from env.rocket_env import N_ACTIONS, OBS_DIM, OBS_SLICES, RocketEnv
from agents.dqn_agent import MLP, Adam, DQNAgent, make_agents, unique, _clip_
from experiments.train import (
    BALL_VIS, SEND_AGE, choose_messages, decision_points, evaluate, make_env, parse_setting,
    rollout_advantage, run_label, sender_kind, train,
)


def _agents(share=True):
    return make_agents(OBS_DIM, N_ACTIONS, share=share)


def _obs(ball_visible, send_age=1.0):
    o = np.zeros(OBS_DIM, np.float32)
    o[BALL_VIS] = 1.0 if ball_visible else 0.0
    o[SEND_AGE] = send_age
    return o


def test_parse_setting():
    assert parse_setting("A") == dict(sender="none", comm_budget=0, comm_every=1)
    assert parse_setting("L4") == dict(sender="learned", comm_budget=4, comm_every=1)
    assert parse_setting("B") == parse_setting("Linf") == dict(sender="learned", comm_budget=None, comm_every=1)
    assert parse_setting("D") == parse_setting("L20")
    assert parse_setting("C") == dict(sender="learned", comm_budget=None, comm_every=5)
    assert parse_setting("R12")["sender"] == "rule" and parse_setting("Rinf")["comm_budget"] is None
    assert parse_setting("F2", max_steps=80) == dict(sender="fixed", comm_budget=None, comm_every=40)
    assert parse_setting("F3", max_steps=80)["comm_every"] == 27
    assert parse_setting("Finf")["comm_every"] == 1
    for bad in ("L0", "X3", "R", "Lx"):
        try:
            parse_setting(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert [sender_kind(s) for s in ("A", "F4", "R4", "L4")] == ["none", "fixed", "rule", "learned"]
    assert run_label("L4") == "L4" and run_label("L4", msg_credit="team") == "L4-team"
    assert run_label("R4", msg_credit="team") == "R4" and run_label("L4", budget_obs=False) == "L4-blind"
    assert run_label("A", share=False) == "A-indep"


def test_learned_sender_silent_without_sighting_or_permission():
    agents = _agents()
    for a in agents:
        a.eps = 1.0   # fully random: would send often if allowed
    for _ in range(200):
        msgs, decided = choose_messages(agents, [_obs(False), _obs(False)], [True, True], "learned")
        assert msgs == [0, 0] and decided == [False, False]
        msgs, decided = choose_messages(agents, [_obs(True), _obs(True)], [False, False], "learned")
        assert msgs == [0, 0] and decided == [False, False]


def test_decision_points_require_sighting_and_permission():
    obs = [_obs(True), _obs(False)]
    assert decision_points(obs, [True, True]) == [True, False]
    assert decision_points(obs, [False, True]) == [False, False]


def test_silence_blocks_all_messages():
    agents = _agents()
    for a in agents:
        a.eps = 1.0
    for sender in ("learned", "rule", "fixed"):
        msgs, _ = choose_messages(agents, [_obs(True), _obs(True)], [True, True], sender, silence=True)
        assert msgs == [0, 0]


def test_gate_exploration_is_silence_biased_and_binary():
    agent = _agents()[0]
    agent.eps = 1.0
    picks = [agent.select_message(_obs(True)) for _ in range(4000)]
    assert set(picks) == {0, 1} and 0.45 < np.mean(np.array(picks) == 0) < 0.55


def test_rule_sender_is_paced():
    agents = _agents()
    msgs, decided = choose_messages(agents, [_obs(True), _obs(False)], [True, True], "rule")
    assert msgs == [1, 0] and decided == [False, False]       # no learned decision recorded
    msgs, _ = choose_messages(agents, [_obs(True, send_age=0.3), _obs(True)], [True, True], "rule")
    assert msgs == [0, 1]                                     # agent 0's last message still held
    msgs, _ = choose_messages(agents, [_obs(True), _obs(True)], [False, True], "rule")
    assert msgs == [0, 1]                                     # budget spent for agent 0


def test_rule_sender_respects_budget_and_pacing_in_the_env():
    env = make_env("R2", obs_frac=1.0)                        # always sees the ball
    obs, info = env.reset(seed=0)
    agents = _agents()
    sent = []
    for t in range(30):
        msgs, _ = choose_messages(agents, obs, info["can_send"], "rule")
        if msgs[0]:
            sent.append(t)
        obs, _, term, trunc, info = env.step([0, 0], msgs)
        if term or trunc:
            break
    assert info["msgs_sent"] == [2, 2] and sent == [0, 11]   # resend once the teammate copy (held through step 11) expired


def test_fixed_sender_follows_its_schedule():
    for setting, expected in (("F2", [20, 60]), ("F4", [10, 30, 50, 70])):
        env = make_env(setting, obs_frac=0.15)
        env.assist_required = False
        obs, info = env.reset(seed=0)
        env._ball_pos[:] = [1.0, 15.0]
        agents = _agents()
        sent = []
        for t in range(1, 81):
            msgs, _ = choose_messages(agents, obs, info["can_send"], "fixed")
            if msgs[0]:
                sent.append(t)
            obs, _, term, trunc, info = env.step([0, 0], msgs)
            if term or trunc:
                break
        assert sent == expected, (setting, sent)


def test_rollout_advantage_is_zero_when_the_send_is_refused_and_leaves_env_untouched():
    env = make_env("L1", obs_frac=0.15)
    obs, info = env.reset(seed=3)
    env.step([0, 0], [1, 1])                                  # both budgets spent
    agents = _agents()
    for a in agents:
        a.eps = 0.0
    state = env.clone()
    adv = rollout_advantage(env, agents, [1, 1], [0, 0], 0, "learned", 0.05, horizon=30)
    assert adv == 0.0
    assert env._step_count == state._step_count and np.array_equal(env._ball_pos, state._ball_pos)


def test_rollout_advantage_matches_manual_rollouts():
    env = make_env("Linf", obs_frac=0.15)
    obs, info = env.reset(seed=5)
    agents = _agents()
    for a in agents:
        a.eps = 0.0
    adv = rollout_advantage(env, agents, [1, 2], [0, 0], 0, "learned", 0.05, horizon=12, gamma=0.9)

    def manual(msg0):
        e = env.clone()
        o, r, te, tr, inf = e.step([1, 2], [msg0, 0])
        total, disc = r * 0.1, 0.9
        for _ in range(11):
            if te or tr:
                break
            acts = [int(np.argmax(a.q_values(x))) for a, x in zip(agents, o)]
            msgs, _ = choose_messages(agents, o, inf["can_send"], "learned", greedy=True, msg_cost=0.05)
            o, r, te, tr, inf = e.step(acts, msgs)
            total += disc * r * 0.1
            disc *= 0.9
        return total
    assert np.isclose(adv, manual(1) - manual(0))


def test_shared_learner():
    agents = make_agents(OBS_DIM, N_ACTIONS, share=True, batch_size=8, learning_starts=8)
    assert agents[0].learner is agents[1].learner and agents[0].gate is agents[1].gate
    assert agents[0].learner.batch_size == 16
    o = np.zeros(OBS_DIM, np.float32)
    assert np.array_equal(agents[0].x(o)[-2:], [1, 0]) and np.array_equal(agents[1].x(o)[-2:], [0, 1])
    assert agents[0].learner.q_net.params[0].shape[0] == OBS_DIM + 2
    indep = make_agents(OBS_DIM, N_ACTIONS, share=False)
    assert len(unique([a.learner for a in indep])) == 2 and indep[0].x(o).shape == (OBS_DIM,)


class _OldMLP:
    """The pre-refactor MLP (list of arrays), kept as a numerical reference."""

    def __init__(self, params):
        self.params = [p.copy() for p in params]

    def forward(self, x):
        W1, b1, W2, b2, W3, b3 = self.params
        self.c = (x, np.maximum(x @ W1 + b1, 0.0))
        h2 = np.maximum(self.c[1] @ W2 + b2, 0.0)
        self.c = self.c + (h2,)
        return h2 @ W3 + b3

    def backward(self, dout):
        W1, _, W2, _, W3, _ = self.params
        x, h1, h2 = self.c
        dh2 = (dout @ W3.T) * (h2 > 0)
        dh1 = (dh2 @ W2.T) * (h1 > 0)
        return [x.T @ dh1, dh1.sum(0), h1.T @ dh2, dh2.sum(0), h2.T @ dout, dout.sum(0)]


def test_flat_mlp_and_adam_match_the_reference_implementation():
    rng = np.random.default_rng(0)
    net = MLP(7, 16, 3, rng=np.random.RandomState(1))
    old = _OldMLP(net.params)
    opt = Adam(net.flat.size, 1e-2)
    m = [np.zeros_like(p) for p in old.params]
    v = [np.zeros_like(p) for p in old.params]
    for t in range(1, 6):
        x = rng.normal(size=(5, 7)).astype(np.float32)
        dout = rng.normal(size=(5, 3)).astype(np.float32)
        assert np.allclose(net.forward(x, cache=True), old.forward(x), atol=1e-5)
        grad = net.backward(dout).copy()
        ogr = old.backward(dout)
        assert np.allclose(grad, np.concatenate([g.ravel() for g in ogr]), atol=1e-5)
        opt.step(net.flat, grad)
        step = 1e-2 * math.sqrt(1 - 0.999 ** t) / (1 - 0.9 ** t)
        for p, g, mm, vv in zip(old.params, ogr, m, v):
            mm *= 0.9; mm += 0.1 * g
            vv *= 0.999; vv += 0.001 * g * g
            p -= step * mm / (np.sqrt(vv) + 1e-8)
        assert np.allclose(net.flat, np.concatenate([p.ravel() for p in old.params]), atol=1e-5)
    g = np.full(4, 3.0, np.float32)
    _clip_(g, 1.0)
    assert np.isclose(np.linalg.norm(g), 1.0, atol=1e-5)


def test_snapshot_views_survive_state_dict_round_trip():
    net = MLP(4, 8, 2)
    state = net.state_dict()
    net.flat[:] = 0.0
    net.load_state_dict(state)
    assert np.allclose(np.concatenate([state[f"p{i}"].ravel() for i in range(6)]), net.flat)
    assert net.params[0].base is net.flat or np.shares_memory(net.params[0], net.flat)


def test_train_smoke_all_kinds_and_credits():
    with tempfile.TemporaryDirectory() as d:
        for setting, kw in (("A", {}), ("L4", {}), ("L4", {"msg_credit": "team"}), ("Rinf", {}), ("F4", {}),
                            ("L2", {"match": True, "max_steps": 40}), ("A", {"share": False}),
                            ("L2", {"budget_obs": False})):
            h = train(setting=setting, n_episodes=4, obs_frac=0.15, max_steps=kw.pop("max_steps", 40),
                      save_dir=os.path.join(d, setting + str(len(kw))), eval_every=2, n_eval_episodes=2,
                      n_final_eval=2, verbose=False, seed=0, rollout_prob=1.0, **kw)
            assert "final_eval" in h and "final_eval_last" in h and len(h["eval"]) == 2
            assert h["best_episode"] in h["eval_episode"]
            assert ("final_eval_shuffled" in h) == (setting != "A")
            assert all(m <= 2 * 4 for m in h["msgs_sent"]) if setting == "L4" else True
            if setting.startswith("L") and kw.get("msg_credit", "rollout") == "rollout":
                assert h["rollouts"] > 0
            with open(os.path.join(d, setting + str(len(kw)), "history.json")) as f:
                assert json.load(f)["label"] == h["label"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All tests passed!")
